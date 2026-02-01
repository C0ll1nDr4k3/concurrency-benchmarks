/**
 * @file hnsw_fine_pessimistic.hpp
 * @brief HNSW implementation with fine-grained pessimistic concurrency control.
 *
 * This implementation uses per-node locks for neighbor lists:
 * - Each node has its own mutex protecting its neighbor lists
 * - Searches acquire shared locks on nodes as they traverse
 * - Insertions acquire exclusive locks only on nodes being modified
 *
 * Upper layers use greedy search (single candidate),
 * lower layers use beam search (multiple candidates).
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_set>
#include "common.hpp"

namespace nilvec {

/**
 * @brief HNSW index with fine-grained pessimistic locking.
 *
 * Each node has its own read-write mutex protecting its neighbor lists.
 * This allows concurrent searches and more fine-grained locking during
 * insertion.
 */
template <typename T>
class HNSWFinePessimistic {
 public:
  /**
   * @brief Construct an HNSW index.
   * @param dim Vector dimensionality
   * @param M Number of neighbors per node (per layer)
   * @param ef_construction Size of dynamic candidate list during construction
   * @param mL Level generation factor (typically 1/ln(M))
   */
  HNSWFinePessimistic(Dim dim,
                      size_t M = 16,
                      size_t ef_construction = 200,
                      float mL = 0.0f)
      : dim_(dim),
        M_(M),
        M_max0_(2 * M),
        ef_construction_(ef_construction),
        mL_(mL > 0 ? mL : 1.0f / std::log(static_cast<float>(M))),
        entry_point_(INVALID_NODE),
        max_level_(-1),
        rng_(std::random_device{}()) {}

  /**
   * @brief Insert a vector into the index.
   * @param data The vector data
   * @return The node ID assigned to this vector
   */
  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);

    // Allocate new node (requires global lock briefly)
    NodeId new_id;
    int new_level;
    {
      std::unique_lock global_lock(global_mutex_);
      new_id = static_cast<NodeId>(nodes_.size());
      new_level = random_level(mL_);

      // Create the new node
      auto node = std::make_unique<Node>();
      node->vector.assign(data.begin(), data.end());
      node->neighbors.resize(new_level + 1);
      nodes_.push_back(std::move(node));
    }

    // Handle first insertion
    NodeId expected_entry = INVALID_NODE;
    if (entry_point_.compare_exchange_strong(expected_entry, new_id)) {
      max_level_.store(new_level, std::memory_order_release);
      return new_id;
    }

    // Get current entry point and max level
    NodeId curr_entry = entry_point_.load(std::memory_order_acquire);
    int curr_max_level = max_level_.load(std::memory_order_acquire);

    // Phase 1: Greedy descent through upper layers (above new_level)
    // Only need shared locks for reading neighbor lists
    for (int level = curr_max_level; level > new_level; --level) {
      curr_entry = greedy_search_layer(data, curr_entry, level);
    }

    // Phase 2: Search and connect at layers [min(new_level, max_level), ..., 0]
    for (int level = std::min(new_level, curr_max_level); level >= 0; --level) {
      auto candidates = search_layer(data, curr_entry, ef_construction_, level);

      size_t M_layer = (level == 0) ? M_max0_ : M_;
      auto neighbors = select_neighbors(candidates, M_layer);

      // Lock the new node and set its neighbors
      {
        std::unique_lock lock(nodes_[new_id]->mutex);
        nodes_[new_id]->neighbors[level] = neighbors;
      }

      // Add reverse edges with fine-grained locking
      for (NodeId neighbor : neighbors) {
        std::unique_lock lock(nodes_[neighbor]->mutex);
        auto& neighbor_list = nodes_[neighbor]->neighbors[level];
        neighbor_list.push_back(new_id);

        // Prune if necessary
        if (neighbor_list.size() > M_layer) {
          prune_neighbors_locked(neighbor, level, M_layer);
        }
      }

      // Update entry point for next layer
      if (!candidates.empty()) {
        curr_entry = candidates.front().id;
      }
    }

    // Update entry point and max level if necessary
    if (new_level > curr_max_level) {
      // Use CAS loop to update atomically
      int expected_level = curr_max_level;
      while (new_level > expected_level) {
        if (max_level_.compare_exchange_weak(expected_level, new_level)) {
          entry_point_.store(new_id, std::memory_order_release);
          break;
        }
      }
    }

    return new_id;
  }

  /**
   * @brief Search for k nearest neighbors.
   * @param query The query vector
   * @param k Number of neighbors to return
   * @param ef Search expansion factor (default: k)
   * @return Search results containing IDs and distances
   */
  SearchResult search(std::span<const T> query, size_t k, size_t ef = 0) const {
    if (ef == 0)
      ef = k;

    NodeId curr_entry = entry_point_.load(std::memory_order_acquire);
    if (curr_entry == INVALID_NODE) {
      return SearchResult{};
    }

    int curr_max_level = max_level_.load(std::memory_order_acquire);

    // Phase 1: Greedy descent through upper layers to layer 1
    for (int level = curr_max_level; level >= 1; --level) {
      curr_entry = greedy_search_layer(query, curr_entry, level);
    }

    // Phase 2: Beam search at layer 0
    auto candidates = search_layer(query, curr_entry, ef, 0);

    // Return top k results
    SearchResult result;
    size_t count = std::min(k, candidates.size());
    result.ids.reserve(count);
    result.distances.reserve(count);

    for (size_t i = 0; i < count; ++i) {
      result.ids.push_back(candidates[i].id);
      result.distances.push_back(candidates[i].distance);
    }

    return result;
  }

  /**
   * @brief Get the number of vectors in the index.
   */
  size_t size() const {
    std::shared_lock lock(global_mutex_);
    return nodes_.size();
  }

  /**
   * @brief Get the current maximum level in the graph.
   */
  int max_level() const { return max_level_.load(std::memory_order_acquire); }

 private:
  /**
   * @brief Node structure with per-node locking.
   */
  struct Node {
    std::vector<T> vector;
    std::vector<std::vector<NodeId>> neighbors;  // [level][neighbor]
    mutable std::shared_mutex mutex;
  };

  /**
   * @brief Greedy search within a single layer (upper layers).
   *
   * Uses shared locks on each node's neighbor list.
   */
  NodeId greedy_search_layer(std::span<const T> query,
                             NodeId entry_point,
                             int level) const {
    NodeId current = entry_point;
    float current_dist = compute_distance(query, current);

    bool improved = true;
    while (improved) {
      improved = false;

      // Read neighbors with shared lock
      std::vector<NodeId> neighbors;
      {
        std::shared_lock lock(nodes_[current]->mutex);
        if (level < static_cast<int>(nodes_[current]->neighbors.size())) {
          neighbors = nodes_[current]->neighbors[level];
        }
      }

      for (NodeId neighbor : neighbors) {
        float neighbor_dist = compute_distance(query, neighbor);
        if (neighbor_dist < current_dist) {
          current = neighbor;
          current_dist = neighbor_dist;
          improved = true;
        }
      }
    }

    return current;
  }

  /**
   * @brief Beam search within a single layer (lower layers).
   *
   * Uses shared locks on each node's neighbor list.
   */
  std::vector<Candidate> search_layer(std::span<const T> query,
                                      NodeId entry_point,
                                      size_t ef,
                                      int level) const {
    std::unordered_set<NodeId> visited;
    MinHeap candidates;
    MaxHeap results;

    float entry_dist = compute_distance(query, entry_point);
    candidates.push({entry_point, entry_dist});
    results.push({entry_point, entry_dist});
    visited.insert(entry_point);

    while (!candidates.empty()) {
      auto [curr_id, curr_dist] = candidates.top();
      candidates.pop();

      if (curr_dist > results.top().distance) {
        break;
      }

      // Read neighbors with shared lock
      std::vector<NodeId> neighbors;
      {
        std::shared_lock lock(nodes_[curr_id]->mutex);
        if (level < static_cast<int>(nodes_[curr_id]->neighbors.size())) {
          neighbors = nodes_[curr_id]->neighbors[level];
        }
      }

      for (NodeId neighbor : neighbors) {
        if (visited.count(neighbor) > 0)
          continue;
        visited.insert(neighbor);

        float neighbor_dist = compute_distance(query, neighbor);

        if (results.size() < ef || neighbor_dist < results.top().distance) {
          candidates.push({neighbor, neighbor_dist});
          results.push({neighbor, neighbor_dist});

          if (results.size() > ef) {
            results.pop();
          }
        }
      }
    }

    std::vector<Candidate> result;
    result.reserve(results.size());
    while (!results.empty()) {
      result.push_back(results.top());
      results.pop();
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  /**
   * @brief Select the best M neighbors.
   */
  std::vector<NodeId> select_neighbors(const std::vector<Candidate>& candidates,
                                       size_t M) const {
    std::vector<NodeId> result;
    result.reserve(M);
    for (size_t i = 0; i < std::min(M, candidates.size()); ++i) {
      result.push_back(candidates[i].id);
    }
    return result;
  }

  /**
   * @brief Prune a node's neighbor list (assumes lock is held).
   *
   * Note: We read neighbor vectors without locks to avoid deadlocks.
   * This is safe because vectors are immutable after insertion.
   */
  void prune_neighbors_locked(NodeId node, int level, size_t max_size) {
    auto& neighbor_list = nodes_[node]->neighbors[level];

    if (neighbor_list.size() <= max_size)
      return;

    std::span<const T> node_vec(nodes_[node]->vector);
    std::vector<Candidate> candidates;
    candidates.reserve(neighbor_list.size());

    for (NodeId neighbor : neighbor_list) {
      // Vectors are immutable after insertion, safe to read without lock
      float dist = squared_distance(
          node_vec, std::span<const T>(nodes_[neighbor]->vector));
      candidates.push_back({neighbor, dist});
    }

    std::sort(candidates.begin(), candidates.end());

    neighbor_list.clear();
    neighbor_list.reserve(max_size);
    for (size_t i = 0; i < max_size; ++i) {
      neighbor_list.push_back(candidates[i].id);
    }
  }

  /**
   * @brief Compute distance from query to a node.
   *
   * Vectors are immutable after insertion, so no lock needed.
   */
  float compute_distance(std::span<const T> query, NodeId node) const {
    return squared_distance(query, std::span<const T>(nodes_[node]->vector));
  }

  int random_level(float mL) {
    std::lock_guard<std::mutex> lock(rng_mutex_);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    return static_cast<int>(-std::log(r) * mL);
  }

  // Configuration
  Dim dim_;
  size_t M_;
  size_t M_max0_;
  size_t ef_construction_;
  float mL_;

  // Graph structure
  std::atomic<NodeId> entry_point_;
  std::atomic<int> max_level_;
  std::vector<std::unique_ptr<Node>> nodes_;

  // Global mutex for node allocation
  mutable std::shared_mutex global_mutex_;

  // Random number generator
  mutable std::mt19937 rng_;
  mutable std::mutex rng_mutex_;
};

}  // namespace nilvec
