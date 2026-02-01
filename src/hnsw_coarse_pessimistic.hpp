/**
 * @file hnsw_coarse_pessimistic.hpp
 * @brief HNSW implementation with per-layer pessimistic concurrency control.
 *
 * This implementation uses a read-write lock per layer of the HNSW graph.
 * - Readers acquire shared locks on layers they traverse
 * - Writers acquire exclusive locks on layers they modify
 *
 * Per-layer locking provides better concurrency than a single global lock
 * while being simpler than per-node fine-grained locking.
 *
 * Upper layers use greedy search (single candidate),
 * lower layers use beam search (multiple candidates).
 */

#pragma once

#include <atomic>
#include <cassert>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <unordered_set>
#include <vector>
#include "common.hpp"

namespace nilvec {

/**
 * @brief HNSW index with per-layer pessimistic locking.
 *
 * Each layer has its own read-write mutex. Searches acquire shared (read)
 * locks on layers as they descend. Insertions acquire exclusive (write) locks
 * on layers they modify.
 */
template <typename T>
class HNSWCoarsePessimistic {
 public:
  /**
   * @brief Construct an HNSW index.
   * @param dim Vector dimensionality
   * @param M Number of neighbors per node (per layer)
   * @param ef_construction Size of dynamic candidate list during construction
   * @param mL Level generation factor (typically 1/ln(M))
   * @param max_layers Maximum number of layers (default: 16)
   */
  HNSWCoarsePessimistic(Dim dim,
                        size_t M = 16,
                        size_t ef_construction = 200,
                        float mL = 0.0f,
                        int max_layers = 16)
      : dim_(dim),
        M_(M),
        M_max0_(2 * M),  // Max neighbors at layer 0
        ef_construction_(ef_construction),
        mL_(mL > 0 ? mL : 1.0f / std::log(static_cast<float>(M))),
        max_layers_(max_layers),
        entry_point_(INVALID_NODE),
        max_level_(-1),
        rng_(std::random_device{}()) {
    // Pre-allocate layer mutexes
    layer_mutexes_.reserve(max_layers_);
    for (int i = 0; i < max_layers_; ++i) {
      layer_mutexes_.push_back(std::make_unique<std::shared_mutex>());
    }
  }

  /**
   * @brief Insert a vector into the index.
   * @param data The vector data
   * @return The node ID assigned to this vector
   */
  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);

    // Acquire global lock for node allocation only
    NodeId new_id;
    int new_level;
    {
      std::unique_lock global_lock(global_mutex_);
      new_id = static_cast<NodeId>(vectors_.size());
      new_level = random_level(mL_, rng_);

      // Store the vector
      vectors_.emplace_back(data.begin(), data.end());

      // Initialize neighbor lists for all layers
      neighbors_.emplace_back();
      neighbors_[new_id].resize(new_level + 1);
    }

    // Handle first insertion
    NodeId expected_entry = INVALID_NODE;
    if (entry_point_.compare_exchange_strong(expected_entry, new_id,
                                             std::memory_order_acq_rel)) {
      max_level_.store(new_level, std::memory_order_release);
      return new_id;
    }

    NodeId curr_entry = entry_point_.load(std::memory_order_acquire);
    int curr_max_level = max_level_.load(std::memory_order_acquire);

    // Phase 1: Greedy descent through upper layers (above new_level)
    // Only need read locks for descent
    for (int level = curr_max_level; level > new_level; --level) {
      std::shared_lock layer_lock(*layer_mutexes_[level]);
      curr_entry = greedy_search_layer(data, curr_entry, level);
    }

    // Phase 2: Search and connect at layers [min(new_level, max_level), ..., 0]
    // Need write locks for modification
    for (int level = std::min(new_level, curr_max_level); level >= 0; --level) {
      std::unique_lock layer_lock(*layer_mutexes_[level]);

      auto candidates = search_layer(data, curr_entry, ef_construction_, level);

      // Select M best neighbors
      size_t M_layer = (level == 0) ? M_max0_ : M_;
      auto neighbors = select_neighbors(candidates, M_layer);

      // Connect new node to neighbors
      neighbors_[new_id][level] = neighbors;

      // Add reverse edges (bidirectional graph)
      for (NodeId neighbor : neighbors) {
        auto& neighbor_list = neighbors_[neighbor][level];
        neighbor_list.push_back(new_id);

        // Prune if necessary
        if (neighbor_list.size() > M_layer) {
          prune_neighbors(neighbor, level, M_layer);
        }
      }

      // Update entry point for next layer
      if (!candidates.empty()) {
        curr_entry = candidates.front().id;
      }
    }

    // Update entry point and max level if necessary (atomic)
    int expected_level = curr_max_level;
    while (new_level > expected_level) {
      if (max_level_.compare_exchange_weak(expected_level, new_level,
                                           std::memory_order_acq_rel)) {
        entry_point_.store(new_id, std::memory_order_release);
        break;
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
    // Acquire read lock per layer
    for (int level = curr_max_level; level >= 1; --level) {
      std::shared_lock layer_lock(*layer_mutexes_[level]);
      curr_entry = greedy_search_layer(query, curr_entry, level);
    }

    // Phase 2: Beam search at layer 0
    std::shared_lock layer0_lock(*layer_mutexes_[0]);
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
    return vectors_.size();
  }

  /**
   * @brief Get the current maximum level in the graph.
   */
  int max_level() const { return max_level_.load(std::memory_order_acquire); }

 private:
  /**
   * @brief Greedy search within a single layer (upper layers).
   *
   * Returns the closest node to the query starting from entry_point.
   * This is a simple greedy best-first search with a single candidate.
   * Assumes caller holds at least a shared lock on the layer.
   */
  NodeId greedy_search_layer(std::span<const T> query,
                             NodeId entry_point,
                             int level) const {
    NodeId current = entry_point;
    float current_dist = compute_distance(query, current);

    bool improved = true;
    while (improved) {
      improved = false;
      for (NodeId neighbor : neighbors_[current][level]) {
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
   * Returns a list of candidates sorted by distance to query.
   * Assumes caller holds at least a shared lock on the layer.
   */
  std::vector<Candidate> search_layer(std::span<const T> query,
                                      NodeId entry_point,
                                      size_t ef,
                                      int level) const {
    std::unordered_set<NodeId> visited;
    MinHeap candidates;  // Nodes to explore (closest first)
    MaxHeap results;     // Current best results (farthest first for pruning)

    float entry_dist = compute_distance(query, entry_point);
    candidates.push({entry_point, entry_dist});
    results.push({entry_point, entry_dist});
    visited.insert(entry_point);

    while (!candidates.empty()) {
      auto [curr_id, curr_dist] = candidates.top();
      candidates.pop();

      // Stop if current candidate is farther than worst result
      if (curr_dist > results.top().distance) {
        break;
      }

      // Explore neighbors
      for (NodeId neighbor : neighbors_[curr_id][level]) {
        if (visited.count(neighbor) > 0)
          continue;
        visited.insert(neighbor);

        float neighbor_dist = compute_distance(query, neighbor);

        // Add to results if better than worst or results not full
        if (results.size() < ef || neighbor_dist < results.top().distance) {
          candidates.push({neighbor, neighbor_dist});
          results.push({neighbor, neighbor_dist});

          if (results.size() > ef) {
            results.pop();
          }
        }
      }
    }

    // Convert results to sorted vector
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
   * @brief Select the best M neighbors using a simple heuristic.
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
   * @brief Prune a node's neighbor list to at most max_size neighbors.
   * Assumes caller holds exclusive lock on the layer.
   */
  void prune_neighbors(NodeId node, int level, size_t max_size) {
    auto& neighbor_list = neighbors_[node][level];

    if (neighbor_list.size() <= max_size)
      return;

    // Compute distances and sort
    std::span<const T> node_vec(vectors_[node]);
    std::vector<Candidate> candidates;
    candidates.reserve(neighbor_list.size());

    for (NodeId neighbor : neighbor_list) {
      float dist =
          squared_distance(node_vec, std::span<const T>(vectors_[neighbor]));
      candidates.push_back({neighbor, dist});
    }

    std::sort(candidates.begin(), candidates.end());

    // Keep only the closest max_size neighbors
    neighbor_list.clear();
    neighbor_list.reserve(max_size);
    for (size_t i = 0; i < max_size; ++i) {
      neighbor_list.push_back(candidates[i].id);
    }
  }

  /**
   * @brief Compute distance from query to a node.
   */
  float compute_distance(std::span<const T> query, NodeId node) const {
    return squared_distance(query, std::span<const T>(vectors_[node]));
  }

  // Configuration
  Dim dim_;
  size_t M_;
  size_t M_max0_;
  size_t ef_construction_;
  float mL_;
  int max_layers_;

  // Graph structure
  std::atomic<NodeId> entry_point_;
  std::atomic<int> max_level_;
  std::vector<std::vector<T>> vectors_;
  std::vector<std::vector<std::vector<NodeId>>>
      neighbors_;  // [node][level][neighbor]

  // Concurrency control - per-layer locks
  mutable std::shared_mutex global_mutex_;  // For vector/neighbor allocation
  mutable std::vector<std::unique_ptr<std::shared_mutex>> layer_mutexes_;

  // Random number generator
  mutable std::mt19937 rng_;
};

}  // namespace nilvec
