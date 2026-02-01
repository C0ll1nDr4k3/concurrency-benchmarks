/**
 * @file hnsw_fine_optimistic.hpp
 * @brief HNSW implementation with fine-grained optimistic concurrency control.
 *
 * This implementation uses optimistic concurrency at the node level:
 * - Each node has a version number for conflict detection
 * - Searches proceed optimistically, validating versions after reading
 * - Insertions use node-level versioning to detect and handle conflicts
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
#include <unordered_map>
#include <unordered_set>
#include "common.hpp"

namespace nilvec {

/**
 * @brief HNSW index with fine-grained optimistic locking.
 *
 * Each node has a version number. Operations read versions before and after
 * accessing data to detect concurrent modifications.
 */
template <typename T>
class HNSWFineOptimistic {
 public:
  /**
   * @brief Construct an HNSW index.
   * @param dim Vector dimensionality
   * @param M Number of neighbors per node (per layer)
   * @param ef_construction Size of dynamic candidate list during construction
   * @param mL Level generation factor (typically 1/ln(M))
   * @param max_retries Maximum retries per node on conflict
   */
  HNSWFineOptimistic(Dim dim,
                     size_t M = 16,
                     size_t ef_construction = 200,
                     float mL = 0.0f,
                     size_t max_retries = 5)
      : dim_(dim),
        M_(M),
        M_max0_(2 * M),
        ef_construction_(ef_construction),
        mL_(mL > 0 ? mL : 1.0f / std::log(static_cast<float>(M))),
        max_retries_(max_retries),
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

    // Allocate new node
    NodeId new_id;
    int new_level;
    {
      std::unique_lock global_lock(global_mutex_);
      new_id = static_cast<NodeId>(nodes_.size());
      new_level = random_level(mL_);

      auto node = std::make_unique<Node>();
      node->vector.assign(data.begin(), data.end());
      node->neighbors.resize(new_level + 1);
      node->version.store(0, std::memory_order_relaxed);
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

    // Phase 1: Greedy descent through upper layers
    for (int level = curr_max_level; level > new_level; --level) {
      curr_entry = greedy_search_layer_optimistic(data, curr_entry, level);
    }

    // Phase 2: Search and connect at lower layers
    for (int level = std::min(new_level, curr_max_level); level >= 0; --level) {
      auto candidates =
          search_layer_optimistic(data, curr_entry, ef_construction_, level);

      size_t M_layer = (level == 0) ? M_max0_ : M_;
      auto neighbors = select_neighbors(candidates, M_layer);

      // Set new node's neighbors (no conflict possible, it's new)
      {
        std::unique_lock lock(nodes_[new_id]->mutex);
        nodes_[new_id]->neighbors[level] = neighbors;
        nodes_[new_id]->version.fetch_add(1, std::memory_order_release);
      }

      // Add reverse edges with optimistic updates
      for (NodeId neighbor : neighbors) {
        add_reverse_edge_optimistic(neighbor, new_id, level, M_layer);
      }

      if (!candidates.empty()) {
        curr_entry = candidates.front().id;
      }
    }

    // Update entry point and max level if necessary
    if (new_level > curr_max_level) {
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

    // Phase 1: Greedy descent through upper layers
    for (int level = curr_max_level; level >= 1; --level) {
      curr_entry = greedy_search_layer_optimistic(query, curr_entry, level);
    }

    // Phase 2: Beam search at layer 0
    auto candidates = search_layer_optimistic(query, curr_entry, ef, 0);

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

  /**
   * @brief Get conflict statistics (only meaningful if NILVEC_TRACK_CONFLICTS
   * is defined).
   */
  const ConflictStats& conflict_stats() const { return conflict_stats_; }

  /**
   * @brief Reset conflict statistics.
   */
  void reset_conflict_stats() { conflict_stats_.reset(); }

 private:
  /**
   * @brief Node structure with version for optimistic concurrency.
   */
  struct Node {
    std::vector<T> vector;
    std::vector<std::vector<NodeId>> neighbors;
    std::atomic<uint64_t> version{0};
    mutable std::shared_mutex mutex;  // Fallback for pessimistic path
  };

  /**
   * @brief Read node's neighbors with version validation.
   * @return Neighbors and version if read was consistent, nullopt otherwise.
   */
  std::optional<std::pair<std::vector<NodeId>, uint64_t>>
  read_neighbors_optimistic(NodeId node, int level) const {
    // Get stable pointer to node under read lock
    Node* node_ptr;
    {
      std::shared_lock lock(global_mutex_);
      if (node >= nodes_.size())
        return std::nullopt;
      node_ptr = nodes_[node].get();
    }

    uint64_t v1 = node_ptr->version.load(std::memory_order_acquire);

    // Check if version indicates write in progress (odd = write in progress)
    if (v1 & 1)
      return std::nullopt;

    std::vector<NodeId> neighbors;
    {
      // Fast path: try to read without lock
      if (level < static_cast<int>(node_ptr->neighbors.size())) {
        neighbors = node_ptr->neighbors[level];
      }
    }

    uint64_t v2 = node_ptr->version.load(std::memory_order_acquire);

    if (v1 != v2)
      return std::nullopt;  // Concurrent modification detected

    return std::make_pair(std::move(neighbors), v1);
  }

  /**
   * @brief Read node's neighbors with retry on conflict.
   */
  std::vector<NodeId> read_neighbors_with_retry(NodeId node, int level) const {
    for (size_t retry = 0; retry < max_retries_; ++retry) {
      NILVEC_TRACK_SEARCH_ATTEMPT(conflict_stats_);
      auto result = read_neighbors_optimistic(node, level);
      if (result.has_value()) {
        return std::move(result->first);
      }
      NILVEC_TRACK_SEARCH_CONFLICT(conflict_stats_);
    }

    // Fallback to pessimistic read
    NILVEC_TRACK_SEARCH_ATTEMPT(conflict_stats_);
    Node* node_ptr;
    {
      std::shared_lock glock(global_mutex_);
      if (node >= nodes_.size())
        return {};
      node_ptr = nodes_[node].get();
    }
    std::shared_lock lock(node_ptr->mutex);
    if (level < static_cast<int>(node_ptr->neighbors.size())) {
      return node_ptr->neighbors[level];
    }
    return {};
  }

  /**
   * @brief Greedy search using optimistic reads.
   */
  NodeId greedy_search_layer_optimistic(std::span<const T> query,
                                        NodeId entry_point,
                                        int level) const {
    NodeId current = entry_point;
    float current_dist = compute_distance(query, current);

    bool improved = true;
    while (improved) {
      improved = false;

      auto neighbors = read_neighbors_with_retry(current, level);

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
   * @brief Beam search using optimistic reads.
   */
  std::vector<Candidate> search_layer_optimistic(std::span<const T> query,
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

      auto neighbors = read_neighbors_with_retry(curr_id, level);

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
   * @brief Add reverse edge with optimistic concurrency.
   */
  void add_reverse_edge_optimistic(NodeId node,
                                   NodeId new_neighbor,
                                   int level,
                                   size_t max_size) {
    // Get stable pointer to node under read lock
    Node* node_ptr;
    {
      std::shared_lock lock(global_mutex_);
      if (node >= nodes_.size())
        return;
      node_ptr = nodes_[node].get();
    }

    for (size_t retry = 0; retry <= max_retries_; ++retry) {
      // Try optimistic update
      uint64_t v1 = node_ptr->version.load(std::memory_order_acquire);

      // Check if write in progress
      if (v1 & 1)
        continue;

      // Try to start write (set odd version)
      uint64_t v_write = v1 | 1;
      if (!node_ptr->version.compare_exchange_strong(
              v1, v_write, std::memory_order_acquire)) {
        continue;  // Someone else started modifying
      }

      // Now we have exclusive access via version
      {
        auto& neighbor_list = node_ptr->neighbors[level];
        neighbor_list.push_back(new_neighbor);

        if (neighbor_list.size() > max_size) {
          prune_neighbors_inline(node_ptr, level, max_size);
        }
      }

      // Complete write (set next even version)
      node_ptr->version.store(v_write + 1, std::memory_order_release);
      return;
    }

    // Fallback to pessimistic
    std::unique_lock lock(node_ptr->mutex);

    // Mark write in progress
    uint64_t v = node_ptr->version.fetch_or(1, std::memory_order_acquire);

    auto& neighbor_list = node_ptr->neighbors[level];
    neighbor_list.push_back(new_neighbor);

    if (neighbor_list.size() > max_size) {
      prune_neighbors_inline(node_ptr, level, max_size);
    }

    // Complete write
    node_ptr->version.store((v | 1) + 1, std::memory_order_release);
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
   * @brief Prune neighbors inline (node must be exclusively accessed).
   */
  void prune_neighbors_inline(Node* node_ptr, int level, size_t max_size) {
    auto& neighbor_list = node_ptr->neighbors[level];

    if (neighbor_list.size() <= max_size)
      return;

    std::span<const T> node_vec(node_ptr->vector);
    std::vector<Candidate> candidates;
    candidates.reserve(neighbor_list.size());

    for (NodeId neighbor : neighbor_list) {
      // Get stable pointer to neighbor node
      Node* neighbor_ptr;
      {
        std::shared_lock lock(global_mutex_);
        if (neighbor >= nodes_.size())
          continue;
        neighbor_ptr = nodes_[neighbor].get();
      }
      float dist =
          squared_distance(node_vec, std::span<const T>(neighbor_ptr->vector));
      candidates.push_back({neighbor, dist});
    }

    std::sort(candidates.begin(), candidates.end());

    neighbor_list.clear();
    neighbor_list.reserve(max_size);
    for (size_t i = 0; i < max_size && i < candidates.size(); ++i) {
      neighbor_list.push_back(candidates[i].id);
    }
  }

  /**
   * @brief Compute distance from query to a node.
   */
  float compute_distance(std::span<const T> query, NodeId node) const {
    // Get stable pointer to node under read lock
    Node* node_ptr;
    {
      std::shared_lock lock(global_mutex_);
      if (node >= nodes_.size())
        return std::numeric_limits<float>::max();
      node_ptr = nodes_[node].get();
    }

    // Vectors are immutable after insertion, so we can read without version
    // check
    return squared_distance(query, std::span<const T>(node_ptr->vector));
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
  size_t max_retries_;

  // Graph structure
  std::atomic<NodeId> entry_point_;
  std::atomic<int> max_level_;
  std::vector<std::unique_ptr<Node>> nodes_;

  // Global mutex for node allocation
  mutable std::shared_mutex global_mutex_;

  // Conflict tracking
  mutable ConflictStats conflict_stats_;

  // Random number generator
  mutable std::mt19937 rng_;
  mutable std::mutex rng_mutex_;
};

}  // namespace nilvec
