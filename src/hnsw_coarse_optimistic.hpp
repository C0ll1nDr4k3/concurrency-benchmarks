/**
 * @file hnsw_coarse_optimistic.hpp
 * @brief HNSW implementation with per-layer optimistic concurrency control.
 *
 * This implementation uses optimistic concurrency at the per-layer level:
 * - Searches proceed with minimal locking, using version numbers per layer
 * - Insertions use per-layer versioning to detect conflicts and retry
 *
 * Per-layer locking provides better concurrency than a single global lock
 * while being simpler than per-node fine-grained locking.
 *
 * Upper layers use greedy search (single candidate),
 * lower layers use beam search (multiple candidates).
 */

#pragma once

#include <algorithm>
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
 * @brief Per-layer state for optimistic concurrency control.
 */
struct LayerState {
  mutable std::shared_mutex mutex;
  std::atomic<uint64_t> version{0};
};

/**
 * @brief HNSW index with per-layer optimistic locking.
 *
 * Uses version numbers per layer to detect concurrent modifications.
 * Operations that detect a conflict on a layer retry that layer's operation.
 */
template <typename T>
class HNSWCoarseOptimistic {
 public:
  /**
   * @brief Construct an HNSW index.
   * @param dim Vector dimensionality
   * @param M Number of neighbors per node (per layer)
   * @param ef_construction Size of dynamic candidate list during construction
   * @param mL Level generation factor (typically 1/ln(M))
   * @param max_retries Maximum number of retries on conflict
   * @param max_layers Maximum number of layers (default: 16)
   */
  HNSWCoarseOptimistic(Dim dim,
                       size_t M = 16,
                       size_t ef_construction = 200,
                       float mL = 0.0f,
                       size_t max_retries = 10,
                       int max_layers = 16)
      : dim_(dim),
        M_(M),
        M_max0_(2 * M),
        ef_construction_(ef_construction),
        mL_(mL > 0 ? mL : 1.0f / std::log(static_cast<float>(M))),
        max_retries_(max_retries),
        max_layers_(max_layers),
        entry_point_(INVALID_NODE),
        max_level_(-1),
        rng_(std::random_device{}()) {
    // Pre-allocate layer states
    layer_states_.reserve(max_layers_);
    for (int i = 0; i < max_layers_; ++i) {
      layer_states_.push_back(std::make_unique<LayerState>());
    }
  }

  /**
   * @brief Insert a vector into the index.
   * @param data The vector data
   * @return The node ID assigned to this vector
   */
  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);

    // Allocate node (requires global lock)
    NodeId new_id;
    int new_level;
    {
      std::unique_lock global_lock(global_mutex_);
      new_id = static_cast<NodeId>(vectors_.size());
      new_level = random_level(mL_, rng_);

      vectors_.emplace_back(data.begin(), data.end());
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
    // Use optimistic reads
    for (int level = curr_max_level; level > new_level; --level) {
      curr_entry = optimistic_greedy_search(data, curr_entry, level);
    }

    // Phase 2: Search and connect at layers [min(new_level, max_level), ..., 0]
    for (int level = std::min(new_level, curr_max_level); level >= 0; --level) {
      NILVEC_TRACK_INSERT_ATTEMPT(conflict_stats_);

      bool success = false;
      for (size_t retry = 0; retry <= max_retries_ && !success; ++retry) {
        auto result =
            try_insert_at_layer(data, new_id, curr_entry, level, new_level);
        if (result.has_value()) {
          curr_entry = result.value();
          success = true;
        } else {
          NILVEC_TRACK_INSERT_CONFLICT(conflict_stats_);
        }
      }

      // Fallback to pessimistic if retries exhausted
      if (!success) {
        curr_entry = pessimistic_insert_at_layer(data, new_id, curr_entry,
                                                 level, new_level);
      }
    }

    // Update entry point and max level if necessary
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

    // Phase 1: Greedy descent through upper layers
    for (int level = curr_max_level; level >= 1; --level) {
      curr_entry = optimistic_greedy_search(query, curr_entry, level);
    }

    // Phase 2: Beam search at layer 0 with optimistic concurrency
    NILVEC_TRACK_SEARCH_ATTEMPT(conflict_stats_);

    for (size_t retry = 0; retry <= max_retries_; ++retry) {
      auto result = try_search_layer0(query, curr_entry, k, ef);
      if (result.has_value()) {
        return result.value();
      }
      NILVEC_TRACK_SEARCH_CONFLICT(conflict_stats_);
    }

    // Fallback to pessimistic search
    return pessimistic_search_layer0(query, curr_entry, k, ef);
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

  /**
   * @brief Get conflict statistics.
   */
  const ConflictStats& conflict_stats() const { return conflict_stats_; }

  /**
   * @brief Reset conflict statistics.
   */
  void reset_conflict_stats() { conflict_stats_.reset(); }

 private:
  /**
   * @brief Optimistic greedy search within a layer.
   * Retries automatically on version mismatch.
   */
  NodeId optimistic_greedy_search(std::span<const T> query,
                                  NodeId entry_point,
                                  int level) const {
    while (true) {
      uint64_t start_version =
          layer_states_[level]->version.load(std::memory_order_acquire);

      NodeId current = entry_point;
      float current_dist = compute_distance(query, current);

      bool improved = true;
      while (improved) {
        improved = false;

        // Check bounds
        if (current >= neighbors_.size() ||
            level >= static_cast<int>(neighbors_[current].size())) {
          break;
        }

        // Take snapshot of neighbors
        std::vector<NodeId> neighbor_snapshot;
        {
          std::shared_lock lock(layer_states_[level]->mutex);
          neighbor_snapshot = neighbors_[current][level];
        }

        size_t num_neighbors = neighbor_snapshot.size();
        for (size_t i = 0; i < num_neighbors; ++i) {
          if (i + 3 < num_neighbors) {
            __builtin_prefetch(vectors_[neighbor_snapshot[i + 3]].data(), 0, 3);
          }

          NodeId neighbor = neighbor_snapshot[i];
          float neighbor_dist = compute_distance(query, neighbor);
          if (neighbor_dist < current_dist) {
            current = neighbor;
            current_dist = neighbor_dist;
            improved = true;
          }
        }
      }

      // Validate version
      if (layer_states_[level]->version.load(std::memory_order_acquire) ==
          start_version) {
        return current;
      }
      // Version changed, retry
    }
  }

  /**
   * @brief Try optimistic insert at a specific layer.
   */
  std::optional<NodeId> try_insert_at_layer(std::span<const T> data,
                                            NodeId new_id,
                                            NodeId entry_point,
                                            int level,
                                            [[maybe_unused]] int new_level) {
    uint64_t start_version =
        layer_states_[level]->version.load(std::memory_order_acquire);

    // Search for neighbors optimistically
    std::vector<Candidate> candidates;
    {
      std::shared_lock lock(layer_states_[level]->mutex);
      candidates = search_layer(data, entry_point, ef_construction_, level);
    }

    // Check version
    if (layer_states_[level]->version.load(std::memory_order_acquire) !=
        start_version) {
      return std::nullopt;
    }

    size_t M_layer = (level == 0) ? M_max0_ : M_;
    auto neighbors = select_neighbors(candidates, M_layer);

    // Acquire exclusive lock and validate + commit
    std::unique_lock lock(layer_states_[level]->mutex);

    if (layer_states_[level]->version.load(std::memory_order_relaxed) !=
        start_version) {
      return std::nullopt;
    }

    // Commit changes
    neighbors_[new_id][level] = neighbors;

    for (NodeId neighbor : neighbors) {
      auto& neighbor_list = neighbors_[neighbor][level];
      neighbor_list.push_back(new_id);

      if (neighbor_list.size() > M_layer) {
        prune_neighbors(neighbor, level, M_layer);
      }
    }

    // Increment version
    layer_states_[level]->version.fetch_add(1, std::memory_order_release);

    return candidates.empty() ? entry_point : candidates.front().id;
  }

  /**
   * @brief Pessimistic fallback for insert at a layer.
   */
  NodeId pessimistic_insert_at_layer(std::span<const T> data,
                                     NodeId new_id,
                                     NodeId entry_point,
                                     int level,
                                     [[maybe_unused]] int new_level) {
    std::unique_lock lock(layer_states_[level]->mutex);

    auto candidates = search_layer(data, entry_point, ef_construction_, level);

    size_t M_layer = (level == 0) ? M_max0_ : M_;
    auto neighbors = select_neighbors(candidates, M_layer);

    neighbors_[new_id][level] = neighbors;

    for (NodeId neighbor : neighbors) {
      auto& neighbor_list = neighbors_[neighbor][level];
      neighbor_list.push_back(new_id);

      if (neighbor_list.size() > M_layer) {
        prune_neighbors(neighbor, level, M_layer);
      }
    }

    layer_states_[level]->version.fetch_add(1, std::memory_order_release);

    return candidates.empty() ? entry_point : candidates.front().id;
  }

  /**
   * @brief Try optimistic search at layer 0.
   */
  std::optional<SearchResult> try_search_layer0(std::span<const T> query,
                                                NodeId entry_point,
                                                size_t k,
                                                size_t ef) const {
    uint64_t start_version =
        layer_states_[0]->version.load(std::memory_order_acquire);

    std::vector<Candidate> candidates;
    {
      std::shared_lock lock(layer_states_[0]->mutex);
      candidates = search_layer(query, entry_point, ef, 0);
    }

    // Validate
    if (layer_states_[0]->version.load(std::memory_order_acquire) !=
        start_version) {
      return std::nullopt;
    }

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
   * @brief Pessimistic fallback for search at layer 0.
   */
  SearchResult pessimistic_search_layer0(std::span<const T> query,
                                         NodeId entry_point,
                                         size_t k,
                                         size_t ef) const {
    std::shared_lock lock(layer_states_[0]->mutex);

    auto candidates = search_layer(query, entry_point, ef, 0);

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
   * @brief Beam search within a single layer.
   * Assumes caller holds at least a shared lock on the layer.
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

      if (curr_id >= neighbors_.size() ||
          level >= static_cast<int>(neighbors_[curr_id].size())) {
        continue;
      }

      std::vector<NodeId> batch_neighbors;
      batch_neighbors.reserve(neighbors_[curr_id][level].size());

      for (NodeId neighbor : neighbors_[curr_id][level]) {
        if (visited.find(neighbor) == visited.end()) {
          visited.insert(neighbor);
          batch_neighbors.push_back(neighbor);
        }
      }

      for (NodeId neighbor : batch_neighbors) {
        __builtin_prefetch(vectors_[neighbor].data(), 0, 3);
      }

      for (NodeId neighbor : batch_neighbors) {
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

  std::vector<NodeId> select_neighbors(const std::vector<Candidate>& candidates,
                                       size_t M) const {
    std::vector<NodeId> result;
    result.reserve(M);
    for (size_t i = 0; i < std::min(M, candidates.size()); ++i) {
      result.push_back(candidates[i].id);
    }
    return result;
  }

  void prune_neighbors(NodeId node, int level, size_t max_size) {
    auto& neighbor_list = neighbors_[node][level];

    if (neighbor_list.size() <= max_size)
      return;

    std::span<const T> node_vec(vectors_[node]);
    std::vector<Candidate> candidates;
    candidates.reserve(neighbor_list.size());

    for (NodeId neighbor : neighbor_list) {
      float dist =
          squared_distance(node_vec, std::span<const T>(vectors_[neighbor]));
      candidates.push_back({neighbor, dist});
    }

    std::sort(candidates.begin(), candidates.end());

    neighbor_list.clear();
    neighbor_list.reserve(max_size);
    for (size_t i = 0; i < max_size; ++i) {
      neighbor_list.push_back(candidates[i].id);
    }
  }

  float compute_distance(std::span<const T> query, NodeId node) const {
    return squared_distance(query, std::span<const T>(vectors_[node]));
  }

  // Configuration
  Dim dim_;
  size_t M_;
  size_t M_max0_;
  size_t ef_construction_;
  float mL_;
  size_t max_retries_;
  int max_layers_;

  // Graph structure
  std::atomic<NodeId> entry_point_;
  std::atomic<int> max_level_;
  std::vector<std::vector<T>> vectors_;
  std::vector<std::vector<std::vector<NodeId>>> neighbors_;

  // Concurrency control - per-layer
  mutable std::shared_mutex global_mutex_;  // For vector/neighbor allocation
  mutable std::vector<std::unique_ptr<LayerState>> layer_states_;
  mutable ConflictStats conflict_stats_;

  // Random number generator
  mutable std::mt19937 rng_;
};

}  // namespace nilvec
