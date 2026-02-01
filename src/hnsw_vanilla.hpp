/**
 * @file hnsw_vanilla.hpp
 * @brief Vanilla HNSW implementation without concurrency control.
 *
 * This is a baseline single-threaded implementation for benchmarking.
 * Upper layers use greedy search (single candidate),
 * lower layers use beam search (multiple candidates).
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <vector>
#include "common.hpp"

namespace nilvec {

/**
 * @brief Vanilla HNSW index without concurrency control.
 *
 * Single-threaded baseline implementation for performance comparison.
 */
template <typename T>
class HNSWVanilla {
 public:
  /**
   * @brief Construct an HNSW index.
   * @param dim Vector dimensionality
   * @param M Number of neighbors per node (per layer)
   * @param ef_construction Size of dynamic candidate list during construction
   * @param mL Level generation factor (typically 1/ln(M))
   */
  HNSWVanilla(Dim dim,
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

    NodeId new_id = static_cast<NodeId>(vectors_.size());
    int new_level = random_level(mL_, rng_);

    // Store the vector
    vectors_.emplace_back(data.begin(), data.end());

    // Initialize neighbor lists for all layers
    neighbors_.emplace_back();
    neighbors_[new_id].resize(new_level + 1);

    // Handle first insertion
    if (entry_point_ == INVALID_NODE) {
      entry_point_ = new_id;
      max_level_ = new_level;
      return new_id;
    }

    NodeId curr_entry = entry_point_;

    // Phase 1: Greedy descent through upper layers (above new_level)
    for (int level = max_level_; level > new_level; --level) {
      curr_entry = greedy_search_layer(data, curr_entry, level);
    }

    // Phase 2: Search and connect at layers [min(new_level, max_level), ..., 0]
    for (int level = std::min(new_level, max_level_); level >= 0; --level) {
      auto candidates = search_layer(data, curr_entry, ef_construction_, level);

      size_t M_layer = (level == 0) ? M_max0_ : M_;
      auto neighbors = select_neighbors(candidates, M_layer);

      // Connect new node to neighbors
      neighbors_[new_id][level] = neighbors;

      // Add reverse edges (bidirectional graph)
      for (NodeId neighbor : neighbors) {
        auto& neighbor_list = neighbors_[neighbor][level];
        neighbor_list.push_back(new_id);

        if (neighbor_list.size() > M_layer) {
          prune_neighbors(neighbor, level, M_layer);
        }
      }

      if (!candidates.empty()) {
        curr_entry = candidates.front().id;
      }
    }

    // Update entry point and max level if necessary
    if (new_level > max_level_) {
      entry_point_ = new_id;
      max_level_ = new_level;
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

    if (entry_point_ == INVALID_NODE) {
      return SearchResult{};
    }

    NodeId curr_entry = entry_point_;

    // Phase 1: Greedy descent through upper layers to layer 1
    for (int level = max_level_; level >= 1; --level) {
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
  size_t size() const { return vectors_.size(); }

  /**
   * @brief Get the current maximum level in the graph.
   */
  int max_level() const { return max_level_; }

 private:
  /**
   * @brief Greedy search within a single layer (upper layers).
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

      for (NodeId neighbor : neighbors_[curr_id][level]) {
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

  // Graph structure
  NodeId entry_point_;
  int max_level_;
  std::vector<std::vector<T>> vectors_;
  std::vector<std::vector<std::vector<NodeId>>> neighbors_;

  // Random number generator
  std::mt19937 rng_;
};

}  // namespace nilvec
