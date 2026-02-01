/**
 * @file flat_vanilla.hpp
 * @brief Flat (brute-force) index without concurrency control.
 *
 * This is the simplest possible index - it stores all vectors and
 * performs exhaustive search. Provides perfect recall but O(n) search.
 *
 * Baseline for accuracy comparison.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include "common.hpp"

namespace nilvec {

/**
 * @brief Flat (brute-force) index without concurrency control.
 *
 * Single-threaded baseline implementation with perfect recall.
 */
template <typename T>
class FlatVanilla {
 public:
  /**
   * @brief Construct a flat index.
   * @param dim Vector dimensionality
   */
  explicit FlatVanilla(Dim dim) : dim_(dim) {}

  /**
   * @brief Insert a vector into the index.
   * @param data The vector data
   * @return The node ID assigned to this vector
   */
  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);

    NodeId new_id = static_cast<NodeId>(vectors_.size());
    vectors_.emplace_back(data.begin(), data.end());

    return new_id;
  }

  /**
   * @brief Search for k nearest neighbors (brute force).
   * @param query The query vector
   * @param k Number of neighbors to return
   * @return Search results containing IDs and distances
   */
  SearchResult search(std::span<const T> query, size_t k) const {
    if (vectors_.empty()) {
      return SearchResult{};
    }

    // Compute distances to all vectors
    MaxHeap results;
    for (size_t i = 0; i < vectors_.size(); ++i) {
      float dist = squared_distance(query, std::span<const T>(vectors_[i]));

      if (results.size() < k || dist < results.top().distance) {
        results.push({static_cast<NodeId>(i), dist});
        if (results.size() > k) {
          results.pop();
        }
      }
    }

    // Build result
    SearchResult result;
    std::vector<Candidate> sorted_results;
    sorted_results.reserve(results.size());
    while (!results.empty()) {
      sorted_results.push_back(results.top());
      results.pop();
    }
    std::sort(sorted_results.begin(), sorted_results.end());

    result.ids.reserve(sorted_results.size());
    result.distances.reserve(sorted_results.size());
    for (const auto& cand : sorted_results) {
      result.ids.push_back(cand.id);
      result.distances.push_back(cand.distance);
    }

    return result;
  }

  /**
   * @brief Get the number of vectors in the index.
   */
  size_t size() const { return vectors_.size(); }

  /**
   * @brief Get vector dimensionality.
   */
  Dim dim() const { return dim_; }

 private:
  Dim dim_;
  std::vector<std::vector<T>> vectors_;
};

}  // namespace nilvec
