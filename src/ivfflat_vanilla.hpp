/**
 * @file ivfflat_vanilla.hpp
 * @brief Vanilla IVFFlat implementation without concurrency control.
 *
 * IVFFlat (Inverted File with Flat quantization) partitions vectors into
 * buckets based on their nearest centroid. Search probes the closest
 * nprobe buckets and performs exhaustive search within them.
 *
 * This is a baseline single-threaded implementation for benchmarking.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include "common.hpp"

namespace nilvec {

/**
 * @brief Vanilla IVFFlat index without concurrency control.
 *
 * Single-threaded baseline implementation for performance comparison.
 */
template <typename T>
class IVFFlatVanilla {
 public:
  /**
   * @brief Construct an IVFFlat index.
   * @param dim Vector dimensionality
   * @param nlist Number of clusters/buckets
   * @param nprobe Number of buckets to search (default: 1)
   */
  IVFFlatVanilla(Dim dim, size_t nlist = 100, size_t nprobe = 1)
      : dim_(dim),
        nlist_(nlist),
        nprobe_(nprobe),
        trained_(false),
        rng_(std::random_device{}()) {
    buckets_.resize(nlist_);
  }

  /**
   * @brief Train the index by computing centroids using k-means.
   * @param training_data Vectors to use for training
   */
  void train(const std::vector<std::vector<T>>& training_data) {
    if (training_data.empty()) {
      return;
    }

    // Initialize centroids using k-means++
    centroids_ = kmeans_plusplus_init(training_data);

    // Run k-means iterations
    constexpr int max_iterations = 20;
    for (int iter = 0; iter < max_iterations; ++iter) {
      // Assign points to clusters
      std::vector<std::vector<size_t>> assignments(nlist_);
      for (size_t i = 0; i < training_data.size(); ++i) {
        size_t nearest = find_nearest_centroid(training_data[i]);
        assignments[nearest].push_back(i);
      }

      // Update centroids
      bool changed = false;
      for (size_t c = 0; c < nlist_; ++c) {
        if (assignments[c].empty()) {
          continue;
        }

        std::vector<float> new_centroid(dim_, 0.0f);
        for (size_t idx : assignments[c]) {
          for (Dim d = 0; d < dim_; ++d) {
            new_centroid[d] += static_cast<float>(training_data[idx][d]);
          }
        }
        for (Dim d = 0; d < dim_; ++d) {
          new_centroid[d] /= static_cast<float>(assignments[c].size());
          if (std::abs(new_centroid[d] - centroids_[c][d]) > 1e-6f) {
            changed = true;
          }
          centroids_[c][d] = new_centroid[d];
        }
      }

      if (!changed) {
        break;
      }
    }

    trained_ = true;
  }

  /**
   * @brief Insert a vector into the index.
   * @param data The vector data
   * @return The node ID assigned to this vector
   */
  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);
    assert(trained_ && "Index must be trained before insertion");

    NodeId new_id = static_cast<NodeId>(vectors_.size());
    vectors_.emplace_back(data.begin(), data.end());

    // Find nearest centroid and add to that bucket
    size_t bucket = find_nearest_centroid(data);
    buckets_[bucket].push_back(new_id);

    return new_id;
  }

  /**
   * @brief Search for k nearest neighbors.
   * @param query The query vector
   * @param k Number of neighbors to return
   * @return Search results containing IDs and distances
   */
  SearchResult search(std::span<const T> query, size_t k) const {
    if (!trained_ || vectors_.empty()) {
      return SearchResult{};
    }

    // Find nprobe nearest centroids
    std::vector<Candidate> centroid_candidates;
    centroid_candidates.reserve(nlist_);
    for (size_t c = 0; c < nlist_; ++c) {
      float dist = squared_distance(
          query, std::span<const float>(centroids_[c].data(), dim_));
      centroid_candidates.push_back({static_cast<NodeId>(c), dist});
    }
    std::partial_sort(centroid_candidates.begin(),
                      centroid_candidates.begin() +
                          std::min(nprobe_, centroid_candidates.size()),
                      centroid_candidates.end());

    // Search within the selected buckets
    MaxHeap results;
    for (size_t i = 0; i < std::min(nprobe_, centroid_candidates.size()); ++i) {
      size_t bucket_idx = centroid_candidates[i].id;
      for (NodeId vec_id : buckets_[bucket_idx]) {
        float dist =
            squared_distance(query, std::span<const T>(vectors_[vec_id]));

        if (results.size() < k || dist < results.top().distance) {
          results.push({vec_id, dist});
          if (results.size() > k) {
            results.pop();
          }
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
   * @brief Check if the index is trained.
   */
  bool is_trained() const { return trained_; }

  /**
   * @brief Set the number of buckets to probe during search.
   */
  void set_nprobe(size_t nprobe) { nprobe_ = std::min(nprobe, nlist_); }

  /**
   * @brief Get the number of buckets.
   */
  size_t nlist() const { return nlist_; }

 private:
  /**
   * @brief Initialize centroids using k-means++ algorithm.
   */
  std::vector<std::vector<float>> kmeans_plusplus_init(
      const std::vector<std::vector<T>>& data) {
    std::vector<std::vector<float>> centroids;
    centroids.reserve(nlist_);

    // Choose first centroid randomly
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    size_t first_idx = dist(rng_);
    centroids.emplace_back(data[first_idx].begin(), data[first_idx].end());

    // Choose remaining centroids
    std::vector<float> min_distances(data.size(),
                                     std::numeric_limits<float>::max());

    for (size_t c = 1; c < nlist_; ++c) {
      // Update minimum distances
      float total_dist = 0.0f;
      for (size_t i = 0; i < data.size(); ++i) {
        float d = squared_distance(std::span<const T>(data[i]),
                                   std::span<const float>(centroids.back()));
        min_distances[i] = std::min(min_distances[i], d);
        total_dist += min_distances[i];
      }

      // Sample next centroid proportional to D^2
      std::uniform_real_distribution<float> rdist(0.0f, total_dist);
      float threshold = rdist(rng_);
      float cumsum = 0.0f;
      size_t next_idx = 0;
      for (size_t i = 0; i < data.size(); ++i) {
        cumsum += min_distances[i];
        if (cumsum >= threshold) {
          next_idx = i;
          break;
        }
      }

      centroids.emplace_back(data[next_idx].begin(), data[next_idx].end());
    }

    return centroids;
  }

  /**
   * @brief Find the nearest centroid to a vector.
   */
  template <typename VecType>
  size_t find_nearest_centroid(const VecType& vec) const {
    size_t nearest = 0;
    float min_dist = std::numeric_limits<float>::max();

    for (size_t c = 0; c < nlist_; ++c) {
      float dist;
      if constexpr (std::is_same_v<VecType, std::vector<T>>) {
        dist = squared_distance(std::span<const T>(vec),
                                std::span<const float>(centroids_[c]));
      } else {
        dist = squared_distance(vec, std::span<const float>(centroids_[c]));
      }

      if (dist < min_dist) {
        min_dist = dist;
        nearest = c;
      }
    }

    return nearest;
  }

  // Configuration
  Dim dim_;
  size_t nlist_;
  size_t nprobe_;
  bool trained_;

  // Index structure
  std::vector<std::vector<float>> centroids_;  // [nlist][dim]
  std::vector<std::vector<NodeId>> buckets_;   // [nlist][vectors in bucket]
  std::vector<std::vector<T>> vectors_;        // All vectors

  // Random number generator
  mutable std::mt19937 rng_;
};

}  // namespace nilvec
