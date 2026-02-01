/**
 * @file ivfflat_coarse_pessimistic.hpp
 * @brief IVFFlat implementation with coarse-grained pessimistic concurrency.
 *
 * Uses a single read-write lock for the entire index.
 * - Readers acquire shared lock
 * - Writers acquire exclusive lock
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <vector>
#include "common.hpp"

namespace nilvec {

/**
 * @brief IVFFlat index with coarse-grained pessimistic locking.
 */
template <typename T>
class IVFFlatCoarsePessimistic {
 public:
  /**
   * @brief Construct an IVFFlat index.
   * @param dim Vector dimensionality
   * @param nlist Number of clusters/buckets
   * @param nprobe Number of buckets to search
   */
  IVFFlatCoarsePessimistic(Dim dim, size_t nlist = 100, size_t nprobe = 1)
      : dim_(dim),
        nlist_(nlist),
        nprobe_(nprobe),
        trained_(false),
        rng_(std::random_device{}()) {
    buckets_.resize(nlist_);
  }

  /**
   * @brief Train the index (requires exclusive lock).
   */
  void train(const std::vector<std::vector<T>>& training_data) {
    std::unique_lock lock(mutex_);

    if (training_data.empty()) {
      return;
    }

    centroids_ = kmeans_plusplus_init(training_data);

    constexpr int max_iterations = 20;
    for (int iter = 0; iter < max_iterations; ++iter) {
      std::vector<std::vector<size_t>> assignments(nlist_);
      for (size_t i = 0; i < training_data.size(); ++i) {
        size_t nearest = find_nearest_centroid(training_data[i]);
        assignments[nearest].push_back(i);
      }

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
   * @brief Insert a vector (requires exclusive lock).
   */
  NodeId insert(std::span<const T> data) {
    std::unique_lock lock(mutex_);
    assert(data.size() == dim_);
    assert(trained_);

    NodeId new_id = static_cast<NodeId>(vectors_.size());
    vectors_.emplace_back(data.begin(), data.end());

    size_t bucket = find_nearest_centroid(data);
    buckets_[bucket].push_back(new_id);

    return new_id;
  }

  /**
   * @brief Search for k nearest neighbors (shared lock).
   */
  SearchResult search(std::span<const T> query, size_t k) const {
    std::shared_lock lock(mutex_);

    if (!trained_ || vectors_.empty()) {
      return SearchResult{};
    }

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

  size_t size() const {
    std::shared_lock lock(mutex_);
    return vectors_.size();
  }

  bool is_trained() const {
    std::shared_lock lock(mutex_);
    return trained_;
  }

  void set_nprobe(size_t nprobe) {
    std::unique_lock lock(mutex_);
    nprobe_ = std::min(nprobe, nlist_);
  }

  size_t nlist() const { return nlist_; }

 private:
  std::vector<std::vector<float>> kmeans_plusplus_init(
      const std::vector<std::vector<T>>& data) {
    std::vector<std::vector<float>> centroids;
    centroids.reserve(nlist_);

    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    size_t first_idx = dist(rng_);
    centroids.emplace_back(data[first_idx].begin(), data[first_idx].end());

    std::vector<float> min_distances(data.size(),
                                     std::numeric_limits<float>::max());

    for (size_t c = 1; c < nlist_; ++c) {
      float total_dist = 0.0f;
      for (size_t i = 0; i < data.size(); ++i) {
        float d = squared_distance(std::span<const T>(data[i]),
                                   std::span<const float>(centroids.back()));
        min_distances[i] = std::min(min_distances[i], d);
        total_dist += min_distances[i];
      }

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

  Dim dim_;
  size_t nlist_;
  size_t nprobe_;
  bool trained_;

  std::vector<std::vector<float>> centroids_;
  std::vector<std::vector<NodeId>> buckets_;
  std::vector<std::vector<T>> vectors_;

  mutable std::shared_mutex mutex_;
  mutable std::mt19937 rng_;
};

}  // namespace nilvec
