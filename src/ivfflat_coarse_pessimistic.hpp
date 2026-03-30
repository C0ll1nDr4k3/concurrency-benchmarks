/**
 * @file ivfflat_coarse_pessimistic.hpp
 * @brief IVFFlat implementation with coarse-grained pessimistic concurrency.
 *
 * Uses per-cluster read-write locks with coarse hold durations:
 * - Searches hold the bucket lock for the full scan (no snapshot/copy)
 * - Insertions lock the vector store, then the target bucket
 * - Centroids are immutable after training
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "common.hpp"

namespace nilvec {

/**
 * @brief IVFFlat index with coarse-grained pessimistic locking.
 */
template <typename T, int32_t D = DynamicDim>
class IVFFlatCoarsePessimistic {
  using Traits = DimTraits<T, D>;

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
    bucket_mutexes_ = std::make_unique<std::shared_mutex[]>(
        nlist_);  // NOLINT(modernize-avoid-c-arrays)
    if constexpr (D > 0) assert(dim == static_cast<Dim>(D));
  }

  /**
   * @brief Train the index (exclusive, single-threaded).
   */
  void train(const std::vector<std::vector<T>>& training_data) {
    std::unique_lock lock(train_mutex_);

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
   * @brief Insert a vector (locks vector store, then target bucket).
   */
  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);
    assert(trained_);

    NodeId new_id;
    {
      std::unique_lock lock(vectors_mutex_);
      new_id = static_cast<NodeId>(vectors_.size());
      vectors_.emplace_back(data.begin(), data.end());
    }

    size_t bucket = find_nearest_centroid(data);

    {
      std::unique_lock lock(bucket_mutexes_[bucket]);
      buckets_[bucket].push_back(new_id);
    }

    return new_id;
  }

  /**
   * @brief Search for k nearest neighbors (per-cluster shared locks).
   *
   * Holds vectors_mutex_ shared for the entire search pass and each
   * bucket lock shared for the full scan of that bucket. This "coarse"
   * hold duration avoids snapshot copies but blocks writers longer than
   * the fine-grained variant.
   */
  SearchResult search(std::span<const T> query, size_t k) const {
    if (!trained_) {
      return SearchResult{};
    }

    // Centroid comparison is lock-free (centroids immutable after training)
    std::vector<Candidate> centroid_candidates;
    centroid_candidates.reserve(nlist_);
    for (size_t c = 0; c < nlist_; ++c) {
      float dist = squared_distance(
          Traits::make_span(query), Traits::make_span(centroids_[c]));
      centroid_candidates.push_back({static_cast<NodeId>(c), dist});
    }
    std::partial_sort(centroid_candidates.begin(),
                      centroid_candidates.begin() +
                          std::min(nprobe_, centroid_candidates.size()),
                      centroid_candidates.end());

    // Hold vectors lock for the entire search pass
    std::shared_lock vec_lock(vectors_mutex_);

    if (vectors_.empty()) {
      return SearchResult{};
    }

    MaxHeap results;
    for (size_t i = 0; i < std::min(nprobe_, centroid_candidates.size()); ++i) {
      size_t bucket_idx = centroid_candidates[i].id;

      // Hold bucket lock for the full scan of this bucket
      std::shared_lock bkt_lock(bucket_mutexes_[bucket_idx]);

      const auto& bucket_vecs = buckets_[bucket_idx];
      size_t bucket_size = bucket_vecs.size();

      for (size_t j = 0; j < bucket_size; ++j) {
        if (j + 4 < bucket_size) {
          NodeId prefetch_id = bucket_vecs[j + 4];
          if (prefetch_id < vectors_.size()) {
            __builtin_prefetch(vectors_[prefetch_id].data(), 0, 3);
          }
        }

        NodeId vec_id = bucket_vecs[j];
        float dist =
            squared_distance(Traits::make_span(query),
                             Traits::make_span(vectors_[vec_id]));

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
    std::shared_lock lock(vectors_mutex_);
    return vectors_.size();
  }

  bool is_trained() const { return trained_; }

  void set_nprobe(size_t nprobe) { nprobe_ = std::min(nprobe, nlist_); }

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
        float d = squared_distance(Traits::make_span(data[i]),
                                   Traits::make_span(centroids.back()));
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
        dist = squared_distance(Traits::make_span(vec),
                                Traits::make_span(centroids_[c]));
      } else {
        dist = squared_distance(Traits::make_span(vec),
                                Traits::make_span(centroids_[c]));
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

  mutable std::mutex train_mutex_;
  mutable std::shared_mutex vectors_mutex_;
  std::unique_ptr<std::shared_mutex[]>
      bucket_mutexes_;  // NOLINT(modernize-avoid-c-arrays)
  mutable std::mt19937 rng_;
};

}  // namespace nilvec
