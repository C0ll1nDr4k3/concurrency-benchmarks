/**
 * @file ivfflat_fine_optimistic.hpp
 * @brief IVFFlat implementation with fine-grained optimistic concurrency.
 *
 * Uses per-bucket versioning with optimistic reads:
 * - Searches read bucket contents optimistically, retry on version mismatch
 * - Insertions update bucket version using even/odd protocol
 * - Centroids are immutable after training
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <mutex>
#include <numeric>
#include <vector>
#include "common.hpp"

namespace nilvec {

/**
 * @brief IVFFlat index with fine-grained optimistic locking.
 */
template <typename T>
class IVFFlatFineOptimistic {
 public:
  /**
   * @brief Construct an IVFFlat index.
   */
  IVFFlatFineOptimistic(Dim dim,
                        size_t nlist = 100,
                        size_t nprobe = 1,
                        size_t max_retries = 10)
      : dim_(dim),
        nlist_(nlist),
        nprobe_(nprobe),
        trained_(false),
        max_retries_(max_retries),
        rng_(std::random_device{}()) {
    buckets_.resize(nlist_);
    bucket_versions_ = std::make_unique<std::atomic<uint64_t>[]>(
        nlist_);  // NOLINT(modernize-avoid-c-arrays)
    bucket_mutexes_ = std::make_unique<std::mutex[]>(
        nlist_);  // NOLINT(modernize-avoid-c-arrays)
    for (size_t i = 0; i < nlist_; ++i) {
      bucket_versions_[i].store(0, std::memory_order_relaxed);
    }
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
   * @brief Insert a vector with optimistic per-bucket versioning.
   */
  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);
    assert(trained_);

    // Allocate vector ID atomically
    NodeId new_id;
    {
      std::unique_lock lock(vectors_mutex_);
      new_id = static_cast<NodeId>(vectors_.size());
      vectors_.emplace_back(data.begin(), data.end());
    }

    // Find target bucket (centroids are immutable)
    size_t bucket = find_nearest_centroid(data);

    // Lock bucket and update with versioning
    {
      std::unique_lock lock(bucket_mutexes_[bucket]);
      // Mark write in progress (odd)
      bucket_versions_[bucket].fetch_add(1, std::memory_order_release);
      buckets_[bucket].push_back(new_id);
      // Mark write complete (even)
      bucket_versions_[bucket].fetch_add(1, std::memory_order_release);
    }

    return new_id;
  }

  /**
   * @brief Search with optimistic per-bucket reads.
   */
  SearchResult search(std::span<const T> query, size_t k) const {
    if (!trained_) {
      return SearchResult{};
    }

    // Find nearest centroids (no lock needed, centroids immutable)
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

    // Search each selected bucket with optimistic reads
    for (size_t i = 0; i < std::min(nprobe_, centroid_candidates.size()); ++i) {
      size_t bucket_idx = centroid_candidates[i].id;
      search_bucket(bucket_idx, query, k, results);
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
    std::unique_lock lock(vectors_mutex_);
    return vectors_.size();
  }

  bool is_trained() const { return trained_; }

  void set_nprobe(size_t nprobe) { nprobe_ = std::min(nprobe, nlist_); }

  size_t nlist() const { return nlist_; }

  const ConflictStats& conflict_stats() const { return conflict_stats_; }
  void reset_conflict_stats() { conflict_stats_.reset(); }

 private:
  void search_bucket(size_t bucket_idx,
                     std::span<const T> query,
                     size_t k,
                     MaxHeap& results) const {
    NILVEC_TRACK_SEARCH_ATTEMPT(conflict_stats_);
    for (size_t retry = 0; retry < max_retries_; ++retry) {
      uint64_t version_before =
          bucket_versions_[bucket_idx].load(std::memory_order_acquire);

      // Odd version means write in progress
      if (version_before & 1) {
        NILVEC_TRACK_SEARCH_CONFLICT(conflict_stats_);
        continue;
      }

      // Read bucket contents
      std::vector<NodeId> bucket_copy = buckets_[bucket_idx];

      // Verify version unchanged
      uint64_t version_after =
          bucket_versions_[bucket_idx].load(std::memory_order_acquire);
      if (version_before != version_after) {
        NILVEC_TRACK_SEARCH_CONFLICT(conflict_stats_);
        continue;
      }

      // Process vectors
      size_t bucket_size = bucket_copy.size();
      for (size_t i = 0; i < bucket_size; ++i) {
        if (i + 4 < bucket_size) {
           // We can't safely prefetch the vector data here because we don't hold the vectors_mutex_
           // and the vector pointers might be reallocated if vectors_ resize. 
           // However, if vectors_ is stable or we assume resize is rare/handled, we might try.
           // BUT: IVFFlatFineOptimistic takes a copy of the vector under lock.
        }

        NodeId vec_id = bucket_copy[i];
        std::vector<T> vec_copy;
        {
          std::unique_lock lock(vectors_mutex_);
          if (vec_id >= vectors_.size())
            continue;
          
          // Prefetching here is tricky because we copy the vector.
          // Better optimization: Avoid the copy if possible, but that breaks the design.
          // Given the constraints, we will just optimize the distance calc.
          vec_copy = vectors_[vec_id];
        }

        float dist = squared_distance(query, std::span<const T>(vec_copy));

        if (results.size() < k || dist < results.top().distance) {
          results.push({vec_id, dist});
          if (results.size() > k) {
            results.pop();
          }
        }
      }
      return;
    }

    // Fallback to locked read
    std::unique_lock lock(bucket_mutexes_[bucket_idx]);
    for (NodeId vec_id : buckets_[bucket_idx]) {
      std::vector<T> vec_copy;
      {
        std::unique_lock vlock(vectors_mutex_);
        if (vec_id >= vectors_.size())
          continue;
        vec_copy = vectors_[vec_id];
      }

      float dist = squared_distance(query, std::span<const T>(vec_copy));

      if (results.size() < k || dist < results.top().distance) {
        results.push({vec_id, dist});
        if (results.size() > k) {
          results.pop();
        }
      }
    }
  }

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
  size_t max_retries_;

  std::vector<std::vector<float>> centroids_;
  std::vector<std::vector<NodeId>> buckets_;
  std::vector<std::vector<T>> vectors_;

  mutable std::mutex train_mutex_;
  mutable std::mutex vectors_mutex_;
  std::unique_ptr<std::atomic<uint64_t>[]>
      bucket_versions_;  // NOLINT(modernize-avoid-c-arrays)
  std::unique_ptr<std::mutex[]>
      bucket_mutexes_;  // NOLINT(modernize-avoid-c-arrays)
  mutable std::mt19937 rng_;
  mutable ConflictStats conflict_stats_;
};

}  // namespace nilvec
