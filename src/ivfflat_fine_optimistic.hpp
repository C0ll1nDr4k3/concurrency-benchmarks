/**
 * @file ivfflat_fine_optimistic.hpp
 * @brief IVFFlat implementation with fine-grained optimistic concurrency.
 *
 * Uses per-node versioning with optimistic reads:
 * - Each vector is stored in a Node with its own version counter
 * - Searches validate per-node versions after reading vector data
 * - Per-bucket versioning protects bucket list reads
 * - Centroids are immutable after training
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>
#include <shared_mutex>
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
   * @brief Insert a vector with per-node versioning.
   */
  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);
    assert(trained_);

    // Construct node: version starts odd (under construction)
    auto node = std::make_unique<Node>();
    node->version.store(1, std::memory_order_relaxed);
    node->vector.assign(data.begin(), data.end());
    // Mark construction complete (even version)
    node->version.store(2, std::memory_order_release);

    // Allocate node ID
    NodeId new_id;
    {
      std::unique_lock lock(nodes_alloc_mutex_);
      new_id = static_cast<NodeId>(nodes_.size());
      nodes_.push_back(std::move(node));
    }

    // Find target bucket (centroids are immutable)
    size_t bucket = find_nearest_centroid(data);

    // Lock bucket and update with versioning
    {
      std::unique_lock lock(bucket_mutexes_[bucket]);
      bucket_versions_[bucket].fetch_add(1, std::memory_order_release);
      buckets_[bucket].push_back(new_id);
      bucket_versions_[bucket].fetch_add(1, std::memory_order_release);
    }

    return new_id;
  }

  /**
   * @brief Search with per-bucket optimistic reads and per-node versioning.
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
    std::shared_lock lock(nodes_mutex_);
    return nodes_.size();
  }

  bool is_trained() const { return trained_; }

  void set_nprobe(size_t nprobe) { nprobe_ = std::min(nprobe, nlist_); }

  size_t nlist() const { return nlist_; }

  const ConflictStats& conflict_stats() const { return conflict_stats_; }
  void reset_conflict_stats() { conflict_stats_.reset(); }

 private:
  /**
   * @brief Per-node storage with version counter.
   */
  struct Node {
    std::vector<T> vector;
    std::atomic<uint64_t> version{0};  // odd = under construction
  };

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

      // Verify bucket version unchanged
      uint64_t version_after =
          bucket_versions_[bucket_idx].load(std::memory_order_acquire);
      if (version_before != version_after) {
        NILVEC_TRACK_SEARCH_CONFLICT(conflict_stats_);
        continue;
      }

      // Gather stable Node* pointers under brief shared lock
      size_t bucket_size = bucket_copy.size();
      std::vector<Node*> node_ptrs(bucket_size);
      {
        std::shared_lock lock(nodes_mutex_);
        for (size_t j = 0; j < bucket_size; ++j) {
          NodeId vec_id = bucket_copy[j];
          if (vec_id < nodes_.size()) {
            node_ptrs[j] = nodes_[vec_id].get();
          } else {
            node_ptrs[j] = nullptr;
          }
        }
      }

      // Process vectors with per-node version checks
      for (size_t j = 0; j < bucket_size; ++j) {
        if (!node_ptrs[j])
          continue;

        // Prefetch ahead
        if (j + 4 < bucket_size && node_ptrs[j + 4]) {
          __builtin_prefetch(node_ptrs[j + 4]->vector.data(), 0, 3);
        }

        Node* node = node_ptrs[j];

        // Check node version (skip if under construction)
        uint64_t node_ver =
            node->version.load(std::memory_order_acquire);
        if (node_ver & 1)
          continue;

        float dist = squared_distance(query, std::span<const T>(node->vector));

        // Verify node version unchanged
        if (node->version.load(std::memory_order_acquire) != node_ver)
          continue;

        if (results.size() < k || dist < results.top().distance) {
          results.push({bucket_copy[j], dist});
          if (results.size() > k) {
            results.pop();
          }
        }
      }
      return;
    }

    // Fallback: locked read of bucket
    std::unique_lock lock(bucket_mutexes_[bucket_idx]);
    std::shared_lock nlock(nodes_mutex_);
    for (NodeId vec_id : buckets_[bucket_idx]) {
      if (vec_id >= nodes_.size())
        continue;
      Node* node = nodes_[vec_id].get();

      float dist = squared_distance(query, std::span<const T>(node->vector));

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
  std::vector<std::unique_ptr<Node>> nodes_;

  mutable std::mutex train_mutex_;
  mutable std::shared_mutex nodes_mutex_;   // shared for pointer reads
  mutable std::mutex nodes_alloc_mutex_;    // exclusive for allocation
  std::unique_ptr<std::atomic<uint64_t>[]>
      bucket_versions_;  // NOLINT(modernize-avoid-c-arrays)
  std::unique_ptr<std::mutex[]>
      bucket_mutexes_;  // NOLINT(modernize-avoid-c-arrays)
  mutable std::mt19937 rng_;
  mutable ConflictStats conflict_stats_;
};

}  // namespace nilvec
