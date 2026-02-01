/**
 * @file common.hpp
 * @brief Common types and utilities for ANN index implementations.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <span>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace nilvec {

/// Node identifier type
using NodeId = std::uint32_t;

/// Dimension of vectors
using Dim = std::uint32_t;

/// Invalid node sentinel
constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();

/**
 * @brief Compute squared Euclidean distance between two vectors.
 */
/**
 * @brief Compute squared Euclidean distance between two vectors.
 */
template <typename T>
inline float squared_distance(std::span<const T> a, std::span<const T> b) {
  float dist = 0.0f;
  size_t n = a.size();
  size_t i = 0;

  // SIMD Optimization for float
  if constexpr (std::is_same_v<T, float>) {
#if defined(__aarch64__) || defined(_M_ARM64)  // NEON (ARM64)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
      float32x4_t a_vec = vld1q_f32(&a[i]);
      float32x4_t b_vec = vld1q_f32(&b[i]);
      float32x4_t diff = vsubq_f32(a_vec, b_vec);
      sum_vec = vmlaq_f32(sum_vec, diff, diff);
    }
    dist += vaddvq_f32(sum_vec);

#elif defined(__x86_64__) || defined(_M_X64)   // AVX2 (x86_64)
    // Check for AVX support (compile-time check mainly, assuming -march=native)
#ifdef __AVX2__
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
      __m256 a_vec = _mm256_loadu_ps(&a[i]);
      __m256 b_vec = _mm256_loadu_ps(&b[i]);
      __m256 diff = _mm256_sub_ps(a_vec, b_vec);
      sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 hsum = _mm_hadd_ps(sum128, sum128);
    hsum = _mm_hadd_ps(hsum, hsum);
    dist += _mm_cvtss_f32(hsum);
#endif
#endif
  }

  // Scalar fallback (and generic types)
  for (; i < n; ++i) {
    float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
    dist += diff * diff;
  }
  return dist;
}

/**
 * @brief Compute Euclidean distance between two vectors.
 */
template <typename T>
inline float distance(std::span<const T> a, std::span<const T> b) {
  return std::sqrt(squared_distance(a, b));
}

/**
 * @brief A candidate node with its distance to the query.
 */
struct Candidate {
  NodeId id;
  float distance;

  bool operator<(const Candidate& other) const {
    return distance < other.distance;
  }

  bool operator>(const Candidate& other) const {
    return distance > other.distance;
  }
};

/**
 * @brief Min-heap of candidates (closest first).
 */
using MinHeap = std::
    priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>;

/**
 * @brief Max-heap of candidates (farthest first).
 */
using MaxHeap = std::
    priority_queue<Candidate, std::vector<Candidate>, std::less<Candidate>>;

/**
 * @brief Generate a random level for a new node in HNSW.
 *
 * The level is drawn from a geometric distribution with parameter mL.
 */
inline int random_level(float mL, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);
  return static_cast<int>(-std::log(r) * mL);
}

/**
 * @brief Result of a k-NN search query.
 */
struct SearchResult {
  std::vector<NodeId> ids;
  std::vector<float> distances;
};

/**
 * @brief Conflict statistics for optimistic concurrency control.
 *
 * Only tracked when NILVEC_TRACK_CONFLICTS is defined.
 */
struct ConflictStats {
  std::atomic<uint64_t> insert_attempts{0};
  std::atomic<uint64_t> insert_conflicts{0};
  std::atomic<uint64_t> search_attempts{0};
  std::atomic<uint64_t> search_conflicts{0};

  void reset() {
    insert_attempts.store(0, std::memory_order_relaxed);
    insert_conflicts.store(0, std::memory_order_relaxed);
    search_attempts.store(0, std::memory_order_relaxed);
    search_conflicts.store(0, std::memory_order_relaxed);
  }

  double insert_conflict_rate() const {
    uint64_t attempts = insert_attempts.load(std::memory_order_relaxed);
    if (attempts == 0)
      return 0.0;
    return static_cast<double>(
               insert_conflicts.load(std::memory_order_relaxed)) /
           attempts * 100.0;
  }

  double search_conflict_rate() const {
    uint64_t attempts = search_attempts.load(std::memory_order_relaxed);
    if (attempts == 0)
      return 0.0;
    return static_cast<double>(
               search_conflicts.load(std::memory_order_relaxed)) /
           attempts * 100.0;
  }
};

// Macros for conditional conflict tracking
#ifdef NILVEC_TRACK_CONFLICTS
#define NILVEC_TRACK_INSERT_ATTEMPT(stats) \
  (stats).insert_attempts.fetch_add(1, std::memory_order_relaxed)
#define NILVEC_TRACK_INSERT_CONFLICT(stats) \
  (stats).insert_conflicts.fetch_add(1, std::memory_order_relaxed)
#define NILVEC_TRACK_SEARCH_ATTEMPT(stats) \
  (stats).search_attempts.fetch_add(1, std::memory_order_relaxed)
#define NILVEC_TRACK_SEARCH_CONFLICT(stats) \
  (stats).search_conflicts.fetch_add(1, std::memory_order_relaxed)
#else
#define NILVEC_TRACK_INSERT_ATTEMPT(stats) ((void)0)
#define NILVEC_TRACK_INSERT_CONFLICT(stats) ((void)0)
#define NILVEC_TRACK_SEARCH_ATTEMPT(stats) ((void)0)
#define NILVEC_TRACK_SEARCH_CONFLICT(stats) ((void)0)
#endif

}  // namespace nilvec
