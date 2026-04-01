/**
 * @file quantization.hpp
 * @brief Scalar quantization for compressing float32 vectors to int8.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "common.hpp"

namespace nilvec {

/**
 * @brief Symmetric per-dimension scalar quantizer (SQ8).
 *
 * Maps float32 vectors to int8_t by computing per-dimension scale factors
 * from training data. Each component x is mapped to:
 *
 *   q = clamp(round(x / scale[d]), -127, 127)
 *
 * where scale[d] = max(|x[d]|) / 127 across all training vectors.
 * Decoding reverses this: x ≈ q * scale[d].
 *
 * Usage:
 *   ScalarQuantizer sq(dim);
 *   sq.train(training_vectors);
 *   auto encoded = sq.encode(float_vec);   // → vector<int8_t>
 *   auto decoded = sq.decode(encoded);     // → vector<float>
 */
struct ScalarQuantizer {
  explicit ScalarQuantizer(Dim dim) : dim_(dim), scales_(dim, 1.0f) {}

  /**
   * @brief Train the quantizer on a set of float32 vectors.
   *
   * Computes per-dimension scale factors from the maximum absolute value
   * seen in the training data.
   *
   * @param data Training vectors, each of length dim_
   */
  void train(const std::vector<std::vector<float>>& data) {
    assert(!data.empty());
    std::fill(scales_.begin(), scales_.end(), 0.0f);

    for (const auto& vec : data) {
      assert(vec.size() == dim_);
      for (size_t i = 0; i < dim_; ++i) {
        scales_[i] = std::max(scales_[i], std::abs(vec[i]));
      }
    }

    for (size_t i = 0; i < dim_; ++i) {
      // Avoid division by zero: if all values are 0, use scale = 1
      scales_[i] = (scales_[i] > 0.0f) ? scales_[i] / 127.0f : 1.0f;
    }

    trained_ = true;
  }

  /**
   * @brief Encode a float32 vector to int8_t.
   * @param vec Input float vector of length dim_
   * @return Quantized int8_t vector of length dim_
   */
  std::vector<int8_t> encode(std::span<const float> vec) const {
    assert(vec.size() == dim_);
    std::vector<int8_t> result(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      float q = vec[i] / scales_[i];
      q = std::max(-127.0f, std::min(127.0f, std::round(q)));
      result[i] = static_cast<int8_t>(q);
    }
    return result;
  }

  /**
   * @brief Decode an int8_t vector back to approximate float32.
   * @param vec Quantized int8_t vector of length dim_
   * @return Approximate float32 vector of length dim_
   */
  std::vector<float> decode(std::span<const int8_t> vec) const {
    assert(vec.size() == dim_);
    std::vector<float> result(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      result[i] = static_cast<float>(vec[i]) * scales_[i];
    }
    return result;
  }

  bool is_trained() const { return trained_; }
  Dim dim() const { return dim_; }

  /// Per-dimension scale factors (exposed for inspection/serialization).
  const std::vector<float>& scales() const { return scales_; }

 private:
  Dim dim_;
  std::vector<float> scales_;
  bool trained_ = false;
};

}  // namespace nilvec
