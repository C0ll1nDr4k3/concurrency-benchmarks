/**
 * @file hnsw_params.hpp
 * @brief Shared construction parameters for all HNSW index variants.
 */

#pragma once

#include "common.hpp"

namespace nilvec {

/**
 * @brief Construction parameters common to all HNSW index variants.
 *
 * Pass this struct to any HNSW constructor to configure the index. The
 * @p max_count field enables pre-allocation of internal storage so that
 * insertions up to that count avoid reallocation.
 */
struct HNSWParams {
  /// Vector dimensionality.
  Dim dim;
  /// Max neighbors per node per layer.
  size_t M = 16;
  /// Dynamic candidate list size during construction.
  size_t ef_construction = 200;
  /// Level generation factor (0 → 1/ln(M)).
  float mL = 0.0f;
  /// Expected maximum number of vectors. Used to pre-allocate storage.
  /// 0 means no pre-allocation.
  size_t max_count = 0;
};

}  // namespace nilvec
