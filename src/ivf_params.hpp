/**
 * @file ivf_params.hpp
 * @brief Shared construction parameters for all IVFFlat index variants.
 */

#pragma once

#include "common.hpp"

namespace nilvec {

/**
 * @brief Construction parameters common to all IVFFlat index variants.
 *
 * Pass this struct to any IVFFlat constructor to configure the index. The
 * @p max_count field enables pre-allocation of internal storage so that
 * insertions up to that count avoid reallocation.
 */
struct IVFParams {
  /// Vector dimensionality.
  Dim dim;
  /// Number of inverted-list clusters.
  size_t nlist = 100;
  /// Number of clusters to probe during search.
  size_t nprobe = 1;
  /// Expected maximum number of vectors. Used to pre-allocate storage.
  /// 0 means no pre-allocation.
  size_t max_count = 0;
};

}  // namespace nilvec
