# Compile-Time Dimensionality

## The problem

Every distance computation bottleneck reduces to a loop over `dim` elements. At runtime, the compiler cannot unroll this loop or eliminate the scalar tail after the SIMD block, because the trip count is unknown. For a 128-dimensional SIFT vector on NEON, the loop runs 32 iterations of 4 floats each with no remainder; the compiler cannot prove this and emits a branch-guarded scalar fallback on every call.

## The mechanism

`std::span<const T, N>` with a non-dynamic extent encodes the element count in the type. When both arguments to `squared_distance` carry a fixed extent `N`, the compiler sees `constexpr size_t n = N` as the loop bound and can fully unroll the SIMD body and eliminate the remainder branch. No code changes to the SIMD logic are required; the same paths execute, but the optimizer now has the information it needs.

This is the same technique used by nanoflann's `KDTreeSingleIndexAdaptor<..., int32_t DIM = -1>`, the only major ANN library that exposes this knob. Every other library (FAISS, hnswlib, USearch, Annoy, Voyager) treats dimensionality as a runtime constructor argument and therefore cannot offer this optimization.

## Why `-1` as the default

Sentinel value `-1` (exposed as `DynamicDim`) matches nanoflann's convention and makes the default path zero-overhead: `HNSWVanilla<float>` is identical to the pre-existing code. Python bindings always use the default, so no behavioral change crosses the Python boundary. Only C++ callers that know their dimension at compile time opt in.

## Tradeoffs

**Template instantiation cost.** Each distinct value of `D` produces a separate instantiation of the index class. For a library used at one or two fixed dimensions this is negligible; for a library serving arbitrary user dimensions it would be impractical. This codebase targets fixed benchmark datasets (SIFT-128, GloVe-200, GIST-960), so the instantiation count is bounded.

**Storage unchanged.** Vectors are still stored as `std::vector<T>`. Switching to `std::array<T, D>` when `D > 0` would improve cache locality (contiguous fixed-size allocations, no per-vector heap pointer) but changes the storage type and propagates through every container. That is a separate optimization with higher complexity cost.

**No safety regression.** The constructor asserts `dim == D` when `D > 0`, so a mismatch between the template parameter and the runtime argument is caught at debug time rather than producing silently wrong distances.
