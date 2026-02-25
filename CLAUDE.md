# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Writing Guidelines

Never use em dashes.

Never make a claim about anything external without first citing specifically where and how it supports said claim.

Avoid adding redundant expositions that experts would already be familiar with.

## Build & Development

This is a C++23 project with Python bindings (pybind11), built with Meson and managed with `uv`.

```bash
# Install Python deps (editable, with dev tools)
uv pip install -e ".[full]"

# Build C++ code and Python extension module
meson setup builddir -Duse_hdf5=true
meson compile -C builddir

# Make the extension importable (needed for benchmarks and manual usage)
export PYTHONPATH=$PYTHONPATH:$(pwd)/builddir
```

The `test/conftest.py` adds `builddir` to `sys.path` automatically, so `export PYTHONPATH` is not needed for pytest.

## Testing

```bash
uv run pytest                              # all tests
uv run pytest test/test_bindings.py        # single file
uv run pytest test/test_bindings.py::test_hnsw  # single test
uv run pytest test/test_gil_release.py -v  # verify GIL release (expect ~3-4x speedup)
```

## Formatting

```bash
clang-format -i **/*.cpp **/*.hpp && uv format
```

## Running Benchmarks

```bash
uv run main.py --dataset data/sift-128-euclidean.hdf5               # full run
uv run main.py --dataset data/sift-128-euclidean.hdf5 --skip-recall  # throughput only
uv run main.py --dataset data/sift-128-euclidean.hdf5 --skip-recall --limit 1000  # quick test
uv run python -m nilvec.benchmark --mode both  # threading vs multiprocessing comparison
```

## Paper

```bash
typst watch paper/nilvec.typ --open
```

## Architecture

**NilVec** is a benchmarking framework for concurrent approximate nearest neighbor (ANN) search indexes. It implements two index families (HNSW and IVFFlat) with multiple concurrency strategies, then benchmarks them against external libraries (FAISS, USearch, Weaviate, Qdrant, Redis).

### C++ Index Implementations (`src/`)

All indexes are header-only C++ templates in the `nilvec` namespace. Each index type has variants with different concurrency control strategies:

- **Vanilla** -- single-threaded baseline, no locking
- **Coarse Pessimistic** -- per-layer `shared_mutex` (readers-writer lock)
- **Coarse Optimistic** -- per-layer version numbers, retry on conflict
- **Fine Pessimistic** -- per-node `shared_mutex`
- **Fine Optimistic** -- per-node version numbers, retry on conflict

Index families:
- `hnsw_*.hpp` -- Hierarchical Navigable Small World graph
- `ivfflat_*.hpp` -- Inverted File with Flat (brute-force) quantizer; requires `train()` before `insert()`
- `flat_vanilla.hpp` -- brute-force linear scan (used for ground truth computation)

`common.hpp` contains shared types (`NodeId`, `Candidate`, `SearchResult`, `ConflictStats`), SIMD-accelerated distance functions (AVX2/NEON), and conflict tracking macros (compiled in when `-DNILVEC_TRACK_CONFLICTS` is set via the `track_conflicts` meson option).

### Python Bindings (`src/python_bindings.cpp`)

pybind11 module `_nilvec`. All performance-critical methods (`insert`, `search`, `train`) release the GIL via `py::call_guard<py::gil_scoped_release>()`, enabling true multithreaded parallelism from Python.

### Benchmark Framework (`nilvec/benchmark.py`)

`nilvec/benchmark.py` is the main benchmark driver. It:
- Wraps external libraries (FAISS, USearch, Milvus, Weaviate, Qdrant, Redis) in a common interface
- Runs throughput-vs-threads and recall-vs-QPS benchmarks
- Supports threading and multiprocessing concurrency modes
- Persists results to DuckDB and can cross-pollinate historical results across runs
- Generates SVG plots to `paper/plots/`

### Meson Build Options (`meson_options.txt`)

- `track_conflicts` (default: true) -- compile conflict counters into optimistic indexes
- `use_hdf5` (default: true) -- HDF5 dataset loading support
- `python_bindings` (default: true) -- build the `_nilvec` Python extension
