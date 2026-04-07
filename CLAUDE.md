# NilVec Concurrency Benchmarks

Concurrent ANN (approximate nearest neighbor) vector index implementations benchmarked against each other and external libraries (FAISS, USearch, hnswlib, Weaviate, Redis).

## Build & Run

```bash
# Install Python deps and build C++ extension
uv run pytest          # runs test suite (always use uv)

# Build C++ extension manually (needed after src/ changes)
uv run meson setup builddir --wipe
uv run meson compile -C builddir

# Run benchmarks
uv run python main.py --limit 10000 --skip-recall
uv run python main.py --all                        # all datasets
uv run python main.py --dataset sift-128-euclidean.hdf5
```

## Project Structure

```
main.py                  # Runner: _run_single_dataset + run_benchmark (CLI entry point)
nilvec/
  __init__.py            # Re-exports C++ extension classes (do not add logic here)
  _nilvec                # Compiled C++ extension (built by Meson into builddir/nilvec/)
  config.py              # Mutable runtime globals: DIM, K, THREAD_COUNTS, NUM_VECTORS, etc.
  formatting.py          # Terminal color output helpers, RW band parsing, _format_elapsed
  datasets.py            # DATASETS dict, download_dataset, load_dataset, generate_data
  metrics.py             # compute_recall, get_ground_truth, Redis connectivity helpers
  store.py               # BenchmarkResultsStore — DuckDB-backed results persistence
  external_indexes.py    # Wrappers for FAISS, USearch, Milvus, Weaviate, Redis, hnswlib
  benchmarks.py          # benchmark_recall_vs_qps, benchmark_throughput_vs_threads
  cli.py                 # build_parser() — argparse definitions only, no execution
src/
  python_bindings.cpp    # pybind11 bindings for all C++ index classes
  *.hpp                  # C++ index implementations (header-only)
plotting/                # Matplotlib plotting: recall_vs_qps, throughput, conflict_rate
test/
  test_bindings.py       # Smoke tests for all index classes
  test_parallelism.py    # GIL-release verification under concurrent insert/search
paper/plots/             # Generated SVG benchmark plots (gitignored except select ones)
data/                    # HDF5 dataset files (downloaded on first run, not committed)
```

## C++ Index Families

All indexes implement `insert(vec)`, `search(query, k)`, and optionally `conflict_stats()`.

- **HNSW**: `HNSWVanilla`, `HNSWCoarseOptimistic`, `HNSWCoarsePessimistic`, `HNSWFineOptimistic`, `HNSWFinePessimistic`
- **IVFFlat**: `IVFFlatVanilla`, `IVFFlatCoarseOptimistic`, `IVFFlatCoarsePessimistic`, `IVFFlatFineOptimistic`, `IVFFlatFinePessimistic`
- **Hybrid**: `HybridOptimistic`, `HybridPessimistic`
- **Flat**: `FlatVanilla` (brute-force, used for ground truth)
- **SQ8 variants**: `HNSWVanillaSQ8`, `HNSWCoarseOptimisticSQ8`, `HNSWFinePessimisticSQ8`, `IVFFlatVanillaSQ8` (int8 scalar-quantized)
- **ScalarQuantizer**: `ScalarQuantizer(dim)` — train + encode float32 → int8

## Key Design Notes

### Runtime globals (`nilvec/config.py`)
`DIM`, `NUM_VECTORS`, `NUM_QUERIES` are overwritten per-dataset in `_run_single_dataset`. All modules read these as `config.DIM` etc. (not imported as `from nilvec.config import DIM`) so mutations are visible at call time.

### SQ8 wrapping
`make_quantized_cls(inner_cls, sq)` in `config.py` wraps SQ8 index classes to accept float32 transparently. Call it after training a `ScalarQuantizer`.

### External index availability
`nilvec/external_indexes.py` exports module-level sentinels (`faiss`, `usearch`, `weaviate`, `redis`, `hnswlib`) that are `None` if the library isn't installed. `main.py` imports and checks these with `if faiss:` guards.

### Results persistence
`BenchmarkResultsStore` (DuckDB) stores runs in `benchmark_results.duckdb`. Cross-pollination injects matching historical results into the current run's plots — controlled by `--cross-pollinate` (on by default).

### Conflict tracking
Enabled at compile time via `-Dtrack_conflicts=true`. The `NILVEC_TRACK_CONFLICTS` preprocessor flag gates conflict stat collection in optimistic indexes. `conflict_stats()` returns `ConflictStats` with `insert_conflict_rate()` and `search_conflict_rate()`.

## Meson Build Options

| Option | Default | Description |
|---|---|---|
| `track_conflicts` | `false` | Enable conflict tracking in optimistic indexes |
| `python_bindings` | `true` | Build pybind11 Python extension |

Enable with: `meson setup builddir -Dtrack_conflicts=true`

## Datasets

Downloaded automatically to `data/` on first run from ann-benchmarks.com:

| Name | Dim | Metric |
|---|---|---|
| sift-128-euclidean | 128 | L2 |
| fashion-mnist-784-euclidean | 784 | L2 |
| glove-25-angular / glove-100-angular | 25/100 | Angular |
| gist-960-euclidean | 960 | L2 |
| nytimes-256-angular | 256 | Angular |
| mnist-784-euclidean | 784 | L2 |

## Benchmark CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `sift-128-euclidean.hdf5` | Dataset path |
| `--all` | false | Run all datasets |
| `--limit N` | 0 (no limit) | Truncate dataset to N vectors |
| `--rw-bands` | `0.01-0.05 0.20-0.50` | Write-ratio bands (ramps linearly across thread counts) |
| `--skip-recall` | false | Skip ANN recall benchmark |
| `--skip-throughput` | false | Skip throughput benchmark |
| `--internal-only` | false | Skip external index benchmarks |
| `--external-only` | false | Skip nilvec index benchmarks |
| `--results-db` | `benchmark_results.duckdb` | DuckDB results file |
| `--cross-pollinate` | true | Inject historical compatible results into plots |
| `--preload-ratio` | 0.5 | Fraction of dataset pre-loaded before threading |
| `--latency-sample-rate` | 0.01 | Fraction of ops timed for p50/p95/p99 |
| `--auto-start-redis` | true | Auto-start Redis Stack via Docker |
