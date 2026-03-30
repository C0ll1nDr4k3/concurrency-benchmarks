# NilVec

Concurrent ANN index implementations benchmarked against FAISS, USearch, Milvus, and Weaviate.

## Setup

```sh
uv sync
```

This builds the C++ extension (via meson-python) and installs all Python dependencies.

## Running benchmarks

```sh
uv run python main.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset PATH` | `sift-128-euclidean.hdf5` | HDF5 dataset to benchmark |
| `--all` | off | Run all datasets (overrides `--dataset`) |
| `--limit N` | 0 (no limit) | Cap vector count |
| `--preload-ratio F` | 0.5 | Fraction of dataset pre-loaded before concurrent phase |
| `--rw-ratio F` | — | Fixed write ratio (0.0=read-only, 1.0=write-only) |
| `--rw-bands LOW-HIGH ...` | `0.01-0.05 0.20-0.50` | Write-ratio bands; each runs a separate sweep |
| `--skip-recall` | off | Skip recall measurement |
| `--skip-throughput` | off | Skip throughput measurement |
| `--only-external` | off | Run only external (FAISS/USearch/Milvus/Weaviate) benchmarks |
| `--results-db PATH` | `benchmark_results.duckdb` | DuckDB file for result history |
| `--run-tag LABEL` | — | Tag this run in the results DB |
| `--latency-sample-rate F` | 0.01 | Fraction of queries timed for latency percentiles |

Datasets are downloaded automatically from ann-benchmarks.com on first use:
`sift-128-euclidean`, `glove-25-angular`, `glove-100-angular`, `nytimes-256-angular`, `fashion-mnist-784-euclidean`, `mnist-784-euclidean`, `gist-960-euclidean`.

## Tests

```sh
uv run pytest
```

## Build options

Passed to meson via `uv sync` or `meson setup`:

| Option | Default | Description |
|--------|---------|-------------|
| `track_conflicts` | `false` | Instrument optimistic indexes with conflict counters |
| `python_bindings` | `true` | Build Python extension |

To rebuild with conflict tracking enabled:

```sh
meson setup builddir -Dtrack_conflicts=true --wipe
meson compile -C builddir
```
