# NilVec

Concurrent Vector Index Implementations for ANN Search.

## Requirements

*   **C++ Compiler**: GCC or Clang with C++17 support.
*   **Build System**: Meson and Ninja.
*   **Python**: Python 3.12+
*   **Package Manager**: `uv` (recommended) or `pip`.
*   **Libraries**: OpenMP (optional, for some baselines), HDF5 (for reading datasets).

## Setup

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install Python Dependencies**:
    ```bash
    uv sync
    ```

3.  **Prepare Data**:
    If you have Git LFS installed, the dataset `data/sift-128-euclidean.hdf5` should be pulled automatically. If not:
    ```bash
    git lfs install
    git lfs pull
    ```
    Alternatively, the benchmark script will attempt to download it if missing.

## Building

The project uses Meson for the C++ build and `uv` for the Python environment.

```bash
# Setup build directory
meson setup builddir -Duse_hdf5=true

# Compile C++ code and Python bindings
meson compile -C builddir
```

## Running Benchmarks

We use `main.py` as the unified entry point for running benchmarks and generating plots.

```bash
# Make sure the build directory is in your PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/builddir

# Run full benchmarks (Throughput and Recall)
uv run main.py --dataset data/sift-128-euclidean.hdf5

# Run only throughput benchmarks (faster)
uv run main.py --dataset data/sift-128-euclidean.hdf5 --skip-recall

# Limit the dataset size for quick testing
uv run main.py --dataset data/sift-128-euclidean.hdf5 --skip-recall --limit 1000
```

### Generated Plots
Plots are saved to the `plots/` directory:
*   `throughput_scaling.png`: Throughput vs. Thread Count.
*   `recall_vs_qps.png`: Recall vs. QPS (if recall benchmark is run).

## Testing

To run the Python binding tests:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/builddir
uv run test/test_bindings.py
```

## Code Formatting

```bash
clang-format -i **/*.cpp **/*.hpp
```

## Paper

```bash
typst watch paper/nilvec.typ --open
```
