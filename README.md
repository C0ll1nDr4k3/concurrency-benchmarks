# Concurrent Vector Index Implementations for ANN Search.

## Requirements

*   **C++ Compiler**: GCC or Clang with C++17 support.
*   **Build System**: Meson and Ninja.
*   **Python**: Python 3.12+
*   **Package Manager**: `uv` (recommended) or `pip`.
*   **Libraries**: OpenMP (optional, for some baselines), HDF5 (for reading datasets).

## Installation

You can install `nilvec` directly from GitHub using pip. This is useful if you want to use the library in your own projects without cloning the repository for development.

```bash
# Basic install
pip install git+https://github.com/C0ll1nDr4k3/concurrency-benchmarks.git

# With FAISS CPU support
pip install "git+https://github.com/C0ll1nDr4k3/concurrency-benchmarks.git#egg=nilvec[cpu]"

# With FAISS GPU support
pip install "git+https://github.com/C0ll1nDr4k3/concurrency-benchmarks.git#egg=nilvec[gpu]"
```

## Setup (For Development)

1.  **Install Python Dependencies**:
    ```bash
    # For CPU
    uv sync --extra cpu

    # For GPU (Linux only)
    uv sync --extra gpu
    ```

2.  **Prepare Data**:
    The benchmark script automatically downloads the dataset (`data/sift-128-euclidean.hdf5`) if it is missing.


## Building

The project uses Meson for the C++ build and `uv` for the Python environment.

```bash
# Setup build directory
meson setup builddir -Duse_hdf5=true

# Compile C++ code and Python bindings
meson compile -C builddir
```

> [!WARNING]
> Windows Troubleshooting
> If you encounter `LNK1104: cannot open file 'kernel32.lib'`, you need to run the build commands from a "x64 Native Tools Command Prompt for VS 2022" or manually activate the environment:
> ```cmd
> call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
> uv sync
> ```


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

## Code Formatting

```bash
clang-format -i **/*.cpp **/*.hpp && uv format
```

## Paper
typst watch paper/nilvec.typ --open
```
