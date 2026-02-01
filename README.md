# NilVec

Concurrent Vector Index Implementations for ANN Search.

## Building

```bash
# Basic build
meson setup builddir
meson compile -C builddir

# With HDF5 support for standard benchmark datasets
meson setup builddir -Duse_hdf5=true
meson compile -C builddir

```

## Python Bindings

Python bindings are built automatically if `pybind11` is found.

```bash
# Install dependencies
source .venv/bin/activate
pip install pybind11

# Build
meson setup builddir
meson compile -C builddir

# Use in Python
export PYTHONPATH=$PYTHONPATH:$(pwd)/builddir
python3 -c "import nilvec; index = nilvec.FlatVanilla(128)"
```

## Running Benchmarks

```bash
./builddir/NilVec
```

## Generating Plots

Generate recall vs throughput plots from benchmark results:

```bash
# Create virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib

# Run benchmarks and generate plots
./builddir/NilVec 2>&1 | python tools/plot_results.py

# Or save results and plot separately
./builddir/NilVec > results.txt 2>&1
python tools/plot_results.py results.txt

# Save plots without displaying
python tools/plot_results.py --no-show results.txt
```

Plots are saved to `plots/` directory:
- `hnsw_recall_vs_qps.png/pdf` - HNSW implementations comparison
- `ivfflat_recall_vs_qps.png/pdf` - IVFFlat implementations comparison  
- `combined_recall_vs_qps.png/pdf` - Side-by-side comparison
- `concurrency_comparison.png` - Concurrency strategy comparison

## Formatting

```bash
clang-format -i -- **.cpp **.h **.hpp
```

## Paper

```bash
typst watch paper/nilvec.typ --open
```
