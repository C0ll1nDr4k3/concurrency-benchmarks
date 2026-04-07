# Test Suite Design

## The Problem

The original tests covered only the C++ extension surface: index construction, insert, search, and removal. That left the Python layer - the benchmark runner, result store, formatting, and CLI - completely untested. Bugs in those components would not be caught until a full benchmark run, which is slow and requires large datasets.

## Scope Decision

The test suite draws a deliberate boundary between what is worth unit testing and what is not.

The C++ indexes are already tested via `test_bindings.py` and `test_parallelism.py`. Adding more unit tests there would just restate what the binding smoke tests already cover. Instead, the new tests focus on the Python modules that contain real logic: input parsing, database persistence, metric computation, and the `make_quantized_cls` wrapper.

External index wrappers (`nilvec/external_indexes.py`) are intentionally excluded. Those wrappers are thin adapters over third-party libraries that may not be installed in every environment, and their correctness is largely a function of the upstream library. Testing that `FaissHNSW` calls `faiss.IndexHNSWFlat` correctly adds little value relative to the maintenance burden.

## Why Pure-Python Logic Needed Its Own Tests

The benchmark runner accumulates results across thread counts and datasets, stores them in DuckDB, and merges historical runs via cross-pollination. These paths have conditional behavior (empty results, missing indexes, incompatible runs) that exercises specific branches. A single end-to-end benchmark run would only exercise one path through that logic.

The DuckDB store tests use an in-memory database (`:memory:`) rather than a temp file. This keeps tests fast and avoids any interaction with the live `benchmark_results.duckdb` file. The trade-off is that the test does not exercise file I/O or the schema migration path for pre-existing databases, but those paths are simple enough that the risk is low.

## Recall Metric Testing

`compute_recall` is the single most important correctness invariant in the codebase - a bug there would silently corrupt all benchmark comparisons. The tests verify edge cases in the denominator logic (`k > len(ground_truth)`, `k < len(ground_truth)`) and confirm the implementation's specific behavior around duplicate results. The function counts each result hit against the truth set independently and does not deduplicate, which is fine for well-formed ANN results but worth making explicit in the tests.

## The Meson Install Gap

The original `meson.build` only installed `nilvec/__init__.py` and the compiled extension into the editable wheel. All other Python modules (`config.py`, `formatting.py`, etc.) existed in the source tree but were invisible to the meson-python import machinery when running tests.

Fixing this required two changes: adding all `nilvec/*.py` files to `install_sources` in `meson.build`, and adding the project root to `sys.path` in `conftest.py`. The second change was necessary because `nilvec/formatting.py` imports from the `plotting` package, which is also source-only and not part of any installed package.

This gap would have affected anyone who tried to import these modules in any test or script relying on the editable install, not just the new tests.

## make_quantized_cls Coverage

The `make_quantized_cls` wrapper is tested through the SQ8 index variants rather than with mocks. Using real `ScalarQuantizer` instances and real SQ8 indexes confirms that the encode-then-forward path actually works end to end, which is the failure mode that matters. The tests check both the `ef=0` (default search) and `ef>0` branches because the wrapper's search method has an explicit branch on `ef`.
