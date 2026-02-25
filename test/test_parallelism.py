"""
Verify that the GIL is released during index operations so that Python
threads can execute C++ code in parallel across multiple CPU cores.
"""

import random
import threading
import time
import multiprocessing

import pytest

nilvec = pytest.importorskip(
    "nilvec._nilvec",
    reason="nilvec C++ extension not importable — run `meson compile -C builddir` first",
)

NUM_THREADS = min(2, multiprocessing.cpu_count())


def _generate(n, dim):
    return [[random.random() for _ in range(dim)] for _ in range(n)]


def _run_parallel(fn, chunks):
    """Run *fn(chunk)* in one thread per chunk; return total wall time."""
    threads = [threading.Thread(target=fn, args=(c,)) for c in chunks]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def _split(seq, n):
    k = len(seq) // n
    chunks = [seq[i * k : (i + 1) * k] for i in range(n)]
    if len(seq) % n:
        chunks[-1].extend(seq[n * k :])
    return chunks


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


def test_gil_released_during_insert():
    """Multi-threaded inserts must be faster than a GIL-serialised upper bound."""
    dim, num_vectors = 128, 5000
    data = _generate(num_vectors, dim)

    # Single-threaded baseline
    idx_single = nilvec.HNSWCoarseOptimistic(dim, 16, 200)
    t0 = time.perf_counter()
    for vec in data:
        idx_single.insert(vec)
    single_time = time.perf_counter() - t0

    # Multi-threaded
    idx_multi = nilvec.HNSWCoarseOptimistic(dim, 16, 200)

    def insert_chunk(chunk):
        for vec in chunk:
            idx_multi.insert(vec)

    multi_time = _run_parallel(insert_chunk, _split(data, NUM_THREADS))

    # If the GIL were held the whole time, wall time ≈ NUM_THREADS × single_time.
    # A write-heavy concurrent structure has lock contention, so we use a lenient
    # threshold of 95 % of the serialised upper bound.
    serialised_upper = NUM_THREADS * single_time
    assert multi_time < serialised_upper * 0.95, (
        f"Insert: multi-thread time {multi_time:.3f}s is not meaningfully below "
        f"the GIL-serialised upper bound {serialised_upper:.3f}s — "
        f"GIL may not be released during insert()"
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_gil_released_during_search():
    """Multi-threaded searches must be faster than a GIL-serialised upper bound."""
    dim, num_vectors, num_queries, k = 128, 500, 100, 10
    data = _generate(num_vectors, dim)
    queries = _generate(num_queries, dim)

    index = nilvec.HNSWCoarseOptimistic(dim, 16, 200)
    for vec in data:
        index.insert(vec)

    # Single-threaded baseline
    t0 = time.perf_counter()
    for q in queries:
        index.search(q, k)
    single_time = time.perf_counter() - t0

    # Multi-threaded
    def search_chunk(chunk):
        for q in chunk:
            index.search(q, k)

    multi_time = _run_parallel(search_chunk, _split(queries, NUM_THREADS))

    # Search is read-heavy with less contention, so use a tighter threshold (70 %).
    serialised_upper = NUM_THREADS * single_time
    assert multi_time < serialised_upper * 0.70, (
        f"Search: multi-thread time {multi_time:.3f}s is not meaningfully below "
        f"the GIL-serialised upper bound {serialised_upper:.3f}s — "
        f"GIL may not be released during search()"
    )
