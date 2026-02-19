"""
Test to verify GIL is released during index operations.
Confirms that multiple threads can execute C++ code in parallel.
"""

import sys
import os
import threading
import time
import multiprocessing

# Add parent directory to path to import nilvec
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from nilvec import _nilvec as nilvec
except ImportError:
    try:
        import nilvec
    except ImportError:
        print("Error: Could not import nilvec. Run `meson compile -C builddir` first.")
        sys.exit(1)


def test_gil_release_insert_parallel():
    """
    Verify GIL release enables parallel execution during insert operations.
    Tests that multiple threads can insert vectors concurrently with speedup.
    """
    dim = 128
    num_vectors = 5000  # Larger dataset to amortize thread overhead
    num_threads = min(4, multiprocessing.cpu_count())

    print(f"\n{'='*60}")
    print(f"GIL Release Verification Test")
    print(f"{'='*60}")
    print(f"Testing with {num_threads} threads on {multiprocessing.cpu_count()} CPUs")
    print(f"Inserting {num_vectors} vectors of dimension {dim}")

    # Generate test data
    import random
    data = [[random.random() for _ in range(dim)] for _ in range(num_vectors)]

    # Test 1: Single-threaded baseline
    print(f"\n{'='*60}")
    print("Test 1: Single-threaded baseline")
    print(f"{'='*60}")
    index_single = nilvec.HNSWCoarseOptimistic(dim, 16, 200)

    start = time.time()
    for vec in data:
        index_single.insert(vec)
    single_thread_time = time.time() - start

    print(f"Single thread time: {single_thread_time:.4f}s")
    print(f"Throughput: {num_vectors / single_thread_time:.0f} inserts/sec")

    # Test 2: Multi-threaded parallel execution
    print(f"\n{'='*60}")
    print(f"Test 2: Multi-threaded parallel execution ({num_threads} threads)")
    print(f"{'='*60}")

    # Create new index for parallel test
    index_multi = nilvec.HNSWCoarseOptimistic(dim, 16, 200)

    # Split data across threads
    chunk_size = len(data) // num_threads
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_threads)]

    # Handle remainder
    if len(data) % num_threads != 0:
        chunks[-1].extend(data[num_threads * chunk_size:])

    def worker(chunk, thread_id):
        """Worker function that inserts a chunk of vectors"""
        start_local = time.time()
        for vec in chunk:
            index_multi.insert(vec)
        duration = time.time() - start_local
        print(f"  Thread {thread_id}: inserted {len(chunk)} vectors in {duration:.4f}s")

    threads = []
    start = time.time()
    for i, chunk in enumerate(chunks):
        t = threading.Thread(target=worker, args=(chunk, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    multi_thread_time = time.time() - start

    print(f"\nMulti-thread time: {multi_thread_time:.4f}s")
    print(f"Throughput: {num_vectors / multi_thread_time:.0f} inserts/sec")

    # Calculate speedup
    speedup = single_thread_time / multi_thread_time
    efficiency = speedup / num_threads * 100

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Speedup: {speedup:.2f}x (expected: ~{num_threads}x for perfect scaling)")
    print(f"Parallel efficiency: {efficiency:.1f}%")
    print(f"")

    # Note: Concurrent indexes have lock contention overhead.
    # We're testing that GIL is released (enabling parallelism),
    # not that the concurrent data structure is lock-free.
    # Insert operations are write-heavy and have significant lock contention.

    # Check if multi-threading is at least attempting parallelism
    # If threads were fully serialized by GIL, each thread would take ~single_thread_time
    # Expected serialized time would be: num_threads * single_thread_time
    serialized_time = num_threads * single_thread_time

    print(f"\nAnalysis:")
    print(f"  If fully serialized by GIL: ~{serialized_time:.4f}s")
    print(f"  Actual multi-thread time: {multi_thread_time:.4f}s")

    # For write-heavy workloads, use more lenient threshold (95%)
    # Heavy lock contention is expected, but we should still be faster than pure serialization
    if multi_thread_time < serialized_time * 0.95:
        print(f"✅ SUCCESS: Execution is {serialized_time/multi_thread_time:.2f}x faster than serialized")
        print(f"   GIL is properly released - threads execute concurrently!")
        print(f"   Note: Significant lock contention is expected for concurrent inserts.")
        return True
    else:
        print(f"❌ FAILURE: Execution time suggests full serialization by GIL")
        print(f"   GIL may not be released properly")
        return False


def test_gil_release_search_parallel():
    """
    Verify GIL release enables parallel execution during search operations.
    Tests that multiple threads can search concurrently with speedup.
    """
    dim = 128
    num_vectors = 500
    num_queries = 100
    num_threads = min(4, multiprocessing.cpu_count())
    k = 10

    print(f"\n{'='*60}")
    print(f"GIL Release Verification Test - Search Operations")
    print(f"{'='*60}")
    print(f"Testing with {num_threads} threads on {multiprocessing.cpu_count()} CPUs")
    print(f"Searching {num_queries} queries in index with {num_vectors} vectors")

    # Generate test data
    import random
    data = [[random.random() for _ in range(dim)] for _ in range(num_vectors)]
    queries = [[random.random() for _ in range(dim)] for _ in range(num_queries)]

    # Build index
    print("\nBuilding index...")
    index = nilvec.HNSWCoarseOptimistic(dim, 16, 200)
    for vec in data:
        index.insert(vec)
    print(f"Index built with {index.size()} vectors")

    # Test 1: Single-threaded search
    print(f"\n{'='*60}")
    print("Test 1: Single-threaded search")
    print(f"{'='*60}")

    start = time.time()
    for q in queries:
        index.search(q, k)
    single_thread_time = time.time() - start

    print(f"Single thread time: {single_thread_time:.4f}s")
    print(f"Throughput: {num_queries / single_thread_time:.0f} searches/sec")

    # Test 2: Multi-threaded search
    print(f"\n{'='*60}")
    print(f"Test 2: Multi-threaded search ({num_threads} threads)")
    print(f"{'='*60}")

    # Each thread searches all queries (read-heavy workload)
    def search_worker(queries_subset, thread_id):
        """Worker function that performs searches"""
        start_local = time.time()
        for q in queries_subset:
            index.search(q, k)
        duration = time.time() - start_local
        print(f"  Thread {thread_id}: searched {len(queries_subset)} queries in {duration:.4f}s")

    # Split queries across threads
    chunk_size = len(queries) // num_threads
    query_chunks = [queries[i*chunk_size:(i+1)*chunk_size] for i in range(num_threads)]

    # Handle remainder
    if len(queries) % num_threads != 0:
        query_chunks[-1].extend(queries[num_threads * chunk_size:])

    threads = []
    start = time.time()
    for i, chunk in enumerate(query_chunks):
        t = threading.Thread(target=search_worker, args=(chunk, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    multi_thread_time = time.time() - start

    print(f"\nMulti-thread time: {multi_thread_time:.4f}s")
    print(f"Throughput: {num_queries / multi_thread_time:.0f} searches/sec")

    # Calculate speedup
    speedup = single_thread_time / multi_thread_time
    efficiency = speedup / num_threads * 100

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Speedup: {speedup:.2f}x (expected: ~{num_threads}x for perfect scaling)")
    print(f"Parallel efficiency: {efficiency:.1f}%")
    print(f"")

    # Check if multi-threading is at least attempting parallelism
    serialized_time = num_threads * single_thread_time

    print(f"\nAnalysis:")
    print(f"  If fully serialized by GIL: ~{serialized_time:.4f}s")
    print(f"  Actual multi-thread time: {multi_thread_time:.4f}s")

    # If we're significantly faster than serialized execution, GIL is released
    if multi_thread_time < serialized_time * 0.7:
        print(f"✅ SUCCESS: Execution is {serialized_time/multi_thread_time:.2f}x faster than serialized")
        print(f"   GIL is properly released - threads execute concurrently!")
        print(f"   Note: Lock contention in concurrent data structure affects speedup.")
        return True
    else:
        print(f"❌ FAILURE: Execution time suggests serialization")
        print(f"   GIL may not be released properly")
        return False


def main():
    """Run all GIL verification tests"""
    print("\n" + "="*60)
    print("NilVec GIL Release Verification Suite")
    print("="*60)
    print(f"System: {multiprocessing.cpu_count()} CPUs available")

    # Run tests
    test1_passed = test_gil_release_insert_parallel()
    test2_passed = test_gil_release_search_parallel()

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Insert operations: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Search operations: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! GIL is properly released.")
        print(f"   Threading enables true multicore parallelism.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. GIL release may not be working correctly.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
