import random
import threading
import time

import numpy as np
from colorama import Fore, Style
from tqdm import tqdm

from nilvec import config
from nilvec.formatting import (
    _format_elapsed,
    format_benchmark_header,
    format_throughput_line,
)
from nilvec.metrics import compute_recall
from nilvec.params import IndexParams


def _unpack_params(index_args, search_params, params):
    """Resolve IndexParams vs legacy (index_args, search_params) arguments."""
    if params is not None:
        return params.construction, params.search_sweep
    if index_args is None:
        index_args = []
    if search_params is None:
        search_params = [{}]
    return index_args, search_params


def _run_search_pass(index, queries, k, search_dict, latency_sample_rate):
    """Execute one search sweep; return (recall_ids_list, latencies_ms)."""
    res_ids_list = []
    latencies_ms = []
    for q in queries:
        sample = latency_sample_rate >= 1.0 or random.random() < latency_sample_rate
        t0 = time.perf_counter() if sample else 0.0
        if "ef" in search_dict:
            res = index.search(q, k, search_dict["ef"])
        else:
            res = index.search(q, k)
        if sample:
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        res_ids_list.append(res.ids)
    return res_ids_list, latencies_ms


def _build_and_insert(index_cls, index_name, construction_args, data, label):
    """Construct index, optionally train (IVF), insert all vectors; return index."""
    if "IVF" in index_name:
        index = index_cls(config.DIM, *construction_args)
        print(f"  {Fore.YELLOW}Training...{Style.RESET_ALL}")
        index.train(data)
    else:
        index = index_cls(config.DIM, *construction_args)

    start = time.time()
    with tqdm(
        total=len(data),
        desc=f"  Insert {index_name} {label}",
        unit="vec",
        dynamic_ncols=True,
    ) as pbar:
        for vec in data:
            index.insert(vec)
            pbar.update(1)
    print(
        f"  {Fore.BLUE}Insert time:{Style.RESET_ALL} "
        f"{Fore.GREEN}{time.time() - start:.2f}s{Style.RESET_ALL}"
    )
    return index


def benchmark_recall_vs_qps(
    index_cls,
    index_name,
    data,
    queries,
    gt,
    k,
    index_args=None,
    search_params=None,
    latency_sample_rate=1.0,
    *,
    params: IndexParams | None = None,
):
    """
    Run Recall vs QPS benchmark.
    search_params: list of dicts, e.g. [{'ef': 10}, {'ef': 20}]

    Prefer passing ``params`` (an IndexParams) instead of separate
    index_args / search_params.

    Builds the index once, then sweeps all search parameters on that
    single build.
    """
    print(
        f"\n{Style.BRIGHT}{Fore.CYAN}Benchmarking Recall vs QPS{Style.RESET_ALL}: "
        f"{Style.BRIGHT}{Fore.MAGENTA}{index_name}{Style.RESET_ALL}"
    )
    recall_start_time = time.time()

    results = []

    try:
        index_args, search_params = _unpack_params(index_args, search_params, params)

        index = _build_and_insert(index_cls, index_name, index_args, data, "")

        for sp in search_params:
            if "nprobe" in sp:
                index.set_nprobe(sp["nprobe"])

            start = time.time()
            res_ids_list, latencies_ms = _run_search_pass(
                index, queries, k, sp, latency_sample_rate
            )
            duration = time.time() - start
            qps = len(queries) / duration
            recall = compute_recall(res_ids_list, gt, k)

            if latencies_ms:
                p50 = float(np.percentile(latencies_ms, 50))
                p95 = float(np.percentile(latencies_ms, 95))
                p99 = float(np.percentile(latencies_ms, 99))
            else:
                p50 = p95 = p99 = None

            print(
                f"  {Fore.WHITE}Params: {sp}{Style.RESET_ALL} -> "
                f"{Fore.BLUE}Recall:{Style.RESET_ALL} {Fore.GREEN}{recall:.4f}{Style.RESET_ALL}, "
                f"{Fore.BLUE}QPS:{Style.RESET_ALL} {Fore.GREEN}{qps:.0f}{Style.RESET_ALL}"
                + (
                    f", {Fore.BLUE}p50/p95/p99:{Style.RESET_ALL} "
                    f"{Fore.GREEN}{p50:.2f}/{p95:.2f}/{p99:.2f}ms{Style.RESET_ALL}"
                    if p50 is not None
                    else ""
                )
            )
            results.append((recall, qps, p50, p95, p99))
    except KeyboardInterrupt:
        print(f"\nSkipping {index_name}: interrupted")
        return None

    recall_elapsed = time.time() - recall_start_time
    print(
        f"  {Fore.WHITE}Elapsed time for {Style.BRIGHT}{index_name}{Style.RESET_ALL}"
        f"{Fore.WHITE}: {_format_elapsed(recall_elapsed)}{Style.RESET_ALL}"
    )
    return results


def benchmark_throughput_vs_threads(
    index_cls,
    index_name,
    data,
    queries,
    k,
    index_args=None,
    latency_sample_rate=0.0,
    *,
    params: IndexParams | None = None,
    preload_ratio: float = 0.5,
):
    """Run throughput vs threads using a fixed thread-count-driven R/W split.

    Prefer passing ``params`` (an IndexParams) instead of index_args.

    ``preload_ratio`` (0.0–1.0) controls what fraction of ``data`` is inserted
    single-threaded before the concurrent phase begins.  This avoids
    re-inserting the full dataset from scratch at every thread count while
    ensuring search threads hit a realistically populated index.
    """
    index_args, search_params = _unpack_params(index_args, None, params)
    # Standardize throughput to the highest-recall operating point from the
    # provided search sweep.
    fixed_search = search_params[-1] if search_params else {}

    print(format_benchmark_header(index_name))
    index_start_time = time.time()
    results = []
    conflict_rates = []
    build_times = []
    latency_data = []
    disjoint_data = []  # list of (N, [rate_layer0, rate_layer1, ...]) per thread count

    # Split data into a preload portion (inserted once, single-threaded,
    # unmeasured) and a concurrent portion (inserted by worker threads during
    # the measured phase).
    preload_ratio = max(0.0, min(1.0, preload_ratio))
    preload_n = int(len(data) * preload_ratio)
    preload_data = data[:preload_n]
    concurrent_data = data[preload_n:]

    # Pre-compute a fixed sampling stride for the whole run.
    if latency_sample_rate >= 1.0:
        sample_interval = 1
    elif latency_sample_rate > 0.0:
        sample_interval = max(1, round(1.0 / latency_sample_rate))
    else:
        sample_interval = 0  # disabled

    stop_event = threading.Event()

    # Build and preload once — reused across all thread-count iterations.
    if "IVF" in index_name:
        index = index_cls(config.DIM, *index_args)
        index.train(data)
    else:
        index = index_cls(config.DIM, *index_args)

    # Vanilla indexes are intentionally non-thread-safe (the point is to show
    # data races during the measured phase).  Preload must be correct, so for
    # those we stay single-threaded; everything else uses a thread pool since
    # nilvec's concurrent indexes release the GIL in C++.
    preload_threads = 1 if "Vanilla" in index_name else max(config.THREAD_COUNTS)

    preload_start = time.time()
    if preload_n > 0:
        with tqdm(
            total=preload_n,
            desc=f"  Preload {index_name} [T={preload_threads}]",
            unit="vec",
            dynamic_ncols=True,
        ) as pbar:
            if preload_threads == 1:
                for vec in preload_data:
                    index.insert(vec)
                    pbar.update(1)
            else:

                def _preload_worker(vecs):
                    for v in vecs:
                        index.insert(v)
                        pbar.update(1)

                workers = []
                chunk = preload_n // preload_threads
                for i in range(preload_threads):
                    lo = i * chunk
                    hi = (i + 1) * chunk if i < preload_threads - 1 else preload_n
                    t = threading.Thread(
                        target=_preload_worker, args=(preload_data[lo:hi],)
                    )
                    workers.append(t)
                    t.start()
                for t in workers:
                    t.join()
    preload_time = time.time() - preload_start

    for num_threads in config.THREAD_COUNTS:
        if stop_event.is_set():
            break

        if hasattr(index, "set_num_threads"):
            index.set_num_threads(num_threads)
        if "nprobe" in fixed_search and hasattr(index, "set_nprobe"):
            index.set_nprobe(fixed_search["nprobe"])

        build_time = preload_time
        preload_time = 0.0  # only charge preload to the first iteration
        build_times.append(build_time)
        start_time = time.time()

        threads = []
        num_insert_threads = max(1, num_threads // 4)
        if num_insert_threads >= num_threads:
            num_insert_threads = num_threads - 1
        num_search_threads = num_threads - num_insert_threads

        search_lat_lists = [[] for _ in range(num_search_threads)]
        insert_lat_lists = [[] for _ in range(num_insert_threads)]
        worker_errors = []
        worker_errors_lock = threading.Lock()
        insert_pbar = None

        def _record_worker_error(worker_kind, err):
            with worker_errors_lock:
                worker_errors.append((worker_kind, err))
            stop_event.set()

        def _thread_target(worker_kind, worker_fn, *worker_args):
            try:
                worker_fn(*worker_args)
            except BaseException as e:
                _record_worker_error(worker_kind, e)

        def search_worker(qs, latency_list):
            ops_until_next = (
                random.randint(0, sample_interval - 1) if sample_interval > 0 else -1
            )
            for _ in range(5):
                if stop_event.is_set():
                    break
                for q in qs:
                    if stop_event.is_set():
                        break
                    if ops_until_next == 0:
                        t0 = time.perf_counter()
                        if "ef" in fixed_search:
                            index.search(q, k, fixed_search["ef"])
                        else:
                            index.search(q, k)
                        latency_list.append((time.perf_counter() - t0) * 1000)
                        ops_until_next = sample_interval - 1
                    else:
                        if "ef" in fixed_search:
                            index.search(q, k, fixed_search["ef"])
                        else:
                            index.search(q, k)
                        if ops_until_next > 0:
                            ops_until_next -= 1

        def insert_worker(vecs, latency_list, progress_bar=None):
            ops_until_next = (
                random.randint(0, sample_interval - 1) if sample_interval > 0 else -1
            )
            for v in vecs:
                if stop_event.is_set():
                    break
                if ops_until_next == 0:
                    t0 = time.perf_counter()
                    index.insert(v)
                    latency_list.append((time.perf_counter() - t0) * 1000)
                    ops_until_next = sample_interval - 1
                else:
                    index.insert(v)
                    if ops_until_next > 0:
                        ops_until_next -= 1
                if progress_bar is not None:
                    progress_bar.update(1)

        if num_insert_threads > 0 and len(concurrent_data) > 0:
            insert_pbar = tqdm(
                total=len(concurrent_data),
                desc=f"  Insert {index_name} [T={num_threads}]",
                unit="vec",
                dynamic_ncols=True,
            )
            chunk_s = len(concurrent_data) // num_insert_threads
            for i in range(num_insert_threads):
                start = i * chunk_s
                end = (
                    (i + 1) * chunk_s
                    if i < num_insert_threads - 1
                    else len(concurrent_data)
                )
                t = threading.Thread(
                    target=_thread_target,
                    args=(
                        "insert",
                        insert_worker,
                        concurrent_data[start:end],
                        insert_lat_lists[i],
                        insert_pbar,
                    ),
                )
                threads.append(t)
                t.start()

        if num_search_threads > 0:
            for i in range(num_search_threads):
                t = threading.Thread(
                    target=_thread_target,
                    args=("search", search_worker, queries, search_lat_lists[i]),
                )
                threads.append(t)
                t.start()

        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            print(f"\nSkipping {index_name}: Operation has been terminated")
            stop_event.set()
            for t in threads:
                t.join(timeout=0.1)

            if hasattr(index, "close"):
                index.close()
            return None, None, None, None, None
        finally:
            if insert_pbar is not None:
                insert_pbar.close()

        if worker_errors:
            first_kind, first_err = worker_errors[0]
            if hasattr(index, "close"):
                index.close()
            raise RuntimeError(
                f"worker {first_kind} error at threads={num_threads}: "
                f"{type(first_err).__name__}: {first_err}"
            ) from first_err

        duration = time.time() - start_time

        all_search_lat = [l for lst in search_lat_lists for l in lst]
        all_insert_lat = [l for lst in insert_lat_lists for l in lst]

        def _pcts(lats):
            if not lats:
                return None, None, None
            return (
                float(np.percentile(lats, 50)),
                float(np.percentile(lats, 95)),
                float(np.percentile(lats, 99)),
            )

        s_p50, s_p95, s_p99 = _pcts(all_search_lat)
        i_p50, i_p95, i_p99 = _pcts(all_insert_lat)
        latency_data.append((s_p50, s_p95, s_p99, i_p50, i_p95, i_p99))

        total_inserts = (
            len(concurrent_data)
            if (num_insert_threads > 0 and len(concurrent_data) > 0)
            else 0
        )
        total_searches = num_search_threads * len(queries) * 5

        total_ops = total_inserts + total_searches
        throughput = total_ops / duration
        prev = results[-1] if results else None
        print(
            format_throughput_line(
                num_threads,
                num_insert_threads,
                num_search_threads,
                throughput,
                prev,
                build_time,
                search_latencies=(s_p50, s_p95, s_p99) if s_p50 is not None else None,
                insert_latencies=(i_p50, i_p95, i_p99) if i_p50 is not None else None,
            )
        )
        results.append(throughput)

        if hasattr(index, "conflict_stats"):
            stats = index.conflict_stats()
            rate = max(stats.insert_conflict_rate(), stats.search_conflict_rate())
            conflict_rates.append(rate)
        else:
            conflict_rates.append(0)

        if hasattr(index, "disjoint_rates"):
            rates = list(index.disjoint_rates())
            disjoint_data.append((int(index.size()), rates))
        else:
            disjoint_data.append((0, []))

    if hasattr(index, "close"):
        index.close()

    index_elapsed = time.time() - index_start_time
    print(
        f"  {Fore.WHITE}Elapsed time for {Style.BRIGHT}{index_name}{Style.RESET_ALL}"
        f"{Fore.WHITE}: {_format_elapsed(index_elapsed)}{Style.RESET_ALL}"
    )
    return results, conflict_rates, build_times, latency_data, disjoint_data
