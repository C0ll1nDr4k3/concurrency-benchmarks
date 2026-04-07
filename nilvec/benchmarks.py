import random
import threading
import time

import numpy as np
from colorama import Fore, Style

from nilvec import config
from nilvec.formatting import (
    _format_elapsed,
    format_benchmark_header,
    format_throughput_line,
)
from nilvec.metrics import compute_recall


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
):
    """
    Run Recall vs QPS benchmark.
    search_params: list of dicts, e.g. [{'ef': 10}, {'ef': 20}]
    """
    if index_args is None:
        index_args = []

    print(
        f"\n{Style.BRIGHT}{Fore.CYAN}Benchmarking Recall vs QPS{Style.RESET_ALL}: "
        f"{Style.BRIGHT}{Fore.MAGENTA}{index_name}{Style.RESET_ALL}"
    )
    recall_start_time = time.time()

    if "IVF" in index_name:
        index = index_cls(config.DIM, *index_args)
        print(f"  {Fore.YELLOW}Training...{Style.RESET_ALL}")
        index.train(data)
    else:
        index = index_cls(config.DIM, *index_args)

    print(f"  {Fore.YELLOW}Inserting...{Style.RESET_ALL}")
    start = time.time()
    for vec in data:
        index.insert(vec)
    print(
        f"  {Fore.BLUE}Insert time:{Style.RESET_ALL} {Fore.GREEN}{time.time() - start:.2f}s{Style.RESET_ALL}"
    )

    results = []

    if search_params is None:
        search_params = [{}]

    for params in search_params:
        if "nprobe" in params:
            index.set_nprobe(params["nprobe"])

        start = time.time()
        res_ids_list = []
        latencies_ms = []
        for q in queries:
            t0 = 0.0
            sample = latency_sample_rate >= 1.0 or random.random() < latency_sample_rate
            if sample:
                t0 = time.perf_counter()
            if "ef" in params:
                res = index.search(q, k, params["ef"])
            else:
                res = index.search(q, k)
            if sample:
                latencies_ms.append((time.perf_counter() - t0) * 1000)
            res_ids_list.append(res.ids)

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
            f"  {Fore.WHITE}Params: {params}{Style.RESET_ALL} -> "
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
    rw_ratio=0.5,
    preload_ratio=0.5,
    latency_sample_rate=0.0,
):
    """
    Run Throughput vs Threads benchmark with configurable R/W split.
    rw_ratio: float for a fixed ratio, or list of floats (one per THREAD_COUNTS entry)
              for a ramping schedule across thread counts.
    preload_ratio: fraction of dataset to pre-load during construction (default: 0.8).
    """
    if index_args is None:
        index_args = []

    # Normalize rw_ratio into a per-thread-count schedule
    if isinstance(rw_ratio, (list, tuple)) and not isinstance(
        rw_ratio[0], (int, float)
    ):
        rw_schedule = rw_ratio
    elif isinstance(rw_ratio, list):
        rw_schedule = rw_ratio
    else:
        rw_schedule = [rw_ratio] * len(config.THREAD_COUNTS)

    # Display header with band info if schedule varies
    if rw_schedule[0] != rw_schedule[-1]:
        header_ratio = (rw_schedule[0], rw_schedule[-1])
    else:
        header_ratio = rw_schedule[0]
    print(format_benchmark_header(index_name, header_ratio))
    index_start_time = time.time()
    results = []
    conflict_rates = []
    build_times = []
    latency_data = []

    # Pre-compute a fixed sampling stride for the whole run.
    if latency_sample_rate >= 1.0:
        sample_interval = 1
    elif latency_sample_rate > 0.0:
        sample_interval = max(1, round(1.0 / latency_sample_rate))
    else:
        sample_interval = 0  # disabled

    stop_event = threading.Event()

    for tc_idx, num_threads in enumerate(config.THREAD_COUNTS):
        if stop_event.is_set():
            break

        build_start = time.time()
        if "IVF" in index_name:
            index = index_cls(config.DIM, *index_args)
            index.train(data)
        else:
            index = index_cls(config.DIM, *index_args)

        if hasattr(index, "set_num_threads"):
            index.set_num_threads(num_threads)

        initial_size = int(len(data) * preload_ratio)
        for i in range(initial_size):
            index.insert(data[i])
        build_time = time.time() - build_start
        build_times.append(build_time)

        remaining_data = data[initial_size:]

        ops_count = 0
        start_time = time.time()

        threads = []

        def search_worker(qs, latency_list):
            nonlocal ops_count
            local_ops = 0
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
                        index.search(q, k)
                        latency_list.append((time.perf_counter() - t0) * 1000)
                        ops_until_next = sample_interval - 1
                    else:
                        index.search(q, k)
                        if ops_until_next > 0:
                            ops_until_next -= 1
                    local_ops += 1
            return local_ops

        def insert_worker(vecs, latency_list):
            nonlocal ops_count
            local_ops = 0
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
                local_ops += 1
            return local_ops

        step_ratio = rw_schedule[tc_idx]
        num_insert_threads = int(num_threads * step_ratio)
        if step_ratio > 0.0 and step_ratio < 1.0:
            if num_insert_threads == 0 and num_threads > 0:
                num_insert_threads = 1
            elif num_insert_threads == num_threads and num_threads > 1:
                num_insert_threads -= 1
        elif step_ratio == 0.0:
            num_insert_threads = 0
        elif step_ratio == 1.0:
            num_insert_threads = num_threads

        num_search_threads = num_threads - num_insert_threads

        search_lat_lists = [[] for _ in range(num_search_threads)]
        insert_lat_lists = [[] for _ in range(num_insert_threads)]

        if num_insert_threads > 0:
            chunk_s = len(remaining_data) // num_insert_threads
            for i in range(num_insert_threads):
                start = i * chunk_s
                end = (
                    (i + 1) * chunk_s
                    if i < num_insert_threads - 1
                    else len(remaining_data)
                )
                t = threading.Thread(
                    target=insert_worker,
                    args=(remaining_data[start:end], insert_lat_lists[i]),
                )
                threads.append(t)
                t.start()

        if num_search_threads > 0:
            for i in range(num_search_threads):
                t = threading.Thread(
                    target=search_worker, args=(queries, search_lat_lists[i])
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
            return None, None, None, None

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

        total_inserts = 0
        if num_insert_threads > 0:
            total_inserts = len(remaining_data)

        total_searches = 0
        if num_search_threads > 0:
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

        if hasattr(index, "close"):
            index.close()

    index_elapsed = time.time() - index_start_time
    print(
        f"  {Fore.WHITE}Elapsed time for {Style.BRIGHT}{index_name}{Style.RESET_ALL}"
        f"{Fore.WHITE}: {_format_elapsed(index_elapsed)}{Style.RESET_ALL}"
    )
    return results, conflict_rates, build_times, latency_data
