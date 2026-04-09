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
    op_mix_ratio=0.5,
    latency_sample_rate=0.0,
):
    """
    Run Throughput vs Threads benchmark with configurable R/W split.
    op_mix_ratio: float for a fixed ratio, or list of floats (one per THREAD_COUNTS entry)
              for a ramping schedule across thread counts.
    """
    if index_args is None:
        index_args = []

    # Normalize op_mix_ratio into a per-thread-count schedule
    if isinstance(op_mix_ratio, (list, tuple)) and not isinstance(
        op_mix_ratio[0], (int, float)
    ):
        op_mix_schedule = op_mix_ratio
    elif isinstance(op_mix_ratio, list):
        op_mix_schedule = op_mix_ratio
    else:
        op_mix_schedule = [op_mix_ratio] * len(config.THREAD_COUNTS)

    # Display header with band info if schedule varies
    if op_mix_schedule[0] != op_mix_schedule[-1]:
        header_ratio = (op_mix_schedule[0], op_mix_schedule[-1])
    else:
        header_ratio = op_mix_schedule[0]
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

        build_time = time.time() - build_start
        build_times.append(build_time)
        start_time = time.time()

        threads = []
        op_counts = [(0, 0) for _ in range(num_threads)]  # (reads, writes)
        search_lat_lists = [[] for _ in range(num_threads)]
        insert_lat_lists = [[] for _ in range(num_threads)]
        insert_pos = [0]
        insert_lock = threading.Lock()
        ops_per_thread = max(1, len(queries) * 5)

        def mixed_worker(worker_id):
            local_reads = 0
            local_writes = 0
            read_lats = search_lat_lists[worker_id]
            write_lats = insert_lat_lists[worker_id]
            write_ratio = max(0.0, min(1.0, float(op_mix_schedule[tc_idx])))
            seed = ((tc_idx + 1) * 1_000_003) ^ ((worker_id + 1) * 65_537)
            rng = random.Random(seed)
            ops_until_next = (
                rng.randint(0, sample_interval - 1) if sample_interval > 0 else -1
            )
            query_pos = rng.randrange(len(queries)) if queries else 0

            for _ in range(ops_per_thread):
                if stop_event.is_set():
                    break
                do_write = rng.random() < write_ratio

                if do_write and data:
                    with insert_lock:
                        if insert_pos[0] < len(data):
                            vec = data[insert_pos[0]]
                            insert_pos[0] += 1
                        else:
                            vec = None
                    if vec is not None:
                        if ops_until_next == 0:
                            t0 = time.perf_counter()
                            index.insert(vec)
                            write_lats.append((time.perf_counter() - t0) * 1000)
                            ops_until_next = sample_interval - 1
                        else:
                            index.insert(vec)
                            if ops_until_next > 0:
                                ops_until_next -= 1
                        local_writes += 1
                        continue

                if queries:
                    q = queries[query_pos]
                    query_pos = (query_pos + 1) % len(queries)
                    if ops_until_next == 0:
                        t0 = time.perf_counter()
                        index.search(q, k)
                        read_lats.append((time.perf_counter() - t0) * 1000)
                        ops_until_next = sample_interval - 1
                    else:
                        index.search(q, k)
                        if ops_until_next > 0:
                            ops_until_next -= 1
                    local_reads += 1

            op_counts[worker_id] = (local_reads, local_writes)

        step_ratio = op_mix_schedule[tc_idx]
        for i in range(num_threads):
            t = threading.Thread(target=mixed_worker, args=(i,))
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

        total_searches = sum(r for r, _ in op_counts)
        total_inserts = sum(w for _, w in op_counts)

        total_ops = total_inserts + total_searches
        throughput = total_ops / duration
        achieved_write_ratio = (total_inserts / total_ops) if total_ops > 0 else 0.0
        num_insert_threads = int(round(num_threads * achieved_write_ratio))
        num_insert_threads = max(0, min(num_threads, num_insert_threads))
        num_search_threads = num_threads - num_insert_threads
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
                target_write_ratio=step_ratio,
                achieved_write_ratio=achieved_write_ratio,
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
