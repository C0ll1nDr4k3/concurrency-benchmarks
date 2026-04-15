#!/usr/bin/env python3
import os
import pickle
import sys
import time

import colorama
from colorama import Fore, Style

colorama.init(autoreset=True, strip=False)

print(
    f"{Style.BRIGHT}{Fore.GREEN}NilVec Benchmark Suite Initializing...{Style.RESET_ALL}"
)

try:
    import nilvec
except ImportError:
    print("Error: Could not import nilvec. Run `uv pip install -e .` first.")
    sys.exit(1)

from nilvec import config as cfg
from nilvec.benchmarks import benchmark_recall_vs_qps, benchmark_throughput_vs_threads
from nilvec.cli import build_parser
from nilvec.config import make_quantized_cls
from nilvec.datasets import DATASETS, generate_data, load_dataset
from nilvec.external_indexes import (
    FaissHNSW,
    FaissIVF,
    HnswLibIndex,
    MilvusIndex,
    RedisIndex,
    USearchIndex,
    WeaviateIndex,
    faiss,
    hnswlib,
    redis,
    usearch,
    weaviate,
)
from nilvec.formatting import _format_elapsed
from nilvec.metrics import get_ground_truth, redis_benchmark_ready
from nilvec.params import hnsw, ivf
from nilvec.store import BenchmarkResultsStore
from plotting.benchmark_plots import (
    plot_conflict_rate,
    plot_recall_vs_qps,
    plot_throughput,
)
from plotting.style import DPI


def _run_single_dataset(args, dataset_path):
    """Run benchmark for a single dataset. Returns elapsed time in seconds."""
    dataset_start_time = time.time()

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    if dataset_path and (os.path.exists(dataset_path) or dataset_name in DATASETS):
        data, queries, gt = load_dataset(dataset_path, args.limit, k=cfg.K)

        if args.limit > 0:
            print(f"Limiting dataset to first {args.limit} vectors...")
            gt = None

        cfg.DIM = len(data[0])
        cfg.NUM_VECTORS = len(data)
        cfg.NUM_QUERIES = len(queries)
        if gt is None:
            gt = get_ground_truth(data, queries, cfg.K)
    else:
        print(
            f"{Fore.CYAN}Generating {cfg.NUM_VECTORS} vectors (dim={cfg.DIM})...{Style.RESET_ALL}"
        )
        data = generate_data(cfg.NUM_VECTORS, cfg.DIM)
        queries = generate_data(cfg.NUM_QUERIES, cfg.DIM)
        gt = get_ground_truth(data, queries, cfg.K)

    print(
        f"{Fore.WHITE}Dataset: N={cfg.NUM_VECTORS}, Q={cfg.NUM_QUERIES}, Dim={cfg.DIM}{Style.RESET_ALL}"
    )

    _plot_limit = args.limit if args.limit > 0 else "full"
    plot_dir = os.path.join("paper", "plots", f"{dataset_name}_{_plot_limit}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    print(f"{Fore.WHITE}Plot output directory: {plot_dir}{Style.RESET_ALL}")

    run_meta = {
        "run_tag": args.run_tag or None,
        "dataset_name": dataset_name,
        "dataset_path": os.path.abspath(dataset_path),
        "dim": cfg.DIM,
        "num_vectors": cfg.NUM_VECTORS,
        "num_queries": cfg.NUM_QUERIES,
        "k": cfg.K,
        "workload_profile": "thread_split_quarter_writers",
        "thread_counts": cfg.THREAD_COUNTS,
        "only_external": args.external_only,
        "internal_only": args.internal_only,
        "skip_recall": args.skip_recall,
        "skip_throughput": args.skip_throughput,
        "limit_rows": args.limit,
    }

    results_store = None
    run_id = None
    if args.results_db:
        try:
            results_store = BenchmarkResultsStore(args.results_db)
            run_id = results_store.start_run(run_meta)
            print(f"Results DB: {args.results_db} (run_id={run_id[:8]})")
        except Exception as e:
            print(f"{Fore.YELLOW}Results DB disabled: {e}{Style.RESET_ALL}")

    results_cache = {
        "recall_vs_qps": {"runs": [], "latencies": {}, "K": cfg.K, "DIM": cfg.DIM},
        "throughput": {},
        "conflicts": {},
        "thread_counts": cfg.THREAD_COUNTS,
        "external_names": [],
    }

    ip = ivf(cfg.NUM_VECTORS)
    # Single fixed-M config for throughput benchmark (explicitly selecting the
    # one returned list element).
    hp = hnsw(M=16)[0]

    # --- recall_vs_qps ---
    if not args.skip_recall:
        print("\n=== recall_vs_qps ===")

        sq = nilvec.ScalarQuantizer(cfg.DIM)
        sq.train(data)

        # Recall is single-threaded, so locking strategy doesn't affect
        # results.  Only benchmark one representative per structural family
        # (Vanilla, Hybrid, SQ8) plus external baselines (FAISS, hnswlib).
        hnsw_classes = [
            (nilvec.HNSWVanilla, "HNSW Vanilla"),
            (nilvec.HybridOptimistic, "Hybrid Opt"),
            (nilvec.HybridPessimistic, "Hybrid Pess"),
            (make_quantized_cls(nilvec.HNSWVanillaSQ8, sq), "HNSW Vanilla SQ8"),
        ]
        # hnsw() returns one IndexParams per M value; expand into separate
        # benchmark entries so each M is built once and ef_search is swept.
        hnsw_param_list = hnsw()
        ann_indexes = [
            (cls, f"{name} M={p.construction[0]}", p)
            for cls, name in hnsw_classes
            for p in hnsw_param_list
        ]
        ann_indexes += [
            (nilvec.IVFFlatVanilla, "IVFFlat Vanilla", ip),
            (
                make_quantized_cls(nilvec.IVFFlatVanillaSQ8, sq),
                "IVFFlat Vanilla SQ8",
                ip,
            ),
        ]
        if faiss is not None:
            ann_indexes += [
                (FaissHNSW, f"FAISS HNSW M={p.construction[0]}", p)
                for p in hnsw_param_list
            ]
            ann_indexes.append((FaissIVF, "FAISS IVF", ip))
        if hnswlib is not None:
            ann_indexes += [
                (HnswLibIndex, f"HnswLib M={p.construction[0]}", p)
                for p in hnsw_param_list
            ]

        for index_cls, index_name, idx_params in ann_indexes:
            res = benchmark_recall_vs_qps(
                index_cls,
                index_name,
                data,
                queries,
                gt,
                cfg.K,
                latency_sample_rate=args.latency_sample_rate,
                params=idx_params,
            )
            if res is None:
                continue
            recalls, qps_vals, p50s, p95s, p99s = zip(*res)
            results_cache["recall_vs_qps"]["runs"].append(
                (index_name, recalls, qps_vals, None)
            )
            results_cache["recall_vs_qps"]["latencies"][index_name] = list(
                zip(p50s, p95s, p99s)
            )

        current_names = {n for n, _, _, _ in results_cache["recall_vs_qps"]["runs"]}
        if args.cross_pollinate and results_store and run_id:
            merged_runs, injected = results_store.cross_pollinate_recall(
                run_meta, run_id, results_cache["recall_vs_qps"]["runs"]
            )
            results_cache["recall_vs_qps"]["runs"] = merged_runs
            if injected:
                print(
                    f"Cross-pollinated recall from history: {', '.join(sorted(injected))}"
                )

        plot_runs = []
        for name, recalls, qps, style in results_cache["recall_vs_qps"]["runs"]:
            if name not in current_names:
                plot_runs.append((f"{name} (history)", recalls, qps, style))
            else:
                plot_runs.append((name, recalls, qps, style))

        recall_path = os.path.join(plot_dir, "recall_vs_qps.svg")
        plot_recall_vs_qps(
            plot_runs,
            k=cfg.K,
            dim=cfg.DIM,
            output_path=recall_path,
        )
        print(f"Saved {recall_path}")

    # --- Throughput vs Threads ---
    if not args.skip_throughput:
        print("\n=== Throughput vs Threads Benchmark ===")

        indexes = [
            (nilvec.HybridPessimistic, "Hybrid Pess", hp),
            (nilvec.HybridOptimistic, "Hybrid Opt", hp),
            (nilvec.HNSWVanilla, "HNSW Vanilla", hp),
            (nilvec.HNSWCoarseOptimistic, "HNSW Coarse Opt", hp),
            (nilvec.HNSWCoarsePessimistic, "HNSW Coarse Pess", hp),
            (nilvec.HNSWFineOptimistic, "HNSW Fine Opt", hp),
            (nilvec.HNSWFinePessimistic, "HNSW Fine Pess", hp),
            (nilvec.IVFFlatCoarseOptimistic, "IVF Coarse Opt", ip),
            (nilvec.IVFFlatFineOptimistic, "IVF Fine Opt", ip),
        ]

        throughput_results = {}
        conflict_results = {}
        build_time_results = {}
        latency_results = {}

        externals = []
        if faiss:
            externals.append((FaissHNSW, "FAISS HNSW", hp))
            externals.append((FaissIVF, "FAISS IVF", ip))
        if usearch:
            externals.append((USearchIndex, "USearch", hp))
        if hnswlib:
            externals.append((HnswLibIndex, "HnswLib", hp))
        if weaviate:
            externals.append((WeaviateIndex, "Weaviate", hp))
        redis_ready, redis_info = redis_benchmark_ready(
            auto_start=args.auto_start_redis
        )
        if redis_ready:
            externals.append((RedisIndex, "Redis", hp))
        elif redis is not None:
            print(
                f"{Fore.YELLOW}Skipping Redis benchmark preflight: cannot connect to {redis_info}.{Style.RESET_ALL}\n"
                f"{Fore.YELLOW}Hint:{Style.RESET_ALL} set REDIS_URL, ensure Docker is running for auto-start, or start Redis Stack manually:\n"
                "  docker run -p 6379:6379 -d redis/redis-stack:latest\n"
                "  (pass --auto-start-redis=false to disable Docker auto-start)"
            )

        for cls, name, idx_p in externals:
            if args.internal_only:
                continue
            try:
                res, conflicts, build, lats = benchmark_throughput_vs_threads(
                    cls,
                    name,
                    data,
                    queries,
                    cfg.K,
                    latency_sample_rate=args.latency_sample_rate,
                    params=idx_p,
                    preload_ratio=args.preload_ratio,
                )
                if res is None:
                    continue
                throughput_results[name] = res
                conflict_results[name] = conflicts
                build_time_results[name] = build
                latency_results[name] = lats
            except Exception as e:
                print(f"Skipping {name}: {e}")

        for cls, name, idx_p in indexes:
            if args.external_only:
                continue
            try:
                res, conflicts, build, lats = benchmark_throughput_vs_threads(
                    cls,
                    name,
                    data,
                    queries,
                    cfg.K,
                    latency_sample_rate=args.latency_sample_rate,
                    params=idx_p,
                    preload_ratio=args.preload_ratio,
                )
                if res is None:
                    continue
                throughput_results[name] = res
                conflict_results[name] = conflicts
                build_time_results[name] = build
                latency_results[name] = lats
            except Exception as e:
                print(f"Skipping {name}: {e}")

        results_cache["throughput"] = throughput_results
        results_cache["conflicts"] = conflict_results
        results_cache["build_times"] = build_time_results
        results_cache["latency_results"] = latency_results
        results_cache["external_names"] = [e[1] for e in externals]

        if args.cross_pollinate and results_store and run_id:
            merged_throughput, merged_conflicts, merged_external, injected = (
                results_store.cross_pollinate_throughput(
                    run_meta,
                    run_id,
                    results_cache["throughput"],
                    results_cache["conflicts"],
                    results_cache["external_names"],
                )
            )
            results_cache["throughput"] = merged_throughput
            results_cache["conflicts"] = merged_conflicts
            results_cache["external_names"] = merged_external
            if injected:
                print(
                    f"Cross-pollinated throughput from history: {', '.join(injected)}"
                )

        title = "Throughput"

        throughput_path = os.path.join(plot_dir, "throughput_scaling.svg")
        plot_throughput(
            results_cache["throughput"],
            cfg.THREAD_COUNTS,
            external_names=results_cache["external_names"],
            title=title,
            output_path=throughput_path,
        )
        print(f"Saved {throughput_path}")

        conflict_path = os.path.join(plot_dir, "conflict_rate.svg")
        plot_conflict_rate(
            results_cache["conflicts"],
            cfg.THREAD_COUNTS,
            output_path=conflict_path,
        )

    if results_store and run_id:
        if results_cache["throughput"]:
            results_store.save_throughput(
                run_id,
                results_cache["throughput"],
                results_cache["conflicts"],
                results_cache["external_names"],
                cfg.THREAD_COUNTS,
                latency_results=results_cache.get("latency_results"),
            )
        if results_cache["recall_vs_qps"]["runs"]:
            results_store.save_recall_runs(
                run_id,
                results_cache["recall_vs_qps"]["runs"],
                results_cache["recall_vs_qps"]["latencies"],
            )
        results_store.close()

    if args.save_results:
        print(f"Saving results to {args.save_results}...")
        with open(args.save_results, "wb") as f:
            pickle.dump(results_cache, f)

    dataset_elapsed = time.time() - dataset_start_time
    print(
        f"\n{Style.BRIGHT}{Fore.GREEN}Dataset '{dataset_name}' completed in "
        f"{_format_elapsed(dataset_elapsed)}{Style.RESET_ALL}"
    )
    return dataset_elapsed


def run_benchmark(args=None):
    if args is None:
        parser = build_parser()
        args = parser.parse_args()

    suite_start_time = time.time()

    if getattr(args, "all", False):
        dataset_names = list(DATASETS.keys())
        print(
            f"\n{Style.BRIGHT}{Fore.CYAN}=== Full Benchmark Suite ==={Style.RESET_ALL}"
        )
        print(f"Running {len(dataset_names)} datasets: {', '.join(dataset_names)}\n")

        dataset_timings = {}
        for i, ds_name in enumerate(dataset_names, 1):
            ds_path = f"{ds_name}.hdf5"
            print(
                f"\n{Style.BRIGHT}{Fore.CYAN}"
                f"--- [{i}/{len(dataset_names)}] Dataset: {ds_name} ---"
                f"{Style.RESET_ALL}"
            )
            elapsed = _run_single_dataset(args, ds_path)
            dataset_timings[ds_name] = elapsed

        suite_elapsed = time.time() - suite_start_time
        print(f"\n{Style.BRIGHT}{Fore.CYAN}=== Full Suite Summary ==={Style.RESET_ALL}")
        for ds_name, elapsed in dataset_timings.items():
            print(f"  {ds_name}: {_format_elapsed(elapsed)}")
        print(
            f"\n{Style.BRIGHT}{Fore.GREEN}Total suite time: "
            f"{_format_elapsed(suite_elapsed)}{Style.RESET_ALL}"
        )
    else:
        _run_single_dataset(args, args.dataset)
        suite_elapsed = time.time() - suite_start_time
        print(
            f"\n{Style.BRIGHT}{Fore.GREEN}Total elapsed time: "
            f"{_format_elapsed(suite_elapsed)}{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    run_benchmark()
