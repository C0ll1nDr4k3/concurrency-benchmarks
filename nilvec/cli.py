import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ann", "--skip-recall", dest="skip_recall", action="store_true"
    )
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sift-128-euclidean.hdf5",
        help="Path to HDF5 dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run benchmark on all available datasets (overrides --dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of vectors (0 = no limit)",
    )
    parser.add_argument(
        "--external-only", action="store_true", help="Run only external benchmarks"
    )
    parser.add_argument(
        "--internal-only", action="store_true", help="Run only internal benchmarks"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="",
        help="Legacy pickle export path (optional)",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default="benchmark_results.duckdb",
        help="DuckDB file for benchmark history",
    )
    parser.add_argument(
        "--cross-pollinate",
        action="store_true",
        default=True,
        help="Merge compatible historical results into current plots (default: enabled)",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional label to tag this benchmark run in results DB",
    )
    parser.add_argument(
        "--auto-start-redis",
        action="store_true",
        default=True,
        help="Attempt to auto-start redis/redis-stack via Docker if REDIS_URL is unreachable (default: enabled)",
    )
    parser.add_argument(
        "--latency-sample-rate",
        type=float,
        default=0.01,
        help="Fraction of queries to time for latency percentiles (0.0-1.0, default: 0.01)",
    )
    parser.add_argument(
        "--preload-ratio",
        type=float,
        default=0.5,
        help="Fraction of dataset to pre-insert single-threaded before the "
        "concurrent phase of the throughput benchmark (0.0-1.0, default: 0.5)",
    )
    return parser
