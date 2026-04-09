import argparse

from nilvec.config import OP_MIX_RATIO


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
        "--op-mix-ratio",
        dest="op_mix_ratio",
        type=float,
        default=OP_MIX_RATIO,
        help="Fixed op-mix write ratio (0.0=Read, 1.0=Write). Overridden by --op-mix-bands.",
    )
    parser.add_argument(
        "--op-mix-bands",
        dest="op_mix_bands",
        nargs="+",
        type=str,
        default=["0.01-0.05", "0.20-0.50"],
        help="Op-mix write-ratio bands (default: '0.01-0.05 0.20-0.50'). "
        "Ratio ramps linearly from low to high across thread counts. "
        "Multiple bands run separate sweeps, e.g. '0.01-0.05 0.20-0.50'.",
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
    return parser
