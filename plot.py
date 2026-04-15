import argparse
import json
import os

import duckdb

from plotting.benchmark_plots import (
    plot_conflict_rate,
    plot_recall_vs_qps,
    plot_throughput,
)
from plotting.style import is_external_index


def _resolve_run_id(conn, run_id):
    if run_id:
        row = conn.execute(
            "SELECT run_id FROM benchmark_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"run_id not found: {run_id}")
        return row[0]

    row = conn.execute(
        "SELECT run_id FROM benchmark_runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("No runs found in benchmark database")
    return row[0]


def _load_plot_data(db_path, run_id=None, external_only=False, internal_only=False):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Results DB {db_path} not found")

    conn = duckdb.connect(db_path, read_only=True)
    selected_run_id = _resolve_run_id(conn, run_id)

    run_meta = conn.execute(
        """
        SELECT rw_ratio, thread_counts_json, k, dim, dataset_name, num_vectors, limit_rows
        FROM benchmark_runs
        WHERE run_id = ?
        """,
        [selected_run_id],
    ).fetchone()
    if run_meta is None:
        raise ValueError(f"No benchmark_runs entry for run_id={selected_run_id}")

    rw_ratio, thread_counts_json, k, dim, dataset_name, num_vectors, limit_rows = (
        run_meta
    )
    thread_counts = json.loads(thread_counts_json)

    throughput_rows = conn.execute(
        """
        SELECT index_name, thread_count, throughput, conflict_rate, is_external
        FROM throughput_points
        WHERE run_id = ?
        ORDER BY index_name, thread_count
        """,
        [selected_run_id],
    ).fetchall()

    throughput = {}
    conflicts = {}
    external_names = set()
    external_by_name = {}
    thread_pos = {int(t): i for i, t in enumerate(thread_counts)}
    for (
        index_name,
        thread_count,
        throughput_val,
        conflict_rate,
        is_external,
    ) in throughput_rows:
        is_external = bool(is_external)
        external_by_name[index_name] = is_external
        if external_only and not is_external:
            continue
        if internal_only and is_external:
            continue
        if index_name not in throughput:
            throughput[index_name] = [float("nan")] * len(thread_counts)
            conflicts[index_name] = [float("nan")] * len(thread_counts)
        i = thread_pos.get(int(thread_count))
        if i is None:
            continue
        throughput[index_name][i] = float(throughput_val)
        conflicts[index_name][i] = float(conflict_rate)
        if is_external:
            external_names.add(index_name)

    recall_rows = conn.execute(
        """
        SELECT index_name, recall, qps, line_style
        FROM recall_points
        WHERE run_id = ?
        ORDER BY index_name, point_order
        """,
        [selected_run_id],
    ).fetchall()
    recall_grouped = {}
    for index_name, recall, qps, line_style in recall_rows:
        is_external = external_by_name.get(index_name)
        if is_external is None:
            is_external = is_external_index(index_name, external_names)
        if external_only and not is_external:
            continue
        if internal_only and is_external:
            continue
        recall_grouped.setdefault(
            index_name, {"recalls": [], "qps": [], "style": line_style}
        )
        recall_grouped[index_name]["recalls"].append(float(recall))
        recall_grouped[index_name]["qps"].append(float(qps))

    recall_runs = [
        (name, vals["recalls"], vals["qps"], vals["style"])
        for name, vals in recall_grouped.items()
    ]

    conn.close()
    _plot_limit = int(limit_rows) if limit_rows and int(limit_rows) > 0 else "full"
    return {
        "run_id": selected_run_id,
        "rw_ratio": float(rw_ratio) if rw_ratio is not None else None,
        "thread_counts": thread_counts,
        "k": int(k),
        "dim": int(dim),
        "throughput": throughput,
        "conflicts": conflicts,
        "external_names": sorted(external_names),
        "recall_runs": recall_runs,
        "dataset_name": dataset_name or "unknown",
        "plot_limit": _plot_limit,
    }


def plot_results(
    results_db,
    output_dir="paper/plots",
    dpi=1200,
    run_id=None,
    external_only=False,
    internal_only=False,
):
    data = _load_plot_data(
        results_db,
        run_id=run_id,
        external_only=external_only,
        internal_only=internal_only,
    )

    dataset_subdir = f"{data['dataset_name']}_{data['plot_limit']}"
    output_dir = os.path.join(output_dir, dataset_subdir)
    os.makedirs(output_dir, exist_ok=True)

    if data["recall_runs"]:
        print("Plotting Recall vs QPS...")
        plot_recall_vs_qps(
            data["recall_runs"],
            k=data["k"],
            dim=data["dim"],
            external_names=data["external_names"],
            output_path=os.path.join(output_dir, "recall_vs_qps.svg"),
            dpi=dpi,
        )

    if data["throughput"]:
        rw_ratio = data["rw_ratio"]
        title = (
            f"Throughput (W:{rw_ratio:.1f}, R:{1.0 - rw_ratio:.1f})"
            if rw_ratio is not None
            else "Throughput"
        )
        print("Plotting Throughput vs Threads...")
        plot_throughput(
            data["throughput"],
            data["thread_counts"],
            external_names=data["external_names"],
            title=title,
            output_path=os.path.join(output_dir, "throughput_scaling.svg"),
            dpi=dpi,
        )

    if data["conflicts"]:
        print("Plotting Conflict Rates...")
        plot_conflict_rate(
            data["conflicts"],
            data["thread_counts"],
            external_names=data["external_names"],
            output_path=os.path.join(output_dir, "conflict_rate.svg"),
            dpi=dpi,
        )

    print(f"Plotted run_id={data['run_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from DuckDB")
    parser.add_argument(
        "--results-db",
        type=str,
        default="benchmark_results.duckdb",
        help="Path to benchmark DuckDB file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run_id to plot (default: latest run)",
    )
    parser.add_argument(
        "--out", type=str, default="paper/plots", help="Output directory"
    )
    parser.add_argument("--dpi", type=int, default=1200, help="DPI for plots")
    parser.add_argument(
        "--external-only", action="store_true", help="Plot only external indexes"
    )
    parser.add_argument(
        "--internal-only", action="store_true", help="Plot only internal indexes"
    )

    args = parser.parse_args()
    plot_results(
        args.results_db,
        args.out,
        args.dpi,
        run_id=args.run_id or None,
        external_only=args.external_only,
        internal_only=args.internal_only,
    )
