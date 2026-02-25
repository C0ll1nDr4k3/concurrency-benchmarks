import argparse
import json
import os

import duckdb
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

try:
    from nilvec.benchmark import COLOR_MAPPING, ICON_MAPPING
except ImportError:
    ICON_MAPPING = {}
    COLOR_MAPPING = {}
    print("Warning: nilvec not found, icons and colors might be missing")


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


def _load_plot_data(db_path, run_id=None):
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
    thread_pos = {int(t): i for i, t in enumerate(thread_counts)}
    for (
        index_name,
        thread_count,
        throughput_val,
        conflict_rate,
        is_external,
    ) in throughput_rows:
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
        "rw_ratio": float(rw_ratio),
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


def plot_results(results_db, output_dir="paper/plots", dpi=1200, run_id=None):
    data = _load_plot_data(results_db, run_id=run_id)

    # Build per-dataset subfolder: {output_dir}/{dataset}_{limit}
    dataset_subdir = f"{data['dataset_name']}_{data['plot_limit']}"
    output_dir = os.path.join(output_dir, dataset_subdir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Plot 1: Recall vs QPS ---
    if data["recall_runs"]:
        print("Plotting Recall vs QPS...")
        plt.figure(figsize=(10, 6))

        for name, recalls, qps, style in data["recall_runs"]:
            plt.plot(recalls, qps, style, label=name)

        plt.xlabel("Recall")
        plt.ylabel("QPS (log scale)")
        plt.yscale("log")
        plt.title(f"Recall vs QPS (K={data['k']}, Dim={data['dim']})")
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(output_dir, "recall_vs_qps.svg")
        plt.savefig(out_path, dpi=dpi)
        print(f"Saved {out_path}")
        plt.close()

    # --- Plot 2: Throughput vs Threads ---
    if data["throughput"]:
        print("Plotting Throughput vs Threads...")
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        thread_counts = data["thread_counts"]
        rw_ratio = data["rw_ratio"]
        external_names = data["external_names"]

        loaded_icons = {}
        for key, (path, zoom) in ICON_MAPPING.items():
            if os.path.exists(path):
                try:
                    loaded_icons[key] = (mpimg.imread(path), zoom)
                except Exception as e:
                    print(f"Could not load icon {path}: {e}")

        for name, res in data["throughput"].items():
            icon_data = None
            for key, (img, zoom) in loaded_icons.items():
                if key in name:
                    icon_data = (img, zoom)
                    break

            color = None
            for key, c in COLOR_MAPPING.items():
                if key in name:
                    color = c
                    break

            if icon_data:
                img, zoom = icon_data
                plt.plot(thread_counts, res, "--", label=name, alpha=0.75, color=color)
                for x, y in zip(thread_counts, res):
                    im = OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
                    ax.add_artist(ab)
            else:
                style = "*--" if name in external_names else "o-"
                plt.plot(thread_counts, res, style, label=name, alpha=0.75, color=color)

        plt.xlabel("Threads")
        plt.ylabel("Ops/sec")
        plt.title(f"Throughput (W:{rw_ratio:.1f}, R:{1.0 - rw_ratio:.1f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "throughput_scaling.svg")
        plt.savefig(out_path, dpi=dpi)
        print(f"Saved {out_path}")
        plt.close()

    # --- Plot 3: Conflict Rates ---
    if data["conflicts"]:
        print("Plotting Conflict Rates...")
        plt.figure(figsize=(8, 6))
        has_conflicts = False
        for name, conflicts in data["conflicts"].items():
            if "Opt" in name:
                plt.plot(data["thread_counts"], conflicts, "x--", label=name)
                has_conflicts = True

        if has_conflicts:
            plt.xlabel("Threads")
            plt.ylabel("Conflict Rate (%)")
            plt.title("Conflict Rate (Optimistic)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(output_dir, "conflict_rate.svg")
            plt.savefig(out_path, dpi=dpi)
            print(f"Saved {out_path}")
        plt.close()

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

    args = parser.parse_args()
    plot_results(args.results_db, args.out, args.dpi, run_id=args.run_id or None)
