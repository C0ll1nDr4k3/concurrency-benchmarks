import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from .style import (
    DPI,
    get_plot_style,
    is_external_index,
    load_icons,
)


def plot_recall_vs_qps(
    recall_runs,
    *,
    k,
    dim,
    external_names=None,
    output_path,
    dpi=DPI,
):
    """Plot recall vs QPS.

    recall_runs: list of (name, recalls, qps_values, style_token).
        Names ending with " (history)" are rendered at reduced alpha.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for name, recalls, qps, _style in recall_runs:
        style_cfg = get_plot_style(name, external_names)
        alpha = 0.6 if name.endswith(" (history)") else style_cfg["alpha"]
        plt.plot(
            recalls,
            qps,
            label=name,
            color=style_cfg["color"],
            linestyle=style_cfg["linestyle"],
            marker=style_cfg["marker"],
            alpha=alpha,
        )

    plt.xlabel("Recall")
    plt.ylabel("QPS (log scale)")
    plt.yscale("log")
    plt.title(f"Recall vs QPS (K={k}, Dim={dim})")
    ax.legend(loc="best")
    plt.grid(True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_throughput(
    throughput,
    thread_counts,
    *,
    external_names=None,
    title,
    output_path,
    dpi=DPI,
):
    """Plot throughput vs threads with icon overlays for external indexes.

    throughput: {index_name: [ops/sec per thread count]}
    title: pre-formatted title string.
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    loaded_icons = load_icons()

    for name, res in throughput.items():
        style_cfg = get_plot_style(name, external_names)

        icon_data = None
        if is_external_index(name, external_names):
            for key, (img, zoom) in loaded_icons.items():
                if key in name:
                    icon_data = (img, zoom)
                    break

        plt.plot(
            thread_counts,
            res,
            label=name,
            alpha=style_cfg["alpha"],
            color=style_cfg["color"],
            linestyle=style_cfg["linestyle"],
            marker="",
        )

        if icon_data:
            img, zoom = icon_data
            for x, y in zip(thread_counts, res):
                if np.isnan(y):
                    continue
                im = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
                ax.add_artist(ab)

    plt.xlabel("Threads")
    plt.ylabel("Ops/sec")
    plt.title(title)
    ax.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_conflict_rate(
    conflicts,
    thread_counts,
    *,
    external_names=None,
    output_path,
    dpi=DPI,
):
    """Plot conflict rate for optimistic indexes only.

    Internal indexes use Matplotlib's default color cycle.
    External indexes preserve assigned branding colors.
    Only plots series whose name contains 'Opt'. No-ops if none found.
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    has_conflicts = False

    for name, vals in conflicts.items():
        if "OPT" not in name.upper():
            continue
        style_cfg = get_plot_style(name, external_names)
        plt.plot(
            thread_counts,
            vals,
            label=name,
            color=style_cfg["color"],
            linestyle=style_cfg["linestyle"],
            marker=style_cfg["marker"],
            alpha=style_cfg["alpha"],
        )
        has_conflicts = True

    if not has_conflicts:
        plt.close()
        return

    plt.xlabel("Threads")
    plt.ylabel("Conflict Rate (%)")
    plt.title("Conflict Rate (Optimistic)")
    ax.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_disjoint_rate(
    disjoint,
    thread_counts,
    *,
    external_names=None,
    output_path,
    dpi=DPI,
    layer=0,
):
    """Plot layer-0 (by default) disjoint rate vs thread count for HNSW indexes.

    disjoint: {index_name: [(N, [rate_layer0, rate_layer1, ...]), ...]} keyed
    by thread-count iteration (parallel to thread_counts).

    The model predicts δ ≈ ρ · T² / N for optimistic strategies. To make the
    T² dependence visible across runs with growing N, we plot δ · N on the
    y-axis: optimistic curves should trace a quadratic in T, while vanilla
    and pessimistic variants stay flat near zero.
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plotted = False

    for name, series in disjoint.items():
        if not series:
            continue
        xs = []
        ys = []
        for t, (n, rates) in zip(thread_counts, series):
            if not rates or layer >= len(rates) or n == 0:
                continue
            xs.append(t)
            ys.append(rates[layer] * n)
        if not xs:
            continue
        style_cfg = get_plot_style(name, external_names)
        plt.plot(
            xs,
            ys,
            label=name,
            color=style_cfg["color"],
            linestyle=style_cfg["linestyle"],
            marker=style_cfg["marker"],
            alpha=style_cfg["alpha"],
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Threads")
    plt.ylabel(f"Disjoint rate × N (layer {layer})")
    plt.title(f"Layer-{layer} Disjoint Rate (scaled by N)")
    ax.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
