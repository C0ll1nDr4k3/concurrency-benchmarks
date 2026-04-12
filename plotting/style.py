import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

DPI = 1200

ICON_MAPPING = {
    "FAISS": ("paper/imgs/meta.png", 0.005),
    "USearch": ("paper/imgs/usearch.png", 0.015),
    "Weaviate": ("paper/imgs/weaviate.png", 0.015),
    "Redis": ("paper/imgs/redis.png", 0.015),
}

COLOR_MAPPING = {
    "FAISS": "#1877F2",  # Facebook Blue
    "USearch": "#192940",  # Dark Blue
    "Weaviate": "#ddd347",  # Yellow
    "Redis": "#d82c20",  # Redis Red
    "HnswLib": "#7b2d8b",  # Purple — recall_vs_qps hnswlib reference
}

STRATEGY_COLOR_MAPPING = {
    "optimistic": "#2ca02c",  # green
    "pessimistic": "#d62728",  # red
    "vanilla": "#7f7f7f",  # neutral gray
}

TYPE_LINESTYLE_MAPPING = {
    "HNSW": "-",
    "IVF": "--",
    "OTHER": "-.",
}

GRANULARITY_MARKER_MAPPING = {
    "coarse": "s",
    "fine": "o",
    "vanilla": "^",
    "other": "d",
}


def _normalize_index_name(name):
    normalized = str(name)
    for suffix in (" (history)",):
        normalized = normalized.replace(suffix, "")
    if " [" in normalized and normalized.endswith("]"):
        normalized = normalized[: normalized.rfind(" [")]
    return normalized.strip()


def get_plot_style(name, external_names=None):
    normalized = _normalize_index_name(name)
    upper = normalized.upper()
    external_names = external_names or []

    for key, color in COLOR_MAPPING.items():
        if key.upper() in upper:
            return {
                "color": color,
                "linestyle": "--",
                "marker": "*",
                "alpha": 0.8,
            }

    if normalized in external_names:
        return {
            "color": "#4c4c4c",
            "linestyle": "--",
            "marker": "*",
            "alpha": 0.8,
        }

    index_type = "HNSW" if "HNSW" in upper else ("IVF" if "IVF" in upper else "OTHER")

    if "OPT" in upper:
        strategy = "optimistic"
    elif "PESS" in upper:
        strategy = "pessimistic"
    elif "VANILLA" in upper:
        strategy = "vanilla"
    else:
        strategy = "vanilla"

    if "COARSE" in upper:
        granularity = "coarse"
    elif "FINE" in upper:
        granularity = "fine"
    elif "VANILLA" in upper:
        granularity = "vanilla"
    else:
        granularity = "other"

    return {
        "color": STRATEGY_COLOR_MAPPING[strategy],
        "linestyle": TYPE_LINESTYLE_MAPPING[index_type],
        "marker": GRANULARITY_MARKER_MAPPING[granularity],
        "alpha": 0.85,
    }


def get_plot_style_token(name, external_names=None):
    style = get_plot_style(name, external_names)
    return f"{style['marker']}{style['linestyle']}"


def add_semantic_style_legend(ax):
    handles = [
        Line2D(
            [0],
            [0],
            color=STRATEGY_COLOR_MAPPING["optimistic"],
            marker="o",
            linestyle="-",
            label="Optimistic",
        ),
        Line2D(
            [0],
            [0],
            color=STRATEGY_COLOR_MAPPING["pessimistic"],
            marker="o",
            linestyle="-",
            label="Pessimistic",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle=TYPE_LINESTYLE_MAPPING["HNSW"],
            label="HNSW",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle=TYPE_LINESTYLE_MAPPING["IVF"],
            label="IVF",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker=GRANULARITY_MARKER_MAPPING["coarse"],
            linestyle="-",
            label="Coarse",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker=GRANULARITY_MARKER_MAPPING["fine"],
            linestyle="-",
            label="Fine",
        ),
    ]
    return ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )


def load_icons():
    """Load icon images from ICON_MAPPING paths. Returns {key: (img_array, zoom)}."""
    loaded = {}
    for key, (path, zoom) in ICON_MAPPING.items():
        if os.path.exists(path):
            try:
                loaded[key] = (mpimg.imread(path), zoom)
            except Exception as e:
                print(f"Could not load icon {path}: {e}")
    return loaded
