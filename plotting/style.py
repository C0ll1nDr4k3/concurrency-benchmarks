import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg

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
    "HnswLib": "#7b2d8b",  # Purple
}


def _normalize_index_name(name):
    normalized = str(name)
    for suffix in (" (history)",):
        normalized = normalized.replace(suffix, "")
    if " [" in normalized and normalized.endswith("]"):
        normalized = normalized[: normalized.rfind(" [")]
    return normalized.strip()


def is_external_index(name, external_names=None):
    normalized = _normalize_index_name(name)
    external_names = set(external_names or [])
    if normalized in external_names:
        return True
    upper = normalized.upper()
    return any(key.upper() in upper for key in COLOR_MAPPING)


def get_plot_style(name, external_names=None):
    normalized = _normalize_index_name(name)
    upper = normalized.upper()
    external_names = external_names or []

    if is_external_index(name, external_names):
        for key, color in COLOR_MAPPING.items():
            if key.upper() in upper:
                return {"color": color, "linestyle": "-", "marker": "", "alpha": 0.8}

    if normalized in external_names:
        return {"color": "#4c4c4c", "linestyle": "-", "marker": "", "alpha": 0.8}

    return {"color": None, "linestyle": "-", "marker": "", "alpha": 0.85}


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
