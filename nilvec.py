from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_bindings() -> bool:
    try:
        mod = importlib.import_module("_nilvec")
    except Exception:
        return False

    public = [name for name in dir(mod) if not name.startswith("_")]
    for name in public:
        globals()[name] = getattr(mod, name)
    globals()["__all__"] = public
    return True


def _try_load_from_build() -> bool:
    repo_root = Path(__file__).resolve().parent
    # Meson build directories can be named 'build' or 'builddir'
    for build_name in ["builddir", "build"]:
        build_root = repo_root / build_name
        if not build_root.exists():
            continue

        # Look for _nilvec extension module
        candidates = list(build_root.glob("**/_nilvec.*"))
        if not candidates:
            candidates = list(build_root.glob("_nilvec.*"))

        if candidates:
            # Add the directory containing _nilvec extension to sys.path
            build_ext_dir = candidates[0].parent
            sys.path.insert(0, str(build_ext_dir))
            return _load_bindings()
    return False


if not _load_bindings():
    _try_load_from_build()
