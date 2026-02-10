from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_bindings() -> bool:
    try:
        mod = importlib.import_module("nilvec._nilvec")
    except Exception:
        return False

    public = [name for name in dir(mod) if not name.startswith("_")]
    for name in public:
        globals()[name] = getattr(mod, name)
    globals()["__all__"] = public
    return True


def _try_load_from_build() -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    build_root = repo_root / "build"
    if not build_root.exists():
        return False

    candidates = list(build_root.glob("**/nilvec/_nilvec.*"))
    if not candidates:
        return False

    # Put the build dir containing the `nilvec/` package on sys.path.
    build_pkg_root = candidates[0].parent.parent
    sys.path.insert(0, str(build_pkg_root))
    return _load_bindings()


if not _load_bindings():
    _try_load_from_build()

try:
    from . import benchmark
except ImportError:
    pass
