"""Centralized index parameter configuration.

Each index needs two kinds of parameters:
  - construction: positional args passed to the constructor (M, ef_construction, etc.)
  - search_sweep: list of param dicts swept during recall-vs-QPS benchmarking

Factory functions (hnsw, ivf) bundle both together into an IndexParams object,
with keyword control over every knob.  This replaces the scattered inline
declarations that previously lived in main.py.

Usage examples:

    # Default HNSW params
    p = hnsw()

    # High-recall FAISS-style configuration
    p = hnsw(M=64, ef_construction=500, ef_search=[200, 400, 600, 800])

    # IVF params (nlist derived from dataset size)
    p = ivf(num_vectors=100_000)

    # Unpack for benchmark functions
    benchmark_recall_vs_qps(cls, name, data, queries, gt, k, p)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt


@dataclass
class IndexParams:
    """Construction args + search sweep for a single index configuration.

    When ``paired_sweep`` is set, each entry is a ``(construction_args, search_dict)``
    pair.  ``benchmark_recall_vs_qps`` builds a fresh index per pair so that
    construction parameters (e.g. M) and search parameters (e.g. ef) can vary
    together along a single recall-vs-QPS curve.  ``construction`` and
    ``search_sweep`` are ignored in this mode.
    """

    construction: list = field(default_factory=list)
    search_sweep: list[dict] = field(default_factory=lambda: [{}])
    paired_sweep: "list[tuple[list, dict]] | None" = None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

# Nine paired (M, ef) values so counts match 1-to-1.
_DEFAULT_HNSW_M_VALUES = [4, 8, 16, 32, 48, 64, 96, 128, 192]
_DEFAULT_HNSW_EF_SEARCH = [10, 20, 40, 80, 120, 200, 400, 600, 800]


def hnsw(
    M: "int | list[int] | None" = None,
    ef_construction: int = 500,
    ef_search: "list[int] | None" = None,
) -> "IndexParams | list[IndexParams]":
    """HNSW-family parameters.

    M can be:
      - None (default): returns a single IndexParams whose ``paired_sweep``
        co-varies M and ef together (one insert + one query per pair).
      - a list of ints: returns one IndexParams per M value (classic sweep)
      - a single int: returns a single IndexParams

    Examples:
        hnsw()           # paired M+ef sweep, single IndexParams
        hnsw(M=16)       # single config
        hnsw(M=[32, 64]) # per-M sweep, list of IndexParams
    """
    if ef_search is None:
        ef_search = list(_DEFAULT_HNSW_EF_SEARCH)

    if M is None:
        m_values = list(_DEFAULT_HNSW_M_VALUES)
        if len(m_values) != len(ef_search):
            raise ValueError(
                f"_DEFAULT_HNSW_M_VALUES ({len(m_values)}) and "
                f"_DEFAULT_HNSW_EF_SEARCH ({len(ef_search)}) must have equal length"
            )
        pairs = [
            ([m, ef_construction], {"ef": ef}) for m, ef in zip(m_values, ef_search)
        ]
        return IndexParams(paired_sweep=pairs)

    sweep = [{"ef": ef} for ef in ef_search]

    if isinstance(M, list):
        return [
            IndexParams(construction=[m, ef_construction], search_sweep=sweep)
            for m in M
        ]

    return IndexParams(construction=[M, ef_construction], search_sweep=sweep)


def ivf(
    num_vectors: int,
    nprobe_values: list[int] | None = None,
) -> IndexParams:
    """IVF-family parameters.  nlist is derived from dataset size."""
    nlist = int(num_vectors**0.5)
    if nprobe_values is None:
        nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128]
    return IndexParams(
        construction=[nlist, int(sqrt(nlist))],
        search_sweep=[{"nprobe": np} for np in nprobe_values if np < nlist],
    )
