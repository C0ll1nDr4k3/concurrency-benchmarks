"""Regression baselines for HNSW disjoint rates.

These tests insert a fixed dataset single-threaded into each HNSW variant
and assert that the resulting per-layer disjoint rates stay under a loose
upper bound. Exact rates are not deterministic across runs (RNG is seeded
from std::random_device), so these tests use generous thresholds intended
to catch regressions (e.g., a bug that drops back-edges or corrupts the
neighbor-list rewrite step) — not to pin down numerical equality.

If one of these fails, the regression is in the index implementation,
not the host environment.
"""

import numpy as np
import pytest

import nilvec

N = 500
DIM = 16
M = 8
EF_CONSTRUCTION = 100

# Loose thresholds calibrated well above observed sequential-insertion rates
# (layer 0: ~1.5%, upper layers: up to ~10% with small N and M). Anything
# materially higher indicates a broken neighbor-list rewrite or lost edges.
PER_LAYER_MAX = 0.25

HNSW_VARIANTS = [
    "HNSWVanilla",
    "HNSWCoarseOptimistic",
    "HNSWCoarsePessimistic",
    "HNSWFineOptimistic",
    "HNSWFinePessimistic",
]


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    return [v.tolist() for v in rng.standard_normal((N, DIM)).astype(np.float32)]


@pytest.mark.parametrize("variant", HNSW_VARIANTS)
def test_disjoint_rate_regression(variant, dataset):
    cls = getattr(nilvec, variant)
    index = cls(DIM, M, EF_CONSTRUCTION, 0.0)
    for v in dataset:
        index.insert(v)
    rates = list(index.disjoint_rates())

    assert rates, (
        f"REGRESSION: {variant} returned empty disjoint_rates after "
        f"inserting {N} vectors. Graph is empty or entry point lost."
    )

    for layer, rate in enumerate(rates):
        assert rate < PER_LAYER_MAX, (
            f"REGRESSION: {variant} layer-{layer} disjoint rate {rate:.4f} "
            f"exceeds threshold {PER_LAYER_MAX}. Sequential insertion should "
            f"preserve the HNSW connectivity invariant up to the pruning "
            f"heuristic's own small residual. Check neighbor-list rewrite / "
            f"back-edge logic. (Run-to-run variation is expected due to "
            f"unseeded RNG; a value this high indicates a real bug.)"
        )
