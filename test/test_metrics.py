"""Tests for nilvec/metrics.py — compute_recall, URL helpers, get_ground_truth."""

import pytest

from nilvec.metrics import _redis_url_is_local, compute_recall

# ---------------------------------------------------------------------------
# compute_recall
# ---------------------------------------------------------------------------


class TestComputeRecall:
    def test_perfect_recall(self):
        ground_truth = [[0, 1, 2, 3, 4]]
        results = [[0, 1, 2, 3, 4]]
        assert compute_recall(results, ground_truth, k=5) == pytest.approx(1.0)

    def test_zero_recall(self):
        ground_truth = [[0, 1, 2, 3, 4]]
        results = [[5, 6, 7, 8, 9]]
        assert compute_recall(results, ground_truth, k=5) == pytest.approx(0.0)

    def test_partial_recall(self):
        ground_truth = [[0, 1, 2, 3, 4]]
        results = [[0, 1, 5, 6, 7]]  # 2 out of 5 correct
        assert compute_recall(results, ground_truth, k=5) == pytest.approx(0.4)

    def test_multiple_queries_averaged(self):
        ground_truth = [[0, 1, 2], [10, 11, 12]]
        # First query: perfect (3/3), second query: zero (0/3)
        results = [[0, 1, 2], [20, 21, 22]]
        assert compute_recall(results, ground_truth, k=3) == pytest.approx(0.5)

    def test_k_smaller_than_ground_truth(self):
        # Ground truth has 5 neighbours, but we only check k=3
        ground_truth = [[0, 1, 2, 3, 4]]
        results = [[0, 1, 2]]  # 3 correct out of top-3
        assert compute_recall(results, ground_truth, k=3) == pytest.approx(1.0)

    def test_k_larger_than_ground_truth(self):
        # Ground truth only has 2 neighbours, k=5 — denominator clamped to len(true_ids)=2
        ground_truth = [[0, 1]]
        results = [[0, 1, 2, 3, 4]]
        assert compute_recall(results, ground_truth, k=5) == pytest.approx(1.0)

    def test_single_query_single_result(self):
        ground_truth = [[42]]
        results = [[42]]
        assert compute_recall(results, ground_truth, k=1) == pytest.approx(1.0)

    def test_order_independent(self):
        """Recall should not depend on the order of results."""
        ground_truth = [[0, 1, 2, 3, 4]]
        results_ordered = [[0, 1, 2, 3, 4]]
        results_shuffled = [[4, 2, 0, 3, 1]]
        r1 = compute_recall(results_ordered, ground_truth, k=5)
        r2 = compute_recall(results_shuffled, ground_truth, k=5)
        assert r1 == pytest.approx(r2)

    def test_duplicate_results_not_deduplicated(self):
        """The implementation counts each hit individually, including duplicates."""
        ground_truth = [[0, 1, 2, 3, 4]]
        # All 5 results are the same correct neighbour — each hits the truth_set
        results = [[0, 0, 0, 0, 0]]
        recall = compute_recall(results, ground_truth, k=5)
        # 5 hits / min(5, 5) = 1.0 (implementation does not deduplicate)
        assert recall == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _redis_url_is_local
# ---------------------------------------------------------------------------


class TestRedisUrlIsLocal:
    def test_localhost(self):
        assert _redis_url_is_local("redis://localhost:6379/0") is True

    def test_127_0_0_1(self):
        assert _redis_url_is_local("redis://127.0.0.1:6379/0") is True

    def test_ipv6_loopback(self):
        # Python's urlparse represents IPv6 addresses with brackets: [::1]
        assert _redis_url_is_local("redis://[::1]:6379/0") is True

    def test_remote_host(self):
        assert _redis_url_is_local("redis://my-redis-server.example.com:6379") is False

    def test_rediss_scheme_localhost(self):
        assert _redis_url_is_local("rediss://localhost:6380/0") is True

    def test_http_scheme_rejected(self):
        assert _redis_url_is_local("http://localhost:6379") is False

    def test_empty_string(self):
        assert _redis_url_is_local("") is False


# ---------------------------------------------------------------------------
# get_ground_truth (requires compiled C++ extension)
# ---------------------------------------------------------------------------


nilvec = pytest.importorskip(
    "nilvec",
    reason="nilvec not importable - run `meson compile -C builddir` first",
)


class TestGetGroundTruth:
    def test_returns_correct_number_of_results(self):
        from nilvec.metrics import get_ground_truth

        data = [[float(i)] * 4 for i in range(50)]
        queries = [[float(i)] * 4 for i in range(5)]
        gt = get_ground_truth(data, queries, k=5)
        assert len(gt) == 5
        assert all(len(row) == 5 for row in gt)

    def test_nearest_neighbour_is_correct(self):
        """Each query should have itself as the nearest neighbour."""
        from nilvec.metrics import get_ground_truth

        data = [[float(i)] * 8 for i in range(100)]
        # Queries are exact copies of the first 10 data vectors
        queries = data[:10]
        gt = get_ground_truth(data, queries, k=1)
        for q_idx, row in enumerate(gt):
            assert row[0] == q_idx, f"Query {q_idx}: expected NN={q_idx}, got {row[0]}"

    def test_ids_are_unique_per_query(self):
        import random

        from nilvec.metrics import get_ground_truth

        random.seed(0)
        data = [[random.random() for _ in range(16)] for _ in range(200)]
        queries = data[:5]
        gt = get_ground_truth(data, queries, k=10)
        for row in gt:
            assert len(set(row)) == len(row), "Duplicate IDs in ground truth"
