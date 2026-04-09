"""Tests for nilvec/store.py — BenchmarkResultsStore (DuckDB-backed)."""

import json
import uuid

import pytest

duckdb = pytest.importorskip("duckdb", reason="duckdb not installed")

from nilvec.store import BenchmarkResultsStore


@pytest.fixture()
def store(tmp_path):
    """Fresh in-memory store for each test."""
    s = BenchmarkResultsStore(":memory:")
    yield s
    s.close()


def _run_meta(**overrides):
    base = {
        "run_tag": "test",
        "dataset_name": "sift-128-euclidean",
        "dataset_path": "data/sift-128-euclidean.hdf5",
        "dim": 128,
        "num_vectors": 1000,
        "num_queries": 100,
        "k": 10,
        "workload_profile": "thread_split_quarter_writers",
        "thread_counts": [2, 4, 8],
        "only_external": False,
        "internal_only": False,
        "skip_recall": False,
        "skip_throughput": False,
        "limit_rows": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


class TestSchema:
    def test_tables_exist(self, store):
        tables = {
            row[0]
            for row in store.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        assert "benchmark_runs" in tables
        assert "throughput_points" in tables
        assert "recall_points" in tables

    def test_view_exists(self, store):
        views = {
            row[0]
            for row in store.conn.execute(
                "SELECT table_name FROM information_schema.views WHERE table_schema='main'"
            ).fetchall()
        }
        assert "scaling_efficiency" in views


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


class TestStartRun:
    def test_returns_valid_uuid(self, store):
        run_id = store.start_run(_run_meta())
        # Should not raise
        uuid.UUID(run_id)

    def test_run_is_persisted(self, store):
        run_id = store.start_run(_run_meta(run_tag="persist-check"))
        rows = store.conn.execute(
            "SELECT run_tag FROM benchmark_runs WHERE run_id = ?", [run_id]
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "persist-check"

    def test_thread_counts_stored_as_json(self, store):
        run_id = store.start_run(_run_meta(thread_counts=[2, 4, 8]))
        row = store.conn.execute(
            "SELECT thread_counts_json FROM benchmark_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        assert json.loads(row[0]) == [2, 4, 8]

    def test_multiple_runs_are_independent(self, store):
        id1 = store.start_run(_run_meta(run_tag="run-1"))
        id2 = store.start_run(_run_meta(run_tag="run-2"))
        assert id1 != id2
        count = store.conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# save_throughput
# ---------------------------------------------------------------------------


class TestSaveThroughput:
    def test_saves_rows(self, store):
        run_id = store.start_run(_run_meta())
        store.save_throughput(
            run_id=run_id,
            throughput_results={"HNSWVanilla": [1000.0, 1800.0, 3000.0]},
            conflict_results={"HNSWVanilla": [0.0, 0.0, 0.0]},
            external_names=[],
            thread_counts=[2, 4, 8],
        )
        rows = store.conn.execute(
            "SELECT thread_count, throughput FROM throughput_points WHERE run_id = ? ORDER BY thread_count",
            [run_id],
        ).fetchall()
        assert len(rows) == 3
        assert rows[0] == (2, 1000.0)
        assert rows[2] == (8, 3000.0)

    def test_external_flag_set_correctly(self, store):
        run_id = store.start_run(_run_meta())
        store.save_throughput(
            run_id=run_id,
            throughput_results={
                "HNSWVanilla": [1000.0, 1800.0, 3000.0],
                "FAISS-HNSW": [900.0, 1600.0, 2800.0],
            },
            conflict_results={},
            external_names=["FAISS-HNSW"],
            thread_counts=[2, 4, 8],
        )
        rows = store.conn.execute(
            "SELECT index_name, is_external FROM throughput_points WHERE run_id = ? AND thread_count = 2",
            [run_id],
        ).fetchall()
        flag_by_name = {r[0]: r[1] for r in rows}
        assert flag_by_name["HNSWVanilla"] is False
        assert flag_by_name["FAISS-HNSW"] is True

    def test_no_rows_saved_for_empty_results(self, store):
        run_id = store.start_run(_run_meta())
        store.save_throughput(
            run_id=run_id,
            throughput_results={},
            conflict_results={},
            external_names=[],
            thread_counts=[2, 4, 8],
        )
        count = store.conn.execute(
            "SELECT COUNT(*) FROM throughput_points WHERE run_id = ?", [run_id]
        ).fetchone()[0]
        assert count == 0

    def test_latency_columns_saved(self, store):
        run_id = store.start_run(_run_meta())
        store.save_throughput(
            run_id=run_id,
            throughput_results={"HNSWVanilla": [1000.0]},
            conflict_results={},
            external_names=[],
            thread_counts=[2],
            latency_results={"HNSWVanilla": [(1.0, 2.0, 3.0, 0.5, 1.5, 2.5)]},
        )
        row = store.conn.execute(
            "SELECT search_p50_ms, search_p99_ms, insert_p50_ms, insert_p99_ms "
            "FROM throughput_points WHERE run_id = ?",
            [run_id],
        ).fetchone()
        assert row[0] == pytest.approx(1.0)
        assert row[1] == pytest.approx(3.0)
        assert row[2] == pytest.approx(0.5)
        assert row[3] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# save_recall_runs
# ---------------------------------------------------------------------------


class TestSaveRecallRuns:
    def test_saves_rows(self, store):
        run_id = store.start_run(_run_meta())
        recall_runs = [("HNSWVanilla", [0.9, 0.95, 0.99], [500.0, 400.0, 300.0], "-")]
        store.save_recall_runs(run_id, recall_runs)
        rows = store.conn.execute(
            "SELECT recall, qps, point_order FROM recall_points WHERE run_id = ? ORDER BY point_order",
            [run_id],
        ).fetchall()
        assert len(rows) == 3
        assert rows[0][0] == pytest.approx(0.9)
        assert rows[0][1] == pytest.approx(500.0)
        assert rows[0][2] == 0

    def test_param_key_format(self, store):
        run_id = store.start_run(_run_meta())
        store.save_recall_runs(run_id, [("Idx", [0.9], [100.0], "-")])
        row = store.conn.execute(
            "SELECT param_key FROM recall_points WHERE run_id = ?", [run_id]
        ).fetchone()
        assert row[0] == "Idx:0"

    def test_latency_data_saved(self, store):
        run_id = store.start_run(_run_meta())
        store.save_recall_runs(
            run_id,
            [("Idx", [0.9], [100.0], "-")],
            latency_data={"Idx": [(1.0, 2.0, 3.0)]},
        )
        row = store.conn.execute(
            "SELECT p50_ms, p99_ms FROM recall_points WHERE run_id = ?", [run_id]
        ).fetchone()
        assert row[0] == pytest.approx(1.0)
        assert row[1] == pytest.approx(3.0)

    def test_empty_recall_runs(self, store):
        run_id = store.start_run(_run_meta())
        store.save_recall_runs(run_id, [])
        count = store.conn.execute(
            "SELECT COUNT(*) FROM recall_points WHERE run_id = ?", [run_id]
        ).fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# cross_pollinate_throughput
# ---------------------------------------------------------------------------


class TestCrossPollinate:
    def _seed_historical_run(self, store, meta, index_name, throughputs):
        run_id = store.start_run(meta)
        store.save_throughput(
            run_id=run_id,
            throughput_results={index_name: throughputs},
            conflict_results={index_name: [0.0] * len(throughputs)},
            external_names=[],
            thread_counts=meta["thread_counts"],
        )
        return run_id

    def test_injects_missing_index(self, store):
        meta = _run_meta()
        hist_id = self._seed_historical_run(
            store, meta, "OldIndex", [1000.0, 2000.0, 3000.0]
        )
        new_id = store.start_run(meta)

        throughput_results = {"NewIndex": [900.0, 1800.0, 2700.0]}
        conflict_results = {"NewIndex": [0.0, 0.0, 0.0]}
        tp, cr, ext, injected = store.cross_pollinate_throughput(
            meta, new_id, throughput_results, conflict_results, []
        )
        assert "OldIndex" in tp
        assert "OldIndex" in injected

    def test_does_not_overwrite_existing_index(self, store):
        meta = _run_meta()
        self._seed_historical_run(store, meta, "SharedIndex", [999.0, 1999.0, 2999.0])
        new_id = store.start_run(meta)

        throughput_results = {"SharedIndex": [1000.0, 2000.0, 3000.0]}
        conflict_results = {"SharedIndex": [0.0, 0.0, 0.0]}
        tp, cr, ext, injected = store.cross_pollinate_throughput(
            meta, new_id, throughput_results, conflict_results, []
        )
        # Current run values should be unchanged
        assert tp["SharedIndex"][0] == pytest.approx(1000.0)
        assert "SharedIndex" not in injected

    def test_no_compatible_runs_returns_unchanged(self, store):
        meta = _run_meta()
        new_id = store.start_run(meta)
        throughput_results = {"Idx": [500.0, 1000.0, 1500.0]}
        conflict_results = {"Idx": [0.0, 0.0, 0.0]}
        tp, cr, ext, injected = store.cross_pollinate_throughput(
            meta, new_id, throughput_results, conflict_results, []
        )
        assert tp == {"Idx": [500.0, 1000.0, 1500.0]}
        assert injected == []

    def test_cross_pollinate_recall(self, store):
        meta = _run_meta()
        hist_id = store.start_run(meta)
        store.save_recall_runs(
            hist_id,
            [("HistIndex", [0.9, 0.95], [500.0, 400.0], "-")],
        )
        new_id = store.start_run(meta)
        existing_runs = [("NewIndex", [0.8], [600.0], "--")]
        merged, injected = store.cross_pollinate_recall(meta, new_id, existing_runs)
        names = [r[0] for r in merged]
        assert "NewIndex" in names
        assert "HistIndex" in names
        assert "HistIndex" in injected

    def test_cross_pollinate_recall_skips_existing(self, store):
        meta = _run_meta()
        hist_id = store.start_run(meta)
        store.save_recall_runs(hist_id, [("SharedIndex", [0.9], [500.0], "-")])
        new_id = store.start_run(meta)
        existing_runs = [("SharedIndex", [0.85], [550.0], "-")]
        merged, injected = store.cross_pollinate_recall(meta, new_id, existing_runs)
        # SharedIndex appears exactly once (the current run's version)
        names = [r[0] for r in merged]
        assert names.count("SharedIndex") == 1
        assert "SharedIndex" not in injected
