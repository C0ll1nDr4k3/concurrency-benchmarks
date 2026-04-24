import json
import uuid

import numpy as np

try:
    import duckdb
except ImportError:
    duckdb = None


class BenchmarkResultsStore:
    def __init__(self, db_path):
        if duckdb is None:
            raise ImportError("duckdb not installed")
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                run_tag VARCHAR,
                dataset_name VARCHAR,
                dataset_path VARCHAR,
                dim INTEGER,
                num_vectors INTEGER,
                num_queries INTEGER,
                k INTEGER,
                workload_profile VARCHAR,
                thread_counts_json VARCHAR,
                only_external BOOLEAN,
                internal_only BOOLEAN,
                skip_recall BOOLEAN,
                skip_throughput BOOLEAN,
                limit_rows INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS throughput_points (
                run_id VARCHAR,
                index_name VARCHAR,
                thread_count INTEGER,
                throughput DOUBLE,
                conflict_rate DOUBLE,
                is_external BOOLEAN,
                search_p50_ms DOUBLE,
                search_p95_ms DOUBLE,
                search_p99_ms DOUBLE,
                insert_p50_ms DOUBLE,
                insert_p95_ms DOUBLE,
                insert_p99_ms DOUBLE
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS disjoint_points (
                run_id VARCHAR,
                index_name VARCHAR,
                thread_count INTEGER,
                layer INTEGER,
                num_vectors INTEGER,
                disjoint_rate DOUBLE
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recall_points (
                run_id VARCHAR,
                index_name VARCHAR,
                param_key VARCHAR,
                recall DOUBLE,
                qps DOUBLE,
                line_style VARCHAR,
                point_order INTEGER,
                p50_ms DOUBLE,
                p95_ms DOUBLE,
                p99_ms DOUBLE
            )
            """
        )
        # Migration: add internal_only column if it doesn't exist
        try:
            self.conn.execute(
                "ALTER TABLE benchmark_runs ADD COLUMN internal_only BOOLEAN"
            )
        except Exception:
            pass  # Column already exists
        try:
            self.conn.execute(
                "ALTER TABLE benchmark_runs ADD COLUMN workload_profile VARCHAR"
            )
        except Exception:
            pass  # Column already exists
        # Migration: add latency columns to existing tables that predate them
        for col in ("p50_ms", "p95_ms", "p99_ms"):
            try:
                self.conn.execute(f"ALTER TABLE recall_points ADD COLUMN {col} DOUBLE")
            except Exception:
                pass  # Column already exists
        for col in (
            "search_p50_ms",
            "search_p95_ms",
            "search_p99_ms",
            "insert_p50_ms",
            "insert_p95_ms",
            "insert_p99_ms",
        ):
            try:
                self.conn.execute(
                    f"ALTER TABLE throughput_points ADD COLUMN {col} DOUBLE"
                )
            except Exception:
                pass  # Column already exists
        self.conn.execute(
            """
            CREATE VIEW IF NOT EXISTS scaling_efficiency AS
            WITH base AS (
                SELECT run_id, index_name,
                       thread_count AS base_threads,
                       throughput   AS base_throughput
                FROM (
                    SELECT run_id, index_name, thread_count, throughput,
                           ROW_NUMBER() OVER (
                               PARTITION BY run_id, index_name
                               ORDER BY thread_count
                           ) AS rn
                    FROM throughput_points
                ) ranked
                WHERE rn = 1
            )
            SELECT
                tp.run_id,
                tp.index_name,
                tp.thread_count,
                tp.throughput,
                b.base_throughput,
                b.base_threads,
                tp.throughput / (
                    b.base_throughput
                    * (CAST(tp.thread_count AS DOUBLE) / b.base_threads)
                ) AS efficiency
            FROM throughput_points tp
            JOIN base b ON tp.run_id = b.run_id AND tp.index_name = b.index_name
            """
        )

    def start_run(self, run_meta):
        run_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO benchmark_runs (
                run_id, run_tag, dataset_name, dataset_path, dim, num_vectors,
                num_queries, k, workload_profile, thread_counts_json, only_external,
                internal_only, skip_recall, skip_throughput, limit_rows
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                run_meta.get("run_tag"),
                run_meta["dataset_name"],
                run_meta["dataset_path"],
                run_meta["dim"],
                run_meta["num_vectors"],
                run_meta["num_queries"],
                run_meta["k"],
                run_meta["workload_profile"],
                json.dumps(run_meta["thread_counts"]),
                run_meta["only_external"],
                run_meta["internal_only"],
                run_meta["skip_recall"],
                run_meta["skip_throughput"],
                run_meta["limit_rows"],
            ],
        )
        return run_id

    def save_throughput(
        self,
        run_id,
        throughput_results,
        conflict_results,
        external_names,
        thread_counts,
        latency_results=None,
    ):
        if latency_results is None:
            latency_results = {}
        rows = []
        for index_name, values in throughput_results.items():
            conflicts = conflict_results.get(index_name, [0] * len(values))
            lats = latency_results.get(
                index_name, [(None, None, None, None, None, None)] * len(values)
            )
            is_external = index_name in set(external_names)
            for thread_count, throughput, conflict_rate, lat in zip(
                thread_counts, values, conflicts, lats
            ):
                s_p50, s_p95, s_p99, i_p50, i_p95, i_p99 = lat
                rows.append(
                    (
                        run_id,
                        index_name,
                        int(thread_count),
                        float(throughput),
                        float(conflict_rate),
                        bool(is_external),
                        float(s_p50) if s_p50 is not None else None,
                        float(s_p95) if s_p95 is not None else None,
                        float(s_p99) if s_p99 is not None else None,
                        float(i_p50) if i_p50 is not None else None,
                        float(i_p95) if i_p95 is not None else None,
                        float(i_p99) if i_p99 is not None else None,
                    )
                )
        if rows:
            self.conn.executemany(
                """
                INSERT INTO throughput_points (
                    run_id, index_name, thread_count, throughput, conflict_rate, is_external,
                    search_p50_ms, search_p95_ms, search_p99_ms,
                    insert_p50_ms, insert_p95_ms, insert_p99_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def save_disjoint(self, run_id, disjoint_results, thread_counts):
        """
        disjoint_results: {index_name: [(N, [rate_by_layer]), ...]} parallel to
        thread_counts.
        """
        rows = []
        for index_name, series in disjoint_results.items():
            if not series:
                continue
            for thread_count, (n, rates) in zip(thread_counts, series):
                for layer, rate in enumerate(rates):
                    rows.append(
                        (
                            run_id,
                            index_name,
                            int(thread_count),
                            int(layer),
                            int(n),
                            float(rate),
                        )
                    )
        if rows:
            self.conn.executemany(
                """
                INSERT INTO disjoint_points (
                    run_id, index_name, thread_count, layer, num_vectors, disjoint_rate
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def save_recall_runs(self, run_id, recall_runs, latency_data=None):
        """
        recall_runs: list of (index_name, recalls, qps_vals, line_style)
        latency_data: optional dict of {index_name: [(p50_ms, p95_ms, p99_ms), ...]}
        """
        if latency_data is None:
            latency_data = {}
        rows = []
        for index_name, recalls, qps, line_style in recall_runs:
            lat = latency_data.get(index_name, [])
            for i, (recall, qps_val) in enumerate(zip(recalls, qps)):
                p50, p95, p99 = lat[i] if i < len(lat) else (None, None, None)
                rows.append(
                    (
                        run_id,
                        index_name,
                        f"{index_name}:{i}",
                        float(recall),
                        float(qps_val),
                        line_style,
                        i,
                        float(p50) if p50 is not None else None,
                        float(p95) if p95 is not None else None,
                        float(p99) if p99 is not None else None,
                    )
                )
        if rows:
            self.conn.executemany(
                """
                INSERT INTO recall_points (
                    run_id, index_name, param_key, recall, qps, line_style, point_order,
                    p50_ms, p95_ms, p99_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _compatible_where(self, meta):
        return (
            """
            dataset_name = ? AND dim = ? AND num_vectors = ? AND num_queries = ?
            AND k = ? AND workload_profile = ? AND thread_counts_json = ?
        """,
            [
                meta["dataset_name"],
                meta["dim"],
                meta["num_vectors"],
                meta["num_queries"],
                meta["k"],
                meta["workload_profile"],
                json.dumps(meta["thread_counts"]),
            ],
        )

    def cross_pollinate_throughput(
        self, meta, run_id, throughput_results, conflict_results, external_names
    ):
        where_clause, params = self._compatible_where(meta)
        rows = self.conn.execute(
            f"""
            WITH matching_runs AS (
                SELECT run_id, created_at
                FROM benchmark_runs
                WHERE {where_clause}
                  AND run_id != ?
            ),
            ranked_points AS (
                SELECT
                    tp.index_name,
                    tp.thread_count,
                    tp.throughput,
                    tp.conflict_rate,
                    tp.is_external,
                    ROW_NUMBER() OVER (
                        PARTITION BY tp.index_name, tp.thread_count
                        ORDER BY mr.created_at DESC
                    ) AS rn
                FROM throughput_points tp
                JOIN matching_runs mr ON tp.run_id = mr.run_id
            )
            SELECT index_name, thread_count, throughput, conflict_rate, is_external
            FROM ranked_points
            WHERE rn = 1
            """,
            params + [run_id],
        ).fetchall()

        thread_pos = {t: i for i, t in enumerate(meta["thread_counts"])}
        injected_indexes = set()
        external_set = set(external_names)
        for index_name, thread_count, throughput, conflict_rate, is_external in rows:
            if thread_count not in thread_pos:
                continue
            if index_name not in throughput_results:
                throughput_results[index_name] = [float("nan")] * len(
                    meta["thread_counts"]
                )
                conflict_results[index_name] = [float("nan")] * len(
                    meta["thread_counts"]
                )
                injected_indexes.add(index_name)
            i = thread_pos[thread_count]
            if np.isnan(throughput_results[index_name][i]):
                throughput_results[index_name][i] = float(throughput)
            if np.isnan(conflict_results[index_name][i]):
                conflict_results[index_name][i] = float(conflict_rate)
            if is_external:
                external_set.add(index_name)
        return (
            throughput_results,
            conflict_results,
            sorted(external_set),
            sorted(injected_indexes),
        )

    def cross_pollinate_recall(self, meta, run_id, existing_runs):
        existing_names = {name for name, _, _, _ in existing_runs}
        where_clause, params = self._compatible_where(meta)
        rows = self.conn.execute(
            f"""
            WITH matching_runs AS (
                SELECT run_id, created_at
                FROM benchmark_runs
                WHERE {where_clause}
                  AND run_id != ?
            ),
            latest_by_index AS (
                SELECT
                    rp.index_name,
                    rp.run_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY rp.index_name
                        ORDER BY mr.created_at DESC
                    ) AS rn
                FROM recall_points rp
                JOIN matching_runs mr ON rp.run_id = mr.run_id
            )
            SELECT rp.index_name, rp.recall, rp.qps, rp.line_style, rp.point_order
            FROM latest_by_index lbi
            JOIN recall_points rp
              ON rp.run_id = lbi.run_id
             AND rp.index_name = lbi.index_name
            WHERE lbi.rn = 1
            ORDER BY rp.index_name, rp.point_order
            """,
            params + [run_id],
        ).fetchall()

        grouped = {}
        for index_name, recall, qps, line_style, point_order in rows:
            if index_name in existing_names:
                continue
            grouped.setdefault(
                index_name, {"recalls": [], "qps": [], "style": line_style}
            )
            grouped[index_name]["recalls"].append(float(recall))
            grouped[index_name]["qps"].append(float(qps))

        injected = []
        for index_name, data in grouped.items():
            injected.append(
                (index_name, tuple(data["recalls"]), tuple(data["qps"]), data["style"])
            )
        return existing_runs + injected, [name for name, _, _, _ in injected]

    def close(self):
        self.conn.close()
