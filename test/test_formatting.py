"""Tests for nilvec/formatting.py"""

import re

from nilvec.formatting import (
    _format_elapsed,
    format_benchmark_header,
    format_throughput_line,
)


def strip_ansi(s):
    """Remove ANSI escape codes from a string."""
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------------
# _format_elapsed
# ---------------------------------------------------------------------------


class TestFormatElapsed:
    def test_seconds_below_60(self):
        assert _format_elapsed(5.0) == "5.00s"
        assert _format_elapsed(0.5) == "0.50s"
        assert _format_elapsed(59.99) == "59.99s"

    def test_exactly_60_seconds(self):
        result = _format_elapsed(60.0)
        assert "1m" in result

    def test_minutes(self):
        result = _format_elapsed(90.0)
        assert "1m" in result
        assert "30.0s" in result

    def test_hours(self):
        result = _format_elapsed(3661.0)
        assert "1h" in result
        assert "1m" in result

    def test_zero(self):
        assert _format_elapsed(0.0) == "0.00s"


# ---------------------------------------------------------------------------
# format_throughput_line
# ---------------------------------------------------------------------------


class TestFormatThroughputLine:
    def test_basic_output_contains_threads_and_throughput(self):
        line = strip_ansi(
            format_throughput_line(
                num_threads=4,
                num_insert_threads=1,
                num_search_threads=3,
                throughput=5000.0,
                prev=None,
            )
        )
        assert "4" in line
        assert "5000" in line
        assert "W=1" in line
        assert "R=3" in line

    def test_no_prev_uses_neutral_color(self):
        # Just verifies it runs without error and produces a string
        line = format_throughput_line(4, 1, 3, 5000.0, None)
        assert isinstance(line, str)

    def test_increase_over_prev(self):
        line = format_throughput_line(4, 1, 3, 6000.0, 5000.0)
        assert isinstance(line, str)
        assert "6000" in strip_ansi(line)

    def test_decrease_over_prev(self):
        line = format_throughput_line(4, 1, 3, 4000.0, 5000.0)
        assert isinstance(line, str)
        assert "4000" in strip_ansi(line)

    def test_stable_over_prev(self):
        line = format_throughput_line(4, 1, 3, 5010.0, 5000.0)
        assert isinstance(line, str)

    def test_with_build_time(self):
        line = strip_ansi(format_throughput_line(4, 1, 3, 5000.0, None, build_time=2.5))
        assert "2.50s" in line
        assert "Build" in line

    def test_with_build_time_milliseconds(self):
        line = strip_ansi(
            format_throughput_line(4, 1, 3, 5000.0, None, build_time=0.0042)
        )
        assert "4.20ms" in line

    def test_with_build_time_microseconds(self):
        line = strip_ansi(
            format_throughput_line(4, 1, 3, 5000.0, None, build_time=0.0000009)
        )
        assert "1us" in line

    def test_with_search_latencies(self):
        line = strip_ansi(
            format_throughput_line(
                4, 1, 3, 5000.0, None, search_latencies=(1.0, 2.0, 3.0)
            )
        )
        assert "1.0" in line
        assert "3.0" in line
        assert "R p50" in line

    def test_with_insert_latencies(self):
        line = strip_ansi(
            format_throughput_line(
                4, 1, 3, 5000.0, None, insert_latencies=(0.5, 1.5, 2.5)
            )
        )
        assert "W p50" in line

    def test_without_optional_args(self):
        line = strip_ansi(format_throughput_line(8, 2, 6, 9000.0, 8000.0))
        assert "Build" not in line
        assert "p50" not in line


# ---------------------------------------------------------------------------
# format_benchmark_header
# ---------------------------------------------------------------------------


class TestFormatBenchmarkHeader:
    def test_returns_string(self):
        header = format_benchmark_header("HNSWVanilla")
        assert isinstance(header, str)

    def test_contains_index_name(self):
        header = strip_ansi(format_benchmark_header("HNSWVanilla"))
        assert "HNSWVanilla" in header

    def test_contains_benchmark_label(self):
        header = strip_ansi(format_benchmark_header("HNSWVanilla"))
        assert "Benchmarking Throughput" in header

    def test_external_names_accepted(self):
        for name in ["Redis", "Weaviate", "USearch", "FAISS-HNSW"]:
            header = format_benchmark_header(name)
            assert isinstance(header, str)
            assert name in strip_ansi(header)
