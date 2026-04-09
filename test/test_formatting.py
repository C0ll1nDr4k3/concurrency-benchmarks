"""Tests for nilvec/formatting.py"""

import re

import pytest

from nilvec.formatting import (
    _format_elapsed,
    format_benchmark_header,
    format_throughput_line,
    make_op_mix_schedule,
    parse_op_mix_bands,
)


def strip_ansi(s):
    """Remove ANSI escape codes from a string."""
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------------
# parse_op_mix_bands
# ---------------------------------------------------------------------------


class TestParseRwBands:
    def test_single_band(self):
        assert parse_op_mix_bands(["0.01-0.05"]) == [(0.01, 0.05)]

    def test_multiple_bands(self):
        result = parse_op_mix_bands(["0.01-0.05", "0.20-0.50"])
        assert result == [(0.01, 0.05), (0.20, 0.50)]

    def test_zero_to_one_extremes(self):
        assert parse_op_mix_bands(["0.0-1.0"]) == [(0.0, 1.0)]

    def test_equal_low_high(self):
        assert parse_op_mix_bands(["0.5-0.5"]) == [(0.5, 0.5)]

    def test_invalid_format_missing_hyphen(self):
        with pytest.raises(ValueError, match="Invalid band format"):
            parse_op_mix_bands(["0.01"])

    def test_invalid_format_too_many_parts(self):
        # split("-", 1) means "0.01-0.05-0.10" => ("0.01", "0.05-0.10")
        # float("0.05-0.10") raises ValueError from float()
        with pytest.raises(ValueError):
            parse_op_mix_bands(["0.01-0.05-0.10"])

    def test_out_of_range_low(self):
        # 1.1 is > 1.0, triggers the range check
        with pytest.raises(ValueError, match="Band values must be in"):
            parse_op_mix_bands(["1.1-0.5"])

    def test_out_of_range_high(self):
        with pytest.raises(ValueError, match="Band values must be in"):
            parse_op_mix_bands(["0.01-1.5"])

    def test_empty_list(self):
        assert parse_op_mix_bands([]) == []


# ---------------------------------------------------------------------------
# make_op_mix_schedule
# ---------------------------------------------------------------------------


class TestMakeRwSchedule:
    def test_single_thread_count(self):
        result = make_op_mix_schedule((0.1, 0.5), [4])
        assert result == [0.1]

    def test_two_thread_counts(self):
        result = make_op_mix_schedule((0.0, 1.0), [2, 4])
        assert result == [0.0, 1.0]

    def test_three_thread_counts_linear(self):
        result = make_op_mix_schedule((0.0, 1.0), [2, 4, 8])
        assert len(result) == 3
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_equal_band_is_flat(self):
        result = make_op_mix_schedule((0.3, 0.3), [2, 4, 8, 16])
        assert all(v == pytest.approx(0.3) for v in result)

    def test_seven_thread_counts(self):
        thread_counts = [2, 4, 8, 12, 16, 20, 24]
        result = make_op_mix_schedule((0.01, 0.05), thread_counts)
        assert len(result) == 7
        assert result[0] == pytest.approx(0.01)
        assert result[-1] == pytest.approx(0.05)
        # Monotonically increasing
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]


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

    def test_with_target_vs_achieved_write_ratio(self):
        line = strip_ansi(
            format_throughput_line(
                8,
                2,
                6,
                9000.0,
                8000.0,
                target_write_ratio=0.20,
                achieved_write_ratio=0.24,
            )
        )
        assert "W tgt/ach" in line
        assert "20.0%/24.0%" in line


# ---------------------------------------------------------------------------
# format_benchmark_header
# ---------------------------------------------------------------------------


class TestFormatBenchmarkHeader:
    def test_returns_string(self):
        header = format_benchmark_header("HNSWVanilla", 0.1)
        assert isinstance(header, str)

    def test_contains_index_name(self):
        header = strip_ansi(format_benchmark_header("HNSWVanilla", 0.1))
        assert "HNSWVanilla" in header

    def test_contains_ratio(self):
        header = strip_ansi(format_benchmark_header("HNSWVanilla", 0.1))
        assert "OpMixW=0.1" in header

    def test_external_names_accepted(self):
        for name in ["Redis", "Weaviate", "USearch", "FAISS-HNSW"]:
            header = format_benchmark_header(name, 0.1)
            assert isinstance(header, str)
            assert name in strip_ansi(header)

    def test_tuple_op_mix_ratio(self):
        header = strip_ansi(format_benchmark_header("HNSWVanilla", (0.01, 0.05)))
        assert "HNSWVanilla" in header
        # Should not contain the plain "OpMixW=..." form
        assert "OpMixW=" not in header
