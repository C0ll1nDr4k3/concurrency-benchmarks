"""Tests for nilvec/cli.py — argument parser defaults and flag behaviour."""

import pytest

from nilvec.cli import build_parser


@pytest.fixture()
def parser():
    return build_parser()


class TestDefaults:
    def test_dataset_default(self, parser):
        args = parser.parse_args([])
        assert args.dataset == "sift-128-euclidean.hdf5"

    def test_limit_default(self, parser):
        args = parser.parse_args([])
        assert args.limit == 0

    def test_results_db_default(self, parser):
        args = parser.parse_args([])
        assert args.results_db == "benchmark_results.duckdb"

    def test_cross_pollinate_default(self, parser):
        args = parser.parse_args([])
        assert args.cross_pollinate is True

    def test_auto_start_redis_default(self, parser):
        args = parser.parse_args([])
        assert args.auto_start_redis is True

    def test_latency_sample_rate_default(self, parser):
        args = parser.parse_args([])
        assert args.latency_sample_rate == pytest.approx(0.01)

    def test_skip_recall_default_false(self, parser):
        args = parser.parse_args([])
        assert args.skip_recall is False

    def test_skip_throughput_default_false(self, parser):
        args = parser.parse_args([])
        assert args.skip_throughput is False

    def test_all_default_false(self, parser):
        args = parser.parse_args([])
        assert args.all is False

    def test_external_only_default_false(self, parser):
        args = parser.parse_args([])
        assert args.external_only is False

    def test_internal_only_default_false(self, parser):
        args = parser.parse_args([])
        assert args.internal_only is False

    def test_run_tag_default_empty(self, parser):
        args = parser.parse_args([])
        assert args.run_tag == ""


class TestFlags:
    def test_skip_recall_flag(self, parser):
        args = parser.parse_args(["--skip-recall"])
        assert args.skip_recall is True

    def test_skip_ann_alias(self, parser):
        args = parser.parse_args(["--skip-ann"])
        assert args.skip_recall is True

    def test_skip_throughput_flag(self, parser):
        args = parser.parse_args(["--skip-throughput"])
        assert args.skip_throughput is True

    def test_all_flag(self, parser):
        args = parser.parse_args(["--all"])
        assert args.all is True

    def test_external_only_flag(self, parser):
        args = parser.parse_args(["--external-only"])
        assert args.external_only is True

    def test_internal_only_flag(self, parser):
        args = parser.parse_args(["--internal-only"])
        assert args.internal_only is True


class TestValueArgs:
    def test_dataset_custom(self, parser):
        args = parser.parse_args(["--dataset", "glove-100-angular.hdf5"])
        assert args.dataset == "glove-100-angular.hdf5"

    def test_limit_custom(self, parser):
        args = parser.parse_args(["--limit", "5000"])
        assert args.limit == 5000

    def test_results_db_custom(self, parser):
        args = parser.parse_args(["--results-db", "my_results.duckdb"])
        assert args.results_db == "my_results.duckdb"

    def test_run_tag_custom(self, parser):
        args = parser.parse_args(["--run-tag", "experiment-1"])
        assert args.run_tag == "experiment-1"

    def test_latency_sample_rate_custom(self, parser):
        args = parser.parse_args(["--latency-sample-rate", "0.1"])
        assert args.latency_sample_rate == pytest.approx(0.1)

    def test_preload_ratio(self, parser):
        args = parser.parse_args(["--preload-ratio", "0.8"])
        assert args.preload_ratio == pytest.approx(0.8)

    def test_preload_ratio_default(self, parser):
        args = parser.parse_args([])
        assert args.preload_ratio == pytest.approx(0.5)

    def test_rw_ratio_flag_removed(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--rw-ratio", "0.3"])

    def test_rw_bands_flag_removed(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--rw-bands", "0.20-0.50"])

    def test_op_mix_ratio_flag_removed(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--op-mix-ratio", "0.3"])

    def test_op_mix_bands_flag_removed(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--op-mix-bands", "0.20-0.50"])
