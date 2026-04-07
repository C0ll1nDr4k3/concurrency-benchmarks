"""Tests for nilvec/datasets.py — generate_data and DATASETS catalogue."""

import random

import pytest

from nilvec.datasets import DATASETS, generate_data

# ---------------------------------------------------------------------------
# generate_data
# ---------------------------------------------------------------------------


class TestGenerateData:
    def test_correct_number_of_vectors(self):
        vecs = generate_data(100, 32)
        assert len(vecs) == 100

    def test_correct_dimensionality(self):
        vecs = generate_data(50, 64)
        assert all(len(v) == 64 for v in vecs)

    def test_values_in_unit_interval(self):
        vecs = generate_data(200, 16)
        for v in vecs:
            for val in v:
                assert 0.0 <= val <= 1.0

    def test_returns_list_of_lists(self):
        vecs = generate_data(10, 4)
        assert isinstance(vecs, list)
        assert all(isinstance(v, list) for v in vecs)

    def test_single_vector(self):
        vecs = generate_data(1, 128)
        assert len(vecs) == 1
        assert len(vecs[0]) == 128

    def test_single_dimension(self):
        vecs = generate_data(10, 1)
        assert all(len(v) == 1 for v in vecs)

    def test_vectors_are_not_identical(self):
        """With random generation, 1000 128-d vectors should not all be the same."""
        random.seed(0)
        vecs = generate_data(1000, 128)
        unique = {tuple(v) for v in vecs}
        assert len(unique) > 1

    def test_zero_vectors(self):
        vecs = generate_data(0, 32)
        assert vecs == []


# ---------------------------------------------------------------------------
# DATASETS catalogue
# ---------------------------------------------------------------------------


REQUIRED_KEYS = {"url", "dim", "metric"}
VALID_METRICS = {"euclidean", "angular"}


class TestDatasetsCatalogue:
    def test_all_expected_datasets_present(self):
        expected = {
            "sift-128-euclidean",
            "fashion-mnist-784-euclidean",
            "glove-100-angular",
            "glove-25-angular",
            "gist-960-euclidean",
            "nytimes-256-angular",
            "mnist-784-euclidean",
        }
        assert set(DATASETS.keys()) == expected

    def test_every_entry_has_required_keys(self):
        for name, info in DATASETS.items():
            missing = REQUIRED_KEYS - set(info.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_dims_are_positive_integers(self):
        for name, info in DATASETS.items():
            assert isinstance(info["dim"], int), f"{name}: dim is not int"
            assert info["dim"] > 0, f"{name}: dim must be positive"

    def test_metrics_are_valid(self):
        for name, info in DATASETS.items():
            assert info["metric"] in VALID_METRICS, (
                f"{name}: unknown metric '{info['metric']}'"
            )

    def test_urls_are_strings_ending_in_hdf5(self):
        for name, info in DATASETS.items():
            assert isinstance(info["url"], str), f"{name}: url is not str"
            assert info["url"].endswith(".hdf5"), f"{name}: url does not end in .hdf5"

    def test_known_dims(self):
        assert DATASETS["sift-128-euclidean"]["dim"] == 128
        assert DATASETS["glove-25-angular"]["dim"] == 25
        assert DATASETS["glove-100-angular"]["dim"] == 100
        assert DATASETS["gist-960-euclidean"]["dim"] == 960
        assert DATASETS["nytimes-256-angular"]["dim"] == 256
        assert DATASETS["fashion-mnist-784-euclidean"]["dim"] == 784
        assert DATASETS["mnist-784-euclidean"]["dim"] == 784

    def test_known_metrics(self):
        assert DATASETS["sift-128-euclidean"]["metric"] == "euclidean"
        assert DATASETS["glove-25-angular"]["metric"] == "angular"
