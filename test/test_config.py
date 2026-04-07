"""Tests for nilvec/config.py — make_quantized_cls wrapper."""

import random

import pytest

nilvec = pytest.importorskip(
    "nilvec",
    reason="nilvec not importable - run `meson compile -C builddir` first",
)

from nilvec.config import make_quantized_cls


def _random_vecs(n, dim):
    return [[random.random() for _ in range(dim)] for _ in range(n)]


class TestMakeQuantizedCls:
    """Tests for the QuantizedIndex wrapper produced by make_quantized_cls."""

    def _make_sq_and_cls(self, dim, inner_cls):
        sq = nilvec.ScalarQuantizer(dim)
        data = _random_vecs(500, dim)
        sq.train(data)
        return make_quantized_cls(inner_cls, sq), sq

    # --- FlatVanilla + SQ8 ---

    def test_insert_and_size_flat(self):
        dim = 32
        cls, _ = self._make_sq_and_cls(dim, nilvec.FlatVanillaSQ8)
        index = cls(dim)
        vecs = _random_vecs(50, dim)
        for v in vecs:
            index.insert(v)
        assert index.size() == 50

    def test_search_returns_ids_flat(self):
        dim = 32
        cls, _ = self._make_sq_and_cls(dim, nilvec.FlatVanillaSQ8)
        index = cls(dim)
        vecs = _random_vecs(100, dim)
        for v in vecs:
            index.insert(v)
        result = index.search(vecs[0], 5)
        assert len(result.ids) == 5

    # --- HNSWVanillaSQ8 ---

    def test_hnsw_sq8_insert_search(self):
        random.seed(42)
        dim = 32
        cls, _ = self._make_sq_and_cls(dim, nilvec.HNSWVanillaSQ8)
        index = cls(dim, 16, 100)
        vecs = _random_vecs(200, dim)
        for v in vecs:
            index.insert(v)
        assert index.size() == 200
        result = index.search(vecs[0], 5)
        assert len(result.ids) == 5

    def test_hnsw_sq8_search_with_ef(self):
        random.seed(0)
        dim = 32
        cls, _ = self._make_sq_and_cls(dim, nilvec.HNSWVanillaSQ8)
        index = cls(dim, 16, 100)
        vecs = _random_vecs(200, dim)
        for v in vecs:
            index.insert(v)
        # ef > 0 branch
        result = index.search(vecs[0], 5, ef=50)
        assert len(result.ids) == 5

    def test_hnsw_sq8_search_without_ef(self):
        random.seed(1)
        dim = 32
        cls, _ = self._make_sq_and_cls(dim, nilvec.HNSWVanillaSQ8)
        index = cls(dim, 16, 100)
        vecs = _random_vecs(200, dim)
        for v in vecs:
            index.insert(v)
        # ef == 0 branch (default path)
        result = index.search(vecs[0], 5, ef=0)
        assert len(result.ids) == 5

    # --- IVFFlatVanillaSQ8 ---

    def test_ivfflat_sq8_train_and_search(self):
        random.seed(7)
        dim = 32
        sq = nilvec.ScalarQuantizer(dim)
        train_data = _random_vecs(500, dim)
        sq.train(train_data)
        cls = make_quantized_cls(nilvec.IVFFlatVanillaSQ8, sq)
        index = cls(dim, 10, 5)
        index.train(train_data)
        vecs = _random_vecs(200, dim)
        for v in vecs:
            index.insert(v)
        assert index.size() == 200
        result = index.search(train_data[0], 5)
        assert len(result.ids) == 5

    def test_ivfflat_sq8_set_nprobe(self):
        random.seed(8)
        dim = 32
        sq = nilvec.ScalarQuantizer(dim)
        train_data = _random_vecs(500, dim)
        sq.train(train_data)
        cls = make_quantized_cls(nilvec.IVFFlatVanillaSQ8, sq)
        index = cls(dim, 10, 5)
        index.train(train_data)
        for v in train_data:
            index.insert(v)
        # set_nprobe should not raise; use full nprobe for reliable recall
        index.set_nprobe(10)
        result = index.search(train_data[0], 5)
        assert len(result.ids) == 5

    # --- Wrapper properties ---

    def test_size_reflects_inserts(self):
        dim = 16
        cls, _ = self._make_sq_and_cls(dim, nilvec.FlatVanillaSQ8)
        index = cls(dim)
        assert index.size() == 0
        for v in _random_vecs(10, dim):
            index.insert(v)
        assert index.size() == 10

    def test_each_call_to_make_quantized_cls_is_independent(self):
        dim = 16
        sq = nilvec.ScalarQuantizer(dim)
        sq.train(_random_vecs(200, dim))
        cls_a = make_quantized_cls(nilvec.FlatVanillaSQ8, sq)
        cls_b = make_quantized_cls(nilvec.FlatVanillaSQ8, sq)
        assert cls_a is not cls_b
        idx_a = cls_a(dim)
        idx_b = cls_b(dim)
        for v in _random_vecs(5, dim):
            idx_a.insert(v)
        assert idx_a.size() == 5
        assert idx_b.size() == 0
