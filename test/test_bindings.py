import random
import pytest

nilvec = pytest.importorskip(
    "nilvec",
    reason="nilvec not importable — run `meson compile -C builddir` first",
)


def generate_data(n, dim):
    return [[random.random() for _ in range(dim)] for _ in range(n)]


def test_flat():
    dim = 64
    index = nilvec.FlatVanilla(dim)
    data = generate_data(100, dim)

    ids = [index.insert(vec) for vec in data]

    assert index.size() == 100

    res = index.search(data[0], 5)
    assert len(res.ids) == 5
    assert res.ids[0] == 0
    assert res.distances[0] < 1e-5


def test_hnsw():
    dim = 64
    index = nilvec.HNSWVanilla(dim, 16, 100)
    data = generate_data(100, dim)

    for vec in data:
        index.insert(vec)

    assert index.size() == 100

    res = index.search(data[0], 5)
    assert res.ids[0] == 0


def test_ivfflat():
    dim = 64
    index = nilvec.IVFFlatVanilla(dim, 10, 5)
    data = generate_data(500, dim)

    index.train(data)
    assert index.is_trained()

    for vec in data:
        index.insert(vec)

    assert index.size() == 500

    res = index.search(data[0], 5)
    # nprobe=5 / nlist=10 covers 50 % of buckets; the query vector should be found.
    assert 0 in res.ids, "Ground-truth vector not found in approximate search results"
