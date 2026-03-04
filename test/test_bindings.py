import random
import pytest

nilvec = pytest.importorskip(
    "nilvec",
    reason="nilvec not importable - run `meson compile -C builddir` first",
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


def test_hnswivf():
    random.seed(42)
    dim = 64
    # M=16 yields ~500/16 ≈ 31 partitions; nprobe=10 covers ~1/3.
    index = nilvec.HNSWIVFCoarsePessimistic(dim, M=16, ef_construction=200, nprobe=10)
    data = generate_data(500, dim)

    for vec in data:
        index.insert(vec)

    assert index.size() == 500
    nparts = index.num_partitions()
    assert nparts > 0

    # Use high enough nprobe relative to partition count
    index.set_nprobe(max(10, nparts // 2))
    res = index.search(data[0], 5, ef=50)
    assert len(res.ids) == 5
    assert 0 in res.ids, "Ground-truth vector not found in HNSWIVF search results"


def test_hnswivf_remove():
    random.seed(123)
    dim = 32
    index = nilvec.HNSWIVFCoarsePessimistic(dim, M=8, ef_construction=100, nprobe=20)
    data = generate_data(200, dim)

    for vec in data:
        index.insert(vec)

    nparts_before = index.num_partitions()
    assert nparts_before > 0

    # Find a partition center (level >= 1 node) by checking num_partitions
    # Remove node 1 -- it may or may not be a center
    index.remove(1)
    assert index.size() == 199

    # Search should still work and not return the deleted node
    index.set_nprobe(nparts_before)
    res = index.search(data[0], 5, ef=50)
    assert 1 not in res.ids, "Deleted node should not appear in results"
