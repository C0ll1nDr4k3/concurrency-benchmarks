import sys
import os
import random

# Attempt to import nilvec
try:
    import nilvec

    print(f"Successfully imported nilvec from {nilvec.__file__}")
except ImportError:
    print(
        "Could not import nilvec. Make sure PYTHONPATH includes the build directory containing the .so file."
    )
    sys.exit(1)


def generate_data(n, dim):
    return [[random.random() for _ in range(dim)] for _ in range(n)]


def test_flat():
    print("\n--- Testing FlatVanilla ---")
    dim = 64
    index = nilvec.FlatVanilla(dim)
    data = generate_data(100, dim)

    print("Inserting 100 vectors...")
    ids = []
    for vec in data:
        ids.append(index.insert(vec))

    print(f"Index size: {index.size()}")
    assert index.size() == 100

    print("Searching for the first vector (k=5)...")
    query = data[0]
    res = index.search(query, 5)
    print(f"Search results ids: {res.ids}")
    print(f"Search results dists: {res.distances}")

    assert len(res.ids) == 5
    # The first result should be the vector itself (dist ~ 0)
    assert res.ids[0] == 0
    assert res.distances[0] < 1e-5


def test_hnsw():
    print("\n--- Testing HNSWVanilla ---")
    dim = 64
    # dim, M, ef_construction
    index = nilvec.HNSWVanilla(dim, 16, 100)
    data = generate_data(100, dim)

    print("Inserting 100 vectors...")
    for vec in data:
        index.insert(vec)

    print(f"Index size: {index.size()}")
    print(f"Max level: {index.max_level()}")
    assert index.size() == 100

    print("Searching...")
    query = data[0]
    res = index.search(query, 5)  # default ef
    print(f"Search results ids: {res.ids}")

    assert res.ids[0] == 0


def test_ivfflat():
    print("\n--- Testing IVFFlatVanilla ---")
    dim = 64
    # dim, nlist, nprobe
    index = nilvec.IVFFlatVanilla(dim, 10, 5)
    data = generate_data(500, dim)

    print("Training with 500 vectors...")
    # train takes vector<vector<float>>
    index.train(data)
    assert index.is_trained()

    print("Inserting 500 vectors...")
    for vec in data:
        index.insert(vec)

    print(f"Index size: {index.size()}")
    assert index.size() == 500

    print("Searching...")
    query = data[0]
    res = index.search(query, 5)
    print(f"Search results ids: {res.ids}")

    # With nprobe=5 and nlist=10, we cover 50% of buckets, recall should be decent.
    # We expect to find the query vector itself.
    found = False
    for id in res.ids:
        if id == 0:
            found = True
            break
    if found:
        print("Found ground truth vector.")
    else:
        print("Did NOT find ground truth vector (approximate search).")


if __name__ == "__main__":
    test_flat()
    test_hnsw()
    test_ivfflat()
    print("\nAll tests passed!")
