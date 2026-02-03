import sys
import os
import time
import random
import argparse
import numpy as np
import threading
import matplotlib.pyplot as plt
import h5py
import requests
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pickle

import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

try:
    import faiss
except ImportError:
    print(f"{Fore.YELLOW}faiss not installed{Style.RESET_ALL}")
    faiss = None

try:
    import usearch.index
except ImportError:
    print(f"{Fore.YELLOW}usearch not installed{Style.RESET_ALL}")
    usearch = None

try:
    import pymilvus
except ImportError:
    print(f"{Fore.YELLOW}Milvus not installed{Style.RESET_ALL}")
    pymilvus = None

try:
    import weaviate
except ImportError:
    print(f"{Fore.YELLOW}Weaviate not installed{Style.RESET_ALL}")
    weaviate = None

try:
    from . import _nilvec as nilvec
except ImportError:
    try:
        import nilvec
    except ImportError:
        print("Error: Could not import nilvec. Run `meson compile -C builddir` first.")
        sys.exit(1)

# --- Configuration ---
DIM = 128
# These will be overridden by dataset if present
NUM_VECTORS = 10000
NUM_QUERIES = 1000
K = 10
DPI = 1200
RW_RATIO = 0.1

# Thread counts to test
THREAD_COUNTS = [2**n for n in range(1, 5)]


# Icon mapping: "Substring": ("path", zoom)
ICON_MAPPING = {
    "FAISS": ("paper/imgs/meta.png", 0.005),
    "USearch": ("paper/imgs/usearch.png", 0.015),
    "Weaviate": ("paper/imgs/weaviate.png", 0.015),
}

# Color mapping: "Substring": "color"
COLOR_MAPPING = {
    "FAISS": "#1877F2",    # Facebook Blue
    "USearch": "#192940",  # Dark Blue
    "Weaviate": "#ddd347", # Yellow
}

# --- Wrappers ---


class FaissHNSW:
    def __init__(self, dim, M=16, ef_construction=200):
        if faiss is None:
            raise ImportError("faiss not installed")
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.M = M
        self.ef_construction = ef_construction

    def insert(self, vec):
        # Faiss expects numpy arrays
        v = np.array([vec], dtype=np.float32)
        self.index.add(v)

    def search(self, query, k, ef=None):
        if ef is not None:
            self.index.hnsw.efSearch = ef
        v = np.array([query], dtype=np.float32)
        D, I = self.index.search(v, k)
        return type(
            "Result", (object,), {"ids": I[0].tolist(), "distances": D[0].tolist()}
        )()

    def set_nprobe(self, n):
        pass  # Not applicable

    def train(self, data):
        pass  # Not needed


class FaissIVF:
    def __init__(self, dim, nlist=100, nprobe=1):
        if faiss is None:
            raise ImportError("faiss not installed")
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        self.index.nprobe = nprobe
        self.nlist = nlist

    def train(self, data):
        # Faiss training needs numpy
        d = np.array(data, dtype=np.float32)
        self.index.train(d)

    def insert(self, vec):
        v = np.array([vec], dtype=np.float32)
        self.index.add(v)

    def search(self, query, k, ef=None):
        v = np.array([query], dtype=np.float32)
        D, I = self.index.search(v, k)
        return type(
            "Result", (object,), {"ids": I[0].tolist(), "distances": D[0].tolist()}
        )()

    def set_nprobe(self, n):
        self.index.nprobe = n


class USearchIndex:
    def __init__(self, dim, M=16, ef_construction=200):
        if usearch is None:
            raise ImportError("usearch not installed")
        self.index = usearch.index.Index(
            ndim=dim,
            metric="l2sq",
            connectivity=M,
            expansion_add=ef_construction,
            expansion_search=ef_construction,
        )
        self.dim = dim

    def insert(self, vec):
        key = random.getrandbits(63)
        v = np.array(vec, dtype=np.float32)
        self.index.add(key, v)

    def train(self, data):
        pass

    def search(self, query, k, ef=None):
        if ef is not None:
            try:
                self.index.change_expansion_search(ef)
            except:
                pass
        v = np.array(query, dtype=np.float32)
        matches = self.index.search(v, k)
        return type(
            "Result",
            (object,),
            {"ids": matches.keys.tolist(), "distances": matches.distances.tolist()},
        )()

    def set_nprobe(self, n):
        pass


class MilvusIndex:
    def __init__(self, dim, M=16, ef_construction=200):
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError("pymilvus not installed")

        self.client = MilvusClient("./milvus_demo.db")
        self.collection_name = "nilvec_bench"
        self.dim = dim
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name, dimension=dim, auto_id=True
        )
        # Note: Milvus Lite parameters are often implicit or set via index creation
        # verifying index creation parameters might be needed for fair comparison

    def insert(self, vec):
        data = [{"vector": vec}]
        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query, k, ef=None):
        search_params = {}
        if ef:
            search_params = {"params": {"ef": ef}}

        res = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            limit=k,
            search_params=search_params,
            output_fields=["id"],
        )
        # res is list of list of matches
        ids = [hit["id"] for hit in res[0]]
        distances = [hit["distance"] for hit in res[0]]
        return type("Result", (object,), {"ids": ids, "distances": distances})()

    def train(self, data):
        pass

    def set_nprobe(self, n):
        pass

    def close(self):
        self.client.close()


class WeaviateIndex:
    def __init__(self, dim, M=16, ef_construction=200):
        try:
            import weaviate
            import weaviate.classes.config as wvc
        except ImportError:
            raise ImportError("weaviate-client not installed")

        # Connect to embedded Weaviate
        self.client = weaviate.connect_to_embedded()
        self.collection_name = "NilVecBench"

        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        # Create collection
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            vector_index_config=wvc.Configure.VectorIndex.hnsw(
                ef_construction=ef_construction,
                max_connections=M,
            ),
            properties=[wvc.Property(name="dummy", data_type=wvc.DataType.INT)],
        )
        self.collection = self.client.collections.get(self.collection_name)

    def insert(self, vec):
        # Weaviate expects vectors as list of floats
        self.collection.data.insert(properties={"dummy": 1}, vector=vec)

    def search(self, query, k, ef=None):
        # Dynamic ef setting not directly exposed in simple query, uses index config
        # Weaviate usually requires re-configuring index to change ef at search time (efSearch)
        # For simple bench we might skip per-query ef setting or assume fixed defaults
        res = self.collection.query.near_vector(
            near_vector=query,
            limit=k,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
        )

        ids = [obj.uuid for obj in res.objects]
        distances = [obj.metadata.distance for obj in res.objects]
        return type("Result", (object,), {"ids": ids, "distances": distances})()

    def train(self, data):
        pass

    def set_nprobe(self, n):
        pass

    def close(self):
        self.client.close()


# --- Helpers ---


# --- Datasets ---

DATASETS = {
    "sift-128-euclidean": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "dim": 128,
        "metric": "euclidean",
    },
    "fashion-mnist-784-euclidean": {
        "url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "dim": 784,
        "metric": "euclidean",
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "dim": 100,
        "metric": "angular",
    },
    "glove-25-angular": {
        "url": "http://ann-benchmarks.com/glove-25-angular.hdf5",
        "dim": 25,
        "metric": "angular",
    },
    "gist-960-euclidean": {
        "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "dim": 960,
        "metric": "euclidean",
    },
    "nytimes-256-angular": {
        "url": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "dim": 256,
        "metric": "angular",
    },
    "mnist-784-euclidean": {
        "url": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "dim": 784,
        "metric": "euclidean",
    },
}


def download_dataset(url, path):
    print(f"Downloading {url} to {path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download dataset: {response.status_code}")
        sys.exit(1)


def load_dataset(path, limit=0):
    # Verify/Create data directory if we need to look there or download there
    data_dir = "data"

    # If the path provided doesn't exist, check if it's in data/
    if not os.path.exists(path):
        candidate_path = os.path.join(data_dir, os.path.basename(path))
        if os.path.exists(candidate_path):
            path = candidate_path
        elif os.sep not in path:
            # If provided path has no separator, assume we should use data_dir for download
            path = candidate_path

    # Infer dataset name from path if possible, or assume it matches one of our keys
    filename = os.path.basename(path)
    dataset_name = os.path.splitext(filename)[0]

    if not os.path.exists(path):
        if dataset_name in DATASETS:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            url = DATASETS[dataset_name]["url"]
            download_dataset(url, path)
        else:
            # Fallback or error?
            # Existing behavior was hardcoded SIFT
            # Let's see if it looks like a standard name
            print(f"Dataset {path} not found.")
            if dataset_name not in DATASETS:
                print(
                    f"Unknown dataset name '{dataset_name}'. Available: {list(DATASETS.keys())}"
                )
                # Try simple default if requested path was literally the default string
                if "sift" in path:
                    url = DATASETS["sift-128-euclidean"]["url"]
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                    download_dataset(url, path)
                else:
                    sys.exit(1)

    print(f"Loading dataset from {path}...")
    f = h5py.File(path, "r")

    # helper to convert hdf5 dataset to list of vectors (float)
    def to_list(dset, limit=0):
        if limit > 0:
            return [list(map(float, vec)) for vec in dset[:limit]]
        return [list(map(float, vec)) for vec in dset]

    train = to_list(f["train"], limit)
    test = to_list(f["test"], limit)

    # Ground truth
    if "neighbors" in f:
        # If we slice the dataset, the original GT is likely invalid/mismatched if it refers to IDs > limit
        # However, for simply running throughput benchmarks, we don't strictly need accurate GT
        # unless we are running recall.
        # But if we limit queries too, we need matching GT size.
        if limit > 0:
            gt = None  # Recompute
        else:
            gt = [list(map(int, vec[:K])) for vec in f["neighbors"]]
    else:
        gt = None

    f.close()
    return train, test, gt


def generate_data(n, dim):
    return [[random.random() for _ in range(dim)] for _ in range(n)]


def compute_recall(results, ground_truth, k):
    """Compute recall@k"""
    recall_sum = 0.0
    for res_ids, true_ids in zip(results, ground_truth):
        truth_set = set(true_ids[:k])
        # Intersection
        found = 0
        for rid in res_ids:
            if rid in truth_set:
                found += 1
        recall_sum += found / min(k, len(true_ids))
    return recall_sum / len(results)


def get_ground_truth(data, queries, k):
    """Compute brute-force ground truth using FlatVanilla"""
    print("Computing ground truth...")
    index = nilvec.FlatVanilla(len(data[0]))
    for vec in data:
        index.insert(vec)

    gt = []
    for q in queries:
        res = index.search(q, k)
        gt.append(res.ids)
    return gt


# --- Benchmarks ---


def benchmark_recall_vs_qps(
    index_cls, index_name, data, queries, gt, k, index_args=None, search_params=None
):
    """
    Run Recall vs QPS benchmark.
    search_params: list of dicts, e.g. [{'ef': 10}, {'ef': 20}]
    """
    if index_args is None:
        index_args = []

    print(f"\nBenchmarking Recall vs QPS: {index_name}")

    # helper to instantiate properly
    if "IVF" in index_name:
        # IVFFlatVanilla takes (dim, nlist, nprobe)
        # We assume index_cls handles the constructor
        index = index_cls(DIM, *index_args)
        print("  Training...")
        index.train(data)
    else:
        index = index_cls(DIM, *index_args)

    print("  Inserting...")
    start = time.time()
    for vec in data:
        index.insert(vec)
    print(f"  Insert time: {time.time() - start:.2f}s")

    results = []

    for params in search_params:
        # Apply search parameters
        if "nprobe" in params:
            index.set_nprobe(params["nprobe"])

        # Search
        start = time.time()
        res_ids_list = []
        for q in queries:
            # HNSW search takes 3 args: query, k, ef
            # IVFFlat search takes 2 args: query, k (nprobe set via setter)
            if "ef" in params:
                res = index.search(q, k, params["ef"])
            else:
                res = index.search(q, k)
            res_ids_list.append(res.ids)

        duration = time.time() - start
        qps = len(queries) / duration
        recall = compute_recall(res_ids_list, gt, k)

        print(f"  Params: {params} -> Recall: {recall:.4f}, QPS: {qps:.0f}")
        results.append((recall, qps))

    return results


def benchmark_throughput_vs_threads(
    index_cls, index_name, data, queries, k, index_args=None, rw_ratio=0.5
):
    """
    Run Throughput vs Threads benchmark with configurable R/W split.
    rw_ratio: Fraction of threads performing inserts (0.0 = Read Only, 1.0 = Write Only)
    """
    if index_args is None:
        index_args = []

    print(f"\nBenchmarking Throughput (W/R={rw_ratio}): {index_name}")
    results = []
    conflict_rates = []
    
    stop_event = threading.Event()

    for num_threads in THREAD_COUNTS:
        if stop_event.is_set():
             break

        # Re-create index for each run to be clean
        if "IVF" in index_name:
            index = index_cls(DIM, *index_args)
            index.train(data)
        else:
            index = index_cls(DIM, *index_args)

        # Pre-load some data
        initial_size = len(data) // 2
        for i in range(initial_size):
            index.insert(data[i])

        remaining_data = data[initial_size:]

        ops_count = 0
        start_time = time.time()

        threads = []

        # Define tasks
        def search_worker(qs):
            nonlocal ops_count
            local_ops = 0
            # Run for fixed iterations
            for _ in range(5):
                if stop_event.is_set(): break
                for q in qs:
                    if stop_event.is_set(): break
                    index.search(q, k)
                    local_ops += 1
            return local_ops

        def insert_worker(vecs):
            nonlocal ops_count
            local_ops = 0
            for v in vecs:
                if stop_event.is_set(): break
                index.insert(v)
                local_ops += 1
            return local_ops

        # Calculate thread split
        num_insert_threads = int(num_threads * rw_ratio)
        if rw_ratio > 0.0 and rw_ratio < 1.0:
            if num_insert_threads == 0 and num_threads > 0:
                num_insert_threads = 1
            elif num_insert_threads == num_threads and num_threads > 1:
                num_insert_threads -= 1
        elif rw_ratio == 0.0:
            num_insert_threads = 0
        elif rw_ratio == 1.0:
            num_insert_threads = num_threads

        num_search_threads = num_threads - num_insert_threads

        # Launch Insert Threads
        if num_insert_threads > 0:
            chunk_s = len(remaining_data) // num_insert_threads
            for i in range(num_insert_threads):
                start = i * chunk_s
                end = (
                    (i + 1) * chunk_s
                    if i < num_insert_threads - 1
                    else len(remaining_data)
                )
                t = threading.Thread(
                    target=insert_worker, args=(remaining_data[start:end],)
                )
                threads.append(t)
                t.start()

        # Launch Search Threads
        if num_search_threads > 0:
            for i in range(num_search_threads):
                t = threading.Thread(target=search_worker, args=(queries,))
                threads.append(t)
                t.start()

        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
             print(f"\nSkipping {index_name}: Operation has been terminated")
             stop_event.set()
             # Cleanup threads
             for t in threads:
                 t.join(timeout=0.1)
             
             if hasattr(index, "close"):
                 index.close()
             return None, None

        duration = time.time() - start_time

        # Estimate ops
        total_inserts = 0
        if num_insert_threads > 0:
            total_inserts = len(remaining_data)  # Approximate given we chunked it

        total_searches = 0
        if num_search_threads > 0:
            total_searches = num_search_threads * len(queries) * 5

        total_ops = total_inserts + total_searches
        throughput = total_ops / duration
        print(
            f"  Threads: {num_threads} (W={num_insert_threads}, R={num_search_threads}) -> Throughput: {throughput:.0f} ops/s"
        )
        results.append(throughput)

        # Check conflict stats
        if hasattr(index, "conflict_stats"):
            stats = index.conflict_stats()
            rate = max(stats.insert_conflict_rate(), stats.search_conflict_rate())
            conflict_rates.append(rate)
        else:
            conflict_rates.append(0)

        if hasattr(index, "close"):
            index.close()

    return results, conflict_rates


# --- Main ---


def run_benchmark(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip-recall", action="store_true")
        parser.add_argument("--skip-throughput", action="store_true")
        parser.add_argument(
            "--dataset",
            type=str,
            default="sift-128-euclidean.hdf5",
            help="Path to HDF5 dataset",
        )
        parser.add_argument(
            "--rw-ratio",
            type=float,
            default=RW_RATIO,
            help="Read/Write ratio (0.0=Read, 1.0=Write)",
        )
        parser.add_argument(
            "--limit", type=int, default=0, help="Limit number of vectors (0 = no limit)"
        )
        parser.add_argument(
            "--only-external", action="store_true", help="Run only external benchmarks"
        )
        parser.add_argument(
            "--save-results", type=str, default="benchmark_results.pkl", help="File to save results to"
        )
        args = parser.parse_args()



    # Load Data
    # Check if we likely have a valid dataset to load
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    if args.dataset and (os.path.exists(args.dataset) or dataset_name in DATASETS):
        data, queries, gt = load_dataset(args.dataset, args.limit)

        if args.limit > 0:
            print(f"Limiting dataset to first {args.limit} vectors...")
            # Data already sliced in load_dataset

            # Recompute GT for subset if needed, but for simplicity we'll just recompute based on subset
            gt = None

        global DIM, NUM_VECTORS, NUM_QUERIES
        DIM = len(data[0])
        NUM_VECTORS = len(data)
        NUM_QUERIES = len(queries)
        if gt is None:
            gt = get_ground_truth(data, queries, K)
    else:
        print(f"Generating {NUM_VECTORS} vectors (dim={DIM})...")
        data = generate_data(NUM_VECTORS, DIM)
        queries = generate_data(NUM_QUERIES, DIM)
        gt = get_ground_truth(data, queries, K)
    
    print(f"Dataset: N={NUM_VECTORS}, Q={NUM_QUERIES}, Dim={DIM}")



    results_cache = {
        "recall_vs_qps": {"runs": [], "K": K, "DIM": DIM},
        "throughput": {},
        "conflicts": {},
        "thread_counts": THREAD_COUNTS,
        "rw_ratio": args.rw_ratio,
        "external_names": []
    }

    # Indexes to test
    # HNSW: M=16, ef_cons=200
    hnsw_args = [16, 200]
    # IVFFlat: nlist=sqrt(N), nprobe variable
    nlist = int(NUM_VECTORS**0.5)
    ivf_args = [nlist, 1]

    # --- Recall vs QPS ---
    if not args.skip_recall:
        print("\n=== Recall vs QPS Benchmark ===")
        plt.figure(figsize=(10, 6))

        # HNSW Variants
        ef_values = [2**n for n in range(1, 4)]
        hnsw_params = [{"ef": ef} for ef in ef_values]

        res = benchmark_recall_vs_qps(
            nilvec.HNSWVanilla,
            "HNSW Vanilla",
            data,
            queries,
            gt,
            K,
            hnsw_args,
            hnsw_params,
        )
        recalls, qps = zip(*res)
        results_cache["recall_vs_qps"]["runs"].append(("HNSW Vanilla", recalls, qps, "o-"))
        plt.plot(recalls, qps, "o-", label="HNSW Vanilla")

        # IVFFlat Variants
        nprobe_values = [1, 2, 4, 8, 16, 32]
        ivf_params = [{"nprobe": np} for np in nprobe_values if np < nlist]

        res = benchmark_recall_vs_qps(
            nilvec.IVFFlatVanilla,
            "IVFFlat Vanilla",
            data,
            queries,
            gt,
            K,
            ivf_args,
            ivf_params,
        )
        recalls, qps = zip(*res)
        results_cache["recall_vs_qps"]["runs"].append(("IVFFlat Vanilla", recalls, qps, "s-"))
        plt.plot(recalls, qps, "s-", label="IVFFlat Vanilla")

        plt.xlabel("Recall")
        plt.ylabel("QPS (log scale)")
        plt.yscale("log")
        plt.title(f"Recall vs QPS (K={K}, Dim={DIM})")
        plt.legend()
        plt.grid(True)
        plt.savefig("paper/plots/recall_vs_qps.svg", dpi=DPI)
        print("Saved paper/plots/recall_vs_qps.svg")

    # --- throughput vs Threads ---
    if not args.skip_throughput:
        print("\n=== Throughput vs Threads Benchmark ===")

        # Define indexes to compare
        indexes = [
            # (nilvec.HNSWVanilla, "HNSW Vanilla", hnsw_args),
            (nilvec.HNSWCoarseOptimistic, "HNSW Coarse Opt", hnsw_args),
            (nilvec.HNSWCoarsePessimistic, "HNSW Coarse Pess", hnsw_args),
            (nilvec.HNSWFineOptimistic, "HNSW Fine Opt", hnsw_args),
            (nilvec.HNSWFinePessimistic, "HNSW Fine Pess", hnsw_args),
            # IVFFlat
            (nilvec.IVFFlatCoarseOptimistic, "IVF Coarse Opt", ivf_args),
            (nilvec.IVFFlatFineOptimistic, "IVF Fine Opt", ivf_args),
        ]

        # Run benchmarks and collect data
        throughput_results = {}
        conflict_results = {}

        # Internal Indexes
        for cls, name, iargs in indexes:
            if args.only_external:
                continue
            try:
                res, conflicts = benchmark_throughput_vs_threads(
                    cls, name, data, queries, K, iargs, rw_ratio=args.rw_ratio
                )
                if res is None:
                    continue
                throughput_results[name] = res
                conflict_results[name] = conflicts
            except Exception as e:
                print(f"Skipping {name}: {e}")

        # External Indexes
        externals = []
        if faiss:
            externals.append((FaissHNSW, "FAISS HNSW", hnsw_args))
            externals.append((FaissIVF, "FAISS IVF", ivf_args))
        if usearch:
            externals.append((USearchIndex, "USearch", hnsw_args))
        # if pymilvus:
        #     externals.append((MilvusIndex, "Milvus", hnsw_args))
        if weaviate:
            externals.append((WeaviateIndex, "Weaviate", hnsw_args))

        for cls, name, iargs in externals:
            try:
                res, conflicts = benchmark_throughput_vs_threads(
                    cls, name, data, queries, K, iargs, rw_ratio=args.rw_ratio
                )
                if res is None:
                    continue
                throughput_results[name] = res
                # External libs don't report conflict rates usually, but we capture 0s
                conflict_results[name] = conflicts
            except Exception as e:
                print(f"Skipping {name}: {e}")

        # Cache results
        results_cache["throughput"] = throughput_results
        results_cache["conflicts"] = conflict_results
        results_cache["external_names"] = [e[1] for e in externals]

        # Plot 1: Throughput
        plt.figure(figsize=(8, 6))
        ax = plt.gca()


        loaded_icons = {}
        for key, (path, zoom) in ICON_MAPPING.items():
            if os.path.exists(path):
                try:
                    loaded_icons[key] = (mpimg.imread(path), zoom)
                except Exception as e:
                    print(f"Could not load icon {path}: {e}")

        for name, res in throughput_results.items():
            external_names = [e[1] for e in externals]

            # Check for icon match
            icon_data = None
            for key, (img, zoom) in loaded_icons.items():
                if key in name:
                    icon_data = (img, zoom)
                    break

            # Check for color match
            color = None
            for key, c in COLOR_MAPPING.items():
                if key in name:
                    color = c
                    break

            if icon_data:
                img, zoom = icon_data
                plt.plot(THREAD_COUNTS, res, "--", label=name, alpha=0.75, color=color)
                for x, y in zip(THREAD_COUNTS, res):
                    im = OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                    ax.add_artist(ab)
            else:
                style = "*--" if name in external_names else "o-"
                plt.plot(THREAD_COUNTS, res, style, label=name, alpha=0.75, color=color)

        plt.xlabel("Threads")
        plt.ylabel("Ops/sec")
        plt.title(f"Throughput (W:{args.rw_ratio:.1f}, R:{1.0 - args.rw_ratio:.1f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("paper/plots/throughput_scaling.svg", dpi=DPI)
        print("Saved plots/throughput_scaling.svg")

        # Plot 2: Conflict Rates (Optimistic only)
        plt.figure(figsize=(8, 6))
        has_conflicts = False
        for name, conflicts in conflict_results.items():
            if "Opt" in name:
                plt.plot(THREAD_COUNTS, conflicts, "x--", label=name)
                has_conflicts = True

        if has_conflicts:
            plt.xlabel("Threads")
            plt.ylabel("Conflict Rate (%)")
            plt.title("Conflict Rate (Optimistic)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("paper/plots/conflict_rate.svg", dpi=DPI)
            print("Saved paper/plots/conflict_rate.svg")

    if args.save_results:
        print(f"Saving results to {args.save_results}...")
        with open(args.save_results, "wb") as f:
            pickle.dump(results_cache, f)


if __name__ == "__main__":
    if not os.path.exists("paper/plots"):
        os.makedirs("paper/plots", exist_ok=True)
    run_benchmark()
