import sys
import os
import time
import random
import argparse
import json
import uuid
import subprocess
from urllib.parse import urlparse
import numpy as np
import threading
import tempfile
import shutil
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
    import qdrant_client
except ImportError:
    print(f"{Fore.YELLOW}Qdrant not installed{Style.RESET_ALL}")
    qdrant_client = None

try:
    import redis
except ImportError:
    print(f"{Fore.YELLOW}Redis client not installed{Style.RESET_ALL}")
    redis = None

try:
    import duckdb
except ImportError:
    print(f"{Fore.YELLOW}duckdb not installed{Style.RESET_ALL}")
    duckdb = None

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
    "Qdrant": ("paper/imgs/qdrant.png", 0.005),
    "Redis": ("paper/imgs/redis.png", 0.015),
}

# Color mapping: "Substring": "color"
COLOR_MAPPING = {
    "FAISS": "#1877F2",  # Facebook Blue
    "USearch": "#192940",  # Dark Blue
    "Weaviate": "#ddd347",  # Yellow
    "Qdrant": "#dc244c",  # Raspberry Red
    "Redis": "#d82c20",  # Redis Red
}

# --- Wrappers ---


def format_benchmark_header(name, rw_ratio):
    if name in {"Qdrant", "Redis", "Weaviate", "USearch"} or "FAISS" in name:
        name_color = Fore.MAGENTA
    else:
        name_color = Fore.CYAN
    return (
        f"\n{Style.BRIGHT}{Fore.CYAN}Benchmarking Throughput{Style.RESET_ALL} "
        f"{Fore.WHITE}(W/R={rw_ratio}){Style.RESET_ALL}: "
        f"{Style.BRIGHT}{name_color}{name}{Style.RESET_ALL}"
    )


def format_throughput_line(
    num_threads, num_insert_threads, num_search_threads, throughput, prev
):
    if prev is None:
        throughput_color = Fore.CYAN
    elif throughput >= prev * 1.05:
        throughput_color = Fore.GREEN
    elif throughput <= prev * 0.95:
        throughput_color = Fore.RED
    else:
        throughput_color = Fore.YELLOW
    return (
        f"  {Fore.BLUE}Threads:{Style.RESET_ALL} {num_threads} "
        f"{Fore.WHITE}(W={num_insert_threads}, R={num_search_threads}){Style.RESET_ALL} -> "
        f"{Fore.BLUE}Throughput:{Style.RESET_ALL} "
        f"{Style.BRIGHT}{throughput_color}{throughput:.0f}{Style.RESET_ALL} ops/s"
    )


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
        self.client = weaviate.connect_to_embedded(
            environment_variables={
                "LOG_LEVEL": "fatal",  # or "fatal"/"panic" for even less
                "DISABLE_TELEMETRY": "true",  # optional, removes telemetry chatter
            }
        )
        self.collection_name = "NilVecBench"

        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        # Create collection
        self.client.collections.create(
            name=self.collection_name,
            vector_config=wvc.Configure.Vectors.self_provided(
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    ef_construction=ef_construction,
                    max_connections=M,
                )
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


class QdrantIndex:
    def __init__(self, dim, M=16, ef_construction=200):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, HnswConfigDiff, VectorParams
        except ImportError:
            raise ImportError("qdrant-client not installed")

        self._tmp_dir = tempfile.mkdtemp(prefix="nilvec_qdrant_")
        self.client = QdrantClient(path=self._tmp_dir)
        self.collection_name = "nilvec_bench_qdrant"
        self._next_id = 0
        self._id_lock = threading.Lock()
        # Qdrant local backend isn't safe for concurrent search+upsert on one collection.
        # Serialize operations to avoid internal shape/race errors.
        self._op_lock = threading.Lock()

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.EUCLID),
            hnsw_config=HnswConfigDiff(m=M, ef_construct=ef_construction),
        )

    def insert(self, vec):
        from qdrant_client.models import PointStruct

        with self._id_lock:
            point_id = self._next_id
            self._next_id += 1

        with self._op_lock:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=vec)],
                wait=True,
            )

    def search(self, query, k, ef=None):
        from qdrant_client.models import SearchParams

        params = SearchParams(hnsw_ef=ef) if ef is not None else None
        with self._op_lock:
            if hasattr(self.client, "search"):
                res = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query,
                    limit=k,
                    search_params=params,
                    with_payload=False,
                    with_vectors=False,
                )
            else:
                query_res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query,
                    limit=k,
                    search_params=params,
                    with_payload=False,
                    with_vectors=False,
                )
                res = query_res.points

        ids = [hit.id for hit in res]
        distances = [hit.score for hit in res]
        return type("Result", (object,), {"ids": ids, "distances": distances})()

    def train(self, data):
        pass

    def set_nprobe(self, n):
        pass

    def close(self):
        if hasattr(self.client, "close"):
            self.client.close()
        if getattr(self, "_tmp_dir", None) and os.path.isdir(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


class RedisIndex:
    def __init__(self, dim, M=16, ef_construction=200):
        try:
            import redis
            from redis.commands.search.field import VectorField
            from redis.commands.search.index_definition import (
                IndexDefinition,
                IndexType,
            )
        except ImportError:
            raise ImportError("redis not installed")

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.index_name = "idx:nilvec_bench"
        self.prefix = "nilvec:vec:"
        self._next_id = 0
        self._id_lock = threading.Lock()

        try:
            self.client.ft(self.index_name).dropindex(delete_documents=True)
        except Exception:
            pass

        schema = (
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dim,
                    "DISTANCE_METRIC": "L2",
                    "M": M,
                    "EF_CONSTRUCTION": ef_construction,
                },
            ),
        )
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
        self.client.ft(self.index_name).create_index(schema, definition=definition)

    def insert(self, vec):
        with self._id_lock:
            point_id = self._next_id
            self._next_id += 1

        key = f"{self.prefix}{point_id}"
        self.client.hset(
            key,
            mapping={"vector": np.array(vec, dtype=np.float32).tobytes()},
        )

    def search(self, query, k, ef=None):
        from redis.commands.search.query import Query

        q = (
            Query(f"*=>[KNN {k} @vector $vec AS score]")
            .return_fields("score")
            .sort_by("score")
            .dialect(2)
        )
        if ef is not None:
            q = q.ef_runtime(int(ef))

        res = self.client.ft(self.index_name).search(
            q, query_params={"vec": np.array(query, dtype=np.float32).tobytes()}
        )

        ids = []
        distances = []
        for doc in res.docs:
            doc_id = doc.id.decode() if isinstance(doc.id, bytes) else str(doc.id)
            ids.append(doc_id.split(self.prefix, 1)[-1])
            distances.append(float(doc.score))
        return type("Result", (object,), {"ids": ids, "distances": distances})()

    def train(self, data):
        pass

    def set_nprobe(self, n):
        pass

    def close(self):
        if hasattr(self.client, "close"):
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


def _run_command(command):
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _redis_url_is_local(redis_url):
    parsed = urlparse(redis_url)
    if parsed.scheme not in {"redis", "rediss"}:
        return False
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _start_redis_stack_container(redis_url):
    if not _redis_url_is_local(redis_url):
        return False, "auto-start supports only localhost REDIS_URL values"

    parsed = urlparse(redis_url)
    host_port = parsed.port or 6379
    container_name = os.getenv("REDIS_STACK_CONTAINER_NAME", "nilvec-redis-stack")
    image = os.getenv("REDIS_STACK_IMAGE", "redis/redis-stack:latest")

    docker_check = _run_command(["docker", "--version"])
    if docker_check.returncode != 0:
        stderr = docker_check.stderr.strip()
        if stderr:
            return False, f"docker not available ({stderr})"
        return False, "docker not available"

    running = _run_command(
        [
            "docker",
            "ps",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.ID}}",
        ]
    )
    if running.returncode == 0 and running.stdout.strip():
        return True, f"container {container_name} already running"

    existing = _run_command(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.ID}}",
        ]
    )
    if existing.returncode == 0 and existing.stdout.strip():
        started = _run_command(["docker", "start", container_name])
        if started.returncode != 0:
            return False, started.stderr.strip() or f"failed to start {container_name}"
        return True, f"started container {container_name}"

    created = _run_command(
        [
            "docker",
            "run",
            "--name",
            container_name,
            "-p",
            f"{host_port}:6379",
            "-d",
            image,
        ]
    )
    if created.returncode != 0:
        return (
            False,
            created.stderr.strip()
            or f"failed to run docker container {container_name} ({image})",
        )
    return True, f"started new container {container_name} ({image})"


def redis_benchmark_ready(auto_start=False):
    if redis is None:
        return False, "redis client not installed"
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=False)
        client.ping()
        client.close()
        return True, redis_url
    except Exception as e:
        if not auto_start:
            return False, f"{redis_url} ({e})"

        started, start_info = _start_redis_stack_container(redis_url)
        if not started:
            return False, f"{redis_url} ({e}); auto-start failed: {start_info}"

        wait_timeout_s = 20
        start = time.time()
        last_error = str(e)
        while time.time() - start < wait_timeout_s:
            try:
                client = redis.Redis.from_url(redis_url, decode_responses=False)
                client.ping()
                client.close()
                return True, f"{redis_url} ({start_info})"
            except Exception as retry_error:
                last_error = str(retry_error)
                time.sleep(1)

        return (
            False,
            f"{redis_url} (auto-started but not ready after {wait_timeout_s}s: {last_error})",
        )


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
                rw_ratio DOUBLE,
                thread_counts_json VARCHAR,
                only_external BOOLEAN,
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
                is_external BOOLEAN
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
                point_order INTEGER
            )
            """
        )

    def start_run(self, run_meta):
        run_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO benchmark_runs (
                run_id, run_tag, dataset_name, dataset_path, dim, num_vectors,
                num_queries, k, rw_ratio, thread_counts_json, only_external,
                skip_recall, skip_throughput, limit_rows
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                run_meta["rw_ratio"],
                json.dumps(run_meta["thread_counts"]),
                run_meta["only_external"],
                run_meta["skip_recall"],
                run_meta["skip_throughput"],
                run_meta["limit_rows"],
            ],
        )
        return run_id

    def save_throughput(
        self, run_id, throughput_results, conflict_results, external_names, thread_counts
    ):
        rows = []
        for index_name, values in throughput_results.items():
            conflicts = conflict_results.get(index_name, [0] * len(values))
            is_external = index_name in set(external_names)
            for thread_count, throughput, conflict_rate in zip(
                thread_counts, values, conflicts
            ):
                rows.append(
                    (
                        run_id,
                        index_name,
                        int(thread_count),
                        float(throughput),
                        float(conflict_rate),
                        bool(is_external),
                    )
                )
        if rows:
            self.conn.executemany(
                """
                INSERT INTO throughput_points (
                    run_id, index_name, thread_count, throughput, conflict_rate, is_external
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def save_recall_runs(self, run_id, recall_runs):
        rows = []
        for index_name, recalls, qps, line_style in recall_runs:
            for i, (recall, qps_val) in enumerate(zip(recalls, qps)):
                rows.append(
                    (
                        run_id,
                        index_name,
                        f"{index_name}:{i}",
                        float(recall),
                        float(qps_val),
                        line_style,
                        i,
                    )
                )
        if rows:
            self.conn.executemany(
                """
                INSERT INTO recall_points (
                    run_id, index_name, param_key, recall, qps, line_style, point_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _compatible_where(self, meta):
        return """
            dataset_name = ? AND dim = ? AND num_vectors = ? AND num_queries = ?
            AND k = ? AND rw_ratio = ? AND thread_counts_json = ?
        """, [
            meta["dataset_name"],
            meta["dim"],
            meta["num_vectors"],
            meta["num_queries"],
            meta["k"],
            meta["rw_ratio"],
            json.dumps(meta["thread_counts"]),
        ]

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
                throughput_results[index_name] = [float("nan")] * len(meta["thread_counts"])
                conflict_results[index_name] = [float("nan")] * len(meta["thread_counts"])
                injected_indexes.add(index_name)
            i = thread_pos[thread_count]
            if np.isnan(throughput_results[index_name][i]):
                throughput_results[index_name][i] = float(throughput)
            if np.isnan(conflict_results[index_name][i]):
                conflict_results[index_name][i] = float(conflict_rate)
            if is_external:
                external_set.add(index_name)
        return throughput_results, conflict_results, sorted(external_set), sorted(injected_indexes)

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
            grouped.setdefault(index_name, {"recalls": [], "qps": [], "style": line_style})
            grouped[index_name]["recalls"].append(float(recall))
            grouped[index_name]["qps"].append(float(qps))

        injected = []
        for index_name, data in grouped.items():
            injected.append((index_name, tuple(data["recalls"]), tuple(data["qps"]), data["style"]))
        return existing_runs + injected, [name for name, _, _, _ in injected]

    def close(self):
        self.conn.close()


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
    recall_start_time = time.time()

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

    recall_elapsed = time.time() - recall_start_time
    print(
        f"  {Fore.WHITE}Elapsed time for {Style.BRIGHT}{index_name}{Style.RESET_ALL}"
        f"{Fore.WHITE}: {recall_elapsed:.2f}s{Style.RESET_ALL}"
    )
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

    print(format_benchmark_header(index_name, rw_ratio))
    index_start_time = time.time()
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
                if stop_event.is_set():
                    break
                for q in qs:
                    if stop_event.is_set():
                        break
                    index.search(q, k)
                    local_ops += 1
            return local_ops

        def insert_worker(vecs):
            nonlocal ops_count
            local_ops = 0
            for v in vecs:
                if stop_event.is_set():
                    break
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
        prev = results[-1] if results else None
        print(
            format_throughput_line(
                num_threads, num_insert_threads, num_search_threads, throughput, prev
            )
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

    index_elapsed = time.time() - index_start_time
    print(
        f"  {Fore.WHITE}Elapsed time for {Style.BRIGHT}{index_name}{Style.RESET_ALL}"
        f"{Fore.WHITE}: {index_elapsed:.2f}s{Style.RESET_ALL}"
    )
    return results, conflict_rates


# --- Main ---


def _format_elapsed(seconds):
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"


def _run_single_dataset(args, dataset_path):
    """Run benchmark for a single dataset. Returns elapsed time in seconds."""
    dataset_start_time = time.time()

    # Load Data
    # Check if we likely have a valid dataset to load
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    if dataset_path and (os.path.exists(dataset_path) or dataset_name in DATASETS):
        data, queries, gt = load_dataset(dataset_path, args.limit)

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

    # Build per-dataset plot directory: paper/plots/{dataset}_{limit}
    _plot_limit = args.limit if args.limit > 0 else NUM_VECTORS
    plot_dir = os.path.join("paper", "plots", f"{dataset_name}_{_plot_limit}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot output directory: {plot_dir}")

    run_meta = {
        "run_tag": args.run_tag or None,
        "dataset_name": dataset_name,
        "dataset_path": os.path.abspath(dataset_path),
        "dim": DIM,
        "num_vectors": NUM_VECTORS,
        "num_queries": NUM_QUERIES,
        "k": K,
        "rw_ratio": args.rw_ratio,
        "thread_counts": THREAD_COUNTS,
        "only_external": args.only_external,
        "skip_recall": args.skip_recall,
        "skip_throughput": args.skip_throughput,
        "limit_rows": args.limit,
    }

    results_store = None
    run_id = None
    if args.results_db:
        try:
            results_store = BenchmarkResultsStore(args.results_db)
            run_id = results_store.start_run(run_meta)
            print(f"Results DB: {args.results_db} (run_id={run_id[:8]})")
        except Exception as e:
            print(f"{Fore.YELLOW}Results DB disabled: {e}{Style.RESET_ALL}")

    results_cache = {
        "recall_vs_qps": {"runs": [], "K": K, "DIM": DIM},
        "throughput": {},
        "conflicts": {},
        "thread_counts": THREAD_COUNTS,
        "rw_ratio": args.rw_ratio,
        "external_names": [],
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
        results_cache["recall_vs_qps"]["runs"].append(
            ("HNSW Vanilla", recalls, qps, "o-")
        )
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
        results_cache["recall_vs_qps"]["runs"].append(
            ("IVFFlat Vanilla", recalls, qps, "s-")
        )
        plt.plot(recalls, qps, "s-", label="IVFFlat Vanilla")

        if args.cross_pollinate and results_store and run_id:
            merged_runs, injected = results_store.cross_pollinate_recall(
                run_meta, run_id, results_cache["recall_vs_qps"]["runs"]
            )
            results_cache["recall_vs_qps"]["runs"] = merged_runs
            for name, recalls, qps, style in merged_runs:
                if name in {"HNSW Vanilla", "IVFFlat Vanilla"}:
                    continue
                plt.plot(recalls, qps, style, label=f"{name} (history)", alpha=0.6)
            if injected:
                print(
                    f"Cross-pollinated recall from history: {', '.join(sorted(injected))}"
                )

        plt.xlabel("Recall")
        plt.ylabel("QPS (log scale)")
        plt.yscale("log")
        plt.title(f"Recall vs QPS (K={K}, Dim={DIM})")
        plt.legend()
        plt.grid(True)
        recall_path = os.path.join(plot_dir, "recall_vs_qps.svg")
        plt.savefig(recall_path, dpi=DPI)
        print(f"Saved {recall_path}")

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
        # Omitted becasue it takes 30 min (alone) on a small dataset
        # if qdrant_client:
        #     externals.append((QdrantIndex, "Qdrant", hnsw_args))
        redis_ready, redis_info = redis_benchmark_ready(
            auto_start=args.auto_start_redis
        )
        if redis_ready:
            externals.append((RedisIndex, "Redis", hnsw_args))
        elif redis is not None:
            print(
                f"{Fore.YELLOW}Skipping Redis benchmark preflight: cannot connect to {redis_info}.{Style.RESET_ALL}\n"
                f"{Fore.YELLOW}Hint:{Style.RESET_ALL} set REDIS_URL, ensure Docker is running for auto-start, or start Redis Stack manually:\n"
                "  docker run -p 6379:6379 -d redis/redis-stack:latest\n"
                "  (use --no-auto-start-redis to disable Docker auto-start)"
            )

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

        if args.cross_pollinate and results_store and run_id:
            merged_throughput, merged_conflicts, merged_external, injected = (
                results_store.cross_pollinate_throughput(
                    run_meta,
                    run_id,
                    results_cache["throughput"],
                    results_cache["conflicts"],
                    results_cache["external_names"],
                )
            )
            results_cache["throughput"] = merged_throughput
            results_cache["conflicts"] = merged_conflicts
            results_cache["external_names"] = merged_external
            if injected:
                print(
                    f"Cross-pollinated throughput from history: {', '.join(injected)}"
                )

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

        for name, res in results_cache["throughput"].items():
            external_names = results_cache["external_names"]

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
                    if np.isnan(y):
                        continue
                    im = OffsetImage(img, zoom=zoom)
                    ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
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
        throughput_path = os.path.join(plot_dir, "throughput_scaling.svg")
        plt.savefig(throughput_path, dpi=DPI)
        print(f"Saved {throughput_path}")

        # Plot 2: Conflict Rates (Optimistic only)
        plt.figure(figsize=(8, 6))
        has_conflicts = False
        for name, conflicts in results_cache["conflicts"].items():
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
            conflict_path = os.path.join(plot_dir, "conflict_rate.svg")
            plt.savefig(conflict_path, dpi=DPI)
            print(f"Saved {conflict_path}")

    if results_store and run_id:
        if results_cache["throughput"]:
            results_store.save_throughput(
                run_id,
                results_cache["throughput"],
                results_cache["conflicts"],
                results_cache["external_names"],
                THREAD_COUNTS,
            )
        if results_cache["recall_vs_qps"]["runs"]:
            results_store.save_recall_runs(run_id, results_cache["recall_vs_qps"]["runs"])
        results_store.close()

    if args.save_results:
        print(f"Saving results to {args.save_results}...")
        with open(args.save_results, "wb") as f:
            pickle.dump(results_cache, f)

    dataset_elapsed = time.time() - dataset_start_time
    print(
        f"\n{Style.BRIGHT}{Fore.GREEN}Dataset '{dataset_name}' completed in "
        f"{_format_elapsed(dataset_elapsed)}{Style.RESET_ALL}"
    )
    return dataset_elapsed


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
            "--all",
            action="store_true",
            help="Run benchmark on all available datasets (overrides --dataset)",
        )
        parser.add_argument(
            "--rw-ratio",
            type=float,
            default=RW_RATIO,
            help="Read/Write ratio (0.0=Read, 1.0=Write)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Limit number of vectors (0 = no limit)",
        )
        parser.add_argument(
            "--only-external", action="store_true", help="Run only external benchmarks"
        )
        parser.add_argument(
            "--save-results",
            type=str,
            default="",
            help="Legacy pickle export path (optional)",
        )
        parser.add_argument(
            "--results-db",
            type=str,
            default="benchmark_results.duckdb",
            help="DuckDB file for benchmark history",
        )
        parser.add_argument(
            "--cross-pollinate",
            action="store_true",
            default=True,
            help="Merge compatible historical results into current plots (default: enabled)",
        )
        parser.add_argument(
            "--no-cross-pollinate",
            action="store_false",
            dest="cross_pollinate",
            help="Disable merging compatible historical results",
        )
        parser.add_argument(
            "--run-tag",
            type=str,
            default="",
            help="Optional label to tag this benchmark run in results DB",
        )
        parser.add_argument(
            "--auto-start-redis",
            action="store_true",
            default=True,
            help="Attempt to auto-start redis/redis-stack via Docker if REDIS_URL is unreachable (default: enabled)",
        )
        parser.add_argument(
            "--no-auto-start-redis",
            action="store_false",
            dest="auto_start_redis",
            help="Disable Docker auto-start for Redis benchmark preflight",
        )
        args = parser.parse_args()

    suite_start_time = time.time()

    if getattr(args, "all", False):
        dataset_names = list(DATASETS.keys())
        print(
            f"\n{Style.BRIGHT}{Fore.CYAN}=== Full Benchmark Suite ==={Style.RESET_ALL}"
        )
        print(f"Running {len(dataset_names)} datasets: {', '.join(dataset_names)}\n")

        dataset_timings = {}
        for i, ds_name in enumerate(dataset_names, 1):
            ds_path = f"{ds_name}.hdf5"
            print(
                f"\n{Style.BRIGHT}{Fore.CYAN}"
                f"--- [{i}/{len(dataset_names)}] Dataset: {ds_name} ---"
                f"{Style.RESET_ALL}"
            )
            elapsed = _run_single_dataset(args, ds_path)
            dataset_timings[ds_name] = elapsed

        suite_elapsed = time.time() - suite_start_time
        print(
            f"\n{Style.BRIGHT}{Fore.CYAN}=== Full Suite Summary ==={Style.RESET_ALL}"
        )
        for ds_name, elapsed in dataset_timings.items():
            print(f"  {ds_name}: {_format_elapsed(elapsed)}")
        print(
            f"\n{Style.BRIGHT}{Fore.GREEN}Total suite time: "
            f"{_format_elapsed(suite_elapsed)}{Style.RESET_ALL}"
        )
    else:
        _run_single_dataset(args, args.dataset)
        suite_elapsed = time.time() - suite_start_time
        print(
            f"\n{Style.BRIGHT}{Fore.GREEN}Total elapsed time: "
            f"{_format_elapsed(suite_elapsed)}{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    # The per-dataset plot directory is created inside run_benchmark()
    run_benchmark()
