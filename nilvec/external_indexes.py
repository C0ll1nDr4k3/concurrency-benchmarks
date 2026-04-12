import os
import random
import threading

import numpy as np

try:
    import faiss
except ImportError:
    print("\033[33mfaiss not installed - skipping related benchmarks\033[0m")
    faiss = None

try:
    import usearch.index
except ImportError:
    print("\033[33musearch not installed - skipping related benchmarks\033[0m")
    usearch = None

try:
    import pymilvus
except ImportError:
    print("\033[33mMilvus not installed - skipping related benchmarks\033[0m")
    pymilvus = None

try:
    import weaviate
except ImportError:
    print("\033[33mWeaviate not installed - skipping related benchmarks\033[0m")
    weaviate = None

try:
    import redis
except ImportError:
    print("\033[33mRedis client not installed - skipping related benchmarks\033[0m")
    redis = None

try:
    import hnswlib
except ImportError:
    print("\033[33mhnswlib not installed - skipping related benchmarks\033[0m")
    hnswlib = None


class FaissHNSW:
    def __init__(self, dim, M=16, ef_construction=200):
        if faiss is None:
            raise ImportError("faiss not installed")
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.M = M
        self.ef_construction = ef_construction
        self._lock = threading.Lock()

    def insert(self, vec):
        v = np.array([vec], dtype=np.float32)
        with self._lock:
            self.index.add(v)  # type: ignore[call-arg]

    def search(self, query, k, ef=None):
        v = np.array([query], dtype=np.float32)
        with self._lock:
            if ef is not None:
                self.index.hnsw.efSearch = ef
            D, I = self.index.search(v, k)  # type: ignore[call-arg]
        return type(
            "Result", (object,), {"ids": I[0].tolist(), "distances": D[0].tolist()}
        )()

    def set_nprobe(self, n):
        pass  # Not applicable

    def set_num_threads(self, n):
        if faiss is not None:
            faiss.omp_set_num_threads(n)

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
        self._lock = threading.Lock()

    def train(self, data):
        d = np.array(data, dtype=np.float32)
        self.index.train(d)  # type: ignore[call-arg]

    def insert(self, vec):
        v = np.array([vec], dtype=np.float32)
        with self._lock:
            self.index.add(v)  # type: ignore[call-arg]

    def search(self, query, k, ef=None):
        v = np.array([query], dtype=np.float32)
        with self._lock:
            D, I = self.index.search(v, k)  # type: ignore[call-arg]
        return type(
            "Result", (object,), {"ids": I[0].tolist(), "distances": D[0].tolist()}
        )()

    def set_nprobe(self, n):
        with self._lock:
            self.index.nprobe = n

    def set_num_threads(self, n):
        if faiss is not None:
            faiss.omp_set_num_threads(n)


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
                self.index.change_expansion_search(ef)  # type: ignore[attr-defined]
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

        self.client = weaviate.connect_to_embedded(
            environment_variables={
                "LOG_LEVEL": "fatal",
                "DISABLE_TELEMETRY": "true",
            }
        )
        self.collection_name = "NilVecBench"

        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

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
        self.collection.data.insert(properties={"dummy": 1}, vector=vec)

    def search(self, query, k, ef=None):
        res = self.collection.query.near_vector(
            near_vector=query,
            limit=k,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),  # type: ignore[attr-defined]
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
        self.client.ft(self.index_name).create_index([schema[0]], definition=definition)  # type: ignore[arg-type]

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
            q = q.ef_runtime(int(ef))  # type: ignore[attr-defined]

        res = self.client.ft(self.index_name).search(
            q, query_params={"vec": np.array(query, dtype=np.float32).tobytes()}
        )

        ids = []
        distances = []
        for doc in res.docs:  # type: ignore[attr-defined]
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


class HnswLibIndex:
    """hnswlib HNSW — the recall_vs_qps reference implementation (pure Python, no Docker)."""

    def __init__(self, dim, M=16, ef_construction=500):
        if hnswlib is None:
            raise ImportError("hnswlib not installed")
        self.index = hnswlib.Index(space="l2", dim=dim)
        self.index.init_index(
            max_elements=1_200_000, ef_construction=ef_construction, M=M
        )
        self.index.ef = 200
        self._count = 0
        self._lock = threading.Lock()

    def train(self, data):
        pass

    def insert(self, vec):
        v = np.array([vec], dtype=np.float32)
        with self._lock:
            if self._count >= self.index.max_elements:
                self.index.resize_index(self._count * 2)
            self.index.add_items(v, [self._count])
            self._count += 1

    def search(self, query, k, ef=None):
        v = np.array([query], dtype=np.float32)
        with self._lock:
            if ef is not None:
                self.index.ef = ef
            labels, distances = self.index.knn_query(v, k=k)
        return type(
            "Result",
            (object,),
            {"ids": labels[0].tolist(), "distances": distances[0].tolist()},
        )()

    def set_nprobe(self, n):
        pass
