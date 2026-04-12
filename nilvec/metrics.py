import hashlib
import os
import pickle
import subprocess
import time
from urllib.parse import urlparse

from colorama import Fore, Style

from nilvec._nilvec import FlatVanilla

try:
    import redis
except ImportError:
    redis = None

_GT_CACHE_DIR = os.path.join("data", "gt_cache")


def compute_recall(results, ground_truth, k):
    """Compute recall@k"""
    recall_sum = 0.0
    for res_ids, true_ids in zip(results, ground_truth):
        truth_set = set(true_ids[:k])
        found = 0
        for rid in res_ids:
            if rid in truth_set:
                found += 1
        recall_sum += found / min(k, len(true_ids))
    return recall_sum / len(results)


def _gt_cache_path(data, queries, k):
    """Derive a deterministic cache filename from the dataset fingerprint."""
    h = hashlib.sha256()
    h.update(f"n={len(data)},q={len(queries)},d={len(data[0])},k={k}".encode())
    # Hash a small sample of the actual data so different datasets with the
    # same shape don't collide.
    for idx in (0, len(data) // 2, len(data) - 1):
        h.update(repr(data[idx][:8]).encode())
    for idx in (0, len(queries) // 2, len(queries) - 1):
        h.update(repr(queries[idx][:8]).encode())
    return os.path.join(_GT_CACHE_DIR, f"gt_{h.hexdigest()[:16]}.pkl")


def get_ground_truth(data, queries, k):
    """Compute brute-force ground truth using FlatVanilla, with disk caching."""
    cache_path = _gt_cache_path(data, queries, k)
    if os.path.exists(cache_path):
        print(
            f"{Fore.GREEN}Loading cached ground truth from {cache_path}{Style.RESET_ALL}"
        )
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"{Fore.YELLOW}Computing ground truth...{Style.RESET_ALL}")
    index = FlatVanilla(len(data[0]))
    for vec in data:
        index.insert(vec)

    gt = []
    for q in queries:
        res = index.search(q, k)
        gt.append(res.ids)

    os.makedirs(_GT_CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(gt, f)
    print(f"{Fore.GREEN}Cached ground truth to {cache_path}{Style.RESET_ALL}")

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
