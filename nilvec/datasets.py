import os
import random
import sys

import h5py
import requests
from colorama import Fore, Style

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
    print(f"{Fore.YELLOW}Downloading {url} to {path}...{Style.RESET_ALL}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{Fore.GREEN}Download complete.{Style.RESET_ALL}")
    else:
        print(
            f"{Fore.RED}Failed to download dataset: {response.status_code}{Style.RESET_ALL}"
        )
        sys.exit(1)


def load_dataset(path, limit=0, k=10):
    data_dir = "data"

    if not os.path.exists(path):
        candidate_path = os.path.join(data_dir, os.path.basename(path))
        if os.path.exists(candidate_path):
            path = candidate_path
        elif os.sep not in path:
            path = candidate_path

    filename = os.path.basename(path)
    dataset_name = os.path.splitext(filename)[0]

    if not os.path.exists(path):
        if dataset_name in DATASETS:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            url = DATASETS[dataset_name]["url"]
            download_dataset(url, path)
        else:
            print(f"Dataset {path} not found.")
            if dataset_name not in DATASETS:
                print(
                    f"Unknown dataset name '{dataset_name}'. Available: {list(DATASETS.keys())}"
                )
                if "sift" in path:
                    url = DATASETS["sift-128-euclidean"]["url"]
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                    download_dataset(url, path)
                else:
                    sys.exit(1)

    print(f"{Fore.CYAN}Loading dataset from {path}...{Style.RESET_ALL}")
    f = h5py.File(path, "r")

    def to_list(dset, limit=0):
        if limit > 0:
            return [list(map(float, vec)) for vec in dset[:limit]]
        return [list(map(float, vec)) for vec in dset]

    train = to_list(f["train"], limit)
    test = to_list(f["test"], limit)

    if "neighbors" in f:
        if limit > 0:
            gt = None  # Recompute
        else:
            gt = [list(map(int, vec[:k])) for vec in f["neighbors"]]  # type: ignore[misc]
    else:
        gt = None

    f.close()
    return train, test, gt


def generate_data(n, dim):
    return [[random.random() for _ in range(dim)] for _ in range(n)]
