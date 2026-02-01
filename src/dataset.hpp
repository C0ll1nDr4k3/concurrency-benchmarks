/**
 * @file dataset.hpp
 * @brief Dataset loading utilities for ANN benchmarks.
 *
 * Supports loading standard ANN benchmark datasets in HDF5 format
 * from ann-benchmarks.com.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

// HDF5 is optional - if not available, only fvecs format is supported
#ifdef NILVEC_USE_HDF5
#include <H5Cpp.h>
#endif

namespace nilvec {

/**
 * @brief A loaded dataset with train vectors, test queries, and ground truth.
 */
template <typename T>
struct Dataset {
  std::string name;
  Dim dimension;
  std::vector<std::vector<T>> train;  // Vectors to index
  std::vector<std::vector<T>> test;   // Query vectors
  std::vector<std::vector<NodeId>>
      neighbors;  // Ground truth (k nearest neighbors)

  size_t train_size() const { return train.size(); }
  size_t test_size() const { return test.size(); }
  size_t k_ground_truth() const {
    return neighbors.empty() ? 0 : neighbors[0].size();
  }
};

/**
 * @brief Available standard datasets.
 */
struct DatasetInfo {
  std::string name;
  std::string url;
  Dim dimension;
  size_t train_size;
  size_t test_size;
  std::string distance;  // "euclidean" or "angular"
};

inline std::vector<DatasetInfo> available_datasets() {
  return {
      {"sift-128-euclidean",
       "http://ann-benchmarks.com/sift-128-euclidean.hdf5", 128, 1000000, 10000,
       "euclidean"},
      {"fashion-mnist-784-euclidean",
       "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", 784, 60000,
       10000, "euclidean"},
      {"glove-100-angular", "http://ann-benchmarks.com/glove-100-angular.hdf5",
       100, 1183514, 10000, "angular"},
      {"glove-25-angular", "http://ann-benchmarks.com/glove-25-angular.hdf5",
       25, 1183514, 10000, "angular"},
      {"gist-960-euclidean",
       "http://ann-benchmarks.com/gist-960-euclidean.hdf5", 960, 1000000, 1000,
       "euclidean"},
      {"nytimes-256-angular",
       "http://ann-benchmarks.com/nytimes-256-angular.hdf5", 256, 290000, 10000,
       "angular"},
      {"mnist-784-euclidean",
       "http://ann-benchmarks.com/mnist-784-euclidean.hdf5", 784, 60000, 10000,
       "euclidean"},
  };
}

/**
 * @brief Download a file using curl.
 */
inline bool download_file(const std::string& url, const std::string& filepath) {
  std::cout << "Downloading " << url << "..." << std::endl;
  std::string cmd = "curl -L -o \"" + filepath + "\" \"" + url + "\"";
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    std::cerr << "Failed to download: " << url << std::endl;
    return false;
  }
  return true;
}

#ifdef NILVEC_USE_HDF5
/**
 * @brief Load a dataset from HDF5 format (ann-benchmarks format).
 */
template <typename T>
Dataset<T> load_hdf5(const std::string& filepath,
                     size_t max_train = 0,
                     size_t max_test = 0) {
  Dataset<T> dataset;

  try {
    H5::H5File file(filepath, H5F_ACC_RDONLY);

    // Read train data
    {
      H5::DataSet ds = file.openDataSet("train");
      H5::DataSpace space = ds.getSpace();
      hsize_t dims[2];
      space.getSimpleExtentDims(dims);

      size_t n_train = dims[0];
      size_t dim = dims[1];
      dataset.dimension = static_cast<Dim>(dim);

      if (max_train > 0 && max_train < n_train) {
        n_train = max_train;
      }

      std::vector<float> buffer(n_train * dim);

      // Read subset if needed
      hsize_t count[2] = {n_train, dim};
      hsize_t start[2] = {0, 0};
      space.selectHyperslab(H5S_SELECT_SET, count, start);
      H5::DataSpace memspace(2, count);

      ds.read(buffer.data(), H5::PredType::NATIVE_FLOAT, memspace, space);

      dataset.train.resize(n_train);
      for (size_t i = 0; i < n_train; ++i) {
        dataset.train[i].resize(dim);
        for (size_t j = 0; j < dim; ++j) {
          dataset.train[i][j] = static_cast<T>(buffer[i * dim + j]);
        }
      }
    }

    // Read test data
    {
      H5::DataSet ds = file.openDataSet("test");
      H5::DataSpace space = ds.getSpace();
      hsize_t dims[2];
      space.getSimpleExtentDims(dims);

      size_t n_test = dims[0];
      size_t dim = dims[1];

      if (max_test > 0 && max_test < n_test) {
        n_test = max_test;
      }

      std::vector<float> buffer(n_test * dim);

      hsize_t count[2] = {n_test, dim};
      hsize_t start[2] = {0, 0};
      space.selectHyperslab(H5S_SELECT_SET, count, start);
      H5::DataSpace memspace(2, count);

      ds.read(buffer.data(), H5::PredType::NATIVE_FLOAT, memspace, space);

      dataset.test.resize(n_test);
      for (size_t i = 0; i < n_test; ++i) {
        dataset.test[i].resize(dim);
        for (size_t j = 0; j < dim; ++j) {
          dataset.test[i][j] = static_cast<T>(buffer[i * dim + j]);
        }
      }
    }

    // Read ground truth neighbors
    {
      H5::DataSet ds = file.openDataSet("neighbors");
      H5::DataSpace space = ds.getSpace();
      hsize_t dims[2];
      space.getSimpleExtentDims(dims);

      size_t n_queries = dims[0];
      size_t k = dims[1];

      if (max_test > 0 && max_test < n_queries) {
        n_queries = max_test;
      }

      std::vector<int32_t> buffer(n_queries * k);

      hsize_t count[2] = {n_queries, k};
      hsize_t start[2] = {0, 0};
      space.selectHyperslab(H5S_SELECT_SET, count, start);
      H5::DataSpace memspace(2, count);

      ds.read(buffer.data(), H5::PredType::NATIVE_INT32, memspace, space);

      // IMPORTANT: If we limited the training set size, we need to filter
      // the ground truth to only include neighbors that exist in our subset.
      // The HDF5 ground truth is computed on the FULL dataset!
      size_t actual_train_size = dataset.train.size();

      dataset.neighbors.resize(n_queries);
      for (size_t i = 0; i < n_queries; ++i) {
        // Only keep neighbors with IDs that exist in our (possibly truncated)
        // training set
        for (size_t j = 0; j < k; ++j) {
          NodeId neighbor_id = static_cast<NodeId>(buffer[i * k + j]);
          if (neighbor_id < actual_train_size) {
            dataset.neighbors[i].push_back(neighbor_id);
          }
        }
      }
    }

    // Extract name from filepath
    std::filesystem::path p(filepath);
    dataset.name = p.stem().string();

  } catch (H5::Exception& e) {
    throw std::runtime_error("Failed to load HDF5 file: " + filepath);
  }

  return dataset;
}
#endif  // NILVEC_USE_HDF5

/**
 * @brief Load vectors from fvecs format (used by FAISS/Texmex).
 *
 * Format: Each vector is stored as:
 *   - 4 bytes: dimension (int32)
 *   - dim * 4 bytes: float32 values
 */
inline std::vector<std::vector<float>> load_fvecs(const std::string& filepath,
                                                  size_t max_vectors = 0) {
  std::vector<std::vector<float>> vectors;
  std::ifstream file(filepath, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filepath);
  }

  while (file.good()) {
    int32_t dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!file.good())
      break;

    std::vector<float> vec(dim);
    file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
    if (!file.good())
      break;

    vectors.push_back(std::move(vec));

    if (max_vectors > 0 && vectors.size() >= max_vectors) {
      break;
    }
  }

  return vectors;
}

/**
 * @brief Load vectors from ivecs format (integer vectors, typically ground
 * truth).
 */
inline std::vector<std::vector<NodeId>> load_ivecs(const std::string& filepath,
                                                   size_t max_vectors = 0) {
  std::vector<std::vector<NodeId>> vectors;
  std::ifstream file(filepath, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filepath);
  }

  while (file.good()) {
    int32_t dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!file.good())
      break;

    std::vector<int32_t> vec(dim);
    file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int32_t));
    if (!file.good())
      break;

    std::vector<NodeId> node_vec(vec.begin(), vec.end());
    vectors.push_back(std::move(node_vec));

    if (max_vectors > 0 && vectors.size() >= max_vectors) {
      break;
    }
  }

  return vectors;
}

/**
 * @brief Load SIFT dataset from fvecs/ivecs files (Texmex format).
 *
 * Expects files: sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs
 */
template <typename T>
Dataset<T> load_sift_fvecs(const std::string& base_path,
                           size_t max_train = 0,
                           size_t max_test = 0) {
  Dataset<T> dataset;
  dataset.name = "sift-1m";
  dataset.dimension = 128;

  std::string base_file = base_path + "/sift_base.fvecs";
  std::string query_file = base_path + "/sift_query.fvecs";
  std::string gt_file = base_path + "/sift_groundtruth.ivecs";

  std::cout << "Loading SIFT base vectors..." << std::endl;
  auto base_vecs = load_fvecs(base_file, max_train);
  dataset.train.reserve(base_vecs.size());
  for (auto& v : base_vecs) {
    std::vector<T> tv(v.begin(), v.end());
    dataset.train.push_back(std::move(tv));
  }

  std::cout << "Loading SIFT query vectors..." << std::endl;
  auto query_vecs = load_fvecs(query_file, max_test);
  dataset.test.reserve(query_vecs.size());
  for (auto& v : query_vecs) {
    std::vector<T> tv(v.begin(), v.end());
    dataset.test.push_back(std::move(tv));
  }

  std::cout << "Loading SIFT ground truth..." << std::endl;
  dataset.neighbors = load_ivecs(gt_file, max_test);

  return dataset;
}

/**
 * @brief Download and load a standard dataset.
 *
 * @param name Dataset name (e.g., "sift-128-euclidean")
 * @param data_dir Directory to store downloaded files
 * @param max_train Maximum number of training vectors (0 = all)
 * @param max_test Maximum number of test queries (0 = all)
 */
template <typename T>
Dataset<T> load_dataset(const std::string& name,
                        const std::string& data_dir = "data",
                        size_t max_train = 0,
                        size_t max_test = 0) {
  // Create data directory if needed
  std::filesystem::create_directories(data_dir);

  // Find dataset info
  auto datasets = available_datasets();
  auto it = std::find_if(datasets.begin(), datasets.end(),
                         [&](const DatasetInfo& d) { return d.name == name; });

  if (it == datasets.end()) {
    throw std::runtime_error("Unknown dataset: " + name +
                             "\nAvailable datasets: sift-128-euclidean, "
                             "fashion-mnist-784-euclidean, glove-100-angular, "
                             "glove-25-angular, gist-960-euclidean");
  }

  std::string filepath = data_dir + "/" + name + ".hdf5";

  // Download if not exists
  if (!std::filesystem::exists(filepath)) {
    std::cout << "Dataset not found locally, downloading..." << std::endl;
    if (!download_file(it->url, filepath)) {
      throw std::runtime_error("Failed to download dataset: " + name);
    }
  }

#ifdef NILVEC_USE_HDF5
  std::cout << "Loading dataset: " << name << std::endl;
  auto dataset = load_hdf5<T>(filepath, max_train, max_test);
  std::cout << "Loaded " << dataset.train_size() << " train vectors, "
            << dataset.test_size() << " test queries, " << dataset.dimension
            << " dimensions" << std::endl;
  return dataset;
#else
  throw std::runtime_error(
      "HDF5 support not enabled. Rebuild with -DNILVEC_USE_HDF5=ON and link "
      "against libhdf5.\n"
      "On macOS: brew install hdf5\n"
      "On Ubuntu: apt-get install libhdf5-dev");
#endif
}

/**
 * @brief Print information about available datasets.
 */
inline void print_available_datasets() {
  std::cout << "Available datasets:" << std::endl;
  std::cout << "  Name                          Dim    Train       Test  "
               "Distance"
            << std::endl;
  std::cout << "  -----------------------------------------------------------"
               "---------"
            << std::endl;
  for (const auto& d : available_datasets()) {
    std::cout << "  " << std::left << std::setw(30) << d.name << std::setw(6)
              << d.dimension << std::setw(12) << d.train_size << std::setw(6)
              << d.test_size << d.distance << std::endl;
  }
}

}  // namespace nilvec
