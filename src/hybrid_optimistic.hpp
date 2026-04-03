/**
 * @file hybrid_optimistic.hpp
 * @brief Hybrid HNSW-IVF index with optimistic concurrency control.
 *
 * Same structural design as HybridPessimistic: upper layers (>= 1) form a
 * standard HNSW graph; layer-0 edges use IVF-style partitioning where each
 * layer-2+ node defines a partition. The difference is the conflict resolution
 * strategy: instead of acquiring shared_mutex locks, operations snapshot
 * version numbers before reading, then validate after. Writers atomically
 * increment versions; readers that observe a version change retry.
 *
 * Upper layers use per-node version numbers (one PartitionState per node).
 * Layer-0 uses per-partition version numbers. On retry exhaustion, both fall
 * back to pessimistic locking with per-node or per-partition mutexes.
 *
 * Lock order (pessimistic fallback):
 *   global -> node[low_id..high_id] -> partition_registry ->
 *   partition[low_idx..high_idx]
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <deque>
#include <memory>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common.hpp"

namespace nilvec {

template <typename T, int32_t D = DynamicDim>
class HybridOptimistic {
  using Traits = DimTraits<T, D>;
  static constexpr size_t NO_PARTITION = std::numeric_limits<size_t>::max();

  struct PartitionState {
    mutable std::shared_mutex mutex;
    std::atomic<uint64_t> version{0};
  };

 public:
  HybridOptimistic(Dim dim,
                   size_t M = 16,
                   size_t ef_construction = 200,
                   float mL = 0.0f,
                   size_t nprobe = 1,
                   size_t max_retries = 10)
      : dim_(dim),
        M_(M),
        M_max0_(2 * M),
        ef_construction_(ef_construction),
        mL_(mL > 0 ? mL : 1.0f / std::log(static_cast<float>(M))),
        nprobe_(nprobe),
        max_retries_(max_retries),
        entry_point_(INVALID_NODE),
        max_level_(-1),
        rng_(std::random_device{}()) {
    if constexpr (D > 0)
      assert(dim == static_cast<Dim>(D));
  }

  NodeId insert(std::span<const T> data) {
    assert(data.size() == dim_);

    NodeId new_id;
    int new_level;
    {
      std::unique_lock global_lock(global_mutex_);
      new_id = static_cast<NodeId>(vectors_.size());
      new_level = random_level(mL_, rng_);

      vectors_.emplace_back(data.begin(), data.end());
      neighbors_.emplace_back();
      neighbors_[new_id].resize(new_level + 1);
      node_levels_.push_back(new_level);
      partition_of_.push_back(NO_PARTITION);
      deleted_.push_back(false);
      node_upper_states_.push_back(std::make_unique<PartitionState>());
    }

    // First insertion
    NodeId expected_entry = INVALID_NODE;
    if (entry_point_.compare_exchange_strong(expected_entry, new_id,
                                             std::memory_order_acq_rel)) {
      max_level_.store(new_level, std::memory_order_release);

      if (new_level >= 2) {
        std::unique_lock reg_lock(partition_registry_mutex_);
        size_t pidx = register_partition(new_id);
        partition_of_[new_id] = pidx;
      } else {
        std::unique_lock reg_lock(partition_registry_mutex_);
        unassigned_.push_back(new_id);
      }

      return new_id;
    }

    std::shared_lock global_read_lock(global_mutex_);

    NodeId curr_entry = entry_point_.load(std::memory_order_acquire);
    int curr_max_level = max_level_.load(std::memory_order_acquire);

    NodeId nearest_partition_node = INVALID_NODE;
    float nearest_partition_dist = std::numeric_limits<float>::max();

    // Phase 1: Greedy descent through upper layers above new_level
    for (int level = curr_max_level; level > new_level; --level) {
      curr_entry = optimistic_greedy_search(data, curr_entry, level);

      if (level >= 2 && node_levels_[curr_entry] >= 2) {
        float d = compute_distance(data, curr_entry);
        if (d < nearest_partition_dist) {
          nearest_partition_dist = d;
          nearest_partition_node = curr_entry;
        }
      }
    }

    // Phase 2: Search and connect at upper layers [min(new_level,
    // max_level)..1]
    for (int level = std::min(new_level, curr_max_level); level >= 1; --level) {
      NILVEC_TRACK_INSERT_ATTEMPT(conflict_stats_);

      bool success = false;
      for (size_t retry = 0; retry <= max_retries_ && !success; ++retry) {
        auto result = try_insert_upper_layer(data, new_id, curr_entry, level);
        if (result.has_value()) {
          // Track nearest partition node from candidates
          for (const auto& c : result->second) {
            if (node_levels_[c.id] >= 2 &&
                c.distance < nearest_partition_dist) {
              nearest_partition_dist = c.distance;
              nearest_partition_node = c.id;
            }
          }
          curr_entry = result->first;
          success = true;
        } else {
          NILVEC_TRACK_INSERT_CONFLICT(conflict_stats_);
        }
      }

      if (!success) {
        auto result =
            pessimistic_insert_upper_layer(data, new_id, curr_entry, level);
        for (const auto& c : result.second) {
          if (node_levels_[c.id] >= 2 && c.distance < nearest_partition_dist) {
            nearest_partition_dist = c.distance;
            nearest_partition_node = c.id;
          }
        }
        curr_entry = result.first;
      }
    }

    // Phase 3: Determine partition assignment
    size_t new_partition = NO_PARTITION;
    if (new_level >= 2) {
      std::unique_lock reg_lock(partition_registry_mutex_);
      new_partition = register_partition(new_id);
      partition_of_[new_id] = new_partition;
      drain_unassigned();
    } else {
      std::unique_lock reg_lock(partition_registry_mutex_);
      if (nearest_partition_node != INVALID_NODE &&
          partition_index_.count(nearest_partition_node)) {
        new_partition = partition_index_[nearest_partition_node];
        partition_of_[new_id] = new_partition;
      } else {
        unassigned_.push_back(new_id);
      }
    }

    // Phase 4: Build layer-0 HNSW edges with optimistic partition locking
    if (curr_max_level >= 0) {
      NILVEC_TRACK_INSERT_ATTEMPT(conflict_stats_);

      bool success = false;
      for (size_t retry = 0; retry <= max_retries_ && !success; ++retry) {
        if (try_insert_layer_0(data, new_id, curr_entry, new_partition)) {
          success = true;
        } else {
          NILVEC_TRACK_INSERT_CONFLICT(conflict_stats_);
        }
      }

      if (!success) {
        pessimistic_insert_layer_0(data, new_id, curr_entry, new_partition);
      }
    }

    // Update entry point and max level if necessary (must happen before
    // releasing the global read lock so that a concurrent search that sees
    // the new entry_point also sees the node's fully-connected edges).
    int expected_level = curr_max_level;
    while (new_level > expected_level) {
      if (max_level_.compare_exchange_weak(expected_level, new_level,
                                           std::memory_order_acq_rel)) {
        entry_point_.store(new_id, std::memory_order_release);
        break;
      }
    }

    global_read_lock.unlock();

    return new_id;
  }

  void remove(NodeId id) {
    std::shared_lock global_read_lock(global_mutex_);

    if (id >= vectors_.size() || deleted_[id])
      return;

    deleted_[id] = true;
    int level = node_levels_[id];

    // Remove from upper-layer graph edges (layers >= 1)
    for (int l = level; l >= 1; --l) {
      // Phase 1: snapshot id's neighbors under a shared lock.
      std::vector<NodeId> id_neighbors;
      {
        std::shared_lock slk(node_upper_states_[id]->mutex);
        id_neighbors = neighbors_[id][l];
      }

      // Phase 2: exclusive locks on id + snapshot neighbors in sorted order.
      std::set<NodeId> write_set;
      write_set.insert(id);
      for (NodeId n : id_neighbors)
        write_set.insert(n);
      std::vector<std::unique_lock<std::shared_mutex>> node_locks;
      node_locks.reserve(write_set.size());
      for (NodeId n : write_set)
        node_locks.emplace_back(node_upper_states_[n]->mutex);

      for (NodeId neighbor : neighbors_[id][l]) {
        if (!write_set.count(neighbor))
          continue;
        auto& nlist = neighbors_[neighbor][l];
        nlist.erase(std::remove(nlist.begin(), nlist.end(), id), nlist.end());
      }
      neighbors_[id][l].clear();

      for (NodeId n : write_set)
        node_upper_states_[n]->version.fetch_add(1, std::memory_order_release);
    }

    // Remove layer-0 edges under partition locks
    {
      std::set<size_t> affected;
      size_t own = partition_of_[id];
      if (own != NO_PARTITION)
        affected.insert(own);
      for (NodeId neighbor : neighbors_[id][0]) {
        size_t np = partition_of_[neighbor];
        if (np != NO_PARTITION)
          affected.insert(np);
      }

      std::vector<std::unique_lock<std::shared_mutex>> locks;
      locks.reserve(affected.size());
      for (size_t pidx : affected) {
        locks.emplace_back(partition_states_[pidx]->mutex);
      }

      for (NodeId neighbor : neighbors_[id][0]) {
        auto& nlist = neighbors_[neighbor][0];
        nlist.erase(std::remove(nlist.begin(), nlist.end(), id), nlist.end());
      }
      neighbors_[id][0].clear();

      for (size_t pidx : affected) {
        partition_states_[pidx]->version.fetch_add(1,
                                                   std::memory_order_release);
      }
    }

    global_read_lock.unlock();

    // Reassign children if this was a partition center
    if (level >= 2) {
      std::unique_lock reg_lock(partition_registry_mutex_);
      auto it = partition_index_.find(id);
      if (it == partition_index_.end())
        return;

      size_t pidx = it->second;

      std::vector<NodeId> orphans;
      for (size_t i = 0; i < partition_of_.size(); ++i) {
        if (partition_of_[i] == pidx && i != id && !deleted_[i]) {
          orphans.push_back(static_cast<NodeId>(i));
        }
      }

      partition_index_.erase(it);

      for (NodeId orphan : orphans) {
        size_t best_pidx = NO_PARTITION;
        float best_dist = std::numeric_limits<float>::max();

        for (const auto& [center_id, pidx2] : partition_index_) {
          if (deleted_[center_id])
            continue;
          float d = squared_distance(Traits::make_span(vectors_[orphan]),
                                     Traits::make_span(vectors_[center_id]));
          if (d < best_dist) {
            best_dist = d;
            best_pidx = pidx2;
          }
        }

        if (best_pidx != NO_PARTITION) {
          partition_of_[orphan] = best_pidx;
        } else {
          partition_of_[orphan] = NO_PARTITION;
          unassigned_.push_back(orphan);
        }
      }
    }
  }

  SearchResult search(std::span<const T> query, size_t k, size_t ef = 0) const {
    if (ef == 0)
      ef = k;

    NodeId curr_entry = entry_point_.load(std::memory_order_acquire);
    if (curr_entry == INVALID_NODE) {
      return SearchResult{};
    }

    std::shared_lock global_read_lock(global_mutex_);

    int curr_max_level = max_level_.load(std::memory_order_acquire);

    // Phase 1: Greedy descent through upper layers to layer 2
    for (int level = curr_max_level; level >= 3; --level) {
      curr_entry = optimistic_greedy_search(query, curr_entry, level);
    }

    // Phase 2: Beam search at layer 2 to find nprobe nearest partition centers
    std::vector<NodeId> probe_centers;

    if (curr_max_level >= 2) {
      size_t np = nprobe_.load(std::memory_order_relaxed);
      size_t search_ef = std::max(np, ef);

      auto candidates = search_layer(query, curr_entry, search_ef, 2);

      std::shared_lock reg_lock(partition_registry_mutex_);
      for (const auto& c : candidates) {
        if (probe_centers.size() >= np)
          break;
        if (deleted_[c.id])
          continue;
        if (partition_index_.count(c.id)) {
          probe_centers.push_back(c.id);
        }
      }
    } else {
      std::shared_lock reg_lock(partition_registry_mutex_);
      for (const auto& [center_id, pidx] : partition_index_) {
        if (!deleted_[center_id])
          probe_centers.push_back(center_id);
      }
    }

    // Phase 3: Walk layer-0 edges with optimistic partition version checks
    MaxHeap results;
    std::unordered_set<NodeId> visited;

    for (NodeId center : probe_centers) {
      walk_layer_0_optimistic(query, center, ef, results, visited, k);
    }

    // Build result
    SearchResult result;
    std::vector<Candidate> sorted;
    sorted.reserve(results.size());
    while (!results.empty()) {
      sorted.push_back(results.top());
      results.pop();
    }
    std::sort(sorted.begin(), sorted.end());

    size_t count = std::min(k, sorted.size());
    result.ids.reserve(count);
    result.distances.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      result.ids.push_back(sorted[i].id);
      result.distances.push_back(sorted[i].distance);
    }

    return result;
  }

  size_t size() const {
    std::shared_lock lock(global_mutex_);
    size_t n = vectors_.size();
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) {
      if (deleted_[i])
        ++d;
    }
    return n - d;
  }

  int max_level() const { return max_level_.load(std::memory_order_acquire); }

  void set_nprobe(size_t nprobe) {
    nprobe_.store(nprobe, std::memory_order_release);
  }

  size_t num_partitions() const {
    std::shared_lock lock(partition_registry_mutex_);
    return partition_index_.size();
  }

  const ConflictStats& conflict_stats() const { return conflict_stats_; }

  void reset_conflict_stats() { conflict_stats_.reset(); }

 private:
  // --- Upper-layer optimistic insert ---

  // Returns (next_entry, candidates) on success, nullopt on conflict.
  // Searches under per-node shared locks, snapshots write-set versions, then
  // acquires exclusive locks and validates before committing.
  std::optional<std::pair<NodeId, std::vector<Candidate>>>
  try_insert_upper_layer(std::span<const T> data,
                         NodeId new_id,
                         NodeId entry_point,
                         int level) {
    auto candidates = search_layer(data, entry_point, ef_construction_, level);
    auto neighbors = select_neighbors(candidates, M_);

    // Collect write nodes: new_id + selected neighbors.
    std::set<NodeId> write_set;
    write_set.insert(new_id);
    for (NodeId n : neighbors)
      write_set.insert(n);

    // Snapshot per-node versions before acquiring locks.
    std::vector<std::pair<NodeId, uint64_t>> version_snapshots;
    version_snapshots.reserve(write_set.size());
    for (NodeId n : write_set) {
      uint64_t v =
          node_upper_states_[n]->version.load(std::memory_order_acquire);
      version_snapshots.emplace_back(n, v);
    }

    // Acquire exclusive locks in sorted NodeId order (deadlock-free).
    std::vector<std::unique_lock<std::shared_mutex>> locks;
    locks.reserve(write_set.size());
    for (NodeId n : write_set)
      locks.emplace_back(node_upper_states_[n]->mutex);

    // Validate: if any node was written to since our search, the candidate
    // selection may be suboptimal -- retry.
    for (const auto& [n, v] : version_snapshots) {
      if (node_upper_states_[n]->version.load(std::memory_order_relaxed) != v)
        return std::nullopt;
    }

    neighbors_[new_id][level] = neighbors;

    for (NodeId neighbor : neighbors) {
      auto& neighbor_list = neighbors_[neighbor][level];
      neighbor_list.push_back(new_id);
      if (neighbor_list.size() > M_) {
        prune_neighbors(neighbor, level, M_);
      }
    }

    for (NodeId n : write_set)
      node_upper_states_[n]->version.fetch_add(1, std::memory_order_release);

    NodeId next = candidates.empty() ? entry_point : candidates.front().id;
    return std::make_pair(next, std::move(candidates));
  }

  std::pair<NodeId, std::vector<Candidate>> pessimistic_insert_upper_layer(
      std::span<const T> data,
      NodeId new_id,
      NodeId entry_point,
      int level) {
    auto candidates = search_layer(data, entry_point, ef_construction_, level);
    auto neighbors = select_neighbors(candidates, M_);

    std::set<NodeId> write_set;
    write_set.insert(new_id);
    for (NodeId n : neighbors)
      write_set.insert(n);

    std::vector<std::unique_lock<std::shared_mutex>> locks;
    locks.reserve(write_set.size());
    for (NodeId n : write_set)
      locks.emplace_back(node_upper_states_[n]->mutex);

    neighbors_[new_id][level] = neighbors;

    for (NodeId neighbor : neighbors) {
      auto& neighbor_list = neighbors_[neighbor][level];
      neighbor_list.push_back(new_id);
      if (neighbor_list.size() > M_) {
        prune_neighbors(neighbor, level, M_);
      }
    }

    for (NodeId n : write_set)
      node_upper_states_[n]->version.fetch_add(1, std::memory_order_release);

    NodeId next = candidates.empty() ? entry_point : candidates.front().id;
    return {next, std::move(candidates)};
  }

  // --- Layer-0 optimistic insert ---

  bool try_insert_layer_0(std::span<const T> data,
                          NodeId new_id,
                          NodeId entry_point,
                          size_t new_partition) {
    // Snapshot partition versions before beam search
    auto candidates =
        search_layer_0_optimistic(data, entry_point, ef_construction_);
    if (!candidates.has_value())
      return false;

    auto neighbors = select_neighbors(*candidates, M_max0_);

    // Collect all partitions that need writes
    std::set<size_t> write_partitions;
    if (new_partition != NO_PARTITION)
      write_partitions.insert(new_partition);
    for (NodeId neighbor : neighbors) {
      size_t np = partition_of_[neighbor];
      if (np != NO_PARTITION)
        write_partitions.insert(np);
    }

    // Snapshot versions of all write partitions
    std::vector<std::pair<size_t, uint64_t>> version_snapshots;
    version_snapshots.reserve(write_partitions.size());
    for (size_t pidx : write_partitions) {
      uint64_t v =
          partition_states_[pidx]->version.load(std::memory_order_acquire);
      version_snapshots.emplace_back(pidx, v);
    }

    // Acquire exclusive locks in index order
    std::vector<std::unique_lock<std::shared_mutex>> write_locks;
    write_locks.reserve(write_partitions.size());
    for (size_t pidx : write_partitions) {
      write_locks.emplace_back(partition_states_[pidx]->mutex);
    }

    // Validate versions
    for (const auto& [pidx, v] : version_snapshots) {
      if (partition_states_[pidx]->version.load(std::memory_order_relaxed) !=
          v) {
        return false;
      }
    }

    // Commit
    neighbors_[new_id][0] = neighbors;

    for (NodeId neighbor : neighbors) {
      auto& neighbor_list = neighbors_[neighbor][0];
      neighbor_list.push_back(new_id);
      if (neighbor_list.size() > M_max0_) {
        prune_neighbors(neighbor, 0, M_max0_);
      }
    }

    // Increment versions of all modified partitions
    for (size_t pidx : write_partitions) {
      partition_states_[pidx]->version.fetch_add(1, std::memory_order_release);
    }

    return true;
  }

  void pessimistic_insert_layer_0(std::span<const T> data,
                                  NodeId new_id,
                                  NodeId entry_point,
                                  size_t new_partition) {
    auto candidates =
        search_layer_0_pessimistic(data, entry_point, ef_construction_);
    auto neighbors = select_neighbors(candidates, M_max0_);

    std::set<size_t> write_partitions;
    if (new_partition != NO_PARTITION)
      write_partitions.insert(new_partition);
    for (NodeId neighbor : neighbors) {
      size_t np = partition_of_[neighbor];
      if (np != NO_PARTITION)
        write_partitions.insert(np);
    }

    std::vector<std::unique_lock<std::shared_mutex>> write_locks;
    write_locks.reserve(write_partitions.size());
    for (size_t pidx : write_partitions) {
      write_locks.emplace_back(partition_states_[pidx]->mutex);
    }

    neighbors_[new_id][0] = neighbors;

    for (NodeId neighbor : neighbors) {
      auto& neighbor_list = neighbors_[neighbor][0];
      neighbor_list.push_back(new_id);
      if (neighbor_list.size() > M_max0_) {
        prune_neighbors(neighbor, 0, M_max0_);
      }
    }

    for (size_t pidx : write_partitions) {
      partition_states_[pidx]->version.fetch_add(1, std::memory_order_release);
    }
  }

  // --- Layer-0 search helpers ---

  // Optimistic beam search at layer 0: reads edge lists under shared partition
  // locks per node, validating partition versions. Returns nullopt if any
  // partition version changed during the search.
  std::optional<std::vector<Candidate>> search_layer_0_optimistic(
      std::span<const T> query,
      NodeId entry_point,
      size_t ef) const {
    std::unordered_set<NodeId> visited;
    MinHeap candidates;
    MaxHeap results;

    // Track which partition versions we observed
    std::unordered_map<size_t, uint64_t> observed_versions;

    float entry_dist = compute_distance(query, entry_point);
    candidates.push({entry_point, entry_dist});
    results.push({entry_point, entry_dist});
    visited.insert(entry_point);

    while (!candidates.empty()) {
      auto [curr_id, curr_dist] = candidates.top();
      candidates.pop();

      if (curr_dist > results.top().distance)
        break;

      std::vector<NodeId> curr_neighbors;
      {
        size_t pidx = partition_of_[curr_id];
        if (pidx != NO_PARTITION) {
          // Snapshot version before read
          uint64_t v =
              partition_states_[pidx]->version.load(std::memory_order_acquire);
          auto it = observed_versions.find(pidx);
          if (it == observed_versions.end()) {
            observed_versions[pidx] = v;
          } else if (it->second != v) {
            return std::nullopt;  // Version changed since first observation
          }

          std::shared_lock plk(partition_states_[pidx]->mutex);
          curr_neighbors = neighbors_[curr_id][0];
        } else {
          curr_neighbors = neighbors_[curr_id][0];
        }
      }

      for (NodeId neighbor : curr_neighbors) {
        if (visited.count(neighbor) > 0)
          continue;
        visited.insert(neighbor);

        if (deleted_[neighbor])
          continue;

        float neighbor_dist = compute_distance(query, neighbor);

        if (results.size() < ef || neighbor_dist < results.top().distance) {
          candidates.push({neighbor, neighbor_dist});
          results.push({neighbor, neighbor_dist});

          if (results.size() > ef)
            results.pop();
        }
      }
    }

    std::vector<Candidate> result;
    result.reserve(results.size());
    while (!results.empty()) {
      result.push_back(results.top());
      results.pop();
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  std::vector<Candidate> search_layer_0_pessimistic(std::span<const T> query,
                                                    NodeId entry_point,
                                                    size_t ef) const {
    std::unordered_set<NodeId> visited;
    MinHeap candidates;
    MaxHeap results;

    float entry_dist = compute_distance(query, entry_point);
    candidates.push({entry_point, entry_dist});
    results.push({entry_point, entry_dist});
    visited.insert(entry_point);

    while (!candidates.empty()) {
      auto [curr_id, curr_dist] = candidates.top();
      candidates.pop();

      if (curr_dist > results.top().distance)
        break;

      std::vector<NodeId> curr_neighbors;
      {
        size_t pidx = partition_of_[curr_id];
        if (pidx != NO_PARTITION) {
          std::shared_lock plk(partition_states_[pidx]->mutex);
          curr_neighbors = neighbors_[curr_id][0];
        } else {
          curr_neighbors = neighbors_[curr_id][0];
        }
      }

      for (NodeId neighbor : curr_neighbors) {
        if (visited.count(neighbor) > 0)
          continue;
        visited.insert(neighbor);

        if (deleted_[neighbor])
          continue;

        float neighbor_dist = compute_distance(query, neighbor);

        if (results.size() < ef || neighbor_dist < results.top().distance) {
          candidates.push({neighbor, neighbor_dist});
          results.push({neighbor, neighbor_dist});

          if (results.size() > ef)
            results.pop();
        }
      }
    }

    std::vector<Candidate> result;
    result.reserve(results.size());
    while (!results.empty()) {
      result.push_back(results.top());
      results.pop();
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  // Walk layer-0 edges optimistically from a starting node. Per-node reads
  // are validated against partition versions; on conflict the walk falls back
  // to pessimistic shared locks for the remainder.
  void walk_layer_0_optimistic(std::span<const T> query,
                               NodeId start,
                               size_t ef,
                               MaxHeap& results,
                               std::unordered_set<NodeId>& visited,
                               size_t k) const {
    MinHeap candidates;
    MaxHeap local_best;

    // Track observed partition versions for validation
    std::unordered_map<size_t, uint64_t> observed_versions;
    bool use_pessimistic = false;

    if (visited.count(start) == 0) {
      float d = compute_distance(query, start);
      candidates.push({start, d});
      if (!deleted_[start])
        local_best.push({start, d});
      visited.insert(start);
    }

    while (!candidates.empty()) {
      auto [curr_id, curr_dist] = candidates.top();
      candidates.pop();

      if (!local_best.empty() && curr_dist > local_best.top().distance)
        break;

      std::vector<NodeId> curr_neighbors;
      {
        size_t pidx = partition_of_[curr_id];
        if (pidx != NO_PARTITION) {
          if (!use_pessimistic) {
            uint64_t v = partition_states_[pidx]->version.load(
                std::memory_order_acquire);
            auto it = observed_versions.find(pidx);
            if (it == observed_versions.end()) {
              observed_versions[pidx] = v;
            } else if (it->second != v) {
              // Conflict detected; switch to pessimistic for the rest
              NILVEC_TRACK_SEARCH_CONFLICT(conflict_stats_);
              use_pessimistic = true;
            }
          }

          if (use_pessimistic) {
            std::shared_lock plk(partition_states_[pidx]->mutex);
            curr_neighbors = neighbors_[curr_id][0];
          } else {
            // Optimistic read: no lock, just read under version protection
            std::shared_lock plk(partition_states_[pidx]->mutex);
            curr_neighbors = neighbors_[curr_id][0];
          }
        } else {
          curr_neighbors = neighbors_[curr_id][0];
        }
      }

      for (NodeId neighbor : curr_neighbors) {
        if (visited.count(neighbor) > 0)
          continue;
        visited.insert(neighbor);

        if (deleted_[neighbor])
          continue;

        float neighbor_dist = compute_distance(query, neighbor);

        if (local_best.size() < ef ||
            neighbor_dist < local_best.top().distance) {
          candidates.push({neighbor, neighbor_dist});
          local_best.push({neighbor, neighbor_dist});
          if (local_best.size() > ef)
            local_best.pop();
        }
      }
    }

    // Merge local results into shared heap
    while (!local_best.empty()) {
      auto c = local_best.top();
      local_best.pop();
      if (results.size() < k || c.distance < results.top().distance) {
        results.push(c);
        if (results.size() > k)
          results.pop();
      }
    }
  }

  // --- Upper-layer helpers ---

  NodeId optimistic_greedy_search(std::span<const T> query,
                                  NodeId entry_point,
                                  int level) const {
    NodeId current = entry_point;
    float current_dist = compute_distance(query, current);

    bool improved = true;
    while (improved) {
      improved = false;
      std::vector<NodeId> curr_neighbors;
      {
        std::shared_lock nlk(node_upper_states_[current]->mutex);
        curr_neighbors = neighbors_[current][level];
      }
      for (NodeId neighbor : curr_neighbors) {
        if (deleted_[neighbor])
          continue;
        float neighbor_dist = compute_distance(query, neighbor);
        if (neighbor_dist < current_dist) {
          current = neighbor;
          current_dist = neighbor_dist;
          improved = true;
        }
      }
    }

    return current;
  }

  std::vector<Candidate> search_layer(std::span<const T> query,
                                      NodeId entry_point,
                                      size_t ef,
                                      int level) const {
    std::unordered_set<NodeId> visited;
    MinHeap candidates;
    MaxHeap results;

    float entry_dist = compute_distance(query, entry_point);
    candidates.push({entry_point, entry_dist});
    results.push({entry_point, entry_dist});
    visited.insert(entry_point);

    while (!candidates.empty()) {
      auto [curr_id, curr_dist] = candidates.top();
      candidates.pop();

      if (curr_dist > results.top().distance)
        break;

      std::vector<NodeId> curr_neighbors;
      {
        std::shared_lock nlk(node_upper_states_[curr_id]->mutex);
        curr_neighbors = neighbors_[curr_id][level];
      }
      for (NodeId neighbor : curr_neighbors) {
        if (visited.count(neighbor) > 0)
          continue;
        visited.insert(neighbor);

        if (deleted_[neighbor])
          continue;

        float neighbor_dist = compute_distance(query, neighbor);

        if (results.size() < ef || neighbor_dist < results.top().distance) {
          candidates.push({neighbor, neighbor_dist});
          results.push({neighbor, neighbor_dist});

          if (results.size() > ef)
            results.pop();
        }
      }
    }

    std::vector<Candidate> result;
    result.reserve(results.size());
    while (!results.empty()) {
      result.push_back(results.top());
      results.pop();
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  // --- Partition management ---

  size_t register_partition(NodeId node) {
    size_t pidx = partition_states_.size();
    partition_index_[node] = pidx;
    partition_states_.push_back(std::make_unique<PartitionState>());
    return pidx;
  }

  void drain_unassigned() {
    if (unassigned_.empty() || partition_index_.empty())
      return;

    for (NodeId nid : unassigned_) {
      size_t best_pidx = NO_PARTITION;
      float best_dist = std::numeric_limits<float>::max();

      for (const auto& [center_id, pidx] : partition_index_) {
        float d = squared_distance(Traits::make_span(vectors_[nid]),
                                   Traits::make_span(vectors_[center_id]));
        if (d < best_dist) {
          best_dist = d;
          best_pidx = pidx;
        }
      }

      if (best_pidx != NO_PARTITION) {
        partition_of_[nid] = best_pidx;
      }
    }
    unassigned_.clear();
  }

  // --- Shared helpers ---

  std::vector<NodeId> select_neighbors(const std::vector<Candidate>& candidates,
                                       size_t M) const {
    std::vector<NodeId> result;
    result.reserve(M);
    for (size_t i = 0; i < std::min(M, candidates.size()); ++i) {
      result.push_back(candidates[i].id);
    }
    return result;
  }

  void prune_neighbors(NodeId node, int level, size_t max_size) {
    auto& neighbor_list = neighbors_[node][level];
    if (neighbor_list.size() <= max_size)
      return;

    auto node_vec = Traits::make_span(vectors_[node]);
    std::vector<Candidate> candidates;
    candidates.reserve(neighbor_list.size());

    for (NodeId neighbor : neighbor_list) {
      float dist =
          squared_distance(node_vec, Traits::make_span(vectors_[neighbor]));
      candidates.push_back({neighbor, dist});
    }

    std::sort(candidates.begin(), candidates.end());

    neighbor_list.clear();
    neighbor_list.reserve(max_size);
    for (size_t i = 0; i < max_size; ++i) {
      neighbor_list.push_back(candidates[i].id);
    }
  }

  float compute_distance(std::span<const T> query, NodeId node) const {
    return squared_distance(Traits::make_span(query),
                            Traits::make_span(vectors_[node]));
  }

  // Configuration
  Dim dim_;
  size_t M_;
  size_t M_max0_;
  size_t ef_construction_;
  float mL_;
  std::atomic<size_t> nprobe_;
  size_t max_retries_;

  // Graph structure
  std::atomic<NodeId> entry_point_;
  std::atomic<int> max_level_;
  std::vector<std::vector<T>> vectors_;
  std::vector<std::vector<std::vector<NodeId>>>
      neighbors_;                 // [node][level][neighbor]
  std::vector<int> node_levels_;  // level assigned to each node
  std::vector<bool> deleted_;     // tombstone flags

  // Partition structures
  std::unordered_map<NodeId, size_t>
      partition_index_;  // center -> partition idx
  std::vector<size_t>
      partition_of_;  // node -> partition idx (NO_PARTITION if unassigned)
  std::vector<NodeId> unassigned_;

  // Concurrency control
  mutable std::shared_mutex global_mutex_;
  mutable std::vector<std::unique_ptr<PartitionState>> node_upper_states_;
  mutable std::deque<std::unique_ptr<PartitionState>> partition_states_;
  mutable std::shared_mutex partition_registry_mutex_;
  mutable ConflictStats conflict_stats_;

  // RNG
  mutable std::mt19937 rng_;
};

}  // namespace nilvec
