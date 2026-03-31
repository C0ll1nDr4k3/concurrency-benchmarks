# Hybrid HNSW-IVF Index Design

## Overview

The Hybrid index combines HNSW's hierarchical navigation structure with IVF's partition-based organization at layer-0. This design eliminates the need for a separate training phase while achieving the locality benefits of partitioned search.

## Architecture

### Layer Structure

- **Upper layers (≥1)**: Standard HNSW graph
  - Coarse navigation using greedy search
  - Each layer has approximately 1/M as many nodes as the layer below
  - Protected by per-layer locks (pessimistic) or version numbers (optimistic)

- **Layer-0**: HNSW graph with IVF-style partitioning
  - Full connectivity graph like standard HNSW
  - Nodes assigned to partitions based on their level
  - Partition centers are nodes that reach level ≥2

### Partition Assignment

Unlike traditional IVF which requires k-means clustering during a training phase, partitions emerge organically from HNSW's level generation:

```
If node level ≥ 2:
    Node becomes a partition center
    Node is assigned to its own partition
Else:
    Node is assigned to nearest level-2+ node's partition
```

**Probability**: With M=16, approximately 1 in M² ≈ 1/256 nodes reach level-2 or higher. For 10,000 vectors, this yields ~39 partition centers without any training step.

### Why This Works

1. **Uniform distribution**: HNSW's exponential level distribution `P(level ≥ ℓ) = 1/M^ℓ` creates naturally dispersed partition centers. And it's worth noting that these centroids are not as well distributed as the k-means clustering found in IVF.

2. **Hierarchical assignment**: Upper-layer edges already connect nodes to diverse regions of the space, so the nearest level-2+ neighbor is typically a good partition representative

3. **Dynamic growth**: Partitions form as vectors are inserted; no separate training phase or rebuild required

## Concurrency Control

### HybridPessimistic

Uses reader-writer locks (`std::shared_mutex`) with different granularities:

**Upper layers**: Per-node locking

- Each node has its own `shared_mutex` protecting its edge lists at layers ≥ 1
- Search: Copies neighbor list under per-node shared lock
- Insert: Acquires exclusive locks on `{new_id} ∪ {selected_neighbors}` in sorted NodeId order (deadlock-free); concurrent inserts to disjoint neighbor sets proceed in parallel

**Layer-0**: Per-partition locking

- Search: Acquires shared locks on partitions as they're visited
  - Locks acquired in sorted partition ID order to prevent deadlocks
  - Multiple readers never block each other (shared locks)
- Insert: Acquires exclusive locks on target partition(s)
  - Writers in disjoint partitions proceed concurrently

**Lock order**: `global → node[low_id..high_id] → partition_registry → partition[low_idx..high_idx]`

### HybridOptimistic

Uses version numbers with retry-based optimistic concurrency control:

**Upper layers**: Per-node versioning (one `PartitionState` per node)

- Search: Reads under per-node shared locks
- Insert: Snapshots write-set versions → acquires exclusive per-node locks → validates versions → commits or retries

```cpp
snapshot = {node_id: node_version[id] for id in write_set}
acquire_exclusive_locks(write_set, sorted_order)
for (nid, v) in snapshot:
    if node_version[nid] != v: retry()
commit_edges(); increment_versions(write_set)
```

**Layer-0**: Per-partition versioning

```cpp
snapshot = {partition_id: partition_version[id] for all visited partitions}
// ... traverse layer-0 ...
for (pid, v) in snapshot:
    if partition_version[pid] != v:
        retry()
```

**Conflict tracking** (ifdef NILVEC_TRACK_CONFLICTS):

- Counts attempts vs. conflicts for inserts
- Exposes `conflict_stats()` method for analysis

**Fallback**: After `max_retries` (default 10), falls back to pessimistic locking to guarantee progress.

**Lock order**: `global → node[low_id..high_id] → partition_registry → partition[low_idx..high_idx]`

## Search Algorithm

```
1. Navigate upper layers (≥1) using standard HNSW greedy search
   - Find entry point at highest layer
   - Descend layer by layer until reaching layer-1

2. At layer-1, find nprobe nearest partition centers
   - Partition centers are nodes with level ≥ 2
   - These become entry points for layer-0 search

3. Traverse layer-0 from each partition center
   - Standard HNSW beam search using M_max0 neighbors
   - Crosses partition boundaries freely
   - Acquires locks/versions per partition as needed

4. Return top-k from layer-0 candidate set
```

**Key insight**: `nprobe` controls search quality vs. throughput. Higher nprobe explores more partitions (better recall) but requires more locks (lower throughput).

## Insert Algorithm

```
1. Assign level to new node using exponential distribution

2. If level ≥ 2:
      Create new partition for this node
      Register as partition center

3. Insert into upper layers (level down to 1):
      For each layer L:
          Lock/version layer L
          Find nearest neighbors
          Add bidirectional edges (prune to M connections)

4. Assign to partition:
      Find nearest partition center (level-2+ node)
      Register node in that partition

5. Insert into layer-0:
      Lock/version target partition
      Find nearest neighbors within M_max0 radius
      Add bidirectional edges (prune to M_max0)
```

## Remove Operation

Both variants support node deletion:

```
1. Mark node as deleted (soft delete)

2. Remove from partition registry if it's a center
   - Reassign orphaned nodes to next-nearest center

3. Remove edges from neighbors at all layers
   - Neighbors prune the deleted node from their edge lists
   - Holes in the graph heal naturally as new edges form
```

**Trade-off**: Soft deletes leave tombstones. A compaction phase (not yet implemented) would reassign node IDs and rebuild edge lists.

## Performance Characteristics

### Space Complexity

- Vectors: `O(N × D)` where N=node count, D=dimension
- Edges: `O(N × M × log(N))` for upper layers + `O(N × M_max0)` for layer-0
- Partitions: `O(P)` where P ≈ N/M² partition centers

### Search Time

- Upper layers: `O(log(N) × M × ef)`
- Layer-0: `O(nprobe × M_max0 × ef)`
- Total: `O((log(N) + nprobe) × M × ef)`

### Concurrency Scalability

- **Read scaling**: Linear with thread count (shared locks don't conflict)
- **Write scaling**: Linear when inserts target different partitions
- **Mixed workload**: Degradation depends on nprobe (more probes = more lock contention)

## Comparison to Alternatives

### vs. Standard HNSW

- **Advantage**: Partitioned layer-0 reduces lock contention for writes
- **Disadvantage**: nprobe parameter tuning required; low nprobe hurts recall

### vs. IVFFlat

- **Advantage**: Hierarchical navigation is much faster than flat exhaustive search within partitions
- **Advantage**: No training phase required
- **Disadvantage**: More complex implementation; higher memory overhead

### vs. Separate HNSW + IVF

- **Advantage**: Single unified index; no need to maintain two structures
- **Advantage**: Organic partition formation from level distribution
- **Disadvantage**: Less flexibility (can't retrain partitions without rebuild)

## Tuning Parameters

| Parameter       | Default  | Effect                                                                      |
| --------------- | -------- | --------------------------------------------------------------------------- |
| M               | 16       | Edges per node in upper layers; affects graph degree and level distribution |
| M_max0          | 32 (2×M) | Edges per node in layer-0; higher = better recall, more memory              |
| ef_construction | 200      | Search width during insertion; higher = better graph quality                |
| mL              | 1/ln(M)  | Level distribution parameter; lower = more partition centers                |
| nprobe          | 1        | Partitions searched at layer-0; higher = better recall, lower throughput    |
| max_retries     | 10       | (Optimistic only) Retries before pessimistic fallback                       |

## Implementation Notes

### Remove Support

- Both `HybridPessimistic` and `HybridOptimistic` expose `remove(NodeId)` method
- Test coverage: `test_hybrid_optimistic_remove` and `test_hnswivf_remove`
- Python bindings include remove method

## Future Work

- **Compaction**: Reclaim deleted node IDs and rebuild partition assignments
- **Adaptive nprobe**: Dynamically adjust based on load or recall targets
- **Partition splitting**: Handle skewed distributions by splitting large partitions
- **Lock-free upper layers**: Replace per-node shared_mutex reads with true lock-free reads (e.g. RCU or copy-on-write edge lists) to eliminate read-path mutex overhead
