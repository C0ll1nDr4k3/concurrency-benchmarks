
#show "University of Colorado Boulder": name => box[
  #box(image(
    "imgs/cu.png",
    height: 0.7em,
  ))
  #name
]

#set page(margin: (x: 1in, y: 1in), numbering: "1", number-align: center + bottom)
#set heading(numbering: "1.")
#set par(justify: true)

#let title = [
  Increasing Throughput with Concurrency Control for ANN Indexes
]

#let authors = [
  Collin Drake \
  University of Colorado Boulder \
  #link("mailto:collin.drake@colorado.edu")
]

// Main header box with gray background
#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 1.5em,
)[
  #box(
    width: 100%,
    fill: rgb("#f5f5f5"), // Light gray background
    // fill: gradient.radial(
    //   (rgb("#ffe8e8"), 0%),     // Light blue at top-left
    //   (rgb("#f5f5f5"), 50%),    // Transition to light gray
    //   (rgb("#f5f5f5"), 100%),   // Light gray at edges
    //   center: (20%, 16%)        // Gradient center at top-left area
    // ),
    radius: 8pt, // Rounded corners
    inset: (x: 24pt, y: 20pt),
    stroke: none,
  )[
    #align(center)[
      #set par(justify: false)
      #text(size: 20pt, weight: "bold", hyphenate: false)[#title]
    ]

    #align(center)[
      #v(12pt)

      // Authors
      #text(size: 11pt)[#authors]

      #v(16pt)
    ]

    // Abstract section
    #block()[
      #text(weight: "bold")[Abstract] #h(0.5em)

      Implementing thread-safety in ANN indexes presents a variety of design choices. Ultimately, the tradeoffs converge to two axes: locking granularity and conflict resolution. In both architectures, conflicts may be handled optimistically or pessimistically. In this paper, we implement and evaluate all four combinations of these two axes against industry-standard implementations.

    ]
  ]
]

#v(2.5em) // Space after the header box

= Introduction

As vector databases increasingly support streaming workloads, the need for concurrent queries against continuously updated indexes has become apparent @gong2025vstream. While approximate nearest neighbor (ANN) search has been extensively studied under the assumption of a static index @ma2025comprehensivesurveyvectordatabase, the concurrency control mechanisms required for dynamic workloads remain underexplored @singh2021freshdiskann. Existing systems either sacrifice read-write concurrency entirely or rely on batch processing strategies incompatible with streaming insertions.

Existing approaches to parallel ANN index construction fall short of these requirements. FAISS, a well-known example of fine-grained locking, utilizes this locking strictly for parallel index construction (protecting concurrent writers), rather than for isolation between readers and writers @douze2024faiss. Consequently, it does not support simultaneous search and update. Furthermore, while FAISS exploits thread-level parallelism to batch-process search queries, the search path itself is completely lock-free @faiss_indexhnsw.

ParlayANN takes a different approach, offering two strategies for parallel construction. Its deterministic batch construction uses prefix sums and parallel primitives to process points in a fixed order, ensuring reproducible results across runs. For incremental updates, it employs lock-free batch insertion using compare-and-swap operations, allowing threads to modify neighbor lists without acquiring locks @manohar2024parlayannscalabledeterministicparallel. However, neither technique is well-suited for online systems. Prefix sum-based construction requires the entire batch of points to be known upfront; offsets must be computed before any insertions can proceed, making it fundamentally incompatible with a stream of arriving vectors. Similarly, batch conflict scheduling assumes all operations are available for analysis before execution, enabling the scheduler to order them to minimize conflicts. In an online setting, there is no batch to analyze.

JVector demonstrates near-linear scaling of nonblocking concurrent construction on SIFT-1M @jvector, but like FAISS and ParlayANN this targets build-time parallelism rather than online read-write concurrency.

We address this gap directly. Concurrency control for ANN indexes varies along two axes: locking granularity (coarse-grained per partition or bucket vs. fine-grained per neighbor list) and conflict resolution (pessimistic blocking vs. optimistic validation and retry) @kung1981optimistic. We implement all four combinations for both IVFFlat and HNSW and benchmark them under mixed read-write workloads.

= Methodology

== Index Implementations

Each index family (IVFFlat and HNSW) is implemented with four concurrency strategies spanning two axes: locking granularity (coarse vs. fine) and conflict resolution (pessimistic vs. optimistic). All variants use SIMD-accelerated distance computation via a common module. All IVFFlat variants additionally apply four-element prefetching in the scan inner loop.

#figure(
  grid(
    columns: 2,
    gutter: 12pt,
    [
      #image("plots/voronoi_pessimistic.svg", width: 100%)
      #align(center)[_Pessimistic_]
      ```
      T1: search → acquire green (shared)
      T2: insert → acquire red (exclusive)
      T3: insert → acquire yellow (exclusive)
      T4: insert → acquire red (blocked)
      T1: scan green → release
      T2: write red → release → T4 unblocks
      T3: write yellow → release
      T4: write red → release
      ```
    ],
    [
      #image("plots/voronoi_optimistic.svg", width: 100%)
      #align(center)[_Optimistic_]
      ```
      // red: v=5; green: v=12; yellow: v=14
      T1: search → snapshot red v=5
      T2: insert → write to red v5→v6
      T4: insert → write to yellow v14→v15
      T3: search → snapshot green v=12
      T1: validate red: 5 ≱ 6 → retry
      T3: validate green: 12 = 12
      ```
    ],
  ),
  caption: [
    Geometric interpretation of IVFFlat coarse locking strategies over a
    Voronoi partition. _Pessimistic_: each cell is guarded by a mutex;
    readers acquire a shared lock and writers an exclusive lock. _Optimistic_:
    threads snapshot the cell version before operating and validate on
    commit; a writer that advances the version mid-search forces the reader
    to retry.
  ],
)

=== HNSW-IVF Hybrid

The four concurrency strategies above apply uniformly within each index family, but the structural asymmetry between HNSW and IVFFlat suggests a hybrid that inherits the strengths of both. HNSW's upper layers naturally partition the vector space: each node promoted to layer 2 or above implicitly defines a Voronoi cell over the lower-layer nodes nearest to it. These upper-layer nodes are the hub nodes of the graph -- a small, well-connected subset responsible for most long-range traversal @munyampirwa2025hubs -- making them effective spatial landmarks for partitioning. By making this partition explicit, we obtain IVF-style write isolation at the base layer while retaining HNSW's sublinear graph walk for search.

The choice of layer 2 rather than layer 1 as the partition threshold follows from the geometry of HNSW's level distribution. Each HNSW layer contains roughly $1 slash M$ of the nodes from the layer below. For $N$ indexed vectors, layer 1 contains approximately $N slash M$ nodes and layer 2 contains approximately $N slash M^2$ nodes. Using layer-1 nodes as partition centers produces $N slash M$ partitions with only $~M$ vectors per cell on average. At this granularity the partitions are too sparse for productive IVF-style probing: reaching acceptable recall requires probing many nearly-empty cells, and the per-cell overhead dominates. Layer 2, by contrast, yields $N slash M^2$ partitions with $~M^2$ vectors per cell, which for typical values of $M$ is within the same order of magnitude as the standard IVF heuristic of $sqrt(N)$ clusters (exact alignment occurs when $N approx M^4$). For example, with $M = 16$ and $N = 1,000,000$, layer 1 contains roughly 62,500 nodes ($~16$ vectors per cell) while layer 2 contains roughly 3,900 nodes ($~256$ vectors per cell), compared to $sqrt(N) = 1,000$ under the IVF heuristic.

Our `HybridPessimistic` implementation maintains a full HNSW graph across all layers, including layer 0. Each layer-2+ node additionally registers a partition upon insertion, and each lower-layer node is assigned to the partition whose center was nearest during the inserting thread's descent. *The partition assignment determines which lock protects a node's layer-0 edge list; it does not restrict the search path.* Layer-0 edges cross partition boundaries freely, connecting each node to its $M$ nearest neighbors regardless of partition membership.

The first nodes that arrive before any layer-2+ node appears are held in an unassigned queue and drained into their nearest center when one appears, using brute-force nearest-center lookup over this small set. Deletion of a partition center triggers reassignment: the center's children are redistributed to their nearest surviving center, and the center's upper-layer graph edges are severed. All deleted nodes are tombstoned and filtered during graph traversal.

Search proceeds in two phases. First, greedy descent through the upper HNSW layers routes the query to layer 2, where a beam search identifies the `nprobe` nearest partition centers. Second, a layer-0 beam search is launched from each probed center, walking edges across partition boundaries. As each node is encountered, its partition's shared lock is acquired to read the edge list. The `nprobe` parameter controls how many entry points seed the layer-0 walk: additional entry points explore spatially distinct regions of the graph, improving recall for queries near partition boundaries.

The locking scheme separates the concurrency domain from the search domain. Upper-layer graph edges are protected by per-layer `shared_mutex` locks (as in `HNSWCoarsePessimistic`). At layer 0, each partition has its own `shared_mutex` protecting the edge lists of all nodes assigned to it. Readers acquire shared partition locks as they traverse nodes, so multiple concurrent searches never block each other. Writers acquire exclusive locks on the partitions they modify: the new node's own partition for its edge list, plus each neighbor's partition for reverse edges, acquired in index order to prevent deadlock. Because a typical insert touches at most 2-3 partitions (bounded by $M$), writers to disjoint regions proceed without contention: the same concurrency property that makes IVFFlat scale well, but with sublinear search rather than linear flat scan.

The tradeoff is that partition quality depends on HNSW's stochastic level generation rather than $k$-means training. The resulting partitions are less balanced than IVFFlat's, and partition sizes follow the distribution of the data rather than an optimized quantizer. However, because the layer-0 search walks edges across partition boundaries rather than scanning within them, recall is less sensitive to partition quality than in pure IVF. Even `nprobe` $= 1$ can achieve reasonable recall if the graph is well-connected. The routing cost through the upper layers is logarithmic in the number of partitions rather than linear, since HNSW's graph replaces the brute-force centroid scan.

== Workload

The benchmarking harness is implemented in Python and invokes the core C++ index implementations through bindings to minimize orchestration overhead.

The primary workload consists of a concurrent mix of document insertions and approximate nearest neighbor searches. We vary the number of concurrent worker threads across seven points -- 2, 4, 8, 12, 16, 20, and 24 -- spanning the full physical core count of the test machine and capturing behavior through the E-core saturation region into combined P+E load. For each thread-count point, the harness uses a fixed split: $T_i = max(1, floor(T/4))$, $T_s = T - T_i$.

We evaluate our implementations using the SIFT-128 ($n = 1 thin 000 thin 000$), Fashion-MNIST-784 ($n = 60 thin 000$) @xiao2017fashionmnist, GloVe ($n = 1 thin 183 thin 514$), GIST-960 ($n = 1 thin 000 thin 000$), NYTimes-256 ($n = 290 thin 000$), and MNIST-784 ($n = 60 thin 000$) datasets @annbenchmarks.

== Measurement

Each throughput data point proceeds in three phases. First, a _construction phase_ creates a fresh index and trains it (for IVF variants). Second, a _preload phase_ inserts a fixed fraction of the dataset single-threaded (50% by default), populating the index to a realistic density before any concurrent operations begin. Build and preload time are reported separately; both are excluded from the throughput calculation. Third, a _mixed-workload phase_ launches insert and search threads concurrently using the fixed split above. Insert threads process the remaining (unpreloaded) fraction of the dataset, sharded evenly across writer threads, while search threads execute five full passes over the query set against the already-populated index. All performance-critical C++ methods release the Python GIL, so threads achieve true parallelism. When a C++ call returns, however, the thread must reacquire the GIL before Python can dispatch the next operation; if another thread holds the GIL at that instant, the thread stalls despite having completed its index work. This queuing at the GIL boundary is absorbed into the wall-clock window, making absolute throughput figures a conservative lower bound on what the underlying indexes can sustain. Wall-clock time is retained rather than instrumented C++ time to maintain consistency with external library baselines (FAISS, USearch, Weaviate, Redis), which cannot be measured otherwise; since all implementations share the same harness overhead, relative comparisons remain valid. Throughput is computed as total operations (inserts plus searches) divided by the wall-clock duration of this second phase only.

All indexes return $k = 10$ nearest neighbors per query. HNSW variants use $M = 16$ bidirectional links per node and $"ef"_"construction" = 200$. IVFFlat variants use $n_"list" = floor(sqrt(N))$ clusters, where $N$ is the number of indexed vectors, and $n_"probe" = sqrt(n_"list")$ for throughput benchmarks.

All experiments were conducted on a Lenovo Legion Pro 7i 16IAX10H with an Intel Core Ultra 9 275HX CPU (24 total cores: 8P+16E) and 32 GiB RAM. Our evaluation is restricted to CPU-based architectures to maintain a direct comparison between our custom concurrency control strategies. Although GPU-acceleration is a common optimization for batch ANN search, implementing our proposed fine-grained locking and optimistic retry mechanisms within GPU kernels involves significant architectural complexity that we reserve for future work.

== Graph Connectivity

The preceding section compared concurrency strategies on throughput; we now analyze a correctness axis particular to graph-based indexes: disjoint nodes. An HNSW graph is a navigable small-world structure whose search correctness depends on every indexed vector being reachable from the entry point via a bounded-length walk @malkov2020hnsw. Under concurrent mutation, this invariant can be violated even when no individual operation appears incorrect in isolation.

=== Failure Mode

Disjoint nodes arise during the neighbor-list rewrite step of HNSW insertion. When a new node $v$ is inserted, it selects $M$ neighbors via the Malkov heuristic, and each of those neighbors in turn has its own neighbor list pruned to bound out-degree at $M$ @malkov2020hnsw. Pruning replaces the neighbor's list wholesale rather than appending to it. Under concurrent inserts, two threads $A$ and $B$ may both read the same neighbor list $N(u) = {a, b, c}$ for some node $u$, each independently compute a replacement (say $A$ writes ${a, b, d}$ and $B$ writes ${a, b, e}$), and the later commit silently clobbers the earlier. If the clobbered edge was the sole remaining inbound edge to a node $w$ at that layer, then $w$ becomes unreachable from the entry point. The node is still present in the index's vector store and may be reachable via a different layer, but at the affected layer it is structurally disjoint.

=== Expected Rate by Strategy

Let $T$ denote the number of concurrent writer threads, $N$ the number of indexed vectors at the relevant layer, and $rho$ the write fraction. We characterize the expected per-operation disjoint rate $delta$ for each strategy.

+ *Vanilla, Pessimistic, and Hybrid*. Both preserve the sequential HNSW invariant. Vanilla by construction, pessimistic by serializing the prune-and-rewrite critical section. So, $delta$ reduces to the baseline established in the original HNSW analysis @malkov2020hnsw. Hybrid varaints inherit the rate of their parent (pessimistic or optimistic), so the same generalizations apply.

+ *Optimistic*.
  Two probabilities govern the rate. Let $p_"race"$ be the probability that two threads concurrently mutate the same neighbor list within a commit window, and $p_"crit"$ the conditional probability that a clobbered edge was the sole remaining path to some node. By a birthday argument over neighbor lists touched per insert,
  $ p_"race" approx binom(T, 2) dot c^2 / N = Theta(T^2 / N), $
  where $c$ is the average number of neighbor lists touched per insert (bounded by $M$). The $c^2$ factor follows from a union bound: each of two threads has a footprint of $c$ lists out of $N$, so the probability that a second thread hits any list in the first thread's footprint is approximately $c dot (c / N)$. The expected disjoint rate per operation is then
  $ delta_"opt" approx rho dot p_"race" dot p_"crit" = Theta(rho dot T^2 / N). $
  The quadratic dependence on thread count is the central formal difference between optimistic and pessimistic strategies on this metric, and it predicts a knee beyond which additional threads trade structural integrity for throughput at an accelerating rate.

=== Measurement Protocol

We quantify disjointness directly rather than inferring from recall, since recall conflates structural breakage with search-tuning suboptimality. After each benchmark run, we perform a breadth-first search from the entry point at each HNSW layer and count the nodes not visited. We report the per-layer disjoint rate, $delta_ell = (|"unreached"_ell|) / (|"nodes at layer"_ell|)$, alongside the overall rate. Layer 0 is of primary interest, since unreachable layer-0 nodes directly reduce achievable recall; higher-layer disjointness is a secondary indicator of upper-level graph damage.

This measurement yields a principled bound on usable concurrency: for any acceptable threshold $delta^*$, the largest $T$ satisfying $delta(T) lt.eq delta^*$ is the honest upper limit on concurrency for a given graph size under a given strategy.

= Results

== Throughput

Incidentally, the lower a dataset's throughput, the higher is cluster density.



#figure(
  grid(
    columns: 2,
    gutter: 12pt,
    figure(
      image("plots/sift-128-euclidean_full/throughput_scaling.svg", width: 100%),
      caption: [SIFT-128],
    ),
    figure(
      image("plots/fashion-mnist-784-euclidean_full/throughput_scaling.svg", width: 100%),
      caption: [Fashion-MNIST-784],
    ),

    figure(
      image("plots/fashion-mnist-784-euclidean_full/throughput_scaling.svg", width: 100%),
      caption: [Fashion-MNIST-784],
    ),
  ),
  caption: [
    Throughput scaling with increasing numbers of threads for each index type, for tested full-run datasets.
  ],
)

== Disjoint Rate

We report the measured per-operation disjoint rate as a function of thread count for each HNSW concurrency strategy, sweeping the write fraction $rho$ across the same range as the throughput sweep. The model in the previous section predicts pessimistic and vanilla variants to remain flat at the sequential baseline, and optimistic variants to exhibit growth consistent with the $Theta(rho dot T^2 / N)$ bound. We report layer 0 and overall rates separately; the gap between them quantifies how much of the concurrency damage is masked by HNSW's multi-layer routing.

// TODO: plots/*/disjoint_rate.svg — generated from the post-run BFS pass

= Discussion

- *_Why hasn't this been done before?_* The predominant paradigm has been offline construction followed by read-only serving, so reader-writer concurrency at the index level simply never arose. Systems that do accept writes typically side-step the problem architecturally: mutable segments are flushed to read-only storage on a rolling basis, and searches hit only sealed data @wang2021milvus. Deletions are handled similarly via tombstoning followed by periodic rebuilds rather than in-place graph repair @pinecone2023hnsw. Streaming workloads driven by retrieval-augmented generation @lewis2020rag have only recently made in-place concurrent update a practical consideration.

- *_Why does HNSW degrade more under concurrent writes than IVF?_* The answer lies in a structural asymmetry between the two index types. HNSW is a navigable small-world graph: its search correctness depends on the graph remaining well-connected across layers @malkov2020hnsw. During insertion, a new node is wired into the graph by selecting neighbors via a heuristic and then back-linking those neighbors to the new node @malkov2020hnsw. Under concurrent writes, these two steps are not atomic. A racing writer can observe a partially-linked node, traverse a stale edge, or have its own neighbor list pruned before its back-links are established, any of which can leave the graph with weakly connected or entirely isolated nodes. Heuristics like deferred or batched pruning reduce the frequency of such breaks but cannot eliminate them: they trade recall loss for throughput, rather than recovering the full structural guarantee. The graph connectivity analysis above quantifies this tradeoff directly, showing that the expected disjoint rate under optimistic concurrency grows as $Theta(rho T^2 / N)$ while pessimistic variants remain at the sequential baseline. The damage is disproportionate when hub nodes are affected, since a small hub subset carries most long-range traversal @munyampirwa2025hubs. IVF does not share this vulnerability. At the coarse quantizer level, cluster assignment is a read-only operation; inserting a vector into a cluster appends to a list and requires only a per-cluster lock @douze2024faiss. Concurrent writers targeting different clusters are entirely independent, and even writers within the same cluster contend only on a flat append structure with no graph invariant to preserve. The degradation seen in HNSW throughput and recall under high write concurrency is therefore not merely an implementation artifact; it reflects a fundamental tension between the graph connectivity invariant that makes HNSW fast and the atomicity that concurrent mutation requires.

= Conclusion

#bibliography("citations.bib")
