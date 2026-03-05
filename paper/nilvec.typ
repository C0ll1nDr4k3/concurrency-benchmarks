
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

As vector databases increasingly support streaming workloads, the need for concurrent queries against continuously updated indexes has become apparent @gong2025vstream. While approximate nearest neighbor (ANN) search has been extensively studied under the assumption of a static index @ma2025comprehensivesurveyvectordatabase, the concurrency control mechanisms required for dynamic workloads remain underexplored @singh2021freshdiskann. Existing systems either sacrifice read-write concurrency entirely or rely on batch processing strategies incompatible with streaming insertions. This paper systematically examines the design space of concurrency control for ANN indexes, focusing on the interplay between locking granularity and conflict resolution strategies.

Existing approaches to parallel ANN index construction fall short of these requirements. FAISS, a well-known example of fine-grained locking, utilizes this locking strictly for parallel index construction (protecting concurrent writers), rather than for isolation between readers and writers @douze2024faiss. Consequently, it does not support simultaneous search and update. Furthermore, while FAISS exploits thread-level parallelism to batch-process search queries, the search path itself is completely lock-free @faiss_indexhnsw.

ParlayANN takes a different approach, offering two strategies for parallel construction. Its deterministic batch construction uses prefix sums and parallel primitives to process points in a fixed order, ensuring reproducible results across runs. For incremental updates, it employs lock-free batch insertion using compare-and-swap operations, allowing threads to modify neighbor lists without acquiring locks @manohar2024parlayannscalabledeterministicparallel. However, neither technique is well-suited for online systems. Prefix sum-based construction requires the entire batch of points to be known upfront; offsets must be computed before any insertions can proceed, making it fundamentally incompatible with a stream of arriving vectors. Similarly, batch conflict scheduling assumes all operations are available for analysis before execution, enabling the scheduler to order them to minimize conflicts. In an online setting, there is no batch to analyze.

JVector demonstrates near-linear scaling of nonblocking concurrent construction on SIFT-1M @jvector, but like FAISS and ParlayANN this targets build-time parallelism rather than online read-write concurrency.

This gap motivates our systematic exploration of the concurrency control design space for online ANN indexes. We identify two key dimensions along which strategies may vary: locking granularity (coarse-grained at the partition or bucket level vs. fine-grained at the individual neighbor list level) and conflict resolution policy (optimistic with validation and retry vs. pessimistic with blocking) @kung1981optimistic. Combining these two dimensions yields four distinct strategies. We implement each strategy for both IVFFlat and HNSW indexes and evaluate their performance under mixed read-write workloads. Our evaluation provides the first empirical comparison of concurrency control approaches for streaming vector search.

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
      T2: insert → write to red, bump v: 5→6
      T4: insert → write to yellow, bump v: 14→15
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

The four concurrency strategies above apply uniformly within each index family, but the structural asymmetry between HNSW and IVFFlat suggests a hybrid that inherits the strengths of both. HNSW's upper layers naturally partition the vector space: each node promoted to layer 1 or above implicitly defines a Voronoi cell over the layer-0 nodes nearest to it. By making this partition explicit, we obtain IVF-style write isolation at the base layer while retaining HNSW's sublinear graph walk for search.

Our `HybridPessimistic` implementation maintains a full HNSW graph across all layers, including layer 0. Each layer-1+ node additionally registers a partition upon insertion, and each layer-0 node is assigned to the partition whose center was nearest during the inserting thread's descent. The partition assignment determines which lock protects a node's layer-0 edge list -- it does not restrict the search path. Layer-0 edges cross partition boundaries freely, connecting each node to its M nearest neighbors regardless of partition membership.

The $~$M nodes that arrive before the first layer-1+ node are held in an unassigned queue and drained into their nearest center when one appears, using brute-force nearest-center lookup over this small set. Deletion of a partition center triggers reassignment: the center's children are redistributed to their nearest surviving center, and the center's upper-layer graph edges are severed. All deleted nodes are tombstoned and filtered during graph traversal.

Search proceeds in two phases. First, greedy descent through the upper HNSW layers routes the query to layer 1, where a beam search identifies the `nprobe` nearest partition centers. Second, a layer-0 beam search is launched from each probed center, walking edges across partition boundaries. As each node is encountered, its partition's shared lock is acquired to read the edge list. The `nprobe` parameter controls how many entry points seed the layer-0 walk: additional entry points explore spatially distinct regions of the graph, improving recall for queries near partition boundaries.

The locking scheme separates the concurrency domain from the search domain. Upper-layer graph edges are protected by per-layer `shared_mutex` locks (as in `HNSWCoarsePessimistic`). At layer 0, each partition has its own `shared_mutex` protecting the edge lists of all nodes assigned to it. Readers acquire shared partition locks as they traverse nodes, so multiple concurrent searches never block each other. Writers acquire exclusive locks on the partitions they modify: the new node's own partition for its edge list, plus each neighbor's partition for reverse edges, acquired in index order to prevent deadlock. Because a typical insert touches at most 2--3 partitions (bounded by M), writers to disjoint regions proceed without contention; the same concurrency property that makes IVFFlat scale well, but with sublinear search rather than linear flat scan.

The tradeoff is that partition quality depends on HNSW's stochastic level generation rather than $k$-means training. The resulting partitions are less balanced than IVFFlat's, and partition sizes follow the distribution of the data rather than an optimized quantizer. However, because the layer-0 search walks edges across partition boundaries rather than scanning within them, recall is less sensitive to partition quality than in pure IVF. Even `nprobe` $= 1$ can achieve high recall if the graph is well-connected. The routing cost through the upper layers is logarithmic in the number of partitions rather than linear, since HNSW's graph replaces the brute-force centroid scan.

== Workload

The benchmarking harness is implemented in Python and invokes the core C++ index implementations through bindings to minimize orchestration overhead.

The primary workload consists of a concurrent mix of document insertions and approximate nearest neighbor searches. We vary the number of concurrent worker threads across seven points -- 2, 4, 8, 12, 16, 20, and 24 -- spanning the full physical core count of the test machine and capturing behavior through the E-core saturation region into combined P+E load.

We evaluate each index configuration under two write-ratio bands, both ramping linearly across the thread-count sweep:

- *Production band (W: 1%--5%)*. The write ratio increases from 1% at 2 threads to 5% at 24 threads. Standard database benchmarks model production workloads as overwhelmingly read-heavy. YCSB Workload B (95% read, 5% update) and Workload D (95% read, 5% insert) represent the read-mostly tier of the Yahoo! Cloud Serving Benchmark suite @cooper2010ycsb, and even the "update-heavy" Workload A allocates only 50% of operations to writes. Production vector databases skew further toward reads: ingestion is typically batched or periodic, while queries arrive continuously. Our 1--5% range falls within this production regime.

- *Stress-test band (W: 20%--50%)*. The write ratio increases from 20% at 2 threads to 50% at 24 threads. At these ratios, write-write and read-write conflicts become frequent enough to expose behavioral differences between concurrency control strategies that the production band masks. This band is not intended to represent a deployment profile; it isolates the locking and conflict-resolution overhead that is the focus of this paper.

We evaluate our implementations using the SIFT-128 @annbench_sift128, Fashion-MNIST-784 @xiao2017fashionmnist, GloVe @annbench_glove100 @annbench_glove25, GIST-960 @annbench_gist960, NYTimes-256 @annbench_nytimes256, and MNIST-784 @annbench_mnist784 datasets.

== Measurement

Each throughput data point proceeds in two phases. First, a _construction phase_ creates a fresh index, trains it (for IVF variants), and bulk-inserts the first 50% of the dataset into the shared index, single-threaded. This pre-load gives the index enough structure to serve queries before the timed phase begins. Build time covers this phase and is reported separately; it is excluded from the throughput calculation. Second, a _mixed-workload phase_ launches writer and reader threads concurrently against the pre-loaded index: writers insert the remaining vectors (split evenly across insert threads) while readers execute five full passes over the query set. All performance-critical C++ methods release the Python GIL, so threads achieve true parallelism. Throughput is computed as total operations (inserts plus searches) divided by the wall-clock duration of this second phase only.

All indexes return $k = 10$ nearest neighbors per query. HNSW variants use $M = 16$ bidirectional links per node and $"ef"_"construction" = 200$. IVFFlat variants use $n_"list" = floor(sqrt(N))$ clusters, where $N$ is the number of indexed vectors, and $n_"probe" = 1$ for throughput benchmarks.

All experiments were conducted on a Lenovo Legion Pro 7i 16IAX10H with an Intel Core Ultra 9 275HX CPU (24 total cores: 8P+16E) and 32 GiB RAM. Our evaluation is restricted to CPU-based architectures to maintain a direct comparison between our custom concurrency control strategies. Although GPU-acceleration is a common optimization for batch ANN search, implementing our proposed fine-grained locking and optimistic retry mechanisms within GPU kernels involves significant architectural complexity that we reserve for future work.

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
  ),
  caption: [
    Throughput scaling with increasing numbers of threads for each index type, for tested full-run datasets.
  ],
)

= Discussion

- *_Why hasn't this been done before?_* The predominant paradigm has been offline construction followed by read-only serving, so reader-writer concurrency at the index level simply never arose. Systems that do accept writes typically side-step the problem architecturally: mutable segments are flushed to read-only storage on a rolling basis, and searches hit only sealed data @wang2021milvus. Deletions are handled similarly via tombstoning followed by periodic rebuilds rather than in-place graph repair @pinecone2023hnsw. Streaming workloads driven by retrieval-augmented generation @lewis2020rag have only recently made in-place concurrent update a practical consideration.

- *_Why does HNSW degrade more under concurrent writes than IVF?_* The answer lies in a structural asymmetry between the two index types. HNSW is a navigable small-world graph: its search correctness depends on the graph remaining well-connected across layers @malkov2020hnsw. During insertion, a new node is wired into the graph by selecting neighbors via a heuristic and then back-linking those neighbors to the new node @malkov2020hnsw. Under concurrent writes, these two steps are not atomic. A racing writer can observe a partially-linked node, traverse a stale edge, or have its own neighbor list pruned before its back-links are established, any of which can leave the graph with weakly connected or entirely isolated nodes. Heuristics like deferred or batched pruning reduce the frequency of such breaks but cannot eliminate them: they trade recall loss for throughput, rather than recovering the full structural guarantee. IVF does not share this vulnerability. At the coarse quantizer level, cluster assignment is a read-only operation; inserting a vector into a cluster appends to a list and requires only a per-cluster lock @douze2024faiss. Concurrent writers targeting different clusters are entirely independent, and even writers within the same cluster contend only on a flat append structure with no graph invariant to preserve. The degradation seen in HNSW throughput and recall under high write concurrency is therefore not merely an implementation artifact; it reflects a fundamental tension between the graph connectivity invariant that makes HNSW fast and the atomicity that concurrent mutation requires.

= Conclusion

#bibliography("citations.bib")
