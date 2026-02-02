#show "University of Colorado Boulder": name => box[
  #box(image(
    "imgs/cu.png",
    height: 0.7em,
  ))
  #name
]

#set page(margin: (x: 1in, y: 1in))
// #set text(font: "Times New Roman", size: 11pt)
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

      Implementing thread-safety in ANN indexes presents a variety of design choices. Ultimately, the tradeoffs converge to two axes: locking granularity and conflict resolution. In IVFFlat, for example, locking can be applied to each bucket; while effective, this approach is coarse-grained. Conversely, HNSW allows for locking at the finer granularity of a node's neighbor list. In both architectures, conflicts may be handled optimistically or pessimistically. In this paper, we implement and evaluate the cross-product of these strategies.

    ]
  ]
]

#v(2.5em) // Space after the header box

= Introduction

As vector databases increasingly support streaming workloads, the need for concurrent queries against continuously updated indexes has become apparent. While approximate nearest neighbor (ANN) search has been extensively studied under the assumption of a static index @ma2025comprehensivesurveyvectordatabase, the concurrency control mechanisms required for dynamic workloads remain underexplored. Existing systems either sacrifice read-write concurrency entirely or rely on batch processing strategies incompatible with streaming insertions. This paper systematically examines the design space of concurrency control for ANN indexes, focusing on the interplay between locking granularity and conflict resolution strategies.

Existing approaches to parallel ANN index construction fall short of these requirements. FAISS, a well-known example of fine-grained locking, utilizes this locking strictly for parallel index construction (protecting concurrent writers), rather than for isolation between readers and writers. Consequently, it does not support simultaneous search and update. Furthermore, while FAISS exploits thread-level parallelism to batch-process search queries, the search path itself is completely lock-free @faiss_indexhnsw.

ParlayANN takes a different approach, offering two strategies for parallel construction. Its deterministic batch construction uses prefix sums and parallel primitives to process points in a fixed order, ensuring reproducible results across runs. For incremental updates, it employs lock-free batch insertion using compare-and-swap operations, allowing threads to modify neighbor lists without acquiring locks @manohar2024parlayannscalabledeterministicparallel. However, neither technique is well-suited for online systems. Prefix sum-based construction requires the entire batch of points to be known upfront—offsets must be computed before any insertions can proceed, making it fundamentally incompatible with a stream of arriving vectors. Similarly, batch conflict scheduling assumes all operations are available for analysis before execution, enabling the scheduler to order them to minimize conflicts. In an online setting, there is no batch to analyze. The system must handle each operation as it arrives while maintaining consistency with concurrent operations already in flight.

JVector implements lock-free concurrent construction but is not benchmarked on standard suites, making performance comparisons difficult @jvector. And neither FAISS's parallel construction nor ParlayANN's batch strategies address the online read-write concurrency problem we examine.

This gap motivates our systematic exploration of the concurrency control design space for online ANN indexes. We identify two key dimensions along which strategies may vary: locking granularity (coarse-grained at the partition or bucket level vs. fine-grained at the individual neighbor list level) and conflict resolution policy (optimistic with validation and retry vs. pessimistic with blocking). The cross-product of these dimensions yields four distinct strategies. We implement each strategy for both IVFFlat and HNSW indexes and evaluate their performance under mixed read-write workloads. Our evaluation provides the first empirical comparison of concurrency control approaches for streaming vector search.

= Methodology

= Results

= Discussion

- *_Why hasn't this been done before?_* That's an excellent question. Techniques as trivial for database concurrency control as these should have been implemented long ago. Well, one possiblity is that for most applications these techniques are unnecessary; most ML workflows simply don't require this level of concurrency control. Another possiblity is that these indexes have had _relatively_ short times spent in the research limelight.

= Conclusion

#bibliography("citations.bib")
