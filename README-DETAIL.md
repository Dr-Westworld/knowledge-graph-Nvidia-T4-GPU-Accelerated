# GPU-Accelerated Knowledge Graph Construction Pipeline

A memory-aware, GPU-first pipeline that converts raw text into an interactive knowledge graph.
Architecture: **CPU ingestion → GPU NLP → GPU graph ops → CPU visualization**
Target: **Google Colab T4 GPU** (tuned for constrained GPU memory/time budgets)

This repository contains a single-file pipeline that demonstrates how to extract entity–relation–entity triples using an on-device LLM, canonicalize and deduplicate them on the GPU with RAPIDS/cuDF, run GPU graph analytics with cuGraph, and render an interactive visualization with PyVis on the CPU.

---

# Table of contents

* [Why this pipeline? (High-level benefits)](#why-this-pipeline-high-level-benefits)
* [What this repo applies / key ideas](#what-this-repo-applies--key-ideas)
* [Architecture & stages](#architecture--stages)
* [How it differs from a “normal” knowledge graph pipeline](#how-it-differs-from-a-normal-knowledge-graph-pipeline)
* [Files & outputs produced](#files--outputs-produced)
* [Quickstart / Usage](#quickstart--usage)
* [Configuration options](#configuration-options)
* [Dependencies & installation notes (Colab/T4 tips)](#dependencies--installation-notes-colabt4-tips)
* [Performance, scaling & memory considerations](#performance-scaling--memory-considerations)
* [Limitations & safety notes](#limitations--safety-notes)
* [Troubleshooting & tips](#troubleshooting--tips)
* [Extensions & future improvements](#extensions--future-improvements)
* [License & Contributing](#license--contributing)

---

# Why this pipeline? (High-level benefits)

* **GPU-first extraction and analytics** — pushes the heavy NLP and graph computation onto the GPU to get dramatically faster extraction, deduplication, and graph algorithms than CPU-only approaches.
* **Memory-aware for smaller GPUs** — tuned defaults and strategies (chunking, batching, offload) to run on constrained GPUs like Colab T4.
* **Streaming-friendly** — processes text in chunks and yields progress so it can be used interactively or embedded in long-running workflows.
* **End-to-end** — from raw text to interactive HTML visualization plus JSON export of edges.
* **Strong deduplication & aggregation on GPU** — uses cuDF groupby/aggregation to canonicalize and collapse duplicate triples at scale before expensive CPU transfers.

---

# What this repo applies / key ideas

* **Token-aware chunking** to keep LLM inputs within model limits while preserving sentence boundaries.
* **Singleton LLM loader** to avoid repeated model loads and limit CPU/GPU memory churn.
* **TOON extraction prompt** — a controlled format the model emits, e.g.

  ```
  TRIPLE source:"Geoffrey Hinton" predicate:"pioneered" target:"deep learning" confidence:0.95
  ```

  This makes parsing deterministic and GPU-friendly.
* **GPU canonicalization** — hash entities to stable int64 keys (xxhash) for efficient join/groupby on cuDF.
* **GPU graph building & analytics** — build a cuGraph graph directly from edge tables and compute PageRank / degree centrally on GPU.
* **CPU visualization** — transfer a trimmed set of edges to the CPU (Polars/Arrow zero-copy conversion) and render with PyVis; avoids rendering massive graphs in the browser by default.
* **Aggressive memory hygiene** — `cleanup_memory()` and careful `del`/`torch.cuda.empty_cache()` calls to minimize OOM on a T4.

---

# Architecture & stages

1. **Stage 1 — CPU ingestion & chunking**

   * `TextChunker` splits long text into token-safe chunks using sentence boundaries to preserve coherence for the LLM.

2. **Stage 2 — GPU NLP extraction**

   * `TripleExtractor` loads an instruction-tuned causal LM (configurable) onto the GPU and generates standardized `TRIPLE` lines per chunk.
   * Batched generation to maximize GPU throughput.

3. **Stage 3 — GPU triple canonicalization**

   * `GPUTripleStore` parses TOON lines, hashes entities/predicates/targets to stable int64 ids, stores them in cuDF and appends new triples in streaming fashion.
   * Aggregation uses GPU groupby to compute mean confidence and counts.

4. **Stage 4 — GPU graph analytics**

   * `GPUGraphAnalyzer` builds a cuGraph directed graph and runs algorithms like PageRank and degree centrality on the GPU.

5. **Stage 5 — CPU projection & visualization**

   * `PyVisAdapter` converts a subset of the edges to Polars (via Arrow zero-copy), applies limits for visualization, applies PageRank sizing, and produces an interactive PyVis `Network` that is written to HTML.

6. **Orchestration**

   * `StreamableKGPipeline` connects all stages, offers streaming `process_text()` with progress yields, and `finalize_graph()` to run aggregation, analytics and produce the visualization.

---

# How it differs from a "normal" knowledge graph pipeline

Typical, CPU-first KG pipelines:

* Use CPU-based NLP (spaCy, rule-based extractors) or remote LLMs (latency + cost).
* Move everything to the CPU/DB for deduplication and analytics (slow for large volumes).
* Visualize raw graphs in the browser—often impractical at scale.

This GPU-accelerated pipeline:

* Runs LLM extraction **on-device (GPU)** — lower latency and no external API calls.
* Performs **deduplication & aggregation on GPU** (cuDF) — orders-of-magnitude faster groupby/joins for millions of edges compared to pandas.
* Runs **graph algorithms on GPU** (cuGraph) — PageRank/centrality computed much faster than CPU graph libs for large graphs.
* Uses **token-aware chunking + batching** for throughput on memory-constrained GPUs.
* Streams progress and performs memory-cleaning to stay within T4 limits.

Net effect: **faster iteration**, **higher throughput**, **reduced CPU-bound bottlenecks**, and **lower cost/latency** relative to remote LLM + CPU-only graph systems.

---

# Files & outputs produced

* The main pipeline file (single-file script) — contains all classes and `run_pipeline()` entrypoint.
* `knowledge_graph.html` — interactive PyVis visualization saved to disk.
* `kg_edges.json` — JSON export of aggregated edge table (Polars JSON).
* `./offload` — directory used for any model offloading (created at runtime).

---

# Quickstart / Usage

Minimal usage (example in script):

```python
from this_module import run_pipeline  # if you rename the file or import accordingly

SAMPLE_TEXT = "Your input text here ..."
network = run_pipeline(SAMPLE_TEXT, output_path="knowledge_graph.html")
```

Or run at command line (if file is `pipeline.py`):

```bash
python pipeline.py
# opens and writes knowledge_graph.html and kg_edges.json
```

Interactive notebook:

* In Colab, run the notebook cell that imports and calls `run_pipeline(SAMPLE_TEXT)`. After completion you can open `knowledge_graph.html` or display with IPython `HTML(...)`.

---

# Configuration options (highlights)

The dataclass `PipelineConfig` controls memory- and behavior-sensitive parameters:

* `model_name` — LLM identifier (default tuned to `nvidia/Nemotron-Mini-4B-Instruct`).
* `max_input_tokens` — token budget per chunk sent to LLM.
* `max_output_tokens` — tokens allowed in LLM output.
* `batch_size` — how many chunks to run per GPU batch (increase for throughput if GPU memory allows).
* `edge_vis_limit` — maximum number of edges marshaled for the browser visualization (prevents browser OOM).
* `edge_batch_limit` — internal batching when constructing edges for GPU ops.
* `top_k_nodes` — limits nodes shown in PyVis.
* `pyvis_height`, `pyvis_width` — visualization size.

Tune these for your GPU. For Colab T4, defaults are chosen conservatively.

---

# Dependencies & installation notes (Colab / T4 tips)

**Core Python packages used**

* `torch` (with CUDA support)
* `transformers`
* `cudf` (RAPIDS)
* `cugraph` (RAPIDS)
* `polars`
* `pyvis`
* `xxhash`
* `tqdm`
* `json`, `re`, `pathlib`, standard library

**Important**: cuDF and cuGraph come from RAPIDS and must be installed with CUDA-compatible packages for your environment. On Colab T4, install RAPIDS with the recommended RAPIDS install script for the appropriate CUDA version (the script/command changes over time). The code expects `torch` + GPU, and cuDF/cuGraph to be available.

**Colab tips**

* Use a runtime with GPU (T4 recommended).
* Install RAPIDS for the matching CUDA version *before* importing cudf/cugraph.
* Ensure `torch` is installed with the same CUDA compatibility.
* `offload` folder created to help with model disk offload if `low_cpu_mem_usage` triggers offload. Keep an eye on disk quota.

Because RAPIDS and CUDA versions are fragile, follow the RAPIDS install docs for the exact colab install snippet. (This code assumes RAPIDS is already available in the runtime.)

---

# Performance, scaling & memory considerations

* **Batching**: Increasing `batch_size` improves GPU throughput but requires more memory. Tune per-GPU.
* **Chunk size** (`max_input_tokens`): Larger chunks reduce LLM call overhead but increase per-call memory. Keep it within model token limits.
* **Entity hashing**: Using stable int64 hashes (xxhash) allows efficient GPU joins and grouping.
* **Aggregation on GPU**: Groupby/aggregation reduces the edge table before moving to CPU—major win for speed and memory.
* **Visualization cap**: The `edge_vis_limit` prevents sending huge graphs to the browser (which would freeze the client).
* **Memory hygiene**: The pipeline calls `torch.cuda.empty_cache()` and `gc.collect()` frequently. These help on constrained GPUs but are not a guarantee against OOM.

---

# Limitations & safety notes

* **Model hallucination**: The LLM may generate incorrect or invented relations. The pipeline includes a confidence scoring rubric, but **human review is recommended** for downstream decisions.
* **Confidence calibration**: The pipeline treats the model-provided `confidence` as guidance. Consider adding post-processing or thresholding to reduce false positives.
* **GPU-only pieces**: cuDF/cuGraph and GPU LLM require compatible GPU and RAPIDS — not trivially reproducible on CPU-only machines.
* **Data privacy**: If you replace the default model with an API-based model, be mindful of data exfiltration. This code favors on-device models.
* **Licensing of models**: Check the model license before using it in production.

---

# Troubleshooting & tips

* **OOM on T4**:

  * Lower `batch_size` and `max_input_tokens`.
  * Set `use_flash_attention = False` (the config includes it as a placeholder).
  * Reduce the model size or use model offloading strategies.
* **RAPIDS import errors**:

  * Ensure RAPIDS/cuDF/cuGraph were installed for the CUDA version in the runtime. In Colab, use the RAPIDS install script for the right CUDA.
* **No triples extracted**:

  * Inspect sample raw output printed in warnings — adjust the model prompt or increase `max_output_tokens`.
  * The TOON prompt is strict: if the LLM doesn’t follow the pattern, the parser may fail. Try relaxing the prompt or parse more flexibly.
* **Slow extraction**:

  * Increase `batch_size` or use a larger GPU. Ensure `torch_dtype=torch.float16` is used (as in the code) for speed/memory improvements.
* **Visualized graph looks odd**:

  * PageRank scores are used to scale node sizes. If nodes overlap, reduce `top_k_nodes` or `edge_vis_limit`.

---

# Extensions & future improvements

* Add **named entity normalization** (linking to Wikidata or internal KB) to reduce duplicate labels.
* Add **confidence recalibration** using an external verifier or rule-based checks.
* Integrate **streaming ingestion** from files, S3, or message queues.
* Export to graph databases (Neo4j/JanusGraph) for persistent storage and query support.
* Replace the single LLM prompt with an ensemble or an extraction model specialized for relation extraction.
* Add unit tests for parsing and GPU aggregation logic.

---

# Example output snippet

Parsed triple (TOON format):

```
TRIPLE source:"Geoffrey Hinton" predicate:"pioneered" target:"deep learning" confidence:0.95
```

Aggregated cuDF row (fields):

* `src_id` (int64 hash)
* `rel_id` (int64 hash)
* `dst_id` (int64 hash)
* `confidence` (mean)
* `source_str`, `predicate_str`, `target_str`
* `count` (how many raw triples collapsed into this aggregated edge)

---

# Contributing

Contributions welcome:

* Raise issues for bugs or feature requests.
* Submit PRs to add tests, CI, or to modularize the script into importable modules.
* If adding models, please include clear licensing for any model weights used.

---

# License

This project is provided under the MIT License (replace as needed). Check model licensing separately before distribution.

---

# A short developer note (non-technical summary)

This pipeline is an engineering-first proof-of-concept that shows how to squeeze the performance advantages of GPUs for the *entire* knowledge-graph construction flow: extraction, canonicalization, and analytics. The biggest wins come from (a) doing heavy grouping/deduplication on the GPU with cuDF instead of pandas, and (b) running graph algorithms with cuGraph. For experimentation, a Colab T4 is a surprisingly capable environment when the pipeline is tuned for memory—hence the many conservative defaults and cleanup steps.

---
