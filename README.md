# GPU-Accelerated Knowledge Graph Construction Pipeline
![Alt image](https://github.com/Dr-Westworld/knowledge-graph-Nvidia-T4-GPU-Accelerated/blob/main/gpu_kg_pipeline_architecture.svg)
A GPU-first, memory-aware pipeline that converts unstructured text into an interactive knowledge graph using on-device LLM extraction, GPU-based canonicalization, and GPU graph analytics.  
Optimized for **Google Colab T4**–class GPUs.

**Architecture:** CPU ingestion → GPU NLP → GPU graph ops → CPU visualization

---

## Why this exists

Traditional knowledge graph pipelines are CPU-bound, slow to scale, and often rely on external LLM APIs.  
This project demonstrates how to build an **end-to-end knowledge graph pipeline that stays on the GPU for all heavy work**, enabling faster extraction, aggregation, and graph analytics at lower cost.

---

## Key benefits over a normal knowledge graph pipeline

- **GPU-based LLM extraction**  
  Entity–relation–entity triples are extracted locally on the GPU using an instruction-tuned LLM (no external APIs).

- **Token-aware chunking**  
  Text is split safely by token count and sentence boundaries, maximizing throughput while avoiding model overflows.

- **GPU canonicalization & deduplication**  
  Triples are hashed and aggregated using cuDF, collapsing duplicates and averaging confidence scores entirely on the GPU.

- **GPU graph analytics**  
  PageRank and degree centrality are computed using cuGraph, scaling far beyond CPU graph libraries.

- **Memory-aware design**  
  Batching, cleanup, and offloading are tuned to run reliably on constrained GPUs like Colab T4.

- **Interactive visualization**  
  Results are projected back to the CPU and rendered as an interactive PyVis HTML graph.

---

## Pipeline stages

1. **CPU Ingestion & Chunking**  
   Token-aware sentence chunking to keep LLM inputs within limits.

2. **GPU NLP Extraction**  
   Batched LLM inference produces structured triples in a strict TOON format.

3. **GPU Triple Canonicalization**  
   cuDF is used to hash entities, deduplicate edges, and aggregate confidence scores.

4. **GPU Graph Analytics**  
   cuGraph runs PageRank and degree calculations directly on the GPU.

5. **CPU Visualization**  
   A size-limited subgraph is rendered using PyVis for interactive exploration.

---

## Output

- **knowledge_graph.html** — Interactive knowledge graph visualization  
- **kg_edges.json** — Aggregated edge data (source, predicate, target, confidence, count)

---

## Example extracted triple
-- TRIPLE source:"Geoffrey Hinton" predicate:"pioneered" target:"deep learning" confidence:0.95


---

## Configuration highlights

Key parameters (via `PipelineConfig`):

- `model_name` — LLM used for extraction  
- `batch_size` — GPU batch size for extraction  
- `max_input_tokens` / `max_output_tokens` — LLM token limits  
- `edge_vis_limit` — Caps graph size for browser visualization  

Defaults are tuned for Colab T4 GPUs.

---

## Troubleshooting

**OOM on T4**
- Lower `batch_size` and `max_input_tokens`
- Use a smaller model or enable model offloading
- Ensure `torch_dtype=torch.float16` is enabled

**RAPIDS import errors**
- Install cuDF/cuGraph for the exact CUDA version of the runtime
- Use the official RAPIDS Colab install script

**No triples extracted**
- Increase `max_output_tokens`
- Relax the extraction prompt or parsing logic if the model output deviates

**Visualization looks cluttered**
- Reduce `edge_vis_limit` or `top_k_nodes`

---

## Future improvements

- Entity normalization (Wikidata / internal KB linking)
- Confidence recalibration or verification
- Streaming ingestion from files or message queues
- Export to graph databases (Neo4j, JanusGraph)
- Dedicated relation-extraction models
- Unit tests for parsing and GPU aggregation
- rmm using for memory management(import rmm
                                  from rmm.allocators.torch import rmm_torch_allocator
                                  from rmm.allocators.cupy import rmm_cupy_allocator
                                  )
---

## Contributing

- Open issues for bugs or feature requests  
- Pull requests welcome (tests and modularization encouraged)  
- Clearly document model licenses when adding new models

---

## License

MIT License (update if needed).  
Model weights and datasets are subject to their respective licenses.
