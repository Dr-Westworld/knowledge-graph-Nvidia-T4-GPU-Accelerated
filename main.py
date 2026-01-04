"""
GPU-Accelerated Knowledge Graph Construction Pipeline
Architecture: CPU ingestion → GPU NLP → GPU graph ops → CPU visualization
Target: Google Colab T4 GPU
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import cudf
from tqdm.auto import tqdm
import time
import cugraph
import polars as pl
from pyvis.network import Network
import re
from pathlib import Path
from typing import List, Dict, Tuple, Iterator
import gc
from dataclasses import dataclass
import xxhash
import json

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Memory-aware configuration for T4 GPU"""
    # LLM constraints
    model_name: str = "nvidia/Nemotron-Mini-4B-Instruct"
    max_input_tokens: int = 1400
    max_output_tokens: int = 128
    batch_size: int = 2
    use_flash_attention: bool = False  # Disable for compatibility

    # Graph constraints
    edge_vis_limit: int = 20_000
    edge_batch_limit: int = 5_000

    # Visualization
    top_k_nodes: int = 100
    pyvis_height: str = "800px"
    pyvis_width: str = "100%"

config = PipelineConfig()



# ============================================================================
# STAGE 1: CPU INGESTION & CHUNKING
# ============================================================================

class TextChunker:
    """Token-aware chunking with safe boundaries"""

    def __init__(self, tokenizer, max_tokens: int = 1400):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def chunk_text(self, text: str) -> List[str]:
        """Split text into LLM-safe chunks"""
        # Sentence boundaries for clean splits
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent, add_special_tokens=False))

            if current_length + sent_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sent]
                    current_length = sent_tokens
                else:
                    # Single sentence exceeds limit - force split
                    chunks.append(sent[:self.max_tokens * 4])  # Rough char estimate
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(sent)
                current_length += sent_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

# ============================================================================
# STAGE 2: GPU NLP EXTRACTION
# ============================================================================

class TripleExtractor:
    """GPU-based triple extraction with strict memory control"""

    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls, config: PipelineConfig):
        """Singleton pattern to prevent multiple model loads"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: PipelineConfig):
        # Only initialize once
        if self._model is not None:
            self.config = config
            self.model = self._model
            self.tokenizer = self._tokenizer
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="./offload"  # Fix for disk offload error
        )
        self.model.eval()

        # Cache for reuse
        self._model = self.model
        self._tokenizer = self.tokenizer

        # TOON extraction prompt
        self.system_prompt = """You are a knowledge graph extraction expert. Extract ALL meaningful relationships from the text.

EXTRACTION RULES:
1. Extract entity-relation-entity triples for:
   - Direct relationships (X created Y, A is part of B)
   - Attribute relationships (X has property Y)
   - Temporal relationships (X happened before Y)
   - Hierarchical relationships (X is a type of Y)
   - Association relationships (X related to Y)

2. IMPORTANT: Extract relationships between entities even if not explicitly stated:
   - If "Deep learning uses neural networks" and "Geoffrey Hinton pioneered deep learning"
   - Also extract: "Geoffrey Hinton" → "contributed_to" → "neural networks"

3. Include co-reference resolution:
   - If text says "AI" then "it enables", treat "it" as "AI"

4. Extract granular relationships:
   - Instead of just "X related_to Y", specify HOW (developed, invented, uses, enables, etc.)

OUTPUT FORMAT (one per line):
TRIPLE source:"entity1" predicate:"relationship" target:"entity2" confidence:0.XX

Confidence scoring:
- 0.95-1.0: Explicitly stated fact
- 0.80-0.94: Strongly implied
- 0.59-0.79: Inferred from context
- Below 0.58: Don't include

Extract triples from this text:"""

    def extract_batch(self, chunks: List[str]) -> List[str]:
        """Process batch on GPU, return TOON lines"""
        prompts = [f"{self.system_prompt}\n\nText: {chunk}\n\nTriples:" for chunk in chunks]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_tokens
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_output_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract TOON lines
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        toon_lines = []

        for text in decoded:
            # Extract only the generated part after prompt
            response = text.split("Triples:")[-1].strip()
            toon_lines.extend([line.strip() for line in response.split('\n') if 'TRIPLE' in line])

        # Free GPU memory
        del inputs, outputs
        torch.cuda.empty_cache()

        return toon_lines

# ============================================================================
# STAGE 3: GPU TRIPLE CANONICALIZATION
# ============================================================================

class GPUTripleStore:
    """cuDF-based triple deduplication and aggregation"""

    def __init__(self):
        self.triples_df = None

    @staticmethod
    def parse_toon_line(line: str) -> Dict:
        """Parse TOON format to structured dict"""
        pattern = r'TRIPLE source:"([^"]+)" predicate:"([^"]+)" target:"([^"]+)" confidence:([\d.]+)'
        match = re.search(pattern, line)

        if match:
            return {
                'source': match.group(1).strip(),
                'predicate': match.group(2).strip(),
                'target': match.group(3).strip(),
                'confidence': min(float(match.group(4)), 1.0)  # Cap at 1.0
            }

        # Fallback: try simpler pattern without confidence
        simple_pattern = r'TRIPLE\s+source:\s*["\']?([^"\']+)["\']?\s+predicate:\s*["\']?([^"\']+)["\']?\s+target:\s*["\']?([^"\']+)["\']?'
        simple_match = re.search(simple_pattern, line)
        if simple_match:
            return {
                'source': simple_match.group(1).strip(),
                'predicate': simple_match.group(2).strip(),
                'target': simple_match.group(3).strip(),
                'confidence': 0.8  # Default confidence
            }
        return None



    @staticmethod
    def hash_entity(entity: str) -> int:
        """Stable int64 hash for GPU operations"""
        return xxhash.xxh64(entity.encode('utf-8')).intdigest() & 0x7FFFFFFFFFFFFFFF

    def add_triples(self, toon_lines: List[str]):
        """Parse and add triples to GPU dataframe"""
        parsed = [self.parse_toon_line(line) for line in toon_lines]
        parsed = [t for t in parsed if t is not None]

        if not parsed:
            print(f"Warning: No valid triples parsed from {len(toon_lines)} raw lines")
            if toon_lines:
                print(f"Sample raw output: {toon_lines[0][:200]}")
            return

        # Hash entities
        data = {
            'src_id': [self.hash_entity(t['source']) for t in parsed],
            'rel_id': [self.hash_entity(t['predicate']) for t in parsed],
            'dst_id': [self.hash_entity(t['target']) for t in parsed],
            'confidence': [t['confidence'] for t in parsed],
            'source_str': [t['source'] for t in parsed],
            'predicate_str': [t['predicate'] for t in parsed],
            'target_str': [t['target'] for t in parsed]
        }

        new_df = cudf.DataFrame(data)

        if self.triples_df is None:
            self.triples_df = new_df
        else:
            self.triples_df = cudf.concat([self.triples_df, new_df])

    def aggregate(self) -> cudf.DataFrame:
        """GPU groupby aggregation"""
        if self.triples_df is None:
            return cudf.DataFrame()

        agg_df = self.triples_df.groupby(['src_id', 'rel_id', 'dst_id']).agg({
            'confidence': 'mean',
            'source_str': 'first',
            'predicate_str': 'first',
            'target_str': 'first'
        }).reset_index()

        # Add edge count
        agg_df['count'] = self.triples_df.groupby(['src_id', 'rel_id', 'dst_id']).size().reset_index()[0]

        return agg_df

# ============================================================================
# STAGE 4: GPU GRAPH ANALYTICS
# ============================================================================

class GPUGraphAnalyzer:
    """cuGraph-based graph analytics"""

    def __init__(self, edges_df: cudf.DataFrame):
        self.edges_df = edges_df
        self.graph = None
        self.build_graph()

    def build_graph(self):
        """Construct cuGraph from edge list"""
        self.graph = cugraph.Graph(directed=True)
        self.graph.from_cudf_edgelist(
            self.edges_df,
            source='src_id',
            destination='dst_id',
            edge_attr='confidence'
        )

    def compute_metrics(self) -> Dict[str, cudf.DataFrame]:
        """Run GPU graph algorithms"""
        metrics = {}

        # PageRank
        metrics['pagerank'] = cugraph.pagerank(self.graph)

        # Degree centrality
        metrics['in_degree'] = self.graph.in_degree()
        metrics['out_degree'] = self.graph.out_degree()

        return metrics

# ============================================================================
# STAGE 5: CPU PROJECTION & VISUALIZATION
# ============================================================================

class PyVisAdapter:
    """Convert cuDF graph to PyVis for local visualization"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def create_network(
        self,
        edges_df: cudf.DataFrame,
        metrics: Dict[str, cudf.DataFrame]
    ) -> Network:
        """Build interactive PyVis network"""

        # Convert to Polars for CPU-side ops (Arrow zero-copy)
        edges_pl = pl.from_arrow(edges_df.to_arrow())

        # Limit visualization size
        if len(edges_pl) > self.config.edge_vis_limit:
            print(f"Limiting visualization to top {self.config.edge_vis_limit} edges by confidence")
            edges_pl = edges_pl.sort('confidence', descending=True).head(self.config.edge_vis_limit)

        # Merge PageRank scores
        pr_pl = pl.from_arrow(metrics['pagerank'].to_arrow())
        pr_dict = dict(zip(pr_pl['vertex'].to_list(), pr_pl['pagerank'].to_list()))

        # Initialize PyVis
        net = Network(
            height=self.config.pyvis_height,
            width=self.config.pyvis_width,
            directed=True,
            notebook=True,
            cdn_resources='in_line'  # Changed from default
        )
        net.barnes_hut()

        # Add nodes with PageRank sizing
        unique_nodes = set(edges_pl['src_id'].to_list() + edges_pl['dst_id'].to_list())
        node_labels = {}

        for row in edges_pl.iter_rows(named=True):
            node_labels[row['src_id']] = row['source_str']
            node_labels[row['dst_id']] = row['target_str']

        for node_id in unique_nodes:
            pr_score = pr_dict.get(node_id, 0.001)
            size = 10 + (pr_score * 1000)  # Scale for visibility

            net.add_node(
                node_id,
                label=node_labels.get(node_id, str(node_id)),
                size=size,
                title=f"PageRank: {pr_score:.4f}"
            )

        # Add edges
        for row in edges_pl.iter_rows(named=True):
            net.add_edge(
                row['src_id'],
                row['dst_id'],
                title=row['predicate_str'],
                label=row['predicate_str'],
                value=float(row['confidence'])
            )

        return net

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class StreamableKGPipeline:
    """Main orchestrator with streaming support"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.extractor = TripleExtractor(config)
        self.chunker = TextChunker(self.extractor.tokenizer, config.max_input_tokens)
        self.triple_store = GPUTripleStore()

    def process_text(self, text: str) -> Iterator[Dict]:
        """Stream processing with yield for progress"""
        chunks = self.chunker.chunk_text(text)
        total_chunks = len(chunks)

        print(f"Processing {total_chunks} chunks...")

        # Progress bar for batches
        num_batches = (total_chunks + self.config.batch_size - 1) // self.config.batch_size
        pbar = tqdm(total=num_batches, desc="Extracting triples", unit="batch")

        start_time = time.time()

        # Batch processing
        for i in range(0, total_chunks, self.config.batch_size):
            batch_start = time.time()
            batch = chunks[i:i + self.config.batch_size]

            # GPU extraction
            toon_lines = self.extractor.extract_batch(batch)
            self.triple_store.add_triples(toon_lines)

            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            remaining_batches = num_batches - (i // self.config.batch_size + 1)
            eta = remaining_batches * (elapsed / (i // self.config.batch_size + 1))

            pbar.set_postfix({
                'triples': len(toon_lines),
                'batch_time': f'{batch_time:.1f}s',
                'ETA': f'{eta:.0f}s'
            })
            pbar.update(1)

            yield {
                'progress': (i + len(batch)) / total_chunks,
                'triples_extracted': len(toon_lines),
                'batch': i // self.config.batch_size + 1,
                'eta_seconds': eta
            }

            # Memory cleanup
            gc.collect()

        pbar.close()
        print(f"✓ Extraction complete in {time.time() - start_time:.1f}s")

    def finalize_graph(self) -> Tuple[cudf.DataFrame, Dict, Network]:
        """Aggregate and analyze graph"""
        print("Aggregating triples on GPU...")
        edges_df = self.triple_store.aggregate()

        print(f"Graph: {len(edges_df)} unique edges")

        if len(edges_df) == 0:
            raise ValueError("No valid triples extracted")

        print("Running graph analytics on GPU...")
        analyzer = GPUGraphAnalyzer(edges_df)
        metrics = analyzer.compute_metrics()

        print("Creating visualization...")
        adapter = PyVisAdapter(self.config)
        network = adapter.create_network(edges_df, metrics)

        return edges_df, metrics, network

# ============================================================================
# EXECUTION
# ============================================================================

def run_pipeline(input_text: str, output_path: str = "knowledge_graph.html"):
    """Execute complete pipeline"""
    overall_start = time.time()

    print("=" * 60)
    print("KNOWLEDGE GRAPH PIPELINE")
    print("=" * 60)
    import os
    os.makedirs("./offload", exist_ok=True)
    # Clean memory from previous runs
    cleanup_memory()
    pipeline = StreamableKGPipeline(config)

    # Stream processing with progress
    for status in pipeline.process_text(input_text):
        pass  # Progress shown by tqdm

    print("\n" + "=" * 60)
    print("FINALIZING GRAPH")
    print("=" * 60)

    # Finalize with timing
    finalize_start = time.time()
    edges_df, metrics, network = pipeline.finalize_graph()
    print(f"✓ Finalization took {time.time() - finalize_start:.1f}s")


    # Save visualization
    network.show(output_path)
    print(f"\n✓ Knowledge graph saved to {output_path}")

    # Export data
    edges_pl = pl.from_arrow(edges_df.to_arrow())
    edges_pl.write_json("kg_edges.json")
    print(f"✓ Edge data exported to kg_edges.json")
    # Clean memory from previous runs
    cleanup_memory()
    return network

# ============================================================================
# SAMPLE INPUT
# ============================================================================

SAMPLE_TEXT = """
Artificial intelligence is transforming modern computing. Machine learning algorithms
enable computers to learn from data without explicit programming. Deep learning,
a subset of machine learning, uses neural networks with multiple layers to model
complex patterns. Geoffrey Hinton pioneered deep learning research at the University
of Toronto. His work on backpropagation revolutionized neural network training.

Natural language processing allows machines to understand human language. BERT,
developed by Google, uses transformer architecture for language understanding.
OpenAI created GPT models that generate human-like text. These models are trained
on massive datasets using powerful GPUs. NVIDIA produces the GPUs that enable
large-scale AI training.

Computer vision enables machines to interpret visual information. Convolutional
neural networks excel at image recognition tasks. ImageNet, a large dataset created
by Fei-Fei Li at Stanford, accelerated computer vision research. Self-driving cars
use computer vision to navigate roads safely. Tesla implements neural networks
for autonomous driving.
"""

if __name__ == "__main__":
    # Run pipeline
    network = run_pipeline(SAMPLE_TEXT)

    # Read and display HTML directly
    with open('knowledge_graph.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    from IPython.display import HTML, display
    display(HTML(html_content))
