
# ============================================================================
# PDF -> Knowledge Graph pipeline, vLLM-backed.
#
# Maps onto the architecture in index.html as follows -- search for these
# tags in comments below to find the corresponding code:
#
#   [PROCESS A]   main pipeline runtime -- THIS Colab kernel: RAPIDS
#                 (cudf/cugraph/cuml/rmm) + orchestration. Never imports
#                 vllm directly (see [PROCESS B] below for why).
#   [OCR/LAYOUT]  extract_sections_from_pdf
#   [CHUNK]       TextChunker
#   [CLASSIFY]    DomainClassifier / DEFAULT_DOMAIN_LABELS
#   [ONTOLOGY]    ALLOWED_ENTITY_TYPES / ALLOWED_PREDICATES / ONTOLOGY_PROFILES
#   [PROCESS B]   vLLM inference server -- ISOLATED venv, localhost:<port>,
#                 OpenAI-compatible HTTP API. launch_vllm_server /
#                 ensure_vllm_server / VLLMTripleExtractor.
#   [VALIDATE]    parse_json_triples(strict=True) / sitl_filter /
#                 INVALID_TYPE_PREDICATE_COMBOS / VALID_SUBJECT_TYPES_PER_PREDICATE
#   [RESOLVE]     resolve_entities_in_edges / resolve_canonical_types
#   [ANN]         SemanticEntityResolver / CuMLEntityResolver / apply_ann_resolution
#   [GRAPH]       GPUGraphAnalyzer (PageRank, in/out-degree, Leiden communities)
#   [STORE]       GPUDirectStorage / export_to_parquet
#   [OBSERVE]     LangfuseTracer (LLM traces/prompts/generations) and
#                 VLLMMetricsPoller (vLLM's native /metrics -> same W&B run
#                 the pipeline already logs batch/graph metrics to -- this
#                 is the direct swap for "vLLM metrics -> Prometheus ->
#                 Grafana" in the original diagram; W&B also auto-collects
#                 host/GPU system metrics on wandb.init() with zero extra
#                 code, covering the "GPU / experiment metrics" column).
#   [ACTIVE-LEARN] VLLMDPODatasetBuilder / run_dpo_training / DSPy section
#                 near the bottom -- optional, not required for the main
#                 pipeline to run; adapted from the transformers-direct
#                 version to call the vLLM HTTP endpoint instead of a
#                 local model.generate().
#
# [PROCESS B] WHY THE ISOLATED VENV:
# vLLM pins its own torch/CUDA/numba stack, which does not have to match
# what RAPIDS (cudf/cugraph/cuml/rmm) needs pinned in THIS process. Pip
# wheels named nvidia-*-cuXX bundle their own .so files and are loaded via
# dlopen from each package's own site-packages -- they do not touch a
# system-wide CUDA install and do not, by themselves, conflict with a
# different CUDA-flavoured stack in a *different* Python environment.
# numba is the one that actually clashes if both stacks land in the SAME
# interpreter: RAPIDS needs numba pinned to whatever its current release
# requires (0.60.x for the 26.2.* RAPIDS release used below), vLLM's own
# dependency tree wants a different numba, and numba's CUDA target is
# sensitive enough to version skew that "just let pip figure it out" in
# one shared environment reliably breaks one side or the other -- this was
# reproduced firsthand across this project's install attempts. A venv
# fully isolates site-packages (and therefore which numba/torch import),
# while `$HOME` -- and so `~/.cache/huggingface` -- is NOT venv-isolated,
# so a model downloaded from the main kernel is immediately visible to the
# vLLM venv's subprocess and vice versa. That's what the pre-download step
# in the notebook relies on.
#
# Also worth knowing before you tune `vllm_gpu_memory_utilization`: the
# vLLM server subprocess and this main kernel's RAPIDS code share the SAME
# physical GPU (isolation above is at the Python-environment level, not
# the hardware level). Whatever fraction you give vLLM is unavailable to
# cudf/cugraph/cuml -- on a 16GB T4 the default below (0.45, ~7.2GB) is
# sized to leave headroom for RAPIDS' own allocations (cuDF frames,
# cuGraph's graph structures, cuML's ANN index) rather than to maximise
# vLLM's KV cache.
# ============================================================================

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer
import cudf
import cugraph
import polars as pl
import re
import gc
import json
import time
import threading
import subprocess
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator, Optional, Set, Callable, Literal
from pydantic import BaseModel, Field, ValidationError
import xxhash
import numpy as np
import fitz
import pymupdf
import ipywidgets as widgets
from IPython.display import display, FileLink
from rapidfuzz import fuzz
from rapidfuzz.process import cdist as rfcdist
import Levenshtein as lev
from tqdm.auto import tqdm
import wandb
warnings.filterwarnings('ignore')

# NOTE: no `import weave` here on purpose. The files this pipeline started
# from imported weave at the top but never called it anywhere (no
# weave.init() / @weave.op() in the original source) -- a dead import. It
# would also duplicate what Langfuse is doing below: your stated split
# puts LLM traces/prompts/generations in Langfuse and experiments/GPU
# metrics/eval/datasets in W&B, so adding W&B's own LLM-tracing layer on
# top would just be two tools doing the same job.

RMM_AVAILABLE = False
try:
    import rmm
    from rmm.allocators.torch import rmm_torch_allocator
    import cupy as cp
    RMM_AVAILABLE = True
except ImportError:
    import cupy as cp

# Split into two independent flags rather than one combined "ANN_AVAILABLE"
# -- the cuML backend (the new default here) needs sentence-transformers
# for embedding but NOT faiss at all, so a box that has cuml +
# sentence-transformers but genuinely skipped installing faiss (a
# reasonable choice: it's a second, CPU-side ANN library this pipeline
# only needs as a fallback) should still get domain classification and
# cuML-backed ANN resolution, not have both silently disabled.
EMBEDDER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDER_AVAILABLE = True
except ImportError:
    pass

FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass

ANN_AVAILABLE = EMBEDDER_AVAILABLE and FAISS_AVAILABLE  # only the FAISS-specific code path needs this combined form

CUML_AVAILABLE = False
try:
    from cuml.neighbors import NearestNeighbors as _CuMLNearestNeighbors
    CUML_AVAILABLE = True
except ImportError:
    pass

DSPY_AVAILABLE = False
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    pass

LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class PipelineConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_input_tokens: int = 900
    max_output_tokens: int = 768
    batch_size: int = 8
    overlap_pct: float = 0.20
    entity_resolution_threshold: float = 0.82
    leiden_resolution: float = 1.2
    confidence_filter: float = 0.4
    debug: bool = False

    # StreamableKGPipeline._process_chunks' low-yield rescue path samples
    # a chunk this many times and keeps a triple only if _consensus_from_runs'
    # Beta(2,1)-prior, 95%-lower-confidence-bound vote clears 0.5. n=3 is
    # the minimum where unanimous agreement clears the bar (lower bound
    # 0.549) -- n=2 cannot, even at perfect 2/2 agreement (lower bound
    # 0.473 < 0.5), so this mechanism would silently reject every triple
    # forever at n=2. Verified numerically, not just asserted.
    bayesian_rerun_samples: int = 3

    # --- [PROCESS B] vLLM inference backend ---
    # StreamableKGPipeline extracts through VLLMTripleExtractor, an
    # OpenAI-compatible HTTP client talking to a vLLM server launched (see
    # launch_vllm_server) in an ISOLATED venv/subprocess. See the module
    # docstring above for why. Requires the venv already set up:
    #   python3 -m venv /content/vllm_env && /content/vllm_env/bin/pip install vllm
    vllm_venv_path: str = "/content/vllm_env"
    vllm_num_speculative_tokens: int = 5
    vllm_gpu_memory_utilization: float = 0.45
    vllm_port: int = 8000
    vllm_quantization: str = "awq"

    # "ngram": no draft model at all -- speculates by matching candidate
    #   tokens directly against the prompt. Chosen as the DEFAULT here
    #   (the transformers-direct/vLLM-v2 predecessors of this file
    #   defaulted to "draft_model") for three concrete reasons on THIS
    #   hardware/task specifically: (1) it needs no second model, saving
    #   ~1GB of VRAM and ~1GB of disk/download on a box with 16GB VRAM and
    #   12GB system RAM to share with RAPIDS; (2) extracted entities are
    #   frequently literal substrings of the source chunk, which is
    #   exactly the pattern n-gram speculation is good at; (3) it
    #   sidesteps needing the draft-model vocabulary/loading path
    #   entirely, which is one fewer thing that can fail at server
    #   startup. "draft_model" (using draft_model_name below, same Qwen2.5
    #   family so vocab is shared) is left in as a fully wired option if
    #   you want to A/B it -- just flip this string.
    vllm_speculative_method: str = "ngram"  # "ngram" | "draft_model"

    # --- extraction quality gates [VALIDATE] ---
    strict_field_validation: bool = True
    max_entity_chars: int = 80
    max_entity_words: int = 7
    max_triples_per_chunk: int = 20
    repetition_penalty: float = 1.1

    # --- confidence calibration ---
    # VLLMTripleExtractor computes confidence as exp(avg_logprob * this
    # scale) -- geometric-mean token probability raised to a tunable
    # power -- from vLLM's completions API `logprobs` field. A guess, not
    # a calibrated constant; see fit_confidence_calibration() to derive it
    # from a labeled sample instead.
    logprob_calibration_scale: float = 8.0

    # --- [ANN] semantic entity resolution ---
    enable_ann_resolution: bool = True
    ann_backend: str = "cuml"  # "cuml" | "faiss" -- cuml is the default here
    # to match the "T4 / GPU ANN" node in the architecture diagram: search
    # runs on-GPU in the same CUDA context as cudf/cugraph rather than
    # round-tripping embeddings to host memory the way FAISS-CPU would.
    # apply_ann_resolution auto-falls-back to whichever backend IS
    # importable if the configured one isn't, rather than silently
    # skipping ANN resolution outright.
    ann_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ann_index_type: str = "flat"  # "flat" | "hnsw" (faiss backend only)
    ann_duplicate_threshold: float = 0.90
    ann_missing_edge_threshold: float = 0.78
    ann_max_proposed_edges: int = 5000

    # --- [CLASSIFY] domain classification ---
    # Honest scope note: this pipeline still extracts against ONE fixed
    # ontology (ALLOWED_ENTITY_TYPES / ALLOWED_PREDICATES below) -- this
    # classifier does not yet switch the grammar or type vocabulary per
    # domain. What it adds is a per-chunk domain TAG threaded into every
    # Langfuse trace and W&B log line, so you can see which domains a
    # document actually contains before investing in a second ontology
    # profile. ONTOLOGY_PROFILES below is a real (if currently
    # single-entry) extension point for when you want that.
    enable_domain_classification: bool = True

    # --- [OBSERVE] ---
    enable_wandb: bool = True
    enable_langfuse: bool = True
    wandb_project: str = "kg-pipeline"
    vllm_metrics_poll_interval_s: float = 15.0


config = PipelineConfig()

# [ONTOLOGY] Mirrors the TYPES / PREDICATES lists inside
# EXTRACTION_SYSTEM_PROMPT. Keep these two in sync if you ever edit the
# prompt's lists.
ALLOWED_ENTITY_TYPES: Set[str] = {
    "PERSON", "ORGANIZATION", "CONCEPT", "PRODUCT", "LOCATION", "EVENT",
    "LAW", "METRIC", "PROCESS", "DOCUMENT", "TECHNOLOGY", "PROVISION",
}

ALLOWED_PREDICATES: Set[str] = {
    "is_a", "instance_of", "part_of", "contains", "includes",
    "develops", "authors", "issues", "regulates", "manages", "governs",
    "performs", "enables", "causes", "requires", "produces", "applies_to",
    "affiliated_with", "located_in", "related_to", "influences", "depends_on",
}


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_sections_from_pdf(pdf_bytes: bytes) -> List[Tuple[str, str]]:
    """[OCR/LAYOUT] stage. This is PyMuPDF's embedded-text extraction, not
    an OCR engine -- for a normal text-layer PDF (reports, filings,
    papers) this is fast and accurate. For a SCANNED/image-only PDF with
    no text layer, per-page text comes back empty and the warning below
    fires; true OCR (e.g. shelling out to Tesseract, or PyMuPDF's own
    get_textpage_ocr) is a real extension point but isn't wired in here to
    keep the dependency footprint down on a 112GB-disk / 12GB-RAM box --
    flagging that explicitly rather than silently handing back a
    near-empty graph."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    toc = doc.get_toc()
    sections = []

    if toc:
        top_level = [(lvl, title, pg) for lvl, title, pg in toc if lvl <= 2]
        for i, (lvl, title, pg) in enumerate(top_level):
            end_pg = top_level[i + 1][2] if i + 1 < len(top_level) else len(doc) + 1
            text = "\n".join(
                doc[p].get_text()
                for p in range(max(0, pg - 1), min(end_pg - 1, len(doc)))
            )
            if len(text.strip()) >= 80:
                sections.append((title, text))

    if not sections:
        block = 8
        for start in range(0, len(doc), block):
            text = "\n".join(
                doc[p].get_text()
                for p in range(start, min(start + block, len(doc)))
            )
            if len(text.strip()) >= 80:
                sections.append((f"pages_{start + 1}_{min(start + block, len(doc))}", text))

    doc.close()

    total_chars = sum(len(t) for _, t in sections)
    if sections and total_chars / max(len(sections), 1) < 200:
        print(
            "  WARNING: extracted text looks very sparse "
            f"({total_chars} chars across {len(sections)} section(s)). "
            "This usually means the PDF is scanned/image-only with no "
            "text layer -- PyMuPDF text extraction can't help there. "
            "OCR (not implemented in this pipeline) would be the fix."
        )
    return sections


# ============================================================================
# [VALIDATE] FIELD VALIDATION -- the "Ontology engine" box in the diagram
#
# CHANGED from a pipe-delimited "S:...|SP:...|P:...|T:...|TP:..." format
# constrained by one big regex to JSON-schema-constrained output validated
# with Pydantic. Reason: the pipe-delimited grammar's regex (12 types
# alternated twice per line + 21 predicates once + two {2,80} char-class
# repeats, wrapped in an outer {0,40} repeat) blew past vLLM/xgrammar's
# 5s regex-compile timeout in practice --
#   ValueError: Regex compilation timed out after 5s. The pattern may be
#   too complex or contain constructs that cause exponential state-space
#   explosion (e.g. nested quantifiers).
# -- a confirmed failure, not a guess (reproduced against a live server,
# traceback landed in backend_xgrammar.py's compile_regex_with_timeout).
# JSON-schema-guided decoding is xgrammar's best-trodden, most-optimized
# path (unlike regex support, which lags behind); enums for
# subject_type/predicate/target_type express the SAME ontology constraint
# a bare regex alternation did, and "triples": {"type": "array", "maxItems":
# N} expresses the same "0 to N triples" bound the outer {0,N} repeat did,
# without the nested-quantifier shape that timed out the regex compiler.
# ============================================================================

_CITATION_RE = re.compile(r'\(\d{4}\)')
_MULTI_SENTENCE_RE = re.compile(r'[.!?;][^.!?;]*[.!?;]')


def _valid_entity_text(text: str, max_chars: int, max_words: int) -> bool:
    """Semantic checks JSON schema can't express by itself: schema-level
    maxLength bounds total characters, but not word count, and can't
    reject "looks like a citation" or "looks like more than one sentence"
    -- those still need this pass after Pydantic's shape/enum validation
    succeeds."""
    text = text.strip()
    if not (1 < len(text) <= max_chars):
        return False
    n_words = len(re.split(r'[\s_]+', text.strip('_')))
    if n_words > max_words:
        return False
    if _CITATION_RE.search(text):
        return False
    if _MULTI_SENTENCE_RE.search(text):
        return False
    return True


# Built directly from the SAME sets used everywhere else in this file
# (ALLOWED_ENTITY_TYPES / ALLOWED_PREDICATES) so there's one source of
# truth for the ontology -- Literal[tuple(...)] is equivalent to writing
# Literal["A", "B", ...] by hand; typing unpacks a single tuple argument
# into individual literal members.
_EntityTypeLiteral = Literal[tuple(sorted(ALLOWED_ENTITY_TYPES))]
_PredicateLiteral = Literal[tuple(sorted(ALLOWED_PREDICATES))]


class TripleModel(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    subject_type: _EntityTypeLiteral
    predicate: _PredicateLiteral
    target: str = Field(min_length=1, max_length=200)
    target_type: _EntityTypeLiteral


class TripleListModel(BaseModel):
    triples: List[TripleModel] = Field(default_factory=list)


def build_triple_json_schema(max_entity_chars: int = 80, max_triples_per_chunk: int = 20) -> dict:
    """Replaces build_triple_grammar_regex(). Hand-built as a plain dict
    (rather than via TripleListModel.model_json_schema()) so max_entity_chars
    / max_triples_per_chunk -- which come from PipelineConfig, i.e. can
    change after this module is imported -- land directly in the schema's
    minLength/maxLength/maxItems instead of being baked into a static
    Pydantic class at import time."""
    entity_type_enum = sorted(ALLOWED_ENTITY_TYPES)
    predicate_enum = sorted(ALLOWED_PREDICATES)
    triple_schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "minLength": 1, "maxLength": max_entity_chars},
            "subject_type": {"type": "string", "enum": entity_type_enum},
            "predicate": {"type": "string", "enum": predicate_enum},
            "target": {"type": "string", "minLength": 1, "maxLength": max_entity_chars},
            "target_type": {"type": "string", "enum": entity_type_enum},
        },
        "required": ["subject", "subject_type", "predicate", "target", "target_type"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "triples": {"type": "array", "maxItems": max_triples_per_chunk, "items": triple_schema},
        },
        "required": ["triples"],
        "additionalProperties": False,
    }


def parse_json_triples(
    text: str,
    strict: bool = True,
    max_entity_chars: int = 80,
    max_entity_words: int = 7,
    stats: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, str, str, str, str, float]]:
    """Parses+validates a `{"triples": [...]}` JSON completion (see
    build_triple_json_schema). Pydantic (via TripleListModel) enforces
    JSON shape + enum membership -- the same job _valid_type_token used to
    do by regex on the pipe-delimited format; strict=True additionally
    applies _valid_entity_text's semantic checks (word count, citation
    pattern, multi-sentence) that schema validation alone can't express."""
    if stats is None:
        stats = {}

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        stats['dropped_json_parse_error'] = stats.get('dropped_json_parse_error', 0) + 1
        return []

    try:
        parsed = TripleListModel(**raw) if isinstance(raw, dict) else TripleListModel(triples=raw)
    except ValidationError:
        stats['dropped_schema_invalid'] = stats.get('dropped_schema_invalid', 0) + 1
        return []

    triples = []
    for t in parsed.triples:
        s, sp, p, tgt, tp = t.subject.strip(), t.subject_type, t.predicate, t.target.strip(), t.target_type
        if not (s and tgt):
            stats['dropped_missing_field'] = stats.get('dropped_missing_field', 0) + 1
            continue
        if strict:
            if not (_valid_entity_text(s, max_entity_chars, max_entity_words)
                    and _valid_entity_text(tgt, max_entity_chars, max_entity_words)):
                stats['dropped_entity_shape'] = stats.get('dropped_entity_shape', 0) + 1
                continue
        triples.append((s, sp, p, tgt, tp, 0.0))

    return triples


PRED_CANON: Dict[str, str] = {
    "was_developed_by": "develops", "created_by": "develops", "built_by": "develops",
    "invented_by": "develops", "developed_by": "develops",
    "works_at": "affiliated_with", "employed_by": "affiliated_with",
    "works_for": "affiliated_with", "member_of": "affiliated_with",
    "is_a": "is_a", "is_an": "is_a", "type_of": "is_a",
    "instance_of": "instance_of",
    "is_part_of": "part_of", "belongs_to": "part_of",
    "uses": "performs", "employs": "performs", "utilizes": "performs",
    "regulated_by": "regulates", "managed_by": "manages", "governed_by": "governs",
    "issued_by": "issues", "authored_by": "authors",
}


def normalize_predicate(pred: str) -> str:
    key = pred.lower().strip().replace(" ", "_")
    return PRED_CANON.get(key, key)


INVALID_TYPE_PREDICATE_COMBOS: Set[Tuple[str, str, str]] = {
    ("METRIC", "develops", "PERSON"), ("METRIC", "regulates", "PERSON"),
    ("LOCATION", "regulates", "TECHNOLOGY"), ("METRIC", "affiliated_with", "ORGANIZATION"),
    ("EVENT", "develops", "CONCEPT"), ("METRIC", "performs", "PROCESS"),
    ("DOCUMENT", "causes", "PERSON"), ("PERSON", "part_of", "METRIC"),
    ("METRIC", "located_in", "PERSON"), ("TECHNOLOGY", "issues", "METRIC"),
    ("LAW", "affiliated_with", "METRIC"), ("METRIC", "governs", "ORGANIZATION"),
    ("LOCATION", "produces", "PERSON"), ("METRIC", "authors", "DOCUMENT"),
    ("METRIC", "develops", "TECHNOLOGY"), ("LOCATION", "requires", "PERSON"),
}

VALID_SUBJECT_TYPES_PER_PREDICATE: Dict[str, Set[str]] = {
    "regulates":       {"ORGANIZATION", "LAW", "PROVISION", "DOCUMENT"},
    "develops":        {"ORGANIZATION", "PERSON", "TECHNOLOGY"},
    "issues":          {"ORGANIZATION", "PERSON", "LAW"},
    "affiliated_with": {"PERSON", "ORGANIZATION"},
    "located_in":      {"PERSON", "ORGANIZATION", "EVENT", "LOCATION"},
    "manages":         {"PERSON", "ORGANIZATION"},
    "governs":         {"ORGANIZATION", "LAW", "PROVISION"},
    "authors":         {"PERSON", "ORGANIZATION"},
}


def sitl_filter(triples: List[Tuple], stats: Optional[Dict[str, int]] = None) -> List[Tuple]:
    """[VALIDATE] Domain/range-style constraint checking on top of the
    type enum: disallowed (subject_type, predicate, target_type) combos,
    predicate-specific valid subject types, minimum entity length,
    self-loop rejection. Re-checks SP/TP against the type enum defensively
    (normally already guaranteed upstream by parse_json_triples(strict=True),
    but keeps this function correct on its own if ever called with
    unvalidated input)."""
    if stats is None:
        stats = {}
    valid = []
    for triple in triples:
        s, st, p, t, tt, c = triple
        if st not in ALLOWED_ENTITY_TYPES or tt not in ALLOWED_ENTITY_TYPES:
            stats['dropped_sitl_type_enum'] = stats.get('dropped_sitl_type_enum', 0) + 1
            continue
        if (st, p, tt) in INVALID_TYPE_PREDICATE_COMBOS:
            stats['dropped_sitl_combo'] = stats.get('dropped_sitl_combo', 0) + 1
            continue
        if p in VALID_SUBJECT_TYPES_PER_PREDICATE and st not in VALID_SUBJECT_TYPES_PER_PREDICATE[p]:
            stats['dropped_sitl_subject_type'] = stats.get('dropped_sitl_subject_type', 0) + 1
            continue
        if len(s.strip()) < 2 or len(t.strip()) < 2:
            stats['dropped_sitl_too_short'] = stats.get('dropped_sitl_too_short', 0) + 1
            continue
        if s.lower() == t.lower():
            stats['dropped_sitl_self_loop'] = stats.get('dropped_sitl_self_loop', 0) + 1
            continue
        valid.append(triple)
    return valid


# ============================================================================
# [RESOLVE] CANONICAL ENTITY IDENTITY
#
# Identity is a function of the normalized label ONLY (not label+type --
# type is a noisy per-extraction LLM guess that can vary chunk to chunk
# for the same real-world entity). Type is tracked as a separate, votable
# attribute reconciled by majority vote in resolve_canonical_types(), so
# one entity = one node = one type in the export.
# ============================================================================

_LEADING_ARTICLE_RE = re.compile(r'^(the|a|an)\s+', re.IGNORECASE)
_WHITESPACE_RE = re.compile(r'\s+')


def normalize_entity_label(label: str) -> str:
    """Used ONLY for bucketing/identity -- never for display."""
    x = label.strip().lower()
    x = _WHITESPACE_RE.sub(' ', x)
    x = _LEADING_ARTICLE_RE.sub('', x)
    return x.strip(' .,;:')


def canonical_entity_id(label: str) -> int:
    norm = normalize_entity_label(label)
    return xxhash.xxh64(norm.encode()).intdigest() & 0x7FFFFFFFFFFFFFFF


def compute_entity_similarity(a: str, b: str) -> float:
    al, bl = a.lower().strip(), b.lower().strip()
    set_a, set_b = set(al.split()), set(bl.split())
    union = set_a | set_b
    jaccard = len(set_a & set_b) / len(union) if union else 0.0
    max_len = max(len(al), len(bl))
    levenshtein = 1.0 - (lev.distance(al, bl) / max_len) if max_len > 0 else 0.0
    return (0.4 * jaccard) + (0.6 * levenshtein)


def resolve_canonical_types(edges_pl: pl.DataFrame) -> pl.DataFrame:
    """Reconciles (label -> type) inconsistencies: collapses each
    canonical id down to a single majority-vote type and majority-vote
    surface form, instead of letting the id fragment silently."""
    obs = pl.concat([
        edges_pl.select([
            pl.col("src_id").alias("id"), pl.col("source_type").alias("type"),
            pl.col("source_str").alias("label"),
        ]),
        edges_pl.select([
            pl.col("dst_id").alias("id"), pl.col("target_type").alias("type"),
            pl.col("target_str").alias("label"),
        ]),
    ])

    type_votes = (
        obs.group_by(["id", "type"])
        .agg(pl.col("label").count().alias("n"))
        .sort("n", descending=True)
        .group_by("id", maintain_order=True)
        .agg(pl.col("type").first().alias("resolved_type"))
    )

    label_votes = (
        obs.group_by(["id", "label"])
        .agg(pl.col("type").count().alias("n"))
        .sort("n", descending=True)
        .group_by("id", maintain_order=True)
        .agg(pl.col("label").first().alias("resolved_label"))
    )

    lut = type_votes.join(label_votes, on="id")

    edges_pl = (
        edges_pl
        .join(lut.rename({"id": "src_id", "resolved_type": "src_type_r", "resolved_label": "src_label_r"}),
              on="src_id", how="left")
        .join(lut.rename({"id": "dst_id", "resolved_type": "dst_type_r", "resolved_label": "dst_label_r"}),
              on="dst_id", how="left")
        .with_columns([
            pl.col("src_type_r").fill_null(pl.col("source_type")).alias("source_type"),
            pl.col("dst_type_r").fill_null(pl.col("target_type")).alias("target_type"),
            pl.col("src_label_r").fill_null(pl.col("source_str")).alias("source_str"),
            pl.col("dst_label_r").fill_null(pl.col("target_str")).alias("target_str"),
        ])
        .drop(["src_type_r", "dst_type_r", "src_label_r", "dst_label_r"])
    )
    return edges_pl


def resolve_entities_in_edges(edges_pl: pl.DataFrame, threshold: float = 0.82) -> pl.DataFrame:
    """Step 1: fuzzy string merge of near-duplicate labels via rapidfuzz.
    Step 2: re-hash on the normalized label ONLY via canonical_entity_id.
    Step 3: majority-vote type + surface form per canonical id via
    resolve_canonical_types. Step 4: collapse duplicate (src, dst,
    predicate) rows now that ids are fully unified."""
    all_labels = edges_pl["source_str"].to_list() + edges_pl["target_str"].to_list()
    unique_labels = sorted(set(all_labels), key=len, reverse=True)

    canon_map: Dict[str, str] = {}
    if len(unique_labels) > 1:
        fast_scores = rfcdist(unique_labels, unique_labels, scorer=fuzz.token_sort_ratio, score_cutoff=70)
        merged: Set[str] = set()
        for i, a in enumerate(unique_labels):
            if a in merged:
                continue
            for j in range(i + 1, len(unique_labels)):
                b = unique_labels[j]
                if b in merged:
                    continue
                al, bl = a.lower(), b.lower()
                if bl in al or al in bl:
                    canon_map[b] = a
                    merged.add(b)
                    continue
                if fast_scores[i][j] >= 70:
                    sim = compute_entity_similarity(a, b)
                    if sim >= threshold:
                        canon_map[b] = a
                        merged.add(b)

    if canon_map:
        def resolve(x: str) -> str:
            return canon_map.get(x, x)

        edges_pl = edges_pl.with_columns([
            pl.col("source_str").map_elements(resolve, return_dtype=pl.Utf8),
            pl.col("target_str").map_elements(resolve, return_dtype=pl.Utf8),
        ])

    edges_pl = edges_pl.with_columns([
        pl.col("source_str").map_elements(canonical_entity_id, return_dtype=pl.Int64).alias("src_id"),
        pl.col("target_str").map_elements(canonical_entity_id, return_dtype=pl.Int64).alias("dst_id"),
    ])

    edges_pl = resolve_canonical_types(edges_pl)

    edges_pl = (
        edges_pl.group_by(["src_id", "dst_id", "predicate_str"])
        .agg([
            pl.col("confidence").mean(), pl.col("count").sum(),
            pl.col("source_str").first(), pl.col("target_str").first(),
            pl.col("source_type").first(), pl.col("target_type").first(),
        ])
    )

    print(f"[RESOLVE] {len(canon_map)} label aliases merged, "
          f"type conflicts reconciled by majority vote -> {len(edges_pl)} unique edges")
    return edges_pl


class TextChunker:
    """[CHUNK] One offset-mapped tokenization of the whole section, then a
    linear pass assigning token counts to sentences by character span --
    one tokenizer call per section instead of one per sentence. The
    overlap backtrack loop appends and reverses once instead of
    List.insert(0, ...) in a loop (which is O(N) per call, O(N^2) overall)."""

    def __init__(self, tokenizer, max_tokens: int = 900, overlap_pct: float = 0.20):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_pct = overlap_pct

    @staticmethod
    def _split_with_spans(text: str) -> List[Tuple[int, int]]:
        sep = re.compile(r'(?<=[.!?])\s+')
        spans = []
        pos = 0
        for m in sep.finditer(text):
            spans.append((pos, m.start()))
            pos = m.end()
        spans.append((pos, len(text)))
        return [s for s in spans if s[1] > s[0]]

    def _token_counts_per_sentence(self, text: str, spans: List[Tuple[int, int]]) -> List[int]:
        try:
            enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
        except Exception:
            return [int(len(text[s:e].split()) * 1.3) for s, e in spans]

        counts = []
        tok_idx = 0
        n_tokens = len(offsets)
        for s_start, s_end in spans:
            count = 0
            while tok_idx < n_tokens and offsets[tok_idx][0] < s_end:
                if offsets[tok_idx][1] <= s_start:
                    tok_idx += 1
                    continue
                count += 1
                tok_idx += 1
            counts.append(count)
        return counts

    def chunk_text(self, text: str) -> List[str]:
        spans = self._split_with_spans(text)
        if not spans:
            return [text]
        sentences = [text[s:e] for s, e in spans]
        token_counts = self._token_counts_per_sentence(text, spans)

        chunks: List[str] = []
        current: List[str] = []
        current_counts: List[int] = []
        length = 0

        for sent, n in zip(sentences, token_counts):
            if length + n > self.max_tokens:
                if current:
                    chunks.append(" ".join(current))
                overlap_budget = int(self.max_tokens * self.overlap_pct)
                carry: List[str] = []
                carry_counts: List[int] = []
                carry_tokens = 0
                for s, st in zip(reversed(current), reversed(current_counts)):
                    if carry_tokens + st > overlap_budget:
                        break
                    carry.append(s)
                    carry_counts.append(st)
                    carry_tokens += st
                carry.reverse()
                carry_counts.reverse()
                current = carry + [sent]
                current_counts = carry_counts + [n]
                length = carry_tokens + n
            else:
                current.append(sent)
                current_counts.append(n)
                length += n

        if current:
            chunks.append(" ".join(current))
        return chunks if chunks else [text]


# ============================================================================
# [CLASSIFY] DOMAIN CLASSIFICATION -- see PipelineConfig.enable_domain_classification
# above for the honest scope note (per-chunk labeling now, per-domain
# ontology switching later). Reuses whatever SentenceTransformer instance
# StreamableKGPipeline already loaded for [ANN] entity resolution -- pass
# it in rather than constructing a second copy, so this costs no extra
# VRAM/RAM on top of ANN resolution's existing ~80MB embedding model.
# ============================================================================

DEFAULT_DOMAIN_LABELS: Dict[str, str] = {
    "financial_regulatory": "banking, credit, monetary policy, financial regulation, compliance filings",
    "legal": "statutes, contracts, litigation, legal provisions and clauses",
    "technical_engineering": "software architecture, hardware systems, engineering specifications",
    "academic_research": "research methodology, literature review, experimental results, citations",
    "general_business": "corporate strategy, operations, general business correspondence",
}


class DomainClassifier:
    def __init__(self, embedder: "SentenceTransformer", labels: Dict[str, str]):
        self.embedder = embedder
        self.label_names = list(labels.keys())
        self._label_emb = embedder.encode(
            list(labels.values()), normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")

    def classify_batch(self, texts: List[str], max_chars: int = 2000) -> List[Tuple[str, float]]:
        """Batched, not one call per chunk -- StreamableKGPipeline calls
        this once per extraction batch (typically 8 chunks) rather than
        once per chunk, since sentence-transformers' own internal
        batching is far cheaper than N separate Python-level calls."""
        if not texts:
            return []
        truncated = [(t[:max_chars] if t.strip() else " ") for t in texts]
        embs = self.embedder.encode(
            truncated, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")
        sims = embs @ self._label_emb.T
        idxs = sims.argmax(axis=1)
        return [(self.label_names[i], float(sims[j, i])) for j, i in enumerate(idxs)]


# [ONTOLOGY] Extension point: ONE profile today. Add a second entry
# (e.g. "financial_regulatory": {"entity_types": {...}, "predicates":
# {...}, "system_prompt": "..."}) and thread DomainClassifier's per-chunk
# label through StreamableKGPipeline._process_chunks to pick a profile per
# chunk instead of always using "general", once you actually want
# per-domain grammars rather than just per-domain labeling.
ONTOLOGY_PROFILES: Dict[str, Dict[str, object]] = {
    "general": {
        "entity_types": ALLOWED_ENTITY_TYPES,
        "predicates": ALLOWED_PREDICATES,
    },
}


EXTRACTION_SYSTEM_PROMPT = """You extract relationships from text into a knowledge graph.

EXTRACT ONLY:
- Relationships explicitly stated in the text
- Direct inferences (if text says "X regulates Y", then "Y is regulated by X" is valid)

DO NOT:
- Hallucinate entities not in the text
- Use predicates not in the allowed list
- Extract generic/meaningless phrases
- Infer relationships requiring external knowledge

DIRECTION RULE (critical):
- Subject = the ACTOR/SOURCE of the relationship
- Target = the RECEIVER/DESTINATION of the relationship
- "RBI regulates banks" -> subject=RBI, predicate=regulates, target=banks
- "Banks are regulated by RBI" -> subject=RBI, predicate=regulates, target=banks (normalize direction to active actor)

ENTITY LENGTH RULE (critical):
- Entities are short noun phrases, not clauses or sentences (2-6 words max)
- If a chunk describes a broad topic or theme, do NOT extract the whole
  phrase as one entity. Extract the core concept instead and relate it
  to other short entities.

OUTPUT FORMAT: a single JSON object with one key, "triples", holding an
array of objects. Each object has exactly these five fields:
subject, subject_type, predicate, target, target_type.

TYPES: PERSON, ORGANIZATION, CONCEPT, PRODUCT, LOCATION, EVENT, LAW, METRIC, PROCESS, DOCUMENT, TECHNOLOGY, PROVISION

PREDICATES (use ONLY these exact active forms):
TYPE: is_a, instance_of, part_of, contains, includes
AGENCY: develops, authors, issues, regulates, manages, governs
ACTION: performs, enables, causes, requires, produces, applies_to
RELATION: affiliated_with, located_in, related_to, influences, depends_on

EXAMPLES:

TEXT: "The RBI regulates export credit through commercial banks."
{"triples": [
  {"subject": "RBI", "subject_type": "ORGANIZATION", "predicate": "regulates", "target": "export credit", "target_type": "CONCEPT"},
  {"subject": "commercial bank", "subject_type": "ORGANIZATION", "predicate": "performs", "target": "export credit", "target_type": "CONCEPT"}
]}

TEXT: "Section 7 of the Companies Act requires annual audits."
{"triples": [
  {"subject": "Section 7", "subject_type": "PROVISION", "predicate": "part_of", "target": "Companies Act", "target_type": "LAW"},
  {"subject": "Section 7", "subject_type": "PROVISION", "predicate": "requires", "target": "annual audit", "target_type": "PROCESS"}
]}

TEXT: "Interest rates increased by 50 basis points, affecting loan costs."
{"triples": [
  {"subject": "interest rate", "subject_type": "METRIC", "predicate": "influences", "target": "loan cost", "target_type": "METRIC"},
  {"subject": "50 basis points", "subject_type": "METRIC", "predicate": "causes", "target": "interest rate", "target_type": "METRIC"}
]}

NEGATIVE EXAMPLES -- DO NOT EXTRACT THESE:
TEXT: "This is important for various reasons." -> {"triples": []} (generic)
TEXT: "The report discusses several factors." -> {"triples": []} (vague)
TEXT: "As mentioned earlier, the policy changes." -> {"triples": []} (no relationship)
TEXT: "There are many options available." -> {"triples": []} (no entities)

TEXT: "The study explores the nature, range, and awareness/utilisation of
banking facilities across a diverse sample of importers and exporters."
WRONG subject: "nature range awareness utilisation of banking facilities across diverse sample importers and exporters" (a whole clause, not an entity)
RIGHT:
{"triples": [
  {"subject": "study", "subject_type": "DOCUMENT", "predicate": "related_to", "target": "banking facility awareness", "target_type": "CONCEPT"},
  {"subject": "importer", "subject_type": "CONCEPT", "predicate": "related_to", "target": "banking facility", "target_type": "CONCEPT"},
  {"subject": "exporter", "subject_type": "CONCEPT", "predicate": "related_to", "target": "banking facility", "target_type": "CONCEPT"}
]}

ENTITY RULES:
- Use exact names from text (keep acronyms if text uses them)
- Compound names: "export credit guarantee" not split
- If text says "RBI", use "RBI" not "Reserve Bank of India"

If unsure whether a relationship exists, DO NOT extract it. If there are
no valid triples, output {"triples": []}.

Extract from:"""


def enhance_connectivity(chunk: str, triples: List) -> List:
    """Standalone (no `self`) so VLLMTripleExtractor can call it without a
    model instance attached -- adds `related_to` edges between entities
    that co-occur in the source chunk but weren't otherwise connected."""
    if len(triples) < 2:
        return triples
    enhanced = list(triples)
    entities: Dict[str, str] = {}
    for s, st, _, t, tt, _ in triples:
        entities[s] = st
        entities[t] = tt
    cl = chunk.lower()
    present = {e for e in entities if e.lower() in cl}
    existing = frozenset(frozenset((s, t)) for s, _, _, t, _, _ in triples)
    el = list(present)[:15]
    for i in range(len(el)):
        for j in range(i + 1, len(el)):
            e1, e2 = el[i], el[j]
            if frozenset((e1, e2)) not in existing:
                enhanced.append((e1, entities[e1], "related_to", e2, entities[e2], 0.65))
    return enhanced


def _consensus_from_runs(
    all_runs: List[List[List]], chunks: List[str], enhance_fn: Callable[[str, List], List],
) -> List[List]:
    """Beta-Binomial agreement consensus across N independent extraction
    runs of the same chunks -- the voting logic behind
    VLLMTripleExtractor._bayesian_rerun's low-yield rescue path. Uses
    scipy.stats.beta.ppf for the lower confidence bound (torch's Beta
    distribution does not implement icdf -- it's declared on the base
    Distribution class and Beta doesn't override it, so calling it always
    raises NotImplementedError; verified numerically that ppf(0.05)
    round-trips through cdf back to 0.05 before relying on it here). Same
    Beta(2, 1) prior, same "keep the triple only if we're 95% confident
    its true agreement rate is >= 0.5" rule."""
    from scipy.stats import beta as beta_dist

    prior_alpha, prior_beta = 2.0, 1.0
    final = []
    for chunk_idx in range(len(chunks)):
        counts: Dict[Tuple, int] = {}
        triple_map: Dict[Tuple, Tuple] = {}
        for run in all_runs:
            for triple in run[chunk_idx]:
                key = (triple[0].lower(), triple[2].lower(), triple[3].lower())
                counts[key] = counts.get(key, 0) + 1
                if key not in triple_map:
                    triple_map[key] = triple

        n = len(all_runs)
        consensus = []
        for key, k in counts.items():
            post_alpha = prior_alpha + k
            post_beta = prior_beta + (n - k)
            posterior_mean = float(post_alpha / (post_alpha + post_beta))
            lower_ci = float(beta_dist.ppf(0.05, post_alpha, post_beta))
            if lower_ci >= 0.5:
                t = triple_map[key]
                consensus.append((*t[:5], posterior_mean))

        seen_keys: Set[Tuple] = set()
        deduped = []
        for t in consensus:
            k = (t[0], t[2], t[3])
            if k not in seen_keys:
                seen_keys.add(k)
                deduped.append(t)

        final.append(enhance_fn(chunks[chunk_idx], deduped) if deduped else [])
    return final


def expected_calibration_error(confidences: List[float], correct: List[bool], n_bins: int = 10) -> float:
    """Standard ECE: bins predictions by confidence, compares each bin's
    average confidence to its actual accuracy, weights by bin size. Lower
    is better; 0 is perfectly calibrated."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    if n == 0:
        return 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = corr[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def fit_confidence_calibration(raw_scores: List[float], correct: List[bool]) -> Callable[[float], float]:
    """Platt scaling: fits a 1-D logistic regression mapping a raw score
    to a calibrated P(correct). Needs a labeled sample -- spot-check
    ~100+ extracted triples, then:
        clf = fit_confidence_calibration(raw_scores, correct_labels)
        calibrated_conf = clf(new_raw_score)
    """
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(raw_scores, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(correct, dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise ValueError("Need both correct and incorrect examples to fit a calibration curve.")
    clf = LogisticRegression()
    clf.fit(X, y)

    def calibrated(raw_score: float) -> float:
        return float(clf.predict_proba([[raw_score]])[0, 1])

    return calibrated


class GPUTripleStore:
    """Identity is label-only (canonical_entity_id), matching [RESOLVE]
    above. No custom CUDA hash kernel here -- a plain xxhash64 Python loop
    costs microseconds at the batch sizes this pipeline actually sees
    (tens of entities per call); a hand-rolled GPU hash kernel would only
    earn its complexity at a much larger scale (millions of entities)."""

    def __init__(self):
        self.triples_df = None
        self._pending: List[cudf.DataFrame] = []

    @staticmethod
    def hash_entity(entity: str, etype: str = None) -> int:
        return canonical_entity_id(entity)

    @staticmethod
    def hash_batch(entities: List[str], etypes: List[str] = None) -> "cp.ndarray":
        return cp.array([canonical_entity_id(e) for e in entities], dtype=cp.int64)

    def add_triples(self, triples: List[Tuple]):
        if not triples:
            return
        s = [t[0] for t in triples]
        st = [t[1] for t in triples]
        p = [t[2] for t in triples]
        tg = [t[3] for t in triples]
        tt = [t[4] for t in triples]
        c = [t[5] for t in triples]
        self._pending.append(cudf.DataFrame({
            "src_id": cudf.Series(self.hash_batch(s, st)),
            "rel_id": cudf.Series(self.hash_batch(p)),
            "dst_id": cudf.Series(self.hash_batch(tg, tt)),
            "confidence": cudf.Series(c, dtype="float32"),
            "source_str": s, "source_type": st, "predicate_str": p,
            "target_str": tg, "target_type": tt,
        }))

    def flush(self):
        if self._pending:
            all_dfs = ([self.triples_df] if self.triples_df is not None else []) + self._pending
            self.triples_df = cudf.concat(all_dfs, ignore_index=True)
            self._pending = []

    def aggregate(self) -> cudf.DataFrame:
        if self.triples_df is None or len(self.triples_df) == 0:
            return cudf.DataFrame()
        agg = self.triples_df.groupby(["src_id", "rel_id", "dst_id"]).agg({
            "confidence": "mean", "source_str": "first", "source_type": "first",
            "predicate_str": "first", "target_str": "first", "target_type": "first",
        }).reset_index()
        counts = self.triples_df.groupby(["src_id", "rel_id", "dst_id"]).size().reset_index()
        counts.columns = ["src_id", "rel_id", "dst_id", "count"]
        return agg.merge(counts, on=["src_id", "rel_id", "dst_id"])


class GPUGraphAnalyzer:
    """[GRAPH] PageRank, in/out-degree, Leiden communities -- matches the
    "In / out-degree · PageRank" node in the architecture diagram
    (Leiden is a bonus beyond what's pictured there, kept since it was
    already load-bearing for the community_id column in the export)."""

    def __init__(self, edges_cudf: cudf.DataFrame, leiden_resolution: float = 1.0):
        self.edges_cudf = edges_cudf
        self.leiden_resolution = leiden_resolution

        self.g_dir = cugraph.Graph(directed=True)
        self.g_dir.from_cudf_edgelist(
            edges_cudf, source="src_id", destination="dst_id",
            edge_attr="confidence", store_transposed=True,
        )

        self.g_undir = cugraph.Graph(directed=False)
        self.g_undir.from_cudf_edgelist(
            edges_cudf, source="src_id", destination="dst_id", edge_attr="confidence",
        )

    def compute_metrics(self) -> Dict:
        m: Dict = {}

        try:
            m["pagerank"] = cugraph.pagerank(self.g_dir)
        except Exception as e:
            print(f"PageRank failed: {e}")
            u = cudf.concat([self.edges_cudf["src_id"], self.edges_cudf["dst_id"]]).unique()
            m["pagerank"] = cudf.DataFrame({"vertex": u, "pagerank": [0.001] * len(u)})

        try:
            m["in_degree"] = self.g_dir.in_degree()
            m["out_degree"] = self.g_dir.out_degree()
        except Exception as e:
            print(f"Degree failed: {e}")

        try:
            leiden_df, modularity = cugraph.leiden(self.g_undir, max_iter=100, resolution=self.leiden_resolution)
            m["leiden"] = leiden_df
            n_comm = int(leiden_df["partition"].n_unique())
            print(f"[GRAPH] Leiden: {n_comm} communities  (modularity={modularity:.4f})")
        except Exception as e:
            print(f"Leiden failed: {e}")

        return m


class GPUDirectStorage:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.path = Path(checkpoint_dir)
        self.path.mkdir(exist_ok=True)
        self.counter = 0

    def save_checkpoint(self, df: cudf.DataFrame, name: str = None) -> Path:
        if name is None:
            name = f"checkpoint_{self.counter:04d}.parquet"
            self.counter += 1
        fp = self.path / name
        try:
            df.to_parquet(str(fp), compression="snappy")
        except Exception as e:
            print(f"Checkpoint failed: {e}")
        return fp


# ============================================================================
# [ANN] Semantic entity resolution beyond fuzzy string matching --
# duplicate merges via embedding similarity, `semantically_similar_to`
# edge proposals between related-but-unconnected entities. Both resolvers
# below accept a pre-built `embedder` so StreamableKGPipeline can share
# the SAME SentenceTransformer instance it already loaded for [CLASSIFY],
# instead of loading a second ~80MB copy.
# ============================================================================

class SemanticEntityResolver:
    def __init__(self, model_name: str, index_type: str = "flat",
                 device: Optional[str] = None, embedder: Optional["SentenceTransformer"] = None):
        # faiss is required unconditionally here (used for indexing/search
        # regardless of whether an embedder was shared) -- checked first
        # so the error names the actual missing piece instead of a
        # confusing later NameError from an unbound `faiss` name.
        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "faiss is required for SemanticEntityResolver (the FAISS ANN backend).\n"
                "  pip install faiss-cpu (or faiss-gpu-cu12)\n"
                "or set PipelineConfig.ann_backend = \"cuml\" to avoid needing faiss at all."
            )
        if embedder is not None:
            self.embedder = embedder
        else:
            if not EMBEDDER_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers is required for the embedding step.\n"
                    "  pip install sentence-transformers\n"
                    "or set PipelineConfig.enable_ann_resolution = False"
                )
            self.embedder = SentenceTransformer(
                model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.index_type = index_type
        self.index = None
        self.ids: List[int] = []
        self.labels: List[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def build_index(self, id_label_pairs: List[Tuple[int, str]]):
        self.ids = [i for i, _ in id_label_pairs]
        self.labels = [l for _, l in id_label_pairs]
        if not self.labels:
            return
        emb = self.embedder.encode(
            self.labels, batch_size=256, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")
        dim = emb.shape[1]
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 64
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(emb)
        self.index = index
        self._embeddings = emb

    def find_duplicates(self, similarity_threshold: float = 0.90, k: int = 5) -> Dict[int, int]:
        if self.index is None or len(self.ids) < 2:
            return {}
        k = min(k + 1, len(self.ids))
        sims, neighbors = self.index.search(self._embeddings, k)
        order = sorted(range(len(self.ids)), key=lambda i: -len(self.labels[i]))
        rank = {idx: r for r, idx in enumerate(order)}
        canon_map: Dict[int, int] = {}
        merged: Set[int] = set()
        for i in range(len(self.ids)):
            if self.ids[i] in merged:
                continue
            for r in range(1, k):
                j = int(neighbors[i, r])
                if j < 0 or j == i or float(sims[i, r]) < similarity_threshold:
                    continue
                if self.ids[j] in merged:
                    continue
                a, b = (i, j) if rank[i] < rank[j] else (j, i)
                canon_map[self.ids[b]] = self.ids[a]
                merged.add(self.ids[b])
        return canon_map

    def find_missing_edges(
        self, existing_pairs: Set[Tuple[int, int]],
        similarity_threshold: float = 0.78, k: int = 8, max_new_edges: int = 5000,
    ) -> List[Tuple[int, int, float]]:
        if self.index is None or len(self.ids) < 2:
            return []
        k = min(k + 1, len(self.ids))
        sims, neighbors = self.index.search(self._embeddings, k)
        proposals = []
        seen_pairs: Set[Tuple[int, int]] = set()
        for i in range(len(self.ids)):
            for r in range(1, k):
                j = int(neighbors[i, r])
                if j < 0 or j == i:
                    continue
                score = float(sims[i, r])
                if score < similarity_threshold:
                    continue
                a, b = self.ids[i], self.ids[j]
                pair = (min(a, b), max(a, b))
                if pair in existing_pairs or pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                proposals.append((a, b, score))
        proposals.sort(key=lambda x: -x[2])
        return proposals[:max_new_edges]

    def neighbors(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if self.index is None:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        sims, idxs = self.index.search(q, min(k, len(self.ids)))
        return [(self.labels[idx], float(sims[0, r])) for r, idx in enumerate(idxs[0]) if idx >= 0]


class CuMLEntityResolver:
    """Same interface as SemanticEntityResolver, backed by
    cuml.neighbors.NearestNeighbors (cuVS-accelerated ANN) instead of
    FAISS-CPU -- default backend here since the pipeline is already
    RAPIDS-native, sharing the same CUDA context/memory pool instead of
    adding a second CPU-side ANN library, with search running on-GPU
    rather than round-tripping embeddings to host memory."""

    def __init__(self, model_name: str, device: Optional[str] = None,
                 embedder: Optional["SentenceTransformer"] = None):
        if embedder is not None:
            self.embedder = embedder
        else:
            # Only needs sentence-transformers here -- unlike
            # SemanticEntityResolver, this class never touches faiss, so
            # gating on EMBEDDER_AVAILABLE (not the faiss-inclusive
            # ANN_AVAILABLE) is what actually reflects this class's real
            # dependency.
            if not EMBEDDER_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers is required for the embedding step.\n"
                    "  pip install sentence-transformers"
                )
            self.embedder = SentenceTransformer(
                model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
        if not CUML_AVAILABLE:
            raise RuntimeError(
                "cuml is required for CuMLEntityResolver.\n"
                "  pip install cuml-cu12 (match your RAPIDS version)\n"
                "or set PipelineConfig.ann_backend = \"faiss\" to use SemanticEntityResolver instead."
            )
        self.ids: List[int] = []
        self.labels: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._nn = None

    def build_index(self, id_label_pairs: List[Tuple[int, str]]):
        self.ids = [i for i, _ in id_label_pairs]
        self.labels = [l for _, l in id_label_pairs]
        if not self.labels:
            return
        emb = self.embedder.encode(
            self.labels, batch_size=256, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")
        self._embeddings = emb
        self._nn = _CuMLNearestNeighbors(n_neighbors=min(16, len(emb)), metric="euclidean")
        self._nn.fit(emb)

    def _search(self, query_emb: np.ndarray, k: int):
        k = min(k, len(self.ids))
        distances, indices = self._nn.kneighbors(query_emb, n_neighbors=k)
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        sims = 1.0 - (distances ** 2) / 2.0
        return sims, indices

    def find_duplicates(self, similarity_threshold: float = 0.90, k: int = 5) -> Dict[int, int]:
        if self._nn is None or len(self.ids) < 2:
            return {}
        sims, neighbors = self._search(self._embeddings, k + 1)
        order = sorted(range(len(self.ids)), key=lambda i: -len(self.labels[i]))
        rank = {idx: r for r, idx in enumerate(order)}
        canon_map: Dict[int, int] = {}
        merged: Set[int] = set()
        for i in range(len(self.ids)):
            if self.ids[i] in merged:
                continue
            for r in range(1, sims.shape[1]):
                j = int(neighbors[i, r])
                if j < 0 or j == i or float(sims[i, r]) < similarity_threshold:
                    continue
                if self.ids[j] in merged:
                    continue
                a, b = (i, j) if rank[i] < rank[j] else (j, i)
                canon_map[self.ids[b]] = self.ids[a]
                merged.add(self.ids[b])
        return canon_map

    def find_missing_edges(
        self, existing_pairs: Set[Tuple[int, int]],
        similarity_threshold: float = 0.78, k: int = 8, max_new_edges: int = 5000,
    ) -> List[Tuple[int, int, float]]:
        if self._nn is None or len(self.ids) < 2:
            return []
        sims, neighbors = self._search(self._embeddings, k + 1)
        proposals = []
        seen_pairs: Set[Tuple[int, int]] = set()
        for i in range(len(self.ids)):
            for r in range(1, sims.shape[1]):
                j = int(neighbors[i, r])
                if j < 0 or j == i:
                    continue
                score = float(sims[i, r])
                if score < similarity_threshold:
                    continue
                a, b = self.ids[i], self.ids[j]
                pair = (min(a, b), max(a, b))
                if pair in existing_pairs or pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                proposals.append((a, b, score))
        proposals.sort(key=lambda x: -x[2])
        return proposals[:max_new_edges]

    def neighbors(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if self._nn is None:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        sims, idxs = self._search(q, k)
        return [(self.labels[idx], float(sims[0, r])) for r, idx in enumerate(idxs[0]) if idx >= 0]


def apply_ann_resolution(edges_pl: pl.DataFrame, cfg: PipelineConfig,
                          shared_embedder: Optional["SentenceTransformer"] = None) -> pl.DataFrame:
    if not cfg.enable_ann_resolution:
        return edges_pl

    def node_view(df):
        return pl.concat([
            df.select([pl.col("src_id").alias("id"), pl.col("source_str").alias("label"),
                       pl.col("source_type").alias("type")]),
            df.select([pl.col("dst_id").alias("id"), pl.col("target_str").alias("label"),
                       pl.col("target_type").alias("type")]),
        ]).unique("id")

    backend = cfg.ann_backend
    have_embedder_source = shared_embedder is not None or EMBEDDER_AVAILABLE

    def _backend_usable(b: str) -> bool:
        # cuML needs sentence-transformers (for embedding) + cuml itself
        # (for search) -- it never touches faiss. FAISS needs
        # sentence-transformers + faiss itself, never touches cuml.
        # Checking each backend's REAL dependency set (rather than the
        # combined ANN_AVAILABLE flag) is what lets a box with cuml +
        # sentence-transformers but no faiss installed still get ANN
        # resolution via cuML instead of being skipped outright.
        if b == "cuml":
            return CUML_AVAILABLE and have_embedder_source
        return FAISS_AVAILABLE and have_embedder_source

    if not _backend_usable(backend):
        other = "faiss" if backend == "cuml" else "cuml"
        if _backend_usable(other):
            print(f"[ANN] ann_backend='{backend}' isn't usable in this environment -- falling back to '{other}'.")
            backend = other
        else:
            print("[ANN] skipped: no usable ANN backend in this environment "
                  "(need sentence-transformers, plus either cuml or faiss).")
            return edges_pl

    if backend == "cuml":
        resolver = CuMLEntityResolver(cfg.ann_embedding_model, embedder=shared_embedder)
    else:
        resolver = SemanticEntityResolver(cfg.ann_embedding_model, index_type=cfg.ann_index_type, embedder=shared_embedder)

    nodes = node_view(edges_pl)
    resolver.build_index(list(zip(nodes["id"].to_list(), nodes["label"].to_list())))

    dup_map = resolver.find_duplicates(similarity_threshold=cfg.ann_duplicate_threshold)
    if dup_map:
        edges_pl = edges_pl.with_columns([
            pl.col("src_id").map_elements(lambda x: dup_map.get(x, x), return_dtype=pl.Int64),
            pl.col("dst_id").map_elements(lambda x: dup_map.get(x, x), return_dtype=pl.Int64),
        ])
        edges_pl = resolve_canonical_types(edges_pl)
        edges_pl = (
            edges_pl.group_by(["src_id", "dst_id", "predicate_str"])
            .agg([
                pl.col("confidence").mean(), pl.col("count").sum(),
                pl.col("source_str").first(), pl.col("target_str").first(),
                pl.col("source_type").first(), pl.col("target_type").first(),
                pl.col("edge_source").first(),
            ])
        )
        print(f"[ANN] {len(dup_map)} semantic duplicates merged (backend={backend})")
        nodes = node_view(edges_pl)

    existing_pairs = set(zip(edges_pl["src_id"].to_list(), edges_pl["dst_id"].to_list()))
    proposals = resolver.find_missing_edges(
        existing_pairs, similarity_threshold=cfg.ann_missing_edge_threshold, max_new_edges=cfg.ann_max_proposed_edges,
    )
    if proposals:
        label_lu = dict(zip(nodes["id"].to_list(), nodes["label"].to_list()))
        type_lu = dict(zip(nodes["id"].to_list(), nodes["type"].to_list()))
        new_edges_pl = pl.DataFrame({
            "src_id": [a for a, b, s in proposals], "dst_id": [b for a, b, s in proposals],
            "predicate_str": ["semantically_similar_to"] * len(proposals),
            "confidence": [s for a, b, s in proposals], "count": [1] * len(proposals),
            "source_str": [label_lu.get(a, "") for a, b, s in proposals],
            "target_str": [label_lu.get(b, "") for a, b, s in proposals],
            "source_type": [type_lu.get(a, "CONCEPT") for a, b, s in proposals],
            "target_type": [type_lu.get(b, "CONCEPT") for a, b, s in proposals],
            "edge_source": ["semantic_similarity"] * len(proposals),
        })
        edges_pl = pl.concat([edges_pl, new_edges_pl], how="vertical_relaxed")
        print(f"[ANN] {len(proposals)} candidate semantic edges proposed (predicate='semantically_similar_to')")

    return edges_pl


# ============================================================================
# [OBSERVE] Langfuse tracing -- the "LLM traces / prompts / generations /
# human evaluation" column in your split (W&B below covers
# experiments/GPU metrics/eval/datasets/training instead).
#
# LangfuseTracer degrades to a real no-op if LANGFUSE_PUBLIC_KEY /
# LANGFUSE_SECRET_KEY aren't set, or if the client can't be constructed --
# tracing is instrumentation, not a correctness dependency, so a
# misconfigured or unreachable Langfuse must never be able to break
# extraction. Uses the OTel-based v3 Python SDK
# (start_as_current_observation / nested start_as_current_generation) --
# checked against Langfuse's current docs while writing this, since the
# pre-3.x trace()/generation()/span() API some older tutorials show no
# longer matches. `as_type="generation"` observations nested under a
# per-batch span is effectively "OpenTelemetry -> Langfuse" in the
# original diagram, since v3's SDK IS the OTel span processor pointed at
# Langfuse's OTLP endpoint.
# ============================================================================

class _NullObservation:
    """No-op stand-in with the subset of the Langfuse observation
    interface this pipeline calls, so call sites never need an `if
    tracer.enabled` branch -- there's always SOMETHING to call
    .update()/.start_as_current_generation() on."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, **kwargs):
        pass

    def start_as_current_generation(self, **kwargs):
        return self


class LangfuseTracer:
    def __init__(self, enabled: bool = True):
        self.client = None
        self.enabled = False
        if not (enabled and LANGFUSE_AVAILABLE):
            return
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            print("[OBSERVE] Langfuse tracing disabled: LANGFUSE_PUBLIC_KEY/SECRET_KEY not set.")
            return
        try:
            self.client = Langfuse()
            self.enabled = True
            print(f"[OBSERVE] Langfuse tracing enabled -> {os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')}")
        except Exception as e:
            print(f"[OBSERVE] Langfuse tracing disabled: client init failed ({e})")

    def batch_span(self, name: str, **kwargs):
        if not self.enabled:
            return _NullObservation()
        return self.client.start_as_current_observation(as_type="span", name=name, **kwargs)

    def flush(self):
        if self.enabled:
            try:
                self.client.flush()
            except Exception as e:
                print(f"[OBSERVE] Langfuse flush failed: {e}")


# ============================================================================
# [OBSERVE] vLLM's own /metrics (Prometheus exposition format) -> the SAME
# W&B run this pipeline already logs batch/graph metrics to. This is the
# direct swap for "vLLM metrics -> Prometheus -> Grafana" in the original
# diagram: same source, different sink, so there's one run to look at
# instead of a second dashboard stack to stand up. Uses
# prometheus_client's own text parser rather than hand-rolled regex --
# Prometheus exposition format has real edge cases (multi-line HELP/TYPE
# comments, quoted label values, histogram _bucket/_sum/_count suffixes)
# that a naive regex eventually gets wrong on some vLLM version's exact
# output; prometheus_client is a tiny, pure-Python package (already a
# transitive dependency inside the vLLM venv, since vLLM uses it to
# EXPOSE /metrics -- installed separately here, in the MAIN env, purely
# to PARSE the text vLLM returns over HTTP).
# ============================================================================

def _parse_vllm_metrics(metrics_text: str) -> Dict[str, float]:
    from prometheus_client.parser import text_string_to_metric_families
    out: Dict[str, float] = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            key = sample.name
            if sample.labels:
                lbl = ",".join(f'{k}="{v}"' for k, v in sorted(sample.labels.items()))
                key = f"{key}{{{lbl}}}"
            out[key] = sample.value
    return out


class VLLMMetricsPoller:
    """Background thread. Only forwards a curated subset of series (not
    every one vLLM exposes) -- logging every histogram bucket individually
    would flood the W&B run with near-duplicate keys."""

    _WANTED_PREFIXES = (
        "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_requests_swapped",
        "vllm:gpu_cache_usage_perc", "vllm:gpu_prefix_cache_hit_rate",
        "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
        "vllm:time_per_output_token_seconds_sum", "vllm:time_per_output_token_seconds_count",
        "vllm:request_success_total", "vllm:request_prompt_tokens_sum", "vllm:request_generation_tokens_sum",
    )

    def __init__(self, base_url: str, interval_s: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll_once(self):
        import requests
        try:
            resp = requests.get(f"{self.base_url}/metrics", timeout=5)
            resp.raise_for_status()
        except Exception:
            return
        try:
            parsed = _parse_vllm_metrics(resp.text)
        except Exception as e:
            print(f"[OBSERVE] vLLM /metrics parse failed: {e}")
            return
        log_dict = {f"vllm/{k}": v for k, v in parsed.items() if any(k.startswith(p) for p in self._WANTED_PREFIXES)}
        if log_dict and wandb.run is not None:
            wandb.log(log_dict)

    def _loop(self):
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval_s)

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[OBSERVE] Polling vLLM /metrics every {self.interval_s}s -> W&B")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None


# ============================================================================
# [PROCESS B] Grammar-constrained decoding: see build_triple_json_schema()
# and TripleListModel near the top of this file (moved there since the
# schema is needed by [VALIDATE]'s parse_json_triples too). The `{"regex":
# ...}` approach that used to live here timed out xgrammar's regex
# compiler -- see the comment above [VALIDATE] for the confirmed
# traceback and why JSON schema replaced it.
# ============================================================================


# ============================================================================
# [PROCESS B] vLLM server launcher -- hardened
#
# Changes vs. the predecessor this was built from, all aimed at the exact
# failure mode hit while developing this project (a launch_vllm_server
# call that just prints "...still waiting" with no detail, gets
# interrupted out of impatience, with no way to tell "still downloading",
# "still compiling", or "actually crashed" apart):
#
#   1. _kill_stale_server_on_port: a re-run of the launch cell after an
#      earlier server was left running (kernel restarted without calling
#      shutdown_vllm_server(), or a cell re-run reset this module's
#      _vllm_state dict while the OS process kept running) leaves the
#      port bound. The new subprocess.Popen would then either silently
#      talk to a STALE server, or fail to bind and exit immediately --
#      which looks IDENTICAL to "didn't come up in time" in the poll
#      loop. Find and kill anything already on the port before launching.
#   2. The poll loop now tails the log file's last line into its own
#      progress print every ~30s, so "still waiting" comes with an actual
#      line from vLLM's own startup log (weight loading %, CUDA graph
#      capture progress, etc.) instead of being a black box.
#   3. poll() is checked every iteration, not just periodically -- a
#      process that exits on its own (config error, OOM, etc.) is reported
#      immediately as a real failure with the log path, not as a timeout
#      45 minutes later.
#   4. ensure_vllm_server() below wraps this in a singleton accessor so a
#      Colab "Process PDF" button click reuses an already-warm server
#      instead of re-paying the full startup cost (weight load +
#      torch.compile + CUDA graph capture, real minutes on a T4) on every
#      single document -- the predecessor this was built from explicitly
#      flagged this as a known, unbuilt limitation.
# ============================================================================

def _venv_vllm_bin(venv_path: str) -> str:
    return str(Path(venv_path) / "bin" / "vllm")


def _port_is_free(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _kill_stale_server_on_port(port: int):
    try:
        import psutil
    except ImportError:
        return  # best-effort only -- not worth a hard dependency for this
    killed = []
    try:
        conns = psutil.net_connections(kind="inet")
    except Exception:
        return
    for conn in conns:
        if conn.laddr and getattr(conn.laddr, "port", None) == port and conn.pid:
            try:
                psutil.Process(conn.pid).terminate()
                killed.append(conn.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    if killed:
        print(f"[PROCESS B] Killed {len(killed)} stale process(es) already bound to port {port}: {killed}")
        time.sleep(2)


def _tail_line(path: str, n_bytes: int = 600) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            data = f.read().decode("utf-8", errors="replace")
        lines = [l for l in data.splitlines() if l.strip()]
        return lines[-1] if lines else ""
    except FileNotFoundError:
        return ""


def tail_log(path: str = "/content/vllm_server.log", n_lines: int = 40):
    """Interactive helper -- call this from its own cell while a launch is
    still in progress to see the last N lines of vLLM's startup log
    directly, instead of waiting for the next poll-loop print."""
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        print("".join(lines[-n_lines:]))
    except FileNotFoundError:
        print(f"(no log file at {path} yet)")


_UNSET = object()  # sentinel distinguishing "argument not passed, use cfg's
# value" from "explicitly passed as None" -- matters specifically for
# `quantization` below, where None is itself a meaningful value ("no
# --quantization flag at all, serve full precision") distinct from
# "I didn't specify one."


def launch_vllm_server(
    cfg: PipelineConfig,
    target_model: Optional[str] = None,
    draft_model: Optional[str] = None,
    speculative_method: Optional[str] = None,
    num_speculative_tokens: Optional[int] = None,
    quantization: Optional[str] = _UNSET,
    venv_path: Optional[str] = None,
    port: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: int = 4096,
    max_num_seqs: Optional[int] = None,
    enforce_eager: bool = False,
    dtype: str = "float16",
    log_path: str = "/content/vllm_server.log",
    timeout_s: int = 900,
    poll_interval_s: float = 5.0,
):
    """Launches vLLM as a background OpenAI-compatible HTTP server from an
    ISOLATED venv -- see the module docstring at the top of this file for
    why. Assumes the venv already exists with vllm installed:
        python3 -m venv /content/vllm_env
        /content/vllm_env/bin/pip install vllm

    MODEL SELECTION: target_model defaults to f"{cfg.model_name}-AWQ",
    e.g. "Qwen/Qwen2.5-3B-Instruct" -> "Qwen/Qwen2.5-3B-Instruct-AWQ" --
    NOT cfg.model_name unchanged. Qwen/Qwen2.5-3B-Instruct-AWQ is an
    official Qwen-published checkpoint.

    QUANTIZATION KERNEL ON THIS HARDWARE, worth knowing explicitly rather
    than assuming: a T4 is Turing (SM 7.5). Per vLLM's own supported-
    hardware table, AWQ IS supported on Turing, but the Marlin-fused
    kernel is NOT (Marlin needs Ampere/SM 8.0+) -- vLLM's awq_marlin
    linear method auto-selects the best AVAILABLE kernel for the current
    platform via choose_mp_linear_kernel, which on a T4 means it silently
    falls back to the plain (non-Marlin) AWQ kernel. So `--quantization
    awq` below is correctly supported and will work on a T4, it just
    won't get the Marlin throughput boost an A100/H100 sees -- worth
    knowing so a T4 benchmark that looks slower than someone else's AWQ
    number isn't mistaken for a misconfiguration.

    SPECULATIVE DECODING: uses vLLM's current `--speculative-config` JSON
    syntax (the older separate `--speculative-model` /
    `--num-speculative-tokens` flags are deprecated).
      "ngram" (PipelineConfig default here): no draft model, no separate
        weights -- speculates by matching candidate tokens directly
        against the prompt. Well-suited to this task specifically since
        extracted entities are frequently literal substrings of the
        source chunk; also the resource-lighter choice on a
        16GB-VRAM/12GB-RAM box since it needs no second model loaded.
      "draft_model": {"method": "draft_model", "model": draft_model,
        "num_speculative_tokens": ...} -- draft and target must share
        vocabulary (same Qwen2.5 family here, so that's satisfied).

    STARTUP TIME: default vLLM startup does torch.compile followed by CUDA
    graph capture across its default batch-size list (which climbs into
    the hundreds) -- on a T4 that legitimately takes several minutes,
    longer on a cold torch.compile cache. Two independent levers here:
      1. timeout_s defaults to 900s (15 min), generous enough for a real
         first-run compile; the poll loop tails the log every ~30s so you
         can see which stage it's actually at rather than guessing.
      2. max_num_seqs, if passed (else defaults to cfg.batch_size), caps
         how many CUDA graph sizes vLLM captures at startup -- capturing
         up to the library default when this pipeline only ever sends
         batches of cfg.batch_size is pure wasted startup time.
    Pass enforce_eager=True to skip torch.compile/CUDA graph capture
    entirely for a startup that completes in well under a minute, trading
    away some runtime throughput -- good for a first correctness smoke
    test before switching it back to False for a real run.

    Guided-decoding backend selection (xgrammar/guidance/etc.) is a
    server-side `vllm serve` startup flag as of current vLLM, not a
    per-request field -- deliberately not pinning one here since the
    exact current flag name is the kind of thing that drifts across
    releases and getting a STARTUP flag wrong fails the whole server, not
    just structured output; vLLM's own "auto" default picks a backend per
    request and is fine to leave alone.
    """
    import requests as _requests

    target_model = target_model or f"{cfg.model_name}-AWQ"
    draft_model = draft_model or cfg.draft_model_name
    speculative_method = speculative_method or cfg.vllm_speculative_method
    num_speculative_tokens = num_speculative_tokens or cfg.vllm_num_speculative_tokens
    quantization = cfg.vllm_quantization if quantization is _UNSET else quantization
    venv_path = venv_path or cfg.vllm_venv_path
    port = port if port is not None else cfg.vllm_port
    gpu_memory_utilization = gpu_memory_utilization if gpu_memory_utilization is not None else cfg.vllm_gpu_memory_utilization
    max_num_seqs = max_num_seqs or max(cfg.batch_size, 8)

    vllm_bin = _venv_vllm_bin(venv_path)
    if not Path(vllm_bin).exists():
        raise FileNotFoundError(
            f"{vllm_bin} not found -- set up the isolated venv first:\n"
            f"  python3 -m venv {venv_path} && {venv_path}/bin/pip install vllm"
        )

    _kill_stale_server_on_port(port)
    if not _port_is_free(port):
        raise RuntimeError(f"Port {port} is still occupied after attempting cleanup -- check `lsof -i:{port}` manually.")

    cmd = [
        vllm_bin, "serve", target_model,
        "--port", str(port),
        "--dtype", dtype,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
    ]
    if quantization:
        cmd += ["--quantization", quantization]
    if enforce_eager:
        cmd += ["--enforce-eager"]
    if speculative_method == "ngram":
        spec_config = {"method": "ngram", "num_speculative_tokens": num_speculative_tokens, "prompt_lookup_max": 4}
        cmd += ["--speculative-config", json.dumps(spec_config)]
    elif speculative_method == "draft_model" and draft_model:
        spec_config = {"method": "draft_model", "model": draft_model, "num_speculative_tokens": num_speculative_tokens}
        cmd += ["--speculative-config", json.dumps(spec_config)]

    print(f"[PROCESS B] Launching: {' '.join(cmd)}")
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    base_url = f"http://localhost:{port}"
    n_iters = max(1, int(timeout_s // poll_interval_s))
    for i in range(n_iters):
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited on its own (code {proc.returncode}) "
                f"before becoming healthy -- this is a real failure, not a timeout. "
                f"Last log line: {_tail_line(log_path)!r}. Full log: {log_path}"
            )
        try:
            if _requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                elapsed = i * poll_interval_s
                print(f"[PROCESS B] vLLM server up at {base_url} (target={target_model}, "
                      f"spec={speculative_method}) after ~{elapsed:.0f}s")
                return proc, base_url
        except _requests.exceptions.ConnectionError:
            pass
        if i > 0 and i % max(1, int(30 // poll_interval_s)) == 0:
            print(f"...still starting ({i * poll_interval_s:.0f}s elapsed, timeout at {timeout_s}s) "
                  f"| log: {_tail_line(log_path)[:160]}")
        time.sleep(poll_interval_s)

    proc.terminate()
    raise RuntimeError(
        f"vLLM server didn't come up within {timeout_s}s -- check {log_path}. "
        f"If the log shows it's still in torch.compile/CUDA graph capture, raise "
        f"timeout_s, lower max_num_seqs, or pass enforce_eager=True for a much "
        f"faster (if slightly slower at runtime) startup."
    )


# Module-level singleton so a fresh StreamableKGPipeline (one per "Process
# PDF" click) reuses an already-warm server instead of relaunching.
_vllm_state: Dict[str, object] = {"proc": None, "base_url": None}


def _server_healthy(base_url: str, timeout_s: float = 3.0) -> bool:
    import requests
    try:
        return requests.get(f"{base_url}/health", timeout=timeout_s).status_code == 200
    except Exception:
        return False


def ensure_vllm_server(cfg: PipelineConfig, **launch_kwargs) -> Tuple[subprocess.Popen, str]:
    """Reuses the already-running server if it's still healthy; otherwise
    launches a fresh one via launch_vllm_server (which itself cleans up
    anything stale bound to the port first)."""
    proc = _vllm_state.get("proc")
    base_url = _vllm_state.get("base_url")
    if proc is not None and base_url is not None:
        if proc.poll() is None and _server_healthy(base_url):
            return proc, base_url
        print("[PROCESS B] Existing vLLM server is no longer healthy -- relaunching.")

    proc, base_url = launch_vllm_server(cfg, **launch_kwargs)
    _vllm_state["proc"] = proc
    _vllm_state["base_url"] = base_url
    return proc, base_url


def shutdown_vllm_server():
    proc = _vllm_state.get("proc")
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[PROCESS B] vLLM server stopped.")
    _vllm_state["proc"] = None
    _vllm_state["base_url"] = None


class VLLMTripleExtractor:
    """HTTP client for a vLLM server running in an ISOLATED venv --
    deliberately NOT an in-process `from vllm import LLM` (see module
    docstring). Sends the whole batch as a single request with prompt as
    a LIST so vLLM's server-side scheduler does the actual
    continuous-batching work.

    GUIDED DECODING FIELD: `{"structured_outputs": {"json": ...}}` --
    JSON-schema-constrained, not regex-constrained (see the comment above
    [VALIDATE]/build_triple_json_schema for why: the regex approach timed
    out xgrammar's compiler on real hardware).
    """

    def __init__(self, cfg: PipelineConfig, base_url: str = "http://localhost:8000",
                 tracer: Optional[LangfuseTracer] = None, request_timeout_s: int = 120):
        import requests
        self._requests = requests
        self.config = cfg
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.json_schema = build_triple_json_schema(
            max_entity_chars=cfg.max_entity_chars, max_triples_per_chunk=cfg.max_triples_per_chunk,
        )
        self.tracer = tracer or LangfuseTracer(enabled=False)

        resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        self.model_name = resp.json()["data"][0]["id"]

        # Loading only the tokenizer files is a plain transformers call
        # with no CUDA/numba/torch-device involvement, so it doesn't
        # reintroduce the RAPIDS/vLLM dependency conflict process
        # isolation exists to avoid -- only loading MODEL weights
        # in-process would do that. $HOME (and so ~/.cache/huggingface)
        # is shared across venvs, so this finds files already pulled down
        # by the pre-download step or by vLLM's own server-startup fetch.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def _confidence_from_choice(self, choice: dict) -> float:
        """Confidence from the average per-token log-probability (vLLM's
        completions API `logprobs` field, requested via `"logprobs": 1`
        below). exp(avg_logprob * scale): the geometric mean of per-token
        probabilities raised to a tunable power -- monotonic and
        correctly bounded to (0, 1] for any avg_logprob <= 0 and any
        scale > 0. `scale` remains an uncalibrated guess (see
        fit_confidence_calibration() above to derive it from labels)."""
        lp = (choice.get("logprobs") or {}).get("token_logprobs") or []
        values = [v for v in lp if v is not None]
        if not values:
            return 0.75  # no logprob data returned -- neutral fallback
        avg_logprob = sum(values) / len(values)
        return float(min(1.0, np.exp(avg_logprob * self.config.logprob_calibration_scale)))

    def _raw_batch_request(
        self, chunks: List[str], domain_tags: Optional[List[str]] = None, span_name: str = "extract_batch",
    ) -> Tuple[List[List], Dict[str, int]]:
        """One HTTP call for the whole batch -> SITL-filtered triples per
        chunk, WITHOUT enhance_connectivity applied. [OBSERVE]: wraps the
        call in a Langfuse span, with one nested "generation" observation
        per chunk (latency is necessarily shared across the batch since
        it's one HTTP call -- recorded as `batch_latency_s` /
        `batched_with_n_chunks` metadata on each generation rather than
        pretending each chunk has its own)."""
        domain_tags = domain_tags or ["unclassified"] * len(chunks)
        prompts = [f"{EXTRACTION_SYSTEM_PROMPT}\n\n{c}\n\nJSON:" for c in chunks]
        payload = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": self.config.max_output_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": self.config.repetition_penalty,
            "structured_outputs": {"json": self.json_schema},
            "logprobs": 1,
        }

        with self.tracer.batch_span(
            span_name, input={"n_chunks": len(chunks), "domains": domain_tags},
            metadata={"vllm_model": self.model_name},
        ) as batch_span:
            t0 = time.time()
            resp = self._requests.post(f"{self.base_url}/v1/completions", json=payload, timeout=self.request_timeout_s)
            resp.raise_for_status()
            elapsed = time.time() - t0
            choices = sorted(resp.json()["choices"], key=lambda c: c.get("index", 0))

            results: List[List] = []
            stats_total: Dict[str, int] = {}
            for i, choice in enumerate(choices):
                raw = choice["text"]
                conf = self._confidence_from_choice(choice)
                stats: Dict[str, int] = {}
                triples = parse_json_triples(
                    raw, strict=self.config.strict_field_validation,
                    max_entity_chars=self.config.max_entity_chars,
                    max_entity_words=self.config.max_entity_words, stats=stats,
                )
                normalized = [(s, sp, normalize_predicate(p), t, tp, conf) for s, sp, p, t, tp, _ in triples]
                filtered = sitl_filter(normalized, stats=stats)
                results.append(filtered)
                for k, v in stats.items():
                    stats_total[k] = stats_total.get(k, 0) + v

                domain = domain_tags[i] if i < len(domain_tags) else "unclassified"
                with batch_span.start_as_current_generation(
                    name=f"chunk_{i}", model=self.model_name,
                    input=chunks[i][:4000],
                    model_parameters={"temperature": 0.1, "top_p": 0.9},
                ) as gen:
                    gen.update(
                        output=raw[:4000],
                        metadata={
                            "domain": domain, "confidence": conf,
                            "n_triples_raw": len(triples), "n_triples_kept": len(filtered),
                            "dropped_stats": stats, "batched_with_n_chunks": len(chunks),
                            "batch_latency_s": round(elapsed, 3),
                        },
                    )
            batch_span.update(output={"total_kept": sum(len(r) for r in results)})
        return results, stats_total

    def _generate_once(self, chunks: List[str], domain_tags: Optional[List[str]] = None) -> List[List]:
        """Bare filtered triples, no stats, no enhancement -- what
        _bayesian_rerun needs for its consensus-sampling loop."""
        filtered, _ = self._raw_batch_request(chunks, domain_tags=domain_tags, span_name="extract_batch_bayesian_sample")
        return filtered

    def extract_batch(
        self, chunks: List[str], previous_entities: Optional[Set[str]] = None,
        domain_tags: Optional[List[str]] = None,
    ) -> Tuple[List[List], Dict[str, int]]:
        # previous_entities accepted, unused -- kept for interface
        # stability with StreamableKGPipeline._process_chunks' call site.
        filtered, stats_total = self._raw_batch_request(chunks, domain_tags=domain_tags)
        results = [enhance_connectivity(chunk, triples) for chunk, triples in zip(chunks, filtered)]
        return results, stats_total

    def _bayesian_rerun(self, chunks: List[str], n_samples: int = 3,
                         domain_tags: Optional[List[str]] = None) -> List[List]:
        """Low-yield rescue path: resamples n_samples times and keeps only
        triples with enough cross-sample agreement (see
        _consensus_from_runs for the Beta-Binomial voting math)."""
        all_runs = [self._generate_once(chunks, domain_tags=domain_tags) for _ in range(n_samples)]
        return _consensus_from_runs(all_runs, chunks, enhance_connectivity)


def _vllm_raw_completion(extractor: "VLLMTripleExtractor", full_prompt: str,
                          temperature: float = 0.1, use_schema: bool = True) -> str:
    """Single-prompt completion against the vLLM server, bypassing
    extract_batch's parsing/filtering -- shared by [ACTIVE-LEARN]'s
    VLLMDPODatasetBuilder (quality-score bootstrapping) and the DSPy LM
    wrapper near the bottom of this file, both of which need the RAW
    text, not filtered triples."""
    payload = {
        "model": extractor.model_name, "prompt": full_prompt,
        "max_tokens": extractor.config.max_output_tokens, "temperature": temperature,
        "top_p": 0.9, "repetition_penalty": extractor.config.repetition_penalty,
    }
    if use_schema:
        payload["structured_outputs"] = {"json": extractor.json_schema}
    resp = extractor._requests.post(f"{extractor.base_url}/v1/completions", json=payload, timeout=extractor.request_timeout_s)
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


class StreamableKGPipeline:
    def __init__(self, cfg: PipelineConfig, base_url: Optional[str] = None):
        self.config = cfg
        if base_url is None:
            _, base_url = ensure_vllm_server(cfg)

        self.tracer = LangfuseTracer(enabled=cfg.enable_langfuse)
        self.extractor = VLLMTripleExtractor(cfg, base_url=base_url, tracer=self.tracer)
        self.chunker = TextChunker(self.extractor.tokenizer, cfg.max_input_tokens, cfg.overlap_pct)
        self.store = GPUTripleStore()
        self.seen: Set[str] = set()
        self.gds = GPUDirectStorage()

        # [CLASSIFY]/[ANN] shared embedder -- loaded once, reused by both.
        # Gated on EMBEDDER_AVAILABLE alone (sentence-transformers), not
        # the faiss-inclusive ANN_AVAILABLE -- domain classification and
        # cuML-backed ANN resolution both only need an embedder, never
        # faiss, so a box without faiss installed shouldn't lose either
        # of those just because the (here, non-default) FAISS backend
        # would also be unusable.
        self.embedder = None
        self.domain_classifier = None
        if (cfg.enable_domain_classification or cfg.enable_ann_resolution) and EMBEDDER_AVAILABLE:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.embedder = SentenceTransformer(cfg.ann_embedding_model, device=device)
                if cfg.enable_domain_classification:
                    self.domain_classifier = DomainClassifier(self.embedder, DEFAULT_DOMAIN_LABELS)
            except Exception as e:
                print(f"[CLASSIFY] Shared embedder load failed ({e}) -- domain tags will read "
                      f"'unclassified'; ANN resolution can still load its own embedder lazily.")

    def process_pdf(self, pdf_bytes: bytes) -> Iterator[Dict]:
        print("[OCR/LAYOUT] Extracting sections...")
        sections = extract_sections_from_pdf(pdf_bytes)
        print(f"[OCR/LAYOUT] Sections: {len(sections)}")
        print("[CHUNK] Adaptive chunking...")
        all_chunks: List[str] = []
        for _, text in sections:
            all_chunks.extend(self.chunker.chunk_text(text))
        yield from self._process_chunks(all_chunks)

    def _process_chunks(self, chunks: List[str]) -> Iterator[Dict]:
        chunk_lengths = [len(self.extractor.tokenizer.encode(c, add_special_tokens=False)) for c in chunks]
        chunks = [c for _, c in sorted(zip(chunk_lengths, chunks))]
        total = len(chunks)
        print(f"[CHUNK] {total} chunks  |  Est. triples: {total * 5}-{total * 10}")
        pbar = tqdm(total=total, desc="[PROCESS B] Extracting", unit="chunk")
        t0 = time.time()

        for i in range(0, total, self.config.batch_size):
            batch = chunks[i: i + self.config.batch_size]
            batch_t0 = time.time()

            domain_tags = None
            if self.domain_classifier is not None:
                domain_tags = [d for d, _score in self.domain_classifier.classify_batch(batch)]

            results, extract_stats = self.extractor.extract_batch(batch, self.seen or None, domain_tags=domain_tags)

            low_yield = [j for j, r in enumerate(results) if len(r) < 3]
            if low_yield:
                rechunks = [batch[j] for j in low_yield]
                rechunk_domains = [domain_tags[j] for j in low_yield] if domain_tags else None
                improved = self.extractor._bayesian_rerun(
                    rechunks, n_samples=self.config.bayesian_rerun_samples, domain_tags=rechunk_domains,
                )
                for local_j, global_j in enumerate(low_yield):
                    if len(improved[local_j]) >= len(results[global_j]):
                        results[global_j] = improved[local_j]

            batch_triples = sum(len(r) for r in results)
            batch_conf = sum(t[5] for r in results for t in r) / max(batch_triples, 1)
            batch_time = time.time() - batch_t0

            for triples in results:
                self.store.add_triples(triples)
                for src, _, _, tgt, _, _ in triples:
                    self.seen.update([src, tgt])

            if wandb.run is not None:
                free_mem, total_mem = torch.cuda.mem_get_info()
                log_dict = {
                    "batch_idx": i // self.config.batch_size,
                    "triples_this_batch": batch_triples,
                    "cumulative_entities": len(self.seen),
                    "gpu_used_gb": (total_mem - free_mem) / 1e9,
                    "gpu_free_gb": free_mem / 1e9,
                    "avg_confidence": batch_conf,
                    "generation_time_s": batch_time,
                    "low_yield_reruns": len(low_yield),
                }
                if domain_tags:
                    for d in set(domain_tags):
                        log_dict[f"domain_count/{d}"] = domain_tags.count(d)
                log_dict.update({f"filter/{k}": v for k, v in extract_stats.items()})
                wandb.log(log_dict)

            bidx = i // self.config.batch_size
            if bidx > 0 and bidx % 5 == 0:
                self.store.flush()
                if self.store.triples_df is not None:
                    self.gds.save_checkpoint(self.store.triples_df, f"batch_{i:04d}.parquet")
            if bidx % 3 == 0:
                cleanup_memory()

            pbar.update(len(batch))
            yield {"progress": min(i + self.config.batch_size, total) / total, "entities": len(self.seen)}

        pbar.close()
        self.tracer.flush()
        print(f"[PROCESS B] Extraction: {time.time() - t0:.1f}s  |  Entities: {len(self.seen)}")

    def finalize_graph(self) -> Tuple[pl.DataFrame, Dict]:
        self.store.flush()
        raw_cudf = self.store.aggregate()
        if len(raw_cudf) == 0:
            raise ValueError("No triples extracted from the document.")

        raw_cudf = raw_cudf[raw_cudf['confidence'] >= self.config.confidence_filter]
        print(f"[VALIDATE] After confidence filter ({self.config.confidence_filter}): {len(raw_cudf)} edges")

        if len(raw_cudf) == 0:
            raise ValueError("All edges filtered. Lower confidence_filter in PipelineConfig.")

        edges_pl = pl.from_arrow(raw_cudf.to_arrow())
        print("[RESOLVE] Global entity resolution...")
        edges_pl = resolve_entities_in_edges(edges_pl, self.config.entity_resolution_threshold)
        edges_pl = edges_pl.with_columns(pl.lit("extracted").alias("edge_source"))
        print("[ANN] Semantic entity resolution...")
        edges_pl = apply_ann_resolution(edges_pl, self.config, shared_embedder=self.embedder)

        resolved_cudf = cudf.DataFrame.from_arrow(edges_pl.to_arrow())
        print("[GRAPH] Computing PageRank / degree / communities...")
        metrics = GPUGraphAnalyzer(resolved_cudf, self.config.leiden_resolution).compute_metrics()

        return edges_pl, metrics


def export_to_parquet(edges_pl: pl.DataFrame, metrics: Dict, output_dir: str = "./kg_output") -> Tuple[Path, Path, Path, Path]:
    """[STORE] the "Canonical KG" node in the architecture diagram."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    pr_pl = pl.from_arrow(metrics["pagerank"].to_arrow())
    pr = {int(k): float(v) for k, v in zip(pr_pl["vertex"].to_list(), pr_pl["pagerank"].to_list())}

    in_d: Dict[int, int] = {}
    out_d: Dict[int, int] = {}
    if "in_degree" in metrics:
        t = pl.from_arrow(metrics["in_degree"].to_arrow())
        in_d = {int(k): int(v) for k, v in zip(t["vertex"].to_list(), t["degree"].to_list())}
    if "out_degree" in metrics:
        t = pl.from_arrow(metrics["out_degree"].to_arrow())
        out_d = {int(k): int(v) for k, v in zip(t["vertex"].to_list(), t["degree"].to_list())}

    leiden_d: Dict[int, int] = {}
    if "leiden" in metrics:
        ld = pl.from_arrow(metrics["leiden"].to_arrow())
        leiden_d = {int(k): int(v) for k, v in zip(ld["vertex"].to_list(), ld["partition"].to_list())}

    src_nodes = edges_pl.select([pl.col("src_id").alias("id"), pl.col("source_str").alias("label"), pl.col("source_type").alias("type")])
    dst_nodes = edges_pl.select([pl.col("dst_id").alias("id"), pl.col("target_str").alias("label"), pl.col("target_type").alias("type")])
    unique_nodes = pl.concat([src_nodes, dst_nodes]).unique("id")

    pr_df = pl.DataFrame({"id": list(pr.keys()), "pagerank": list(pr.values())})
    ind_df = pl.DataFrame({"id": list(in_d.keys()), "in_degree": list(in_d.values())})
    outd_df = pl.DataFrame({"id": list(out_d.keys()), "out_degree": list(out_d.values())})
    leid_df = pl.DataFrame({"id": list(leiden_d.keys()), "community_id": list(leiden_d.values())})

    nodes_pl = (
        unique_nodes
        .join(pr_df, on="id", how="left").join(ind_df, on="id", how="left")
        .join(outd_df, on="id", how="left").join(leid_df, on="id", how="left")
        .with_columns([
            pl.col("pagerank").fill_null(0.0),
            pl.col("in_degree").fill_null(0).cast(pl.Int64),
            pl.col("out_degree").fill_null(0).cast(pl.Int64),
            pl.col("community_id").fill_null(-1).cast(pl.Int64),
            (pl.col("in_degree").fill_null(0) + pl.col("out_degree").fill_null(0)).alias("degree"),
        ])
    )

    edges_out = edges_pl.select([
        pl.col("src_id").alias("source"), pl.col("dst_id").alias("target"),
        pl.col("source_str").alias("source_label"), pl.col("target_str").alias("target_label"),
        pl.col("predicate_str").alias("label"), pl.col("confidence").alias("weight"),
        pl.col("count"), pl.col("edge_source"),
    ])

    np_ = out / "nodes.parquet"
    ep_ = out / "edges.parquet"
    nc_ = out / "nodes.csv"
    ec_ = out / "edges.csv"

    nodes_pl.write_parquet(str(np_))
    edges_out.write_parquet(str(ep_))
    nodes_pl.write_csv(str(nc_))
    edges_out.write_csv(str(ec_))

    n_comm = nodes_pl["community_id"].n_unique() - (1 if -1 in nodes_pl["community_id"].to_list() else 0)
    print(f"[STORE] Nodes: {len(nodes_pl)}  |  Edges: {len(edges_out)}  |  Leiden communities: {n_comm}")
    return np_, ep_, nc_, ec_


_state: Dict = {"pipeline": None}


def launch_ui():
    for d in ["./offload", "./checkpoints", "./kg_output"]:
        Path(d).mkdir(exist_ok=True)

    uploader = widgets.FileUpload(accept=".pdf", multiple=False)
    button = widgets.Button(description="\u25b6 Process PDF", button_style="success",
                             layout=widgets.Layout(width="160px", margin="8px 0"))
    status = widgets.HTML("")
    out_widget = widgets.Output()

    def on_click(b):
        with out_widget:
            out_widget.clear_output()

            # NOTE: no server shutdown/relaunch here -- StreamableKGPipeline
            # reuses the already-warm singleton server via
            # ensure_vllm_server(). Only the in-memory triple store/seen
            # set from any previous run gets cleared.
            if _state["pipeline"] is not None:
                prev = _state["pipeline"]
                if prev.store.triples_df is not None:
                    del prev.store.triples_df
                prev.store._pending.clear()
                prev.seen.clear()
                del prev
                _state["pipeline"] = None
            cleanup_memory()

            if config.enable_wandb and wandb.run is not None:
                wandb.finish()

            if not uploader.value:
                print("No PDF uploaded.")
                return

            try:
                f = uploader.value[0]
                pdf_bytes, filename = bytes(f["content"]), f["name"]
            except (KeyError, IndexError, TypeError):
                fi = list(uploader.value.values())[0]
                pdf_bytes, filename = bytes(fi["content"]), list(uploader.value.keys())[0]

            status.value = f"<b>Processing:</b> {filename}  ({len(pdf_bytes) / 1024:.1f} KB)"
            cleanup_memory()

            # Everything below -- including wandb.init() and the vLLM
            # health check StreamableKGPipeline's constructor triggers --
            # now lives inside this single try/except/finally, so a failed
            # W&B login or a not-yet-healthy server degrades to a clean
            # "Failed" status instead of an unhandled exception escaping
            # the button handler entirely.
            metrics_poller = None
            try:
                if config.enable_wandb:
                    try:
                        wandb.init(
                            project=config.wandb_project, reinit=True,
                            config={
                                "model": config.model_name, "vllm_speculative_method": config.vllm_speculative_method,
                                "vllm_quantization": config.vllm_quantization, "max_input_tokens": config.max_input_tokens,
                                "max_output_tokens": config.max_output_tokens, "batch_size": config.batch_size,
                                "overlap_pct": config.overlap_pct, "entity_resolution_threshold": config.entity_resolution_threshold,
                                "leiden_resolution": config.leiden_resolution, "confidence_filter": config.confidence_filter,
                                "strict_field_validation": config.strict_field_validation, "ann_backend": config.ann_backend,
                                "enable_ann_resolution": config.enable_ann_resolution,
                                "enable_domain_classification": config.enable_domain_classification,
                                "filename": filename,
                            },
                        )
                        # W&B auto-collects host/GPU system metrics
                        # (utilization, memory, temp, power) on init with
                        # zero extra code here.
                    except Exception as e:
                        print(f"[OBSERVE] W&B init failed ({e}) -- continuing without W&B logging.")

                pipeline = StreamableKGPipeline(config)
                _state["pipeline"] = pipeline

                if config.enable_wandb and wandb.run is not None:
                    metrics_poller = VLLMMetricsPoller(
                        pipeline.extractor.base_url, interval_s=config.vllm_metrics_poll_interval_s,
                    )
                    metrics_poller.start()

                for _ in pipeline.process_pdf(pdf_bytes):
                    pass

                edges_pl, metrics = pipeline.finalize_graph()

                n_nodes = len(pl.concat([
                    edges_pl.select(pl.col("source_str").alias("n")),
                    edges_pl.select(pl.col("target_str").alias("n")),
                ]).unique("n"))
                n_edges = len(edges_pl)
                n_unique_preds = edges_pl["predicate_str"].n_unique()
                avg_conf = float(edges_pl["confidence"].mean())
                n_communities = 0
                if "leiden" in metrics:
                    ld = pl.from_arrow(metrics["leiden"].to_arrow())
                    n_communities = int(ld["partition"].n_unique())

                if wandb.run is not None:
                    wandb.log({
                        "total_nodes": n_nodes, "total_edges": n_edges,
                        "leiden_communities": n_communities, "unique_predicates": n_unique_preds,
                        "avg_edge_confidence": avg_conf,
                    })

                np_, ep_, nc_, ec_ = export_to_parquet(edges_pl, metrics)

                for v in metrics.values():
                    del v
                del edges_pl, metrics
                cleanup_memory()

                print("\n\u2500\u2500 Output files \u2500\u2500")
                display(FileLink(str(np_), result_html_prefix="nodes.parquet \u2192 "))
                display(FileLink(str(ep_), result_html_prefix="edges.parquet \u2192 "))
                display(FileLink(str(nc_), result_html_prefix="nodes.csv  (Gephi) \u2192 "))
                display(FileLink(str(ec_), result_html_prefix="edges.csv  (Cosmograph) \u2192 "))
                status.value = "<b style='color:green'>\u2713 Complete</b>"

            except Exception as e:
                print(f"Error: {e}")
                status.value = "<b style='color:red'>\u2717 Failed</b>"
                import traceback
                traceback.print_exc()
            finally:
                if metrics_poller is not None:
                    metrics_poller.stop()
                if config.enable_wandb and wandb.run is not None:
                    wandb.finish()

    button.on_click(on_click)
    display(widgets.VBox([
        widgets.HTML("<h3 style='font-family:sans-serif;margin:4px 0'>PDF \u2192 Knowledge Graph</h3>"),
        widgets.HTML(
            "<small style='color:#555'>Qwen2.5-3B-AWQ \u00b7 vLLM (isolated venv, PagedAttention + "
            "speculative decoding) \u00b7 Leiden \u00b7 SITL ontology engine \u00b7 canonical entity IDs \u00b7 "
            "cuML ANN resolution \u00b7 Bayesian voting \u00b7 domain classification \u00b7 "
            "W&amp;B + Langfuse observability</small>"
        ),
        uploader, button, status, out_widget,
    ]))


# ============================================================================
# [ACTIVE-LEARN] Evaluation / Active Learning / Distillation -- OPTIONAL,
# not required for the main pipeline above. Adapted from the
# transformers-direct predecessor to call the vLLM HTTP endpoint via
# _vllm_raw_completion instead of a local model.generate(). The pipeline's
# own validation gates (parse_json_triples strict mode + sitl_filter)
# ARE the reward function here -- no labeled dataset needed.
# ============================================================================

class VLLMDPODatasetBuilder:
    """For the SAME chunk, generate several completions under settings
    likely to spread quality apart (JSON-schema constraint on/off, varied
    temperature), score each by what fraction of its parsed triples
    survive validation, and keep the best/worst as a chosen/rejected pair
    when the gap is large enough to be signal rather than noise."""

    def __init__(self, extractor: "VLLMTripleExtractor", min_score_margin: float = 0.25, min_attempted: int = 2):
        self.extractor = extractor
        self.min_score_margin = min_score_margin
        self.min_attempted = min_attempted

    @staticmethod
    def _quality_score(raw_text: str, cfg: PipelineConfig) -> Tuple[float, int]:
        attempted = parse_json_triples(raw_text, strict=False)
        if len(attempted) < 1:
            return 0.0, 0
        stats: Dict[str, int] = {}
        strict = parse_json_triples(raw_text, strict=True, max_entity_chars=cfg.max_entity_chars,
                                     max_entity_words=cfg.max_entity_words, stats=stats)
        normalized = [(s, sp, normalize_predicate(p), t, tp, 0.0) for s, sp, p, t, tp, _ in strict]
        survived = sitl_filter(normalized, stats=stats)
        return len(survived) / len(attempted), len(attempted)

    def generate_candidates(self, chunk: str, n: int = 4) -> List[Tuple[str, float]]:
        variants = [
            dict(use_schema=True, temperature=0.1), dict(use_schema=True, temperature=0.3),
            dict(use_schema=False, temperature=0.5), dict(use_schema=False, temperature=0.8),
        ][:max(2, n)]
        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n{chunk}\n\nJSON:"
        candidates: List[Tuple[str, float]] = []
        for v in variants:
            raw_text = _vllm_raw_completion(self.extractor, prompt, temperature=v["temperature"], use_schema=v["use_schema"])
            score, attempted = self._quality_score(raw_text, self.extractor.config)
            if attempted >= self.min_attempted:
                candidates.append((raw_text, score))
        return candidates

    def build_pair(self, chunk: str, n: int = 4) -> Optional[Dict[str, object]]:
        candidates = self.generate_candidates(chunk, n=n)
        if len(candidates) < 2:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_text, best_score = candidates[0]
        worst_text, worst_score = candidates[-1]
        if best_score - worst_score < self.min_score_margin:
            return None
        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\n{chunk}\n\nJSON:"
        return {"prompt": prompt, "chosen": best_text, "rejected": worst_text,
                "chosen_score": best_score, "rejected_score": worst_score}

    def build_dataset(self, chunks: List[str], out_path: str, n: int = 4) -> int:
        written = 0
        with open(out_path, "w") as f:
            for chunk in tqdm(chunks, desc="Building DPO pairs"):
                pair = self.build_pair(chunk, n=n)
                if pair is not None:
                    f.write(json.dumps(pair) + "\n")
                    written += 1
        print(f"DPO dataset: {written}/{len(chunks)} chunks yielded a usable pair -> {out_path}")
        return written


def run_dpo_training(
    dataset_path: str, base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "./dpo_output", lora_r: int = 16,
):
    """Skeleton for LoRA + DPO fine-tuning on the dataset
    VLLMDPODatasetBuilder.build_dataset() produces. Trains a FRESH local
    transformers copy of the base model (needs gradient access vLLM's
    inference server doesn't provide) -- backend-agnostic relative to
    whether serving runs through vLLM or not. A starting point, not a
    tuned recipe -- batch size, LoRA rank, and beta need adjusting against
    your actual dataset size and GPU memory.

    Needs: pip install trl peft datasets"""
    from transformers import AutoModelForCausalLM, AutoTokenizer as _AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb, device_map="auto")
    tokenizer = _AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")

    training_args = DPOConfig(
        output_dir=output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=5e-6, beta=0.1, num_train_epochs=1, logging_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
    )

    trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=lora_config)
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


if DSPY_AVAILABLE:
    import types

    class _VLLMExtractorLM(dspy.LM):
        """Wraps a VLLMTripleExtractor so DSPy's optimizers drive the vLLM
        server instead of an API model. forward() must return an object
        shaped like a litellm ModelResponse (.model, .choices[i].message.content)."""

        def __init__(self, extractor: "VLLMTripleExtractor"):
            super().__init__(model=f"vllm:{extractor.model_name}")
            self.extractor = extractor

        def forward(self, prompt=None, messages=None, **kwargs):
            text = messages[-1]["content"] if messages else (prompt or "")
            raw = _vllm_raw_completion(self.extractor, text, temperature=kwargs.get("temperature", 0.1), use_schema=True)
            return types.SimpleNamespace(
                model=self.model,
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=raw))],
            )

    class RawPassthroughAdapter(dspy.adapters.Adapter):
        """Only correct for single-output-field signatures like
        ExtractTriplesSignature below. DSPy's default adapters construct
        their OWN prompt asking for "[[ ## field ## ]]" markers -- a
        different, incompatible JSON shape than the {"triples": [...]}
        object the model here is already prompted (and, via
        structured_outputs.json, generation-time constrained) to emit, so
        the default adapters fail immediately. format() sends the
        extraction prompt as-is; parse() takes the raw completion as the
        single output field with no marker extraction."""

        def format(self, signature, demos, inputs):
            input_field_name = list(signature.input_fields.keys())[0]
            chunk_text = inputs[input_field_name]
            prompt = f"{signature.instructions}\n\n{chunk_text}\n\nJSON:"
            return [{"role": "user", "content": prompt}]

        def parse(self, signature, completion):
            output_field_name = list(signature.output_fields.keys())[0]
            return {output_field_name: completion.strip()}

    class ExtractTriplesSignature(dspy.Signature):
        """Extract relationship triples from text into a knowledge graph,
        as a JSON object {"triples": [...]}, using only the allowed types
        and predicates."""
        chunk_text = dspy.InputField(desc="a chunk of source document text")
        triples = dspy.OutputField(desc='JSON object: {"triples": [{"subject": ..., "subject_type": ..., "predicate": ..., "target": ..., "target_type": ...}, ...]}')

    class TripleExtractionModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract = dspy.Predict(ExtractTriplesSignature)

        def forward(self, chunk_text):
            return self.extract(chunk_text=chunk_text)


def dspy_extraction_metric(example, pred, trace=None) -> float:
    """Reuses the pipeline's own validators as the DSPy optimization
    metric -- same idea as VLLMDPODatasetBuilder._quality_score above."""
    raw = getattr(pred, "triples", "") or ""
    attempted = parse_json_triples(raw, strict=False)
    if not attempted:
        return 0.0
    stats: Dict[str, int] = {}
    strict = parse_json_triples(raw, strict=True, stats=stats)
    normalized = [(s, sp, normalize_predicate(p), t, tp, 0.0) for s, sp, p, t, tp, _ in strict]
    survived = sitl_filter(normalized, stats=stats)
    return len(survived) / len(attempted)


def optimize_extraction_prompt(extractor: "VLLMTripleExtractor", train_chunks: List[str], auto: str = "light"):
    """Runs DSPy's MIPROv2 against your own validators as the metric.
    train_chunks can be any chunks you already have from a processed PDF
    -- unlabeled is fine, the metric IS the label. Needs: pip install dspy"""
    if not DSPY_AVAILABLE:
        raise RuntimeError("dspy is required.\n  pip install dspy")

    lm = _VLLMExtractorLM(extractor)
    dspy.settings.configure(lm=lm, adapter=RawPassthroughAdapter())
    module = TripleExtractionModule()
    trainset = [dspy.Example(chunk_text=c).with_inputs("chunk_text") for c in train_chunks]
    optimizer = dspy.MIPROv2(metric=dspy_extraction_metric, auto=auto)
    return optimizer.compile(module, trainset=trainset)

# No auto-launch_ui() call at module import time (unlike the predecessors
# this was built from) -- `import kg_pipeline` here would need to happen
# BEFORE the vLLM smoke-test cell (to reach PipelineConfig /
# ensure_vllm_server), and an import-time side effect that immediately
# displays the UI widget would fire at the wrong moment in that flow. The
# notebook calls kg_pipeline.launch_ui() itself, explicitly, once the
# server smoke test has already passed.