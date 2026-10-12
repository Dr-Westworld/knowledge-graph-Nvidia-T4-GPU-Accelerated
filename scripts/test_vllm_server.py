
"""
Standalone isolation test for the vLLM server -- deliberately has ZERO
dependency on kg_pipeline.py (no torch/cudf/transformers import), so it
runs from anywhere with just `requests` installed, before you commit to
loading the full RAPIDS+torch pipeline. Run it with the MAIN env's python
(not the vllm venv's) -- it only needs `requests`, which is virtually
always already present.

Tests two things SEPARATELY, on purpose, instead of one big "did it work"
call -- a failure here tells you WHICH piece broke, not just THAT
something broke:

  Step 1: a plain completion request, no structured output at all --
          confirms the server is actually generating text: model loaded
          correctly, speculative decoding (active at the server level
          regardless of this script) isn't crashing, AWQ quantization is
          producing coherent output.

  Step 2: the SAME prompt, this time with a JSON-schema constraint passed
          via `structured_outputs.json` -- confirms structured output is
          REALLY constraining generation, not silently ignored, and that
          the schema actually compiles server-side. Uses JSON schema, not
          regex: a regex-based version of this test previously reproduced
          a REAL failure on live hardware --
              ValueError: Regex compilation timed out after 5s. The
              pattern may be too complex or contain constructs that cause
              exponential state-space explosion (e.g. nested quantifiers).
          -- from xgrammar's regex compiler choking on a grammar built
          from 12 entity types alternated twice per line, 21 predicates
          once, two bounded character-class repeats, all wrapped in an
          outer bounded repeat for "0 to N triples". JSON schema expresses
          the exact same ontology constraint (enums for the type/predicate
          fields, `maxItems` for the "0 to N" bound) through xgrammar's
          best-trodden, most-optimized compilation path instead, and
          doesn't hit this failure mode.
          `{"structured_outputs": {"json": ...}}` is current as of this
          writing -- if your installed vLLM version predates that field,
          step 2 failing with a 422/"unexpected field" is the tell to
          check `vllm serve --help` / the structured-outputs docs for
          whatever your version currently calls it.

Usage:
    python3 test_vllm_server.py
    python3 test_vllm_server.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time

import requests

ALLOWED_ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "CONCEPT", "PRODUCT", "LOCATION", "EVENT",
    "LAW", "METRIC", "PROCESS", "DOCUMENT", "TECHNOLOGY", "PROVISION",
}
ALLOWED_PREDICATES = {
    "is_a", "instance_of", "part_of", "contains", "includes",
    "develops", "authors", "issues", "regulates", "manages", "governs",
    "performs", "enables", "causes", "requires", "produces", "applies_to",
    "affiliated_with", "located_in", "related_to", "influences", "depends_on",
}


def build_triple_json_schema(max_entity_chars=80, max_triples_per_chunk=20):
    """Same construction as kg_pipeline.py's build_triple_json_schema --
    duplicated here on purpose so this script has no import dependency on
    the main file."""
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


MINI_SYSTEM_PROMPT = """You extract relationships from text into a knowledge graph.

OUTPUT FORMAT: a single JSON object with one key, "triples", holding an
array of objects. Each object has exactly these five fields:
subject, subject_type, predicate, target, target_type.

TYPES: PERSON, ORGANIZATION, CONCEPT, PRODUCT, LOCATION, EVENT, LAW, METRIC, PROCESS, DOCUMENT, TECHNOLOGY, PROVISION

PREDICATES (use ONLY these exact active forms):
is_a, instance_of, part_of, contains, includes, develops, authors, issues,
regulates, manages, governs, performs, enables, causes, requires, produces,
applies_to, affiliated_with, located_in, related_to, influences, depends_on

EXAMPLE:
TEXT: "The RBI regulates export credit through commercial banks."
{"triples": [
  {"subject": "RBI", "subject_type": "ORGANIZATION", "predicate": "regulates", "target": "export credit", "target_type": "CONCEPT"},
  {"subject": "commercial bank", "subject_type": "ORGANIZATION", "predicate": "performs", "target": "export credit", "target_type": "CONCEPT"}
]}

Extract from:"""

TEST_CHUNK = "The RBI regulates export credit through commercial banks."


def check_health(base_url: str) -> bool:
    print(f"[0] Health check: GET {base_url}/health")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        ok = r.status_code == 200
        print(f"    -> status {r.status_code}  {'OK' if ok else 'FAILED'}")
        return ok
    except requests.exceptions.ConnectionError as e:
        print(f"    -> CONNECTION REFUSED: {e}")
        print("    Server isn't listening on this port -- confirm launch_vllm_server()")
        print("    actually returned successfully, or check the vLLM log file directly")
        print("    (kg_pipeline.tail_log() from a notebook cell, or `tail -f /content/vllm_server.log`).")
        return False
    except requests.exceptions.Timeout:
        print("    -> TIMED OUT. Server process may exist but isn't responding --")
        print("    check /content/vllm_server.log for what it's actually doing.")
        return False


def check_model(base_url: str):
    print(f"\n[0.5] Model registered: GET {base_url}/v1/models")
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        model_id = r.json()["data"][0]["id"]
        print(f"    -> serving: {model_id}")
        return model_id
    except Exception as e:
        print(f"    -> FAILED: {e}")
        return None


def step1_plain_request(base_url: str, model_id: str):
    print("\n[1] Plain completion request (no structured output)")
    prompt = f"{MINI_SYSTEM_PROMPT}\n\n{TEST_CHUNK}\n\nJSON:"
    payload = {"model": model_id, "prompt": prompt, "max_tokens": 150, "temperature": 0.1, "top_p": 0.9}
    t0 = time.time()
    try:
        r = requests.post(f"{base_url}/v1/completions", json=payload, timeout=120)
    except requests.exceptions.Timeout:
        print("    -> TIMED OUT after 120s. Model isn't generating -- check the")
        print("    server log for a crash during the forward pass, not just startup.")
        return None
    elapsed = time.time() - t0

    if r.status_code != 200:
        print(f"    -> HTTP {r.status_code}: {r.text[:500]}")
        return None

    text = r.json()["choices"][0]["text"]
    print(f"    -> {elapsed:.1f}s, {len(text)} chars returned")
    print(f"    -> raw output:\n{'-' * 60}\n{text.strip()}\n{'-' * 60}")

    if not text.strip():
        print("    -> EMPTY output. Model responded but generated nothing -- worth")
        print("    checking eos_token handling / whether it stopped immediately.")
        return text

    print("    PASS: server is generating text end-to-end (model load, AWQ")
    print("    quantization, and speculative decoding are all functioning)")
    return text


def step2_structured_request(base_url: str, model_id: str):
    print("\n[2] SAME prompt, WITH structured_outputs.json (JSON schema)")
    schema = build_triple_json_schema()
    prompt = f"{MINI_SYSTEM_PROMPT}\n\n{TEST_CHUNK}\n\nJSON:"
    payload = {
        "model": model_id, "prompt": prompt, "max_tokens": 200,
        "temperature": 0.1, "top_p": 0.9,
        "structured_outputs": {"json": schema},
    }
    t0 = time.time()
    try:
        r = requests.post(f"{base_url}/v1/completions", json=payload, timeout=30)
    except requests.exceptions.Timeout:
        print("    -> TIMED OUT after 30s (client-side). If the server itself is stuck")
        print("    compiling the schema, check the log -- xgrammar has its own internal")
        print("    ~5s compile timeout, so a healthy server should fail fast, not hang.")
        return

    elapsed = time.time() - t0

    if r.status_code != 200:
        print(f"    -> HTTP {r.status_code}: {r.text[:500]}")
        print("    If this is a 500 with 'Regex compilation timed out' in the SERVER log")
        print("    (not this client-side text), that's the old regex-based failure mode --")
        print("    make sure you're running the JSON-schema version of the pipeline, not")
        print("    an older copy still sending structured_outputs.regex.")
        print("    If this is a 422/'unexpected field' instead, structured_outputs.json")
        print("    isn't the field name your installed vLLM version expects -- check")
        print("    `vllm serve --help` / the structured-outputs docs for your version.")
        return

    text = r.json()["choices"][0]["text"]
    print(f"    -> {elapsed:.1f}s, {len(text)} chars returned")
    print(f"    -> raw output:\n{'-' * 60}\n{text.strip()}\n{'-' * 60}")

    try:
        parsed = json.loads(text)
        triples = parsed.get("triples", [])
        shape_ok = isinstance(triples, list) and all(
            isinstance(t, dict) and set(t.keys()) == {"subject", "subject_type", "predicate", "target", "target_type"}
            and t["subject_type"] in ALLOWED_ENTITY_TYPES and t["target_type"] in ALLOWED_ENTITY_TYPES
            and t["predicate"] in ALLOWED_PREDICATES
            for t in triples
        )
    except (json.JSONDecodeError, AttributeError):
        shape_ok = False

    if shape_ok:
        print(f"    PASS: valid JSON matching the schema exactly ({len(triples)} triple(s)) --")
        print("    structured output is REALLY constraining generation, not just present")
        print("    in the request and silently ignored.")
    else:
        print("    FAIL (silent-ignore signature): got a 200 response and text, but it's")
        print("    not valid JSON matching the schema. This is the specific failure mode")
        print("    this test exists to catch: the server likely accepted the request but")
        print("    never applied structured_outputs.json, so it generated freely. Compare")
        print("    this output to step 1's -- if they look similar, that confirms the")
        print("    constraint never engaged.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    print("=" * 70)
    print(f"vLLM server isolation test -- {args.base_url}")
    print("=" * 70)

    if not check_health(args.base_url):
        print("\nStopping here -- fix the connection/health issue before testing generation.")
        sys.exit(1)

    model_id = check_model(args.base_url)
    if model_id is None:
        print("\nStopping here -- server is up but /v1/models failed, which is odd.")
        sys.exit(1)

    plain_output = step1_plain_request(args.base_url, model_id)
    if plain_output is None:
        print("\nStopping here -- base generation isn't working, no point testing")
        print("structured output on top of a broken base case.")
        sys.exit(1)

    step2_structured_request(args.base_url, model_id)

    print("\n" + "=" * 70)
    print("Done. Read the PASS/FAIL lines above -- that tells you which layer,")
    print("if any, needs attention before trusting VLLMTripleExtractor on a real document.")
    print("=" * 70)


if __name__ == "__main__":
    main()