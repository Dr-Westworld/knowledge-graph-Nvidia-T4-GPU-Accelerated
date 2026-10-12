"""Fast, GPU-free integrity checks for the v1 repository."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kg_pipeline_v1.ipynb"


def notebook_cell_source(cell_index: int, magic: str) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][cell_index]["source"])
    expected_prefix = f"%%writefile {magic}\n"
    if not source.startswith(expected_prefix):
        raise AssertionError(f"cell {cell_index} does not start with {expected_prefix!r}")
    return source[len(expected_prefix) :]


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 44

    expected_pipeline = notebook_cell_source(15, "kg_pipeline.py")
    expected_test = notebook_cell_source(16, "test_vllm_server.py")
    assert (ROOT / "kg_pipeline.py").read_text(encoding="utf-8") == expected_pipeline
    assert (ROOT / "scripts" / "test_vllm_server.py").read_text(encoding="utf-8") == expected_test

    ast.parse(expected_pipeline)
    ast.parse(expected_test)
    print("v1 notebook/source parity: OK")
    print("Python syntax: OK")
    print("GPU-free repository validation: PASS")


if __name__ == "__main__":
    main()
