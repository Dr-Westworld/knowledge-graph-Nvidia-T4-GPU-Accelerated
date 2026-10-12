# Profiling artifacts

The repository includes the two supplied Nsight Systems reports:

| Artifact | Process | Size | SHA-256 |
|---|---|---:|---|
| [`vllm_profile.nsys-rep`](../profiling/vllm_profile.nsys-rep) | Isolated vLLM server running Qwen2.5-3B-Instruct-AWQ | 8,907,189 bytes | `da72f3f925fc133127efe4523eb75b0d5bbcca5b1d25f2158cada10e73a5f419` |
| [`kg_pipeline_profile.nsys-rep`](../profiling/kg_pipeline_profile.nsys-rep) | Main pipeline process (`profile_kg_pipeline.py`) | 4,365,937 bytes | `69eed38d9669825719ada65c49ca38105e0caa33787c8cbc86ee2dd1c1264` |

The reports were produced in a Colab-style Linux environment with Nsight Systems `2026.4.1.191-264138605071v0`, CUDA 12.8 libraries, and a T4 runtime. They are binary evidence artifacts, not generated benchmark summaries. A full kernel table should be generated with the matching Nsight CLI/GUI rather than inferred from file strings.

## Open in the Nsight Systems GUI

Install Nsight Systems on a CUDA workstation, then open each `.nsys-rep` in the GUI. The most useful views are:

1. CUDA GPU trace and CUDA API rows;
2. process/thread lanes around vLLM request handling;
3. kernel summary grouped by name;
4. memory operations and synchronization gaps;
5. CPU↔GPU overlap between Process A and Process B.

## Export a machine-readable summary

With the `nsys` CLI available:

```bash
nsys stats --report cudaapisum,gpukernsum,osrtsum \
  profiling/vllm_profile.nsys-rep

nsys stats --report cudaapisum,gpukernsum,osrtsum \
  profiling/kg_pipeline_profile.nsys-rep
```

If your installed Nsight version uses a different report name, run `nsys stats --help` and choose the closest CUDA API, GPU kernel, and OS runtime summaries. Keep exported text/CSV summaries next to the report with the Nsight version and command recorded.

## Re-run profiling

The notebook’s one-off vLLM path uses a throwaway port and explicitly avoids putting Nsight in the always-on observability path:

```bash
nsys profile --force-overwrite=true \
  -o vllm_profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --duration=300 \
  /content/vllm_env/bin/vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
  --port 8001 \
  --quantization awq
```

For the main pipeline, profile the exact script and input used for the experiment, and record the PDF characteristics, GPU, driver, CUDA, package versions, batch size, chunk count, and whether throughput mode was enabled. Without that context, a trace is difficult to compare across runs.

## Privacy note

Nsight reports can preserve command lines, paths, environment variables, model names, and notebook metadata. Review them before publishing outside the intended audience. The supplied artifacts are included because the project request explicitly asked to use them; do not upload new traces containing secrets or private documents.
