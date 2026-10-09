# Reproduce the measurements

There are two tasks: regenerate published numbers locally, or collect a fresh GPU reproduction. The first needs only Python 3.10+ and this repository. The second needs a separately authorized Linux 8×H100 80GB SXM node. This branch was prepared without new GPU work.

## 1. Rebuild tables without GPUs

```bash
python tools/build_blog_results.py
python tools/audit_references.py
```

The builder reads the selected `serving.csv`, `components.csv` and `memory.csv` under both October studies. It generates [five-model results](../reports/five-model/results.md), including individual blocks and the MiMo control. The audit checks hashes, raw timing arithmetic, selection counts, generated tables, links and publication hygiene. Retained JSON measurements are unchanged; [provenance](provenance.md) documents filtering of shared summaries.

Full profiler traces and public input texts are not in Git. Published component CSVs can be reaggregated locally; generating them again from GPU kernels requires recapturing traces. Source texts can be rebuilt from pinned corpus sources and compared against the published input hashes.

## 2. Prepare the Linux GPU node

Complete local preparation before renting. Download sizes, exact model revisions and expected backend log lines are in [model-setup.md](model-setup.md). V4.1 also needs substantial host RAM for Engram tables (about 190 GB pinned across eight ranks); the measured node had about 1.7 TB RAM. Allow disk space for the selected checkpoints and avoid simultaneous downloads during timed measurement.

```bash
nvidia-smi
nvidia-smi topo -m
lscpu
free -h
df -h /workspace
```

Install the exact October runtime into its own environment. These are the recorded installation settings; wheel availability and dependency resolution may change, so compare the resolved environment with the published node records.

```bash
python3 -m venv /root/vllm-latest
/root/vllm-latest/bin/python -m pip install --upgrade uv
/root/vllm-latest/bin/uv pip install --python /root/vllm-latest/bin/python \
  'vllm==0.31.1rc1.dev50+g554340f3d' pandas \
  --torch-backend=auto --index-strategy unsafe-best-match --prerelease=allow \
  --extra-index-url https://wheels.vllm.ai/554340f3d3259e321be4c07282be7a02a5aeef83
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
export PATH="$VLLM_VENV/bin:$PATH"
python -c 'import vllm, torch; print(vllm.__version__); print(torch.__version__)'
```

Expect vLLM `0.31.1rc1.dev50+g554340f3d` and the recorded torch `2.13.0+cu132`. Preserve a different resolution as a declared deviation or a separate runtime arm; do not label it the same environment. [install.sh](../install.sh) also records the older runtime used by the previous studies; it installs two environments, so it is not required for an October-only reproduction.

Download each model with its exact `hf download ... --revision ...` command in [model setup](model-setup.md). The input builder needs tokenizers for the first three models even when reproducing only Qwen. If `hf` is unavailable, install the Hugging Face CLI in a separate downloader environment. Credentials, when needed, belong in environment/credential storage and never in Git.

## 3. Rebuild public inputs

The commands below use the completed study identities only as reproduction profiles **in a fresh checkout with no existing `results/`**. Do not overwrite or publish fresh measurements into the completed study directories. For a new question or combined five-model session, declare a new study entry in `STUDIES` and the report model map, then freeze its own plan and cost.

The code corpus builder reads the local Python standard library. The measured source was CPython 3.12.3 under `/usr/lib/python3.12`; distro patches can change its files. Use a matching installation for the scratch corpus environment and verify corpus/request hashes against the published manifests. A version label alone is insufficient. Both reproduction profiles now check the published request hashes in `dry-run`.

```bash
export BLOG_STUDY=blog-architecture-h100-v1
hf download EleutherAI/hendrycks_math --repo-type dataset --revision 21a5633873b6a120296cce3e2df9d5550074f4a3
hf download OpenAssistant/oasst1 --repo-type dataset --revision fdf72ae0827c1cda404aff25b6603abec9e3399b
python3 -m venv /tmp/blog-inputs-venv
/tmp/blog-inputs-venv/bin/pip install pyarrow huggingface_hub
/tmp/blog-inputs-venv/bin/python bench/blog_corpus.py
$VLLM_VENV/bin/python bench/blog_inputs.py
$VLLM_VENV/bin/python bench/blog_inputs.py extra glm-53
$VLLM_VENV/bin/python bench/blog_study.py env
```

For the Qwen profile, repeat corpus/input generation with `BLOG_STUDY=qwen-bf16-h100-v1`, use `blog_inputs.py extra qwen-38-bf16`, and record the node again. Keep the first-node manifest in `reports/blog-architecture-h100-v1/study/_inputs/manifest.json`: the Qwen dry run compares every request-list and request hash against it.

## 4. Diagnostics, plan and timing

Run these stages sequentially. `blog_launch.sh` detaches the bounded chain and returns before it finishes. **Wait for its process to exit, inspect the log and verify shutdown before starting the next stage.** Every step owns its server process group and attempts cleanup on success, error and timeout. Never kill a process using a broad pattern that can match your shell.

```bash
export BLOG_STUDY=blog-architecture-h100-v1
bash bench/blog_launch.sh first-diagnostics diag:v4-0731 diag:mimo-v26 diag:v41 diag:glm-53
tail -f results/$BLOG_STUDY/_logs/chain.log
# After the chain exits, verify functional/pilot/trace/KV outputs and shutdown:
nvidia-smi --query-gpu=memory.used --format=csv,noheader
python bench/blog_report.py pilot
```

Check model classes, precision, attention and MoE backends against [model setup](model-setup.md), not only checkpoint config names. GLM's natural-answer thinking-off check is expected to fail because the feature is unavailable; check its answers and forced-256 result. One retry after a first-load JIT failure was needed for 0731 and GLM on the original node.

For a protocol reproduction, freeze the recorded counts. A new experiment must justify counts and cost from its own pilot.

```bash
$VLLM_VENV/bin/python bench/blog_study.py plan \
  --counts '{"16k:c1":36,"16k:c8":75,"1k:c1":48,"1k:c8":204,"64k:c1":18,"64k:c8":33}' \
  --note 'Fresh reproduction of selected deployments; all models use three blocks. Record node, order, deviations and estimated node-hours here.'

steps=(timing:v4-0731:1 timing:mimo-v26:1 timing:v41:1
       timing:mimo-v26:2 timing:v41:2 timing:v4-0731:2
       timing:v41:3 timing:v4-0731:3 timing:mimo-v26:3
       timing:glm-53:1 timing:glm-53:2 timing:glm-53:3)
$VLLM_VENV/bin/python bench/blog_study.py dry-run "${steps[@]}"
# Proceed only after DRY RUN OK:
bash bench/blog_launch.sh first-timing "${steps[@]}"
```

The branch's profile includes GLM in each declared block. This is a reproduction selection; the original first three rotated, and GLM was added later. The original shared addendum is linked through [provenance](provenance.md), not rewritten as a new frozen plan.

For Qwen on the second-node profile, with rebuilt inputs and inventory:

```bash
export BLOG_STUDY=qwen-bf16-h100-v1
bash bench/blog_launch.sh qwen-diagnostics diag:qwen-38-bf16
# Wait, inspect the pilot, then freeze the same counts using blog_study.py plan above.
steps=(timing:qwen-38-bf16:1 timing:mimo-v26:1 timing:qwen-38-bf16:2 timing:qwen-38-bf16:3)
$VLLM_VENV/bin/python bench/blog_study.py dry-run "${steps[@]}"
bash bench/blog_launch.sh qwen-timing "${steps[@]}"
# After timing has finished and the server has stopped:
bash bench/blog_launch.sh mimo-trace-control diag:mimo-v26 --diag-parts functional,kv,trace
```

MiMo is a control, not a replacement fifth model. Do not drop its timing block or trace control when placing new-node evidence beside the first node. Record a fresh `_env/` and inspect differences before comparing metrics.

## 5. Analyze a fresh run

Only after each chain finishes:

```bash
python bench/blog_report.py serving
python bench/blog_report.py memory
python bench/blog_report.py components
python bench/blog_report.py tables
# Second-node profile only, after both serving CSVs exist:
BLOG_STUDY=qwen-bf16-h100-v1 python bench/blog_report.py compare
```

These commands write the current checkout's report files. Use a separate reproduction checkout and preserve original evidence. `components` requires full local traces. `compare` reports only the MiMo node control in this branch. Curate any newly authorized study separately with `blog_report.py publish`; do not overwrite this branch's completed evidence or its provenance hashes.

Record every failure and rerun, actual timing windows, trace overhead and shutdown. Preserve plain-launch timing separately from diagnostics. Do not infer HBM traffic from nominal parameter counts or treat token-pool capacity as measured request capacity.

Before publishing edits, commit in an isolated branch and audit a clean clone:

```bash
git clone --no-local --branch blog/five-model-reproduction . /tmp/five-model-audit
cd /tmp/five-model-audit
python tools/audit_references.py
```
