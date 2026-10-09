# Reproduce: run each model and repeat the measurements

This folder shows how to run the five checkpoints on an 8×H100 80GB node the way the [article](../index.html) measured them: the code to run, what the server log should say, and what went wrong on the original nodes. The observations are from 2026-10-07 and 2026-10-08 on vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`). On another build, re-check the log lines: backends and even the executed layers can change.

| You want to | Read |
| --- | --- |
| Rebuild the published tables on a laptop, no GPU | [Section 1](#1-rebuild-tables-without-gpus) |
| Budget disk, GPU memory, host RAM and hours before renting | [What a reproduction needs](#what-a-reproduction-needs-disk-memory-and-time) |
| Launch one model and send it a request | That model's page, below |
| Repeat the whole measurement | Sections [2](#2-prepare-the-linux-gpu-node) to [5](#5-analyze-a-fresh-run), in order |
| Know what can go wrong before renting | [Known problems on every model](#known-problems-on-every-model), then each page's own list |
| Understand why each step exists | The [lessons](../teach_me/README.md) |

Sections 2 to 5 are recipes for a **separately authorized** GPU session. This branch was prepared without new GPU work, and nothing here authorizes a rental.

## The five models

| Key | Page | Checkpoint | Size on disk | First launch | Relaunch | Thinking off | Template adds | Node |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| `v4-0731` | [DeepSeek V4 Flash 0731](v4-0731.md) | `deepseek-ai/DeepSeek-V4-Flash-0731` | 156 GB | 11 min | 2 min | `{"thinking": false}` | 4 tokens | First |
| `v41` | [DeepSeek V4.1 Flash](v41.md) | `deepseek-ai/DeepSeek-V4.1-Flash` | 476 GB | 6 min | 2.5 min | `{"thinking": false}` | 4 tokens | First |
| `mimo-v26` | [MiMo-V2.6-Flash-MOPD](mimo-v26.md) | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | 166 GB | 7.5 min | 2 min | `{"enable_thinking": false}` | 9 tokens | First; control on the second |
| `qwen-38-bf16` | [Qwen3.8-Flash-Next](qwen-38-bf16.md) | `Qwen/Qwen3.8-Flash-Next` (original BF16) | 336 GB | 9.7 min | 4.3 min | `{"enable_thinking": false}` | 12 tokens | Second |
| `glm-53` | [GLM-5.3-Flash](glm-53.md) | `zai-org/GLM-5.3-Flash` | 306 GB | 8.5 min | 3.5 min | **not available**: `{"reasoning_effort": "low"}` | 12 tokens | First |

The **key** is the name used by every command (`diag:v4-0731`), every result folder (`reports/<study>/data/<key>/`) and every page in this folder. Each page pins the immutable revision. "Template adds" is the server-counted prompt tokens minus the tokens of the user text, measured on one 1K request. All repositories are public; no token is needed. Qwen's row was measured on a different rented node than the others (another CPU and driver), so its launch times are not directly comparable with theirs.

Architecture notes and links to each model's card and technical report are in [docs/models.md](../docs/models.md).

## What a reproduction needs: disk, memory and time

Everything in this section was read from the original October sessions on 8×H100 80GB: checkpoint sizes from the nodes' disks, memory from each server's startup log and `nvidia-smi`, and times from the saved server logs. `python tools/launch_times.py` prints the memory and time tables again from those logs. The values describe those two nodes and that build. Use them as a budget, not as a promise.

### Disk and memory per model

| Key | Checkpoint on disk | Weights, each GPU | Weights, eight GPUs | KV pool reserved, each GPU | GPU memory in use, each GPU | Host RAM besides the page cache |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `v4-0731` | 156 GB | 19.79 GiB | 158 GiB | 47.43 GiB | 75.7 GiB | None recorded |
| `v41` | 476 GB | 36.32 GiB | 291 GiB | 29.14 GiB | 76.4 GiB | About 190 GB pinned for Engram tables (11.80 GiB per rank, logged twice) |
| `mimo-v26` | 166 GB | 20.1 GiB | 161 GiB | 49.18 GiB | 73.7 GiB | None recorded |
| `qwen-38-bf16` | 336 GiB (360 GB by the hub's count) | 31.42 GiB | 251 GiB | 37.37 GiB | 74.4 GiB | The n-gram embedding sits in pinned host memory; its size was not recorded |
| `glm-53` | 306 GB | 38.8 GiB | 310 GiB | 28.73 GiB | 74.1 GiB | None recorded |
| **All five** | **about 1.44 TB** | | | | | |

How to read the columns:

- **Checkpoint on disk** is what you download. All five need about 1.44 TB, plus room for the two public datasets, the Python environments and the raw results of your own session.
- **Weights** is the server's `Model loading took … GiB` line, per GPU. Under tensor parallel 8 each GPU holds a slice of every layer, so the model occupies eight times that.
- **KV pool reserved** is the `Available KV cache memory` line. It is what remains of 0.90 × the GPU after weights and workspace, reserved at startup whether or not requests use it. It is not live state and not a measured request capacity.
- **GPU memory in use** is `nvidia-smi` on GPU 0 while one 64K request was live, out of 79.6 GiB (81,559 MiB) per H100. It is nearly the same for all five, because the pool takes whatever the weights leave. A larger model does not use more of the GPU; it leaves a smaller pool (GLM 28.73 GiB against MiMo 49.18 GiB).
- **Host RAM.** V4.1's checkpoint is 476 GB but only 291 GiB of it goes to the GPUs: its Engram tables stay in pinned host memory. Weights are also read through the page cache, which is why a relaunch is fast. Our nodes had 1,771 and 1,511 GiB of RAM, so every checkpoint stayed cached; a smaller host was not tested.

### Time per model

| Key | Download | First launch until ready | Relaunch until ready | Diagnostics step | One timing block | Three timing blocks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `v4-0731` | With MiMo and V4.1: about 800 GB in 10 min | 10.8 min | 2.0 min | 19.4 min | 15.6–15.9 min | 47 min |
| `v41` | About 4 min | 5.9 min | 2.4 min | 13.2 min | 15.3–15.4 min | 46 min |
| `mimo-v26` | See `v4-0731` | 7.5 min | 2.0 min | 18.4 min | 14.4–14.7 min | 44 min |
| `glm-53` | Not recorded | 8.4 min | 3.5 min | 15.9 min | 14.8–15.1 min | 45 min |
| `qwen-38-bf16` | 23 min at about 240 MiB/s | 9.7 min | 4.2–4.3 min | 24.9 min | 21.3–21.5 min | 64 min |

- **Until ready** is the launch command to the server log's `Application startup complete`. It can differ by a few seconds from the rounded launch times in the table of the five models, which were noted when the server first answered.
- **Diagnostics step** is one `diag:<key>` launch from the launch command to the server's last log line: first load, functional check, pilot, six live-KV snapshots, six traces and shutdown.
- **One timing block** is one `timing:<key>:<block>` launch measured the same way: relaunch, warm-up at each input length, the six unprofiled points and shutdown. The benchmark windows themselves take 6.4 to 9.9 minutes of it.
- The first four rows are from the first node and Qwen's from the second, so its row also carries that node's CPU. On the second node MiMo's first launch took 10.7 min and its control block 30.2 min.
- **Failed first launches cost time too.** A 0731 diagnostics launch died in the JIT build after 8.3 min and a GLM one after 4.7 min; both models worked on the relaunch.

### Total for all five

| Stage | Time | Basis |
| --- | ---: | --- |
| Rebuild tables and audit on a laptop, no GPU | about 4 min | Measured on a Windows laptop: the builders take seconds, the audit 3.5 min |
| Download 1.44 TB | 20 min to 1.7 h | Estimate from the two measured rates: about 800 GB in 10 min on the first node, 240 MiB/s on the second |
| Install the runtime, record the node, build inputs, review the pilot and freeze the plan | allow 30 min | Estimate; not recorded |
| Diagnostics, five models | 1.5 h | Sum of the recorded steps |
| Timing, fifteen launches | 4.1 h | Sum of the recorded blocks |
| Retries after a failed first launch | about 15 min | What 0731 and GLM cost us |
| **Servers up, five models on one node** | **about 5.9 h** | Sum of the three rows above |
| Pauses between stages | about 1 h | Gaps in the first-node session |
| **Node time to plan for** | **8 to 10 h** | Estimate with margin for a slow download or one more retry |

For comparison, the original work used two nodes: 5.6 h from the first launch to the last shutdown on the first node (servers up 4.4 h) and 2.3 h on the second (servers up 2.2 h). Running all five on one node needs no node control, because MiMo is already one of the five. Splitting across two nodes adds the MiMo control block and its trace launch, about 45 minutes, and a second setup and download.

## Common setup

```bash
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
```

All five models are launched by one script, [bench/serve.sh](../bench/serve.sh), with the same flags:

| Setting | Value | Flag |
| --- | --- | --- |
| GPUs | 8, tensor parallel 8 + expert parallel | `--tensor-parallel-size 8 --enable-expert-parallel` |
| Memory | 0.90 of each GPU | `--gpu-memory-utilization 0.90` |
| Context / sequences | 262,144 / 64 | `--max-model-len 262144 --max-num-seqs 64` |
| Prefill chunk | 8,192 tokens | `--max-num-batched-tokens 8192` (passed as an extra argument) |
| Prefix caching | off | `--no-enable-prefix-caching` (configs `off`, `off-profidle`) |
| Modality | text only | `--language-model-only` |
| Requests | temperature 0, thinking off where possible | `chat_template_kwargs`, see each page |

`bench/serve.sh` also sets the environment every launch needs. If you start `vllm serve` yourself, set the same variables first:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=3600          # a first launch can take 11 minutes
export TRITON_CACHE_DIR=/tmp/triton_cache        # local disk, see "Known problems"
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_ws
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
source "$VLLM_VENV/bin/activate"
```

There are two ways to run a model:

- **By hand**, to look at one server: `bash bench/serve.sh <key> off /tmp/serve-<key>.log --max-num-batched-tokens 8192`. It runs in the foreground. You stop it yourself ([stop and verify](#stop-the-server-and-prove-it)).
- **Through the bounded chain**, for anything that will be reported: `bash bench/blog_launch.sh <name> diag:<key>` or `timing:<key>:<block>`. Each chain step owns its server, waits for readiness, measures and stops the process group on success, error or timeout. Do not run a diagnostic chain and a timing chain at the same time.

## Known problems on every model

Each row was observed on the original nodes or is enforced by the scripts. Model-specific problems are on each model's page.

| What you see | Why | What to do |
| --- | --- | --- |
| The first launch takes 6 to 11 minutes | Weights are read from disk once and kernels are JIT-compiled once | Wait. Later launches reuse the page cache and the `/tmp` JIT caches |
| `CUDA error: invalid argument` right after `flashinfer.jit: Building JIT module …` | Eight ranks build the same module at once | **Relaunch once**; the module is then cached. The chain does this automatically. It happened to 0731 and GLM, not to the other three |
| `FileNotFoundError` on `kernel.ttir` or `kernel.ptx` during startup | On the original node `/root` was a gocryptfs FUSE mount; the eight ranks' concurrent Triton compiles race on rename visibility there. FlashInfer's JIT modules have the same problem | Keep `TRITON_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE` on local disk (`/tmp`), as `bench/serve.sh` does |
| Log lines or speeds do not match these pages | `bench/serve.sh` defaults to the older runtime in `/root/vllm` when `VLLM_VENV` is unset | `export VLLM_VENV=/root/vllm-latest` and check `python -c 'import vllm; print(vllm.__version__)'` |
| The request succeeds but the model still thinks | The thinking switch has a different name per model, and an unknown name is ignored | Use the exact `chat_template_kwargs` from the model's page; check that the answer has no reasoning text |
| Fewer text chunks than output tokens in a forced-length run | With `ignore_eos`, a model past its natural end emits tokens that produce no text | Count tokens from `usage` and from trace annotations, never from text chunks |
| The first timing block at a new shape is slower | Seen on 0731 (1K, eight clients) and GLM (one client) | Keep the block. Report mean, sample SD and median |
| `GPUs are not free; refusing to launch over another server` | A previous server is still up, or its memory is not yet released | [Stop and verify](#stop-the-server-and-prove-it); wait for 0 MiB on all eight GPUs |
| The next chain stage starts while the previous one is still running | `bench/blog_launch.sh` detaches and returns immediately | Wait for the chain's process to exit and read `results/<study>/_logs/chain.log` first |
| `dry-run` reports `differs from the reference inputs` | The code corpus is built from the node's installed Python standard library; another build of Python changes the text | Use a CPython 3.12.3 installation whose files match, and compare the corpus and request hashes with the published manifest |
| The installed packages differ from the recorded node | Wheel availability and dependency resolution move over time | Compare with `reports/<study>/study/_env/freeze.txt`. Keep a different resolution as a declared deviation, not as the same environment |
| A download is slower than expected | The node's network was the limit (about 240 MiB/s for Qwen) | Start the download first and prepare everything else while it runs |

Two rules protect the node and the bill: never kill processes with a broad pattern that can match your own shell, and always confirm shutdown before the next launch.

## 1. Rebuild tables without GPUs

Needs only Python 3.10+ and this repository.

```bash
python tools/build_blog_results.py
python tools/build_blog_page.py --check
python tools/audit_references.py
```

The builder reads the selected `serving.csv`, `components.csv` and `memory.csv` under both October studies. It generates the [five-model results](../reports/five-model/results.md), including individual blocks and the MiMo control. The page check confirms that the numbers embedded in the [article](../index.html) equal the same CSVs. The audit checks hashes, raw timing arithmetic, selection counts, generated tables, links, the repository layout and publication hygiene. Retained JSON measurements are unchanged; [provenance](../docs/provenance.md) documents filtering of shared summaries. [tools/README.md](../tools/README.md) describes each tool.

Full profiler traces and public input texts are not in Git. Published component CSVs can be reaggregated locally; generating them again from GPU kernels requires recapturing traces. Source texts can be rebuilt from pinned corpus sources and compared against the published input hashes.

## 2. Prepare the Linux GPU node

Complete local preparation before renting. Download sizes, exact model revisions and expected backend log lines are on each model's page. V4.1 also needs substantial host RAM for Engram tables (about 190 GB pinned across eight ranks); the measured node had about 1.7 TB RAM. Allow disk space for the selected checkpoints and avoid simultaneous downloads during timed measurement.

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

Download each model with the exact `hf download ... --revision ...` command on its page. The input builder needs tokenizers for the first three models even when reproducing only Qwen. If `hf` is unavailable, install the Hugging Face CLI in a separate downloader environment. Credentials, when needed, belong in environment/credential storage and never in Git.

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

Check model classes, precision, attention and MoE backends against each model's page, not only checkpoint config names. GLM's natural-answer thinking-off check is expected to fail because the feature is unavailable; check its answers and forced-256 result. One retry after a first-load JIT failure was needed for 0731 and GLM on the original node.

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

The branch's profile includes GLM in each declared block. This is a reproduction selection; the original first three rotated, and GLM was added later. The original shared addendum is linked through [provenance](../docs/provenance.md), not rewritten as a new frozen plan.

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

## Health check after any launch

```bash
KEY=v4-0731                      # or v41, mimo-v26, glm-53, qwen-38-bf16
L=/tmp/serve-$KEY.log            # chain launches: results/$BLOG_STUDY/$KEY/_server/serve-*.log
grep -c "CUDA error\|Traceback" $L                                   # expect 0
grep -m1 "non-default args" $L | grep -o "'enable_prefix_caching': [A-Za-z]*"   # expect False
grep -m1 "Chunked prefill is enabled" $L                             # expect max_num_batched_tokens=8192
grep -m1 "GPU KV cache size" $L                                      # compare with the model's page
curl -s localhost:8000/metrics | grep -E "^vllm:(num_requests_running|kv_cache_usage_perc)"   # expect 0 and 0 when idle
```

The studies use the following functional check (rendered request, three natural-EOS prompts, one forced 256-token request). For Qwen set `BLOG_STUDY=qwen-bf16-h100-v1` and `KEY=qwen-38-bf16`.

```bash
bash bench/blog_launch.sh check-$KEY diag:$KEY --diag-parts functional   # drop the flag to also run pilot, KV snapshots and traces
cat results/${BLOG_STUDY:-blog-architecture-h100-v1}/$KEY/_functional/functional.json
```

`"ok": true` means natural answers stopped by themselves with no reasoning text and the forced request returned exactly 256 tokens. GLM reports `false` by design; read its three answers instead.

## Stop the server and prove it

The chain does this for you and logs it in `results/<study>/_logs/shutdown.jsonl`. After a launch by hand:

```bash
PG=$(ps -o pgid= -p $(pgrep -f "bin/[v]llm serv[e]" | head -1) | tr -d ' ')
kill -INT -- -$PG; sleep 20; kill -KILL -- -$PG 2>/dev/null
pgrep -fc "bin/[v]llm serv[e]"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Expect `0` processes and `0 MiB` eight times. The node is billed by the hour, and a leftover server makes the next launch refuse to start.

## Before publishing edits

Commit in an isolated branch and audit a clean clone:

```bash
git clone --no-local --branch main . /tmp/five-model-audit
cd /tmp/five-model-audit
python tools/audit_references.py
```

## Adding a model to this folder

1. Add the key, checkpoint and immutable revision to `bench/serve.sh` and to `MODELS` in `bench/run_matrix.py`.
2. Copy an existing page to `reproduce/<key>.md` and fill every section from **your own** launch: download size, launch times, log lines, the request and its thinking switch, and each problem you hit. Do not copy another model's observations.
3. Add the row to the table above and the model to [docs/models.md](../docs/models.md).
4. Run `python tools/audit_references.py`. It fails if a key has no page here, or if a page does not state the checkpoint and revision that `bench/serve.sh` launches.

A new checkpoint of an already measured model is a new key and a new study, not an edit of the old one: see [lesson 6](../teach_me/06_new_checkpoint_new_study.md).
