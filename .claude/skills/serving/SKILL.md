---
name: serving
description: Serve and measure one of the five pinned checkpoints (DeepSeek V4 Flash 0731, DeepSeek V4.1 Flash, MiMo-V2.6-Flash-MOPD, Qwen3.8-Flash-Next BF16, GLM-5.3-Flash) with vLLM on an 8×H100 node, the way this repository does it. Use when asked to launch a model, check that a server is healthy, run a diagnostics or timing step, stop a server, or publish a finished step. Not for new GPU rentals or new experiments, which need the user's explicit authorization.
---

# Serving the five models on 8×H100

This skill is the short operating procedure. The reasons are in [teach_me/](../../../teach_me/README.md), the per-model details in [reproduce/](../../../reproduce/README.md), and the rules in [AGENTS.md](../../../AGENTS.md). Read `AGENTS.md` and `target.md` first: they decide what may run.

## Before anything runs

1. **Authorization.** No rental, measurement, prefix-cache experiment or speculative sweep without the user's explicit go-ahead. Prefix caching and speculation stay off in every blog launch.
2. **One study, one build.** A study has one frozen vLLM commit. Never upgrade inside a study and never use a number from one build as a control for another. The builds are listed in [reports/README.md](../../../reports/README.md).
3. **Credentials** come from the environment or the node's credential storage (`HF_TOKEN`, a Git credential helper). Never write a token into a file of this repository; the audit scans for them.

## Set up the node

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv   # 8 GPUs, 0 MiB used
df -h /workspace; free -g                                                  # 1.5 TB for all five, 190 GB RAM for V4.1
python3 -m venv /workspace/vllm-rerun
uv pip install --python /workspace/vllm-rerun/bin/python "vllm==<frozen version>" pandas requests \
  --torch-backend=auto --index-strategy unsafe-best-match --prerelease=allow \
  --extra-index-url https://wheels.vllm.ai/<frozen commit>
export VLLM_VENV=/workspace/vllm-rerun HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
export BLOG_STUDY=<study id>          # for the rerun session: five-model-rerun-h100-v1
```

- Put the environment, the weights and `results/` on **local disk**. On the rented nodes `/root` is an encrypted FUSE mount: slow for traces, and eight ranks compiling the same kernel there can race.
- Download in measurement order with the pinned revision from the model's page: `hf download <repo> --revision <40-char sha>`. Two downloads side by side used the link well (1.44 TB in about 16 minutes on the session node).
- Build inputs once per study: `bench/blog_corpus.py` (scratch venv with `pyarrow`), then `bench/blog_inputs.py`. Compare the corpus hashes with the published manifest before trusting that the request texts are the same.
- Record the node and freeze the plan before the first launch: `bench/blog_study.py env`, `plan --counts ...`, `dry-run <steps>`. Proceed only on `DRY RUN OK`.

## Launch

Everything that will be reported goes through the bounded chain, which owns its server:

```bash
bash bench/blog_launch.sh <name> diag:<key>            # idle-profiler launch: functional check, pilot, snapshots, traces
bash bench/blog_launch.sh <name> timing:<key>:<block>  # plain launch: the unprofiled points of one block
tail -f results/$BLOG_STUDY/_logs/chain.log
```

Keys: `v4-0731`, `v41`, `mimo-v26`, `qwen-38-bf16`, `glm-53`. `bench/serve.sh` holds each key's checkpoint, revision and parsers, and the common flags: TP8 plus expert parallel, utilization 0.90, context 262144, at most 64 sequences, 8,192-token chunk budget, prefix caching off, text only. To look at one server by hand: `bash bench/serve.sh <key> off /tmp/serve-<key>.log --max-num-batched-tokens 8192` (foreground; you stop it).

`blog_launch.sh` detaches and returns at once. **Wait for the chain to exit before starting another one**, and never run two chains together.

### What a first launch looks like

A first launch of a model on a build compiles kernels and can take 10 to 25 minutes; a relaunch reuses the caches and takes 2 to 5. Seeing these is normal, not a hang:

| Log line | Meaning |
| --- | --- |
| `Capturing CUDA graphs (PIECEWISE)` at 20 to 35 s per item | First-time graph compilation |
| `JIT kernel warmup (N compile keys)`, `DeepGEMM warmup` | Kernel JIT; on the session build 0731 compiled 232 TileLang keys and 1,261 DeepGEMM kernels |
| `No available shared memory broadcast block found in 60 seconds` | The engine is waiting for the workers that are compiling |
| `CUDA error: invalid argument` right after `Building JIT module` | Eight ranks built one module at once. Relaunch once; the chain does it |

Keep every JIT cache on local disk; `bench/serve.sh` sets `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `TILELANG_CACHE_DIR` and `VLLM_CACHE_ROOT` under `/tmp`.

## Check that the server is what you think it is

```bash
L=$(ls -t results/$BLOG_STUDY/<key>/_server/serve-*.log | head -1)
grep -c "CUDA error\|Traceback" $L                                             # 0
grep -m1 "non-default args" $L | grep -o "'enable_prefix_caching': [A-Za-z]*"  # False
grep -m1 "Chunked prefill is enabled" $L                                       # max_num_batched_tokens=8192
grep -m1 "Model loading took" $L; grep -m1 "Available KV cache memory" $L      # weights and reserved pool per GPU
grep -m1 "MoE backend\|kv cache block size" $L                                 # executed kernels and KV format
curl -s localhost:8000/metrics | grep -E "^vllm:(num_requests_running|kv_cache_usage_perc)"   # 0 and 0 when idle
```

Read the model class, precision, attention and expert backends from the log and compare them with the model's page. A config field or a paper feature does not prove execution.

Requests: temperature 0, one user message, the server applies the chat template. Thinking off uses a different switch per model (`thinking` for DeepSeek, `enable_thinking` for MiMo and Qwen); an unknown name is silently ignored. GLM has no off switch and runs with `reasoning_effort=low`.

## Measure

- **Timing** comes only from plain launches (`off`). **Traces** come only from the idle-profiler launch (`off-profidle`). Keep them in separate tables: a kernel sum is not a latency.
- **Count tokens from the server**, never from streamed text chunks: `usage` in the response, and the `vllm:generation_tokens_total`, `vllm:time_to_first_token_seconds_count` and `vllm:iteration_tokens_total` counters. With `ignore_eos` a model past its natural end emits tokens that produce no text.
- **Client `c` is not engine batch `B`.** For a decode step at a known batch, admit B requests together, wait until the server has counted B first tokens, admit nothing else, then read the counters at both ends of the interval.
- A run is valid when every request completed with exactly the forced output length, none failed, and the server counted at least 98% of the expected prompt tokens. Keep invalid and slow runs; never select the fastest attempt.
- Live state is the usage gauge times the per-rank pool: an approximation of bytes held by live requests. It is not the reserved pool, not a request capacity and not reusable-prefix capacity.

## Stop the server and prove it

The chain stops its own server and writes `results/<study>/_logs/shutdown.jsonl`. After a launch by hand:

```bash
PG=$(ps -o pgid= -p $(pgrep -f "bin/[v]llm serv[e]" | head -1) | tr -d ' ')
kill -INT -- -$PG; sleep 20; kill -KILL -- -$PG 2>/dev/null
pgrep -fc "bin/[v]llm serv[e]"; nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 0, then 0 MiB eight times
```

Never kill by a broad pattern that can match your own shell. The node is billed by the hour and a leftover server makes the next launch refuse to start.

## Publish a finished step

Raw runs live in the git-ignored `results/`. In the rerun session the chain publishes after each step when started with `--publish`; by hand:

```bash
python bench/session_publish.py diag:<key>      # tables, curated evidence, timeline, ledger, commit, push
python tools/audit_references.py                # on a clean clone, before calling anything published
```

The publisher copies reviewed runs into `reports/<study>/data/<key>/` (logs with private addresses masked, traces excluded), rewrites the study's hash ledger and `timeline.md`, and pushes. Raw measurements and ledgers are never edited by hand, derived tables are regenerated by their tool, and a new measurement always gets a new study identity.

## Problems met in the rerun session, and how each was fixed

These happened on 2026-10-10 on vLLM `a98247ab4`. Each row says how the problem was noticed, what the cause was, what was changed and where the change lives, so the same fix is not worked out twice. Fix the cause, keep the failed attempt in the evidence, and push the fix only after a run has shown that it works.

| What was seen | Cause | Fix, and where it lives |
| --- | --- | --- |
| Pilot runs invalid: `output tokens 768 != 6144` | `vllm bench serve` overrides the request list's `output_tokens` with its own default, `--custom-output-len 256` | Pass the forced length explicitly (`bench_cmd(..., osl=)` in `bench/run_matrix.py`). The chain was stopped, the four invalid runs kept, and the pilot repeated. The validity check is what caught it: never loosen it |
| First launch of 0731 took 16 minutes instead of 11 | This build adds a TileLang warm-up (232 compile keys) and a DeepGEMM warm-up (1,261 kernels), and their caches defaulted to the home directory on the encrypted FUSE mount | Wait; it is not a hang. `bench/serve.sh` now sets `TILELANG_CACHE_DIR` and `VLLM_CACHE_ROOT` under `/tmp`. The relaunch was ready in 90 seconds |
| GLM's first launch died after 16 minutes with `CUDA error: invalid argument` | Eight ranks finished building the FlashInfer `fused_moe_90` module at once | Nothing to change: the chain relaunches once after a failure before readiness, and the module is cached. Ready in 375 seconds on the relaunch |
| Trace and result files were slow to write | `results/` was on the FUSE mount | Move it to local disk and leave a symlink (`results -> /workspace/...`). `.gitignore` needs `/results` without the slash, because a symlink is not a directory |
| `git push` rejected in the middle of the session | Someone else pushed to `main` | `git pull --rebase --autostash origin main`, then push. `bench/session_publish.py` now does this by itself and commits only `reports/<study>/` |
| The audit rejected the tree | A stray file at the root, and a new top-level folder for skills | Remove the stray file; register a new top-level entry in `tools/audit_references.py` and in the README map in the same commit |
| Live state at 1K looked 2.5 times too large for MiMo | Read right at the first token, window layers still hold the whole prompt chunk | Read the gauge a few steps into decode (`kv_usage_frac_decoding` in `bench/blog_report.py`); keep the first-token reading beside it |
| Streamed chunk counts did not equal tokens | With `ignore_eos` some tokens produce no text, and a chunk can carry several tokens | Read progress from the server's counters (`counters()` and `Live.window()` in `bench/blog_study.py`) |
| A version lookup ran beside live requests | `runtime()` imports vLLM in a subprocess, seconds of CPU each time | Read it once at chain start and cache it (`bench/blog_study.py`) |
| All-reduce time ten times higher in the 1K and 16K prefill captures | Those captures consist of the first step after the profiler starts, where ranks wait for each other | Leave communication out of the kernel totals (`tools/build_rerun_results.py`) and compare it only at 64K and 128K. A primer request before the capture would be the better fix for a next session |
| New kernels landed in "other" | Kernel names changed with the build | Read the rank-0 kernel inventory, add rules to `EXTRA` in `bench/blog_report.py`, raise `CLASSIFIER`, and let the publisher rebuild at a step boundary (`touch results/<study>/_logs/recompute_components`), never beside a timing run |
| Per-rank profiler summaries were about to be published eight times | The build names them `..._rankN.profiler_out.txt` | `bench/curate.py` keeps rank 0 and excludes the others, as before |
| `ncu` says `ERR_NVGPUCTRPERM` | `RmProfilingAdminOnly: 1` and a container in a user namespace: only the host can allow performance counters | Cannot be fixed from inside the rental. Record it (`_logs/hbm_counter_attempt.txt`) and use the labelled estimate (`tools/estimate_state_traffic.py`). Check this **before** paying for a node |
| `--enable-return-routed-experts` refuses to start | The capture requires prefix caching and Model Runner V2 | Not enabled, because the plan keeps prefix caching off in every launch; recorded in `_logs/routing_attempt.txt`. It needs the user's decision, not a workaround |
| A command of the agent waited more than an hour on a permission prompt | It touched a path outside the working directory | Nothing was lost, because measurements run in the detached chain and `bench/session_run.sh`, not in the agent's shell. Keep it that way |

Two habits behind these fixes: a check that fails is information, so read why before changing anything; and a fix that makes evidence pass by weakening a check (a looser validity rule, a narrower audit scan) is not a fix. Report it instead.

## When something is off

| Symptom | Likely cause | Do |
| --- | --- | --- |
| `GPUs are not free; refusing to launch` | A previous server or its memory is still there | Stop and verify, wait for 0 MiB on all eight |
| Speeds or log lines differ from the pages | Wrong environment: `VLLM_VENV` unset selects an older build | `python -c 'import vllm; print(vllm.__version__)'` |
| The answer contains reasoning | Wrong thinking switch for this model | Use the `chat_template_kwargs` from the model's page |
| A run at many clients is slow to start | Prefill of all admitted prompts comes first; queueing is valid, a failure is not | Read `running`, `waiting` and preemptions in the run's telemetry |
| The first block at a new shape is slower | First-use compilation or allocation | Keep the block; report mean, SD and the individual blocks |
