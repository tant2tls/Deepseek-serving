---
name: serving
description: Serve and measure DeepSeek V4 Flash 0731, DeepSeek V4.1 Flash, MiMo-V2.6-Flash-MOPD, Qwen3.8-Flash-Next or GLM-5.3-Flash with vLLM on a rented 8×H100 node in this repository. Use when asked to set up a fresh node, launch or stop one of these servers, run a timing or diagnostic chain, or check why a launch failed.
---

# Serving the three models on 8×H100

GPUs are billed by the hour. Do everything that needs no GPU first, never leave a ready server idle, and stop the server you started on every exit path.

## 1. Read before acting

- `AGENTS.md` for the current scope and standing rules. Prefix caching stays **off** unless Tan explicitly asks; speculation is off for the blog study.
- `blog_target.md` for the blog study, `target.md` for model revisions and the later speculative study.
- `docs/reproduce.md` section 0 for the verified fresh-node procedure and timings.

## 2. Fresh node checklist (no GPU needed)

1. Inventory: `nvidia-smi`, `df -h /workspace`, `ls /workspace/hf/hub`, `ls /root/vllm /root/vllm-latest`. Assume nothing survived from the last rental.
2. Runtime: `bash install.sh`. Two venvs exist on purpose:
   - `/root/vllm`: pinned build `44af287e…` of the completed studies (default).
   - `/root/vllm-latest`: build used by `blog-architecture-h100-v1`. Select with `export VLLM_VENV=/root/vllm-latest`.
3. Weights: `export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1`, then `hf download <model> --revision <sha> --max-workers 32` for each revision in `target.md`. The repositories are public. Smallest first (0731, MiMo, then V4.1 at about 476 GB).
4. Inputs: `bench/blog_corpus.py` in a scratch venv with `pyarrow`, then `bench/blog_inputs.py` with the vLLM venv.
5. Record the node: `$VLLM_VENV/bin/python bench/blog_study.py env`.

## 3. Launching

Prefer the chain; it owns the server and cleans up:

```bash
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf
bash bench/blog_launch.sh <name> diag:<model> ...            # functional, pilot, live KV, traces
bash bench/blog_launch.sh <name> timing:<model>:<block> ...  # six unprofiled points of one block
tail -f results/blog-architecture-h100-v1/_logs/chain.log
```

Model keys: `v4-0731`, `v41`, `mimo-v26`, and for the blog study only `qwen-38` and `glm-53` (GLM has no thinking-off switch; it runs with `reasoning_effort=low`). Server configs allowed in the blog study: `off` (timing) and `off-profidle` (idle torch profiler for diagnostics). Both pass `--no-enable-prefix-caching`.

A manual launch, only when the chain does not fit:

```bash
PROFILE_DIR=$PWD/results/<study>/<model>/profiles \
  bash bench/serve.sh <model> off <log> --max-num-batched-tokens 8192
curl -s localhost:8000/v1/models        # ready when the model id is listed
```

Common deployment, do not change inside a study: TP 8 + expert parallel, memory utilization 0.90, `max_model_len` 262144, `max_num_seqs` 64, chunk budget 8192, native precision, text only, temperature 0, thinking off (`thinking=false` for DeepSeek, `enable_thinking=false` for MiMo).

## 4. Check often while it runs

- Readiness is bounded (30 min). A first launch of a model takes 6–11 minutes (weights from disk plus JIT; Qwen took 19); later launches take about 2. The chain relaunches once by itself if the first launch dies before readiness.
- Watch `chain.log` for `ready`, `valid=`, `FAILED`, `stopped`. A run with `valid=False` is kept; rerun into a new directory, never overwrite.
- Before trusting numbers, read `<model>/_functional/functional.json`: natural-EOS answers stop, no reasoning text, forced request returns exactly 256 tokens.
- Client concurrency is not the engine batch. Read the actual batch from the trace annotations or the running-requests gauge.

## 5. Stopping, always

The chain stops its own process group and writes `_logs/shutdown.jsonl`. After any manual launch:

```bash
kill -INT -- -<pgid>; sleep 20; kill -KILL -- -<pgid> 2>/dev/null
pgrep -af "[v]llm serve"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Confirm no `vllm serve` process and 0 MiB on all eight GPUs before reporting that the server is stopped.

## 6. Known failures

| Symptom | Cause | Action |
| --- | --- | --- |
| `CUDA error: invalid argument` during the first launch of a model, right after `flashinfer.jit: Building JIT module …` | Eight ranks JIT-build the same module concurrently | Relaunch once; the module is cached under `/tmp/flashinfer_ws` |
| `FileNotFoundError` in Triton or TileLang compile | `/root` is a FUSE mount and concurrent compiles race | Keep JIT caches on `/tmp` (`serve.sh` sets `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`) |
| `No available shared memory broadcast block found in 60 seconds` | Ranks are loading weights or compiling | Wait; it is not a failure by itself |
| Chain dies after about 10 minutes | It was started as a tool-managed background command | Start it with `bench/blog_launch.sh` (detached) |
| Your shell is killed when stopping a chain | `pkill -f <pattern>` matched your own command line | Use `kill -TERM <pid>`; never `pkill -f` a pattern that appears in your command |
| Streams stop yielding text during a forced-length run | With `ignore_eos` the model emits special tokens after its natural end | Expected; count tokens from usage and trace annotations, not from text chunks |

## 7. Reporting

- Keep timing (unprofiled `off` launches) separate from diagnostics (`off-profidle`, traces).
- Kernel sums are not wall time; live KV bytes are not reserved pools or prefix capacity; unavailable counters are unavailable, not zero.
- Results from different vLLM builds are not comparable without fresh controls. Always state the build.
- Curate small evidence into `reports/<study>/`; keep weights, traces and private paths out of git. Run `python tools/audit_references.py` before publishing. Commit and push only when asked.
