# Reproducing a serving study on one 8×H100 node

**Current Qwen rule (2026-10-08):** use only original BF16 `Qwen/Qwen3.8-Flash-Next` for future Qwen work; see [the pinned policy](qwen-checkpoint-policy.md). The Qwen download sizes, startup timings and `qwen-38` commands in the completed October workflow describe **FP8 history**, not a BF16 procedure. Do not run the old Qwen entry. BF16 requires separate harness wiring, a new study ID and fresh matched controls before an authorized session; no new FP8 runs.

**Active-phase routing, 2026-09-30:** follow [target.md](../target.md) and [the three-model GPU plan](three-model-h100-plan.md) for `spec-realtext-h100-v1`. Extend the harness locally first; use fresh AR controls, variable 256/2048 output lengths, pinned real text and a new no-prefix chain. All new prefix tests, including reuse checks and prewarming, are deferred. H200 and new models are outside the next session. Do not run the historical launch matrix or `chain_mimo.sh` as the active workflow.

## 0. Fresh-node quick setup (verified 2026-10-07, blog study)

This records the completed October setup. Its general inventory/environment preparation applies to a fresh node; model choice and launch authorization follow current policy. In particular, the old Qwen FP8 procedure is superseded. It took about 15 minutes from login to the first server launch; local preparation belongs before rental.

| Step | Command | Time on the 2026-10-07 node | Notes |
| --- | --- | --- | --- |
| 1. Inventory | `nvidia-smi`, `lscpu`, `df -h /workspace`, `ls /workspace/hf/hub` | seconds | Do not assume the previous node's weights, caches or venvs survived. `/workspace/hf` was empty |
| 2. Runtime | `bash install.sh` (pinned venv `~/vllm` and blog venv `~/vllm-latest`) | about 2 min per venv | Pick the runtime with `export VLLM_VENV=/root/vllm-latest`; the default stays the pinned build |
| 3. Weights | `export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1`, then `hf download <model> --revision <sha> --max-workers 32` for the three revisions in [target.md](../target.md) | 0731 156 GB in 2 min, MiMo 166 GB in 5 min, V4.1 476 GB in 4 min | All repositories are public; no `HF_TOKEN` was needed. Download the smallest first so a launch can start early. The two models added to the blog study took 4 min (Qwen, 186 GB) and 3 min (GLM, 328 GB); their revisions are in [blog_target.md](../blog_target.md) section 10 |
| 4. Public inputs | `python3 -m venv /tmp/blog-inputs-venv && /tmp/blog-inputs-venv/bin/pip install pyarrow`; `/tmp/blog-inputs-venv/bin/python bench/blog_corpus.py`; `$VLLM_VENV/bin/python bench/blog_inputs.py` | about 2 min | The vLLM venv has no `pyarrow`; keep it out of the measured environment. The math and chat datasets are pinned by revision in `bench/blog_corpus.py` |
| 5. Record the node | `$VLLM_VENV/bin/python bench/blog_study.py env` | seconds | Writes `results/<study>/_env/` |
| 6. Diagnostics and pilot | `bash bench/blog_launch.sh chain-diag diag:v4-0731 diag:mimo-v26 diag:v41` | 15–25 min per model, mostly the first launch | One idle-profiler launch per model: functional checks, disjoint pilot, live-KV snapshots, traces |
| 7. Freeze and dry-run | `python bench/blog_report.py pilot`, then `blog_study.py plan --counts '<json>'` and `blog_study.py dry-run <steps>` | seconds | Timing starts only after the dry run prints `DRY RUN OK` |
| 8. Timing blocks | `bash bench/blog_launch.sh chain-timing timing:<model>:<block> ...` in the declared block order | 14.5–16 min per model-block (six points); 54 runs took 2 h 18 min | Each step owns its server and stops it on every exit path |
| 9. Tables and publishing | `python bench/blog_report.py serving`, `memory`, `components`, `tables`, `publish` | seconds; `components` reads every rank trace and takes about 15 min for five models | Standard library only |

Traps met on this node:

- **Detach long chains.** A chain started as a tool-managed background command can be killed by that tool's timeout. `bench/blog_launch.sh` starts it with `setsid nohup`; watch `results/<study>/_logs/chain.log`.
- **Never `pkill -f` a pattern that also appears in your own command line.** It killed the calling shell and left a half-started chain. Stop a chain with `kill -TERM <pid>` from the launcher's output; the chain then stops its own server group.
- **First load of a model reads the whole checkpoint from disk**; expect several minutes before graph capture. The engine prints `No available shared memory broadcast block found in 60 seconds` while ranks load; that message alone is not a failure.
- **No Nsight Compute (`ncu`) on the image.** Hardware HBM byte counters are unavailable unless it is installed and validated beforehand.

The sections below preserve the procedure that produced [report.md](../report.md) (study `v41-vs-0731`) and [report_mimo.md](../report_mimo.md). Their run lists, fixed-256 validity rule and timing estimates are **historical reproduction instructions**, not defaults for the new phase. Environment fixes and curation remain applicable; the new plan's per-request validity, counting and scope take precedence. GPUs are rented by the hour: chain launches, measure promptly and stop servers as soon as measurement ends.

## 1. Environment

| Item | Value |
| --- | --- |
| venv | `source /root/vllm/bin/activate`; vLLM `0.30.1rc1.dev223+g44af287eb`, torch `2.13.0+cu132`. Keep it pinned within a study. `pandas` is required for `--dataset-name custom`. |
| Credentials | `HF_TOKEN` from the environment only. Never write it into scripts. |
| Weights | `HF_HOME=/workspace/hf` (shared xet/blob store). Use `du -shL` for the real size. Download with a pinned `--revision <sha>`. |
| JIT caches | `/root` is a gocryptfs FUSE mount, where concurrent TP-rank JIT compiles race. `bench/serve.sh` sets `TRITON_CACHE_DIR=/tmp/triton_cache` and `FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_ws`. If a new model dies in warmup or graph capture with `FileNotFoundError` or `CUDA error: invalid argument`, move that model's JIT cache (for example `DG_JIT_CACHE_DIR`) to `/tmp` first. |
| tmux | `ds-serve` (server), `ds-bench` (harness), `ds-chain` (chained arms), `ds-dl` (downloads) |
| Stopping | `tmux send-keys -t ds-serve C-c`. Never run `pkill -f <pattern>` from a shell whose own command line contains that pattern; use bracket patterns such as `pgrep -f "[v]llm serve"`. |

## 2. Adding a model

Historical extension procedure; no fourth model is authorized by the current three-model plan.

1. Check `config.json` `architectures` against `vllm.model_executor.models.registry.ModelRegistry.get_supported_archs()` in the pinned venv.
2. Add one entry each to `MODELS` in `bench/run_matrix.py`, the `case` block in `bench/serve.sh`, and `bench/profile_trace.py`: model ID, pinned revision, tokenizer mode, and reasoning/tool parsers.
3. Confirm that `chat_template_kwargs.thinking=false` (or the model's equivalent) produces non-thinking output, and record the encoded request.
4. Check that the shared prefix prompt files (`results/<study>/_prompts/`) tokenize to the same counts. If they don't, record it and relax the 98% prompt-token check to the measured count.
5. Write a plan JSON (`results/<study>/plan*.json`, copied to `reports/<study>/`) **before** collecting.

## 3. Launch plan (each restart costs 3–10 min)

Historical matrix only. The active phase disables prefix caching and uses its own manifest/chain; do not execute the prefix launch or old chained scripts below for it.

| Launch | Config | Workloads | Approx. time |
| --- | --- | --- | --- |
| 1 | `off-profidle` (prefix caching off, idle profiler; `PROFILE_DIR` required) | `run_matrix.py --workloads concurrency,context,prefix,isolated --prefix-states cache-off`, then traces with `profile_trace.py` (16K and 64K prefill) | concurrency ~45 min, context ~30, prefix cache-off ~45, isolated ~25, traces ~3 each |
| 2 | `off-prefix` | `run_matrix.py --config off --workloads prefix --prefix-states cold,prewarmed`, then `prefix_reuse_check.py` | ~30 min + 1 min |
| 3+ | one per speculative arm (for example `dspark-fixed-k5`) | concurrency c1/4/16/64 | ~17–25 min per arm plus ~3.5 min launch |

For new studies, write a chain script on top of `bench/chain_lib.sh` (`launch`, `trace`, `wait_sweep`, `stop_server`); `bench/chain_mimo.sh` is the template (traces → prefix launch → speculative arms → stop). `bench/chain_dspark.sh '<model> <config>' ...` is the older DSpark-only chain. It waits for any running sweep, stops the server, launches the next arm, polls `/v1/models` (aborting if the process dies or 30 min pass), runs the sweep, and stops the server at the end. Watch `results/<study>/_chain_dspark.log` with a Monitor filter on `CHAIN|valid=False|Traceback|Error`. **Do not hand-roll readiness loops:** a broken `until` condition once left a ready server idle for 1 h 50 min.

The first launch of a new model takes about 8–10 min (JIT, DeepGEMM warmup, graph capture); later launches take about 3.5–5 min. Adding a speculative method changes graph-capture memory and KV capacity; record both from the server log.

## 4. Harness rules

- `run_matrix.py` writes immutable run directories under `results/<study>/<model>/<workload>/<config>/<point>/repeat-N`. Reruns become `repeat-N-rerunK`, and nothing is overwritten.
- Historical fixed-256 runs are valid only with all requests completed, output = n × 256, and server prompt tokens ≥ 98% of the target. New real-text runs validate every request against its declared output budget and actual rendered prompt count; natural-EOS task checks use separate validity rules.
- Speculative counters (`spec_num_drafts`, `spec_num_draft_tokens` = scheduled drafts, `spec_num_accepted_tokens`) are in `summary.json`. Per-position acceptance comes from `telemetry/metrics_{before,after}.prom`.
- **Historical prewarm caveat (deferred in the active phase):** a model whose SWA prefix reuse needs a second touch (0731-style) requires **two** requests per prefix using distinct suffixes for a fully warm control. The old one-touch measurements remain unchanged; do not run a new prewarm/reuse experiment without an explicit request.
- Some models are slower on the first repeat at new shapes (0731: 5–9%, and once 50% on a new DSpark shape). Keep those runs, and report medians next to means.
- Traces: decode under full CUDA graphs is not attributable. After adding a model, check that no large kernel lands in `other_elementwise` in `trace_breakdown.py`.

## 5. Analysis and publishing

```bash
python bench/summarize.py <study> <model> --csv reports/<study>/<model>_off_summary.csv > reports/<study>/<model>_off_tables.md
python bench/compare.py --csv reports/<study>/comparison.csv > reports/<study>/comparison_tables.md
python bench/spec_compare.py --csv reports/<study>/dspark_comparison.csv > reports/<study>/dspark_tables.md
python bench/compare_models.py --csv ... > ...          # cross-study ratios (edit MODELS/SUBJECT/REFS)
python bench/spec_compare.py --preset mimo ...          # per-study speculative presets
python bench/compare_outputs.py <model> <spec-config> <ar-config> --study <study>
python bench/quality_smoke.py --model <key> --study <study>   # also verifies the thinking switch
python bench/curate.py <study> <model>     # lossless gzip, IP/ANSI removed from server logs, prompts/full traces excluded, SHA-256 ledger
```

`compare.py` and `spec_compare.py` contain per-model config maps (`CFG`, `AR`); extend them for a new model pair. CSV writers use `lineterminator="\n"`, and `.gitattributes` marks `reports/**/data/**` as `-text` so hashes survive checkout.

Audit on a clean clone, because `tools/audit_references.py` also scans the git-ignored `results/`:

```bash
git clone --branch <branch> . /tmp/audit && cd /tmp/audit && python tools/audit_references.py
```

Commit on a feature branch, push, and merge into `main` only when the user asks.

## 6. Reproducibility status (checked 2026-09-28)

**Reproducible from code:**
- **Runtime:** `install.sh` pins vLLM commit `44af287ebe38d6dc4e102948025f5e3e175aefd6` (it previously installed the moving nightly). A fresh resolution matched the working environment's full lock (`reports/environment-lock.txt`, 205 packages) for 200/201 packages, and `filelock` is now pinned too.
- **Models:** every checkpoint is pinned by revision in `bench/serve.sh` and `run_matrix.MODELS`, and the drafts are pinned too (DSpark `revision`, DFlash snapshot path).
- **Requests:** today's `run_matrix.bench_cmd` regenerates byte-identical `vllm bench serve` commands for recorded V4.1, 0731, and MiMo runs (checked against `manifest.json`). Seeds are deterministic, and prompts regenerate from seeds and the pinned tokenizer.
- **Analysis:** summaries, comparisons, trace breakdowns, and curation are scripted. Re-running them on the curated data reproduces the published tables; the DeepSeek tables were byte-identical after the later tool changes.

**Small gaps (noted, not fixed):**
- Timings are hardware-specific: 8× H100 80GB SXM, all-pairs NV18 NVLink, driver 580.105.08. `--torch-backend=auto` picks the CUDA build from the driver. Expect repeats to land within the reported SD, not bit-identical.
- Greedy outputs are not verified deterministic across runs (no same-seed repeat), so the speculative output-match tables are not losslessness evidence.
- The 0731 sequential prefix check came from an unsaved ad hoc script. `bench/prefix_reuse_check.py` reconstructs it (same prompts, endpoint, and 67,588-token queries) and was used for V4.1 and MiMo.
- DeepSeek traces were filed into `profiles/<label>/` by hand, and the DeepSeek quality smoke was ad hoc. Both are now scripted (`chain_lib.sh trace`, `bench/quality_smoke.py` with the same 4 prompts).
- Not published (git-ignored `results/`): prefix prompt JSONL (regenerable from seeds) and full 8-rank torch traces (~73 MB per model; must be re-captured).
- `environment.txt` was captured only for the DeepSeek study; MiMo ran on the same node and venv the next day, which `reports/environment-lock.txt` covers.
- Failed or crashed runs are kept as `repeat-N`, with valid reruns as `repeat-N-rerunK`. A reproduction without the failure gets plain `repeat-N` names.
- Weights must still be downloadable from Hugging Face at the pinned revisions (needs `HF_TOKEN`), and the vLLM per-commit wheel index must stay online.
