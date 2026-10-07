# Handoff: blog-architecture-h100-v1

Session of 2026-10-07 on one rented 8×H100 80GB node. Results: [findings.md](findings.md). Plan: [blog_target.md](../../blog_target.md). Build: vLLM `0.31.1rc1.dev50+g554340f3d` (`554340f3d3259e321be4c07282be7a02a5aeef83`) in `/root/vllm-latest`.

## 1. Status of every planned item

### First three models (0731, V4.1, MiMo)

| Item | Planned | State | Evidence |
| --- | ---: | --- | --- |
| B0 architecture and implementation table | 1 | **Done** from config, card, server logs, source and traces | [architecture.md](architecture.md) |
| Functional checks (rendered request, natural EOS, forced 256) | 3 | **Done**, 3/3 pass | `data/<model>/_functional/functional.json` |
| Disjoint pilot | 18 runs | **Done**, 18/18 valid | `data/<model>/pilot/` |
| Frozen plan and dry run | 1 | **Done** before timing | [plan.json](plan.json) |
| B1 unprofiled timing | 54 runs | **Done, 54/54 valid**, three launch-separated blocks with rotated model order | [serving.csv](serving.csv), [serving_tables.md](serving_tables.md) |
| B2 diagnostic traces | 18 captures | **Done, 18/18**: prefill 16K/64K, decode 1K/64K at engine B=1 and B=8, all 8 ranks | [components.csv](components.csv), [components_tables.md](components_tables.md) |
| B2 hardware-counter point sets | 6 | **Not collected**: no Nsight Compute on the image. HBM comparison is unanswered; estimates only | findings section 4 |
| B3 live KV snapshots | 18 | **Done, 18/18** at 1K/16K/64K × B=1/8 | [memory.csv](memory.csv), [memory_tables.md](memory_tables.md) |
| B3 logical-state derivation per layer type | 3 | **Partial**: MiMo derived and matched; DeepSeek models not derived tensor by tensor | findings section 5 |
| B4 FFN component costs | from B2 | **Done** from traces | findings section 6 |
| B4 routing statistics (tokens per expert, coverage, imbalance, padding) | 3 | **Unsupported**: no runtime counters, no instrumentation added | – |
| V4.1 prefill versus technical report | 1 | **Done** | [v41_prefill_check.md](v41_prefill_check.md) |
| Natural-EOS task set, quality comparison | – | Not planned for this session; only the three-prompt functional check exists | – |
| Prefix capacity or reuse | – | **Deferred by instruction**; nothing run | – |
| Conditional follow-ups (1K/c64, 128K, single-change ablation, five-block confirmation) | – | **Not run**; none declared | section 5 |

### Added models (Qwen3.8-Flash-Next, GLM-5.3-Flash)

Declared in [plan_addendum_glm_qwen.json](plan_addendum_glm_qwen.json). State at the time this file was first written: **queued and starting** (diagnostics, then three timing blocks each: 36 timing runs, 12 traces, 12 KV snapshots). The section "Added models" at the end is updated when they finish.

## 2. Failures, deviations and what they affect

| What happened | Effect |
| --- | --- |
| First 0731 launch on the new build died with `CUDA error: invalid argument` while the ranks JIT-built a FlashInfer module | No data. Relaunched after the other two diagnostics; server log kept under `data/v4-0731/_server/`. The chain now relaunches once automatically |
| The session started on the pinned build; Tan then asked for the latest vLLM | One 0731 launch on the old build was stopped before any measurement; its log is kept as `_logs/aborted-pinned-build-launch` locally. All published numbers are from the new build |
| A chain started as a tool-managed background command, then restarted detached; a `pkill -f` pattern matched its own shell | About 2 minutes lost, no data affected. Fixed with `bench/blog_launch.sh` |
| The first two diagnostic servers took about 3 minutes to stop because the unreaped group leader looked alive | About 6 minutes of rented time. Fixed (stops now take seconds). V4.1's diagnostic server was verified gone by hand and the waiting chain process ended; its `shutdown.jsonl` line is hand-written and says so |
| Diagnostic decode windows on MiMo ran 5 s instead of about 0.3 s (two streams stopped yielding text after their natural end) | Larger traces only; step averages use 446–491 steps. Fixed before V4.1 and 0731 |
| GLM/Qwen weights were downloaded 21:33–21:40 while MiMo's block 2 was timing | No visible effect: MiMo block 2 equals blocks 1 and 3 within 1% at every point |
| 0731 at 1K/c8, block 1: 551.9 tok/s against 621.8 and 622.5 | Kept. Median reported next to the mean |
| Timing ran on `off`; diagnostics on `off-profidle` (idle profiler) | By design. Pilot and trace numbers are never mixed into timing tables |
| Model order within a block is rotated, but the three blocks ran back to back in one evening | Blocks are launch-separated, not day-separated |

## 3. Cost and time

| Phase | Wall time |
| --- | --- |
| Downloads (0731, MiMo, V4.1), runtime install, input building | 19:24–19:35, GPUs idle |
| Diagnostics: three launches, one failed launch | 19:37–20:41 |
| Timing: nine launches, 54 runs | 20:41–22:59 |
| Recorded server-up time for the three models | about 3.3 hours |

A timing step (launch, warmup, six points, stop) takes 14.5–16 minutes; a relaunch of a cached model takes about 2 minutes, a first launch 6–11. No hourly price was supplied, so cost is in node-hours only. Every server was stopped by its chain; `study/_logs/shutdown.jsonl` records 0 MiB on all GPUs after each stop.

## 4. Exact commands

```bash
# environment
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf
# inputs (already frozen; rebuild only on a fresh node, hashes must match study/_inputs/manifest.json)
/tmp/blog-inputs-venv/bin/python bench/blog_corpus.py && $VLLM_VENV/bin/python bench/blog_inputs.py
$VLLM_VENV/bin/python bench/blog_inputs.py extra qwen-38 glm-53
# what was run
bash bench/blog_launch.sh chain-diag diag:v4-0731 diag:mimo-v26 diag:v41
$VLLM_VENV/bin/python bench/blog_study.py plan --counts '{"16k:c1": 36, "16k:c8": 75, "1k:c1": 48, "1k:c8": 204, "64k:c1": 18, "64k:c8": 33}'
$VLLM_VENV/bin/python bench/blog_study.py dry-run timing:v4-0731:1 timing:mimo-v26:1 timing:v41:1 ...
bash bench/blog_launch.sh chain-timing timing:v4-0731:1 timing:mimo-v26:1 timing:v41:1 \
     timing:mimo-v26:2 timing:v41:2 timing:v4-0731:2 timing:v41:3 timing:v4-0731:3 timing:mimo-v26:3
bash bench/blog_launch.sh chain-glm-qwen diag:qwen-38 diag:glm-53 timing:qwen-38:1 timing:glm-53:1 \
     timing:glm-53:2 timing:qwen-38:2 timing:qwen-38:3 timing:glm-53:3
# tables and publishing
python bench/blog_report.py serving && python bench/blog_report.py memory
python bench/blog_report.py components && python bench/blog_report.py tables
python bench/blog_report.py publish && python tools/audit_references.py
```

Raw runs live in the git-ignored `results/blog-architecture-h100-v1/`. The curated copies are under `data/<model>/` and `study/` here, each with a `CURATION.json` ledger (lossless gzip, private IPs masked, full traces and public input texts excluded; inputs are regenerable from pinned sources and their hashes are published).

## 5. Next steps, in order of value

1. **HBM counters (closes Kan's HBM question).** Before renting: install Nsight Compute, confirm it can attach to or replay one rank's kernels, and write the six point sets (each model at a late 64K prefill chunk and at 64K/B=1 decode) as a declared manifest. If it cannot work with an eight-rank server, record that and keep the estimate.
2. **Expert backend ablation for MiMo prefill.** MiMo's expert GEMM is 70–76% of its prefill chunk on the `HUMMING` backend. One variable (another supported MXFP4 MoE backend), one model, two points (16K/c1, 16K/c8), baseline and intervention, three blocks: 12 runs, about 35 minutes. Check `vllm serve --help=all | grep -i moe` for the supported choices first.
3. **128K points** to locate where MiMo's growing attention and KV cross V4.1 and 0731: 18 runs, about 60 minutes.
4. **1K/c64** to test the "more distinct experts at larger batch" reading: 9 runs plus three B=64 traces.
5. **Routing statistics** need instrumentation in the runtime (a separate diagnostic runtime with its own overhead control); do not start on rented time without a prepared patch.
6. **Prefix capacity** stays deferred until Tan asks.
7. **Confirmation** of small differences (for example V4.1 versus MiMo at 64K/c1, within 1%) needs held-out prompts and at least five independently scheduled blocks.

## 6. Added models: Qwen3.8-Flash-Next and GLM-5.3-Flash

Pending at first publication. This section is replaced with their status table, deviations and any failed or unsupported points when the chain finishes.
