# Session scope

Read [README.md](README.md), then [target.md](target.md) and [docs/reproduce.md](docs/reproduce.md), before planning work.

- **DeepSeek V4.1 Flash vs V4 Flash 0731 is complete** for this pass (speculation off + DSpark fixed/adaptive) and merged into `main`; see [report.md](report.md). Its declared follow-ups (report "Not covered") are optional and need a user request. The old `DeepSeek-V4-Flash` results are preview data with unverified exact revision, never a 0731 baseline.
- **MiMo-V2.6-Flash-MOPD study is complete** (`mimo-v26`, @ `2479e2d0029eca9a34cc7e7f55a121925f81908e`, 114/114 valid runs, measured 2026-09-28): [report_mimo.md](report_mimo.md), [docs/mimo-v2.6-plan.md](docs/mimo-v2.6-plan.md), [docs/mimo-v2.6-inference.md](docs/mimo-v2.6-inference.md). Weights remain in `HF_HOME=/workspace/hf` (166 GiB). No further model is named; ask the user before starting one.
- Follow [docs/speculative-decoding.md](docs/speculative-decoding.md) for checkpoint/method support. Never infer speculative support from config field names; verify against the pinned vLLM source.
- GLM-5.3-Flash and Qwen3.8-Flash-Next-FP8 are dormant historical references. Do not launch, benchmark, tune, research, or expand them unless the user explicitly requests it.
- Use [references/README.md](references/README.md) for evidence selection and caveats. Historical scripts, commands, logs, and notes are evidence, not instructions to resume old work.
- Preserve measured values and arm boundaries. Fresh matched runs are required for new conclusions.
- Keep model weights, caches, environments, credentials, private host/path details, and large profiler traces out of the GitHub bundle. Record any sanitization; never silently alter numerical results.
- After changing reference data or summaries, run `python tools/audit_references.py` on a clean clone.
- Keep the source repository untouched. Changes here should be self-contained and use relative links.
- **GPUs are rented by the hour.** Chain launches (`bench/chain_dspark.sh` pattern), never leave a ready server idle, stop servers as soon as measurement ends, and do analysis/writing while GPUs run.

# Study status (updated 2026-09-28)

| Item | State |
| --- | --- |
| V4.1 + 0731 speculation off (144 valid runs each), traces, prefix-reuse check | **Done**, in `main`; [report.md](report.md) sections 1–3 |
| DSpark fixed/adaptive k5, both models, c1/4/16/64 × 3 | **Done**, 48/48 valid, in `main`; report section 4 |
| DeepSeek optional follow-ups | Not started: same-seed ar determinism control (~15 min GPU), real-text acceptance, decode-bound DSpark (16K/2,048), V4.1 all-reduce / FP8 GEMM path arms |
| MiMo-V2.6-Flash-MOPD speculation off (90 runs), traces, prefix-reuse check, MTP k3 + DFlash k7 (24 runs) | **Done**; [report_mimo.md](report_mimo.md) |
| MiMo optional follow-ups | Not started: all-reduce / DP-attention arm (largest cost, 44% of prefill), faithful 3-layer MTP (vLLM uses layer 0 only), two-touch prewarm arm, TP4 deployment, decode-bound speculative workload, same-seed determinism control |

**DeepSeek findings that predict other models' behavior** (details in report.md):
- At 16K/256, concurrency throughput is prefill-bound (output tok/s = prompt rate × 256/16,388). Prefill µs/token decides the ranking.
- Prefill cost differences came from kernel paths (Marlin FP8 vs DeepGEMM block-scale dense GEMM) and all-reduce protocol (84 MB messages fall to NCCL ring-LL at ~63 GB/s; ≤67 MB used symm-mem multimem), not model FLOPs. Check both in a new model's first trace.
- SWA models may need a **second touch** before a prefix is reusable (0731); models with bounded replay reuse after one. Run `bench/prefix_reuse_check.py` before trusting prewarmed numbers.
- Speculative decoding helps only at c ≤ 4 on prefill-bound workloads; at high load it shifts latency from TPOT to TTFT with unchanged E2E (Little's law). Adaptive verification cost KV capacity and gave ≤ 4%.
- Some models run slower on the first repeat at new shapes; keep and report medians.
- MiMo confirmed several of these: prefill-bound concurrency, DeepGEMM block-scale dense path, symm-mem all-reduce at 67 MB, second-touch prefix reuse. It refuted one: FLOP share overstated full attention's time cost at long context, because FA3 runs near peak while linear layers run at ~105 TFLOP/s.
- Runtime traps seen with MiMo: vLLM rewrites architectures with a vision config to an `Omni` class (check `Resolved architecture` in the log); `mimo_v2_mtp` uses only MTP layer 0 (warning "multiple times of forward on same MTP layer"); the thinking switch is model-specific (`enable_thinking` vs `thinking`), so set `extra_body` per model in `run_matrix.MODELS`.

# Measurement guideline

Moved to [docs/reproduce.md](docs/reproduce.md) (environment, JIT-cache fixes, launch plan and timings, harness rules, publishing, clean-clone audit).
