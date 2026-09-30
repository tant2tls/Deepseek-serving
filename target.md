# Target: strengths and weaknesses of three models on 8×H100

Updated 2026-09-30. **This session prepares the next GPU study; it does not run or rent GPUs.** The September DeepSeek and MiMo studies are complete. Preserve their results in [report.md](report.md) and [report_mimo.md](report_mimo.md). The new phase is `spec-realtext-h100-v1`, currently **planned, not measured**.

**Objective:** establish which deployment performs best for which workload on one 8×H100 80GB SXM node, and explain the evidence for its strengths and weaknesses. Compare absolute throughput, responsiveness, sustained generation, speculative progress/cost and task checks. A valid outcome can be a regression, an inconclusive comparison or an unsupported optimization. Speed alone does not establish equivalent task quality or an intrinsic architecture ranking.

Read [update.md](update.md) for the checked evidence and experiment rationale, [docs/three-model-h100-plan.md](docs/three-model-h100-plan.md) for the concrete run plan, and [docs/reproduce.md](docs/reproduce.md) for environment and curation. These current scope rules supersede older experiment lists in historical plans and reports.

## Scope and immutable identities

| Key | Checkpoint | Revision | Initial arms |
| --- | --- | --- | --- |
| `v4-0731` | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | AR off; fixed DSpark k5 |
| `v41` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | AR off; fixed DSpark k5 |
| `mimo-v26` | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | `2479e2d0029eca9a34cc7e7f55a121925f81908e` | AR off; MTP k3 (layer 0 reused); DFlash k7 |

Pin target, tokenizer/remote code and draft separately. DSpark and MTP use the target snapshot; MiMo DFlash uses its `dflash/` subfolder. Verify resolved implementation, loaded layers and supported widths against vLLM commit `44af287ebe38d6dc4e102948025f5e3e175aefd6` (`0.30.1rc1.dev223+g44af287eb`). A newer build is a separate runtime arm with new AR controls. Follow [the DeepSeek support audit](docs/speculative-decoding.md) and [the MiMo implementation record](docs/mimo-v2.6-plan.md), not config names alone.

**All new prefix-cache experiments are deferred until Tan explicitly requests them.** No cold/warm comparison, prewarming, reuse check, prefix pressure or speculation-plus-prefix arm. Disable prefix caching in every active run. Ordinary per-request KV allocation, occupancy and traffic remain in scope. Preserve and appropriately qualify existing prefix results.

Preview DeepSeek, GLM, Qwen and other models remain outside scope. The preview's unverified revision is never a 0731 baseline. H200 is an optional separately requested future study, not part of this H100 session or its completion requirements. TP4×DP2/DP-attention and runtime changes are conditional tuned deployments, separate from the common baseline.

## What existing evidence supports

| Model | Current strength | Current limitation and new question |
| --- | --- | --- |
| MiMo | Highest reported uncached throughput; lowest c1 decode latency; DFlash c1 219.2 tok/s on 16K/256 | All-reduce dominates the sampled prefill kernel sum; MTP reuses one head. Does the advantage hold for real text and long output? |
| 0731 | Faster uncached than V4.1; fixed DSpark c1 gives 1.87× over its own AR baseline | Below MiMo's absolute throughput; smaller reported KV pool and slow first repeats. Does its draft agreement pay off at high load? |
| V4.1 | Lower ≤2K TTFT than 0731; largest reported KV pool; advantage in the historical one-touch prefix procedure | MiMo is faster on the tested short uncached prompts too. Dense GEMM/collective cost and inactive CED skipping limit this build. Which supported intervention changes measured cost? |

These observations apply to the saved deployment settings and workloads. Native weight/KV formats and kernel paths differ; equal utilization flags do not mean equal usable KV capacity. The reported pools are not measured maximum concurrent requests. [update.md](update.md) links numerical anchors, raw runs, uncertainty and interpretive corrections, including the distinction between estimated and measured committed tokens per round.

## Active matrix and priorities

| Stage | Required core | Result needed |
| --- | --- | --- |
| Preparation | Extend the existing harness locally; freeze public real-text prompts, pins, manifest, validation and no-prefix chain | A dry-run manifest and tested collection/analysis before GPU rental |
| E0 | Each model: repeated same-seed AR and planned speculative methods, c1/c64, natural-EOS task checks; validate counters and implementations | Correctness/support record; unavailable telemetry explicitly identified |
| E1 screen | Balanced code/math/chat, approximately 1K input / 2048 output, c1/c64, 7 deployments × 3 repeats | 42 fresh timing runs; absolute and within-model speculative results |
| E1 controls | 16K/2048 and 16K/256 at c1/c64, same prompts between output-length arms | 84 more runs; isolate output-length sensitivity without changing prompt distribution |
| E1 crossover | 1K/2048 at c4/c16, same seven deployments × 3 | 42 more runs; a four-load short-input curve (168 core timing runs total) |
| E3 core | All seven deployments, initial 1K KV, actual B=1/64 decode-only intervals | 14 short diagnostic captures; progress, draft/verification cost and exposed communication |

E0, pilots, warmups, overhead controls, failed attempts and profiles are additional to the 168 timing runs. Stage the work; estimate wall time from a disjoint pilot, and record the session allocation before long sweeps. A partially run stage stays pending rather than becoming a completed study.

**Targeted tier, declare before collection:** E2 supported widths and fixed/adaptive DSpark; E4 real-text long-context sensitivity; separate domain throughput; held-out confirmation and latency validation; a model-specific communication or kernel ablation selected from traces. Do not launch the entire Cartesian product. The run-count and selection rules are in [the GPU plan](docs/three-model-h100-plan.md). No width, alternative kernel, faithful multi-layer MTP or CED implementation is assumed available.

## Comparison and evidence rules

- Keep the common deployment at TP8 + EP, memory utilization 0.90, context limit 262,144, max sequences 64, explicit 8,192-token chunk budget, text only, temperature 0 and thinking off. Preserve native precision; record resolved graph, quantization and KV formats. DSpark draft sampling/rejection settings stay pinned. Separate any change to these settings into its own arm.
- Collect new AR controls for every model with the new workload. Compare the same texts across models and publish token counts; a separate token-count-matched view must be labelled. Context checks include rendered template, requested output and required speculative lookahead. Keep forced 256/2048-token timing separate from natural-EOS quality checks.
- Save immutable manifests, request IDs, prompt hashes, per-request status/tokens/timings, raw counters, launch logs and failures. Validate every completion and per-request output budget, identity and cache-off setting. Reruns use new directories and are paired by manifest identity. Missing telemetry is unavailable, never zero.
- Show mean, sample SD, median, repeat count and paired ratios. Three repeats screen large effects; small gains and final recommendations need held-out confirmation with independently scheduled blocks. Do not treat token samples as independent observations or reuse the old SD heuristic as a significance test. Screen p95 estimates are descriptive only.
- Separate client concurrency from actual engine batch and verification positions. Report actual committed tokens and critical-path wall time where observable; the old `1 + accepted/drafts` is a proxy. Never sum overlapping ranks/kernels or infer a prefill bottleneck from input/output accounting alone.
- Keep authoritative timing unprofiled. Quantify profiler overhead; eager/graph-disabled or patched instrumentation is a separate diagnostic runtime. Record source paths/commit, layers, shapes, fusion and unattributed time. Trace shares suggest interventions; a matched ablation is needed to claim the intervention caused an improvement.
- Evaluate each speculative arm against its own AR baseline first, then compare all models' absolute throughput and latency. Native methods are deployment comparisons, not isolated algorithm comparisons. Use `8/output_tok_s` for allocated GPU-seconds/token; dollar cost needs a supplied hourly price.
- Provide measured throughput/latency Pareto curves. No application SLO is supplied, so do not invent one or claim production goodput from closed-loop load. Open-loop SLO validation and capacity saturation require separate manifests. Task checks bound quality claims to the tested tasks.

## Completion and handoff

The bounded H100 core is complete when every core point is measured and validated or explicitly failed/unsupported with evidence, all omitted points are accounted for, and the report contains:

1. Three-model serving curves and absolute/relative results with sample counts, uncertainty, output policy and task outcomes.
2. Support and acceptance tables separating scheduled, verified, accepted and committed tokens; unavailable fields and proxy quantities labelled.
3. Diagnostic phase/component costs with explicit denominators, critical-path versus kernel-sum distinction, trace/source links and unresolved explanations.
4. A strengths/weaknesses table for each model: workload, measured difference, explanation confidence, confounders and recommended or inconclusive operating point.
5. A portable evidence bundle and handoff listing completed/pending/failed/unsupported points, exact commands, pins, durations and next questions. Run `python tools/audit_references.py` on a clean clone before publication.

Missing core runs stay pending. Missing causal telemetry restricts explanations; it does not justify fabricated component timings. Conditional extensions are not core completion gates and cannot be reported as measured. Existing numerical reports remain intact, and the earlier source repository remains untouched.
