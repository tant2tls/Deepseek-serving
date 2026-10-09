# Next GPU session: three-model H100 study

Prepared 2026-09-30. Study ID: `spec-realtext-h100-v1`. **Status: documentation ready; harness extensions, datasets and all new measurements pending.** No GPU was launched for this review. Scope: [target.md section 12](../target.md#12-later-study-spec-realtext-h100-v1); analysis and experiment definitions: [update.md](../update.md). Historical studies remain complete and unchanged.

## 1. Questions and evidence needed

| Question | Comparison that answers it | Conclusion this does not establish |
| --- | --- | --- |
| Which model serves uncached real text fastest on eight H100s? | Fresh AR across all three, same texts/output policy and limits; absolute tok/s, requests/s and latency | Equal task quality, pure architecture advantage or a result on other hardware |
| When does native speculation help? | Each arm / its own fresh AR across four short-input loads; long-input controls | Draft agreement alone as the cause, or equal methods across checkpoints |
| Why can acceptance stay high without speedup? | Committed progress versus round critical path at actual B=1/64; short/long-output controls | Prefill saturation inferred only from input/output rate arithmetic |
| Which implementation limitation matters? | Trace-selected, supported single-change ablation, paired AR/spec and correctness | Unmeasured CED, kernel, multi-head MTP or topology gains |

Record the answer as measured, trace/source-supported, hypothesis or unavailable. The three-model strength table must name the workload, evidence, uncertainty and limitation of each recommendation.

## 2. Work to finish locally before rental

The existing harness is useful but **does not execute this plan yet**. Preserve historical defaults/commands and add a phase-specific manifest path or runner. Do not paste hypothetical flags into a rental session.

| Area | Required change and acceptance check |
| --- | --- |
| Variable outputs | Remove the new phase's dependence on global `OSL=256` in commands, seeds, point IDs, manifests, per-request validity and analysis. Fixture checks for 256 and 2048 outputs must detect one short request even when aggregate totals appear correct. |
| Real text | Prepare redistributable public or authored code/math/chat records: stable request/domain IDs, source/license, expected checks, short and long prompts, exact hashes. Freeze screen/held-out splits before measuring. Save public sanitized artifacts or regeneration recipes; never include private prompts. |
| Tokenization | Pin client/server tokenizer and remote-code revision. Validate the actual rendered chat template and model-specific thinking switch. The old custom-prefix path's `--skip-chat-template` must not be copied without verifying exactly one template application. Record actual per-model token counts; 1K/16K are buckets, not a claim of identical token lengths across tokenizers. |
| Pairing and aggregation | Explicit workload/hardware/runtime/config IDs; pair by prompt hash, seed, output policy, load and repetition. Accept the selected valid rerun for a planned point without counting both attempts or requiring identical directory suffixes. Historical comparison scripts use hardcoded maps and must not overwrite old tables. |
| Collection | Write the attempted manifest before HTTP/subprocess work; preserve raw responses/counts/status and startup configuration. Bound timeouts, reconcile request IDs/completions/tokens, and detect server restarts/counter resets. Save null for unavailable counters. |
| Correctness | Add same-seed AR-vs-AR and AR-vs-spec token comparison where available, natural-EOS scoring, truncation reporting and a fixed rubric. Existing `quality_smoke.py` is only a four-prompt smoke; `compare_outputs.py` compares characters and misses differently named reruns. |
| Diagnostics | Add synchronized decode-only collection or faithful step segmentation; actual active sequences, context, scheduled/verified/accepted/committed tokens, timestamps and fallback accounting. Keep profiler/instrumentation code versioned separately when it changes execution. |
| Launch chain | Create a new no-prefix chain based on `chain_lib.sh`. Enforce this study ID and the seven allowed deployments; reject prefix-on settings. Check served identity and resolved settings, measure immediately after readiness, stop only the owned server on success/error/signal/timeout, and confirm its processes and GPU allocation have exited before continuing. Bound shutdown/readiness and stop on failures. |
| Dry-run checks | Generate the full planned command/manifest list without GPUs. Check exact run counts, unique output paths, no historical study writes, no prefix operations, pins and constraints. Exercise parser/validity/rerun/failure cases with small fixtures or a mock endpoint; run Python compilation and Bash syntax checks on changed files. |

`run_matrix.py` currently defaults to concurrency/context, **not prefix**. `chain_mimo.sh` includes prefix stages; both it and `chain_dspark.sh` hardcode old study IDs. Keep these for historical reproduction and do not use them for this phase. `serve.sh` already supports the seven initial deployments, but needs the new chain's identity/config validation and an explicitly pinned chunk budget. `install.sh` is pinned, not a moving-nightly recipe.

Do not begin the rental with data preparation, analysis plumbing or a multi-layer MTP/CED implementation still to write. Runtime-dependent source/shape checks and a small pilot belong on the actual node.

## 3. Node and support gate

Use the exact checkpoint SHAs in [target.md](../target.md) and the historical runtime commit first. Capture GPU SKU/count/memory, topology, driver/CUDA, CPU/RAM, clocks/power, dependency lock and launch command. Reuse verified cached weights if available; do not assume the previous rented node or paths still exist. Credentials come from the environment and never enter the evidence bundle.

| Setting | Common baseline |
| --- | --- |
| Hardware | One exclusive 8×H100 80GB SXM node, topology recorded |
| Runtime | vLLM `44af287ebe38d6dc4e102948025f5e3e175aefd6`; preserve dependency pins and record resolved versions |
| Parallelism / memory | TP8 + EP; utilization 0.90; native checkpoint precision; resolved KV formats and graph pools recorded |
| Scheduler | Context 262144, max sequences 64, **explicit** max batched tokens 8192, chunked prefill resolved and logged |
| Cache | Prefix caching disabled, verified in resolved startup config; no prewarm/reset/reuse workload |
| Requests | Text only, target temperature 0, model-specific thinking off; fixed-length timing uses `ignore_eos` |
| DSpark | k5 fixed, target-SHA draft, probabilistic draft sampling, standard rejection, adaptive false |
| MiMo | MTP k3 labelled layer 0 reused; DFlash k7 from pinned `dflash/`; record resolved defaults and feature taps |

Verify requested and resolved identity, quantization, parser/template, draft class/depth, graph mode, width and fallback behavior. An OpenAI model ID alone does not prove the loaded revision. Check logs/source and immutable snapshots. Do not force unsupported widths or checkpoint fields. Unsupported arms retain a reason and planned-point entries.

If the old runtime cannot run on the node, stop the matched-baseline claim. A supported replacement requires a separate runtime ID and fresh baselines for all models; do not silently compare a new runtime to old numbers. No classic MTP for either DeepSeek checkpoint without explicit new source support.

## 4. E0 and dataset freeze

Run natural-EOS regression/task checks before interpreting timing. For each concurrency use **two identical AR executions per model**, then each planned speculative arm: `(2+1) + (2+1) + (2+2) = 10` executions. At c1 use 24 balanced prompts; at c64 use 96 balanced prompts. Thus E0 has **20 batch executions** with 512-token caps, separate from timing and warmup counts. AR duplicates measure same-seed stability; every new width/runtime selected later needs its own check. Define in advance which task checks constitute failure; investigate unexplained regressions before recommending an arm. No quality-equivalence claim from this small set.

Use separate performance prompts that request sustained responses. Screen mixture is balanced code/math/chat, shuffled with a pinned schedule; no after-the-fact cherry-picking for high acceptance. Build approximately 1K and 16K inputs from meaningful context, with token counts saved for each tokenizer. For the two 16K output arms, prompt text/order/request seeds are identical. Output policy, not prompt text, changes.

Pilot a disjoint prompt sample. Start timing lists at `max(16,4c)`, round up to a multiple of three for domain balance, and increase to reach ≥60 seconds in the fastest compared deployment. Freeze common counts/lists across all seven deployments before accepting comparative timing. If a later arm is faster than predicted, append a consistently sized matched set for all arms; retain the pilot/undersized set with its status. Do not quietly lengthen only the slower arm. For each shape/concurrency, publish actual list size and measured window.

Screen p95 is descriptive. Tail-latency confirmation uses a separately declared larger sample (at least `max(200,8c)` as a starting floor, plus duration/uncertainty checks). A stream chunk carrying multiple tokens is not a single-token ITL; keep burst/chunk latency separate when token timestamps are unavailable.

## 5. Core run ledger and cost control

The seven deployments are `v41/off`, `v41/dspark-fixed-k5`, `v4-0731/off`, `v4-0731/dspark-fixed-k5`, `mimo-v26/off`, `mimo-v26/mtp-k3`, `mimo-v26/dflash-k7`. Fresh AR uses the unprofiled `off` launch for all three; historical `off-profidle` controls do not replace it.

| Block | Shapes (input/output tokens) | Client c | Deployments × repeats | Timing runs |
| --- | --- | --- | --- | ---: |
| A: initial screen | ~1K/2048 | 1,64 | 7 × 3 | 42 |
| B: output/prompt controls | ~16K/2048; ~16K/256 | 1,64 | 7 × 3 | 84 |
| C: short-input crossover | ~1K/2048 | 4,16 | 7 × 3 | 42 |
| **Core total** | 8 shape/load points per deployment | | | **168** |
| E3 diagnostic capture | Initial ~1K KV, decode-only | Actual B=1,64 | 7, one short capture each | **14 captures**, not timing repeats |

Start with A and E0, review results, then execute the predeclared B/C points within the session allocation. The default economical A screen groups its three repeats per deployment launch (seven launches); disclose this blocking and do not present the repeats as independent restart samples. For B/C use three launch cycles with one repeat per point per cycle, rotating model and AR/spec order across cycles (21 launches if all seven deployments remain supported). Save the exact order before collection. Final small-gain confirmation also needs independent launch/time blocks. Do not quietly omit a model because it looks slow.

Logical block counts are not rental estimates. For a fixed request list, pilot timing predicts `run_seconds ≈ n × output_len / output_tok_s`; use measurements for **each** arm and add startup, JIT/shape warmup, E0, profile relaunches, teardown and a declared rerun allowance. Record node-hours and eight times that number in GPU-hours, plus price only if supplied. The old 17–25-minute speculative sweeps used 256-token output and cannot budget this plan.

Freeze a machine-readable `results/spec-realtext-h100-v1/plan.json` with identities, dataset hashes/splits, points, seeds/orders, counts, output/EOS policy, warmup, timeouts, validity, rerun selection, allocation and extension triggers **before** measured collection. When the session has only a partial allocation, finish complete comparison blocks, stop the server and leave remaining core points pending. Never leave a ready server waiting for analysis. Do analysis while queued runs execute; do not start an undeclared extension simply to occupy the node.

## 6. E3 profiles and targeted extensions

Use a separate diagnostic launch per arm where needed; add its restart and overhead-control cost to the budget. The initial 14 captures must distinguish initial prefill from a measured decode-only interval. Save active B, KV-length distribution, committed output, round span, draft, verification, rejection/state, CPU gaps, padding and exposed collectives. Existing full CUDA graphs can hide V4.1 decode kernels: use graph-aware tooling if available; label eager fallback as diagnostic and measure its overhead. Do not turn a kernel-time sum into end-to-end wall time. If the instrumentation cannot answer a component question, leave it unavailable and narrow the explanation.

| Extension | Trigger / bounded first step | Incremental count rule |
| --- | --- | --- |
| E2 width economics | Different widths could repay cost at the observed low/high-load points; select on screen prompts, validate on held-out prompts | `declared arm/workload/load combinations × repeats`; include own AR and current-width controls unless all identities and collection blocks match |
| Adaptive DSpark | Fixed verification appears expensive at a crossover/high load | Same maximum width and target/draft settings; verify admitted lengths or leave adaptive mechanism unresolved |
| E4 context | Need a real-text context-scaling conclusion beyond 16K | AR + one selected method/model, 1K/16K/64K, c1/c16, 3 repeats = 108 cells; the core supplies 54 compatible cells, so **54 new runs** (18 at 16K/c16 + 36 at 64K/c1,c16) if all six selected deployments are reusable |
| Domain performance | Mixed-domain acceptance suggests a material difference | Separate code/math/chat timing at selected points; aggregate counters from mixed traffic cannot establish per-domain acceptance or throughput |
| Confirmation | Gain near break-even or a final operating recommendation | Held-out prompts; ≥5 independently scheduled repeat blocks, predeclared practical threshold (proposed 5%), latency and task checks; expand if uncertainty remains |
| Runtime/topology ablation | A supported intervention tests dominant exposed cost | One changed variable, fresh AR/spec controls, quality gate; preserve TP8 results and label tuned deployment separately |

E4 reuse requires the selected six deployments to use the original widths/policies and identical workload definitions. Tuning, changed datasets or incompatible blocks increase the count; list the actual missing cells rather than assuming 54. Longer 128K contexts, c32 crossover probes, extra profiles and all other extensions require a declared list and cost. No automatic full Cartesian sweep. No new prefix work. No H200 launch in this session.

For an ablation, choose the question from [update.md](../update.md): V4.1 dense FP8/collective paths, MiMo exposed communication or faithful MTP support, and real-text DSpark progress on 0731. Record source compatibility before trying alternative flags. CED or multi-layer MTP development belongs outside rented measurement time when no supported implementation exists.

## 7. Validation, reporting and stop conditions

Accept a timing point only if the served config/identity matches, all requests complete, failures are zero, every forced-length output meets its budget, prompt counts match the rendered inputs, and required measurement telemetry is valid. Missing optional verified counts/component attribution limit the explanation and remain null. Keep failed attempts with their manifests/logs; reruns get new paths. Stop on identity mismatch, unexpected prefix-on config, unresolved correctness regression, repeated readiness failure or an exhausted session allocation. Preserve the blocker and stop the owned server before analysis.

Report throughput, requests/s, TTFT/TPOT/E2E, request counts, failures, memory/pool occupancy and allocated GPU-seconds/token. Show all three models in the same tables and plots, separate AR from native methods, and show each method relative to its own AR. Do not rank DSpark/MTP/DFlash draft quality from acceptance percentages at different widths. The historical `1+accepted/drafts` remains a labelled proxy until direct committed counts reconcile with end-of-request/fallback behavior.

Do not interpret three-repeat SD as a confidence interval. Publish repeat ratios, median/mean/SD and block design. For confirmed claims, report repeat-block uncertainty and the practical threshold; near-zero gain remains inconclusive if not resolved. Rank-0 traces, small quality checks, no open-loop SLO test and no capacity saturation must remain visible limitations. No supplied SLO means throughput/latency Pareto curves, not invented production requirements.

Suggested new artifacts (paths are created by the next session):

```text
results/spec-realtext-h100-v1/plan.json
results/spec-realtext-h100-v1/_datasets/manifest.json
results/spec-realtext-h100-v1/<model>/<workload>/<config>/<point>/repeat-N/
  manifest.json
  requests.jsonl
  summary.json
  telemetry/
results/spec-realtext-h100-v1/<model>/_server/
results/spec-realtext-h100-v1/<model>/profiles/
reports/spec-realtext-h100-v1/report.md
reports/spec-realtext-h100-v1/strengths-weaknesses.md
reports/spec-realtext-h100-v1/handoff.md
```

Keep weights, caches, credentials, private node/path details and full profiler traces out of Git. Curate small evidence with hashes and explicit sanitization; never change numerical results. Run `python tools/audit_references.py` on a clean clone including the intended published changes. Do not edit the earlier source repository. Commit/push/merge only as requested.

The handoff must give the last completed point and next exact command, all pending/failed/unsupported cells, dataset and runtime pins, actual rental duration, server-stopped confirmation, provisional findings and unresolved questions. Core completeness follows [target.md section 12.4](../target.md#124-completion-and-handoff); optional/deferred work is not disguised as required unfinished work.
