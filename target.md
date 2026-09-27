# Target: DeepSeek V4 Flash 0731 vs V4.1 Flash

Measurement and comparison plan for the next GPU session. No new GPU measurements or V4.1 results are claimed in this document.

**Goal:** measure DeepSeek V4 Flash **0731 official release** (`deepseek-ai/DeepSeek-V4-Flash-0731`) and V4.1 Flash (`deepseek-ai/DeepSeek-V4.1-Flash`) on matched hardware, compare both whole-serving performance and the speed of individual components, and explain each model's measured strengths, weaknesses, and bottlenecks. The sequence is **measure each model -> compare matched results -> investigate differences -> recommend operating points**. Do not assume V4.1 is faster. In new result labels below, V4 means 0731, never the preview.

**Checkpoint correction (2026-09-27):** the retained old results name `deepseek-ai/DeepSeek-V4-Flash`, the preview repository, with no immutable checkpoint revision. They are not measurements of 0731. Both primary checkpoints use DSpark; do not assume classic MTP support from an `mtp` weight prefix or a `num_nextn_predict_layers` field. See the sourced [checkpoint audit and speculative-decoding plan](docs/speculative-decoding.md) for the method matrix, runtime checks, and optional preview MTP/DSpark bridge.

**Scope priority:** DeepSeek V4-0731/V4.1 is the only active study. [GLM](references/glm-5.3-flash/README.md) and [Qwen](references/qwen3.8-flash-next-fp8/README.md) are dormant historical references for a later explicit user request. Do not start their benchmarks, tuning, research, or expanded comparison automatically, including after DeepSeek finishes. See [agent instructions](AGENTS.md) and the [curated reference index](references/README.md).

**Reference:** [local V4 preview reference](references/deepseek-v4-flash/README.md), including original results, logs, environment, methodology, and corrected caveats. Everything needed from the prior study is included in this repo. Historical results are context, not a substitute for a fresh matched 0731 baseline.

## Alignment with slides 13 and 15

Source: physical slides 13 and 15 of `SyFI_ML_Serving_refined.pptx` in the earlier `LLMs_frontier_serving` study. The requirements below are preserved here so the GPU session does not need access to that file. Physical slide 13 has a visible `09` label; use its title to identify it.

| Slide | Question to carry forward | Required DeepSeek comparison |
| --- | --- | --- |
| 13: Higher concurrency buys throughput at a latency cost | How much throughput does higher concurrency buy, and how much do TTFT and token latency worsen? | At 16,384 input / 256 output tokens, compare V4 and V4.1 at the same concurrency, including 1, 4, 16, 64; plot throughput and latency together. Record actual engine batch sizes. |
| 15: More MTP steps can be slower | Do accepted speculative tokens repay drafting, verification, scheduling, and state-management costs at each concurrency? | For 0731 and V4.1, compare speculation off, fixed DSpark, and adaptive DSpark at a validated common proposal width, initially five. Sweep supported widths separately; classic preview MTP belongs to the optional mechanism bridge. Acceptance alone is not a performance conclusion. |

Slide 13 reports historical DeepSeek/GLM/Qwen deployments; slide 15 reports a historical GLM experiment. Neither establishes V4.1 performance or the cause of a DeepSeek bottleneck. Reuse their questions and measurement dimensions, not their model rankings or conclusions.

## What the agent should measure

| Experiment | Proposed workload | Record |
| --- | --- | --- |
| Concurrency scaling | 16,384 input / 256 output tokens; concurrency 1, 2, 4, 8, 16, 32, 48, 64 | Output tokens/s, requests/s, TTFT, TPOT, end-to-end latency; throughput/latency tradeoff |
| Context scaling | Input 16,384 / 65,536 / 131,072 / 260,000 tokens; 256 output; concurrency 8, within verified shared limits | Latency and throughput versus context, cache occupancy, preemptions, OOMs and failures |
| Prefix reuse | 65,536-token shared prefix + 2,048-token suffix; 256 output; concurrency 8; 1 / 4 / 16 distinct prefixes | Cache-off, cold-fill, and prewarmed results; cache-hit rate and TTFT savings |
| Speculative decoding | Each primary model: off / DSpark fixed / DSpark adaptive; initially five proposed candidates, validated on the pinned runtime; c1/4/16/64 at 16K input / 256 output, at least three repeats | Per-model off/on speedups and V4.1/0731 ratios; proposed/verified/accepted/committed counts; draft, confidence/scheduling, verification and state time; memory and correctness. Follow the [detailed matrix](docs/speculative-decoding.md). |
| Speculation extensions | 16K input / 2,048 output at c1/16; 64K and 128K input / 256 output at c8; fixed real-text code/math/chat; supported widths 1/3/7 after the primary matrix | Sustained-generation and context sensitivity, workload-dependent acceptance, break-even load, actual verification-length distributions, and separate tuned results |
| Resource efficiency | All experiments | Peak GPU memory separately from KV-pool occupancy, allocated cache capacity, GPU-seconds/output token; cost only with an explicit GPU-hour price |

Report p50/p95 TTFT, TPOT, inter-token latency, and end-to-end latency, plus variation across at least three measured repeats after warmup. Save request-level samples and sample counts; small runs do not establish reliable tail latency. Output throughput includes input processing; TTFT includes queueing. Client concurrency is not engine batch size.

For speculation, distinguish drafter layer count, trained block size, requested proposal width, actual verified length, accepted candidates, and committed output tokens. Pin draft sampling and rejection policy separately from target sampling. Verify the resolved method and whether adaptive verification actually executes. Use common supported widths for the controlled comparison; if an arm is unsupported, retain that status and reason. Do not force incompatible checkpoint fields or substitute classic MTP. Keep model-specific width/policy tuning separate. The initial primary matrix has 72 measured runs if all arms are supported; extensions are additional and must be declared before collection.

## Measure the components, not only the endpoint

First inspect each exact checkpoint configuration and the runtime implementation to map the real execution graph. A component name below is a measurement category, not a claim that both models implement the same architecture. Mark absent components as not applicable and unsupported instrumentation as unavailable. Save source paths, revisions, layer IDs/types, tensor shapes, dtypes, and kernel names used in the mapping.

Measure prefill and decode separately. Within prefill, record prompt length, chunk size, and cached versus newly computed tokens. Within decode, record KV length, active sequence count, and useful committed tokens per step. Collect a per-layer/type breakdown as well as component totals; do not extrapolate one layer across heterogeneous layers without evidence.

| Component or stage | What to separate | Measurements and diagnostic questions |
| --- | --- | --- |
| Request handling and scheduling | Tokenization/template work, queue wait, CPU scheduling, host/device transfers, output streaming | Wall time, queue delay, CPU/GPU gaps, scheduled tokens and actual active sequences; is low GPU utilization caused by host work or batching? |
| Embeddings and projections | Input embedding and attention input/output projections | GPU time and shapes per layer/type; how does cost change with prefill tokens and decode batch? |
| Attention and context selection | Any indexer/scoring, selection/top-k, compression, positional transforms, cache reads/writes, and core attention that exist | Time each separable stage versus context length; does selection or cache access erase core-attention savings? |
| Feed-forward / MoE | Routing, expert dispatch, expert compute, combine, and shared/dense experts where present | GPU time, tokens per expert/rank, load imbalance and padding; is the cost compute, imbalance, or communication? |
| Distributed communication | TP collectives and EP transfers, including dispatch/combine communication | Per-rank time, bytes where available, overlap and exposed waits; which communication lies on the critical path? |
| Residual, normalization, and other layer operations | Actual residual/mixing/normalization operations, conversions, and unfused elementwise work | GPU time, launch count and gaps; are many small operations significant in decode? |
| Output head and sampling | Final projection, logits processing, sampling and token delivery | GPU/CPU time per step and per committed token; include any reasoning-mode differences in output accounting. |
| DSpark / speculative execution | Parallel draft backbone, sequential/Markov module, confidence computation, verification scheduling, target verification, accept/reject, rollback/cache updates | Time and memory per round; proposed, verified, accepted and committed tokens; pruned/rejected work and actual verified-token batch size; which stage explains the net gain or regression? |
| KV/cache management | Allocation, prefix lookup/reuse, eviction, preemption/recompute, and transfers if used | Capacity and occupancy, hit/miss counts, bytes where available, events and time; distinguish allocated memory from useful occupancy. |

Use timeline/operator profiling first, with CPU ranges and GPU kernels correlated to layers and stages. Use targeted hardware-counter profiling for dominant kernels when needed to test compute, bandwidth, or communication hypotheses. Choose profiler commands supported by the installed runtime and GPU tools and save them with the results.

- Capture representative short traces at 16K context with concurrency 1, 16, and 64; add the longest common feasible context at concurrency 8. Include cache-off versus prewarmed prefix traces and speculation-off/fixed-DSpark/adaptive-DSpark traces at low and high concurrency. Extend profiling when a curve changes unexpectedly.
- Keep authoritative throughput/latency runs unprofiled. Collect comparable profiled runs separately, quantify profiling overhead, and keep graph-capture or execution-mode changes in separate diagnostic arms.
- Report component GPU milliseconds per prefill chunk and per decode step, plus milliseconds per processed prefill token or useful committed decode token with an explicit denominator. For speculation, also report time per round; proposed or verified candidates are not useful output throughput. Split V4.1 encoder/decoder execution and KV/index reuse where applicable.
- Report critical-path wall time alongside kernel time. Overlapping kernels, communication, nested ranges, and ranks cannot simply be summed into end-to-end latency. Identify exposed communication, waiting, and unattributed time; do not add component percentiles together.
- Use a non-overlapping accounting scheme for component shares, with its denominator stated. Attribute a fused kernel to a combined component if its internals cannot be timed; do not invent separate timings. Microbenchmarks may test a hypothesis, but must be labeled and confirmed against full-model traces.

## Turn measurements into strengths and weaknesses

For every matched workload, report V4, V4.1, and `V4.1 / V4`. A ratio above one is better for throughput and worse for latency, time, memory, or GPU-seconds per output token. Report absolute differences and repeat variation alongside ratios; label results inconclusive when the variation does not support a clear ordering.

Compare both absolute component time and its share of the measured execution time. A smaller percentage can still mean a slower component if total time grew. Map functionally corresponding blocks across architectures and disclose different layer counts, shapes, precision, or fusion. Whole-model gains need not imply faster individual kernels.

For each claimed strength or weakness, record:

1. The workload and operating point where it appears, including exact checkpoint, context, concurrency, cache state, speculative method, proposal width and verification policy.
2. The measured endpoint difference and the component-level evidence, with links to raw results and traces.
3. A proposed explanation and a controlled follow-up that tests it, such as changing context, batch, cache state, proposal width, or verification policy while keeping other settings fixed.
4. The supported conclusion, remaining confounders, and the practical operating recommendation.

Investigate both improvements and regressions: low-concurrency responsiveness, saturation throughput, long-context scaling, prefix reuse, MoE/communication balance, and speculative-decoding break-even points. Compare fixed and adaptive DSpark against each checkpoint's own off baseline before comparing their gains across models. Treat bottlenecks as workload-dependent. High KV occupancy, parameter counts, acceptance rate, or runtime FLOP estimates alone do not establish the cause.

Use the pinned model config, official model documentation, runtime source, and collected traces when researching explanations. Save the relevant source URL/path and revision. Separate published claims, hypotheses, and local measurements. Include a fixed real-text task set with expected-answer or task-success checks; speed alone cannot establish equal quality. Application search/retrieval quality would require its own specified corpus, queries, and relevance metrics and is not measured by synthetic serving traffic.

## Comparison rules

- Use the same GPU node/count, pinned runtime where compatible, workload, sampling/reasoning mode, precision, TP/EP, scheduler limits, and memory budget. Record exact model revisions, resolved settings, and unavoidable differences. If a common runtime or precision is impossible, label the result a deployment comparison and do not attribute the full difference to architecture.
- Start with text-only, speculation-off baselines. Keep fixed DSpark, adaptive DSpark, and width/sampling tuning separate; do not merge preview/0731 identities, historical builds, or memory settings into one curve. Classic MTP is confined to the optional preview mechanism bridge.
- Match seeds across models/repeats, isolate cache state between runs, and verify actual token counts. Check representative text responses for correctness, truncation, and reasoning-output differences before interpreting speed.
- Alternate model run order; retain raw JSON, server logs, manifests, and every failure. Require all planned requests to complete for a valid performance point. Mark missing telemetry as unavailable, not zero.
- Current `run.sh` sets 32,768 maximum context and eight server sequences. Validate support and configure appropriate limits before larger sweeps; label unsupported points explicitly.

## Next GPU session: execution order and handoff

The repo currently contains installation, launch, and smoke-request scripts plus historical V4 evidence. It does **not** yet contain the automated benchmark/profiling harness described here. Build and validate that harness during GPU setup; do not treat the current smoke request as a benchmark.

1. **Inventory and pin.** Read this plan and the local reference. Record GPU model/count/memory, topology/interconnect, driver/CUDA, clocks/power settings, CPU/RAM, runtime and dependency versions, exact checkpoint/tokenizer revisions, and model configs. Verify access to 0731 and V4.1 and support for DSpark widths and fixed/adaptive verification. Pin the bundled draft as well as the target; follow the identity gate in docs/speculative-decoding.md. `install.sh` installs a moving nightly: pin the resolved build before comparative runs. If either model is unavailable, report the blocker and any one-model measurements separately; do not invent the missing baseline.
2. **Prepare matched launch configurations.** Parameterize the model, parser/tokenizer settings, precision, TP/EP, memory budget, context limit, sequence limit, batching/chunking, prefix cache, graph mode, speculative method, proposal width, draft sampling, and verification policy. Save both requested and resolved settings. Adapt `run.sh` for the matrix: its 32K context, eight sequences, and 4,096 batched-token limit are setup defaults, not validated sweep settings. Start one model server at a time on the reserved node and verify the served identity.
3. **Implement collection and smoke-test it.** Create a harness that writes immutable run directories, request-level timings/counts, server logs, raw telemetry, and manifests. Validate a small request and a small concurrent batch for each model; confirm streaming timestamps, token accounting, reasoning settings, no truncation, and expected completions. Verify profiler access and component attribution with a short trace before launching a long sweep. Record instrumentation gaps explicitly.
4. **Declare the run manifest.** Fix workload/token construction, seeds, request counts or measurement duration, warmup, cache reset/prewarm procedure, repeat count, timeout/failure rules, and model order before collection. Size samples for a sustained measurement window and meaningful latency estimates; do not silently reuse the historical tiny request counts. Match token counts for controlled synthetic experiments and use identical text for the real-text task set, recording tokenizer differences. Include both input and output plus template headroom in context-limit checks.
5. **Run the unprofiled baselines.** Warm representative shapes using disjoint prompts, establish the intended cache state, and run the concurrency and context sweeps with speculation off. Alternate model order across at least three measured repeats. Baseline batch/context measurements should disable prefix reuse or demonstrate no unintended hits. Retain failures and infeasible points rather than silently reducing the workload.
6. **Run cache and DSpark experiments.** Measure cache-off, cold-fill, and prewarmed prefix controls with matched request order. Run the primary off/fixed/adaptive DSpark matrix before supported-width tuning and the optional preview mechanism bridge. Follow docs/speculative-decoding.md for sustained-generation, real-text, and context extensions. Verify proposed/verified/accepted/committed-token accounting and collect confidence/scheduler, memory, and state-management telemetry. Preserve each exact checkpoint's off baseline under identical settings.
7. **Profile and investigate.** Collect the representative component traces specified above, build the layer-to-kernel mapping, and compare prefill/decode component costs between models. Use focused counter collection or controlled reruns to investigate dominant differences. Record unresolved explanations instead of asserting an unsupported bottleneck.
8. **Validate and report.** Check model/config identity, all planned completions, actual token counts, cache state, repeat coverage, and telemetry units before accepting each performance point. Keep invalid points in a failure table. Produce the comparison tables, curves, component breakdowns, and evidence-backed strengths/weaknesses below. Leave exact commands and remaining experiments for continuation.

Suggested artifact layout (to be created by the GPU session):

```text
results/<study-id>/plan.json
results/<study-id>/<model>/<workload>/<config-id>/repeat-<n>/
  manifest.json          # target/draft revisions, runtime/hardware, resolved method/policy, commands, seed
  requests.jsonl         # per-request status, tokens, timing and task checks
  summary.json           # aggregates, sample counts and repeat identity
  server.log
  telemetry/            # raw GPU, cache, scheduler and speculative-round observations
  profiles/             # traces/counters from explicitly labeled diagnostic runs
reports/<study-id>/comparison.md
reports/<study-id>/components.csv
reports/<study-id>/figures/
reports/<study-id>/handoff.md
```

The handoff must identify completed and pending matrix points, failures, exact rerun commands, artifact paths, pinned versions, known limitations, and the next unresolved profiling questions. Missing measurements remain pending or unavailable, never zero.

## Deliverable

The study is complete when it supplies:

- A whole-serving comparison table and throughput/latency curves across concurrency, context, prefix states, and supported DSpark widths/policies, including V4.1/V4 ratios, repeat variation, sample counts, failures, and configuration caveats.
- Prefill and decode component/layer tables with absolute times, normalized costs, shares with explicit denominators, comparison ratios, trace links, and absent/fused/unavailable entries. Cover every applicable component category above.
- A checkpoint/method support table and speculative cost breakdown relating parallel/sequential draft, confidence/scheduling, verification, and state time to useful committed output. Report fixed/adaptive gains against each model's own off baseline, correctness evidence, break-even loads, and per-position/proposal/verification length distributions. Keep any preview MTP/DSpark bridge separate from the 0731/V4.1 comparison.
- A strengths/weaknesses table for **both** models, with workload-specific evidence, explanation confidence, and recommended operating points. Quality checks and deployment differences accompany speed claims.
- A reproducible artifact bundle and GPU-session handoff. Every planned point is either measured and validated or explicitly classified as failed, unsupported, or still pending; pending points mean the study remains incomplete.

The desired outcome is not simply a winner by output tokens/s. It is a measured account of **which model is faster, in which blocks and workloads, why the evidence supports that conclusion, and how to configure the next deployment or experiment**.
