# Target: DeepSeek V4 Flash vs V4.1 Flash

Measurement and comparison plan for the next GPU session. No new GPU measurements or V4.1 results are claimed in this document.

**Goal:** measure DeepSeek V4 Flash and V4.1 Flash on matched hardware, compare both whole-serving performance and the speed of individual components, and explain each model's measured strengths, weaknesses, and bottlenecks. The sequence is **measure each model -> compare matched results -> investigate differences -> recommend operating points**. Do not assume V4.1 is faster.

**Scope priority:** DeepSeek V4/V4.1 is the only active study. [GLM](references/glm-5.3-flash/README.md) and [Qwen](references/qwen3.8-flash-next-fp8/README.md) are dormant historical references for a later explicit user request. Do not start their benchmarks, tuning, research, or expanded comparison automatically, including after DeepSeek finishes. See [agent instructions](AGENTS.md) and the [curated reference index](references/README.md).

**Reference:** [local V4 reference](references/deepseek-v4-flash/README.md), including original results, logs, environment, methodology, and corrected caveats. Everything needed from the prior study is included in this repo. Historical results are context, not a substitute for a fresh matched V4 baseline.

## Alignment with slides 13 and 15

Source: physical slides 13 and 15 of `SyFI_ML_Serving_refined.pptx` in the earlier `LLMs_frontier_serving` study. The requirements below are preserved here so the GPU session does not need access to that file. Physical slide 13 has a visible `09` label; use its title to identify it.

| Slide | Question to carry forward | Required DeepSeek comparison |
| --- | --- | --- |
| 13: Higher concurrency buys throughput at a latency cost | How much throughput does higher concurrency buy, and how much do TTFT and token latency worsen? | At 16,384 input / 256 output tokens, compare V4 and V4.1 at the same concurrency, including 1, 4, 16, 64; plot throughput and latency together. Record actual engine batch sizes. |
| 15: More MTP steps can be slower | Do accepted speculative tokens repay drafting, verification, and state-management costs at each concurrency? | Compare MTP off, one speculative token, and a larger supported depth (target five) for both models; measure the costs of each stage and net speedup. Acceptance alone is not a performance conclusion. |

Slide 13 reports historical DeepSeek/GLM/Qwen deployments; slide 15 reports a historical GLM experiment. Neither establishes V4.1 performance or the cause of a DeepSeek bottleneck. Reuse their questions and measurement dimensions, not their model rankings or conclusions.

## What the agent should measure

| Experiment | Proposed workload | Record |
| --- | --- | --- |
| Concurrency scaling | 16,384 input / 256 output tokens; concurrency 1, 2, 4, 8, 16, 32, 48, 64 | Output tokens/s, requests/s, TTFT, TPOT, end-to-end latency; throughput/latency tradeoff |
| Context scaling | Input 16,384 / 65,536 / 131,072 / 260,000 tokens; 256 output; concurrency 8, within verified shared limits | Latency and throughput versus context, cache occupancy, preemptions, OOMs and failures |
| Prefix reuse | 65,536-token shared prefix + 2,048-token suffix; 256 output; concurrency 8; 1 / 4 / 16 distinct prefixes | Cache-off, cold-fill, and prewarmed results; cache-hit rate and TTFT savings |
| MTP effect | Off / one speculative token / a larger supported depth (target five); concurrency 1 / 4 / 16 / 64 at 16K input / 256 output | Throughput/latency changes, acceptance rate by position where available, committed tokens per round, draft/verification/state costs, memory overhead; include real-text prompts |
| Resource efficiency | All experiments | Peak GPU memory separately from KV-pool occupancy, allocated cache capacity, GPU-seconds/output token; cost only with an explicit GPU-hour price |

Report p50/p95 TTFT, TPOT, inter-token latency, and end-to-end latency, plus variation across at least three measured repeats after warmup. Save request-level samples and sample counts; small runs do not establish reliable tail latency. Output throughput includes input processing; TTFT includes queueing. Client concurrency is not engine batch size.

For MTP, define depth as the number of speculative candidate tokens, and save the runtime's actual interpretation of the setting. Use common supported depths for matched comparisons. If five is unavailable, record why and use a common supported larger depth; if none exists, mark that arm unsupported. Keep model-specific depths as separate tuning results. Record metric definitions: accepted draft tokens, committed output tokens (including any target/bonus token), and accepted length need not mean the same thing.

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
| MTP / speculative execution | Draft, target verification, accept/reject, rollback/cache updates and other state handling | Time and memory per round, candidates and committed tokens, rejection cost; which overhead explains the net gain or regression? |
| KV/cache management | Allocation, prefix lookup/reuse, eviction, preemption/recompute, and transfers if used | Capacity and occupancy, hit/miss counts, bytes where available, events and time; distinguish allocated memory from useful occupancy. |

Use timeline/operator profiling first, with CPU ranges and GPU kernels correlated to layers and stages. Use targeted hardware-counter profiling for dominant kernels when needed to test compute, bandwidth, or communication hypotheses. Choose profiler commands supported by the installed runtime and GPU tools and save them with the results.

- Capture representative short traces at 16K context with concurrency 1, 16, and 64; add the longest common feasible context at concurrency 8. Include cache-off versus prewarmed prefix traces and MTP off/one/larger-depth traces at low and high concurrency. Extend profiling when a curve changes unexpectedly.
- Keep authoritative throughput/latency runs unprofiled. Collect comparable profiled runs separately, quantify profiling overhead, and keep graph-capture or execution-mode changes in separate diagnostic arms.
- Report component GPU milliseconds per prefill chunk and per decode step, plus milliseconds per processed prefill token or useful committed decode token with an explicit denominator. For MTP, also report time per round; candidate tokens are not useful output throughput.
- Report critical-path wall time alongside kernel time. Overlapping kernels, communication, nested ranges, and ranks cannot simply be summed into end-to-end latency. Identify exposed communication, waiting, and unattributed time; do not add component percentiles together.
- Use a non-overlapping accounting scheme for component shares, with its denominator stated. Attribute a fused kernel to a combined component if its internals cannot be timed; do not invent separate timings. Microbenchmarks may test a hypothesis, but must be labeled and confirmed against full-model traces.

## Turn measurements into strengths and weaknesses

For every matched workload, report V4, V4.1, and `V4.1 / V4`. A ratio above one is better for throughput and worse for latency, time, memory, or GPU-seconds per output token. Report absolute differences and repeat variation alongside ratios; label results inconclusive when the variation does not support a clear ordering.

Compare both absolute component time and its share of the measured execution time. A smaller percentage can still mean a slower component if total time grew. Map functionally corresponding blocks across architectures and disclose different layer counts, shapes, precision, or fusion. Whole-model gains need not imply faster individual kernels.

For each claimed strength or weakness, record:

1. The workload and operating point where it appears, including context, concurrency, cache state and MTP depth.
2. The measured endpoint difference and the component-level evidence, with links to raw results and traces.
3. A proposed explanation and a controlled follow-up that tests it, such as changing context, batch, cache state, or MTP depth while keeping other settings fixed.
4. The supported conclusion, remaining confounders, and the practical operating recommendation.

Investigate both improvements and regressions: low-concurrency responsiveness, saturation throughput, long-context scaling, prefix reuse, MoE/communication balance, and MTP break-even points. Treat bottlenecks as workload-dependent. High KV occupancy, parameter counts, acceptance rate, or runtime FLOP estimates alone do not establish the cause.

Use the pinned model config, official model documentation, runtime source, and collected traces when researching explanations. Save the relevant source URL/path and revision. Separate published claims, hypotheses, and local measurements. Include a fixed real-text task set with expected-answer or task-success checks; speed alone cannot establish equal quality. Application search/retrieval quality would require its own specified corpus, queries, and relevance metrics and is not measured by synthetic serving traffic.

## Comparison rules

- Use the same GPU node/count, pinned runtime where compatible, workload, sampling/reasoning mode, precision, TP/EP, scheduler limits, and memory budget. Record exact model revisions, resolved settings, and unavoidable differences. If a common runtime or precision is impossible, label the result a deployment comparison and do not attribute the full difference to architecture.
- Start with text-only, MTP-off baselines. Keep tuning and MTP arms separate; do not merge historical builds or memory settings into one curve.
- Match seeds across models/repeats, isolate cache state between runs, and verify actual token counts. Check representative text responses for correctness, truncation, and reasoning-output differences before interpreting speed.
- Alternate model run order; retain raw JSON, server logs, manifests, and every failure. Require all planned requests to complete for a valid performance point. Mark missing telemetry as unavailable, not zero.
- Current `run.sh` sets 32,768 maximum context and eight server sequences. Validate support and configure appropriate limits before larger sweeps; label unsupported points explicitly.

## Next GPU session: execution order and handoff

The repo currently contains installation, launch, and smoke-request scripts plus historical V4 evidence. It does **not** yet contain the automated benchmark/profiling harness described here. Build and validate that harness during GPU setup; do not treat the current smoke request as a benchmark.

1. **Inventory and pin.** Read this plan and the local reference. Record GPU model/count/memory, topology/interconnect, driver/CUDA, clocks/power settings, CPU/RAM, runtime and dependency versions, exact checkpoint/tokenizer revisions, and model configs. Verify access and runtime support for both models and MTP depths. `install.sh` installs a moving nightly: pin the resolved build before comparative runs. If either model is unavailable, report the blocker and any one-model measurements separately; do not invent the missing baseline.
2. **Prepare matched launch configurations.** Parameterize the model, parser/tokenizer settings, precision, TP/EP, memory budget, context limit, sequence limit, batching/chunking, prefix cache, graph mode, and MTP depth. Save both requested and resolved settings. Adapt `run.sh` for the matrix: its 32K context, eight sequences, and 4,096 batched-token limit are setup defaults, not validated sweep settings. Start one model server at a time on the reserved node and verify the served identity.
3. **Implement collection and smoke-test it.** Create a harness that writes immutable run directories, request-level timings/counts, server logs, raw telemetry, and manifests. Validate a small request and a small concurrent batch for each model; confirm streaming timestamps, token accounting, reasoning settings, no truncation, and expected completions. Verify profiler access and component attribution with a short trace before launching a long sweep. Record instrumentation gaps explicitly.
4. **Declare the run manifest.** Fix workload/token construction, seeds, request counts or measurement duration, warmup, cache reset/prewarm procedure, repeat count, timeout/failure rules, and model order before collection. Size samples for a sustained measurement window and meaningful latency estimates; do not silently reuse the historical tiny request counts. Match token counts for controlled synthetic experiments and use identical text for the real-text task set, recording tokenizer differences. Include both input and output plus template headroom in context-limit checks.
5. **Run the unprofiled baselines.** Warm representative shapes using disjoint prompts, establish the intended cache state, and run the concurrency and context sweeps with MTP off. Alternate model order across at least three measured repeats. Baseline batch/context measurements should disable prefix reuse or demonstrate no unintended hits. Retain failures and infeasible points rather than silently reducing the workload.
6. **Run cache and MTP experiments.** Measure cache-off, cold-fill, and prewarmed prefix controls with matched request order. Measure each common MTP depth at the planned concurrency points using synthetic and real-text prompts. Verify acceptance/committed-token accounting and collect memory and state-management telemetry. Preserve the off baseline under identical settings.
7. **Profile and investigate.** Collect the representative component traces specified above, build the layer-to-kernel mapping, and compare prefill/decode component costs between models. Use focused counter collection or controlled reruns to investigate dominant differences. Record unresolved explanations instead of asserting an unsupported bottleneck.
8. **Validate and report.** Check model/config identity, all planned completions, actual token counts, cache state, repeat coverage, and telemetry units before accepting each performance point. Keep invalid points in a failure table. Produce the comparison tables, curves, component breakdowns, and evidence-backed strengths/weaknesses below. Leave exact commands and remaining experiments for continuation.

Suggested artifact layout (to be created by the GPU session):

```text
results/<study-id>/plan.json
results/<study-id>/<model>/<workload>/<config-id>/repeat-<n>/
  manifest.json          # checkpoint/runtime/hardware/resolved settings, commands, seed
  requests.jsonl         # per-request status, tokens, timing and task checks
  summary.json           # aggregates, sample counts and repeat identity
  server.log
  telemetry/            # raw GPU, cache, scheduler and MTP observations
  profiles/             # traces/counters from explicitly labeled diagnostic runs
reports/<study-id>/comparison.md
reports/<study-id>/components.csv
reports/<study-id>/figures/
reports/<study-id>/handoff.md
```

The handoff must identify completed and pending matrix points, failures, exact rerun commands, artifact paths, pinned versions, known limitations, and the next unresolved profiling questions. Missing measurements remain pending or unavailable, never zero.

## Deliverable

The study is complete when it supplies:

- A whole-serving comparison table and throughput/latency curves across concurrency, context, prefix states, and supported MTP depths, including V4.1/V4 ratios, repeat variation, sample counts, failures, and configuration caveats.
- Prefill and decode component/layer tables with absolute times, normalized costs, shares with explicit denominators, comparison ratios, trace links, and absent/fused/unavailable entries. Cover every applicable component category above.
- An MTP cost breakdown relating draft, verification, and state time to useful committed output and the observed net gain or slowdown at each measured operating point.
- A strengths/weaknesses table for **both** models, with workload-specific evidence, explanation confidence, and recommended operating points. Quality checks and deployment differences accompany speed claims.
- A reproducible artifact bundle and GPU-session handoff. Every planned point is either measured and validated or explicitly classified as failed, unsupported, or still pending; pending points mean the study remains incomplete.

The desired outcome is not simply a winner by output tokens/s. It is a measured account of **which model is faster, in which blocks and workloads, why the evidence supports that conclusion, and how to configure the next deployment or experiment**.
