- Kan (SyFI lab) message "We want to teach people architecture of new generation of models. Specifically we want to compare the attention and FFN side. Which model's attention take minimal runtime loading/HBM or runs fastest? For prefix caching, which model is less space consuming? FFN side what are there sparsity differences? Feel free to add more!"
- Our task: run experiment to measure these requirement carefully on H100, same cpu, same set up. Based on carefully number + architect technical report, inference of model -> they match and explanable in an easy to understand with reader
- Example blog: https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/
- Model: Deepseek v4 Flash, v4.1 Flash, MiMo-V2.6-Flash-MOPD first

## Experiment plan for the measurement agent — 2026-10-07

**Status: proposed, not measured; this change is planning only.** Focus this blog on explaining attention, memory movement and FFN/MoE execution on H100. The broader speculative-serving matrix in [target.md](target.md) remains for later. Use a separate proposed study ID, `blog-architecture-h100-v1`; its results do not complete or replace `spec-realtext-h100-v1`, and must not be appended to historical curves.

The reader should leave understanding why fewer attended positions or fewer active parameters do not automatically mean a faster model. Build each explanation from three distinct kinds of evidence: architecture/checkpoint facts, what the pinned runtime actually executes, and measured time/bytes. A deployment may win one component and lose overall.

### 1. Priorities and scope

| Priority | Question the agent should answer | Evidence to collect |
| --- | --- | --- |
| P0 | What does each model actually execute? | Architecture and tensor-shape table, pinned source paths, loaded decoder, layer types, native precision and kernel mapping |
| P1 | Whose attention is faster during prefill and decode, and how does context change that? | Phase-specific latency, complete attention-path breakdown, matched batch/context diagnostics |
| P1 | Which attention path moves fewer bytes through HBM? | Targeted hardware read/write counters; distinguish weight, KV and temporary-state traffic where attribution permits |
| P1 | How much memory does a live request's attention state occupy? | Logical state calculation and measured physical live-block bytes, including replication and allocator overhead |
| P1 | What do FFN sparsity differences save or cost in practice? | Routed/shared/dense expert structure, actual routing, expert GEMMs, dispatch/combine, padding and exposed communication |
| P2 | Do these component differences explain user-visible performance? | Fresh unprofiled AR controls, TTFT, TPOT, end-to-end latency and throughput on the same public texts |
| Conditional | Can one supported implementation change test a specific explanation? | A separately budgeted, single-change ablation with a fresh matched baseline |

Start with **autoregressive decoding only, speculation off** for all three models. DSpark, MTP/DFlash widths, adaptive verification, 2048-token generation, topology sweeps and H200 belong to later work. Do not require them to finish this architecture blog.

**Prefix-cache boundary:** Kan's brief includes prefix memory, but the current project instruction still defers new prefix-cache experiments until Tan explicitly requests them. Keep prefix caching disabled in every proposed launch. Measure ordinary live-request KV/state memory now; describe retained-prefix capacity/reuse as unanswered. Existing prefix results can appear as historical context with their warming caveats. Neither live KV bytes nor advertised token-pool capacity proves which model retains more reusable prefixes. This document does not authorize cold/warm, reuse, prewarming or prefix-pressure runs.

### 2. Common deployment and local preparation

Use the three immutable checkpoints in [target.md](target.md): **V4 Flash 0731**, V4.1 Flash and MiMo-V2.6-Flash-MOPD. The older DeepSeek preview is not a substitute. Pin vLLM `44af287ebe38d6dc4e102948025f5e3e175aefd6`, tokenizer and remote code. Preserve TP8 + EP, utilization 0.90, context limit 262144, max sequences 64, explicit 8192-token chunk budget and native precision. Record resolved graph modes, quantization, KV format and attention/MoE backends. Use text only, temperature 0, `thinking=false` for DeepSeek and `enable_thinking=false` for MiMo.

Run on the same exclusive 8×H100 80GB SXM node and CPU setup. Record GPU topology, clocks/power, CPU model/affinity, RAM/NUMA, driver/CUDA and dependency versions. Keep those settings fixed. Native formats differ, so conclusions compare these deployments; they do not isolate architecture from quantization or kernel quality.

Before rental, the agent must prepare:

1. **Source worksheet:** architecture-report/model-card URLs and revisions, checkpoint config/tensor shapes, pinned runtime symbols and a proposed operator-to-component map. Verify paper features against the loaded implementation on the node. In particular, resolve MiMo's text decoder behind the Omni wrapper and check actual DeepSeek compression/indexer/CED execution.
2. **Public inputs:** balanced code/math/chat texts with licenses or authored provenance, immutable hashes, request IDs and held-out examples. Construct meaningful approximately 1K/16K/64K inputs. Apply exactly one chat template; save actual per-model token counts. Same-text serving comparisons and exact-token-shape diagnostics are separate views.
3. **Collection:** a study-specific manifest, per-request 256-token validation, phase segmentation, live KV-block accounting, trace classification and a small counter-collection pilot. Do not claim the existing single-request profiler already provides synchronized B=8 decode or HBM attribution.
4. **Execution:** a new no-prefix manifest-driven chain using owned-process cleanup patterns from `bench/chain_lib.sh`. Do not invoke `chain_mimo.sh` or `chain_dspark.sh`. Dry-run unique paths, allowed models, pins, counts, no-prefix checks, failure handling and shutdown before renting.

### 3. B0 — Architecture and implementation facts

Create one side-by-side table, with every value linked to its source. Record total layers and layer types; hidden/head dimensions; query/KV heads and TP replication; full/sliding/compressed/sparse attention; window and selection sizes; KV/cache dtype and auxiliary index state; attention projection shapes; total/routed/shared experts; top-k, expert width, dense FFN layers, weight precision and runtime backend.

Separate three meanings of sparsity:

- **Attention sparsity:** which past positions are accessed, including selection/compression work needed to find them.
- **MoE activation sparsity:** which experts execute per token, including shared experts and dense layers.
- **Weight sparsity/quantization:** zeros or lower-bit storage, only when actually present and supported. MoE routing does not establish hardware structured sparsity.

Compute nominal active parameters and linear FLOPs as labelled estimates. Also report resident weight bytes, including quantization scales/metadata. All resident experts consume memory even when an individual token routes to only a few. Never equate active parameter bytes with measured HBM reads.

### 4. B1 — Fresh serving controls and phase timing

Proposed initial matrix:

| Axis | Values |
| --- | --- |
| Models/arms | All three models, AR only, prefix caching off |
| Input buckets | Approximately 1K, 16K, 64K; publish actual rendered token counts |
| Output | Forced 256 tokens per request; natural-EOS checks collected separately |
| Client concurrency | c1 and c8 |
| Repeats | Three declared repeat blocks |
| Total | **3 models × 3 contexts × 2 loads × 3 repeats = 54 timing runs** |

c1 exposes per-request costs; c8 tests modest batching while keeping the long-context comparison bounded. This matrix does not establish high-load saturation or maximum capacity. Start with a complete 16K/c1,c8 comparison across all three models (18 timing runs); then complete the 1K/64K points (36 more). Rotate model order across the three blocks, and freeze the exact launch/workload order before collection. Nine AR launches can cover the full matrix if all six points per model run within each block; staging separate sessions adds launches and must be budgeted.

Use a disjoint pilot to size request lists: start at `max(16, 4c)`, round up for domain balance, and increase until the fastest model has at least a 60-second measurement window. Freeze identical text lists/counts/order across models for each point. Save startup/drain behavior and actual engine batch distributions; client c8 is not proof of an eight-sequence decode step. If counts prove insufficient, retain that attempt and collect a new matched set for every model.

Report output tok/s, requests/s, TTFT, per-request TPOT and end-to-end latency with mean, sample SD, median and repeat values. Collect engine prefill/decode intervals where observable. TTFT includes scheduling and first-token work; throughput proportionality does not prove prefill dominance. Keep the authoritative runs unprofiled. Before timing, check rendered inputs, output accounting and natural-EOS responses for each model; these are functional checks, not evidence of equal model quality.

### 5. B2 — Attention runtime and HBM traffic

**Define attention consistently.** Publish both the attention-core kernel and the complete attention path: Q/K/V and output projections, positional operations/norms, KV insertion, compression, indexer/top-k and core attention. Keep residual/mHC/Engram or other model-specific work in explicit categories where applicable. Assign fused kernels once; when a kernel crosses categories, label its combined scope instead of inventing a split. Separate FFN/shared-expert projections from attention projections even when historical reports grouped both under dense GEMM.

Collect these initial diagnostic points for every model:

| Diagnostic | Points per model | Total |
| --- | --- | ---: |
| Prefill trace | Single request at 16K and 64K input | 6 captures |
| Decode trace | Initial KV 1K and 64K, actual active B=1 and B=8 | 12 captures |
| **Trace budget** | Six captures/model | **18 captures** |

For prefill, record the full request phase and compare representative early/late chunks with their actual query-token count and prior KV length. An 8192 scheduler budget does not guarantee every chunk contains 8192 query tokens. For decode, finish prefill first, admit no new prompts during the measured window, and record exact active B, context distribution and generated-token positions. If synchronized batches cannot be maintained, segment faithful decode-only steps by actual B; unmatched states remain pending rather than being relabelled.

Use graph-aware tracing where possible. If graph replay hides attribution, make an explicitly separate eager/instrumented diagnostic arm and measure its overhead against the default execution. Collect all-rank timing or disclose rank coverage; report exposed collective waits and wall-clock spans separately from kernel sums. Overlapping GPU work is not additive latency.

After the traces identify relevant kernels, collect **six initial hardware-counter point sets**: each model at a late 64K prefill chunk and at 64K/B=1 decode. Cover representative attention-core, projection, indexer/compressor and FFN kernels, including distinct layer types. Freeze kernel/rank selections and actual replay-pass counts before collection; six point sets are not six cheap timing runs. Add B=8 counters only if needed to resolve a batching hypothesis.

Measure HBM read/write bytes, L2 behavior, kernel duration and compute activity where supported. Report bytes per processed prefill token and per committed decode token, with rank coverage and precision explicit. Separately identify weight/KV/intermediate bytes only where kernels or instrumentation support that attribution; total DRAM counters alone do not identify the source. Report attained bandwidth using matching byte/time scopes. Do not call a kernel bandwidth-bound solely because GPU utilization is high or a theoretical byte estimate is large.

Profiler replay, serialization and hardware-cache handling can change the execution being measured. Record tool/version, replay mode, pass count, cache/clock policy and overhead; validate multi-rank compatibility in a short pilot. Hardware L1/L2 cache handling is distinct from the prohibited application prefix-cache experiments. Use [NVIDIA's profiling guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) to choose a supported collection mode. If counters are inaccessible or replay is unreliable, retain timings and labelled byte estimates, and mark the measured HBM comparison unavailable.

Here, “loading/HBM cost” means memory traffic during inference. Checkpoint download, disk loading and server startup are recorded for rental budgeting, not mixed into attention runtime.

### 6. B3 — Live KV/state memory, with prefix caching off

At the B1 context/load states, collect diagnostic snapshots after prefill and at a fixed decode position, recording actual live sequences and lengths. Use matched unprofiled controls if hooks change execution. Reuse the workload definitions; no separate cache-reuse requests are needed.

Publish four separate quantities:

1. **Logical state:** source-derived KV/auxiliary bytes for each layer type as a function of context, including sliding-window retention, compression and indexer state.
2. **Physical live allocation:** occupied blocks × actual bytes/block for each cache group and rank, with partial-block waste, metadata and replicated state included or explicitly unavailable.
3. **Reserved pools:** total engine KV reservation, allocated/unused blocks and graph/workspace reservations.
4. **Total device memory:** weights plus runtime pools/workspaces and other allocations, measured independently.

Report per-rank values and the physical sum across eight GPUs; do not confuse replicated bytes with unique logical state. Because the engine can reserve KV pools at startup, a flat `nvidia-smi` memory reading does not mean request KV is free. Derive marginal live-state bytes/token from adjacent context points where valid, and explain nonlinear/block-rounded behavior rather than forcing one constant slope.

The figure should answer **“What does one live context cost?”** It must not claim “How many shared prefixes fit?” Record any preemption/recomputation and inability to sustain the intended active batch. Maximum request capacity needs a separate experiment.

### 7. B4 — FFN/MoE sparsity and its realized cost

Reuse B2 traces and B3 memory metadata instead of launching another full matrix. For prefill and decode separately, report router, dispatch/permutation, routed-expert GEMMs, shared experts/dense FFNs, combine and communication. Keep TP collectives distinct from EP dispatch/combine; verify the executed backend rather than assuming EP implies a particular all-to-all path.

Collect tokens assigned per expert/rank, unique experts touched per step, max/mean routing load, actual GEMM shapes and padding where exposed. A larger batch can touch many experts even if each token activates few. Measure routing/communication instrumentation overhead and keep this diagnostic execution separate from serving timing. Preserve domain labels; mixed-traffic counters cannot establish separate code/math/chat performance.

Show both total model FFN cost and per-layer/per-token cost. Different layer counts, shared experts, widths, quantization and batching can reverse rankings based on top-k alone. Attribute straggler waits only with aligned rank timelines: uneven routing by itself is not proof of a bottleneck. Pair sparsity estimates with measured expert time and sampled HBM bytes; explain missing coverage explicitly.

### 8. Conditional follow-ups and budget discipline

Select additional work only after the initial comparison identifies an unresolved question:

| Trigger | Bounded proposed extension |
| --- | --- |
| Need to explain large-batch expert/weight reuse | 1K/c64 AR for all three, three repeats = 9 timing runs, plus three actual B=64 decode captures if feasible |
| 64K leaves a context trend unresolved | 128K/c1,c8 AR across all three, three repeats = 18 timing runs, with selected additional diagnostics |
| A supported kernel or communication path may explain a measured cost | One variable changed for one model; select two workload points; baseline + intervention × two points × three repeats = 12 fresh timing runs, plus correctness/overhead checks |
| A small difference supports a final recommendation | Held-out prompts and at least five independently scheduled blocks at selected points; declare practical threshold and incremental count before collection |

Unsupported CED, new kernels or faithful multi-head MTP development must not consume rental time. Topology/precision changes require separate deployment arms and fresh controls. No automatic prefix, speculation or H200 extension.

The initial ledger is **54 unprofiled timing runs + 18 trace captures + 6 counter point sets**, with KV/routing diagnostics attached where feasible. Functional checks, pilots, warmups, diagnostic controls, counter replays, reruns and relaunches are additional costs. Estimate node-hours from the actual pilot and startup times, record an explicit allocation and stop point, and never treat run count as an hourly quote. Finish complete three-model comparisons first. Measure immediately after readiness, chain bounded work, stop only the owned server on every exit path, and verify GPU/process cleanup. Analyze while queued measurements run.

### 9. Blog figures and agent handoff

Use a question → measurement → explanation → limitation structure, inspired by the linked [TraceLab example](https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/). That article is a presentation reference, not benchmark evidence for these checkpoints.

Prepare these outputs:

- **Architecture diagram/table:** one token through each model's attention and FFN; paper design versus verified runtime behavior.
- **Context curves:** prefill and decode latency versus actual context at declared batch/load, alongside end-to-end serving anchors.
- **Component breakdown:** attention core, projections, selection/compression, FFN, communication and other/unattributed work. Label kernel-sum charts; show wall time separately where overlap prevents an additive stack.
- **HBM and KV figure:** inference bytes moved and live state bytes in separate panels, with measured versus estimated values visibly distinguished.
- **FFN sparsity figure:** nominal active experts/parameters alongside actual expert coverage, GEMM time, routing imbalance and exposed communication.
- **Findings table:** each model's workload-specific strength/weakness, supporting artifact, explanation confidence and unresolved confounders. Report negative and inconclusive findings too.

Suggested artifacts under `reports/blog-architecture-h100-v1/`: `architecture.md`, `serving.csv`, `components.csv`, `memory.csv`, `routing.csv`, `findings.md` and `handoff.md`, backed by immutable manifests/raw results under the matching `results/` study. Each row needs model/runtime identity, phase, prompt hash, actual token shape, client c/engine B, rank coverage, units, repeat/attempt and evidence path.

Historical [DeepSeek](report.md) and [MiMo](report_mimo.md) results select hypotheses only; apply the corrections in [update.md](update.md), especially kernel-sum versus critical-path time, token-pool versus capacity, and throughput accounting versus bottleneck evidence. Three repeats are a screen, not a confidence interval or a production tail-latency claim. Do not infer equal task quality from speed.

The handoff must list every planned point as measured/validated, pending, failed or unsupported; include next exact commands, missing instrumentation, actual rental cost/time and server-stopped confirmation. Missing HBM counters leave the HBM question unanswered even if the timing figures are complete. Curate small public artifacts with hashes and documented sanitization; exclude weights, private prompts/host details, credentials and large traces. Before publication, validate the evidence bundle with `python tools/audit_references.py` on a clean clone. Commit/push only when requested.
