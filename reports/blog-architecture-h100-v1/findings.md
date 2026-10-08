# Findings: attention, KV memory and FFN sparsity of three models on 8×H100

Study `blog-architecture-h100-v1`, measured 2026-10-07 on one 8×H100 80GB node. Plan: [blog_target.md](../../blog_target.md). Status of every planned item: [handoff.md](handoff.md).

**Read this first**

- **Build:** vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`). The completed studies in [report.md](../../report.md) and [report_mimo.md](../../report_mimo.md) used an older build with different kernels and a different V4.1 prefill path. **Do not compare numbers across the two.**
- **Deployment, identical for all models:** TP8 + expert parallel, memory utilization 0.90, context limit 262,144, 64 sequences, 8,192-token chunks, native precision, prefix caching off, speculation off, temperature 0, thinking off.
- **Workload:** public code, math and chat text, the same text for every model, 256 forced output tokens. Token counts differ slightly by tokenizer and are reported.
- **Evidence labels:** *measured* (unprofiled timing, three launch-separated repeat blocks), *trace* (profiler diagnostics, GPU kernel sums, not wall time), *snapshot* (live KV gauge), *source/log/config*, *estimate* (arithmetic, never a measurement).
- **Scope of a "win":** these are comparisons of three *deployments* (checkpoint + precision + kernels + this build), on these workloads. They do not rank architectures in isolation and say nothing about answer quality.

Sections 1–7 and 9 cover the three target models: DeepSeek V4 Flash 0731, DeepSeek V4.1 Flash, MiMo-V2.6-Flash-MOPD. Qwen3.8-Flash-Next and GLM-5.3-Flash were added during the session and are in [section 8](#8-added-models-qwen38-flash-next-and-glm-53-flash), with all five models side by side. **Qwen is the fastest of the five at five of six points; it and GLM also hold the most KV per token.**

## 1. Answers in one table

| Kan's question | Answer on this build | Evidence | Confidence |
| --- | --- | --- | --- |
| Whose attention runs fastest in **prefill**? | **MiMo up to about 16K of context; V4.1 from there on.** Attention work per 8,192-token chunk (core + indexer + KV work, projections excluded): MiMo 5.6 ms at 16K but 36.5 ms at 57K of prior context; V4.1 33.0 → 47.6 ms; 0731 61.5 → 110.7 ms | trace, section 3 | Medium: one request per point, kernel sums |
| Whose attention runs fastest in **decode**? | **MiMo at small batch or short context; a tie at 64K with eight sequences.** Per step: MiMo 0.64 ms at 1K/B=1, 1.95 ms at 64K/B=8; DeepSeek models stay at 1.6–2.1 ms everywhere | trace, section 3 | Medium |
| Whose attention moves the least data through HBM? | **Not measured.** No hardware counters were available. The estimate from live KV bytes says MiMo's full-attention layers read up to its whole live KV every step (about 2.8 TB/s implied at 64K/B=8), while DeepSeek's decode attention cost is flat with context | estimate, section 4 | Low: an inference, not a counter |
| Which model's KV state is smallest? | **V4.1, by a wide margin.** Per live token per GPU at 64K: V4.1 1.85 KiB, 0731 4.22 KiB, MiMo 5.69 KiB. One live 64K request holds 0.93, 2.14 and 2.77 GiB across the node | snapshot, section 5 | Medium: gauge × pool approximation |
| Which model keeps more *reusable prefixes*? | **Not measured.** Prefix caching was off by standing instruction. Live KV bytes are not prefix capacity | – | Open |
| What does FFN sparsity cost or save? | **Top-k does not predict FFN time.** Expert GEMM for an 8,192-token chunk costs 3.6 ms per layer on 0731 (6+1 experts), 5.0 ms on MiMo (8 experts) and 5.4 ms on V4.1 (6+1 larger experts). In decode at batch 1 the routing and activation work is 40–45% of the FFN path | trace, section 6 | Medium |
| Does this explain user-visible speed? | **Yes in direction.** MiMo decodes fastest (5.8–6.0 ms per token alone), V4.1 prefills fastest (34–35 µs per prompt token at 16–64K versus 46–50), so MiMo wins short or lightly loaded work and V4.1 wins long-context work under load | measured, section 2 | High for the measured points |

## 2. Serving: what a user sees

*Measured.* 54 of 54 runs valid: every request finished with exactly 256 output tokens. Mean ± sample SD over three blocks, each on a fresh server launch with rotated model order. Full tables with medians and every block value: [serving_tables.md](serving_tables.md); per-run rows: [serving.csv](serving.csv).

### Output throughput (tokens per second)

| Input | Client concurrency | 0731 | V4.1 | MiMo | Fastest |
| --- | ---: | ---: | ---: | ---: | --- |
| 1K | 1 | 118.1 ± 0.2 | 106.0 ± 0.1 | **160.6 ± 0.4** | MiMo, 1.36× 0731 |
| 1K | 8 | 598.7 ± 40.5 | 562.9 ± 12.6 | **687.7 ± 2.7** | MiMo, 1.15× 0731 |
| 16K | 1 | 93.4 ± 0.3 | 89.3 ± 0.2 | **114.3 ± 0.7** | MiMo, 1.22× 0731 |
| 16K | 8 | 237.5 ± 2.2 | **278.9 ± 1.2** | 250.9 ± 1.6 | V4.1, 1.11× MiMo |
| 64K | 1 | 48.5 ± 0.4 | **55.9 ± 0.4** | 55.4 ± 0.1 | V4.1 and MiMo within 1% |
| 64K | 8 | 71.5 ± 1.1 | **100.5 ± 1.1** | 78.5 ± 1.0 | V4.1, 1.28× MiMo |

0731 at 1K/c8 had a slower first block (551.9, then 621.8 and 622.5); its median is 621.8. The run is kept.

### Time to first token and time per output token (p50)

| Input | c | TTFT ms: 0731 | V4.1 | MiMo | TPOT ms: 0731 | V4.1 | MiMo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 208 ± 3 | 121 ± 1 | **114 ± 2** | 7.68 | 8.99 | **5.79** |
| 1K | 8 | 487 ± 1 | **397 ± 74** | 409 ± 10 | 10.81 | 12.32 | **9.93** |
| 16K | 1 | 759 ± 8 | **559 ± 6** | 762 ± 8 | 7.76 | 9.04 | **5.86** |
| 16K | 8 | 2,866 ± 40 | **1,815 ± 26** | 2,878 ± 86 | 21.98 | 20.82 | **20.16** |
| 64K | 1 | 3,282 ± 39 | **2,278 ± 35** | 3,142 ± 6 | 7.81 | 9.01 | **6.01** |
| 64K | 8 | 8,950 ± 371 | **5,807 ± 240** | 8,406 ± 246 | 73.17 | **54.90** | 68.26 |

TPOT SDs at c1 are below 0.01 ms.

### What this says

1. **Decode alone:** MiMo is fastest at every context (5.8–6.0 ms per token), 0731 next (7.7–7.8), V4.1 slowest (9.0). All three are nearly flat from 1K to 64K at one sequence.
2. **Prefill alone:** dividing c1 TTFT by the prompt tokens gives about 34–35 µs per token for V4.1 at 16K and 64K, 46–48 for MiMo and 46–50 for 0731 (*arithmetic*; TTFT also contains queueing and the first decode step). V4.1's lead is what its prefill skip predicts ([check](v41_prefill_check.md)).
3. **Why the ranking flips with context and load:** a 256-token answer after a 1K prompt is almost all decode, where MiMo leads. After a 64K prompt, prefill dominates; with eight clients, other requests' prefill chunks also interleave with everyone's decode steps, so the model with the cheapest prefill (V4.1) also shows the lowest TPOT at 64K/c8 (54.9 ms versus 68.3 and 73.2). TPOT at c8 is therefore not a decode-speed measurement.
4. **Client concurrency is not the engine batch.** The running-requests samples are in each run's `blog_validation.json`; exact engine batches are only known in the traces.

Limits: c1 and c8 only, so no saturation or capacity claim; closed loop, so no production latency claim; three blocks screen effects and are not confidence intervals; forced 256 tokens, so nothing about natural answer length.

## 3. Attention: where the time goes

*Trace.* GPU kernel time per engine step, mean over the eight ranks, from the idle-profiler launches. Kernel sums are not wall time. Full table: [components_tables.md](components_tables.md); rows per rank: [components.csv](components.csv).

"Attention path" below is attention core + indexer/top-k/candidates + Q/K/V norm, RoPE, KV insertion and attention metadata. **Projections are not included**: the kernels that run them are shared with other dense work, so they are listed separately as an upper bound.

### Prefill, one 8,192-token chunk

| ms per chunk | 0731 at 16K | V4.1 at 16K | MiMo at 16K | 0731 at 64K | V4.1 at 64K | MiMo at 64K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Step span on the rank | 355 | 263 | 329 | 405 | 262 | 361 |
| Attention core | 41.0 | 25.3 | 4.4 | 55.3 | 26.6 | 35.4 |
| Indexer / top-k / candidates | 6.4 | 2.4 | – | 41.1 | 15.5 | – |
| KV insert, norms, RoPE, metadata | 14.2 | 5.3 | 1.2 | 14.3 | 5.5 | 1.2 |
| **Attention path** | **61.5** | **33.0** | **5.6** | **110.7** | **47.6** | **36.5** |
| Dense GEMM (projections; MiMo also its layer-0 FFN) | 39.7 | 29.4 | 11.5 | 40.1 | 30.7 | 11.9 |
| mHC / residual / norm | 28.4 | 17.5 | 8.6 | 28.4 | 17.4 | 8.6 |
| MoE expert GEMM | 155.3 | 124.1 | 234.1 | 158.5 | 125.4 | 244.0 |
| TP all-reduce | 26.7 | 21.5 | 31.7 | 31.9 | 21.8 | 33.7 |

The 64K column is the last full chunk, with about 57K tokens already in the KV.

**How to explain it:**

- **MiMo** has no selection machinery. Its 39 sliding-window layers look at 128 tokens, which is nearly free. Its 9 full-attention layers read the whole context, so their cost grows with context: 4.4 ms at 16K, 35.4 ms at 57K. Simple and cheap when short, linear growth when long.
- **0731** attends to few positions (a 128-token window plus the top 512 compressed positions), so its attention core grows slowly (41 → 55 ms). But it must *find* those positions: the indexer scores all earlier positions and grows from 6 to 41 ms. At 64K the search costs almost as much as the attention it enables.
- **V4.1** uses the same idea with three savings that show up in the trace: only 21 of 40 layers process the full chunk in prefill, KV is shared across layers (4 source layers), and later indexing layers are limited to a candidate pool. Its attention path is about half of 0731's at 16K and 43% at 64K.
- **The crossover.** MiMo's attention path is 6× cheaper than V4.1's at 16K and 1.3× cheaper at 57K. Its growth between the two points is 31 ms per chunk against 15 ms for V4.1, so the lines cross somewhere beyond 64K. That is an extrapolation; the 128K point was not run.

### Decode, one step

| ms per step | 0731 1K B=1 | V4.1 1K B=1 | MiMo 1K B=1 | 0731 64K B=8 | V4.1 64K B=8 | MiMo 64K B=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Client step time in the window | 8.19 | 9.73 | 6.49 | 11.71 | 12.83 | 11.23 |
| Attention core | 0.97 | 0.91 | 0.56 | 1.11 | 1.01 | 1.87 |
| Indexer / top-k / candidates | 0.40 | 0.27 | – | 0.68 | 0.55 | – |
| KV insert, norms, RoPE, metadata | 0.22 | 0.55 | 0.08 | 0.25 | 0.58 | 0.08 |
| **Attention path** | **1.60** | **1.72** | **0.64** | **2.04** | **2.14** | **1.95** |
| Dense GEMM (projections) | 3.59 | 2.81 | 1.09 | 3.97 | 3.12 | 1.10 |
| mHC / residual / norm | 1.54 | 1.70 | – | 2.18 | 2.25 | – |
| MoE expert GEMM | 1.29 | 1.49 | 1.44 | 3.73 | 4.23 | 4.82 |
| MoE routing, activation, combine | 0.88 | 1.20 | 0.95 | 0.91 | 0.91 | 0.69 |
| TP all-reduce | 0.87 | 0.94 | 1.06 | 1.41 | 1.49 | 1.74 |

**How to explain it:**

- **The attention path is a small part of a decode step** for all three: 10–20%.
- **DeepSeek's decode attention is flat** with context and batch (1.6 → 2.1 ms): sparse selection reads a bounded number of positions.
- **MiMo's grows** with context × batch (0.64 → 1.95 ms) because its full-attention layers read every live token's KV. At 64K with eight sequences the three are level.
- **What actually makes DeepSeek decode slower is next to attention, not in it:** the projections (64 heads × 512 through low-rank Q and grouped O layers) cost 2.8–3.6 ms against MiMo's 1.1 ms, and the four-stream mHC mixing costs another 1.5–2.3 ms that MiMo does not have. Together that is more than the whole gap between the models at batch 1.
- If projections are counted as attention, the complete attention path at 1K/B=1 is at most 5.2 ms (0731), 4.5 ms (V4.1) and 1.7 ms (MiMo).

Trace quality: the step length seen in the trace equals the client's step time within 0–7% for all twelve decode captures (`span_share_of_step` in the CSV). The profiler lengthened decode steps by 0.4–12% (same window, measured with and without it). One request per prefill point; no repeat.

## 4. HBM traffic: what we can and cannot say

**No hardware byte counters were collected.** The node image has no Nsight Compute, and replaying kernels of an eight-process server under it was not attempted on rented time. Kan's question "which attention moves the least data through HBM" is therefore **not answered by measurement**.

What follows is an *estimate* built from two measured quantities: live KV bytes (section 5) and attention kernel time (section 3).

| Case (MiMo decode) | Extra live KV versus 1K, per GPU | Extra attention-core time per step | Implied read rate |
| --- | ---: | ---: | ---: |
| 64K, 1 sequence | about 339 MiB | 0.27 ms | about 1.3 TB/s |
| 64K, 8 sequences | about 2,848 MiB | 1.07 ms | about 2.8 TB/s |

Reading: if MiMo's full-attention layers read all of the added live KV once per step, the eight-sequence case moves data at about 2.8 TB/s per GPU, close to the H100's specified 3.35 TB/s. That is consistent with those layers being limited by memory bandwidth at long context and larger batch. It is an inference: the bytes come from an allocation gauge, the time from a kernel sum, and neither is a counter.

For the DeepSeek models the same reasoning gives the opposite picture: attention time does not grow with context in decode, so the bytes read per step do not grow either, apart from the indexer's pass over its own (small) state. V4.1 also holds the least KV per token. On this evidence the ordering "least KV traffic in decode" is plausibly V4.1, then 0731, then MiMo at long context; at short context MiMo's traffic is the smallest because there is almost nothing to read.

Weight traffic is separate from KV traffic and is larger at small batch: each decode step reads the active expert and projection weights once whatever the context. Estimates from shapes are in [architecture.md](architecture.md); they are not HBM measurements.

To close this question: install Nsight Compute before a session, validate it on one rank, and collect the six planned point sets ([handoff](handoff.md)).

## 5. Live KV/state memory

*Snapshot.* B requests decoding, nothing else running, prefix caching off. Live bytes = KV-usage gauge × per-GPU KV pool from the server log. Table for all 18 snapshots: [memory_tables.md](memory_tables.md).

| | 0731 | V4.1 | MiMo |
| --- | ---: | ---: | ---: |
| One live 16K request, per GPU | 88.9 MiB | 33.7 MiB | 94.2 MiB |
| One live 64K request, per GPU | 273.4 MiB | 119.4 MiB | 354.2 MiB |
| One live 64K request, whole node (8 GPUs) | 2.14 GiB | 0.93 GiB | 2.77 GiB |
| Eight live 64K requests, whole node | 17.0 GiB | 7.4 GiB | 23.1 GiB |
| KiB per live token per GPU at 64K | 4.22 | 1.85 | 5.69 |
| KiB per live token per GPU at 16K | 5.31 | 2.03 | 5.88 |
| KiB per live token per GPU at 1K | 18.0 | 4.2 | 7.7 |
| KV pool reserved per GPU | 47.43 GiB | 29.14 GiB | 49.18 GiB |
| Weights per GPU | 19.79 GiB | 36.32 GiB | 20.10 GiB |

**How to explain it:**

- **V4.1 holds 2.3× less than 0731 and 3.1× less than MiMo per token at 64K.** It shares one KV across layer groups (4 source layers) and compresses it. Its model card quotes 890 bytes per token for the global KV with FP4 KV caching; this build runs the FP8 KV format (`fp8_ds_mla`, log), and 1.85 KiB per token is about twice that figure. That is consistent, not a proof.
- **MiMo's number matches its design.** Nine full-attention layers keep (192 + 128) values per token in BF16 for the KV head each GPU holds: 9 × 320 × 2 bytes = 5.6 KiB (*estimate*), against 5.69 KiB measured. Its 39 window layers keep only 128 tokens.
- **0731's per-token cost falls with length** (18.0 → 5.3 → 4.2 KiB) because part of its state is fixed per request (the sliding-window state of 43 layers) and its KV blocks hold 256 tokens, so a short request pays for mostly empty blocks. This is the block-rounding effect the plan asked us to show, not a different slope.
- **Linear in batch.** Eight requests use 8.0× the bytes of one at 16K and 64K for all three models.
- **V4.1 trades KV for weights.** It reserves the smallest KV pool because its weights are the largest (36.3 GiB per GPU, plus two 11.8 GiB Engram tables held in host memory per rank). Per byte of pool it still fits the most context.

**What this does not say:**

- **Nothing about prefix caching.** Live bytes of running requests are not the capacity for retained, reusable prefixes, and no reuse or warming was run.
- **Nothing about maximum concurrency.** No saturation run.
- **Token capacities are misleading.** The server reports pools of 1.73M, 9.09M and 6.97M tokens. A 64K request occupied 0.56%, 0.40% and 0.70% of the pools, which by those token counts would be 9.7K, 36K and 49K tokens, not 66K. The reported capacity assumes every token keeps all of its state, while sliding-window state is released as the window moves. Compare bytes.
- **Approximation.** Usage is a block fraction; with several cache groups per model the conversion to bytes assumes blocks of one pool have one size.

## 6. FFN: nominal sparsity versus realized cost

All three FFNs are mixtures of experts. "Sparsity" here means that a token runs only a few experts. No model has structured zero weights.

| | 0731 | V4.1 | MiMo |
| --- | ---: | ---: | ---: |
| Routed experts per layer (config) | 256 | 384 | 256 |
| Active per token (config) | 6 routed + 1 shared | 6 routed + 1 shared | 8 routed |
| Fraction of routed experts used by a token | 2.3% | 1.6% | 3.1% |
| Active expert weights per layer (*estimate*) | 176M | 248M | 201M |
| Expert GEMM per layer for an 8,192-token chunk (*trace*) | 3.6 ms | 5.4 ms (full-chunk layers, rank 0) | 5.0 ms |
| Expert GEMM per prefill chunk, all layers (*trace*) | 155 ms | 124 ms | 234 ms |
| Expert GEMM per decode step, B=1 (*trace*) | 1.29 ms | 1.49 ms | 1.44 ms |
| Expert GEMM per decode step, B=8 (*trace*) | 3.7 ms | 4.2 ms | 4.9 ms |
| Routing + activation + combine per decode step, B=1 (*trace*) | 0.88 ms | 1.20 ms | 0.95 ms |
| FFN path share of a decode step, B=1 | 27% | 28% | 37% |

**How to explain it:**

1. **The lowest activation fraction is not the cheapest FFN.** V4.1 uses the smallest share of its experts (1.6%), yet each of its layers costs the most, because its experts are wider (2,304 versus 2,048) on a wider hidden state (5,120 versus 4,096) and it adds a shared expert that every token runs.
2. **Per chunk, depth and the prefill skip decide.** V4.1's layers are the most expensive, but only 21 of its 40 run on the whole chunk, so its chunk total (124 ms) is the lowest. MiMo runs 47 MoE layers on every token and pays the most (234 ms, 70–76% of its whole prefill chunk).
3. **In decode at batch 1, time follows weights read, not tokens.** One token costs 1.3–1.5 ms of expert GEMM; eight tokens cost 2.8–3.4× that, not 8×. A step must read each active expert's weights once, however few tokens use it, and a larger batch touches more distinct experts. MiMo grows the most (3.4×): eight tokens × eight experts can touch up to 64 experts per layer, against up to 48 routed plus one shared for DeepSeek.
4. **Selection is not free.** At batch 1 the router, top-k, token dispatch, activation and combine take 0.9–1.2 ms per step, which is 40–45% of the FFN path. In prefill the same work is 2–3% because one routing pass serves 8,192 tokens.
5. **MoE time is similar across the three in decode** (1.3–1.5 ms at batch 1) even though the active counts differ (7, 7 and 8). The FFN is not what separates their decode speed; projections and mHC are (section 3).

**Not measured:** tokens per expert, expert coverage per step, routing imbalance across ranks, and padding. The runtime exposes no such counters and no instrumentation was added, so statements about imbalance or stragglers are not supported. EP dispatch and TP all-reduce are both under "all-reduce / other communication" only as far as kernel names allow.

**One hypothesis worth a single-change test:** on this build all three models run experts on the `HUMMING` backend, and MiMo's expert GEMM is 70–76% of its prefill chunk. Whether another supported MXFP4 backend is faster for prefill is unknown; it needs a matched launch with only the backend changed ([handoff](handoff.md)).

## 7. Does the V4.1 measurement match its technical report?

Full check: [v41_prefill_check.md](v41_prefill_check.md).

- The report describes 20 encoder + 20 decoder layers, with about half the parameters active per prompt token.
- On this build, layers 0–20 process every prompt token and layers 21–39 only each request's last 128 tokens (*source, log, trace*: 21 long and 19 very short attention calls per chunk).
- Differences: 21 full layers instead of 20, and no skip for prefill steps under 768 tokens.
- vLLM's timer is a wall-clock interval around what runs. It was never wrong; on the old build the skipped layers were simply executed.

## 8. Added models: Qwen3.8-Flash-Next and GLM-5.3-Flash

Requested by Tan during the session and declared in [plan_addendum_glm_qwen.json](plan_addendum_glm_qwen.json) and [blog_target.md](../../blog_target.md) section 10. Same build, deployment, request lists, counts and rules. **36 of 36 timing runs valid**, 12 traces and 12 live-KV snapshots. Architecture facts: [architecture.md](architecture.md).

**Read these with three caveats:**

- **Order.** Both models ran after the first three, in the same night. Model order was not rotated across all five, so a slow drift of the node would look like a group difference.
- **Thinking.** GLM-5.3-Flash has no thinking-off switch. It ran at the lowest reasoning effort; one of its three natural-EOS test answers still carried a short reasoning passage. Forced 256-token timing counts the same number of decode steps either way.
- **Window length.** Counts were sized on the first three models. Qwen's windows were 46–85 s (46 s at 16K/c8, 58 s at 1K/c8 and 64K/c8); GLM's were 59–91 s.

### Serving, all five models

*Measured.* Mean ± sample SD over three blocks. The same text tokenizes to about 5% more tokens for Qwen (17,199 against 16,351 at "16K"), so Qwen does slightly more work per request than the others.

| Output tok/s | 0731 | V4.1 | MiMo | Qwen | GLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K, c1 | 118.1 ± 0.2 | 106.0 ± 0.1 | **160.6 ± 0.4** | 145.0 ± 0.1 | 137.3 ± 1.8 |
| 1K, c8 | 598.7 ± 40.5 | 562.9 ± 12.6 | 687.7 ± 2.7 | **897.5 ± 3.0** | 705.8 ± 1.7 |
| 16K, c1 | 93.4 ± 0.3 | 89.3 ± 0.2 | 114.3 ± 0.7 | **120.5 ± 0.1** | 109.4 ± 6.9 |
| 16K, c8 | 237.5 ± 2.2 | 278.9 ± 1.2 | 250.9 ± 1.6 | **412.6 ± 1.6** | 322.3 ± 6.1 |
| 64K, c1 | 48.5 ± 0.4 | 55.9 ± 0.4 | 55.4 ± 0.1 | **77.0 ± 0.2** | 66.4 ± 2.1 |
| 64K, c8 | 71.5 ± 1.1 | 100.5 ± 1.1 | 78.5 ± 1.0 | **145.1 ± 0.3** | 115.7 ± 0.2 |

| p50 | 0731 | V4.1 | MiMo | Qwen | GLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTFT ms, 1K c1 | 208 | 121 | **114** | 153 | 166 |
| TTFT ms, 16K c1 | 759 | 559 | 762 | 529 | **527** |
| TTFT ms, 64K c1 | 3,282 | 2,278 | 3,142 | **1,724** | 2,080 |
| TTFT ms, 64K c8 | 8,950 | 5,807 | 8,406 | **4,845** | 5,410 |
| TPOT ms, 1K c1 | 7.68 | 8.99 | **5.79** | 6.31 | 6.59 |
| TPOT ms, 64K c1 | 7.81 | 9.01 | **6.01** | 6.36 | 6.62 |
| TPOT ms, 1K c8 | 10.81 | 12.32 | 9.93 | **7.61** | 9.91 |
| TPOT ms, 64K c8 | 73.2 | 54.9 | 68.3 | **34.7** | 46.7 |

GLM's first block was slower at one client (16K: 101.4, then 113.7 and 113.0 tok/s; 64K: 64.0, then 67.5 and 67.7); medians are 113.0 and 67.5. Kept.

**What this says.** Qwen has the highest throughput at five of the six points, by 1.31× to 1.48× over the best of the first three at eight clients. MiMo keeps the fastest single-stream decode (5.8 ms per token) and so wins 1K with one client. GLM is second at the three eight-client points and at 64K with one client, and third at 1K and 16K with one client.

### Why: where the time goes

*Trace*, rank mean, ms. Full table: [components_tables.md](components_tables.md).

| One 8,192-token prefill chunk at 64K | 0731 | V4.1 | MiMo | Qwen | GLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step span on the rank | 405 | 262 | 361 | 192 | 242 |
| Attention path (softmax core + recurrent update + indexer + KV work) | 110.7 | 47.6 | 36.5 | 47.4 | 69.4 |
| of which recurrent-layer state update | – | – | – | 5.1 | 15.7 |
| MoE expert GEMM | 158.5 | 125.4 | 244.0 | 10.4 | 25.1 |
| MoE routing, activation, combine | 3.1 | 3.4 | 4.8 | 7.5 | 9.4 |
| Dense GEMM (projections) | 40.1 | 30.7 | 11.9 | 32.8 | 27.1 |
| Residual streams / norm | 28.4 | 17.4 | 8.6 | 32.1 | 39.0 |
| TP all-reduce | 31.9 | 21.8 | 33.7 | 33.0 | 34.5 |

| One decode step | 0731 | V4.1 | MiMo | Qwen | GLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Client step time, 1K, B=1 (unprofiled window) | 7.75 | 9.01 | 5.78 | 6.33 | 6.67 |
| Client step time, 64K, B=8 (unprofiled window) | 11.50 | 12.66 | 11.11 | 7.94 | 10.20 |
| MoE expert GEMM, B=1 → B=8 | 1.29 → 3.7 | 1.49 → 4.2 | 1.44 → 4.9 | 1.04 → 1.45 | 0.92 → 2.9 |
| Attention path, 64K, B=8 | 2.04 | 2.14 | 1.95 | 0.77 | 0.76 |

**How to explain it:**

1. **The FFN decides the prefill ranking, and the expert kernel matters more than the expert count.** Qwen and GLM run FP8 experts on the DeepGEMM path; the first three run 4-bit experts on `HUMMING`. GLM's experts have the same shape as 0731's and MiMo's (hidden 4,096, width 2,048) and nine are active per token, yet its expert time per chunk is 25 ms against 158 and 244 ms. That is a 6–10× gap at equal or larger nominal work. Four-bit experts halve the bytes but, on this build, cost far more compute per token in prefill. This is a comparison across models, not a controlled test; it is the strongest single-change candidate in the handoff.
2. **Qwen's experts are small.** 512 experts of width 640 on a 2,560-wide state: about 54M active expert weights per layer (*estimate*) against 176–248M for the others. Its expert time is 10 ms per chunk and grows only 1.4× from one to eight decoding sequences (the others grow 2.8–3.4×), which is why its step time barely rises with batch (6.3 → 7.9 ms) and why it wins every eight-client point.
3. **Recurrent attention is cheap and flat.** In both models three of every four layers keep a fixed-size state instead of a growing KV. Their update costs 5–16 ms per chunk and 0.2–0.3 ms per decode step, independent of context. The remaining sparse-attention layers behave like DeepSeek's: an attention core that barely grows and an indexer that grows with context (Qwen 4.8 → 12.2 ms, GLM 2.0 → 13.1 ms per chunk from 16K to 64K).
4. **Decode attention is the smallest of the five** (0.76–0.77 ms per step at 64K with eight sequences, against about 2 ms for the first three): few softmax layers, and sparse reads in those.
5. **What they pay instead:** residual-stream mixing (32–39 ms per chunk, 1.4–1.9 ms per decode step) and projections (27–33 ms per chunk, 2.3 ms per decode step) are as large as in the DeepSeek models. MiMo remains the only model without that cost, which is why it still has the fastest single-stream decode.

### Live KV, all five models

*Snapshot.* Same method and caveats as section 5; table in [memory_tables.md](memory_tables.md).

| | 0731 | V4.1 | MiMo | Qwen | GLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| KiB per live token per GPU at 64K | 4.22 | **1.85** | 5.69 | 13.16 | 12.00 |
| One live 64K request, whole node | 2.14 GiB | **0.93 GiB** | 2.77 GiB | 6.85 GiB | 5.77 GiB |
| Eight live 64K requests, share of the KV pool | 4.5% | 3.2% | 5.9% | 13.6% | 20.7% |
| KV pool per GPU | 47.43 GiB | 29.14 GiB | 49.18 GiB | 51.44 GiB | 28.73 GiB |
| Weights per GPU | 19.79 GiB | 36.32 GiB | 20.10 GiB | 17.36 GiB | 38.8 GiB |

**The fastest models hold the most state.** Qwen and GLM keep 2–7× more KV bytes per token than the first three. Their sparse-attention layers *read* few positions but *store* all of them uncompressed: for Qwen, 12 layers × one KV head per GPU × 512 values × 2 bytes ≈ 12.3 KiB per token (*estimate*) against 13.2 measured. The recurrent layers add a fixed state per request, visible as a higher cost per token at 1K (27–35 KiB). GLM combines that with the smallest pool, so eight 64K requests already occupy a fifth of it. For long shared contexts or many long sessions this is the opposite end of the trade-off from V4.1. It says nothing about prefix reuse, which was not run.

### Strengths and weaknesses of the added models

| Model | Strength on this build | Weakness on this build | Explanation and confidence |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | Highest throughput at five of six points (412.6 tok/s at 16K/c8, 145.1 at 64K/c8); fastest TTFT at 64K (1.72 s); step time nearly flat with batch | Largest KV per token (13.2 KiB per GPU); MiMo decodes faster alone | Small FP8 experts and mostly recurrent layers (config + trace, medium). Stores uncompressed KV in 12 layers (estimate + snapshot, medium) |
| GLM-5.3-Flash | Second-fastest at eight clients (322 tok/s at 16K, 116 at 64K); TTFT 527 ms at 16K | KV 12.0 KiB per token with the smallest pool; largest weights (38.8 GiB per GPU); no thinking-off switch; slower first block at one client | FP8 experts on DeepGEMM (trace, medium). mHC and projections as costly as DeepSeek's (trace, medium) |

## 9. Strengths and weaknesses

| Model | Strength on this build | Weakness on this build | Explanation and confidence | Not controlled |
| --- | --- | --- | --- | --- |
| MiMo-V2.6-Flash-MOPD | Fastest decode (5.8–6.0 ms/token alone); highest throughput at 1K (1.15–1.36× 0731) and at 16K with one client; cheapest attention up to 16K | Expert GEMM dominates prefill (234 ms of a 329 ms chunk); attention and KV grow linearly with context; largest KV per token (5.7 KiB per GPU) | No indexer, no mHC, light projections make each step cheap (trace, medium). Full-attention layers explain the growth (trace + snapshot, medium) | Expert backend is the build's choice; BF16 KV versus FP8 for DeepSeek |
| DeepSeek V4.1 Flash | Fastest prefill (TTFT 559 ms at 16K, 2.28 s at 64K, about 26–31% below the others); highest throughput at 16K/c8 and 64K/c8 (1.11× and 1.28× MiMo); smallest KV per token (1.85 KiB per GPU) | Slowest decode (9.0 ms/token); largest weights (36.3 GiB per GPU) and smallest KV pool | Prefill skip runs 21 of 40 layers on prompt tokens (source + log + trace, high). Shared, compressed KV (config + snapshot, medium). Decode pays projections, mHC and the widest experts (trace, medium) | Skip applies only to steps ≥ 768 tokens; Engram tables sit in host memory |
| DeepSeek V4 Flash 0731 | Flat decode with context; smaller weights than V4.1 | Never the fastest; slowest at 16K/c8 and at both 64K points, in the middle at 1K and 16K/c1; indexer cost grows to match attention at 64K; KV blocks of 256 tokens waste memory on short requests | Sparse attention keeps the core cheap but the search grows (trace, medium). Every layer runs on every prompt token (trace, medium) | Slow first block at 1K/c8 kept |

Among these three, no single model is best. The rule that survives the data, and that the two added models in section 8 confirm (fewer softmax-attention layers, smaller or faster experts): **work that is skipped is the only reliably cheap work.** MiMo skips the selection machinery, V4.1 skips half the layers during prefill, and each wins where its skipped work would have dominated.

## 10. What is still open

- Measured HBM bytes (no counters), prefix capacity and reuse (deferred by instruction), per-expert routing statistics (no instrumentation), maximum concurrency, 128K behavior, and any quality comparison.
- All trace statements rest on one capture per point.
- The exact continuation commands and the list of pending, failed and unsupported items are in [handoff.md](handoff.md).
