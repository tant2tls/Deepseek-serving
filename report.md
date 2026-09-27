# DeepSeek V4.1 Flash — speculation-off serving baseline

Study `v41-vs-0731`, phase 1 of [target.md](target.md): **V4.1 Flash only, speculation off**. Measured 2026-09-27 on one reserved 8×H100 node. The matched DeepSeek V4 Flash **0731** baseline and all DSpark arms are still **pending**; this report makes no V4.1/0731 comparison. The historical preview (`deepseek-ai/DeepSeek-V4-Flash`, revision unverified) is not a 0731 baseline and is not compared here.

All 75 planned performance points (25 points × 3 repeats) completed and passed validation. An added prefill/decode phase (section 4: 6 control runs, 27 isolated runs, and 2 diagnostic traces) also passed. Two harness failures from an aborted launch attempt are kept in the [failure table](#failures-and-deviations). Machine-readable aggregates: [reports/v41-vs-0731/v41_off_summary.csv](reports/v41-vs-0731/v41_off_summary.csv); full tables including telemetry: [reports/v41-vs-0731/v41_off_tables.md](reports/v41-vs-0731/v41_off_tables.md).

## Setup (identical for every point)

| Item | Value |
| --- | --- |
| Hardware | 8× NVIDIA H100 80GB HBM3, all-pairs NV18 NVLink, 2 NUMA nodes (GPU0–3 / GPU4–7), 208 CPUs, 1.7 TiB RAM; driver 580.105.08; SM max clock 1980 MHz, 700 W limit |
| Runtime | vLLM `0.30.1rc1.dev223+g44af287eb`, torch `2.13.0+cu132` (CUDA 13.2), Triton `3.7.1`, env `~/vllm` |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277` (served ID verified before every sweep) |
| Precision | FP8 attention/dense (block 32×32, UE8M0 scales), FP4 experts; KV cache `fp8_ds_mla` (resolved), block size 64 |
| Parallelism | TP 8 + expert parallel |
| Limits | `max_model_len` 262,144; `max_num_seqs` 64; `max_num_batched_tokens` 8,192 (vLLM default, chunked prefill); GPU mem util 0.90 |
| KV capacity | 270,198 blocks = 9,179,728 tokens (identical in both launches) |
| Speculation | off (`speculative_config=None` in engine log) |
| Requests | `vllm bench serve`, `openai-chat`, `/v1/chat/completions`, unlimited arrival rate with client concurrency cap, `ignore_eos`, 256 output tokens, `temperature=0`, `chat_template_kwargs.thinking=false` |
| Launches | (1) prefix caching **off** → concurrency, context, prefix cache-off; (2) prefix caching **on** → prefix cold/prewarmed; (3) prefix caching off + idle profiler → control, isolated c1 sweep, traces (section 4). |

Launch script: [bench/serve.sh](bench/serve.sh). Harness: [bench/run_matrix.py](bench/run_matrix.py). Declared plan (written before collection): [reports/v41-vs-0731/plan.json](reports/v41-vs-0731/plan.json). Environment inventory: [reports/v41-vs-0731/environment.txt](reports/v41-vs-0731/environment.txt). Per-run evidence (manifests, per-request `bench.json`, summaries, telemetry, server/harness logs): [reports/v41-vs-0731/data/v41/](reports/v41-vs-0731/data/v41/), with what was changed or excluded recorded in its `CURATION.json`.

**Method notes.** Each point ran 3 repeats in alternating order (ascending, then descending). Warmup used disjoint seeds. Measured prompts differ per repeat. Every run has its own immutable directory with raw `bench.json` (per-request TTFT/ITL samples), `manifest.json`, `/metrics` scrapes before and after, 1 Hz scheduler/KV polling, and 1 Hz `nvidia-smi` samples. A point is valid only if all requests completed, none failed, output tokens equal n×256, and server-counted prompt tokens are at least 98% of the target. Values below are **mean ± sample std across 3 repeats**. The percentiles (p50/p95) are computed within each repeat over its requests. Latencies are in ms.

## 1. Concurrency scaling (16,384 in / 256 out, prefix cache off)

| Conc. | Requests/rep | Output tok/s | Req/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | ITL p95 | E2E p50 | E2E p95 | Peak running / waiting | Peak KV | GPU-s / out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 88.4 ± 0.0 | 0.345 | 882 ± 2 | 896 ± 4 | 7.9 ± 0.0 | 7.9 | 8.1 | 2,893 | 2,906 | 1 / 0 | 1.4% | 0.090 |
| 2 | 16 | 138.1 ± 0.1 | 0.540 | 1,477 ± 4 | 1,691 ± 3 | 8.7 ± 0.0 | 9.5 | 8.2 | 3,703 | 3,732 | 2 / 0 | 1.5% | 0.058 |
| 4 | 16 | 190.1 ± 0.1 | 0.742 | 2,465 ± 2 | 3,269 ± 8 | 11.5 ± 0.0 | 16.1 | 9.0 | 5,379 | 5,418 | 4 / 2 | 1.7% | 0.042 |
| 8 | 32 | 231.4 ± 0.8 | 0.904 | 4,058 ± 14 | 5,647 ± 17 | 18.7 ± 0.0 | 29.5 | 19.0 | 8,836 | 9,630 | 8 / 6 | 2.2% | 0.035 |
| 16 | 64 | 260.3 ± 0.6 | 1.017 | 4,385 ± 284 | 10,812 ± 33 | 44.2 ± 0.8 | 55.3 | 401.8 | 15,702 | 21,953 | 16 / 13 | 3.0% | 0.031 |
| 32 | 128 | 280.0 ± 0.3 | 1.094 | 3,850 ± 205 | 21,151 ± 16 | 98.8 ± 0.9 | 105.3 | 419.2 | 29,182 | 46,234 | 32 / 28 | 4.7% | 0.029 |
| 48 | 192 | 287.0 ± 1.0 | 1.121 | 3,604 ± 223 | 31,644 ± 86 | 152.8 ± 1.0 | 155.7 | 426.6 | 42,627 | 70,670 | 48 / 44 | 6.5% | 0.028 |
| 64 | 256 | 292.3 ± 0.4 | 1.142 | 3,308 ± 6 | 42,164 ± 154 | 205.4 ± 0.5 | 206.5 | 430.1 | 55,795 | 94,418 | 64 / 59 | 8.3% | 0.027 |

Slide-13 question (how much throughput concurrency buys, at what latency cost): going from c1 to c8 gives 2.6× output throughput for 4.6× TTFT p50 and 2.4× TPOT. Going from c8 to c64 adds only **+26%** throughput, while TPOT p50 rises **11×** (18.7 → 205 ms) and TTFT p95 rises **7.5×** (5.6 → 42 s). Engine batch equals client concurrency: peak running requests reached the cap at every point. There were no preemptions, and KV use never exceeded 8.3% of the pool.

## 2. Context scaling (256 out, concurrency 8, 16 requests/repeat, prefix cache off)

| Input tokens | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | ITL p50 | ITL p95 | E2E p50 | E2E p95 | Peak KV | Preempt | GPU-s / out tok |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 231.0 ± 0.5 | 4,056 ± 8 | 6,215 ± 11 | 18.7 ± 0.1 | 29.4 | 9.4 | 18.8 | 8,809 | 10,158 | 2.2% | 0 | 0.035 |
| 65,536 | 68.5 ± 0.1 | 14,491 ± 154 | 25,206 ± 20 | 59.9 ± 0.5 | 99.9 | 9.6 | 440.5 | 29,755 | 40,046 | 4.5% | 0 | 0.117 |
| 131,072 | 33.3 ± 0.0 | 30,296 ± 515 | 54,020 ± 80 | 121.6 ± 2.2 | 206.5 | 9.8 | 504.0 | 61,267 | 84,670 | 7.5% | 0 | 0.241 |
| 260,000 | 14.8 ± 0.0 | 68,344 ± 298 | 123,490 ± 165 | 273.0 ± 1.8 | 467.2 | 422.2 | 676.5 | 137,700 | 193,082 | 13.5% | 0 | 0.541 |

All four context points, including 260,000 input + 256 output inside the 262,144 limit, completed without failures or preemptions. The 16K point matches the concurrency sweep's c8 point (231.0 vs 231.4 tok/s) even though request counts differ (16 vs 32). This batch/context agreement in one controlled session is what the preview reference flagged as unresolved for the old runs.

## 3. Prefix reuse (65,536 shared prefix + 2,048 suffix, 256 out, c8, 64 requests/repeat)

The three cache states use **identical prompt files and request order** within each repeat, with new prompts for each repeat. *Cache-off* ran on launch 1. *Cold* ran after `POST /reset_prefix_cache`. *Prewarmed* ran after a reset followed by one request per prefix, each with a disjoint suffix.

| Distinct prefixes | State | Output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | E2E p50 | E2E p95 | Token hit rate | vs cache-off (tok/s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cache off | 66.6 ± 0.2 | 14,843 ± 18 | 17,853 ± 144 | 62.4 | 30,759 | 32,913 | — | 1.00× |
| 1 | cold | 475.3 ± 0.8 | 867 ± 2 | 4,663 ± 53 | 11.7 | 3,846 | 7,044 | 95.5% | 7.14× |
| 1 | prewarmed | 524.4 ± 0.3 | 791 ± 73 | 1,331 ± 36 | 12.0 | 3,838 | 4,418 | 97.0% | 7.87× |
| 4 | cache off | 66.3 ± 0.1 | 14,900 ± 21 | 17,783 ± 175 | 62.7 | 30,857 | 32,884 | — | 1.00× |
| 4 | cold | 369.5 ± 1.3 | 835 ± 73 | 12,400 ± 1,979 | 12.0 | 3,862 | 15,948 | 90.9% | 5.57× |
| 4 | prewarmed | 520.7 ± 0.4 | 773 ± 78 | 1,328 ± 31 | 12.1 | 3,865 | 4,334 | 97.0% | 7.85× |
| 16 | cache off | 66.4 ± 0.1 | 14,860 ± 47 | 17,863 ± 97 | 62.7 | 30,848 | 32,526 | — | 1.00× |
| 16 | cold | 194.2 ± 0.5 | 2,015 ± 1,919 | 15,599 ± 1,568 | 12.7 | 7,503 | 24,891 | 72.7% | 2.92× |
| 16 | prewarmed | 520.6 ± 0.7 | 765 ± 91 | 1,332 ± 17 | 12.2 | 3,871 | 4,387 | 97.0% | 7.84× |

Hit counts match the design exactly in every repeat. Cold runs hit (64 − N) × 65,536 tokens; prewarmed runs hit all 64 × 65,536. Cache-off runs recorded zero prefix queries. The cold 16-prefix TTFT p50 is order-sensitive: per repeat it was 0.87, 0.94, and 4.23 s. Its mean is reported but should not be read as a stable value.

## 4. Prefill and decode in isolation, and the CED check

This phase was added after the first three workloads, in response to the V4.1 paper ([arXiv:2609.19969](https://arxiv.org/abs/2609.19969), Section 2.2). The paper says the **Causal Encoder-Decoder (CED)** design lets prompt tokens pass through only the bottom L/2 = 20 layers. The upper 20 "decoder" layers take their global KV from the layer-20 hidden state, and only the last n_win = 128 prompt tokens are replayed through them for sliding-window KV ("Decoder SWA Bounded Replay"). This gives "8B activated parameters per token during prefill and 16B during decode" and "nearly halves prefill computation."

It ran on one extra launch (`off-profidle`): the same settings as `off`, plus a torch profiler that is configured but idle except between `/start_profile` and `/stop_profile`. **Control:** repeating c1 and c64 at 16K on this launch gave 88.7 ± 0.1 and 292.2 ± 0.3 output tok/s, against 88.4 ± 0.0 and 292.3 ± 0.4 on the baseline launch. The idle profiler setting has no measurable effect, so these points are comparable to sections 1–3.

### 4a. Isolated sweep: one request at a time (c1, 4 requests × 3 repeats, 256 output tokens)

At c1 there is no queueing, so TTFT ≈ prefill plus the first step, and TPOT is pure decode at that context.

| Input tokens | TTFT p50 (ms) | Prefill tok/s (input ÷ TTFT) | µs per prompt token | TPOT p50 (ms) | Output tok/s |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 148 ± 1 | 6,960 | 143.7 | 7.84 | 118.6 |
| 2,048 | 153 ± 1 | 13,389 | 74.7 | 7.84 | 118.5 |
| 4,096 | 245 ± 0 | 16,749 | 59.7 | 7.84 | 113.7 |
| 8,192 | 452 ± 10 | 18,118 | 55.2 | 7.85 | 104.1 |
| 16,384 | 894 ± 15 | 18,335 | 54.5 | 7.84 | 88.4 |
| 32,768 | 1,769 ± 10 | 18,530 | 54.0 | 7.81 | 68.0 |
| 65,536 | 3,618 ± 21 | 18,114 | 55.2 | 7.78 | 45.7 |
| 131,072 | 7,740 ± 8 | 16,934 | 59.1 | 7.70 | 26.4 |
| 260,000 | 17,796 ± 34 | 14,610 | 68.4 | 7.56 | 13.0 |

- **Decode is flat in context.** TPOT is 7.56–7.85 ms from 1K to 260K input. This matches the paper's near-constant decode FLOPs claim (its Figure 2). Decode is not where V4.1's long-context cost lives.
- **Prefill is about 54–55 µs per prompt token (≈18.3K tok/s) from 8K to 64K input.** It rises to 59 µs at 128K and 68 µs at 260K. Below 4K, fixed per-request overhead dominates.
- The same ≈18.5K tok/s ceiling limits the c64 concurrency point (section 1), so it is a property of prefill on this runtime, not of batching.

### 4b. Kernel breakdown of one prefill chunk (diagnostic trace, rank 0, 8,192-token chunk)

Trace manifests, rank-0 kernel summaries, and per-step breakdown CSVs: [reports/v41-vs-0731/data/v41/profiles/](reports/v41-vs-0731/data/v41/profiles/). The full 8-rank torch traces (~73 MB) are kept locally under `results/`, not published. Breakdown script: [bench/trace_breakdown.py](bench/trace_breakdown.py). The unit is summed GPU kernel time inside the main-stream step span. The categories do not overlap each other, and the denominator is the step's kernel sum.

| Component | 16K run, chunk 1 | 16K run, chunk 2 | 64K run, chunk 1 | 64K run, chunk 8 (57K context) | Share (16K chunk 2) |
|---|---:|---:|---:|---:|---:|
| Step wall span | 416.5 | 398.8 | 424.7 | 434.2 | — |
| Kernel sum | 389.8 | 399.2 | 392.2 | 434.4 | 100% |
| All-reduce (TP) | 106.6 | 107.7 | 108.4 | 106.7 | 27.0% |
| Dense GEMM (projections, Marlin FP8) | 90.3 | 91.5 | 90.3 | 91.9 | 22.9% |
| MoE expert GEMM (Marlin W4A16) | 73.2 | 73.0 | 73.3 | 73.4 | 18.3% |
| Core sparse attention | 48.1 | 49.5 | 48.2 | 50.1 | 12.4% |
| mHC / residual / norm | 32.1 | 32.1 | 32.3 | 32.1 | 8.0% |
| Other elementwise | 13.8 | 15.1 | — | — | 3.8% |
| QKV norm / RoPE / KV insert | 9.4 | 9.6 | 9.4 | 9.5 | 2.4% |
| MoE routing / combine | 7.8 | 7.9 | — | — | 2.0% |
| Indexer / top-k | 2.2 | 6.1 | 2.2 | 28.4 | 1.5% |
| Engram | 3.5 | 4.0 | — | — | 1.0% |

All values are in ms; "—" means that component was not extracted for the 64K chunks. Over the 64K prompt's eight chunks, only the indexer grows with context (2.2 → 28.4 ms). Core attention stays about 48–50 ms, consistent with fixed top-k sparse attention. Everything else is per-token work repeated in every layer. The final short chunk of each prompt (4 tokens after 2×8,192 or 8×8,192) still runs all 40 layers.

**Decode steps** are captured by CUDA graphs. The per-step annotation spans only 84 launch-level kernels (about 0.46 ms), not the about 7.8 ms step. **The decode component breakdown is unavailable** from this trace and needs graph-aware attribution (for example, the profiler's `capture_torch_profiler`, or an eager diagnostic arm).

### 4c. Is CED active in this runtime? No.

In both prefill chunks of the 16K trace, **all 40 layers ran the full 8,192-token chunk**:

- 80 MoE expert GEMM calls per chunk (40 layers × w13/w2), with identical launch grids in every layer. The mean per-layer MoE time was 1.83 ms for layers 1–20 and 1.83 ms for layers 21–40.
- 40 sparse prefill attention calls per chunk, at 1.22–1.25 ms in every layer from 2 to 40, with no drop in the upper half.
- The runtime source agrees. `models/deepseek_v41/nvidia/model.py` loops every scheduled token through all layers. The cross-layer KV/indexer reuse (CSA2) is implemented: only layers 2/8/14/20 own compressed KV, and 24/28/32/36 reindex. `swa_bounded_replay` is active (V2 model runner), but in this build it only rebuilds sliding-window KV after a prefix hit. No path stops prompt tokens at layer 20.

**Conclusion:** vLLM `0.30.1rc1.dev223` runs V4.1 prefill through all 40 layers, so it activates the decode-size parameter set for every prompt token rather than the paper's 8B prefill path. Every prefill number in this report is **V4.1 on this runtime**, not the CED design point.

**Projection, not a measurement:** the upper 20 layers account for about half of each chunk's per-layer work (the MoE and attention times above are symmetric between halves). A CED-aware runtime could therefore approach the paper's "nearly halves prefill computation", roughly 1.8–2× prompt throughput at ≥ 8K input. At 16K/256, concurrency throughput is prefill-bound (finding 1), so it could rise by a similar factor. Only a CED-capable runtime can confirm this.

### 4d. All-reduce runs far below NVLink bandwidth (hypothesis)

Every prefill all-reduce moves a `[8192, 5120]` BF16 tensor (83.9 MB), 2 per layer, and averages 1.32 ms. That is about 63 GB/s algorithm bandwidth. NCCL 2.29.7 chose `AllReduce_Sum_bf16_RING_LL` (the low-latency protocol) for these large messages. vLLM does not set `NCCL_PROTO`/`NCCL_ALGO` in this configuration. The dispatch order for the TP group is FlashInfer → custom → symm-mem → PyNCCL, and the large prefill messages fell through to PyNCCL. At 27% of prefill kernel time, this is the largest single component. **Controlled follow-up:** run `nccl-tests` all-reduce at 84 MB on this node, then a separate tuning arm with `NCCL_PROTO=Simple` (or LL128) or a symm-mem/custom all-reduce size limit that covers 84 MB, measuring prefill tok/s and the c64 point against this baseline.

## Findings (V4.1 only, speculation off)

1. **At 16K input, uncached prefill caps whole-serving throughput.** Server-counted prompt processing saturates at about **18.7K prompt tok/s** (c64: 4.20M prompt tokens in 224 s). Output throughput follows it arithmetically: 18.7K × 256 / 16,388 ≈ 292 tok/s, which is exactly the measured c64 value. Uncached prompt rates at c8 cluster at 14.8–17.5K tok/s across 16K–260K contexts and the cache-off prefix runs. *Status: measured endpoint relationship. Section 4 attributes the prefill cost: all 40 layers run on every prompt token, and all-reduce 27%, dense GEMM 23%, MoE GEMM 18%, and attention 12% of chunk kernel time.*
2. **Decode per step is cheap and nearly flat in context, but prefill chunks stall it.** With no prefill in flight, ITL p50 stays at 9.4–9.8 ms from 16K to 131K context at c8, and 7.9 ms at c1. ITL p95 jumps to about 400–500 ms whenever new prompts are prefilling (c ≥ 16, or ≥ 64K context). At 260K, prefill occupies most of the run, so even ITL p50 is 422 ms. This pattern is consistent with decode steps sharing batches with 8,192-token prefill chunks. The chunk-size link is a **hypothesis**: the controlled test is to rerun c16/c64 at 16K with `max_num_batched_tokens` 4,096/16,384 as a separate tuning arm, plus a timeline trace.
3. **Memory is not the constraint in these workloads.** Peak KV use was at most 13.5% of the 9.18M-token pool, with zero preemptions anywhere. The vLLM-reported `kv_cache_max_concurrency` at 262,144 tokens is 35 sequences. `max_num_seqs=64` was the binding limit, not KV. Peak GPU memory (about 77–80 GB per GPU) reflects the 0.90 memory budget, not useful occupancy.
4. **Prefix reuse is V4.1's largest lever for long shared prompts.** A prewarmed 64K prefix gives **7.8×** output throughput and cuts TTFT p50 from 14.9 s to about 0.77 s, with cached prompt tokens processed at about 138K tok/s. Prewarmed results do not depend on prefix count (1, 4, or 16), while cold performance drops with it (475 → 370 → 194 tok/s) because each distinct prefix pays a full 64K fill. Prewarming matters most when many distinct prefixes are in use.
5. **V4.1's CED prefill saving is not realized on this runtime.** Traces show all 40 layers processing every prompt token (section 4c). The measured prefill cost is about 54.5 µs per prompt token at 8–64K. Decode is context-independent at about 7.8 ms/token (c1). The largest prefill component is TP all-reduce (27%), which runs at about 63 GB/s on the NCCL ring-LL protocol (section 4d, hypothesis). *Status: CED absence confirmed by trace plus source; the size of the missed gain is a projection; the all-reduce inefficiency is a hypothesis with a declared test.*
6. **Operating points (this hardware, speculation off).** For latency (TPOT < 20 ms, TTFT p95 < 6 s at 16K), stay at **≤ 8 concurrent requests** (231 tok/s, 0.035 GPU-s/token). For throughput with bounded latency, **c16** gives 89% of the c64 throughput at about 1/5 of the TPOT. c32–c64 buys only +8–12% more throughput for 2–4.6× worse TPOT and 2–4× worse TTFT p95. Long-context (≥ 64K) serving without prefix reuse is dominated by TTFT, about 0.22–0.26 s of TTFT p50 per 1K input tokens at c8.

Cost per token, if needed: GPU-s/output token × (GPU-hour price / 3600). No price is assumed here. All workloads are synthetic random-token prompts with forced 256-token outputs. They measure serving speed, not answer quality.

**Correctness smoke test** (same server, thinking off, greedy): 4/4 fixed prompts answered correctly (arithmetic, fact, code, word problem), all with `finish_reason=stop` and no reasoning text. This is a sanity check, not the real-text task set the plan requires. Saved at [reports/v41-vs-0731/data/v41/_quality_smoke/responses.json](reports/v41-vs-0731/data/v41/_quality_smoke/responses.json).

## Failures and deviations

| Item | What happened | Effect |
| --- | --- | --- |
| Earlier launch crash (before this study) | Triton `FileNotFoundError` in `~/.triton/cache`: 8 TP ranks compiling concurrently on the gocryptfs FUSE mount at `/root` | Fixed by `TRITON_CACHE_DIR=/tmp/triton_cache` (local overlay) in `run.sh` and `bench/serve.sh` |
| `prefix/off-cache-off/p65536_s2048_n1_c8/repeat-1` | Invalid: `ImportError: Please install vllm[bench]` (missing `pandas` for the custom dataset); no request was sent | Kept on disk; `pandas` installed; valid rerun is `repeat-1-rerun1` |
| `prefix/off-cache-off/p65536_s2048_n4_c8/repeat-1` | Aborted by the operator while stopping the failing loop; empty log, no summary | Kept on disk; valid rerun is `repeat-1-rerun1` |
| Prefix prompts | Harness-built JSONL (random tokens, decode/re-encode to exact length) instead of vLLM `prefix_repetition`, so cold/prewarmed/cache-off could share identical prompts and order | Deliberate; files in `results/v41-vs-0731/_prompts/` |
| `max_num_seqs` | 64 (historical preview used 256); covers the planned c ≤ 64 | Must be matched for 0731 |
| Telemetry not collected | Per-request queue delay split, CPU scheduling gaps, and all component/kernel timings | Unavailable, not zero; requires the profiling phase |

## Pending (study is incomplete)

- **DeepSeek V4 Flash 0731**, the same three workloads with identical settings, harness, and prompts. Pin its snapshot in `V4_0731_REVISION`, and alternate model order for any reruns. Two launches, same as V4.1.
- **DSpark fixed/adaptive** for both checkpoints (c1/4/16/64 at 16K/256, width 5, 3 repeats), per [docs/speculative-decoding.md](docs/speculative-decoding.md). One launch per arm.
- **Component profiling, remaining**: prefill traces at c1 for 16K and 64K are done (section 4). Still pending: decode-step attribution under CUDA graphs; traces at 16K c16/c64, 260K c8, and cache-off vs prewarmed; per-rank communication/imbalance analysis (MoE per-layer time varies 0.5–3.8 ms across layers on rank 0); and hardware counters for the dominant GEMMs.
- **All-reduce tuning arm** (section 4d) and a **CED-capable runtime**, if one exists, as separately labeled arms.
- Real-text quality task set, and the `max_num_batched_tokens` tuning arm (separate from baselines).

## Reproduce

```bash
# launch 1 (prefix cache off), in tmux
bash bench/serve.sh v41 off results/v41-vs-0731/v41/_server/serve-off-<ts>.log
python bench/run_matrix.py --model v41 --config off \
    --workloads concurrency,context,prefix --prefix-states cache-off
# launch 2 (prefix cache on)
bash bench/serve.sh v41 off-prefix results/v41-vs-0731/v41/_server/serve-off-prefix-<ts>.log
python bench/run_matrix.py --model v41 --config off --workloads prefix --prefix-states cold,prewarmed
# launch 3 (idle profiler; section 4)
PROFILE_DIR=$PWD/results/v41-vs-0731/v41/profiles bash bench/serve.sh v41 off-profidle results/v41-vs-0731/v41/_server/serve-off-profidle-<ts>.log
python bench/run_matrix.py --model v41 --config off-profidle --workloads concurrency,isolated --conc-list 1,64
python bench/profile_trace.py --label prefill16k_decode8 --isl 16384 --max-tokens 8   # move files into profiles/<label>/
python bench/trace_breakdown.py results/v41-vs-0731/v41/profiles/<label>/dp0_pp0_tp0_*rank0*.json.gz
# aggregate
python bench/summarize.py v41-vs-0731 v41 --csv reports/v41-vs-0731/v41_off_summary.csv
```

`HF_TOKEN` must come from the environment. The raw local tree is `results/` (git-ignored, 454 MB, mostly prefix prompt JSONL). The published subset is produced by `python bench/curate.py v41-vs-0731 v41` and audited on a clean checkout with `python tools/audit_references.py`.
