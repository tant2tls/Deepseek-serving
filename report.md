# DeepSeek V4.1 Flash vs V4 Flash 0731 — serving, prefill components, and DSpark

Study `v41-vs-0731` ([target.md](target.md)). One reserved 8×H100 node, one pinned vLLM build, identical server settings, harness, prompts, and seeds for both checkpoints. Measured 2026-09-27/28.

| Checkpoint | Revision | Role |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | V4.1 |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | V4 official release ("0731") |

The historical `deepseek-ai/DeepSeek-V4-Flash` results (preview, revision unverified) are **not** a 0731 baseline and are not compared here.

**Scope and validity.** Speculation off: 144 valid runs per model (concurrency c1–c64, context 16K–260K, prefix cache-off/cold/prewarmed, isolated c1 prefill/decode sweep), plus prefill traces. DSpark fixed and adaptive (width 5): 48 of 48 runs valid (2 models × 2 arms × c1/4/16/64 × 3 repeats). Plus a sequential prefix-reuse check on both models. All failures are kept and listed in [Failures and deviations](#failures-and-deviations). Values are **mean ± sample SD over 3 repeats**. Ratios are **V4.1 / 0731** (> 1 is better for throughput, worse for latency) unless a row says otherwise. A difference is "inconclusive" when |Δmean| ≤ 2·√(SD₁² + SD₂²).

Evidence labels: **measured** (endpoint numbers), **trace+source** (profiler kernels plus runtime code), **arithmetic** (follows from measured numbers), **hypothesis** (a declared test is still needed).

## Key insights

1. **On this runtime, 0731 is faster than V4.1 on nearly every uncached workload. V4.1's prefill is the reason.** V4.1/0731 output throughput is 0.74–0.90 across concurrency, 0.78–0.87 across context, and about 0.79 for cache-off prefix. Uncached prefill costs V4.1 **54.5 µs/token vs 41.6 µs for 0731** at 16K (c1). At 16K/256 both models are prefill-bound, so this gap flows directly into throughput. *Measured.*
2. **V4.1's prefill deficit is kernel-path and communication cost, not model FLOPs.** Per 8,192-token chunk (rank 0), V4.1 needs 399 ms of kernel time vs 293 ms for 0731, even though V4.1 has fewer layers (40 vs 43). The largest gaps are dense GEMM (91.5 vs 39.7 ms; V4.1 runs Marlin FP8, 0731 runs DeepGEMM/FlashInfer FP8 block-scale), TP all-reduce (107.7 vs 79.4 ms; ring-LL vs symm-mem multimem), and MoE (73.0 vs 53.3 ms). *Trace+source.* This vLLM build also does not implement V4.1's CED prefill skip: all 40 layers process every prompt token. The paper's 8B-prefill design point is therefore not what was measured.
3. **V4.1 wins everything with prefix caching on: 1.05–1.33× cold and 1.07–2.24× prewarmed.** The mechanism is measured directly. On V4.1 a new 64K prefix becomes reusable **on the 2nd request**. 0731 needs a **3rd**, because its SWA prefix checkpoint is pinned only when a second request computes the shared junction. As a result, "prewarm with one request per prefix" does not warm 0731. *Measured (sequential check and exact hit counters) + source.*
4. **DSpark speeds up low-concurrency decode, but gains vanish once prefill dominates.** Relative to each model's own speculation-off run, V4.1 gets 1.55× at c1 and 1.11× at c4, and 0731 gets 1.87× and 1.30×. At c16–c64 both land at 0.98–1.04×. *Measured.*
5. **0731's drafter accepts more:** 45–49% of proposed tokens vs V4.1's 29–34% (3.2–3.5 vs 2.4–2.7 tokens per verification round) on random-token prompts. V4.1 therefore stays at 0.70–0.85× of 0731 with DSpark on. *Measured; acceptance is workload-specific.*
6. **At c64, DSpark moves latency from TPOT to TTFT without changing end-to-end latency.** TPOT falls by 22–36% (mean), while TTFT p50 rises from about 3 s to 17–18 s. Mean E2E is unchanged (V4.1 55.9 → 56.0 s, 0731 41.6 → 40.4 s) and equals concurrency ÷ request rate. This is Little's law in a closed loop: when throughput is fixed by prefill, finishing decode sooner only adds time spent waiting to prefill. *Arithmetic on measured values.*
7. **Adaptive verification is active but performs about the same as fixed (≤ 4%).** It costs memory: an extra variable-length CUDA-graph route takes KV capacity from 7.38M to 6.25M tokens on V4.1 and from 1.56M to 1.38M on 0731. *Measured (server logs and endpoint numbers).*
8. **Capacity differs by 5.4×.** At 0.90 memory utilization, V4.1 holds 9.18M KV tokens vs 1.71M for 0731. No workload here came close to either limit: peak KV use was ≤ 13.5% on V4.1 and ≤ 29.7% on 0731, with zero preemptions. *Measured.*

## Setup (identical for both models)

| Item | Value |
| --- | --- |
| Hardware | 8× NVIDIA H100 80GB HBM3, all-pairs NV18 NVLink, 2 NUMA nodes, driver 580.105.08 |
| Runtime | vLLM `0.30.1rc1.dev223+g44af287eb`, torch `2.13.0+cu132`, Triton `3.7.1` ([environment.txt](reports/v41-vs-0731/environment.txt)) |
| Server | TP 8 + expert parallel, `--language-model-only`, GPU memory utilization 0.90, `max_model_len` 262,144, `max_num_seqs` 64, default `max_num_batched_tokens` 8,192 (chunked prefill), default KV dtype |
| Tokenizer / parsers | `deepseek_v41` vs `deepseek_v4`. `chat_template_kwargs.thinking=false` maps to chat mode in both |
| Requests | `vllm bench serve`, `openai-chat`, unlimited arrival rate with a client concurrency cap, `ignore_eos`, 256 output tokens, `temperature=0`, thinking off, random-token prompts (same token counts; prefix prompt files are shared text and tokenize to the same 67,584 tokens in both) |
| KV capacity (speculation off) | V4.1 9,179,728 tokens; 0731 1,711,998 tokens |
| Launches per model | `off` or `off-profidle` (prefix caching off; idle torch profiler, shown to have no effect: V4.1 c1 88.4 vs 88.7, c64 292.3 vs 292.2 tok/s) → concurrency, context, prefix cache-off, isolated, traces · `off-prefix` → prefix cold/prewarmed · one launch per DSpark arm |

Plans were declared before collection: [plan.json](reports/v41-vs-0731/plan.json), [plan_0731_addendum.json](reports/v41-vs-0731/plan_0731_addendum.json), and [plan_dspark.json](reports/v41-vs-0731/plan_dspark.json). A run is valid only if every request completed, output equals n×256 tokens, and server-counted prompt tokens are at least 98% of the target. Warmups use disjoint seeds, and repeat order alternates. Per-run evidence (manifests, per-request `bench.json`, summaries, Prometheus scrapes, 1 Hz scheduler/KV/GPU polling, and server/harness logs) is in [reports/v41-vs-0731/data/](https://github.com/tant2tls/Deepseek-serving/blob/be9ee6abfc251d8145c44a375cd2de8907228513/reports/v41-vs-0731/data). Each model's `CURATION.json` records what was gzipped, sanitized, or excluded.

Full tables: [comparison_tables.md](reports/v41-vs-0731/comparison_tables.md) (every metric, V4.1 vs 0731) · [v41_off_tables.md](reports/v41-vs-0731/v41_off_tables.md) and [v4-0731_off_tables.md](reports/v41-vs-0731/v4-0731_off_tables.md) (per model, with telemetry) · [dspark_tables.md](reports/v41-vs-0731/dspark_tables.md) (DSpark vs own baseline, acceptance). CSV versions sit alongside them.

## 1. Speculation off: V4.1 vs 0731

### 1a. Concurrency (16,384 in / 256 out)

| Conc. | V4.1 tok/s | 0731 tok/s | Ratio | TTFT p50 V4.1 / 0731 (ms) | TPOT p50 V4.1 / 0731 (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 88.4 ± 0.0 | 98.3 ± 0.0 | 0.90 | 882 / 681 | 7.9 / 7.5 |
| 2 | 138.1 ± 0.1 | 159.3 ± 0.1 | 0.87 | 1,477 / 1,125 | 8.7 / 8.2 |
| 4 | 190.1 ± 0.1 | 232.5 ± 0.0 | 0.82 | 2,465 / 1,864 | 11.5 / 10.0 |
| 8 | 231.4 ± 0.8 | 295.7 ± 0.4 | 0.78 | 4,059 / 3,052 | 18.7 / 15.2 |
| 16 | 260.3 ± 0.6 | 329.2 ± 15.7 | 0.79 | 4,385 / 3,537 | 44.2 / 33.2 |
| 32 | 280.0 ± 0.3 | 356.1 ± 29.0 | 0.79 | 3,850 / 3,264 | 98.8 / 75.7 |
| 48 | 287.0 ± 1.0 | 377.0 ± 14.2 | 0.76 | 3,605 / 3,284 (inconclusive) | 152.8 / 111.6 |
| 64 | 292.3 ± 0.4 | 392.4 ± 0.2 | 0.74 | 3,308 / 3,191 (inconclusive) | 205.4 / 150.0 |

At c64 the server-counted prompt rate is about 18.7K tok/s for V4.1 and about 25.1K for 0731. Output throughput equals prompt rate × 256 / 16,388 for both models, which means **concurrency throughput at 16K is prefill-bound** and the ratio widens as prefill dominates (0.90 at c1 → 0.74 at c64). The larger 0731 SDs at c16–c48 come from slower first repeats at new shapes (5–9% lower). These runs are kept, not rerun.

### 1b. Context (c8, 16 requests/repeat)

| Input | V4.1 tok/s | 0731 tok/s | Ratio | TTFT p50 ratio | TPOT p50 ratio |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 231.0 ± 0.5 | 294.8 ± 0.6 | 0.78 | 1.33 | 1.24 |
| 65,536 | 68.5 ± 0.1 | 85.3 ± 4.2 | 0.80 | 1.22 | 1.23 (inconclusive) |
| 131,072 | 33.3 ± 0.0 | 39.8 ± 1.8 | 0.84 | 1.18 | 1.23 |
| 260,000 | 14.8 ± 0.0 | 17.0 ± 0.1 | 0.87 | 1.09 | 1.21 |

The gap narrows with context: V4.1's indexer is cheaper at long context (section 3), and a larger share of time goes to attention and indexing, where the models are closer.

### 1c. Isolated prefill and decode (c1, 4 requests × 3 repeats)

| Input | TTFT p50 V4.1 / 0731 (ms) | Prefill µs/token V4.1 / 0731 | TPOT p50 V4.1 / 0731 (ms) | Output tok/s ratio |
|---:|---:|---:|---:|---:|
| 1,024 | **148 / 185** | 144 / 180 | 7.84 / 7.5 | 0.97 |
| 2,048 | **153 / 196** | 75 / 96 | 7.84 / 7.5 | 0.98 |
| 4,096 | 245 / 210 | 60 / 51 | 7.84 / 7.5 | 0.95 |
| 8,192 | 452 / 344 | 55.2 / 42.0 | 7.85 / 7.5 | 0.92 |
| 16,384 | 894 / 682 | 54.5 / 41.6 | 7.84 / 7.5 | 0.90 |
| 32,768 | 1,769 / 1,369 | 54.0 / 41.8 | 7.81 / 7.5 | 0.88 |
| 65,536 | 3,618 / 2,855 | 55.2 / 43.6 | 7.78 / 7.5 | 0.85 |
| 131,072 | 7,741 / 6,359 | 59.1 / 48.5 | 7.70 / 7.5 | 0.85 |
| 260,000 | 17,796 / 15,538 | 68.4 / 59.8 | 7.56 / 7.4 | 0.88 |

- **Decode is flat in context for both models**, at 7.4–7.85 ms/token from 1K to 260K. 0731 is about 4% faster per step.
- **V4.1 wins TTFT only at ≤ 2K input.** From 4K upward, 0731 prefills 1.16–1.31× faster.

### 1d. Prefix reuse (65,536 shared prefix + 2,048 suffix, c8, 64 requests/repeat)

| Prefixes | State | V4.1 tok/s | 0731 tok/s | Ratio | TTFT p50 V4.1 / 0731 (ms) | Prefix hits V4.1 / 0731 (× 65,536) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | cache off | 66.6 ± 0.2 | 84.3 ± 0.6 | 0.79 | 14,844 / 12,152 | — |
| 1 | cold | 475.3 ± 0.8 | 453.6 ± 1.1 | **1.05** | 867 / 1,263 | 63 / **62** |
| 1 | prewarmed | 524.4 ± 0.3 | 488.7 ± 1.9 | **1.07** | 791 / 1,265 | 64 / **63** |
| 4 | cache off | 66.3 ± 0.1 | 84.6 ± 0.0 | 0.78 | 14,900 / 12,172 | — |
| 4 | cold | 369.5 ± 1.3 | 320.7 ± 1.2 | **1.15** | 835 / 1,380 | 60 / **56** |
| 4 | prewarmed | 520.7 ± 0.4 | 399.2 ± 4.4 | **1.30** | 773 / 1,314 | 64 / **60** |
| 16 | cache off | 66.4 ± 0.1 | 84.6 ± 0.1 | 0.79 | 14,860 / 12,158 | — |
| 16 | cold | 194.2 ± 0.5 | 146.2 ± 2.8 | **1.33** | 2,015 / 6,941 | 48 / **32** |
| 16 | prewarmed | 520.6 ± 0.7 | 232.9 ± 1.3 | **2.24** | 765 / 2,900 | 64 / **48** |

Hit counts are per repeat (identical in every repeat). V4.1 misses exactly once per prefix when cold, and never when prewarmed. **0731 misses exactly twice per prefix when cold, and once per prefix even after prewarming**, so its prewarmed result degrades with prefix count (489 → 399 → 233 tok/s) while V4.1's stays flat (≈ 521 tok/s).

**Sequential check** ([bench/prefix_reuse_check.py](bench/prefix_reuse_check.py); c1, chat completions, `max_tokens=1`, reset before each sequence; raw JSON in `data/<model>/_prefix_reuse_check/`):

| Request on a new 64K prefix | V4.1 hit / wall | 0731 hit / wall |
| --- | --- | --- |
| 1st | 0 / 3.75 s | 0 / 3.0 s |
| 2nd (new suffix) | **65,536 / 0.44 s** | 0 / 3.0 s |
| 3rd (new suffix) | 65,536 / 0.44 s | **65,536 / 0.52 s** |
| Exact repeat | 67,584 / 0.43 s | 67,584 / 0.36 s |

This holds for every prefix tested, and is not specific to the first prefix after a reset. **Mechanism (source):** vLLM's SWA prefix checkpointing (`shared_prefix_boundary` in `vllm/v1/core/kv_cache_manager.py`) pins the shared-prefix junction only when a second request computes it. V4.1's SWA Bounded Replay keeps SWA state out of prefix caching and rebuilds the window on a hit, so the first computation is already reusable. A cached hit is cheaper on V4.1 (0.44 vs 0.52 s), even though its cold prefill is slower.

## 2. Operating implications

- **Throughput-oriented, uncached, or unique prompts:** 0731 delivers 1.1–1.35× more output tokens per GPU-second on this runtime at every concurrency and context tested.
- **Shared long prefixes (RAG/agents with system prompts):** V4.1 wins. It needs no special warming, it reuses after one computation, and its 5.4× larger KV pool leaves more room for cached prefixes. When running 0731 with prefix caching, **warm each prefix with two requests using different suffixes**, and verify the hit counters before relying on it.
- **Short prompts (≤ 2K), latency-sensitive:** V4.1 has 20–22% lower TTFT. Decode is within 4%.
- **Speculative decoding:** enable DSpark for low-concurrency interactive serving (c ≤ 4): 1.5–1.9× output tok/s and 2–3× lower TPOT at c1. At c ≥ 16 with long prompts it gains nothing and moves latency into TTFT. Use fixed verification: adaptive gives no measured benefit here and costs KV capacity.

## 3. Where prefill time goes (diagnostic traces, rank 0)

One c1 request at 16K (2 chunks + a 4-token tail) and one at 64K (8 chunks + tail) per model, on the `off-profidle` launch. Unit: summed GPU kernel time (ms) inside each 8,192-token prefill step, split by [bench/trace_breakdown.py](bench/trace_breakdown.py) categories. Summaries and per-step CSVs are in `data/<model>/profiles/`; the full 8-rank traces are kept locally only.

| Component | V4.1 16K chunk 2 | 0731 16K chunk 2 | V4.1 − 0731 | V4.1 64K chunk 8 (57K ctx) | 0731 64K chunk 8 |
|---|---:|---:|---:|---:|---:|
| **Kernel sum** | **399.2** | **293.1** | **+106.1** | 434.4 | 338.3 |
| TP all-reduce | 107.7 | 79.4 | +28.3 | 106.7 | 78.7 |
| Dense GEMM | 91.5 | 39.7 | **+51.8** | 91.9 | 39.8 |
| MoE expert GEMM | 73.0 | 53.3 | +19.7 | 73.4 | 52.6 |
| Core sparse attention | 49.5 | 41.9 | +7.6 | 50.1 | 53.2 |
| mHC / residual / norm | 32.1 | 27.9 | +4.2 | 32.1 | 27.9 |
| Indexer / top-k | 6.1 | 8.9 | −2.8 | **28.4** | **40.5** |
| QKV norm / RoPE / KV insert | 9.6 | 10.2 | −0.6 | 9.5 | 10.2 |
| MoE routing / combine | 7.9 | 7.3 | +0.6 | — | 7.3 |
| Other elementwise (+ Engram on V4.1) | 19.1 | 24.5 | −5.4 | — | 28.2 |
| Layers (attention calls per chunk) | 40 | 43 | | 40 | 43 |

- **Dense GEMM explains about half the gap** (2.29 vs 0.92 ms per layer). V4.1's dense FP8 layers (32×32 blocks, UE8M0 scales) resolve to `MarlinFP8ScaledMMLinearKernel`, while 0731's resolve to DeepGEMM/FlashInfer FP8 block-scale GEMM. *Trace+source.* A kernel-path fix is a runtime question, not a model property: *hypothesis*, testable by forcing a block-scale FP8 path for V4.1 if one supports its scale format.
- **All-reduce explains about a quarter.** V4.1's 83.9 MB messages (`[8192, 5120]` BF16) fall through to PyNCCL ring-LL at about 63 GB/s, while 0731's 67 MB messages use symm-mem multimem all-reduce. *Trace; the cause of the size threshold is a hypothesis.* Test: `nccl-tests` at 84 MB, then a V4.1 arm with `NCCL_PROTO=Simple`/LL128 or a symm-mem size limit that covers 84 MB.
- **MoE explains about a fifth** (larger per-token expert work in V4.1).
- **V4.1's indexer scales better** (28.4 vs 40.5 ms at 57K context). This is the only component where V4.1 is cheaper, and it is why the context-scaling ratio improves toward 260K.
- **CED is not active in this runtime (V4.1).** All 40 layers run the full chunk: 80 MoE GEMM calls with identical grids, 40 attention calls, and equal per-layer time in layers 1–20 and 21–40. The source (`models/deepseek_v41/nvidia/model.py`) loops every scheduled token through all layers. CSA2 cross-layer KV reuse and `swa_bounded_replay` are implemented; the early exit of prompt tokens at layer 20 is not. *Trace+source.* **Projection, not measured:** a CED-aware runtime could cut V4.1's per-chunk work by up to about half, which would more than close the 1.36× chunk-time gap to 0731.
- **Decode.** On 0731, decode steps are kernel-attributable: 7.8 ms span and 9.9 ms kernel sum, with overlapping streams, so the sum is not the critical path. Dense GEMM is 35%, all-reduce 15%, and MoE 12–14% of the kernel sum. On V4.1, decode runs inside full CUDA graphs, and the trace shows only about 0.47 ms of launch-level work per step. **A V4.1 decode component breakdown is unavailable** from these traces; it needs graph-aware attribution or an eager diagnostic arm.

## 4. DSpark speculative decoding

Setup per [docs/speculative-decoding.md](docs/speculative-decoding.md):

```json
{"method": "dspark", "num_speculative_tokens": 5, "revision": "<target sha>",
 "draft_sample_method": "probabilistic", "rejection_sample_method": "standard",
 "enable_adaptive_verification": false | true}
```

The draft ships inside each target checkpoint, but vLLM resolves its revision separately (default `main`), so it is pinned to the target SHA. Resolved draft classes: `DSparkV41DraftModel` (V4.1) and `DSparkDraftModel` (0731). Prefix caching is off. Workload: the concurrency points at c1/4/16/64 with the same seeds, and each model is compared with its own speculation-off run. Launch order: V4.1 fixed → V4.1 adaptive → 0731 fixed → 0731 adaptive, chained without restarts in between ([bench/chain_dspark.sh](bench/chain_dspark.sh)). Tables: [dspark_tables.md](reports/v41-vs-0731/dspark_tables.md), generated by [bench/spec_compare.py](bench/spec_compare.py).

### 4a. Throughput vs each model's own speculation-off baseline

| Conc. | V4.1 off | V4.1 fixed | V4.1 adaptive | 0731 off | 0731 fixed | 0731 adaptive |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 88.4 | 136.7 ± 4.9 (**1.55×**) | 133.1 ± 6.6 (**1.50×**) | 98.3 | 183.4 ± 6.4 (**1.87×**) | 182.0 ± 4.1 (**1.85×**) |
| 4 | 190.1 | 211.6 ± 2.4 (**1.11×**) | 217.8 ± 3.2 (**1.15×**) | 232.5 | 301.9 ± 7.0 (**1.30×**) | 300.8 ± 4.0 (**1.29×**) |
| 16 | 260.3 | 261.5 ± 1.2 (1.00×, inconcl.) | 265.3 ± 1.4 (1.02×) | 329.2 | 307.9 ± 106.3 (0.94×, inconcl.; median 366.1) | 342.7 ± 20.2 (1.04×, inconcl.) |
| 64 | 292.3 | 286.5 ± 0.3 (**0.98×**, worse) | 287.6 ± 0.5 (**0.98×**, worse) | 392.4 | 392.8 ± 2.9 (1.00×, inconcl.) | 394.4 ± 0.8 (1.01×) |

0731 fixed at c16, repeat 1, ran at 185 tok/s with TPOT 68.5 ms, vs 366 and 372 tok/s for repeats 2–3, with normal acceptance and no preemptions. This matches 0731's slow-first-repeat-at-new-shapes pattern. It is kept, and the median is shown.

**V4.1 / 0731 output-throughput ratio by arm:**

| Conc. | Off | Fixed | Adaptive |
| --- | --- | --- | --- |
| c1 | 0.90 | 0.75 | 0.73 |
| c4 | 0.82 | 0.70 | 0.72 |
| c16 | 0.79 | 0.85 | 0.77 |
| c64 | 0.74 | 0.73 | 0.73 |

### 4b. Acceptance (sums over 3 repeats)

| Model / arm | Accepted / proposed (c1, c4, c16, c64) | Tokens per round | Per-position acceptance, pos 0 → 4 (c64) |
| --- | --- | --- | --- |
| V4.1 fixed | 0.315, 0.325, 0.341, 0.341 | 2.57–2.71 | see dspark_tables.md |
| V4.1 adaptive | 0.291, 0.314, 0.286, 0.332 | 2.43–2.66 | 0.631, 0.397, 0.282, 0.202, 0.150 |
| 0731 fixed | 0.449, 0.487, 0.490, 0.478 | 3.25–3.45 | 0.699, 0.558, 0.433, 0.370, 0.328 |
| 0731 adaptive | 0.436, 0.469, 0.445, 0.464 | 3.18–3.35 | 0.695, 0.546, 0.418, 0.353, 0.311 |

"Proposed" is vLLM's `spec_decode_num_draft_tokens`, which counts **scheduled** drafts: exactly 5.00 per round in all arms, including adaptive. **Verified-candidate counts under adaptive verification are not exported by this build and are unavailable.** Acceptance is flat or slightly rising with load, so the lost gain at high concurrency is not a drafter-quality effect. These are random-token prompts; real-text acceptance will differ (not measured).

### 4c. Why the gain disappears, and where the latency goes (c64)

| Model / arm | Mean TTFT | Mean TPOT | Mean E2E | c ÷ request rate |
| --- | ---: | ---: | ---: | ---: |
| V4.1 off | 9.2 s | 183.1 ms | 55.9 s | 56.1 s |
| V4.1 fixed | 19.8 s | 141.9 ms | 56.0 s | 57.2 s |
| V4.1 adaptive | 19.9 s | 140.7 ms | 55.8 s | 57.0 s |
| 0731 off | 7.4 s | 134.1 ms | 41.6 s | 41.8 s |
| 0731 fixed | 18.6 s | 85.3 ms | 40.4 s | 41.7 s |
| 0731 adaptive | 18.7 s | 85.4 ms | 40.4 s | 41.5 s |

At 16K/256 the prompt side needs 64× more tokens than the output side, and prefill already saturates the GPUs (section 1a). DSpark speeds up only decode, so request throughput, and therefore mean E2E at fixed concurrency (Little's law), cannot change. Requests finish decoding sooner and then wait longer in the prefill queue. *Arithmetic on measured values.* Consequence: DSpark gains at high concurrency require a decode-bound workload (short prompts or long outputs, for example 16K/2,048 as in docs/speculative-decoding.md step 4, not run here).

The ITL p95 of DSpark arms is **not comparable** with speculation off: several tokens can arrive in one stream chunk. See dspark_tables.md for the raw values.

### 4d. Memory cost

| Arm | V4.1 KV tokens | 0731 KV tokens | Peak graph capture (per-launch log) |
| --- | ---: | ---: | --- |
| Off | 9,179,728 | 1,711,998 | — |
| Fixed | 7,381,775 | 1,558,746 | ≈ 3 GiB |
| Adaptive | 6,248,184 | 1,375,898 | 6.8 GiB (V4.1), 8.0 GiB (0731) |

Adaptive verification captures an additional full-plus-piecewise CUDA-graph route for variable-length verification. The larger capture is the evidence that the adaptive path is active, since the module logs nothing itself.

### 4e. Output agreement with speculation off (greedy target)

[dspark_output_match.md](reports/v41-vs-0731/dspark_output_match.md) ([bench/compare_outputs.py](bench/compare_outputs.py)) compares each prompt's text with the speculation-off run on the same prompt. Exact matches, summed over repeats:

| Arm | c1 | c4 | c16 | c64 |
| --- | --- | --- | --- | --- |
| V4.1 fixed | 3/48 | 1/48 | 2/192 | 18/768 |
| V4.1 adaptive | 2/48 | 2/48 | 3/192 | 14/768 |
| 0731 fixed | 7/48 | 11/48 | 47/192 | 147/768 |
| 0731 adaptive | 12/48 | 14/48 | 52/192 | 148/768 |

Outputs typically diverge after about 60–140 characters. **This is not evidence of lossy speculation.** No same-seed speculation-off repeat exists, so the determinism of greedy decoding itself under batching (random-token prompts invite near-tie logits) is unmeasured. The needed control is a same-seed speculation-off rerun on each model (about 15 min of GPU time), followed by a real-text task comparison.

## Failures and deviations

| Item | What happened | Effect |
| --- | --- | --- |
| Pre-study V4.1 launch crash | Triton `FileNotFoundError`: 8 TP ranks compiling concurrently on the gocryptfs FUSE mount at `/root` | Fixed by `TRITON_CACHE_DIR=/tmp/triton_cache` |
| 0731 first launch (`serve-off-profidle-20260927-221524.log`) | `CUDA error: invalid argument` during JIT/warmup; worker died | Kept. Fixed by `FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_ws`; the relaunch 6 min later succeeded |
| V4.1 `prefix/off-cache-off/…n1…/repeat-1` | Invalid: `pandas` missing for the custom dataset; no request sent | Kept; valid `repeat-1-rerun1` |
| V4.1 `prefix/off-cache-off/…n4…/repeat-1` | Aborted by the operator while stopping the failing loop | Kept; valid `repeat-1-rerun1` |
| 0731 slower first repeats | 5–9% lower at 64K/128K context and c16–c48; 0731 DSpark fixed c16 r1 at half speed | Kept; medians shown where they matter |
| Model order | V4.1 measured first, 0731 second; no cross-model alternation (would have needed extra restarts) | Recorded in the addendum |
| Launch config for 0731 | Concurrency/context/prefix cache-off ran on `off-profidle` (V4.1 on `off`); control shows the idle profiler has no effect | Recorded |
| Prefix prewarm | Harness prewarms with one request per prefix, which does not fully warm 0731 (section 1d) | Reported as a measured model/runtime difference, not rerun |
| Sequential prefix check | 0731's original script was not saved; `bench/prefix_reuse_check.py` reconstructs it (same prompt files, endpoint, and 67,588-token queries) and was used for V4.1 | Recorded |
| GPU idle time | A broken readiness loop left the V4.1 DSpark server idle for about 1 h 50 min before its sweep | No data effect; the remaining arms ran through `chain_dspark.sh` |
| Telemetry unavailable | V4.1 decode components (CUDA graphs); DSpark verified-candidate counts; per-request queue split | Unavailable, not zero |

## Not covered (declared follow-ups)

- **DSpark:** same-seed speculation-off determinism control; real-text prompt sets (code/math/chat); decode-bound workload (16K/2,048, short prompts); widths k1/3/7; long context; DSpark component traces; greedy-draft arm.
- **Prefill:** V4.1 all-reduce tuning arm; a block-scale FP8 dense-GEMM path for V4.1; a CED-capable runtime; `max_num_batched_tokens` tuning; V4.1 decode attribution under CUDA graphs.
- **Harness:** a 0731-aware prewarm (two touches per prefix) as a separately labeled arm.
- **Quality:** real-text task set. The 4-prompt smoke checks (`data/<model>/_quality_smoke/`) are sanity checks only.

## Reproduce

See [docs/reproduce.md](docs/reproduce.md) for the full command sequence, environment fixes, and timings. In short:

```bash
source /root/vllm/bin/activate            # vLLM 0.30.1rc1.dev223+g44af287eb
export HF_TOKEN=...  HF_HOME=/workspace/hf
# per model (v41 | v4-0731): launch in tmux ds-serve, harness in ds-bench
bash bench/serve.sh <model> off-profidle <log>     # PROFILE_DIR=... required
python bench/run_matrix.py --model <model> --config off-profidle --workloads concurrency,context,prefix,isolated --prefix-states cache-off
bash bench/serve.sh <model> off-prefix <log>
python bench/run_matrix.py --model <model> --config off --workloads prefix --prefix-states cold,prewarmed
python bench/prefix_reuse_check.py --model <model>
bash bench/chain_dspark.sh 'v41 dspark-fixed-k5' 'v41 dspark-adaptive-k5' 'v4-0731 dspark-fixed-k5' 'v4-0731 dspark-adaptive-k5'
python bench/summarize.py v41-vs-0731 <model> --csv ...; python bench/compare.py --csv ...; python bench/spec_compare.py --csv ...
python bench/curate.py v41-vs-0731 <model>
```
