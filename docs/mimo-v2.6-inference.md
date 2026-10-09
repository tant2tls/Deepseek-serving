# How MiMo-V2.6-Flash runs in vLLM, and why it may differ from DeepSeek V4/V4.1

This is a source- and config-level walkthrough of MiMo-V2.6-Flash-MOPD prefill and decode on the pinned runtime (vLLM `0.30.1rc1.dev223+g44af287eb`, TP 8 + EP, H100). It sits next to the DeepSeek paths measured in [the DeepSeek report](../reports/v41-vs-0731/report.md). Every statement is labelled: **source** (vLLM or config code read on 2026-09-28), **log** (server startup log), **estimate** (arithmetic from weight shapes, not measured), or **measured** (study `mimo-v26`, filled in as runs complete). Measured sections are marked *pending* until the data exists.

## 1. One decoder layer

```
x ─ RMSNorm ─ QKV proj (FP8 128×128 block, DeepGEMM) ─ partial RoPE (33% of dims) ─ V × 0.707
          ─ attention: FlashAttention-3 "DiffKV" (K head 192, V head 128)
                 · 39 layers: sliding window 128 keys + learned per-head sink logit
                 ·  9 layers: full causal attention (layers 0,5,11,17,23,29,35,41,47)
          ─ o_proj (BF16, row-parallel) ─ TP all-reduce ─ + residual
  ─ RMSNorm ─ router (BF16 linear, sigmoid + bias-corrected top-8 of 256)
          ─ 8 experts, MXFP4 weights (Marlin W4A16 MoE) ─ reduce ─ + residual
```

*Source:* `vllm/model_executor/models/mimo_v2.py` (`MiMoV2FlashDecoderLayer`, `MiMoV2Attention`, `MiMoV2MoE`). The served class is `MiMoV2OmniForCausalLM` (vLLM rewrites the architecture because the config has a vision tower). It runs this same decoder as `language_model`; multimodal inputs are disabled. *Log:* `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for Fp8LinearMethod`, `Using 'MARLIN' Mxfp4 MoE backend`, `Using FLASH_ATTN_DIFFKV for attention` (FA3). Layer 0 uses a dense 16,384-wide FFN instead of MoE. KV cache dtype is BF16 (`auto`). CUDA graphs run in `FULL_AND_PIECEWISE` mode.

### The same layer in DeepSeek V4 (0731) / V4.1, as measured

```
x (4 parallel residual streams, hc_mult=4) ─ mHC mixing (Sinkhorn-normalised) ─ RMSNorm
  ─ q-LoRA (1024/1280) → 64 heads × 512 ; single shared KV head 512 (MQA-style, fp8_ds_mla cache)
  ─ attention: sliding window 128 + compressed long-range KV (per-layer compress ratio)
               selected by a learned indexer (top-512), sparse attention kernel
  ─ grouped o-LoRA (8 groups × 1024) ─ all-reduce ─ mHC mixing
  ─ MoE: 6 routed + 1 shared expert (FP4) ─ all-reduce ─ (V4.1: Engram n-gram memory on some layers)
```

## 2. Per-token work implied by weight shapes (*estimate*)

| Per layer (active weights) | MiMo | 0731 | V4.1 |
| --- | ---: | ---: | ---: |
| Attention linear (QKV + o) | ≈ 94M | ≈ 107M | ≈ 127M |
| MoE active | 8 × 25.2M = 201M | (6 + 1) × 25.2M = 176M | (6 + 1) × 35.4M = 248M |
| Layers | 48 | 43 | 40 |
| **Linear weights touched per token** | **≈ 14.2B** | **≈ 12.2B** | **≈ 15.0B** |
| Extra per-layer ops | none | mHC ×4 streams, indexer, KV compressor | same + Engram |

**MiMo does about 16% more dense matrix work per token than 0731.** Any prefill advantage therefore has to come from (a) the operations it doesn't have, (b) cheaper attention, or (c) better kernels. It cannot come from a smaller model.

## 3. Prefill (8,192-token chunks, chunked prefill)

1. **Linear layers** (QKV, o, router, experts): about 2 × 14.2B FLOPs per token (*estimate*). QKV and dense FFN use the same block-scaled FP8 DeepGEMM path that made 0731's dense GEMM 2.3× cheaper per layer than V4.1's Marlin FP8 ([DeepSeek report §3](../reports/v41-vs-0731/report.md)). `o_proj` stays BF16, which costs extra compute and bandwidth at 4096×8192 per layer.
2. **Sliding-window layers (39):** each query attends to at most 128 keys plus a sink. Cost is O(n · 128) and independent of context. KV outside the window is freed (hybrid KV manager).
3. **Full-attention layers (9):** dense causal FlashAttention, O(n²). Estimated attention FLOPs per token per layer ≈ 2 × 64 heads × (192 + 128) × L_ctx ≈ 41K × L_ctx. Averaged over a prompt of length n, that is ≈ 9 × 41K × n/2 per token: ≈ 3 GFLOP/token at 16K (about 10% of the linear work), ≈ 24 GFLOP at 128K (comparable to the linear work), and ≈ 48 GFLOP at 260K (≈ 1.7× the linear work). **Prediction:** MiMo's prefill µs/token should be flat to 16–32K, then rise more steeply than DeepSeek's. DeepSeek's sparse top-512 attention kept core attention at ≈ 50 ms per chunk at any context; only its indexer grew.
4. **Communication:** two TP all-reduces per layer, 96 per chunk, each `[8192, 4096]` BF16 = 67.1 MB. That is the message size 0731 sent through symm-mem multimem (79 ms per 87 all-reduces). V4.1's 84 MB messages fell to NCCL ring-LL (108 ms).
5. **Ops DeepSeek pays that MiMo does not:** mHC mixing/norms (0731: 27.9 ms per chunk, 9.5%), indexer + top-k (0731: 8.9 ms at 16K, 40.5 ms at 57K context), KV compressor and other elementwise work (0731: 24.5 ms), and Engram on V4.1.

## 4. Decode (one token per sequence per step)

- **Weights:** the same ≈ 14.2B active linear weights per token, read once per step for the whole batch. At small batch this is bandwidth-bound: MXFP4 experts (≈ 0.53 B/param) plus FP8 attention.
- **KV reads:** the 9 full layers read the whole context. With 4 KV heads and TP 8, each rank holds 1 KV head (replicated): 9 × (192 + 128) × 2 B = 5.8 KB per context token per rank, about 1.5 GB per step at 260K. That is about 0.45 ms at 3.35 TB/s (*estimate*), so TPOT should rise mildly with context. The 39 sliding layers read ≤ 128 tokens each, a constant cost.
- **DeepSeek for comparison:** compressed single-head KV plus top-512 sparse selection kept decode flat at 7.4–7.85 ms/token from 1K to 260K (measured).

## 5. Measured (study `mimo-v26`, launch 1: 72/72 valid runs; traces rank 0)

### 5a. End-to-end, same axes as the DeepSeek report

| Workload | MiMo | MiMo / 0731 | MiMo / V4.1 |
| --- | --- | --- | --- |
| Concurrency 16K/256, c1 → c64 | 130.1 → 504.6 tok/s (SD ≤ 0.9) | 1.27–1.33× | 1.47–1.73× |
| Context c8, 16K / 64K / 128K / 260K | 376.2 / 115.9 / 55.4 / 24.1 tok/s | 1.28 / 1.36 / 1.39 / **1.42×** | 1.63–1.69× |
| Isolated c1, 1K → 260K | TTFT 93 ms → 10.95 s | 1.31–1.40× | 1.43–1.57× |
| Prefix cache off, 64K + 2K, c8 | 112.9–113.0 tok/s | 1.33–1.34× | 1.70× |

- **Prefill cost:** 33.2–34.1 µs/token from 8K to 64K, 36.2 at 128K, and 42.1 at 260K (isolated c1). That is about 20% below 0731 (41.6 at 16K) and about 40% below V4.1 (54.5). At c64 the server-side prompt rate is ≈ 32.3K tok/s (0731 25.1K, V4.1 18.7K), and output throughput again equals prompt rate × 256 / 16,388: **prefill-bound**, as predicted.
- **Short prompts:** TTFT is 93 ms at 1K vs 148 (V4.1) and 185 (0731), so fixed per-request overhead is lower too.
- **Decode (c1):** TPOT rises 5.54 → 6.27 ms from 1K to 260K, as predicted for full-layer KV reads (+0.73 ms vs the ≈ 0.45 ms estimate). It starts about 25% below DeepSeek's flat 7.4–7.9 ms, so MiMo decodes faster at every measured context.
- **Prediction 4 was wrong:** the full-attention layers do **not** make MiMo lose ground at long context. Its lead over 0731 *grows* from 1.28× to 1.42× (c8 context). Section 5b explains why: attention FLOPs run on FA3 near peak, while the linear layers run far below it, so FLOP share overstated attention's time share.

### 5b. Where the prefill time goes: one 8,192-token chunk, rank 0, kernel ms

| Component | MiMo 16K chunk 2 | 0731 16K chunk 2 | MiMo − 0731 | MiMo 64K chunk 8 (57K ctx) | 0731 64K chunk 8 |
|---|---:|---:|---:|---:|---:|
| **Kernel sum** | **216.3** | **293.1** | **−76.8** | **244.7** | **338.3** |
| TP all-reduce (symm-mem multimem in both) | 94.8 (95 calls, 1.00 ms each) | 79.4 (87, 0.91 ms) | +15.4 | 93.3 | 78.7 |
| MoE expert GEMM (Marlin W4A16) | 71.6 (top-8) | 53.3 (top-6 + shared in dense) | +18.3 | 71.8 | 52.6 |
| MoE routing / combine | 4.6 | 7.3 | −2.7 | 4.6 | 7.3 |
| Attention (MiMo: 39 SWA + 9 full FA3; 0731: sparse) | 9.1 (SWA 1.6 + full 7.5) | 41.9 | −32.8 | **38.8** | **53.2** |
| Indexer / top-k | — | 8.9 | −8.9 | — | **40.5** |
| QKV norm / RoPE / KV insert (+ compressor on 0731) | 1.1 | 10.2 | −9.1 | 1.1 | 10.2 |
| Dense GEMM (QKV, o_proj, layer-0 FFN; 0731 + q/o-LoRA, shared expert) | 11.2 | 39.7 | −28.5 | 11.2 | 39.8 |
| Norm / residual (0731: mHC ×4 streams) | 8.5 | 27.9 | −19.4 | 8.5 | 27.9 |
| Other elementwise (MiMo: SwiGLU 11.8, FP8 act-quant 2.4) | 15.5 | 24.5 | −9.0 | 15.5 | 28.2 |

*Trace.* `bench/trace_breakdown.py` now also files FA3's `FlashAttnFwdSm90` under attention (the DeepSeek breakdowns are byte-identical after the change). **MiMo is faster because it skips DeepSeek's attention machinery (−51 ms: sparse attention, indexer, compressor), has lighter projections with no shared expert (−29 ms), and uses plain residual norms instead of mHC (−19 ms).** It gives back 34 ms on MoE (8 active experts vs 6) and all-reduce (96 vs 87 calls per chunk). At 57K context, MiMo's full attention grows to 38.8 ms, but 0731's attention plus indexer grows to 93.7 ms, which is why MiMo's lead widens with context.

**All-reduce is now MiMo's largest cost:** 44% of the prefill chunk and 27–29% of a decode step. Each 67 MB all-reduce takes about 1.0 ms (≈ 67 GB/s algorithm bandwidth), the same inefficiency seen on 0731 (0.91 ms). *Hypothesis:* the model card's DP-attention deployment (SGLang `--dp 2 --enable-dp-attention`) or a faster all-reduce would target MiMo's biggest component. Measure it as a separate tuning arm.

### 5c. Decode step (c1, 16K context, rank 0)

| Component | MiMo (5.67 ms span, 880 kernels) | 0731 (7.77 ms span, 1,834 kernels) |
|---|---:|---:|
| Dense GEMM | 0.93 | 3.48 |
| TP all-reduce | 1.54 | 1.34 |
| MoE expert GEMM | 1.34 | 1.39 |
| Attention | 0.60 (0.82 at 64K) | 0.99 |
| MoE routing / combine | 0.50 | 0.68 |
| mHC / norm | — | 0.89 |
| Other elementwise + indexer + KV | 0.34 | 1.15 |

0731's kernel sum (9.9 ms) exceeds its span because streams overlap, so compare spans. **MiMo decodes 27% faster mainly because of dense GEMM (−2.5 ms)** — 0731's q-LoRA with 64 × 512-wide heads, grouped o-LoRA and shared expert — plus mHC (−0.9 ms) and half the kernel launches. Weight bandwidth for MoE is similar (1.34 vs 1.39 ms), even though MiMo activates 8 experts instead of 6, because the per-expert size is the same and MXFP4 halves weight bytes.

### 5d. Prefix reuse and speculation (launches 2–4)

- **Prefix reuse needs a second touch**, as on 0731 (the hybrid SWA prefix checkpoint): requests 1–2 on a new 64K prefix miss, and request 3 hits in 0.35 s. See [MiMo report §1c](../reports/mimo-v26/report.md).
- **Speculation:** DFlash-7 gives 1.68× and MTP (layer 0 reused ×3) 1.53× at c1; both are ≤ 1.06× at c ≥ 16 because this workload is prefill-bound. See [MiMo report §3](../reports/mimo-v26/report.md).
