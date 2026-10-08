# B0: architecture and implementation facts

Study `blog-architecture-h100-v1`, recorded 2026-10-07 on vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`), TP8 + EP, 8×H100.

Every entry carries its evidence label: **config** (the checkpoint's `config.json` at the pinned revision), **card** (model card in the same snapshot), **log** (server startup log of this study), **source** (vLLM code on the node), **estimate** (arithmetic from shapes). Entries are facts about these deployments, not measurements of speed.

## Structure (config)

| | V4 Flash 0731 | V4.1 Flash | MiMo-V2.6-Flash-MOPD |
| --- | --- | --- | --- |
| Revision | `7872f01b…5a386062` | `dba1be0a…0a90277` | `2479e2d0…5f81908e` |
| Decoder layers | 43 | 40 (card: 20 encoder + 20 decoder) | 48 |
| Hidden size | 4096 | 5120 | 4096 |
| Query heads × head dim | 64 × 512 | 64 × 512 | 64 × 192 (value dim 128) |
| KV heads | 1 | 1 | 4 in full-attention layers, 8 in sliding-window layers |
| Low-rank projections | q rank 1024, o rank 1024 in 8 groups | q rank 1280, o rank 1024 in 8 groups | none (fused QKV) |
| Residual streams (mHC) | 4 | 4 | 1 (plain residual) |
| Vocabulary | 129,280 | 129,280 | 152,576 |

## Attention (config, card)

| | V4 Flash 0731 | V4.1 Flash | MiMo-V2.6-Flash-MOPD |
| --- | --- | --- | --- |
| Layer types | Layers 0–1 sliding window only; layers 2–42 alternate compression ratio 4 and 128 on the long-range KV | Layers 0–1 sliding window only; layers 2–19 ratio 2; layers 20–39 ratio 1 | 9 full causal layers (0, 5, 11, 17, 23, 29, 35, 41, 47); 39 sliding-window layers |
| Sliding window | 128 tokens | 128 tokens | 128 tokens, with a learned sink on window layers |
| Long-range selection | Learned indexer, 64 heads × 128, top-512 | Learned indexer, 32 heads × 128, top-512; indexer state from 8 source layers; later layers restricted to 2,048 candidate blocks of 8 chosen at layer 20 | None: full layers read the whole context |
| KV sharing across layers | No | Yes: main KV comes from 4 source layers (2, 8, 14, 20) | No |
| Positional scheme | RoPE on 64 dims, YaRN factor 16 | RoPE on 64 dims, YaRN factor 16 | Partial RoPE on about a third of the dims |
| Extra memory modules | 3 hash layers | Engram n-gram memory on layers 1 and 14 | None |

## FFN (config)

| | V4 Flash 0731 | V4.1 Flash | MiMo-V2.6-Flash-MOPD |
| --- | --- | --- | --- |
| Routed experts per layer | 256 | 384 | 256 |
| Routed experts active per token | 6 | 6 | 8 |
| Shared experts (run for every token) | 1 | 1 | 0 |
| Expert intermediate width | 2048 | 2304 | 2048 |
| Router scoring | sqrt-softplus | sqrt-softplus | sigmoid |
| Dense (non-MoE) FFN layers | none | none | layer 0, width 16,384 |
| Fraction of routed experts a token uses | 6/256 = 2.3% | 6/384 = 1.6% | 8/256 = 3.1% |

**Estimate**, not measured: one expert has about 3 × hidden × width weights. That is 25.2M for 0731 and MiMo and 35.4M for V4.1. Active expert weights per MoE layer are then (6+1) × 25.2M = 176M for 0731, (6+1) × 35.4M = 248M for V4.1 and 8 × 25.2M = 201M for MiMo. The model with the lowest activation fraction (V4.1) touches the most expert weights per layer, because its experts are larger and it adds a shared expert.

## Precision and executed backends (config, log)

| | V4 Flash 0731 | V4.1 Flash | MiMo-V2.6-Flash-MOPD |
| --- | --- | --- | --- |
| Served class | `DeepseekV4ForCausalLM` | `DeepseekV41ForCausalLM` | `MiMoV2OmniForCausalLM` wrapping the text decoder (multimodal inputs disabled) |
| Dense weights | FP8 E4M3, 128×128 blocks, UE8M0 scales | FP8, 32×32 blocks, UE8M0 scales | FP8 E4M3, 128×128 blocks; `o_proj` left in BF16 |
| Expert weights | FP4 | FP4 | MXFP4 |
| FP8 linear kernel | `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | DeepGEMM enabled; low-rank projections also run on the `humming` low-bit GEMM (trace) | `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` |
| MoE backend | `HUMMING` | `HUMMING` | `HUMMING` |
| Attention backend | `FLASHMLA_SPARSE_DSV4` + `DEEPSEEK_SPARSE_SWA` + indexer + compressor | `FLASHMLA_SPARSE_DSV41` + `DEEPSEEK_SPARSE_SWA` + V4.1 indexer + compressor | `FLASH_ATTN_DIFFKV` (FlashAttention 3, upgraded to 4 for sinks) |
| KV cache format | `fp8_ds_mla`, FP8 indexer cache | `fp8_ds_mla`, FP8 indexer cache | BF16 (`auto`) |
| KV block size (cache groups) | 256 tokens (groups 64, 64, 256, 4, 8) | 64 tokens (groups 32 ×7, 64, 8) | 16 tokens (six groups of 16) |
| TP all-reduce | FlashInfer first, fused with RMS norm (`allreduce_rms`) | same | same |
| Weights per GPU | 19.79 GiB | 36.32 GiB, plus two Engram tables of 11.80 GiB each in pinned host memory per rank | 20.1 GiB |
| KV pool per GPU at 0.90 utilization | 47.43 GiB (1,728,531 tokens) | 29.14 GiB (9,092,347 tokens) | 49.18 GiB (6,969,972 tokens) |
| Prefill skip | not applicable | **active**: layers 21–39 run on each request's last 128 tokens in prefill steps of at least 768 tokens ([check](v41_prefill_check.md)) | not applicable |

## What changed against the old pinned build

These are implementation changes between vLLM `44af287e…` (completed studies) and `554340f3…` (this study). They are the reason old and new numbers must not be mixed.

| Item | Old build | New build |
| --- | --- | --- |
| MXFP4/FP4 expert GEMM | Marlin W4A16 | `HUMMING` for all three models |
| V4.1 dense FP8 | Marlin FP8 | DeepGEMM plus `humming` for low-rank projections |
| V4.1 prefill | All 40 layers on every prompt token | Layers 21–39 on the last 128 tokens only |
| TP all-reduce | symm-mem / NCCL ring, separate from the norm | FlashInfer all-reduce fused with RMS norm |
| V4.1 Engram | on GPU path | tables offloaded to pinned host memory |

## Three meanings of "sparse" in these models

| Kind | 0731 | V4.1 | MiMo |
| --- | --- | --- | --- |
| Attention sparsity: which past tokens are read | Window of 128 plus top-512 compressed positions chosen by an indexer | Same idea, with shared KV and a candidate pool | Window of 128 in 39 layers; everything in 9 layers |
| MoE activation sparsity: which experts run | 6 of 256 routed, plus 1 shared | 6 of 384 routed, plus 1 shared | 8 of 256 routed, no shared |
| Weight sparsity or low-bit storage | No structured zeros; FP8 dense, FP4 experts | No structured zeros; FP8 dense, FP4 experts | No structured zeros; FP8 dense, MXFP4 experts |

None of the three has hardware structured weight sparsity. "Sparse" here always means *fewer positions* or *fewer experts*, and both come with selection work: an indexer for attention, a router and token dispatch for experts.

## Not verified here

- Tensor-by-tensor shapes of every projection; the table uses config fields.
- Resident weight bytes split by dense, expert and metadata. Only the per-GPU total from the load log is recorded.
- Which exact kernels serve 0731's low-rank projections on this build; see `components.csv` and `components_unclassified.md`.

## Added models: Qwen3.8-Flash-Next and GLM-5.3-Flash

Same build and deployment. Entries are from the pinned checkpoint configs and this study's server logs; kernel-level statements are in [components_tables.md](components_tables.md). Neither model was in the first three, so read this section together with the order caveat in [findings.md](findings.md).

Both models replace most softmax-attention layers with **recurrent ("linear") attention**: a layer keeps a fixed-size state per request and updates it once per token, so its cost and memory do not grow with context. Every fourth layer is a sparse softmax-attention layer with an indexer, which is where long-range lookup happens.

| | Qwen3.8-Flash-Next-FP8 | GLM-5.3-Flash |
| --- | --- | --- |
| Revision | `236dfdf2…a3469b13` | `eb9eb208…5f52574a` |
| Served class | `Qwen4ExpForConditionalGeneration` (text only) | `Glm5NextForConditionalGeneration` (text only) |
| Layers | 48: 36 recurrent + 12 softmax-attention (every 4th: 3, 7, …, 47) | 45: 34 recurrent + 11 sparse-attention (3, 7, …, 43) |
| Hidden size | 2560 | 4096 |
| Softmax-attention layers | 24 heads × 256, 2 KV heads; the config calls them "full attention", the executed kernels are sparse (`_qsa_sparse_…`) with an indexer of budget 2,048 and compression ratio 4 | 64 heads, MLA without positional part (`mla_use_nope`), Q rank 1536, KV rank 512; indexer 32 heads × 128, top-2,048, K-pool 4 |
| Recurrent layers | gated delta rule, 16 key heads × 128 and 48 value heads × 128, causal conv kernel 4 | recurrent layers handled through the same Mamba-style state pages |
| Residual scheme | 4 hyper-connection streams, low rank 320 | mHC, 4 streams |
| Extra modules | n-gram (PLE) embedding at one layer | – |
| Routed experts / active per token | 512 / 10, plus a shared expert of the same width | 288 / 8, plus 1 shared |
| Expert intermediate width | 640 | 2048 |
| Dense FFN layers | none | first 3 layers, width 12,288 |
| Fraction of routed experts a token uses | 10/512 = 2.0% | 8/288 = 2.8% |
| Weight format | FP8, 128×128 blocks (experts too) | FP8 E4M3 |
| MoE backend (log) | `FLASHINFER_CUTLASS` FP8 | `FLASHINFER_CUTLASS` FP8 |
| FP8 linear kernel (log) | DeepGEMM / FlashInfer | `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` |
| KV block size (log) | attention block 400 tokens, equal to the recurrent-state page | attention block 640 tokens, equal to the recurrent-state page |
| Weights per GPU (log) | 17.36 GiB | 38.8 GiB |
| KV pool per GPU (log) | 51.44 GiB (4,194,304 tokens) | 28.73 GiB (2,618,281 tokens) |
| Thinking off | `enable_thinking=false`, verified | **not available**; lowest `reasoning_effort` used |

**Estimate**, not measured: one Qwen expert has about 3 × 2560 × 640 = 4.9M weights, so a token's eleven experts touch about 54M weights per layer, against 176–248M for the first three models. Qwen's experts are small and numerous; GLM's are the same size as 0731's and MiMo's (25.2M) with nine active (227M per layer).
