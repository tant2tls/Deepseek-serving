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
