# MiMo-V2.6-Flash-MOPD serving study plan

Study ID `mimo-v26`. Checkpoint `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @ `2479e2d0029eca9a34cc7e7f55a121925f81908e` (177.8 GB, downloaded 2026-09-28). The user confirmed starting this study on 2026-09-28, after the DeepSeek report was pushed. It uses the same node, vLLM build (`0.30.1rc1.dev223+g44af287eb`), server limits, harness, and workloads as [the DeepSeek report](../reports/v41-vs-0731/report.md), so its results sit on the same axes as V4.1 and 0731. Tokenizers differ, so random-token prompts are regenerated per model to the same token counts. Declared plan: `results/mimo-v26/plan.json`, copied to `reports/mimo-v26/` at publication.

**Scope update, 2026-09-30:** this is the completed MiMo study's implementation record and historical plan. New work follows [target.md](../target.md) and [the three-model GPU plan](https://github.com/tant2tls/Deepseek-serving/blob/be9ee6abfc251d8145c44a375cd2de8907228513/docs/three-model-h100-plan.md). Do not run its historical prefix/prewarm/reuse instructions or `chain_mimo.sh` for the active phase. Its pre-run predictions are not new results; use [the MiMo report](../reports/mimo-v26/report.md) for measured findings.

## Architecture (from the pinned `config.json`) vs DeepSeek V4/V4.1

| Property | MiMo-V2.6-Flash-MOPD | DeepSeek V4 0731 / V4.1 (as measured) | Expected serving effect |
| --- | --- | --- | --- |
| Layers | 48: layer 0 dense FFN (16,384), layers 1–47 MoE | 43 / 40 | — |
| Attention | **Hybrid GQA**: 9 full-attention layers (0, 5, 11, 17, 23, 29, 35, 41, 47; 64 Q / 4 KV heads, head 192, V 128) and 39 sliding-window layers (window **128**, 64 Q / 8 KV heads, with learned attention-sink bias) | Sparse top-k attention with an indexer (compressed KV) | Full layers are dense O(n²) in prefill, so long-context prefill should grow faster than DeepSeek's (whose core attention stayed flat at ~50 ms per chunk). SWA layers are O(n·128). |
| Position | Partial RoPE (33.4% of dims), θ 10M full / 10K SWA; `max_position_embeddings` 1,048,576 | — | Serve at 262,144, matching DeepSeek |
| MoE | 256 routed experts, top-8, expert FFN 2,048, sigmoid + `noaux_tc`, no shared expert; hidden 4,096 | Hidden 4,096 (0731) / 5,120 (V4.1) | Roughly 300B total and 11–12B active parameters (estimate from shapes) |
| Precision | FP8 e4m3, 128×128 weight blocks, dynamic activations; experts stored **MXFP4** (`store_dtype: mxfp4` → vLLM `Mxfp4MoEMethod`); `o_proj` kept BF16 (49 ignored layers) | V4.1: FP8 32×32 blocks with UE8M0 scales (Marlin), FP4 experts; 0731: DeepGEMM/FlashInfer block-scale | Dense GEMMs should take a block-scale FP8 path like 0731's, not V4.1's Marlin. Verify in the trace. MoE kernel on Hopper for MXFP4 is to be identified from the log/trace. |
| Speculative heads | Classic **MTP, 3 layers** (`model.mtp.*` with `eh_proj`/`enorm`/`hnorm`); separate **DFlash** drafter in `dflash/` (qwen3-type, 5 SWA layers, block 8, taps layers 0/11/23/35/47, sink bias) | DSpark (both) | Different method families; compare with DSpark only as deployment outcomes |
| Extra weights | Vision (360 tensors) and audio encoder tensors in the same checkpoint | — | Verified: loaded by the Omni wrapper (vision tower built, audio tokenizer not found so audio is disabled); inputs disabled, text decoder only |

## Runtime support (pinned vLLM source, checked 2026-09-28)

- `MiMoV2ForCausalLM` is registered (`models/mimo_v2.py`), but because the config carries a vision config, vLLM's `model_arch_config_convertor.py` rewrites the architecture to **`MiMoV2OmniForCausalLM`** (`models/mimo_v2_omni.py`). Every MiMo launch logged `Resolved architecture: MiMoV2OmniForCausalLM`. That class wraps the same `MiMoV2FlashForCausalLM` text decoder as `language_model`. The vision tower is built, but multimodal inputs are disabled (`--language-model-only`), so the served path is the text decoder. Implements hybrid SWA/full attention with sinks, per-layer `attention_value_scale`, and fused/unfused FP8 QKV loading, and skips `mtp.*` weights in the target.
- The config class is not in transformers 5.17, so it needs `--trust-remote-code` with `--code-revision` pinned to the same SHA.
- Parsers: `--reasoning-parser mimo` and `--tool-call-parser mimo` (Qwen3-engine adapters), with the HF tokenizer (`--tokenizer-mode auto`).
- **Thinking off:** `chat_template_kwargs.enable_thinking=false` makes the template emit `<think></think>`. DeepSeek's `thinking=false` would be ignored, so the harness now carries a per-model request body.
- Classic MTP: `mimo_v2_mtp` is registered, and speculative config handles `MiMoV2MTPModel` (`method: "mtp"`).
- DFlash: vLLM has a `dflash` method with `dflash_config` fields (`target_layer_ids`, `mask_token_id`). Whether it supports this drafter's `attention_sink_bias`/`attention_value_scale` and loads from the `dflash/` subfolder is **unverified**. It gets a smoke launch before any timing.
- The model card recommends TP4 with `--gpu-memory-utilization 0.95`. This study uses **TP8 + EP at 0.90** to match DeepSeek. A TP4 or DP-attention deployment arm is optional.

## Predictions from the DeepSeek study (to confirm or refute)

1. **16K/256 concurrency will be prefill-bound.** Output tok/s ≈ prompt rate × 256 / 16,388 at c64. MiMo's rank depends on prefill µs/token.
2. **All-reduce path:** hidden 4,096 gives 67.1 MB messages per 8,192-token chunk, the same size that took symm-mem multimem on 0731. Expect all-reduce closer to 0731 (~79 ms/chunk for 43 layers) than V4.1 (ring-LL, ~108 ms).
3. **Dense GEMM path:** 128×128 FP8 blocks should resolve to a block-scale FP8 GEMM (DeepGEMM/CUTLASS), not Marlin. If the log shows Marlin, expect a V4.1-like dense-GEMM penalty.
4. **Long context:** full attention in 9 of 48 layers should make prefill µs/token rise with context more steeply than DeepSeek's (0731 went 41.6 → 59.8 µs from 16K to 260K).
5. **Prefix caching:** SWA layers use vLLM's hybrid KV manager. Whether reuse needs a second touch (as on 0731) must be measured with `prefix_reuse_check.py` **before** the prefix arm. If it does, record it and interpret prewarmed runs accordingly.
6. **Decode:** GQA KV reads grow with context in the 9 full layers, so c1 TPOT may rise with context, unlike DeepSeek's flat 7.4–7.85 ms.
7. **Speculation:** gains are expected only at c ≤ 4 on this prefill-bound workload. A decode-bound point (16K/2,048 or short prompts) is needed to see MTP/DFlash at load.

## Launches and matrix (same definitions as the DeepSeek study)

| Launch | Config | Workloads | Est. GPU time |
| --- | --- | --- | --- |
| 1 | `off-profidle` (prefix caching off, idle profiler) | smoke (thinking off, correctness 4 prompts) → concurrency c1–c64 ×3 → context 16K/64K/128K/260K ×3 → prefix cache-off ×3 → isolated c1 1K–260K ×3 → traces `prefill16k_decode8`, `prefill64k_decode4` | 10 min launch + ~2 h 55 min |
| 2 | `off-prefix` | `prefix_reuse_check.py` first, then prefix cold/prewarmed ×3 | 5 + 35 min |
| 3 | `mtp-k3` (native classic MTP, 3 tokens) | concurrency c1/4/16/64 ×3 | 5 + 25 min |
| 4 | `dflash` (only if the smoke load passes) | concurrency c1/4/16/64 ×3 | 5 + 25 min |

Total ≈ 4.5 h of GPU time if nothing fails. Launches 2–4 run through a chain script (no idle gaps). Validity, repeats, seeds, and publication rules are identical to [docs/reproduce.md](reproduce.md).

## Deliverables

A `report.md` section or separate report for MiMo, with the same tables as the DeepSeek report and a three-way ratio view (MiMo / 0731 / V4.1) on the identical axes. Also: the trace component table with a kernel-path check for predictions 2–3, prefix-reuse behavior, speculative arms vs own baseline, and failures. The speculative method differs between families, so cross-model speculative comparisons are deployment outcomes only.
