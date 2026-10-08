# Qwen3.8-Flash-Next-FP8 historical reference

**September FP8 archive; separate from the completed October comparison.** Qwen3.8-Flash-Next FP8 was subsequently measured alongside the two DeepSeek deployments, MiMo and GLM on the same node/runtime/core settings; see the [October five-model findings](../../reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash) and [Qwen run evidence](../../reports/blog-architecture-h100-v1/data/qwen-38/). The older results below use different builds/settings and are not controls for that study. All future Qwen work uses the [original BF16 checkpoint only](../../docs/qwen-checkpoint-policy.md); both archives preserve FP8 evidence, and no new FP8 runs are permitted.

This bundle preserves 19 result JSONs, their 19 benchmark logs, three manifests, three selected startup logs, and one small checkpoint configuration from `LLMs_frontier_serving/Qwen3.8-Flash-Next-FP8`. Numerical JSONs are unchanged. Text exports and original hashes are recorded in [provenance.json](../provenance.json). See [RESULTS.md](RESULTS.md) for every retained point and the [shared methodology and caveats](../README.md).

## Select the correct arm

| Arm | Points | Intended use | Boundary |
| --- | ---: | --- | --- |
| [base-util082](results/base-util082/) | 4 | Corrected main batch at c1/c4/c16/c64 | Warm compilation/startup, utilization 0.82. No corrected context/prefix or MTP points exist in this arm. |
| [base](results/base/) | 11 | Original batch, context, and prefix evidence | Utilization 0.85, earlier smaller pool. Batch is superseded for headline comparisons, but retained as the historical MTP comparator. |
| [mtp-n1](results/mtp-n1/) | 4 | Historical MTP comparison against original `base` batch only | Utilization 0.85; startup/pool differences remain. Do not use `base-util082` as this MTP control. |

The corrected batch result is not a correction factor for other axes. Do not mix all `base` and `base-util082` points into a single controlled curve.

## Recorded deployment and config

- Model ID: `Qwen/Qwen3.8-Flash-Next-FP8`; eight H100 80GB HBM3 GPUs, TP8 and EP enabled. vLLM `0.1.dev20073+g8e685d198`, PyTorch `2.13.0+cu130`. The old script's TP4 defaults are not the retained measured deployment.
- Text-only, maximum model length 262,144, maximum sequences 256, prefix caching enabled, Triton MoE backend, `qwen3` reasoning parser, and `qwen3_xml` tool parser. Main batch MTP is off; the MTP arm drafts one token.
- Saved metadata describes FP8 attention/dense and expert weights with BF16 KV. This is a metadata shorthand, not proof that every tensor is FP8. The recorded resolved cache grouping is 4 in the base arms and 8 in the MTP arm; do not infer a byte-cost ratio from those values.
- [config.json](config.json) was extracted from cached snapshot `236dfdf285828023ca3bcd3f37366c58a3469b13`. It contains 48 text layers: 36 `linear_attention` and 12 `full_attention`, 512 experts and ten selected per token. The source describes these as GDN and QSA layers and also discusses PLE n-gram embedding work. Inspect the config and pinned runtime to map each actual operation before profiling.
- The cached config revision does not prove that every benchmark used the same immutable checkpoint. The manifests do not pin that identity. No weights or model cache are bundled, and this MoE model is not a dense-attention control.

## Startup correction and remaining limitations

| Recorded observation | Original `base` | Corrected `base-util082` |
| --- | ---: | ---: |
| GPU memory utilization | 0.85 | 0.82 |
| Peak activation reported at startup | 17.07 GiB | 0.99 GiB |
| Reported KV pool tokens | 2,048,645 | 3,197,331 |
| c4 output throughput | 162.2 tok/s | 256.8 tok/s |

The lower utilization setting accompanies a larger pool, so startup conditions changed beyond the intended flag. This undermines the older claim that Qwen's c4 slowdown proved an architectural routing weakness. It does not prove which allocation was recoverable or that preemption caused the observed slowdown. The context/prefix arm still uses the original pool, and the original MTP pair retains startup/pool confounds.

The warm replacement c1 results in `base` and `mtp-n1` do not make all points repeat-controlled or erase the pool differences. Original cold c1 observations remain quarantined in the source and are recorded as exclusions in [arms.json](../arms.json).

The saved engine manifest may describe named `attn`, `ffn`, and `unembed` estimators as full coverage. Those names do not establish accounting for every GDN/QSA/PLE/state operation and are not hardware counters. Operator bottlenecks, phase-only throughput, and quality parity remain unmeasured in this bundle.

## Supporting startup logs

| Session | Log |
| --- | --- |
| Original TP8 base / smaller pool | [07:31 startup](logs/serve_base-tp8_20260902-073122.log) |
| Corrected TP8 batch / warm startup | [09:10 startup](logs/serve_base-tp8_20260902-091050.log) |
| Original TP8 MTP n1 | [08:00 startup](logs/serve_mtp1-tp8_20260902-080051.log) |

These are sanitized full log exports. Use them with result dates and manifests, while retaining uncertainty about per-point configuration identity. TP4 probes, download logs, hub cache files other than the config, and superseded launch narratives are omitted.

## If Qwen is requested later

Pin the checkpoint/runtime and obtain a fresh MTP-off baseline with matched startup and explicit pool accounting. Repeat context, prefix controls, and MTP with the same pool/startup protocol before comparing axes. Map and time GDN state operations, QSA indexing/selection and core attention, embedding/PLE work, MoE, communication, and speculative state management where applicable. Do not start that work automatically after the DeepSeek study.
