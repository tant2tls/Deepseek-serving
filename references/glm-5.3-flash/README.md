# GLM-5.3-Flash historical reference

**September archive; separate from the October comparison.** The [October GLM measurements](../../reports/blog-architecture-h100-v1/findings.md) use a different build and setup. Do not merge these observations with them.

This bundle preserves 42 result JSONs, their 42 benchmark logs, six manifests, and seven selected startup logs from `LLMs_frontier_serving/GLM-5.3-Flash`. Numerical JSONs are unchanged. Text exports and original hashes are recorded in [provenance.json](../provenance.json). See [RESULTS.md](RESULTS.md) for every retained point and the [shared methodology and caveats](../README.md).

## Select the correct arm

| Arm | Points | Intended use | Boundary |
| --- | ---: | --- | --- |
| [bf16kv](results/bf16kv/) | 15 | Main batch, context, and prefix baseline | Eight batch points span sessions; the later arm manifest is not a per-point snapshot. |
| [bf16kv-mtp-n1](results/bf16kv-mtp-n1/) | 4 | MTP one-token batch comparison against `bf16kv` at c1/c4/c16/c64 | Historical single observations; fresh matched repeats still required. |
| [bf16kv-mtp-n5](results/bf16kv-mtp-n5/) | 4 | MTP five-token batch comparison against the same base and n1 grids | More accepted tokens per round do not establish net speedup. |
| [bf16kv-mtp-n1-context](results/bf16kv-mtp-n1-context/) | 4 | One-token MTP context comparison against `bf16kv` context | Separate startup session; preserve dates and state/pool differences. |
| [bf16kv-fi618](results/bf16kv-fi618/) | 4 | BF16 control for the alternate software stack, batch only | Experimental overlay/version-check bypass and DeepGEMM MoE path. |
| [fp8kv-fi618](results/fp8kv-fi618/) | 11 | Alternate FP8 stack, batch/context/prefix | Compare batch to `bf16kv-fi618`; no corresponding alternate-stack BF16 context/prefix controls are included. Numerical parity is unverified. |

Do not compare alternate FP8 directly to the original BF16 arm and call the full difference a KV-dtype effect. At c64 the retained output rates are 447.1 tok/s for original BF16, 348.8 for alternate-stack BF16, and 262.0 for alternate-stack FP8. Both the software/backend change and the dtype-associated path change matter. These measurements do not isolate storage bytes alone.

## Recorded deployment and architecture context

- Model ID: `zai-org/GLM-5.3-Flash`; eight H100 80GB HBM3 GPUs, TP8 and EP enabled. vLLM `0.1.dev20051+g487ecf187`, PyTorch `2.13.0+cu130`; see each arm's manifest for the historical environment.
- Main baseline: text-only requests, MTP off, GPU memory utilization 0.82, maximum model length 262,144, maximum sequences 256, CUDA graph capture limit 256. Historical weight metadata says FP8 attention/dense and experts; this shorthand does not mean all tensors are FP8. Main KV is BF16.
- Historical parser settings are `glm45` for reasoning and `glm47` for tools. Requested cache block size 128 resolves to a recorded attention block size of 640; keep the requested/resolved distinction and recurrent-state grouping. The alternate FP8 manifest records 1,152. These are layout/grouping properties, not efficiency ratios.
- The source guide describes 34 recurrent KDA and 11 sparse-attention layers, 288 routed experts, and eight selected experts per token. These are inherited architecture notes, not a new tensor/config audit; no standalone GLM checkpoint config was found in the selected source folders. Verify the exact checkpoint before future component work.
- Immutable model revisions are not recorded in the retained manifests. Do not infer a revision from the model name or date.

## Corrections to retain

The original FP8 route failed with `pe_dim must be 64 for fp8_ds_mla`. That demonstrates an incompatible layout for the tested NoPE model/backend, not that every FP8 KV path on H100 is impossible. The later FlashInfer 0.6.18 overlay enabled an alternate path and selected DeepGEMM after a different API mismatch. The historical scripts disabled version checking and mixed package versions. These results are experimental evidence, not a deployment recommendation.

GLM's MTP n5 c64 [result](results/bf16kv-mtp-n5/batch_isl16k_c64.json) reports approximately 30.1% draft acceptance, 2.51 acceptance length, 99.2% peak cache-pool occupancy, and 408.3 output tok/s, below the [base](results/bf16kv/batch_isl16k_c64.json) at 447.1. This motivates separating draft, verification, and state costs; it does not identify the dominant overhead. The saved acceptance length is not a component timing or necessarily the same definition as useful committed tokens per round in a future harness.

The old harness has the shared telemetry/validation limitations documented in the [reference index](../README.md). Particularly, recorded engine FLOP/byte estimates omit attention, missing telemetry can appear as zero, and the finer batch grid combines sessions. One manifest captures a shell wrapper instead of the effective server command; consult startup logs for resolved flags, and treat per-point configuration identity as incomplete.

## Supporting startup logs

These are sanitized full log exports, not profiler traces. Timestamps support the source narrative; they do not supply immutable run IDs for every point.

| Session | Log |
| --- | --- |
| Early main baseline | [01:38 startup](logs/serve_bf16kv_20260902-013803.log) |
| Later main baseline / finer grid | [10:43 startup](logs/serve_bf16kv_20260902-104317.log) |
| MTP n1 batch | [05:11 startup](logs/serve_bf16kv-mtp-n1_20260902-051151.log) |
| MTP n1 context | [11:12 startup](logs/serve_bf16kv-mtp-n1_20260902-111207.log) |
| MTP n5 batch | [05:28 startup](logs/serve_bf16kv-mtp-n5_20260902-052824.log) |
| Alternate BF16 stack | [04:51 startup](logs/serve_bf16kv-fi618_20260902-045116.log) |
| Alternate FP8 stack | [04:04 startup](logs/serve_fp8kv-fi618_20260902-040441.log) |

Quarantined warmup/queueing arms, the supplemental `util085` sweep, and the failed original `fp8kv` route are listed in [arms.json](../arms.json), with the originals preserved outside this repo. Superseded narratives and historical launch wrappers are not imported.

Historical commands describe prior experiments, not an active run queue. New GPU, prefix or speculative experiments require explicit authorization.
