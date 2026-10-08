# Qwen3.8-Flash-Next: original BF16 checkpoint only for future work

**Decision: Tan, 2026-10-08.** From now on, every new Qwen3.8-Flash-Next run uses the original **BF16 checkpoint**, `Qwen/Qwen3.8-Flash-Next`. Do not use the `-FP8` checkpoint, quantize the original weights to FP8, or fall back to FP8 if native BF16 does not fit or is unsupported. This decision updates checkpoint selection; it does not authorize a GPU rental or a new measurement session.

## Identities and evidence boundaries

| Role | Checkpoint | Immutable revision | Status |
| --- | --- | --- | --- |
| Future Qwen work | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Original BF16; source metadata verified, local runtime support and measurements pending |
| Completed October blog study, key `qwen-38` | `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` | Historical FP8 results only; no new FP8 runs |

The original repository and revision were checked on 2026-10-08 through the [official model metadata](https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next). The metadata identifies BF16 floating-point weight tensors; the [config at the pinned revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) declares `text_config.dtype = bfloat16` and contains no weight-quantization configuration. This is source evidence, not proof that a particular vLLM build loads or executes it correctly.

Pin the tokenizer and any remote code to the same original-checkpoint revision. Do not reuse the FP8 revision for the BF16 repository. Model dtype, weight quantization and KV/state dtype are separate: adding `--dtype bfloat16` to the FP8 checkpoint does not select the original BF16 weights, and a BF16 weight policy does not by itself establish every cache/state dtype.

## Existing results stay FP8

All Qwen values in `blog-architecture-h100-v1` came from the FP8 checkpoint. Keep the original plan, addendum, manifests, raw results, CSVs and provenance unchanged. Display the model as **Qwen3.8-Flash-Next (FP8)** or **Qwen FP8** in the existing blog and tables. Its throughput, memory, kernel selections and ranking must not be attributed to the unmeasured BF16 deployment.

Keep the older [FP8 reference bundle](../references/qwen3.8-flash-next-fp8/README.md) as historical evidence. Retaining records does not authorize new FP8 reproduction runs. A future BF16 comparison needs a separate study ID and fresh matched AR controls; never append BF16 points to an FP8 curve or overwrite a completed study's identity.

## What must happen before a BF16 run

The current `qwen-38` entries in [serve.sh](../bench/serve.sh) and [run_matrix.py](../bench/run_matrix.py) still identify the archived FP8 checkpoint. [blog_study.py](../bench/blog_study.py) also fixes the completed study ID. **Do not invoke those Qwen launch/chain entries for new work.** This documentation change does not claim that a BF16 harness is ready.

1. Prepare a distinct BF16 model key (proposed `qwen-38-bf16`) and a fresh study manifest/output directory. Do not repoint the historical `qwen-38` key or substitute weights inside a frozen plan.
2. Preserve the common deployment: TP8 + EP, utilization 0.90, max context 262144, max sequences 64, chunk budget 8192, native BF16 weights, text only, AR only, prefix caching off. Pin the runtime; verify parser/template behavior and `enable_thinking=false` from the original checkpoint.
3. Verify the actual loaded weight dtype and absence of FP8 weight quantization, plus dense/expert kernels, graph settings, per-layer KV/state formats, resident weights and usable pool. Existing FP8 memory figures, launch times and kernel choices do not establish BF16 behavior or fit.
4. After an authorized node is supplied, run support/functional checks and a disjoint pilot, then freeze the counts, launch order, budget and controls before collecting. If native BF16 fails, record the unsupported/failed point; do not silently substitute FP8 or tune topology/precision.

**Pending:** BF16 harness wiring and dry run, verified runtime execution, functional checks, pilot, serving timings, profiles and live-memory measurements. The later three-model speculative study remains separate; this checkpoint decision does not expand its matrix or authorize prefix, speculative or H200 work.

Only public source metadata was consulted for this note. No weights, credentials, private paths, new GPU results or large traces were added.
