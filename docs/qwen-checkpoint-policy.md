# Qwen3.8-Flash-Next: original BF16 checkpoint only for future work

**Decision: Tan, 2026-10-08.** From now on, every new Qwen3.8-Flash-Next run uses the original **BF16 checkpoint**, `Qwen/Qwen3.8-Flash-Next`. Do not use the `-FP8` checkpoint, quantize the original weights to FP8, or fall back to FP8 if native BF16 does not fit or is unsupported. The decision itself did not authorize a GPU session. **Tan authorized one later the same day**, and the BF16 checkpoint was measured in study `qwen-bf16-h100-v1`: [findings](../reports/qwen-bf16-h100-v1/findings.md), [handoff](../reports/qwen-bf16-h100-v1/handoff.md).

## Identities and evidence boundaries

| Role | Checkpoint | Immutable revision | Status |
| --- | --- | --- | --- |
| All Qwen work from 2026-10-08, key `qwen-38-bf16` | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Original BF16; loaded and measured on 2026-10-08 in `qwen-bf16-h100-v1` (vLLM `554340f3…`) |
| Completed October blog study, key `qwen-38` | `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` | Historical FP8 results only; no new FP8 runs |

The original repository and revision were checked on 2026-10-08 through the [official model metadata](https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next). The metadata identifies BF16 floating-point weight tensors; the [config at the pinned revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) declares `text_config.dtype = bfloat16` and contains no weight-quantization configuration. This is source evidence. Execution evidence for vLLM `554340f3…` is the server log of `qwen-bf16-h100-v1`: `dtype=torch.bfloat16`, `quantization=None`, `Using TRITON Unquantized MoE backend`, and a BF16 n-gram embedding (`Qwen4ExpPLEUnquantizedEmbeddingMethod`, `weight_dtype=torch.bfloat16`). Re-check those lines on any other build.

Pin the tokenizer and any remote code to the same original-checkpoint revision. Do not reuse the FP8 revision for the BF16 repository. Model dtype, weight quantization and KV/state dtype are separate: adding `--dtype bfloat16` to the FP8 checkpoint does not select the original BF16 weights, and a BF16 weight policy does not by itself establish every cache/state dtype.

## Existing results stay FP8

**Article naming:** the main model is displayed as **Qwen3.8-Flash-Next**, or **Qwen** in compact charts. That name refers to the original BF16 checkpoint; precision and physical-node details are documented in the measurement notes, not appended to the name. All five deployments share the core serving configuration and workload, with the recorded node and model-specific exceptions preserved.

All Qwen values in `blog-architecture-h100-v1` came from the FP8 checkpoint. Keep the original plan, addendum, manifests, raw results, CSVs and provenance unchanged. Wherever those historical values appear, display the model as **Qwen3.8-Flash-Next (FP8)** or **Qwen FP8**. Its throughput, memory, kernel selections and ranking must not be attributed to BF16, which has its own measured evidence.

**Blog selection, Tan's instruction of 2026-10-09:** the main article, charts and takeaways now use **Qwen BF16** from `qwen-bf16-h100-v1`; Qwen FP8 is **historical reference only**. The other four main deployments come from the first-node study. Follow [blog_target.md section 11](../blog_target.md#11-current-blog-evidence-selection-2026-10-09): label the second node, show the MiMo control, compare throughput across nodes, and keep BF16 latency separate from first-node rankings. This does not exclude GLM's native FP8 deployment or DeepSeek's native FP8 KV. It selects the original Qwen checkpoint for the article.

Keep the older [FP8 reference bundle](../references/qwen3.8-flash-next-fp8/README.md) as historical evidence. Retaining records does not authorize new FP8 reproduction runs. BF16 results live in their own study; never append BF16 points to an FP8 curve or overwrite a completed study's identity.

## How BF16 is run (wired 2026-10-08)

1. **Key and study.** `qwen-38-bf16` in [serve.sh](../bench/serve.sh) and [run_matrix.py](../bench/run_matrix.py); study `qwen-bf16-h100-v1`, selected with `export BLOG_STUDY=qwen-bf16-h100-v1` for [blog_study.py](../bench/blog_study.py), [blog_launch.sh](../bench/blog_launch.sh), [blog_inputs.py](../bench/blog_inputs.py) and [blog_report.py](../bench/blog_report.py). Without that variable the tools address the completed `blog-architecture-h100-v1`.
2. **The FP8 key is retired, not repointed.** `qwen-38` still names the FP8 checkpoint as a record. `serve.sh` refuses it unless `ALLOW_HISTORICAL_QWEN_FP8=1` is set, and the chain refuses it always. Do not set that variable for new work.
3. **Deployment.** The common one: TP8 + EP, utilization 0.90, max context 262144, max sequences 64, chunk budget 8192, text only, AR only, prefix caching off. No `--dtype` and no quantization flag: native precision comes from the checkpoint. Parsers `qwen3` / `qwen3_xml`, thinking off with `enable_thinking=false`.
4. **Same inputs.** The request lists are rebuilt on the node and must equal the published lists of the blog study hash for hash; the dry run checks it. The BF16 tokenizer gives the same token count as the FP8 one on all 543 requests.
5. **Controls.** No FP8 run may be repeated, so a BF16-versus-FP8 difference is a comparison of two deployments measured on different nodes and days. One previously measured model (MiMo) runs one timing block in the BF16 study as a **node control**; read it before placing BF16 numbers next to the 2026-10-07 tables.
6. **If native BF16 fails** on a build, record the unsupported or failed point. Do not substitute FP8 and do not tune topology or precision inside the study.

The later three-model speculative study remains separate; this checkpoint decision does not expand its matrix or authorize prefix, speculative or H200 work.

No weights, credentials, private paths or large traces are stored in this repository.
