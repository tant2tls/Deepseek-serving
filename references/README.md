# Curated historical evidence

**This directory preserves the September reference archive.** The completed October architecture study includes fresh measurements of DeepSeek V4 Flash 0731, V4.1 Flash, MiMo-V2.6-Flash-MOPD, GLM-5.3-Flash and Qwen3.8-Flash-Next FP8 on the same node/runtime/core settings; see the [five-model findings](../reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash). The older GLM/Qwen points below used different builds/settings and are not controls for that comparison. Qwen work now follows the [original-BF16-only policy](../docs/qwen-checkpoint-policy.md); the BF16 checkpoint was measured on 2026-10-08 in [`qwen-bf16-h100-v1`](../reports/qwen-bf16-h100-v1/findings.md), on another node than the five-model study.

The historical V4 bundle below is from the preview model ID, with exact revision unverified; it is not a 0731 baseline. See the [checkpoint/speculation audit](../docs/speculative-decoding.md). These references preserve evidence; they are not a queue of experiments to run. The later speculative-study scope remains in [target.md](../target.md).

## Included study arms

All imported measurement points below are from September 2, 2026. Each model guide identifies the correct comparison and its limitations.

| Model | Retained arms | Use |
| --- | --- | --- |
| [DeepSeek V4](deepseek-v4-flash/README.md) | `mtp-off-image` | Historical baseline context for the active DeepSeek study |
| [GLM-5.3-Flash](glm-5.3-flash/README.md) | `bf16kv`, `bf16kv-mtp-n1`, `bf16kv-mtp-n5`, `bf16kv-mtp-n1-context`, `bf16kv-fi618`, `fp8kv-fi618` | Main baseline, MTP controls, and separately labeled experimental alternate-stack controls |
| [Qwen3.8-Flash-Next-FP8](qwen3.8-flash-next-fp8/README.md) | `base-util082`, `base`, `mtp-n1` | Corrected batch; qualified original context/prefix and original MTP pair |

[arms.json](arms.json) records roles and exclusions. [results.csv](results.csv) is a machine-readable extraction of retained measurements with model, arm, workload, source path, saved date, and units. Blank cells mean absent, not zero. GLM and Qwen also have per-arm tables linked from their guides. Do not combine these arms into a single performance ranking.

## Shared method and interpretation

- Historical hardware: eight H100 80GB HBM3 GPUs per retained deployment, TP8 with EP enabled. The manifests record PyTorch `2.13.0+cu130`; GLM and the retained DeepSeek arm use vLLM `0.1.dev20051+g487ecf187`, while Qwen uses `0.1.dev20073+g8e685d198`. They are not universally engine-matched.
- Workloads used `vllm bench serve`, `openai-chat`, `/v1/chat/completions`, unlimited offered rate with a client concurrency cap, and forced 256-token outputs via `--ignore-eos`. Input counts include tokenizer/template effects. These synthetic finite batches are not traffic replay or equal-quality task comparisons.
- Batch uses 16,384 target input tokens. Main concurrency points are 1, 4, 16, 64; GLM's baseline also retains 2, 8, 32, 48 from a later session. Context uses 16,384 / 65,536 / 131,072 / 260,000 target input tokens at concurrency 8. Batch/context seeds use `(input_length % 100000) + concurrency * 7919 + output_length * 31`.
- Historical request counts use `min(max(8, 2 * concurrency), max(8, floor(8000000 / input_length)))`. Prefix workloads use 65,536 shared + 2,048 suffix tokens, 1/4/16 prefixes, 64 requests, concurrency 8, and seed `4200 + prefix_count`. Prefix comparisons lack matched cache-off and prewarmed controls.
- Output throughput is `total_output_tokens / duration`, including input processing and scheduling time. TTFT includes queueing; TPOT is a per-request average, not individual inter-token latency or aggregate decode throughput. Client concurrency is not fixed engine batch size.
- Saved summaries provide medians, means, and p99, not repeat-based p95 estimates or confidence intervals. They generally retain one observation per point. GLM's finer grid spans sessions; an arm-wide manifest can postdate some points. A log with a nearby timestamp is supporting evidence, not a verified immutable per-point configuration ID.
- `peak_kv_cache_usage_perc` is a fraction of the allocated engine cache pool; multiply by 100 for percent. It is not total GPU memory. High occupancy alone does not prove preemption. Prefix points lack some cache telemetry.
- Saved `cold_run=true` and zero cache-hit deltas are not independently validated: the old harness could replace missing telemetry with zero, retain warned cache hits or partial completions, and overwrite manifests while skipping result files. Collection success is not scientific validation.
- FLOP/byte counters are partial engine estimates, not hardware counters or guaranteed lower bounds. GLM's recorded estimators omit attention; DeepSeek's omit much of the model. Qwen's named `attn/ffn/unembed` coverage does not establish correct accounting for every hybrid/state/PLE operation. Raw manifests retain historical overclaims about these counters; this correction governs interpretation.
- No retained component trace establishes a dominant operator bottleneck. No dense control, full quality comparison, or repeat-based causal finding is supplied. Fresh matched measurements and profiling are required under [target.md](../target.md).

## Curation and provenance

[provenance.json](provenance.json) lists every imported file with its path relative to `LLMs_frontier_serving`, source SHA-256, export SHA-256, byte counts, and transformations. It also hashes the source documents used for the corrected guides. Numerical JSONs are unchanged. Private cluster roots, host identifiers, private IP addresses, temporary session identifiers, and terminal control sequences are removed from exported logs/manifests where encountered; each file records the actual transformation counts. Public model IDs, software versions, flags, numeric metrics, and timestamps are retained. Sanitized historical command lines are documentary, not executable recipes.

The provenance ledger includes the existing DeepSeek bundle. Its manifest is now a sanitized export; benchmark JSONs remain identical to the source. The original repository has not been modified. Hash checks verify identity, not truthfulness or validity of a measurement.

The imported Qwen `config.json` is a small configuration extracted from the cached snapshot, not model weights. Its snapshot directory identifies a config revision; the historical benchmark manifests do not prove that every run used that revision. GLM has no standalone checkpoint config in the selected source folders, so do not imply an equivalent pinned config is available.

## Deliberate exclusions

- GLM `_discarded-warmup-contaminated`, `_mtpctx_queued_firstrun`, `_util085_queued_firstrun`, and Qwen `_base_c1_jitcold`, `_mtp_c1_jitcold`: quarantined observations. Original evidence remains in the source; retain their exclusion reasons rather than promoting them as baselines.
- GLM `util085`: supplemental utilization experiment outside this curated selection. GLM `fp8kv`: failed original route, with no accepted performance point; the failure and alternate-path caveat are summarized in the GLM guide.
- DeepSeek's other historical arms: outside the existing `mtp-off-image` baseline selection. The active task requires fresh V4/V4.1 measurements.
- Old launch/submission/send scripts: tied to historical clusters, cache locations, package overlays, or superseded causal claims. Relevant flags remain in manifests and startup logs. No old package/version-check bypass is promoted as a supported deployment recipe.
- Old `history/`, `RESULT-*.md`, root narratives, decks, screenshots, and archives: duplicate or superseded presentation material. Relevant corrections and methods are distilled here instead of copying broken cross-references or old agent instructions.
- Weights, hub blobs, general download caches, broad debug/console logs, and failed TP4 experiments: unnecessary for this TP8 reference bundle. Only selected startup logs and the small Qwen config are retained.
- Later `new_run_report_real/` packages, root final reports, and the reported DeepSeek n5 package: outside the requested model-folder snapshot. This bundle does not claim to be the latest or complete inventory of the source repository. A reported n5 package was also recorded as missing locally in the source handoff; no unverified numbers are imported from it.

If GLM or Qwen is requested later, read its guide, choose compatible arms, and design fresh controls before drawing new comparisons. Until then, follow the active three-model H100 scope in [target.md](../target.md); historical experiment lists do not authorize new work, and all new prefix-cache experiments are deferred.
