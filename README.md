# Five-model serving measurements on 8×H100

**Current blog selection (2026-10-09):** use the latest completed measurements of **DeepSeek V4 Flash 0731, DeepSeek V4.1 Flash, MiMo-V2.6-Flash-MOPD, Qwen3.8-Flash-Next, and GLM-5.3-Flash**. **Qwen FP8 is historical reference only**, excluded from the main figures and rankings. Start with [the blog](index.html), [the source-selection rules](target.md#11-current-blog-evidence-selection-2026-10-09), [BF16 findings](reports/qwen-bf16-h100-v1/findings.md), and [first-node findings](reports/blog-architecture-h100-v1/findings.md).

In the article, **Qwen3.8-Flash-Next** means the original BF16 checkpoint; compact chart labels use **Qwen**. Precision and node details belong in the measurement notes, rather than the model name.

Four main deployments come from `blog-architecture-h100-v1`, measured on the first 8×H100 80GB node on October 7–8. Qwen comes from `qwen-bf16-h100-v1`, measured on October 8 on a **second** 8×H100 node with MiMo as a node control. These are two completed studies, not five models measured on one node. No further GPU work is authorized by this editorial update.

**All five models were measured with the same core serving setup:** vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`), TP8 + expert parallel, utilization 0.90, max context 262144, max sequences 64 and an 8192-token prefill chunk budget. They used the same public code/math/chat request lists and counts, approximately 1K/16K/64K input, 256 forced output tokens, c1/c8 and three timing blocks. Prefix caching and speculation were off. The [first-node addendum](reports/blog-architecture-h100-v1/plan_addendum_glm_qwen.json) and [BF16 plan](reports/qwen-bf16-h100-v1/plan.json) record the settings and rules.

| Main blog deployment | Node | Valid timing runs | Diagnostic captures | Live-KV snapshots |
| --- | --- | ---: | ---: | ---: |
| DeepSeek V4 Flash 0731 | First | 18/18 | 6 | 6 |
| DeepSeek V4.1 Flash | First | 18/18 | 6 | 6 |
| MiMo-V2.6-Flash-MOPD | First | 18/18 | 6 | 6 |
| Qwen3.8-Flash-Next | Second | 18/18 | 6 | 6 |
| GLM-5.3-Flash | First | 18/18 | 6 | 6 |
| **Main article total** | **Two nodes** | **90/90** | **30** | **30** |

Separately: the second-node MiMo control has 6 timing runs, 6 traces and 6 snapshots; historical Qwen FP8 has 18 timing runs, 6 traces and 6 snapshots. These article counts select existing rows and do not change either study's frozen completion ledger.

**Comparison limits:** GLM has no thinking-off switch and used `reasoning_effort=low`; the other four disabled thinking. GLM followed the first three on the first node, and Qwen ran on the second node. Launch order was not rotated across all five. Native weight/KV precision, kernel paths and tokenizer counts differ, and some timing windows were shorter than the intended 60 seconds. The MiMo control supports a **throughput-only comparison across nodes**, with differences under about 2% unresolved; BF16 TTFT and eight-client TPOT must not enter first-node latency rankings. Hardware HBM counters and expert-routing statistics were not collected.

**Latest serving results:** output tok/s, mean of three blocks, vLLM `554340f3d3259e321be4c07282be7a02a5aeef83`. Four columns are first-node measurements; Qwen is second-node. The last column shows the separately measured MiMo control, without adjusting any model's values. SDs and individual blocks: [first-node tables](reports/blog-architecture-h100-v1/serving_tables.md), [BF16 tables](reports/qwen-bf16-h100-v1/serving_tables.md), [node control](reports/qwen-bf16-h100-v1/comparison.md).

| Input / clients | 0731 | V4.1 | MiMo | Qwen | GLM | MiMo control, second / first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K / c1 | 118.1 | 106.0 | **160.6** | 154.7 | 137.3 | 0.989 |
| 1K / c8 | 598.7 | 562.9 | 687.7 | **793.0** | 705.8 | 0.988 |
| 16K / c1 | 93.4 | 89.3 | 114.3 | **125.4** | 109.4 | 0.986 |
| 16K / c8 | 237.5 | 278.9 | 250.9 | **385.5** | 322.3 | 0.995 |
| 64K / c1 | 48.5 | 55.9 | 55.4 | **77.5** | 66.4 | 0.983 |
| 64K / c8 | 71.5 | 100.5 | 78.5 | **142.5** | 115.7 | 0.987 |

Within this selected five-deployment comparison, Qwen leads throughput at five of six points; MiMo leads at 1K/c1. The original BF16 checkpoint loads natively with unquantized `TRITON` experts, 31.42 GiB of weights and a 37.37 GiB KV pool per GPU on the second node, vLLM `554340f3…`. Use its own [findings](reports/qwen-bf16-h100-v1/findings.md) and [handoff](reports/qwen-bf16-h100-v1/handoff.md) for Qwen's runtime and memory explanation.

**Qwen FP8 reference:** the first-node FP8 measurements are preserved in the [historical findings](reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash) and [BF16-versus-FP8 comparison](reports/qwen-bf16-h100-v1/comparison.md). They are not the main article's Qwen baseline and must not be relabelled BF16. No new FP8 runs, conversion or fallback; see the [checkpoint policy](docs/qwen-checkpoint-policy.md). GLM's native FP8 deployment remains in the main comparison.

**Earlier completed studies** compare DeepSeek V4 Flash 0731, V4.1 Flash and MiMo-V2.6-Flash-MOPD on the older vLLM build `44af287e…`; see the [DeepSeek report](reports/v41-vs-0731/report.md) and the [MiMo report](reports/mimo-v26/report.md). Their numbers must not be mixed with the October five-model study. The separate later speculative study `spec-realtext-h100-v1` remains **planned, not measured**, with those three models only. See [target.md section 12](target.md#12-later-study-spec-realtext-h100-v1), the [checked evidence review](update.md) and the [later GPU-session plan](docs/three-model-h100-plan.md). That plan's three-model scope does not limit the completed five-model blog comparison.

The old V4 measurements used the **preview** model ID, with exact revision unverified. They are historical context, not 0731 results. The [speculative support record](docs/speculative-decoding.md) separates classic preview MTP from native DSpark. **All new prefix-cache experiments are deferred; prefix caching stays off in the active phase.** Existing prefix results are preserved. H200 is a separate optional future study.

This repository also contains a curated, portable subset of the earlier `LLMs_frontier_serving` study under [references/](references/README.md). Its September GLM and Qwen archives used different builds/settings and remain historical context. The current article uses October GLM from `blog-architecture-h100-v1` and Qwen from `qwen-bf16-h100-v1`.

| Need | File |
| --- | --- |
| Next agent's scope | [AGENTS.md](AGENTS.md) |
| **Target: Kan's questions, the measurement plan and what is answered so far** (attention, KV memory, FFN sparsity) | [target.md](target.md) |
| **Current architecture blog** | [article](index.html) · [source selection and comparison rules](target.md#11-current-blog-evidence-selection-2026-10-09) · [BF16 findings](reports/qwen-bf16-h100-v1/findings.md) |
| First-node study on vLLM `554340f3…` (four main deployments plus Qwen FP8 reference) | [findings](reports/blog-architecture-h100-v1/findings.md) · [handoff](reports/blog-architecture-h100-v1/handoff.md) · [architecture](reports/blog-architecture-h100-v1/architecture.md) |
| **GLM-5.3-Flash: measured in the same October environment** | [five-model comparison](reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash) · [GLM run evidence](reports/blog-architecture-h100-v1/data/glm-53/) |
| **Qwen3.8-Flash-Next FP8: measured in the same October environment** (history; no new FP8 runs) | [five-model comparison](reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash) · [Qwen run evidence](reports/blog-architecture-h100-v1/data/qwen-38/) |
| **Qwen3.8-Flash-Next original BF16: measured 2026-10-08 on a second node, with a MiMo node control** | [findings](reports/qwen-bf16-h100-v1/findings.md) · [handoff](reports/qwen-bf16-h100-v1/handoff.md) · [BF16 against FP8 and the control](reports/qwen-bf16-h100-v1/comparison.md) · [run evidence](reports/qwen-bf16-h100-v1/data/qwen-38-bf16/) · [checkpoint policy](docs/qwen-checkpoint-policy.md) |
| Does vLLM time V4.1 prefill as its report describes? | [v41_prefill_check.md](reports/blog-architecture-h100-v1/v41_prefill_check.md) |
| **Per-model setup guide** (download, launch, expected log lines, request, pitfalls) | [docs/model-setup.md](docs/model-setup.md) |
| Fresh-node setup, serving skill, step-by-step lessons | [docs/reproduce.md](docs/reproduce.md) §0 and §0.1 · [.claude/skills/serving/](.claude/skills/serving/SKILL.md) · [teach_me/](teach_me/README.md) (lesson 6: a new checkpoint is a new study) |
| **V4.1 vs 0731 measured results (speculation off + DSpark)** | [reports/v41-vs-0731/report.md](reports/v41-vs-0731/report.md) |
| **MiMo-V2.6-Flash vs 0731 vs V4.1 (speculation off + MTP/DFlash)** | [reports/mimo-v26/report.md](reports/mimo-v26/report.md) |
| How MiMo runs prefill/decode, and why it is faster | [docs/mimo-v2.6-inference.md](docs/mimo-v2.6-inference.md) |
| Reproduce a study / add a model | [docs/reproduce.md](docs/reproduce.md) |
| Benchmark/profiling harness | [bench/](bench/) |
| **Every study in one place** (report, tables, frozen plans and curated run evidence per study) | [reports/README.md](reports/README.md) |
| Separate later three-model speculative objective and evidence rules | [target.md section 12](target.md#12-later-study-spec-realtext-h100-v1) |
| Reviewed findings, corrections and experiment rationale | [update.md](update.md) |
| Later speculative GPU study: prerequisites, staged counts and handoff | [docs/three-model-h100-plan.md](docs/three-model-h100-plan.md) |
| Checkpoint identity and MTP/DSpark experiments | [docs/speculative-decoding.md](docs/speculative-decoding.md) |
| Historical evidence, selection, and shared caveats | [references/README.md](references/README.md) |
| Historical DeepSeek V4 preview | [DeepSeek reference](references/deepseek-v4-flash/README.md) |
| September GLM archive (different runtime/settings) | [GLM reference](references/glm-5.3-flash/README.md) |
| September Qwen FP8 archive (different runtime/settings) | [Qwen reference](references/qwen3.8-flash-next-fp8/README.md) |
| Verify the curated bundle locally | [tools/audit_references.py](tools/audit_references.py) |

`install.sh`, `run.sh`, and `request.sh` are the Linux installation, V4.1 launch, and smoke-request scaffolds. The matched benchmark harness is in `bench/`. **September results on the older runtime:** the V4.1 vs 0731 study is complete for speculation off (concurrency, context, prefix cache states, isolated prefill/decode, prefill kernel breakdown) and DSpark fixed/adaptive at c1/4/16/64; see the [DeepSeek report](reports/v41-vs-0731/report.md). In that study, 0731 is 1.1–1.35× faster uncached (V4.1 prefill kernel paths, no CED), V4.1 wins 1.05–2.24× with prefix caching (reuse after one computation vs two), and DSpark helps only at low concurrency.

**September MiMo-V2.6-Flash-MOPD results** (study `mimo-v26`, same node, older runtime and workloads as the DeepSeek study above): MiMo is 1.27–1.42× faster than 0731 and 1.43–1.73× faster than V4.1 uncached. It skips DeepSeek's sparse-attention/indexer, mHC, and heavy projections; all-reduce is its largest sampled prefill kernel component on that runtime. Details: the [MiMo report](reports/mimo-v26/report.md).

The GitHub bundle includes small result JSONs, benchmark logs, selected startup logs, manifests, corrected summaries, and source/export SHA-256 provenance. Numeric result JSONs are preserved byte-for-byte. Exported manifests/logs may have private host/path information and terminal formatting removed; [provenance.json](references/provenance.json) records each transformation. Caches, weights, presentations, old executable launch wrappers, quarantined runs, and superseded narratives are excluded with reasons in the reference index. Original files remain in the source repository.

Validate from this repository root with Python 3.10 or newer (standard library only):

```bash
python tools/audit_references.py
```

The imported references are a bounded historical snapshot, not a copy of every result from the older repository. The October five-model study has its own evidence and manifests under `reports/blog-architecture-h100-v1/`. New GPU evidence belongs to a separate study ID and must preserve model/runtime/workload provenance.
