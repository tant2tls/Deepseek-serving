# Five-model serving measurements on 8×H100

**Latest completed study (2026-10-07/08):** `blog-architecture-h100-v1` measured **DeepSeek V4 Flash 0731, DeepSeek V4.1 Flash, MiMo-V2.6-Flash-MOPD, GLM-5.3-Flash and Qwen3.8-Flash-Next FP8** on the same 8×H100 80GB node. Start with [the five-model blog](index.html), [findings](reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash), or [the completion ledger](reports/blog-architecture-h100-v1/handoff.md).

All five used vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`), TP8 + expert parallel, utilization 0.90, max context 262144, max sequences 64 and an 8192-token prefill chunk budget. They used the same public code/math/chat request lists and counts, approximately 1K/16K/64K input, 256 forced output tokens, c1/c8 and three timing blocks. Prefix caching and speculation were off. The [GLM/Qwen addendum](reports/blog-architecture-h100-v1/plan_addendum_glm_qwen.json) records the shared settings and rules.

| Measured deployment | Valid timing runs | Diagnostic captures | Live-KV snapshots |
| --- | ---: | ---: | ---: |
| DeepSeek V4 Flash 0731 | 18/18 | 6 | 6 |
| DeepSeek V4.1 Flash | 18/18 | 6 | 6 |
| MiMo-V2.6-Flash-MOPD | 18/18 | 6 | 6 |
| **GLM-5.3-Flash** | **18/18** | **6** | **6** |
| Qwen3.8-Flash-Next **FP8** | 18/18 | 6 | 6 |
| **Total** | **90/90** | **30** | **30** |

**Comparison limits:** GLM has no thinking-off switch and used `reasoning_effort=low`; the other four disabled thinking. GLM and Qwen ran after the first three in the same session, so launch order was not rotated across all five. Native weight/KV precision, kernel paths and tokenizer counts differ. Some timing windows were shorter than the intended 60 seconds; [findings section 8](reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash) records the deviations. These are measured deployment comparisons. Hardware HBM counters and expert-routing statistics were not collected.

**Qwen policy (2026-10-08):** all future Qwen work uses the original **BF16** checkpoint `Qwen/Qwen3.8-Flash-Next`; no new FP8 runs. The verified source pin, historical boundary and pending BF16 preparation are in [docs/qwen-checkpoint-policy.md](docs/qwen-checkpoint-policy.md). The table above preserves the completed FP8 measurements; BF16 performance has not been measured here.

**Earlier completed studies** compare DeepSeek V4 Flash 0731, V4.1 Flash and MiMo-V2.6-Flash-MOPD on the older vLLM build `44af287e…`; see [report.md](report.md) and [report_mimo.md](report_mimo.md). Their numbers must not be mixed with the October five-model study. The separate later speculative study `spec-realtext-h100-v1` remains **planned, not measured**, with those three models only. See [target.md](target.md), the [checked evidence review](update.md) and the [later GPU-session plan](docs/three-model-h100-plan.md). That plan's three-model scope does not limit the completed five-model blog comparison.

The old V4 measurements used the **preview** model ID, with exact revision unverified. They are historical context, not 0731 results. The [speculative support record](docs/speculative-decoding.md) separates classic preview MTP from native DSpark. **All new prefix-cache experiments are deferred; prefix caching stays off in the active phase.** Existing prefix results are preserved. H200 is a separate optional future study.

This repository also contains a curated, portable subset of the earlier `LLMs_frontier_serving` study under [references/](references/README.md). Its September GLM and Qwen archives used different builds/settings and remain historical context. The fresh October GLM and Qwen measurements above belong to `reports/blog-architecture-h100-v1/` and participate in the five-model comparison.

| Need | File |
| --- | --- |
| Next agent's scope | [AGENTS.md](AGENTS.md) |
| Architecture blog plan (attention, KV memory, FFN sparsity) | [blog_target.md](blog_target.md) |
| **Architecture blog results, five models on vLLM `554340f3…`** (attention, live KV, FFN sparsity) | [findings](reports/blog-architecture-h100-v1/findings.md) · [handoff](reports/blog-architecture-h100-v1/handoff.md) · [architecture](reports/blog-architecture-h100-v1/architecture.md) |
| **GLM-5.3-Flash: measured in the same October environment** | [five-model comparison](reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash) · [GLM run evidence](reports/blog-architecture-h100-v1/data/glm-53/) |
| **Qwen3.8-Flash-Next FP8: measured in the same October environment** | [five-model comparison](reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash) · [Qwen run evidence](reports/blog-architecture-h100-v1/data/qwen-38/) · [future BF16 policy](docs/qwen-checkpoint-policy.md) |
| Does vLLM time V4.1 prefill as its report describes? | [v41_prefill_check.md](reports/blog-architecture-h100-v1/v41_prefill_check.md) |
| **Per-model setup guide** (download, launch, expected log lines, request, pitfalls) | [docs/model-setup.md](docs/model-setup.md) |
| Fresh-node setup, serving skill, step-by-step lessons | [docs/reproduce.md](docs/reproduce.md) §0 · [.claude/skills/serving/](.claude/skills/serving/SKILL.md) · [teach_me/](teach_me/README.md) |
| **V4.1 vs 0731 measured results (speculation off + DSpark)** | [report.md](report.md) |
| **MiMo-V2.6-Flash vs 0731 vs V4.1 (speculation off + MTP/DFlash)** | [report_mimo.md](report_mimo.md) |
| How MiMo runs prefill/decode, and why it is faster | [docs/mimo-v2.6-inference.md](docs/mimo-v2.6-inference.md) |
| Reproduce a study / add a model | [docs/reproduce.md](docs/reproduce.md) |
| Benchmark/profiling harness | [bench/](bench/) |
| Curated V4.1 and 0731 run evidence | [reports/v41-vs-0731/](reports/v41-vs-0731/) |
| Separate later three-model speculative objective and evidence rules | [target.md](target.md) |
| Reviewed findings, corrections and experiment rationale | [update.md](update.md) |
| Later speculative GPU study: prerequisites, staged counts and handoff | [docs/three-model-h100-plan.md](docs/three-model-h100-plan.md) |
| Checkpoint identity and MTP/DSpark experiments | [docs/speculative-decoding.md](docs/speculative-decoding.md) |
| Historical evidence, selection, and shared caveats | [references/README.md](references/README.md) |
| Historical DeepSeek V4 preview | [DeepSeek reference](references/deepseek-v4-flash/README.md) |
| September GLM archive (different runtime/settings) | [GLM reference](references/glm-5.3-flash/README.md) |
| September Qwen FP8 archive (different runtime/settings) | [Qwen reference](references/qwen3.8-flash-next-fp8/README.md) |
| Verify the curated bundle locally | [tools/audit_references.py](tools/audit_references.py) |

`install.sh`, `run.sh`, and `request.sh` are the Linux installation, V4.1 launch, and smoke-request scaffolds. The matched benchmark harness is in `bench/`. **September results on the older runtime:** the V4.1 vs 0731 study is complete for speculation off (concurrency, context, prefix cache states, isolated prefill/decode, prefill kernel breakdown) and DSpark fixed/adaptive at c1/4/16/64; see [report.md](report.md). In that study, 0731 is 1.1–1.35× faster uncached (V4.1 prefill kernel paths, no CED), V4.1 wins 1.05–2.24× with prefix caching (reuse after one computation vs two), and DSpark helps only at low concurrency.

**September MiMo-V2.6-Flash-MOPD results** (study `mimo-v26`, same node, older runtime and workloads as the DeepSeek study above): MiMo is 1.27–1.42× faster than 0731 and 1.43–1.73× faster than V4.1 uncached. It skips DeepSeek's sparse-attention/indexer, mHC, and heavy projections; all-reduce is its largest sampled prefill kernel component on that runtime. Details: [report_mimo.md](report_mimo.md).

The GitHub bundle includes small result JSONs, benchmark logs, selected startup logs, manifests, corrected summaries, and source/export SHA-256 provenance. Numeric result JSONs are preserved byte-for-byte. Exported manifests/logs may have private host/path information and terminal formatting removed; [provenance.json](references/provenance.json) records each transformation. Caches, weights, presentations, old executable launch wrappers, quarantined runs, and superseded narratives are excluded with reasons in the reference index. Original files remain in the source repository.

Validate from this repository root with Python 3.10 or newer (standard library only):

```bash
python tools/audit_references.py
```

The imported references are a bounded historical snapshot, not a copy of every result from the older repository. The October five-model study has its own evidence and manifests under `reports/blog-architecture-h100-v1/`. New GPU evidence belongs to a separate study ID and must preserve model/runtime/workload provenance.
