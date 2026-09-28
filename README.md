# DeepSeek serving measurements

The active study compares **DeepSeek V4 Flash 0731 (official release) and V4.1 Flash**, from serving throughput/latency down to prefill, decode, and individual component costs. See [target.md](target.md) for the measurement matrix and ordered GPU-session checklist.

The old V4 measurements used the **preview** model ID, with exact revision unverified. They are historical context, not 0731 results. The [speculative-decoding plan](docs/speculative-decoding.md) separates classic preview MTP from native DSpark and compares speculation off, fixed DSpark, and adaptive DSpark for the two primary checkpoints.

This repository contains a curated, portable subset of the earlier `LLMs_frontier_serving` study. Historical GLM and Qwen evidence is retained for a future comparison **only when requested**. It does not expand the active DeepSeek scope.

| Need | File |
| --- | --- |
| Next agent's scope | [AGENTS.md](AGENTS.md) |
| **V4.1 vs 0731 measured results (speculation off + DSpark)** | [report.md](report.md) |
| **MiMo-V2.6-Flash vs 0731 vs V4.1 (speculation off + MTP/DFlash)** | [report_mimo.md](report_mimo.md) |
| How MiMo runs prefill/decode, and why it is faster | [docs/mimo-v2.6-inference.md](docs/mimo-v2.6-inference.md) |
| Reproduce a study / add a model | [docs/reproduce.md](docs/reproduce.md) |
| Benchmark/profiling harness | [bench/](bench/) |
| Curated V4.1 and 0731 run evidence | [reports/v41-vs-0731/](reports/v41-vs-0731/) |
| DeepSeek measurement and GPU handoff plan | [target.md](target.md) |
| Checkpoint identity and MTP/DSpark experiments | [docs/speculative-decoding.md](docs/speculative-decoding.md) |
| Historical evidence, selection, and shared caveats | [references/README.md](references/README.md) |
| Historical DeepSeek V4 preview | [DeepSeek reference](references/deepseek-v4-flash/README.md) |
| Dormant GLM evidence | [GLM reference](references/glm-5.3-flash/README.md) |
| Dormant Qwen evidence | [Qwen reference](references/qwen3.8-flash-next-fp8/README.md) |
| Verify the curated bundle locally | [tools/audit_references.py](tools/audit_references.py) |

`install.sh`, `run.sh`, and `request.sh` are the Linux installation, V4.1 launch, and smoke-request scaffolds. The matched benchmark harness is in `bench/`. The **V4.1 vs 0731 study is complete** for speculation off (concurrency, context, prefix cache states, isolated prefill/decode, prefill kernel breakdown) and DSpark fixed/adaptive at c1/4/16/64; see [report.md](report.md). Headline: 0731 is 1.1–1.35× faster uncached on this runtime (V4.1 prefill kernel paths, no CED), V4.1 wins 1.05–2.24× with prefix caching (reuse after one computation vs two), and DSpark helps only at low concurrency.

**MiMo-V2.6-Flash-MOPD** (study `mimo-v26`, same node, runtime, and workloads) is 1.27–1.42× faster than 0731 and 1.43–1.73× faster than V4.1 uncached. It skips DeepSeek's sparse-attention/indexer, mHC, and heavy projections; all-reduce is now its largest cost. Details: [report_mimo.md](report_mimo.md).

The GitHub bundle includes small result JSONs, benchmark logs, selected startup logs, manifests, corrected summaries, and source/export SHA-256 provenance. Numeric result JSONs are preserved byte-for-byte. Exported manifests/logs may have private host/path information and terminal formatting removed; [provenance.json](references/provenance.json) records each transformation. Caches, weights, presentations, old executable launch wrappers, quarantined runs, and superseded narratives are excluded with reasons in the reference index. Original files remain in the source repository.

Validate from this repository root with Python 3.10 or newer (standard library only):

```bash
python tools/audit_references.py
```

The bundle is a bounded historical snapshot, not a copy of every result from the older repository. New GPU evidence belongs to a separate study ID and must preserve the model/runtime/workload provenance required by `target.md`.
