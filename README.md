# DeepSeek serving measurements

The active study compares **DeepSeek V4 Flash and V4.1 Flash**, from serving throughput/latency down to prefill, decode, and individual component costs. See [target.md](target.md) for the measurement matrix and ordered GPU-session checklist.

This repository contains a curated, portable subset of the earlier `LLMs_frontier_serving` study. Historical GLM and Qwen evidence is retained for a future comparison **only when requested**. It does not expand the active DeepSeek scope.

| Need | File |
| --- | --- |
| Next agent's scope | [AGENTS.md](AGENTS.md) |
| DeepSeek measurement and GPU handoff plan | [target.md](target.md) |
| Historical evidence, selection, and shared caveats | [references/README.md](references/README.md) |
| DeepSeek V4 baseline | [DeepSeek reference](references/deepseek-v4-flash/README.md) |
| Dormant GLM evidence | [GLM reference](references/glm-5.3-flash/README.md) |
| Dormant Qwen evidence | [Qwen reference](references/qwen3.8-flash-next-fp8/README.md) |
| Verify the curated bundle locally | [tools/audit_references.py](tools/audit_references.py) |

`install.sh`, `run.sh`, and `request.sh` are the existing Linux installation, V4.1 launch, and smoke-request scaffolds. They are not an automated benchmark harness. Installation currently selects a moving nightly, and the launch defaults are 32K context and eight server sequences; follow `target.md` before a GPU comparison. No new GPU execution, V4.1 result, or component profile is included yet.

The GitHub bundle includes small result JSONs, benchmark logs, selected startup logs, manifests, corrected summaries, and source/export SHA-256 provenance. Numeric result JSONs are preserved byte-for-byte. Exported manifests/logs may have private host/path information and terminal formatting removed; [provenance.json](references/provenance.json) records each transformation. Caches, weights, presentations, old executable launch wrappers, quarantined runs, and superseded narratives are excluded with reasons in the reference index. Original files remain in the source repository.

Validate from this repository root with Python 3.10 or newer (standard library only):

```bash
python tools/audit_references.py
```

The bundle is a bounded historical snapshot, not a copy of every result from the older repository. New GPU evidence belongs to a separate study ID and must preserve the model/runtime/workload provenance required by `target.md`.
