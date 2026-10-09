# Earlier completed experiments

The September studies use **vLLM `44af287ebe38d6dc4e102948025f5e3e175aefd6`**, version `0.30.1rc1.dev223+g44af287eb`, on the same earlier 8×H100 node. They used random-token prompts and their own manifests. The October article uses a newer build and public text; do not interpret a difference between the two as a runtime improvement.

| Study | Completed measurements | Evidence |
| --- | --- | --- |
| `v41-vs-0731` | 144 valid speculation-off runs per model; 48 valid fixed/adaptive DSpark k5 runs across both models; historical prefix checks and traces | [DeepSeek report](../reports/v41-vs-0731/report.md), [tables and raw data](../reports/v41-vs-0731/) |
| `mimo-v26` | 90 valid speculation-off runs; 12 MTP k3 and 12 DFlash k7 runs; historical prefix checks and traces | [MiMo report](../reports/mimo-v26/report.md), [tables and raw data](../reports/mimo-v26/) |

GLM also has a [42-point September 2 archive](../reports/glm-53-september/glm-5.3-flash/RESULTS.md) on vLLM `0.1.dev20051+g487ecf187`: baseline, MTP n1/n5, and alternate-stack KV-format experiments. Its checkpoint revision was not recorded, settings differ from October, and most points are single observations. The [GLM guide](../reports/glm-53-september/glm-5.3-flash/README.md) preserves arm boundaries and telemetry caveats. It is historical context, not a control for the five-model results.

0731 was 1.1–1.35× faster than V4.1 uncached on the older runtime. MiMo was 1.27–1.42× faster than 0731 and 1.43–1.73× faster than V4.1. These ranges belong only to the saved workloads in those reports. The old build did not execute V4.1's CED prefill skip; the October build does. The older expert backend was Marlin; October selected HUMMING for the three models.

Read the historical reports with these evidence corrections:

- `1 + accepted / drafts` is an estimated tokens-per-round metric, not measured committed progress. Scheduled drafts are not actual verified counts under adaptive verification.
- Three repeats screen differences; the reports' SD heuristic is not a confidence interval.
- Greedy text matches and four-prompt smokes do not prove losslessness or equal quality. MiMo's original output comparison omitted its c1 AR rerun, so its coverage is limited as stated in the report.
- Historical prefix advantages depend on the warming procedure. They are not general retained-prefix capacity measurements. The saved results remain unchanged; no new prefix experiment is authorized.
- KV token-pool capacities use different formats and accounting. Compare bytes for memory claims.
- GPU kernel sums can overlap and do not predict an achievable optimization gain.

The old reports retain their original commands for documentary reproduction. Do not invoke `chain_mimo.sh` or `chain_dspark.sh` as the October workflow: they hardcode historical studies and can include prefix experiments. Use [the blog reproduction guide](../reproduce/README.md) for the five-model comparison. The separate later real-text speculative study remains pending.
