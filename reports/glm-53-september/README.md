# Earlier GLM reference archive

This directory retains **42 GLM-5.3-Flash result points from September 2, 2026**, plus their manifests, benchmark logs and selected startup logs. Read the [model guide](glm-5.3-flash/README.md) and [all numbers](glm-5.3-flash/RESULTS.md). These observations are separate from the October five-model comparison.

The historical runtime was vLLM `0.1.dev20051+g487ecf187`, with PyTorch `2.13.0+cu130`, TP8 and expert parallel on eight H100 80GB GPUs. Main utilization was 0.82, not October's 0.90; package overlays, KV formats and launch settings differ by arm. Immutable checkpoint revisions were not recorded. The arm manifests and logs qualify each result; a single arm-wide manifest is not a verified per-point configuration identity.

The same requested output length (256 tokens, `ignore_eos`) does not make these experiments controls for October. Historical batch/context inputs were synthetic, generally one retained observation per point. The finer concurrency grid spans sessions. There is no repeat-based confidence interval, equal-quality validation or production latency claim.

Output throughput includes prefill and scheduling. TTFT includes queueing; TPOT is a per-request average. Recorded cache usage is a fraction of the reserved pool, not device memory. Historical engine FLOP/byte estimates omit attention and are not hardware counters. Missing telemetry could be replaced with zero in the old harness; collection success does not prove validity. Prefix results lack matched cache-off and prewarmed controls and do not authorize fresh prefix experiments.

[arms.json](arms.json) retains the included arms and excluded-arm reasons for GLM. [results.csv](results.csv) preserves the corresponding source values. [provenance.json](provenance.json) is a labelled selection of the original export ledger: numerical JSONs are unchanged, while logs/manifests retain the recorded masking of private paths/hosts and terminal formatting. The complete original ledger remains in parent history.

No historical launcher, dependency overlay or version-check bypass is promoted as a new setup recommendation. Use [the current per-model guides and reproduction procedure](../../reproduce/README.md) for the October protocol.
