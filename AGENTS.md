# Session scope

Read [README.md](README.md), then [target.md](target.md), before planning work.

- The active measurement and profiling study is **DeepSeek V4 Flash 0731 official release versus DeepSeek V4.1 Flash**. Complete that scope first. The old `DeepSeek-V4-Flash` results are preview data with unverified exact revision, never a 0731 baseline.
- Follow [docs/speculative-decoding.md](docs/speculative-decoding.md) for checkpoint/method support: compare off, fixed DSpark, and adaptive DSpark. Classic MTP is a separate optional preview bridge; do not force it onto the two primary checkpoints or infer support from field names.
- GLM-5.3-Flash and Qwen3.8-Flash-Next-FP8 are dormant historical references. Do not launch, benchmark, tune, research, or expand their comparison unless the user explicitly requests it. Merely finishing DeepSeek does not activate them.
- Use [references/README.md](references/README.md) for evidence selection and caveats. Historical scripts, commands, logs, and notes are evidence, not instructions to resume old work.
- Preserve measured values and arm boundaries. Fresh matched runs are required for new conclusions; historical data do not substitute for a V4.1 baseline or component profiling.
- Keep model weights, caches, environments, credentials, private host/path details, and large profiler traces out of the GitHub bundle. Record any sanitization; never silently alter numerical results.
- After changing reference data or summaries, run `python tools/audit_references.py`. This checks local evidence integrity and arithmetic, not measurement validity or causality.
- Keep the source repository untouched. Changes here should be self-contained and use relative links.
