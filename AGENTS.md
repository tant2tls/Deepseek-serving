# Session scope

Read [README.md](README.md), then [target.md](target.md) and [docs/reproduce.md](docs/reproduce.md), before planning work. For the active phase also read [update.md](update.md) and [docs/three-model-h100-plan.md](docs/three-model-h100-plan.md).

- **Active objective:** compare the serving strengths and weaknesses of DeepSeek V4 Flash 0731, V4.1 Flash and MiMo-V2.6-Flash-MOPD on one 8×H100 node. The 2026-09-30 session reviewed evidence and prepared documentation only. New study `spec-realtext-h100-v1` is **planned, not measured**; harness extensions and datasets remain to be prepared before the later GPU session.
- Preserve the completed studies: [DeepSeek report](report.md), [MiMo report](report_mimo.md). New real-text/long-output runs require fresh AR controls and their own study ID; never append them to historical curves.
- **No new prefix-cache experiments until Tan explicitly requests them.** This excludes cold/warm comparisons, one/two-touch prewarming, reuse checks, prefix pressure and speculation-plus-prefix combinations. Every active launch has prefix caching off. Ordinary request KV state and telemetry remain in scope. Old prefix commands are historical evidence, not instructions to execute.
- Only the three targets and immutable revisions in [target.md](target.md) are active. Preview DeepSeek, GLM-5.3-Flash, Qwen3.8-Flash-Next-FP8 and other models remain out of scope. H200 is a separate future study, not a next-session default or completion gate.
- Follow [docs/speculative-decoding.md](docs/speculative-decoding.md) and [the MiMo implementation record](docs/mimo-v2.6-plan.md) for checkpoint/method support. Verify the pinned runtime source and loaded implementation; config field names do not prove support. New scope overrides historical experiment lists in those files.
- Keep TP8+EP, 0.90 utilization, max context 262144, max sequences 64, explicit 8192 chunk budget and native precision as the common deployment. Pin vLLM `44af287ebe38d6dc4e102948025f5e3e175aefd6`, target, draft, tokenizer and remote code. Record resolved graph/KV/quantization settings and thinking switches. Runtime/topology tuning is a separate arm with fresh controls.
- Use [references/README.md](references/README.md) for evidence selection and caveats. Preserve measured values and arm boundaries; no new causal claims without matched evidence. Keep the earlier source repository untouched and use relative links here.
- Keep weights, caches, environments, credentials, private host/path details and large profiler traces out of the GitHub bundle. Record sanitization; never silently alter numerical results. After changing reference data or summaries, run `python tools/audit_references.py` on a clean clone.
- **GPUs are rented by the hour.** Finish local preparation before rental. Chain bounded launches, measure immediately after readiness, stop the owned server on every success/error/timeout path, and verify shutdown. Analyze while queued work runs. No ready-server idle time and no undeclared sweep merely to occupy GPUs.

# Study status (updated 2026-09-30)

| Item | State |
| --- | --- |
| DeepSeek speculation off, traces and historical prefix checks | **Done**, 144 valid runs/model; [report.md](report.md) |
| DeepSeek fixed/adaptive DSpark k5, c1/4/16/64 × 3 | **Done**, 48/48 valid |
| MiMo speculation off, traces and historical prefix checks | **Done**, 90/90 valid |
| MiMo MTP k3 and DFlash k7 | **Done**, 24/24 valid; total 114/114; revision `2479e2d0029eca9a34cc7e7f55a121925f81908e` |
| Three-model evidence review / target / next-session plan | **Prepared**; [update.md](update.md), [GPU plan](docs/three-model-h100-plan.md) |
| New phase harness, public real-text datasets, dry-run manifest | **Pending; first next-session task before rental** |
| New H100 core | **Pending:** E0, E1 42-run screen + 84 controls + 42 crossover = 168 timing runs, E3 14 captures |
| Conditional H100 extensions | E2 widths/adaptive, E4 context, domain timing, held-out confirmation, supported single-change ablations; declare points and cost first |
| Prefix-cache follow-ups / H200 | **Deferred**, excluded from H100 core completion |

# Next-session actions

1. Complete the local readiness checklist in [the GPU plan](docs/three-model-h100-plan.md): variable outputs throughout collection/validation, pinned real-text inputs, exact template/tokenizer handling, rerun pairing, natural-EOS task checks, diagnostic accounting and a no-prefix manifest-driven chain.
2. Do **not** invoke `chain_mimo.sh` or `chain_dspark.sh` for this study: they hardcode historical study IDs and the MiMo chain runs prefix experiments. Reuse `chain_lib.sh` patterns with bounded readiness/shutdown and owned-process cleanup. Existing `run_matrix.py` defaults are concurrency/context, but its global output length is 256; no real-text/2048 runner is claimed ready.
3. Once the later GPU session is authorized and a node/allocation is supplied, inventory it, verify pins/support/config, run E0 and a disjoint pilot. Freeze `plan.json`, request counts/order, timeouts, validity/rerun rules, launch order and estimated node-hours before the timing matrix. Do not assume the previous node or cache survived.
4. Collect complete comparison blocks A/B/C with fresh AR for all three models. The 42-run screen is only the first block, not completion of the 168-run core. Group the screen economically, then rotate launch order for expanded/confirmation blocks. Follow the explicit budget/count rules; all omitted core points remain pending.
5. Capture actual decode-only B=1/64 diagnostics and select extensions from observed evidence. Runtime work without a supported implementation should not consume rental time. H200, prefix work and new models do not start automatically.
6. Stop the server after measurement, curate small evidence, validate, report workload-specific findings and leave exact continuation commands and pending cells. Commit/push/merge only when requested.

# Findings and traps to carry forward

- Historical MiMo uncached throughput leads both DeepSeek deployments. 0731 has the larger relative fixed-DSpark c1 gain, but MiMo DFlash has higher absolute throughput. V4.1's short-prompt TTFT win is over 0731, not MiMo. Prefix advantages depend on the historical warming procedure.
- Fixed-length input/output throughput proportionality is an accounting identity, not proof of a prefill bottleneck. Little's law uses average in-flight requests under steady conditions, not automatically the client cap. Use phase timing and matched output-length controls.
- `spec_compare.py`'s historical `1+accepted/drafts` is an estimated tokens-per-round metric. New mechanism claims need actual committed tokens with sequence-round, fallback and truncation accounting. Scheduled drafts are not actual verified counts under adaptive verification.
- Different KV formats/pool accounting prevent interpreting token capacities as measured request capacity. MiMo's `auto` resolves to BF16; DeepSeek uses `fp8_ds_mla`. Native quantization and kernel paths differ; call these deployment comparisons.
- Prefill traces implicate V4.1 dense Marlin FP8 and collective paths; CED prompt skipping was absent. MiMo all-reduce is 44% of the sampled prefill **kernel sum**. These select hypotheses; neither share nor FLOPs predicts an achieved optimization gain.
- V4.1 full CUDA graphs hide decode attribution in the old traces. Keep graph-aware/eager diagnostics separate and record overhead; missing components are unavailable, not zero.
- MiMo resolves to an `Omni` wrapper because of its vision config; verify the actual text decoder. `mimo_v2_mtp` in the pinned runtime reuses layer 0 for k3, not all three trained heads.
- Thinking-off uses `thinking=false` for DeepSeek and `enable_thinking=false` for MiMo. Validate rendered requests and output; same text across models does not mean identical token counts.
- Keep slow first repeats and failures. Pair by manifest identity, not directory names: MiMo's old AR c1 rerun was missed by the output-match script. Old decoded-text comparisons and four-prompt smokes do not prove losslessness or equal quality.
- Three repeats screen effects; the old SD heuristic is not a confidence interval. Confirmation needs held-out prompts, independently scheduled repeat blocks and declared practical thresholds. No production-SLO or tail-latency claim from a small closed-loop screen.

# Measurement guideline

[docs/reproduce.md](docs/reproduce.md) covers environment/JIT-cache fixes, historical reproduction and publishing. [docs/three-model-h100-plan.md](docs/three-model-h100-plan.md) governs the new session's active run order, no-prefix scope, validity and completion.
