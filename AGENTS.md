# Session scope

Read [README.md](README.md), then [target.md](target.md) and [docs/reproduce.md](docs/reproduce.md), before planning work. Read [blog_target.md](blog_target.md) for the immediate objective, and [update.md](update.md) for evidence corrections. [docs/three-model-h100-plan.md](docs/three-model-h100-plan.md) governs the separate, later speculative-serving study.

- **Immediate objective (2026-10-07):** prepare the architecture blog in [blog_target.md](blog_target.md), comparing attention runtime/HBM traffic, live KV/state memory and FFN/MoE sparsity for DeepSeek V4 Flash 0731, V4.1 Flash and MiMo-V2.6-Flash-MOPD on one 8×H100 node. Proposed study `blog-architecture-h100-v1` is **planned, not measured**; this session updates documentation only. Finish local preparation before a later authorized GPU session.
- **Later objective:** the broader AR/speculative-serving study `spec-realtext-h100-v1` in [target.md](target.md) remains pending. Do not automatically execute its 168-run matrix, speculative arms or 2048-token workloads for the blog. The blog's separate study and completion ledger do not replace or complete that study.
- Preserve the completed studies: [DeepSeek report](report.md), [MiMo report](report_mimo.md). New real-text/long-output runs require fresh AR controls and their own study ID; never append them to historical curves.
- **No new prefix-cache experiments until Tan explicitly requests them.** This excludes cold/warm comparisons, one/two-touch prewarming, reuse checks, prefix pressure and speculation-plus-prefix combinations. Every active launch has prefix caching off. Ordinary request KV state and telemetry remain in scope. Old prefix commands are historical evidence, not instructions to execute.
- Only the three targets and immutable revisions in [target.md](target.md) are active. Preview DeepSeek, GLM-5.3-Flash, Qwen3.8-Flash-Next-FP8 and other models remain out of scope. H200 is a separate future study, not a next-session default or completion gate.
- Verify the pinned runtime source and loaded implementation; architecture descriptions and config field names do not prove execution or support. The blog starts with AR only, speculation off. When the later speculative study resumes, follow [docs/speculative-decoding.md](docs/speculative-decoding.md) and [the MiMo implementation record](docs/mimo-v2.6-plan.md) for checkpoint/method support. Current scope overrides historical experiment lists in those files.
- Keep TP8+EP, 0.90 utilization, max context 262144, max sequences 64, explicit 8192 chunk budget and native precision as the common deployment. Pin vLLM `44af287ebe38d6dc4e102948025f5e3e175aefd6`, target, draft, tokenizer and remote code. Record resolved graph/KV/quantization settings and thinking switches. Runtime/topology tuning is a separate arm with fresh controls.
- Use [references/README.md](references/README.md) for evidence selection and caveats. Preserve measured values and arm boundaries; no new causal claims without matched evidence. Keep the earlier source repository untouched and use relative links here.
- Keep weights, caches, environments, credentials, private host/path details and large profiler traces out of the GitHub bundle. Record sanitization; never silently alter numerical results. After changing reference data or summaries, run `python tools/audit_references.py` on a clean clone.
- **GPUs are rented by the hour.** Finish local preparation before rental. Chain bounded launches, measure immediately after readiness, stop the owned server on every success/error/timeout path, and verify shutdown. Analyze while queued work runs. No ready-server idle time and no undeclared sweep merely to occupy GPUs.

# Study status (updated 2026-10-07)

| Item | State |
| --- | --- |
| DeepSeek speculation off, traces and historical prefix checks | **Done**, 144 valid runs/model; [report.md](report.md) |
| DeepSeek fixed/adaptive DSpark k5, c1/4/16/64 × 3 | **Done**, 48/48 valid |
| MiMo speculation off, traces and historical prefix checks | **Done**, 90/90 valid |
| MiMo MTP k3 and DFlash k7 | **Done**, 24/24 valid; total 114/114; revision `2479e2d0029eca9a34cc7e7f55a121925f81908e` |
| Three-model evidence review / target / next-session plan | **Prepared**; [update.md](update.md), [GPU plan](docs/three-model-h100-plan.md) |
| Architecture blog plan | **Prepared:** [blog_target.md](blog_target.md); proposed study `blog-architecture-h100-v1`, no new measurements |
| Blog source worksheet, public inputs, collection and dry-run manifest | **Pending; immediate local preparation before rental** |
| Blog initial measurements | **Pending:** 54 AR timing runs, 18 trace captures, 6 hardware-counter point sets; live KV/routing diagnostics and extra collection costs declared separately |
| Later speculative-study harness and datasets | **Pending for later:** variable outputs, real-text inputs and manifest-driven execution |
| Later speculative H100 core | **Pending for later:** E0, E1 42-run screen + 84 controls + 42 crossover = 168 timing runs, E3 14 captures |
| Conditional extensions | Blog: selected batch/context probes or supported single-change ablations. Later speculative study: E2/E4, domain timing and confirmation. Declare points and cost first |
| Prefix-cache follow-ups / H200 | **Deferred**, excluded from both studies' core completion |

# Next-session actions

1. Follow [blog_target.md](blog_target.md) sections 2–3: prepare source-linked architecture/implementation facts, public code/math/chat inputs, pinned templates/tokenizers and an operator-to-component map. Separate architecture facts, executed implementation, measured results and estimates.
2. Prepare the blog harness locally: per-request 256-token validity, actual batch/context tracking, decode-only segmentation, live KV-block accounting, FFN routing diagnostics and targeted HBM collection. Treat unavailable counters as unavailable. Existing single-request traces do not establish synchronized B=8 profiling or hardware byte attribution.
3. Build and dry-run a new no-prefix chain for `blog-architecture-h100-v1`; reject historical study paths and speculative/prefix-enabled launches. Reuse bounded readiness and owned-process cleanup patterns from `chain_lib.sh`. Do **not** invoke `chain_mimo.sh` or `chain_dspark.sh`.
4. Only after a GPU session is authorized and a node/allocation supplied, inventory the node, verify pins/resolved settings, run functional checks and a disjoint pilot. Freeze `plan.json`, request lists/counts/order, timeouts, validity/rerun rules, launch order and node-hour budget before timing. Do not assume the previous node/cache survived.
5. Collect fresh unprofiled AR for all three models at approximately 1K/16K/64K input, 256 output, c1/c8, three repeats: **54 timing runs**. Prioritize a complete three-model 16K comparison. Follow the blog plan's launch/block design and keep incomplete points pending.
6. Collect **18 diagnostic traces**: each model's 16K/64K single-request prefill plus 1K/64K decode at actual B=1/8. Select **6 initial HBM-counter point sets** at late 64K prefill and 64K/B=1 decode across all models. Reuse diagnostics for live KV and FFN analysis; budget counter replay, instrumentation overhead controls and relaunches separately.
7. Explain full attention-path cost, physical live KV bytes and realized FFN sparsity, including communication and unattributed work. Keep native timing separate from eager/instrumented diagnostics. Select extensions from observed evidence and declare their cost first; no automatic speculation, prefix or H200 work.
8. Stop the owned server on every exit and verify shutdown. Curate small evidence, produce the blog figures/findings and leave exact continuation commands and pending/failed/unsupported points. Commit/push/merge only when requested.

# Later speculative-study continuation

These actions remain preserved for when Tan resumes [target.md](target.md); they are not the immediate blog queue.

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
- For the blog, distinguish attention-core time from projections/indexer/compression and the complete attention path. Keep FFN/shared-expert projections out of attention totals and assign fused kernels once. Nominal active parameters are not measured HBM bytes; total DRAM counters alone cannot separate weight, KV and temporary traffic.
- Live KV/state bytes, reserved cache pools and total device memory are different quantities. Record per-rank replication and block rounding. Ordinary live KV measurements do not establish reusable-prefix capacity; Kan's quoted prefix question does not override the no-prefix instruction.
- Separate attention sparsity, MoE activation sparsity and weight sparsity/quantization. Top-k alone does not predict FFN time; measure expert coverage, padding, routing and exposed communication. Uneven routing alone does not prove a bottleneck.
- Prefill traces implicate V4.1 dense Marlin FP8 and collective paths; CED prompt skipping was absent. MiMo all-reduce is 44% of the sampled prefill **kernel sum**. These select hypotheses; neither share nor FLOPs predicts an achieved optimization gain.
- V4.1 full CUDA graphs hide decode attribution in the old traces. Keep graph-aware/eager diagnostics separate and record overhead; missing components are unavailable, not zero.
- MiMo resolves to an `Omni` wrapper because of its vision config; verify the actual text decoder. `mimo_v2_mtp` in the pinned runtime reuses layer 0 for k3, not all three trained heads.
- Thinking-off uses `thinking=false` for DeepSeek and `enable_thinking=false` for MiMo. Validate rendered requests and output; same text across models does not mean identical token counts.
- Keep slow first repeats and failures. Pair by manifest identity, not directory names: MiMo's old AR c1 rerun was missed by the output-match script. Old decoded-text comparisons and four-prompt smokes do not prove losslessness or equal quality.
- Three repeats screen effects; the old SD heuristic is not a confidence interval. Confirmation needs held-out prompts, independently scheduled repeat blocks and declared practical thresholds. No production-SLO or tail-latency claim from a small closed-loop screen.

# Measurement guideline

[docs/reproduce.md](docs/reproduce.md) covers environment/JIT-cache fixes, historical reproduction and publishing. [blog_target.md](blog_target.md) governs immediate blog preparation and its proposed measurement scope. [target.md](target.md) remains authoritative for immutable model identities and common deployment constraints; its broader speculative matrix and [docs/three-model-h100-plan.md](docs/three-model-h100-plan.md) are for later. Keep the two studies' manifests, results and completion status separate.
