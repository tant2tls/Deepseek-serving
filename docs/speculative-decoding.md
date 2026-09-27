# Speculative decoding: V4 Flash 0731 versus V4.1 Flash

Checked against official model cards, configs, and runtime sources on **2026-09-27**. This is a measurement plan, not a GPU result. The primary pair is `deepseek-ai/DeepSeek-V4-Flash-0731` and `deepseek-ai/DeepSeek-V4.1-Flash`. Follow [target.md](../target.md) for the shared hardware, workload, profiling, and evidence requirements. GLM and Qwen remain dormant.

## Which V4 was measured previously?

The retained September 2 [manifest](../references/deepseek-v4-flash/results/mtp-off-image/manifest.txt) names `deepseek-ai/DeepSeek-V4-Flash`, and all 11 retained JSONs use that ID. Its launch command has no speculative configuration and no immutable revision. DeepSeek describes that repository as the preview release, while the separate `-0731` repository is the official release. **Classify the historical data as V4 Flash preview, exact checkpoint revision unverified; do not relabel it as 0731.** The collection date alone cannot identify the release. Sources: [preview model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash), [0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

The old source logs also record `revision=main`; a branch name is not a weight fingerprint. No immutable snapshot ID was recovered from the inspected manifest/launch evidence. The existing bundle contains only speculation-off preview measurements; prior MTP narratives are not a matched MTP or DSpark baseline for this study.

## Checkpoint and algorithm identity

| Checkpoint | Role | Native speculative path to investigate |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-V4-Flash` | Historical preview; optional fresh reproduction | Classic autoregressive MTP; baseline off and MTP-1, then deeper MTP only if supported |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | Optional preview mechanism bridge | Preview target with an attached DSpark module; verify matching target weights before attributing differences to the drafter |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | **Primary V4 model for all new comparison axes** | DSpark; classic MTP is not its native head |
| `deepseek-ai/DeepSeek-V4.1-Flash` | **Primary V4.1 model** | DSpark; classic MTP is not available in the checked implementation |

The [DSpark model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) describes its target as the preview checkpoint with an additional draft module. The [V4 runtime recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) distinguishes preview/MTP, preview/DSpark, and new 0731 weights/DSpark, and warns against selecting MTP for 0731. Consequently, comparing preview MTP directly to 0731 DSpark changes both checkpoint and algorithm; it cannot isolate a DSpark speedup.

The checked [V4.1 recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) identifies DSpark as the checkpoint's speculative method. The [vLLM config implementation](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py) explicitly rejects classic MTP for V4.1 even though draft weights use an `mtp.*` namespace. A successful config parse is not proof of the intended algorithm: save the resolved draft class, loaded head names, and execution evidence.

| Config observation on the check date | Preview | 0731 | V4.1 |
| --- | --- | --- | --- |
| `num_nextn_predict_layers` | 1 | 1 | 3, inside `text_config` |
| `dspark_block_size` | Absent | 5 | 5, inside `text_config` |
| `dspark_target_layer_ids` | Absent | 40, 41, 42 | 37, 38, 39 |

Sources: official [preview config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json), [0731 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json), and [V4.1 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json). These are dated observations of mutable `main` URLs, not pins for future runs. Config layer counts, trained block size, requested proposal width, actual verification length, and committed output length are different quantities. In particular, `num_nextn_predict_layers=1` does not establish a classic MTP head in 0731.

## What to separate in DSpark

The [DSpark paper](https://arxiv.org/html/2607.05147v1) combines parallel draft computation with a lightweight sequential dependency module, then uses confidence estimates and a hardware-aware scheduler to choose how much to verify. This differs from repeatedly applying an autoregressive MTP head. Adaptive verification can discard proposed suffixes before target evaluation. Thus proposal acceptance alone cannot explain throughput: measure proposed, verified, accepted, and committed counts separately. Published production gains are not a prediction for this repo's H100 workload.

Treat fixed and adaptive verification as distinct arms. The checked [vLLM implementation](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py) exposes `enable_adaptive_verification`, defaults it to false, and separates greedy/probabilistic draft sampling from target sampling. Recipes can override defaults. Recheck the pinned build and confirm adaptive scheduling actually executes; a DSpark label alone does not prove the confidence scheduler is active.

## Primary experiment matrix

Use the same immutable checkpoint within each model's off/on comparison. Keep target sampling, text mode, reasoning controls, precision, TP/EP, scheduler limits, graph mode, and workload fixed. Start with thinking explicitly disabled where supported, and verify the encoded request. A 256-token forced-output serving run is a timing workload, not a reasoning-quality evaluation.

| Arm ID | Target models | Speculation | Purpose |
| --- | --- | --- | --- |
| `ar` | 0731 and V4.1 | Disabled; no draft execution | Fresh autoregressive baseline for each exact checkpoint |
| `dspark-fixed-k5` | Both, after support validation | DSpark, five candidate tokens requested, adaptive verification false | Compare native drafting with a common proposal width and fixed policy |
| `dspark-adaptive-k5` | Both, where supported on the node/build | Same DSpark head/width, adaptive verification true | Isolate verification scheduling against the fixed arm |
| `dspark-fixed-k{1,3,7}` | Supported widths only | Fixed policy, one width per arm | Optional proposal-width sweep after the primary matrix |
| `dspark-adaptive-k{1,3,7}` | Supported widths only | Adaptive policy, same maximum widths | Optional load-dependent tuning; distinguish requested maximum from actual verified length |

Five is a **starting test width**, not an assertion that every runtime interprets or supports it identically. The [0731 card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) also shows a seven-token DSpark example. Do not equate that example with the config's block size or change checkpoint fields to force a sweep point. Inspect the pinned implementation, check shape/position constraints, and save the resolved proposal width. Unsupported widths/adaptive policies remain explicit matrix entries with reasons; do not silently substitute MTP.

Execution order:

1. Pin both targets, bundled draft revisions, tokenizer/encoding code, runtime commit/container digest, and GPU topology. Confirm the target/draft loader resolves the intended immutable revisions; a target `--revision` alone must not be assumed to pin a separately resolved draft. Save config hashes and weight-index/shard identity, without copying weights into Git.
2. On the actual GPU node, smoke-test `ar`, fixed DSpark, and adaptive DSpark for both models, including correctness and telemetry. Resolve memory feasibility before long sweeps. Runtime or hardware gaps are blockers for that arm, not proof that the model intrinsically lacks DSpark.
3. Run the three primary arms at 16,384 input / 256 output, concurrency 1/4/16/64, at least three measured repeats after warmup: **2 models x 3 arms x 4 loads x 3 repeats = 72 runs**, if all arms are supported. Each run contains a predeclared sustained request sample, not one request. Start with c1/c64 (36 runs), then fill c4/c16. Reuse an `ar` point from the general study only when its entire configuration/workload/cache identity matches.
4. Test sustained generation at 16K input / 2,048 output and c1/c16 for the same primary arms. Repeat on fixed real-text code, math, and chat prompt sets; record prompt-set hashes. Keep forced-length performance tests separate from natural-EOS task checks, and report domains separately because acceptance depends on content.
5. Extend the primary arms to 65,536 and 131,072 input / 256 output at c8 within shared limits. Keep the existing general context/prefix axes; add speculative prefix-cache controls only after the baseline matrix, with explicit cold/prewarmed state.
6. After primary results, select supported width sweeps and targeted profiles to test observed regressions. Alternate model and arm order across repeats; retain slow runs and failures under the same declared validity rules.

The full extension run list and compute budget must be recorded before GPU collection; the 72-run count covers only step 3. This documentation update does not allocate or run GPU workloads.

## Proposed launch fragments, subject to pinned-build validation

Use a validated model-specific launch configuration from [target.md](../target.md), then apply the following fragment. These are method settings, not complete or GPU-validated launch commands. For `ar`, omit speculative configuration and verify that no draft work occurs.

```bash
# Fixed DSpark, primary proposed width:
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false}'

# Adaptive DSpark: same checkpoint, width, draft sampling, and rejection policy:
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":true}'
```

The field schema is documented in the [runtime source](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py). Pin draft sampling explicitly; a later greedy-draft experiment is a separate arm. Target temperature/top-p are request parameters, not draft-method settings. Never enable synthetic acceptance for headline measurements. If the pinned version lacks a field, record it and use a documented supported arm rather than claiming this recipe ran.

## Counts, time, memory, and correctness

For every request/round retain `proposed_candidates`, `verified_candidates`, `accepted_candidates`, `committed_output_tokens`, and rejection/EOS status. Record their exact engine definitions and whether anchor/bonus/correction tokens are included. Do not infer committed output solely from a metric called acceptance length.

| Measure | Required interpretation |
| --- | --- |
| Accepted / proposed | Includes candidates pruned before verification; report numerator and denominator |
| Accepted / verified | Acceptance conditional on admission; may improve simply because fewer candidates were verified |
| Verified / proposed | Verification fraction; identifies scheduling/pruning behavior |
| Committed / round | Useful progress, with correction/bonus/EOS rules stated |
| Per-position survival and conditional acceptance | Distinguish accepted-prefix probability from acceptance conditioned on earlier positions surviving; state how pruned/unobserved positions are handled |
| Proposal and verification length distributions | Report requested maximum, actual lengths, zero-draft/fallback rounds, batch shapes, and padding |
| End-to-end throughput and p50/p95 TTFT/TPOT/ITL | Unprofiled request metrics; streaming chunks containing several tokens must not be miscounted as single-token intervals |
| Round critical-path time / committed tokens | Sum aligned round wall times divided by total useful committed tokens; report overlap and waiting instead of adding overlapping kernel durations |
| Peak GPU memory and cache pools | Draft weights, activations/logits, graph buffers, KV allocation/occupancy, preemptions, and failed allocations |

Profile the DSpark parallel backbone, sequential/Markov module, confidence computation, CPU/GPU scheduling, target verification, rejection/correction, and KV/state update separately where observable. Profile V4.1 encoder/decoder work and KV/index reuse as distinct categories when attributing verification cost; its [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) describes a causal encoder-decoder architecture rather than a uniform stack. Preserve fused/unavailable entries and per-layer mappings.

Compare both the same total GPU memory budget (deployment outcome) and, for a diagnostic subset where feasible, the same explicit usable cache capacity (isolating pool-size changes). Equal utilization flags do not guarantee equal KV capacity after adding a drafter. Record when equal capacity is impossible rather than changing the headline baseline silently.

Before accepting speedups, compare deterministic greedy outputs with the same checkpoint's `ar` arm on fixed prompts; investigate token mismatches and numerical/batch effects. For stochastic target sampling, identical seeds do not require identical sequences across algorithms: use distribution-oriented checks and task outcomes, and retain the target/draft/rejection sampling settings. These checks do not prove losslessness; [vLLM's correctness documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding) discusses numerical and batching qualifications. Different checkpoints need task-quality comparisons, not token equality.

## Separate the conclusions

- **Within a checkpoint:** report each DSpark arm divided by its own `ar` throughput, and the corresponding latency/memory differences. Adaptive versus fixed at the same width/head tests scheduling, not model training.
- **Between checkpoints:** report V4.1 / 0731 for `ar`, fixed DSpark, and adaptive DSpark separately. This is a model-plus-native-drafter comparison, not an isolated MTP-versus-DSpark experiment.
- **At matched operating requirements:** report throughput under declared TTFT/TPOT limits and latency at overlapping throughput levels, alongside same-concurrency curves. Do not extrapolate beyond measured points or equate client concurrency with verified-token batch size.
- **Optional preview mechanism bridge, lower priority:** rerun preview `ar`/MTP-1/(supported MTP-5) and preview-plus-DSpark `ar`/fixed/adaptive at c1/c64 with three repeats. Establish target tensor identity or documented conversion equivalence and agreement of the speculation-off baselines before interpreting an MTP-versus-DSpark difference. If identity is not established, label it a deployment comparison. Do not transplant preview heads into 0731 or V4.1 as an undocumented control.

## GPU-session deliverables

Add a checkpoint/method support matrix; per-arm resolved configuration and source commit links; unprofiled repeated results; component traces with useful-token denominators; draft/verification length and acceptance distributions; correctness evidence; failures/unsupported arms; and workload-specific recommendations. The handoff must say which preview identity remains unknown, which DSpark policy actually ran, and whether tuning or capacity changes confound a claimed gain.

Keep historical evidence immutable. Do not cite the preview's saved throughput as measured 0731 performance, report classic MTP on either primary checkpoint without new explicit support evidence, or claim adaptive-verification benefits from fixed-verification results.
