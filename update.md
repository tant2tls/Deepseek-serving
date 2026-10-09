# Three-model serving study: evidence review and experiment design

**Current blog decision, 2026-10-09:** refine the article from the completed October results, using **Qwen BF16 as the main Qwen deployment** and Qwen FP8 only as a historical reference. The other four deployments retain their first-node results; BF16 comes from the second node with a MiMo control. [target.md section 11](target.md#11-current-blog-evidence-selection-2026-10-09) governs source selection, counts and the throughput-only cross-node comparison. The review below governs the separate, later speculative study and does not authorize a new GPU session.

**Later decision, 2026-10-08:** Tan requires the original **BF16** `Qwen/Qwen3.8-Flash-Next` for all future Qwen work, with no new FP8 checkpoint runs, conversion or fallback. The source pin and preparation boundary are recorded in [docs/qwen-checkpoint-policy.md](docs/qwen-checkpoint-policy.md). The October blog's Qwen measurements remain explicitly FP8. BF16 was loaded and measured later the same day in the separate study `qwen-bf16-h100-v1` ([findings](reports/qwen-bf16-h100-v1/findings.md)). This does not expand the separate three-model speculative study reviewed below.

Reviewed 2026-09-30 against the local reports, curated per-run JSON, launch scripts and analysis code. This is a plan for future measurements, not a new GPU result. [target.md section 12](target.md#12-later-study-spec-realtext-h100-v1) defines the later speculative objective; [the GPU-session plan](docs/three-model-h100-plan.md) defines its readiness gates, run counts and execution order. The immediate editorial objective is in [target.md](target.md), sections 1–11.

## 1. Scope and binding decision

**Keep every existing prefix-cache result, but run no new prefix-caching experiment for any model until Tan explicitly asks.** This includes cold/warm comparisons, one-touch/two-touch warming, prefix-reuse checks, prefix working-set pressure, and speculative decoding combined with prefix caching. These are deferred, not failed, and are not completion gates for this phase. Disable prefix caching in every active experiment below. Ordinary per-request autoregressive KV state remains necessary; monitoring its occupancy is allowed and is not a prefix-caching experiment.

Active models are DeepSeek-V4-Flash-0731, DeepSeek-V4.1-Flash, and XiaomiMiMo/MiMo-V2.6-Flash-MOPD. Preserve the exact checkpoint revisions from the existing study. Historical preview DeepSeek, GLM and Qwen remain outside this phase. The common protocol can be applied to another model only after that model is added to scope.

Priority: establish each model's **workload-specific strengths and weaknesses on one 8×H100 node**, with fresh AR (ordinary autoregressive decoding) baselines, real-text speculative acceptance, useful tokens per round, draft/verification costs, latency and task checks. Then investigate model-specific runtime limitations. Do not design the study to guarantee a winner or a speculative speedup. H200 is a separate optional follow-up, outside the next H100 session and its completion criteria; hardware availability alone is not a reason to launch it.

## 2. What the existing results establish

The recorded baseline uses one 8×H100 80GB node, TP=8 with expert parallel enabled, memory utilization 0.90, max_model_len=262144, max_num_seqs=64, and an 8192-token chunked-prefill budget. Runtime: vLLM 0.30.1rc1.dev223+g44af287eb. Do not silently revert to older TP-only / 0.82 settings.

On random-token 16K-input/256-output traffic, fixed DSpark gives about 1.55× at c1 for V4.1 and 1.87× for 0731. At c64 it gives about 0.98× and 1.00×. MiMo DFlash k7 gives 1.68× at c1 and 1.01× at c64. MiMo MTP k3 gives 1.53× and 1.02×. These are within-model speedups, not equal-quality cross-model comparisons.

0731 fixed DSpark accepts roughly 45–49% of scheduled candidates, versus V4.1's 31–34%. Acceptance did not collapse with concurrency. Thus the high-load loss of benefit cannot be explained solely by deteriorating draft agreement. Long-input/short-output traffic and verification/scheduling costs need separate tests.

Important limitations: random prompts; no same-seed ordinary-decoding determinism control; no decode-heavy speculative matrix; unavailable adaptive verified-candidate counts; incomplete V4.1 CUDA-graph decode attribution. MiMo's measured MTP reuses layer 0 three times, not all three checkpoint MTP layers. Existing profiles suggest V4.1 kernel/communication disadvantages and inactive CED prefill skipping, but optimization gains remain unmeasured.

### Checked numerical anchors

The endpoint values below were recomputed from valid curated `summary.json` files, retaining all three repeats. Units are output tokens/s, mean ± sample SD. All are historical random-token 16K/256 runs with prefix caching off. MiMo ran the following day on the same node/build; the DeepSeek baselines were not rerun with it. These are comparisons of the measured checkpoint/runtime deployments, including their native quantization and kernel paths.

| Model | AR c1 | AR c64 | Speculative arm | Spec c1 / own AR | Spec c64 / own AR |
| --- | ---: | ---: | --- | ---: | ---: |
| V4.1 | 88.4 ± 0.0 | 292.3 ± 0.4 | Fixed DSpark k5 | 136.7 ± 4.9 / **1.55×** | 286.5 ± 0.3 / **0.980×** |
| 0731 | 98.3 ± 0.0 | 392.4 ± 0.2 | Fixed DSpark k5 | 183.4 ± 6.4 / **1.87×** | 392.8 ± 2.9 / **1.001×** |
| MiMo | 130.1 ± 0.3 | 504.6 ± 0.6 | MTP k3, layer 0 reused | 199.1 ± 3.0 / **1.53×** | 516.6 ± 0.5 / **1.024×** |
| MiMo | Same AR | Same AR | DFlash k7 | 219.2 ± 5.3 / **1.68×** | 512.2 ± 0.3 / **1.015×** |

Rounded SD `0.0` does not mean zero variability. Evidence: [DeepSeek raw runs](reports/v41-vs-0731/data/), [MiMo raw runs](reports/mimo-v26/data/), [DeepSeek speculative tables](reports/v41-vs-0731/dspark_tables.md), [MiMo speculative tables](reports/mimo-v26/spec_tables.md). Compare absolute throughput alongside the multiplier: 0731 has the larger c1 speedup, while MiMo DFlash has higher c1 absolute throughput. Small c64 gains are observations on these three repeats, not robust production recommendations.

### What can be called a strength or weakness today?

| Model | Supported strength on the measured deployment | Weakness or limit | What the next phase must establish |
| --- | --- | --- | --- |
| MiMo | Highest uncached throughput in the reported sweeps; c1 TPOT 5.5–6.3 ms from 1K–260K; 24.1 tok/s at 260K/c8 versus 17.0 (0731) and 14.8 (V4.1) | TPOT rises with context; all-reduce is 44% of the sampled prefill kernel sum; MTP does not use all trained heads | Whether its lead survives real text, long generation and task checks; whether exposed communication limits AR and verification |
| 0731 | Faster than V4.1 on the tested long uncached prompts; flatter c1 TPOT with context; largest fixed-k5 c1 relative speedup | Slower than MiMo in those uncached tests; smaller reported KV pool; some large first-repeat slowdowns | Whether stronger DSpark agreement persists on real tasks and repays verification at high load |
| V4.1 | Lower short-prompt TTFT than 0731 (1K: 148 vs 185 ms, but MiMo is 93 ms); largest reported AR KV pool; historical one-touch prefix reuse advantage | Slowest long-prompt uncached deployment here; Marlin dense GEMM and collective costs; CED prompt skipping absent in the measured build | Real-text low-load behavior, graph-aware decode cost, and a supported single-change runtime ablation before claiming optimization gains |

Historical prefix evidence remains useful but bounded: V4.1 beats MiMo with 16 prefixes warmed once (520.6 vs 303.0 tok/s), whereas MiMo's individual cached-hit check was faster. This is an advantage under the **tested warming procedure**, not a universal ranking of fully warm prefix serving. No new prefix experiments are in scope. See the [DeepSeek report](reports/v41-vs-0731/report.md) and the [MiMo report](reports/mimo-v26/report.md).

### Corrections that govern the new interpretation

- **Bottleneck proof:** `output_rate = input_rate × output_length / input_length` follows from token accounting for fixed lengths. It does not by itself prove a prefill bottleneck. Likewise Little's law relates average in-flight requests, request rate and residence time under steady conditions; substituting the concurrency cap is only approximate during startup/drain. Existing traces and latency changes motivate the hypothesis. Phase timing and the short/long-output controls must test it.
- **Useful progress:** [spec_compare.py](bench/spec_compare.py) computes historical “tokens/round” as `1 + accepted / drafts`. It is a proxy assuming one bonus/correction token per sequence-round, not directly measured committed progress. At V4.1 fixed/c1 the proxy is 2.5741 while total output/draft-count is 2.5670; even the latter mixes first-token and fallback accounting. Preserve the historical table and collect aligned committed counts for the new analysis.
- **Capacity and precision:** the reported AR pools are V4.1 9,179,728, MiMo 6,906,032 and 0731 1,711,998 tokens. Different hybrid-cache accounting, compression and KV formats mean these are not measured concurrent-request limits or equally expensive tokens. MiMo's `auto` KV resolves to BF16; DeepSeek uses `fp8_ds_mla`. Match the serving budget and preserve native formats, disclose them, and avoid an equal-precision or architecture-only claim. No capacity-saturation experiment has been run.
- **Component shares:** rank-0 kernel sums from selected chunks are diagnostic evidence, not whole-node critical-path time. MiMo's 44% all-reduce share is a hypothesis selector, not a promised 44% optimization gain. CED layer counts, FLOP estimates and nominal bandwidth cannot predict an achieved speedup.
- **Correctness and uncertainty:** the old four-prompt smokes do not establish equal model quality. The old `2×sqrt(SD1²+SD2²)` “inconclusive” rule is a descriptive heuristic, not a confidence interval or significance test. New quality and repeat-level inference need their own declared protocol.

## 3. Acceptance economics on H100, and the optional H200 question

**Possible, but not guaranteed. High acceptance is useful only when the saved target work exceeds drafting, verification and state-management cost.**

For a synchronized decode-only diagnostic with B active requests, define:

- T_AR(B,L): ordinary target step wall time at KV length L; it produces B useful tokens.
- T_round(B,L,k): speculative round critical-path wall time, including drafting, verification, scheduling and state updates, with overlap counted once.
- U: actual useful committed tokens across those B requests in that round; g=E[U]/B.
- Approximate decode speedup: S ≈ g × T_AR / T_round.
- Break-even: T_round / T_AR < g.

This approximation requires comparable batches and contexts and a ratio of aggregate progress to aggregate time, not an unweighted mean of per-round ratios. In continuous serving, use measured total useful tokens divided by elapsed time; changing active batch sizes and queueing prevent direct substitution of client concurrency for B. Do not sum overlapping kernel times to obtain T_round.

For fixed-width, linear speculation without early stopping, g is often close to 1+k×a, where a is accepted candidates / proposed candidates and one correction/bonus token is committed. Verify the engine's accounting and use measured g: EOS, truncation, adaptive pruning and fallback rounds change this relation. With k=5 and a=0.80, g≈5. If the speculative round costs 6 ordinary steps, high acceptance still yields only about 0.83× speedup; if it costs 3 steps, it yields about 1.67×. These are illustrative calculations, not measurements.

At high concurrency, ordinary decoding already amortizes weight reads across requests. Verification can schedule roughly B×(k+1) target positions, depending on the algorithm, increasing compute, KV traffic, expert work and padding. Large B does not imply cheap verification, even when almost every candidate is accepted.

NVIDIA's HGX specification lists H100 SXM at 80GB and 3.35TB/s HBM bandwidth, and H200 SXM at 141GB and 4.8TB/s. Both use fourth-generation NVLink with 900GB/s GPU-to-GPU bandwidth. The upgrade therefore does not automatically remove an all-reduce bottleneck. These are platform specifications, not application speed predictions. [NVIDIA specifications](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)

For the same model, workload and arm define:

    S_H100 = Q_spec,H100 / Q_AR,H100
    S_H200 = Q_spec,H200 / Q_AR,H200
    interaction = S_H200 / S_H100

Also report Q_AR,H200/Q_AR,H100 and Q_spec,H200/Q_spec,H100. A higher Q_spec on H200 alone does not show a larger speculation benefit. If acceptance and g stay constant, a larger relative benefit requires the speculative round to accelerate more proportionally than the AR step. Faster HBM might accelerate AR more than compute-heavy verification, making the relative gain smaller. More memory can also permit different batch sizes, but that is a separate deployment-capacity effect.

Hypotheses to distinguish:

| Hypothesis | Evidence needed |
|---|---|
| Better draft agreement unlocks high-load speedup | Natural real-text cases with high g and measured T_round/T_AR below g |
| Verification saturates at large batches | Target verification time and actual scheduled-token shapes grow faster than useful committed tokens |
| H200 improves speculation disproportionately | interaction >1 beyond repeat variation at matched settings and workload |
| H200 mainly improves the ordinary baseline | Higher absolute throughput but interaction ≤1 |
| Communication remains limiting | Exposed collective time persists while local kernels accelerate |

Hardware should not inherently improve statistical draft/target agreement with identical weights, prompts and sampling. Numerical paths and scheduling may still change generated trajectories; measure rather than assume equal acceptance.

## 4. Shared experiments: core first, targeted extensions second

### E0 — Support, correctness and instrumentation gate

Pin target, draft, tokenizer, runtime/container, precision and sampling settings. Save resolved draft classes, actual loaded layers, graph modes and resolved scheduling limits. Verify proposal-width support against the pinned implementation; config field names are not evidence of support.

For each model run a fixed real-text set twice with speculation off and identical request seeds/order, then with every planned speculative method. Start with 24 prompts at c1 and 96 at c64, equally divided among code, math and chat; the latter offers enough requests to reach c64, but is not a tail-latency benchmark. Use natural EOS, a common 512-token cap and explicit thinking-off settings. Compare token IDs and first divergence positions where available; decoded-text comparison alone must be labelled as such. Score code using pinned tests in an isolated executor with time/resource limits, math with a fixed answer checker, and chat with a predeclared rubric. Report truncation, refusals and failures. These are regression/task checks, not comprehensive model-quality evaluation.

Declare correctness acceptance rules before timing: unexplained new task failures or divergence beyond the AR-vs-AR control require investigation; do not recommend an affected arm as quality-preserving while unresolved. Stable greedy AR plus a speculative mismatch warrants diagnosis but is not alone proof of algorithmic lossiness. Separate forced-length timing (`ignore_eos`) from these natural-EOS checks; do not score forced continuation after a completed answer as normal task output. A changed width/runtime needs its own correctness gate. Stochastic sampling is a separate extension: identical seeds need not produce identical sequences across algorithms.

Instrument accepted, proposed and committed counts before the main sweep. Verified counts and component timing can be marked unavailable if the runtime cannot expose them. A correct instrumentation patch is a separately versioned diagnostic runtime; quantify overhead and keep authoritative throughput on the uninstrumented build. Never change acceptance logic to manufacture a headline speedup.

### E1 — Real-text acceptance and concurrency: primary experiment

| Dimension | Required initial setting |
|---|---|
| Workloads | Screen approximately 1K input / 2048 output; then 16K/2048 and matched 16K/256 controls |
| Concurrency | All three shapes at c1/c64; add c4/c16 at 1K/2048; c32 and intermediate 16K loads only if needed to locate a crossover |
| Content | Fixed balanced code/math/chat mixture for the screen; report task checks by domain; separate domain timing is a declared extension |
| 0731 arms | AR off; fixed DSpark k5 |
| V4.1 arms | AR off; fixed DSpark k5 |
| MiMo arms | AR off; MTP k3 (layer 0 reused); DFlash k7 |
| Repeats | Three measured repetitions after representative warmup |
| Cache | Prefix caching disabled for every arm |

Use identical text within each target's off/on pairs. Across different tokenizers, publish actual input lengths. Maintain a same-text real-task view and, where needed, a separately labeled token-count-matched performance view; do not claim both exact text and exact token counts when they differ. Use meaningful documents/examples to construct long prompts, not repeated filler. Pin domain assignments before running; do not choose only prompts known afterward to have high acceptance.

Efficient screen: first use a fixed balanced mixed-domain set at 1K/2048, c1/c64, all seven model/arm combinations, three repeats: **42 runs**. Report domain-level acceptance if per-request instrumentation permits, but the mixed run does not establish domain-specific throughput. Then declare the expanded manifest: 16K/2048, crossover loads, and separate domain runs at informative points. Do not run the entire Cartesian product automatically.

For the throughput screen start at `n=max(16,4×concurrency)` requests, round up for equal domain counts, and increase if needed to obtain at least 60 seconds in the fastest compared arm after a disjoint pilot. Use the same fixed list/order and count across all seven deployments for a point; pair request seeds within each model. Across repeats use a predeclared order/seed schedule shared by compared arms. Preserve startup/drain effects; a central window is secondary and must have a fixed inclusion rule. These counts support screening means/medians, not reliable p95 comparisons. A later latency-validation subset should use at least `max(200,8×concurrency)` requests, duration chosen from the pilot, and uncertainty for its tails. Do not claim production p95 from the small screen.

The fresh 16K/256 control and 16K/2048 arm must use the same prompts/order/seeds at each concurrency; only output budget changes. Together with the 42-run screen these are **126 runs** (7 deployments × 3 shapes × 2 loads × 3 repeats). Adding c4/c16 at 1K/2048 adds **42**, giving **168 core timing runs**, before E0, warmups, pilots and diagnostic profiles. Keep historical random-token results separate. Reusing old AR numbers would confound the new prompt/output distribution and is not allowed.

### E2 — Proposal width and acceptance economics

Targeted tier after E1; declare selected points and incremental cost before collection. It is not an automatic full width × domain sweep or a gate for the core screen.

After E1, choose one natural high-acceptance domain and one lower-acceptance domain using the declared screening results; retain held-out prompts for final validation. Label selection and avoid claiming a universal model property from a favorable subset.

At c1 and one high load (c32 or c64), sweep supported widths around the measured optimum. Candidate sets: DSpark k1/3/5/7, MiMo MTP k1/2/3, DFlash supported widths up to k7. These are requests for support validation, not promises that all widths run. Keep trained block size, drafter depth and requested width separate. Do not edit checkpoint configs to force unsupported points.

Measure whether additional positions increase useful tokens enough to repay marginal cost. A lower aggregate acceptance at larger k can still yield more tokens per round; a higher acceptance at small k can still yield less throughput. Choose the optimum by useful tok/s and latency, not acceptance percentage alone.

### E3 — Decode-only and speculative critical-path profiles

For the core, capture all seven deployments at B=1 and B=64 with initial 1K KV context: **14 short diagnostic captures**, separate from timing runs. Use the baseline widths so both MiMo methods are explained. Complete initial prefill before the measured decode interval; prohibit new prefills in that interval. Record actual KV lengths and batch occupancy as generation proceeds. Extend to B=16, 16K context and tuned widths only to resolve a specific observation. If a fixed-batch harness is unavailable, segment actual decode-only engine steps and report their batch-size distribution; do not relabel mixed serving as decode-only. If no faithful attribution is available, publish that limitation and leave the cost explanation unresolved.

Measure draft backbone and any sequential module, target verification, confidence/admission, rejection/correction, KV/state updates, TP/EP communication, CPU gaps and padding. Record critical-path wall time, overlap and useful tokens, plus actual scheduled verification tokens. GPU utilization percentage alone cannot establish compute or bandwidth saturation. Use targeted hardware counters for dominant kernels when needed.

A high-acceptance/no-speedup case is valuable evidence: identify which cost defeats g. An idealized 100%-acceptance cost bound may be computed from validated measurements, but any modified verifier or forced acceptance is diagnostic only, cannot establish correctness, and must never enter serving rankings.

### E4 — Context sensitivity without prefix reuse

Targeted tier, after the core; declare its manifest separately. This is needed before making a new real-text long-context recommendation, but not to complete the bounded core study.

For each model compare AR and the selected native speculative arm at 1K,16K,64K input, output=2048, c1/c16. Reuse exact compatible E1 points. Extend to 128K only if 64K shows a meaningful change or the research question needs it. Record acceptance by output-position bins (1–256,257–1024,1025–2048), KV length, attention/indexer costs and memory occupancy. Long output is only a candidate decode-heavy workload; establish phase proportions from evidence.

This tests whether long-context verification cost erases speculative gains and whether compressed/sparse versus full-attention paths behave differently. It does not test shared-prefix caching.

## 5. Acceptance metrics: mandatory definitions

| Metric | Definition and interpretation |
|---|---|
| Proposal acceptance | Total accepted candidates / total proposed candidates; aggregate counts, not an unweighted mean of percentages |
| Verified acceptance | Accepted / actually verified; unavailable when actual verification count is not exposed |
| Verification fraction | Verified / proposed; essential for adaptive pruning |
| Useful progress | Actual committed output tokens / sequence-round, with correction/bonus tokens and end truncation documented; include zero-draft/fallback sequence-rounds in a separately reconciled total |
| Position survival s_i | Fraction of eligible rounds accepting the entire candidate prefix through position i; denominator and skipped positions explicit |
| Conditional acceptance | Acceptance at position i conditional on earlier positions surviving and that position being attempted |
| Cost per useful token | Measured critical-path interval time / useful output tokens; batch denominator explicit |
| Acceptance variation | Domain, prompt, context, concurrency and output-position distributions; identify fallback rounds |

Do not reconstruct survival as a^i from one aggregate acceptance rate unless explicitly using an illustrative independence model. DSpark, MTP and DFlash percentages at different widths are not directly comparable measures of draft quality. Report committed progress and cost alongside them. Existing scheduled-draft counters are not actual verified-token counts under adaptive verification. An engine batch-round contains multiple sequence-rounds: record both denominators and reconcile their committed totals with server/client output counts. If only aggregate counters exist, label proxy estimates and leave direct progress unavailable; this limits the mechanism claim, not the validity of independently measured endpoint throughput.

## 6. Model-specific experiments: choose from observed evidence

These are conditional, separately budgeted H100 investigations, not promises that a runtime feature exists. Prioritize one supported intervention that tests the largest **exposed** cost. A negative or unsupported result is useful; do not spend the rental implementing a new CED or MTP engine.

| Model | Targeted question | Experiment and supported conclusion |
|---|---|---|
| 0731 | Why does its DSpark accept more than V4.1 on historical random prompts? | Compare fixed k5 on matched real-text domains, target sampling and contexts; add supported greedy versus probabilistic draft sampling as separate arms. This compares each model's native draft/target pair, not one draft against an identical target. |
| Both DeepSeek | Does adaptive verification help where fixed speculation becomes costly? | At E2's crossover and one high-acceptance high-load point, compare fixed/adaptive with the same maximum width. Instrument admitted verification lengths, padding, confidence overhead and graph/KV allocation. Higher accepted/verified after pruning is not proof of a better drafter. |
| V4.1 | Is slow target execution masking DSpark's value? | Obtain graph-aware decode/verification attribution, then test a supported alternative dense FP8 kernel and communication path one at a time. Pair AR and DSpark under each change; optimizing AR can reduce the relative speculation multiplier while improving absolute speed. |
| V4.1 | Is CED prefill skipping really inactive, and would enabling it help? | Preserve the existing trace evidence. Only add an implementation arm after verifying compatible support, correctness and which tokens/layers execute. Record new runtime separately; never project a 2× measured improvement from layer count alone. Lower priority than speculation profiling. |
| MiMo | Is repeated layer-0 MTP limiting later-position acceptance? | First compare supported k1/k2/k3 and position survival. If a faithful multi-layer implementation becomes available, compare it to repeated-layer-0 on the same target/runtime if possible. If another runtime is needed, rerun AR and available MTP controls there; runtime plus head changes cannot isolate head quality. |
| MiMo | Is DFlash's longer draft worth the work? | Compare supported widths, actual useful progress, draft/verify costs and later-position survival. Validate target feature taps and draft revision. |
| MiMo | Does communication dominate after local compute is reduced? | Profile exposed collective time under AR and speculation. Test supported DP-attention or TP4×DP2 on the same eight GPUs only as a separately labeled tuned deployment; verify model/draft fit and request routing first. Do not assume feasibility. |
| All MoE targets | Does speculative verification amplify expert imbalance? | On selected E3 points compare AR versus verification tokens per expert/rank, padding, dispatch/combine and exposed waits. Analyze draft and target MoE separately where applicable. Routing skew alone is not evidence of a bottleneck. |

Runtime changes and deployment tuning are secondary arms. They must not replace the matched TP8+EP baseline or mix with it in a single curve.

## 7. Optional matched 8×H200 confirmation

Retained as future study design only; start it after a separate decision to run H200. Complete the H100 phase on its own terms. H100 measurements can explain candidate mechanisms, but cannot prove an H200 result.

Select one held-out real-text workload with high observed acceptance and one representative mixed-domain workload. Start with 1K/2048 at c1,c16,c64; add 16K/256 at c64 to test whether the prefill-heavy limitation remains. For each model use AR and the selected native method; MiMo's second method is optional if it addresses a distinct finding. Three repeats, identical prompts/seeds/output policy.

Two separately reported comparisons:

1. **Matched configuration:** 8 GPUs on both machines, TP8+EP, same pinned runtime/checkpoints/precision, same concurrency, max sequences, token budget and context. Keep utilization=0.90 but document that it gives different absolute memory and KV pools. Use workloads comfortably fitting H100, verify no preemption, and optionally cap usable KV blocks equally if the runtime supports it. Record CPU, driver, clocks, power, NVLink topology and actual kernel selection; node differences are confounders.
2. **Best feasible deployment:** only after the matched comparison, tune concurrency or topology separately on each platform under the same declared latency limits. c128/256 are optional feasibility probes that require raising max_num_seqs and warmup; they are not part of the c≤64 matched baseline. Prefix caching remains off. Report failures and saturation, not just the best successful point.

H200 success criterion: the confidence interval for interaction=S_H200/S_H100 is above 1 on the matched workload, while correctness is acceptable and acceptance/cost differences are explained. If speculative throughput increases but interaction does not, conclude hardware improves serving but not the relative benefit of speculation. Do not extrapolate HBM bandwidth ratios directly into application throughput.

## 8. Outputs and decision rules

Publish these tables with raw links:

1. Support: target/draft revisions, method, trained block size, loaded head depth, requested/scheduled/verified width, runtime and unsupported reasons.
2. Acceptance: model, hardware, domain, input/output lengths, c, actual B, width, proposed/verified/accepted/committed counts, per-position survival and variability.
3. Serving: AR and speculative tok/s, speedup, TTFT/TPOT/E2E p50/p95, completions, actual tokens and repeats. ITL must account for multi-token stream chunks.
4. Cost: AR step, draft/verification/other critical-path cost, useful progress, measured versus approximate break-even, overlap and unattributed time.
5. Operating points: all three models' absolute throughput, latency and `8/output_tok_s` allocated GPU-seconds/token on H100; include same-concurrency and measured overlapping-throughput comparisons. Show Pareto curves when no application latency limits were supplied. Hardware interaction tables belong only to a separately executed H200 study.
6. Feature ablations: one changed variable, paired baseline, runtime/deployment differences, quality, measured benefit and remaining confounders.

Keep raw failed runs and reruns immutable. Warm representative shapes before timed repeats; alternate arm order across launch cycles and disclose model-order blocks. Report means, SD, medians and all repeat ratios, including slow first runs. Pair by manifest identity (model, prompt hash/order, seed, output policy, load, configuration and repetition), not directory suffix: `repeat-1-rerun1` may be the valid match for `repeat-1`. A statistical pair also needs a declared repeat/block design; shared seed alone does not remove time drift. Keep one selected valid attempt per planned repeat under a predeclared failure/rerun rule, never choose the fastest attempt.

Three repeats are a screen. For a near-break-even finding or an operating recommendation, confirm on held-out prompts with at least five independently scheduled repeat blocks (more if unresolved). Predeclare a practical gain threshold, proposed **5% throughput** at acceptable latency and task checks. A gain smaller than this may be measurable but is not automatically worthwhile. Use repeat-block uncertainty on paired log-ratios; bootstrap intervals with only three blocks are not strong evidence. Report inconclusive findings when the interval crosses break-even or the practical threshold. Tokens and overlapping windows are not independent trials. Cross-model task quality and request-level tail latency need separate sample-size justification.

The core uses closed-loop concurrency caps. It does not establish stability under external arrivals or production SLO goodput. If a deployment SLO is supplied later, add an explicitly budgeted open-loop arrival-rate test (common arrival trace, queue drain, all failures counted) and report completed requests meeting **both** latency limits per wall second. Without that test, provide workload-specific screening operating points, not a production capacity guarantee. Dollar cost requires an explicit node/GPU hourly price; allocated GPU-seconds/token is not energy consumption.

The equation output_rate=input_rate×output_length/input_length is an accounting identity for fixed lengths; it does not prove a prefill bottleneck. Establish bottlenecks using phase timing and interventions. Separate measured findings, source-supported implementation facts and hypotheses. No speed ranking establishes equivalent task quality.

## 9. Required harness changes before running

- Parameterize output length: the current run_matrix.py hardcodes OSL=256 and validity expects n×OSL. Add output_len per workload to commands, manifests, checks and analysis. Fix downstream filenames/parsers that assume 256.
- Add pinned real-text datasets with hashes and per-request domain IDs; use unique requests with prefix caching off. Save actual token counts and model-specific thinking switches.
- Pin the client tokenizer and remote code, not just the server weights. `bench_cmd` currently passes a model ID/tokenizer mode without a revision; use validated immutable snapshot paths or supported revision arguments on the GPU build. Save rendered chat inputs and token counts, with public-data/licensing checks before curation.
- Add per-request/round acceptance logging where supported; retain raw engine metric definitions. Instrument adaptive verified counts rather than relabeling scheduled counts.
- Add supported width/policy arms with explicit draft pinning. Validate actual head depth and fallback execution.
- Add a decode-only diagnostic runner or faithful trace segmentation with actual active batch counts.
- Add hardware/study IDs to every result key and pairing rule. Existing comparison scripts hardcode model/arm maps; extend them without overwriting historical tables.
- Match reruns by manifest identity and compare token IDs where possible. `compare_outputs.py` currently requires matching directory names and compares decoded characters, which omitted MiMo's valid AR c1 rerun from its old match table.
- Parameterize max_num_seqs only for separately declared c>64 arms. Keep default chunk budget fixed unless a scheduler experiment explicitly changes it.
- Add a new phase-specific, manifest-driven chain using `chain_lib.sh`; reject prefix-enabled arms, historical study IDs and unsupported model/method combinations before launch. Current `run_matrix.py` defaults to concurrency/context (not prefix), but `chain_mimo.sh` unconditionally runs prefix stages and both old chains hardcode historical study IDs. Keep them as historical recipes, never call them as the active phase chain.
- Fail closed on identity/config mismatches, output-length errors, missing required token validation or readiness failures. Save the attempted manifest before subprocess execution, bound run/launch timeouts, stop the owned server on every exit, and keep failure status separate from an unavailable diagnostic counter. Dry-run the full chain before reserving GPUs. These are required extensions, not capabilities already implemented by this documentation change.

## 10. Active documents and completion boundary

[target.md](target.md) now covers all three models. [The GPU-session plan](docs/three-model-h100-plan.md) specifies the bounded core (E0, E1 and initial E3), targeted extensions, launch order and required harness work. [AGENTS.md](AGENTS.md) is the next-session handoff. The reproduction and speculative-support guides distinguish historical recipes from this phase. Existing reports and numerical artifacts remain unchanged.

Use fresh study ID `spec-realtext-h100-v1`. No new GPU runs or new real-text harness are claimed by this documentation update. Core completion requires every declared point to be validated or documented failed/unsupported, no unexplained omission, and a three-model strengths/weaknesses table whose claims match the available evidence. An unavailable profile permits a descriptive serving result, but leaves its causal explanation unresolved. Pending core points mean the core is incomplete. E2/E4, domain-specific performance, optimization and production-SLO claims require their separately declared validation; deferred prefix and H200 work never block core completion.

Chain launches, run measurements promptly after readiness, and stop servers when each arm finishes. Estimate node-hours from a pilot and count restarts, warmups, correctness checks, repeats, profiles and reruns. Record `GPU-hours = 8 × node-hours`. The old 256-output timing estimates are not a budget for this phase. Finish local harness preparation before starting a rental.

## Sources

- [R1: DeepSeek results and caveats](reports/v41-vs-0731/report.md)
- [R2: MiMo results and runtime limitations](reports/mimo-v26/report.md)
- [R3: Active target and comparison rules](target.md)
- [R4: Speculative support and metric definitions](docs/speculative-decoding.md)
- [R5: Existing harness](bench/run_matrix.py)
- [S1: NVIDIA HGX H100/H200 specifications](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html), accessed 2026-09-30.
- [S2: Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), foundational speculative-decoding algorithm. Cost equations and hardware hypotheses in this plan are analytical reasoning, not predictions from that paper for these checkpoints.
