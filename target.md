# Target: answer Kan's architecture questions on 8×H100, and publish a blog everyone can use

Updated 2026-10-10. **This is the single target file of the five-model publication branch.** It states what the measurements are for, which checkpoints they cover, what is answered so far, and the [planned five-model rerun](#five-model-rerun-plan). The completed evidence comes from the full archive's [target.md](https://github.com/tant2tls/Deepseek-serving/blob/4cfa8d5f4d4fc7da3ed6248fda8e5a09004f6bcf/target.md) on `all-data`; the rerun is a new proposal, not a revision of a frozen study.

**Two purposes, in this order:**

1. **Measure, to answer Kan's question.** Each part of [the brief](#the-brief) gets an answer backed by evidence collected on 8×H100, or is stated plainly as unanswered.
2. **Publish a blog that is useful to everyone who reads it.** A reader new to inference systems can follow every figure, and a practitioner can trace every number to its run, build and node.

The user requested rerun planning, including speculative decoding, on 2026-10-10, and asked the same day that the plan answer every part of Kan's brief and test each model's strengths and weaknesses. This update defines that plan; no new measurements have been run. GPU rental and execution await an explicit session allocation. Prefix caching will not be measured: the user confirmed this on 2026-10-10. Kan's prefix-caching question is therefore answered only as far as stored state bytes allow. The rerun measures one output length, 2,048 tokens; the 256-token numbers of the completed October study are a reference only and are not measured again (user decision, 2026-10-10). The rental is one 8×H100 node for 15 hours (user decision, same day): [the 15-hour session](#the-15-hour-session) is the part of this plan that runs, and the rest is deferred. Prefix-cache experiments remain outside scope. Sessions on a 4×H200 and a 4×B200 server, which add speculative decoding, have their own guide: [docs/H200_B200_plan.md](docs/H200_B200_plan.md).

## The brief

Kan (SyFI lab) asked:

> We want to teach people architecture of new generation of models. Specifically we want to compare the attention and FFN side. Which model's attention take minimal runtime loading/HBM or runs fastest? For prefix caching, which model is less space consuming? FFN side what are there sparsity differences? Feel free to add more!

The [article](index.html) explains the completed measurements using each model's architecture report and its actual inference path, so that the measurements and the explanation agree and a reader new to these models can follow them. The [TraceLab post](https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/) is the presentation reference: an example of style, not evidence about these checkpoints.

**The one idea the reader should leave with:** fewer attended positions or fewer active parameters do not automatically make a model faster. A deployment can win one component and still lose overall, because the cost also depends on the work needed to *find* the sparse subset, on the kernel that executes it, on precision, and on communication between GPUs.

## What is answered so far

| Kan's question | What was measured | State | Evidence |
| --- | --- | --- | --- |
| What is the architecture of each model? | Layer types, shapes, precision and the kernels the runtime really loads | **Answered** from config, server log and trace on the blog build | [Article](index.html); [models](docs/models.md); [per-model guides](reproduce/README.md) |
| Whose attention runs fastest? | Attention-path time in prefill (16K and 64K) and decode (1K and 64K, engine batch 1 and 8) | **Answered** at those points: diagnostic kernel sums, read beside unprofiled timing | [Trace components](reports/five-model/results.md#trace-components) |
| Whose attention moves the least data through HBM? | Hardware read/write counters were planned | **Unanswered.** No counters were collected (no Nsight Compute on the image); byte-traffic statements are estimates. Rerun: a shape-based estimate for all five and a hardware-counter attempt | [Experiments and limits](docs/experiments.md) |
| Which model's cached state takes less space? | Bytes held by live requests' KV/state, six snapshots per model | **Partly answered.** Live state is measured as an approximation (usage gauge × per-rank pool). Reusable-prefix capacity is **unanswered**: prefix caching was off. Rerun: stored bytes per cache group, with caching still off; prefix caching will not be measured | [Live state memory](reports/five-model/results.md#live-state-memory-at-64k-b1) |
| What are the FFN sparsity differences? | Nominal expert structure, and trace-level expert and routing time | **Partly answered.** Routing histograms, expert coverage and padding statistics were unavailable. Rerun: [expert-routing statistics](#experiments-for-the-blog) as a required attempt | [Trace components](reports/five-model/results.md#trace-components) |
| "Feel free to add more" | TTFT, TPOT and throughput on public text, from unprofiled runs | **Answered for 256-token outputs.** The article displays all five deployments; the node control limits cross-node latency interpretation | [Output throughput](reports/five-model/results.md#output-throughput); [protocol](docs/experiments.md#two-physical-nodes) |
| Does the ranking hold during sustained generation and as context grows? | Only short-output timing and sampled decode points exist | **Planned, not measured:** 2,048-token outputs at three inputs and two loads in the [15-hour session](#the-15-hour-session), with context-scaling diagnostics; the other loads are deferred | [Rerun plan](#five-model-rerun-plan) |
| When does speculation improve each deployment? | Historical evidence uses other workloads/builds; no matched five-model rerun exists | **Deferred on H100:** outside the 15-hour session. Planned for the H200 and B200 servers ([guide](docs/H200_B200_plan.md)): supported native methods against fresh AR controls | [Speculative comparison](#speculative-comparison) |
| What turns a speed ranking into a useful architectural explanation? | Existing traces suggest mechanisms but do not isolate their causal gains or serving capacity | **Deferred:** the bounded mechanism, capacity and workload checks are outside the 15-hour session | [Experiments for the blog](#experiments-for-the-blog) |
| What is each model good at, what is it bad at, and why? | The October numbers suggest a pattern for each model; none has been tested as a claim | **Planned, not measured:** six cross-model and ten per-model claims, each with a deciding measurement and a failure condition. Parts that need a deferred experiment are reported as untested | [Claims the rerun must test](#claims-the-rerun-must-test) |

The [answer map](#answer-map-for-the-brief) names, for each of Kan's questions, the measurement that decides it and what the blog will say if that measurement is blocked.

The answers themselves, with their numbers and limits, live in the [five-model results](reports/five-model/results.md) and the [article](index.html). This file states the target and its state; it does not restate results.

## Models and immutable identities

Qwen3.8-Flash-Next denotes the original BF16 checkpoint throughout.

| Key | Checkpoint | Immutable revision | Node |
| --- | --- | --- | --- |
| `v4-0731` | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | First |
| `v41` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | First |
| `mimo-v26` | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | `2479e2d0029eca9a34cc7e7f55a121925f81908e` | First; node control on the second |
| `qwen-38-bf16` | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Second |
| `glm-53` | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | First |

Pin the tokenizer and remote code with the checkpoint. [docs/models.md](docs/models.md) links each model's card and technical report and states what the runtime executed. The [per-model guides](reproduce/README.md) document the actual runtime paths, including the BF16 Qwen expert backend and GLM's thinking exception.

## Completed comparison: setup and boundaries

- **Runtime:** vLLM `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`) for all five. The earlier DeepSeek/MiMo studies used `44af287ebe38d6dc4e102948025f5e3e175aefd6`; a number from one build is never a control for the other.
- **Deployment:** 8×H100 80GB SXM, TP8 plus expert parallel, utilization 0.90, context limit 262144, at most 64 sequences, explicit 8192-token chunk budget, native precision.
- **Workload:** the same public code/math/chat text for every model, approximately 1K/16K/64K input, 256 forced output tokens, one or eight clients, three timing blocks. Autoregressive decoding only: speculation off and prefix caching off in every launch.
- **Two physical nodes:** four deployments on the first node; Qwen and a MiMo control on the second. The article displays all five in each serving metric at the user's request. The control reproduced throughput closely but not latency; displaying latency together does not remove that limitation. The rerun will use one physical node for all five.
- **Thinking:** GLM has no thinking-off switch and runs with `reasoning_effort=low`; the other four disable thinking.
- **Scope of a result:** these compare deployments (checkpoint, precision, kernels and this build) on these workloads. They do not rank architectures in isolation and say nothing about answer quality.

The [experiment protocol](docs/experiments.md) defines the measurements, counts and limits in full; [reproduce](reproduce/README.md) gives the commands.

## Evidence labels and terms

Every statement in the blog carries exactly one label:

| Label | Meaning | Example |
| --- | --- | --- |
| Architecture fact | What the paper, model card or checkpoint config says | "Top-8 of 256 experts" |
| Executed implementation | What the pinned runtime verifiably loads and runs | "Dense FP8 layers resolve to the Marlin kernel" |
| Measurement | Time or bytes recorded on the node, with its scope | "ms per 8,192-token chunk, rank 0, kernel sum" |
| Estimate | Arithmetic from shapes; never presented as measured | "≈ 14.2B linear weights touched per token" |

Architecture descriptions and config field names do not prove execution.

| Term | Plain meaning |
| --- | --- |
| Prefill | Processing the input prompt. Many tokens are processed per step, in chunks of at most 8,192 scheduled tokens |
| Decode | Generating output, one new token per active request per step |
| TTFT / TPOT | Time to first token / time per output token after the first |
| KV state | Keys and values (plus any index or compressed state) that attention keeps for earlier tokens of a live request |
| HBM traffic | Bytes read from and written to GPU memory during inference. An H100 SXM is specified at 3.35 TB/s; that is a platform specification, not a prediction |
| MoE | Mixture of experts: the FFN is split into many experts and a router picks a few per token |
| Routed / shared expert | A routed expert runs only for tokens sent to it; a shared expert runs for every token |
| TP8 + EP | Tensor parallel over 8 GPUs (each GPU holds a slice of every layer) plus expert parallel (experts distributed across GPUs) |
| All-reduce | The collective operation that sums partial results across the 8 GPUs; it happens inside every layer under TP |
| Client c vs engine B | c is how many requests the client keeps open; B is how many sequences the engine actually puts in one step. They are not the same number |
| Kernel sum vs wall time | Adding GPU kernel durations versus elapsed time. Kernels on different streams and ranks overlap, so a sum is not a latency |
| Rank | One of the 8 GPU worker processes |

## Earlier studies

The September DeepSeek and MiMo studies ran on the older build with random-token prompts. Their reports now sit beside their evidence: [reports/v41-vs-0731/report.md](reports/v41-vs-0731/report.md) and [reports/mimo-v26/report.md](reports/mimo-v26/report.md). They selected hypotheses for this work and are never controls for the October numbers; read them with the corrections in [earlier experiments](docs/previous-experiments.md).

## Still open

- **Measured HBM traffic.** Needs Nsight Compute installed and validated on the node before a rental. The rerun also requires a shape-based estimate for all five, so the question keeps a labelled answer if no counter can be collected.
- **Expert-routing statistics** (tokens per expert, coverage, padding). Need runtime instrumentation; the rerun makes them a [required attempt](#experiments-for-the-blog).
- **Reusable-prefix capacity and reuse.** Not measured, by decision: prefix caching stays off in every launch. Live state bytes do not establish it.
- **Answer quality, saturation and production tail latency** were not measured.

The archived `spec-realtext-h100-v1` study remains planned, not measured. Its three-model design informs the new five-model proposal below; its old runtime, run counts and completion status are not carried over as new measurements.

## Five-model rerun plan

**Status: proposed; harness changes, pilots and collection pending.** Proposed new study identity: `five-model-rerun-h100-v1`. Preserve all completed reports and curation ledgers byte-for-byte. New numbers must live under a new study identity and must not replace the article's results until reviewed and explicitly selected for publication.

The objective is to answer each part of Kan's brief and to identify where each deployment is strong or weak: prompt processing, sustained generation, context growth, live state, expert computation and supported speculation. Distinguish a deployment observation from an architectural explanation. An architectural cause needs source/trace evidence and, for a claimed causal speedup, a matched intervention.

This proposal follows the [archived speculative target](https://github.com/tant2tls/Deepseek-serving/blob/5732235e0ea75ba33944f7f653df2d0d8e3cfa77/target.md#12-later-study-spec-realtext-h100-v1) and [concrete run plan](https://github.com/tant2tls/Deepseek-serving/blob/5732235e0ea75ba33944f7f653df2d0d8e3cfa77/docs/three-model-h100-plan.md), inspected from `origin/all-data`. It extends the model set and adds sustained-generation and context measurements. It does not execute or complete the archived study.

### The 15-hour session

**Status: proposed. This is the part of the plan that fits the rental.** The user set the budget on 2026-10-10: one 8×H100 node for 15 hours, spent as close to Kan's questions as possible. The session is autoregressive only. Everything else in this plan is deferred: it stays pending, and it is neither cancelled nor run.

Kan asks about attention, stored state and the FFN, so the diagnostics run first and keep their full scope. Serving runs come last and are cut to two loads.

| Order | Stage | Serves | Hours |
| ---: | --- | --- | ---: |
| 1 | Setup: download, install the frozen build, record the node, verify inputs, dry run | Everything | 1.25 |
| 2 | One diagnostics launch per model: first load; architecture and support check from the log; functional check with 12 natural-ending prompts; pilot of the session's serving points; 12 trace captures (prefill at 1K/16K/64K/128K, decode at actual B=1/8 on the same four contexts); state snapshots at the same eight decode conditions | Architecture; attention speed; stored state; expert time | 2.75 |
| 3 | Expert-routing statistics: one instrumented launch per model, as defined in [Experiments for the blog](#experiments-for-the-blog) | FFN sparsity | 1.25 |
| 4 | HBM counters on the attention kernels, as defined in [Experiments for the blog](#experiments-for-the-blog) | HBM traffic | 1.0 |
| 5 | Timing: three blocks, each one plain launch per model in rotating order. Every launch runs the six serving points below, plus unprofiled decode intervals of 512 tokens per request at actual B=8 on 1K/16K/64K/128K and at B=1 on 128K | Attention speed without a profiler; "add more" | 6.7 |
| | Reserve: failed launches, reviews, a slow download | | 2.05 |
| | **Total** | | **15.0** |

| Serving points in the session | Output tokens | Clients | Requests per run | Runs at three blocks, five models |
| --- | ---: | ---: | ---: | ---: |
| 1K, 16K, 64K | 2,048 | 1 | 12 | 45 |
| 1K, 16K, 64K | 2,048 | 16 | 48 | 45 |
| **Total** | | | | **90** |

One client gives TTFT (prompt cost) and TPOT (one decode step at batch 1) with nothing else in the engine, which is the cleanest serving view of Kan's attention question. Sixteen clients show one batched load. The request counts are below the starting rule in [Serving matrix and repetition](#serving-matrix-and-repetition) so that three blocks fit; they stay balanced across code, math and chat (4 and 16 per domain), and by the estimate the shortest run still lasts about 80 seconds.

Rules for the session:

- **Three blocks behind every published serving number.** If time runs short, drop points, never blocks.
- **Gates on stages 3 and 4.** Both need instrumentation that did not exist in October. Each stops if it has no valid result on the first model within 20 minutes, and stops at its hour limit in any case. The blog then prints the fallback from the [answer map](#answer-map-for-the-brief). The shape-based HBM estimate is computed before the rental and does not depend on stage 4.
- **Checkpoint before timing.** One full block takes about 2.25 hours. Choose the serving matrix before block 1 and keep it for all three blocks. If less than 7.5 hours remain, drop c16 at 16K, which saves about half an hour. If that is not enough, drop c16 entirely, which saves about 1.75 hours in total. The one-client runs and the decode intervals are not dropped.
- **Downloads.** Start them in measurement order. Diagnostics may begin on a model whose download has finished; timing blocks start only after every download has finished.
- **Unused time is returned.** It is not filled with a measurement this section does not list.

Not in the session, and therefore pending: speculative decoding (screen and confirmation); client loads 4, 32 and 64; the five held-out blocks; the 60-task natural-ending set; the drift reference runs; and the five conditional extensions (domain comparison, mixed traffic, capacity test, V4.1 replay control, expert-kernel control). A difference inside the practical threshold or the spread between blocks is reported as unresolved rather than chased with more blocks.

The hours are an estimate, not a measurement. Run times come from October's TTFT and TPOT per model, in a model that reproduces October's own benchmark windows within about 9%; launch and capture times are scaled from October's logs. Decode speed at 16 clients is extrapolated from 1 and 8 clients. By the same estimate the full plan needs about 65–80 node-hours, or 40–50 without speculation, which is why most of it is deferred.

### Answer map for the brief

Each question Kan asked has one deciding measurement. A row counts as answered only when that measurement exists for all five models on the single rerun node. Otherwise the blog prints the fallback in the last column, with its evidence label. The prefix-caching row is the exception: prefix caching will not be measured, so the rerun can answer it only in part.

| Kan's question | Deciding measurement | Stage in this plan | If the measurement is blocked, the blog says |
| --- | --- | --- | --- |
| What is the architecture of each model? | Layer types, shapes, precision and loaded kernels, read again from the rerun's own server logs and traces | [Shared system and workload](#shared-system-and-workload) | October's implementation notes, labelled with their older build |
| Whose attention runs fastest? | *Attention-path* time, defined [below](#architecture-measurements-and-hypotheses), per 8,192-token prefill chunk and per decode step, at 1K to 128K and actual B=1/8, beside the unprofiled whole-step time | Prefill and decode scaling in [Architecture measurements](#architecture-measurements-and-hypotheses) | The kernel-sum order for each phase and context, stated as a sum and not as a latency |
| Whose attention moves the least data through HBM? | Two numbers per model: an estimate of the state bytes read and written per step, from executed shapes, and hardware counters on the attention kernels | FLOP/HBM row in [Architecture measurements](#architecture-measurements-and-hypotheses); HBM-counter attempt in [Experiments for the blog](#experiments-for-the-blog) | The estimate alone, labelled Estimate; measured traffic stays unanswered |
| For prefix caching, which model takes less space? | Stored bytes per token for a live request, per cache group and summed over eight GPUs including replicated copies, with caching off. This shows which model stores less per token. It does not show what a prefix cache keeps after a request ends, or whether a later request can reuse it | Live-state row in [Architecture measurements](#architecture-measurements-and-hypotheses) | The usage gauge × pool approximation, as in October, with the same limit stated |
| What are the FFN sparsity differences? | Nominal sparsity (k of N experts, expert width, shared experts) beside realized sparsity: distinct experts used, tokens per expert, load imbalance, padding and the share of expert weights touched per step | Expert-routing statistics in [Experiments for the blog](#experiments-for-the-blog); expert-cost row in [Architecture measurements](#architecture-measurements-and-hypotheses) | Nominal sparsity and expert kernel time; realized routing stays unavailable |
| "Feel free to add more" | TTFT, TPOT and throughput with 2,048-token outputs at 1 and 16 clients and three inputs, all five on one node. Deferred: the other loads, tested capacity, mixed traffic and speculation | [The 15-hour session](#the-15-hour-session); the full designs are in [Serving matrix](#serving-matrix-and-repetition), [Speculative comparison](#speculative-comparison) and [Experiments for the blog](#experiments-for-the-blog) | Each unrun cell stays pending |

Kan's phrase "runtime loading/HBM" is read here as data moved through GPU memory while the model runs. Start-up load time and weight bytes per GPU are different quantities; October's values are in the [reproduction budget](reproduce/README.md#what-a-reproduction-needs-disk-memory-and-time), and the rerun records them again on its node.

### Output length decision

**The rerun measures one output length: 2,048 tokens, for both autoregressive (AR) and speculative decoding, at every input and load. The 256-token numbers are the completed October study; they stay as a reference and are not measured again. There is no 8,192-token tier.** The user chose this on 2026-10-10 so that every new number has the same standing and none rests on a check at a few points. These are design choices, not universal statistical sufficiency thresholds.

| Output length | Role | How to read it |
| --- | --- | --- |
| 256 | Reference only: the [published October results](reports/five-model/results.md) | Not measured in the rerun. It used an older build and two nodes, so a difference between it and a rerun number mixes build, node and output length. It is never a control for the rerun and cannot show the effect of output length |
| 2,048 | The only measured workload, all inputs and loads | Eight times the published generation length: sustained decode work with enough budget for varied prompts, loads and independent repeats; matches the archived speculative plan |

An 8,192-token tier was considered and dropped. At four points per model it could only be a check, and it was expensive: with 24 requests and October's single-request TPOT of about 7 ms, its single-client runs alone would take about 11.5 hours, against about 4.3 hours for every 2,048-token single-client run together. Those hours are an estimate, not a measurement. A longer output does not make a number more reliable; repeated blocks do. In October's 256-token runs the spread between three blocks was already under 2% of the mean at 26 of 30 points ([throughput](reports/five-model/results.md#output-throughput)). Every 2,048-token point gets the same three blocks, and [five held-out blocks](#serving-matrix-and-repetition) stand behind any final recommendation or small difference.

Nothing beyond 2,048 generated tokens is measured, and the blog must say so. Whether decoding slows later in a response is read from position windows inside the 2,048-token runs. For longer contexts it is read from the decode-scaling diagnostic at 1K to 128K, which measures step time against context directly. With a 64K input, another 2,048 tokens grow the context by only about 3%, so output length cannot replace the input-context sweep.

With no short-output run, throughput in the rerun leans toward decode speed, most of all at one client. Prompt cost is read from TTFT in the same runs and from the prefill-scaling diagnostic at 1K to 128K.

Every AR/spec pair uses the same 2,048-token output.

### Shared system and workload

- Run all five pinned checkpoints above on **one exclusive physical 8×H100 80GB SXM node**, sequentially. Record CPU/NUMA, GPU topology, driver, clocks/power and dependencies; finish downloads before timing. If the node changes, create a separate hardware block and rerun all compared deployments there.
- **Use the latest upstream vLLM build available when preparing this rerun**, as requested on 2026-10-10. Resolve the latest main/nightly build once, then freeze its full commit SHA, package version, wheel hash or container digest, CUDA/PyTorch versions and dependency lock before the pilot and measurements. The latest nightly resolved on 2026-10-10 is `c41b2639e29c3bc01add1d34bef3032a6d9d8aca` (vLLM `0.31.1rc1.dev261+gc41b2639e`), installed for the B200 session as `~/vllm-b200` by [install.sh](install.sh). It is not yet smoke-tested on a B200, and its wheel hash and lock are not recorded; neither historical build above is the rerun default. Follow the [official installation guidance](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for commit-specific artifacts. Use that one frozen environment for all five models and matched AR/spec arms; never upgrade midway through the matrix. A necessary later upgrade creates a separate runtime block with fresh controls, not a continuation of the same block.
- Recheck all model loaders, graph paths, attention/expert backends, KV formats, thinking controls and speculative support on the frozen latest build. Historical implementation notes are a checklist, not proof of current behavior. In particular, record V4.1's actual full-prompt/replay layer split, replay window and trim threshold from source, startup logs and traces; do not assume the previously measured 21/19 split still applies. Unsupported methods retain their status rather than silently falling back to an old runtime.
- Keep TP8 + EP, memory utilization 0.90, context limit 262144, max sequences 64, explicit 8192-token prefill chunk budget, text only and native precision. Record actual weight/KV formats, loaded kernels and graph settings. Qwen stays original BF16; GLM stays native FP8. Prefix caching remains off in every launch. AR launches have speculation off.
- Keep target temperature zero and fixed request seeds; thinking off where supported, GLM `reasoning_effort=low`. Count and retain reasoning tokens where emitted. This is not equal-answer-quality control.
- Freeze balanced public code/math/chat prompts with source/license, exact text, request hashes and disjoint pilot, screen and held-out splits. Use the same texts/order across models; report actual rendered token counts. Add a separately labelled token-count-matched diagnostic view when comparing architecture scaling, because equal text is not equal tokens.
- Separate **forced-length timing** (`ignore_eos`, every request validated at 2,048 tokens) from **natural-EOS task checks** with a 2K cap. Record natural lengths, truncation, task outcomes and output repetition. Use prompts that genuinely request sustained output. Forced post-EOS/repetitive tails can distort speculative acceptance; they must not support a claim about useful long answers. Natural-EOS speedups need their own matched results and actual-length distributions.
- Confirm template length + requested output + speculative lookahead fits both target and draft limits. Insufficient memory or unsupported configurations remain explicit failures/unsupported points, not silently smaller batches or changed precision.

### Serving matrix and repetition

**In the 15-hour session only 1 and 16 clients run, with the request counts given in [The 15-hour session](#the-15-hour-session): 90 runs. The full sweep below, 225 runs, is the complete design; its loads 4, 32 and 64 are deferred.**

Here K denotes powers of two for requested budgets: 1K = 1,024, 16K = 16,384, 64K = 65,536; real-text input buckets are approximate and actual counts are saved. Client concurrency `c` is distinct from actual engine batch `B`.

| AR stage, all five models | Input buckets | Output tokens | Clients | Runs at three blocks |
| --- | --- | ---: | --- | ---: |
| Main sustained-generation comparison | 1K, 16K, 64K | 2,048 | 1, 4, 16, 32, 64 | **225** |

That is 5 models × 3 inputs × 5 loads × 3 blocks. No 256-token run is part of the rerun.

**The main concurrency sweep is c1, c4, c16, c32 and c64**, for both AR and the speculative screen. c1 establishes single-request responsiveness; c4 tests light batching; c16 tests a moderate load; c32 helps locate a throughput/latency crossover; c64 probes high-load scheduling and memory pressure. These roles are experimental questions, not a claim that a given point saturates every model. c8 is not a point in the sweep; batch 8 appears only as actual B=8 in the context diagnostics.

Plot output throughput, TTFT, TPOT and end-to-end latency against client concurrency, and plot paired speculative speedup against the same loads. Record actual engine-B distributions, running/queued requests, cache occupancy, preemptions and failures. Client c64 means up to 64 outstanding requests; it does not promise 64 simultaneously decoding sequences. At 64K input, KV capacity may force queueing even on eight H100s. Queueing is part of a valid serving measurement; OOM/failure is not a valid completed timing run. Keep memory-limited serving behavior separate from isolated compute-scaling diagnostics, and never silently lower the requested client cap.

Pilot in ascending load order to establish feasibility; rotate the frozen measured workload order afterward. Increasing concurrency multiplies live state and verification work; a useful speculative gain at c1 may disappear at c32/c64.

Each run is a request set, not a single request. Begin a disjoint pilot with `max(24, 4c)` requests, rounded up to a multiple of three for domain balance. Increase to reach at least 60 seconds in the fastest compared arm; then freeze the same request count/list across models and paired methods at that point. Do not copy the historical 256-token request counts blindly. If pilot counts must change after collection starts, preserve the earlier block and collect the added matched block for every compared arm.

Use three separate launch/time blocks for screening, with a predeclared rotating order of all five models, alternating workload order and balanced AR/spec order. Keep warmups separate and retain slow valid blocks. For a final recommendation or a small difference, collect at least five independent held-out blocks at the selected points, evaluate paired block uncertainty and a predeclared practical threshold (proposed: 5%). Five blocks are a starting requirement, not a guarantee that a difference can be resolved. Requests and tokens within one run are not independent repeat blocks.

Publish mean, median, sample SD, block values and paired ratios in the evidence. The blog may show average-only labels, but unresolved differences remain unresolved. Report p95 descriptively unless a separately sized tail-latency study is performed; this closed-loop plan does not establish production SLOs or saturation.

### Architecture measurements and hypotheses

Longer outputs do not isolate prefill, total decode FLOPs or memory traffic. Pair the serving results with these AR diagnostics on all five models:

| Question | Measurement | Interpretation boundary |
| --- | --- | --- |
| How does prefill scale? | 1K/16K/64K/128K inputs at c1; unprofiled TTFT and engine prefill elapsed time if available, with separate short captures of first/late chunks | Include every executed layer and window replay. TTFT includes more than GPU prefill; a chunk kernel sum is not whole-prompt elapsed time |
| How does decode scale with context? | Actual B=1/8 at initial 1K/16K/64K/128K; synchronize after prefill, admit no new requests, record actual context and committed progress in a short decode-only interval | Separate attention core, indexer/search, recurrent updates, projections, experts, communication and whole-step elapsed time. Report ratios/slopes against actual context |
| Does generation slow within a response? | On the 2,048-token runs, report generated-position windows 1–256, 257–1,024 and 1,025–2,048, with matching context and batch distributions | Serving windows can include other requests' prefill. Use isolated diagnostics to distinguish context growth from scheduling; retain shorter/EOS responses in the natural-length report. Nothing beyond 2,048 generated tokens is measured; longer contexts come from the decode-scaling row |
| How much live state is saved? | Snapshots at the same initial contexts and B=1/8, plus matched checkpoints at generated positions 256, 1,024 and 2,048 during a 2,048-token generation | Prefer per-cache-group occupied block bytes, including replicated state; report recurrent state, weights, workspace and reserved pools separately. Gauge × pool remains an approximation if exact allocation is unavailable |
| Which work consumes FLOPs and HBM bandwidth? | Check executed shapes/source for operation-count estimates; collect validated hardware FLOP/HBM counters on short diagnostic steps where supported | Report estimated versus hardware-measured quantities separately. Flat latency is not proof of constant FLOPs; KV capacity is not HBM traffic |
| What explains expert cost? | Expert GEMM and routing times/shapes, active experts, tokens per expert and padding where instrumentation is available | Nominal top-k and precision do not establish actual traffic or throughput. Missing routing/counter telemetry remains unavailable |

The context diagnostics comprise **5 models × 4 contexts × 2 batches = 40 decode conditions**, plus **20 c1 prefill conditions**. Repeat their unprofiled timing intervals across three blocks; collect bounded profiles and memory snapshots separately. Shared points may reuse a capture only when context, batch, request and runtime identities match. 128K/B8 must pass a feasibility gate; any preemption/OOM is recorded as a capacity limit, not mixed into a compute-scaling curve. Instrumentation that changes graph behavior is a separate diagnostic arm with an overhead control. Full traces stay outside Git.

The 15-hour session keeps all of these conditions. Profiles and state snapshots are taken once, in each model's diagnostics launch. The unprofiled values have three blocks: B=1 at 1K/16K/64K is the TTFT and TPOT of the one-client serving runs, and B=8 at all four contexts and B=1 at 128K are the decode intervals of the timing launches.

**The attention path, defined once.** To say whose attention is fastest, every model reports the same four parts: the *core* (the attention or state-lookup kernel), the *search* (indexer and top-k selection), the *recurrent update*, and *state preparation* (writing, compressing or converting stored state). Their sum is the attention path. Attention projections are reported beside it only where dense GEMMs can be attributed to attention modules; otherwise dense GEMM stays one mixed column, as in October. Give each value per 8,192-token prefill chunk and per decode step, split by layer type (window, full, sparse, recurrent), next to the unprofiled whole-step time. State the order separately for prefill and for decode, and at each context: the [October traces](reports/five-model/results.md#trace-components) already show a different leader in the two phases.

### Claims the rerun must test

A strength or a weakness is a claim, and a claim needs a measurement that could contradict it. Each row below starts from a pattern in the [October results](reports/five-model/results.md), names the measurement in this plan that decides it, and states the outcome that would make the blog drop it. A difference counts only when it exceeds both the practical threshold in [Serving matrix and repetition](#serving-matrix-and-repetition) and the spread between blocks.

The first table splits the [one idea](#the-brief) of the article into six claims that hold across models.

| # | Claim | Deciding measurement | The claim fails if |
| --- | --- | --- | --- |
| 1 | Sparse attention moves cost into the search: as context grows, the time to *find* positions grows faster than the time to attend to them | Search and core time per prefill chunk and per decode step at 1K to 128K, for the four models that run an indexer | The search share of the attention path stays flat from 16K to 128K |
| 2 | Recurrent layers keep their own decode time flat as context grows, but do not by themselves make stored state small | Recurrent-update time against context; stored bytes per token by cache group | Recurrent time in Qwen or GLM grows with context, or their bytes per token turn out the smallest of the five |
| 3 | Expert time follows the kernel, precision and expert width more than the number of experts selected | Expert time per routed token beside k, width, precision and kernel; the expert-kernel control in [Experiments for the blog](#experiments-for-the-blog) where it is available | Expert time across the five is proportional to k × width whatever the kernel |
| 4 | MoE sparsity is a per-token property: one decode token touches k of N experts, but a prefill chunk or a large batch touches nearly all of them | Distinct experts used per step against the tokens in that step (1, 8, 64 and 8,192) | The share of experts used stays near k of N at 8,192 tokens |
| 5 | At batch 1 a decode step is mostly fixed per-layer work (projections, routing, communication), not attention or experts | Whole-step elapsed time and classified kernel time at actual B=1, by component and per layer | The attention path plus experts exceed half of the classified kernel time |
| 6 | No deployment wins everywhere: the leader changes with input length and load | Throughput, TTFT and TPOT curves from the serving matrix, with the crossover points | One deployment leads at every measured point on the single node |

The second table gives one strength and one weakness per model.

| Model | Kind | Claim, from the October pattern | Deciding measurement | The claim fails if |
| --- | --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 | Strength | A small footprint: the smallest weights per GPU and compact compressed state leave room for many requests, and native DSpark may speed up low-load generation | Stored bytes by cache group; tested capacity at 16K/64K; paired DSpark gain at c1 | Its tested capacity at 64K is not above Qwen's and GLM's, and DSpark shows no paired gain at c1 |
| | Weakness | Slow prompt processing at long inputs: the search grows with context and the 4-bit expert kernel is the largest part of every prefill chunk | Prefill scaling at 1K to 128K: TTFT, first and late chunk components | The search share does not grow, or its TTFT at 64K and 128K is no longer above V4.1's and GLM's |
| DeepSeek V4.1 Flash | Strength | The smallest stored state per token, and less prefill work than 0731 because later layers process only a window | Stored bytes by cache group, with host-side tables listed beside them; prefill scaling; the replay control if a supported one exists | Its bytes per token are not the smallest once every cache group and replicated copy is counted, or its TTFT from 16K upward is not below 0731's |
| | Weakness | The highest fixed cost per decode step: the highest single-request TPOT in October at every input length, so the prefill saving pays off only at long inputs or higher load. Also the largest checkpoint and about 190 GB of pinned host memory | Whole-step decode time at B=1 by component; serving curves at 1K with 2,048 outputs; the input length and load at which V4.1 overtakes 0731 | Its B=1 decode step is not slower than 0731's, or no crossover with 0731 appears |
| MiMo-V2.6-Flash | Strength | Fast generation at short inputs and light load without a learned search: 39 of 48 layers read a 128-token window | Decode step at B=1 from a 1K context; serving at 1K, c1 and c4; attention path in a 1K and a 16K prefill chunk | Another deployment on the same node has a faster B=1 step at 1K and higher 1K/c1 throughput |
| | Weakness | Attention grows with context through nine full-attention layers, and the expert kernel is the slowest of the five in prefill, so the short-input lead does not carry to long inputs or more clients | Attention core by layer type at 1K to 128K; expert time per routed token; the expert-kernel control; serving curves at 64K | The full-attention layers' time does not grow with context, or its expert time per routed token is not the highest |
| Qwen3.8-Flash-Next | Strength | The highest throughput at most points: recurrent layers make decode attention cheap, and 640-wide experts keep expert time low although ten are selected | The serving matrix on the same node as the others (October's Qwen ran on a second machine); decode attention path; expert time per routed token | On the shared node its lead falls inside the threshold at most points |
| | Weakness | The largest stored state per token, and the most routing and communication time in a prefill chunk | Stored bytes and their slope by cache group; tested capacity at 16K/64K; routing time and the unprofiled communication share (October's all-reduce value includes profiler waiting) | Its bytes per token are not among the two largest, or its tested capacity at 64K matches the compact-state models |
| GLM-5.3-Flash | Strength | A short time to first token and the lowest eight-client TPOT on its node at 16K and 64K: FP8 experts on a CUTLASS kernel took about a tenth of MiMo's expert time in a prefill chunk, at the same expert dimensions | Prefill scaling; TPOT under load in the serving matrix; expert time per routed token; the expert-kernel control | Its TTFT from 16K upward is not among the two lowest on the single node, or its expert time per routed token is not below MiMo's |
| | Weakness | The second-largest state per token with the smallest reserved pool, padding in every 640-token page, and no thinking-off switch, so some output tokens are low-effort reasoning | Stored bytes including padding; tested capacity; natural-EOS task checks with reasoning tokens counted | Its tested capacity at 64K is not below the compact-state models, and reasoning tokens are a negligible share of its natural answers |

In the 15-hour session the capacity test, the expert-kernel control, the V4.1 replay control, speculation and client loads other than 1 and 16 do not run. Any part of a claim that needs one of them is reported as untested, not as held. That covers the capacity and DSpark parts of 0731's strength (its stored bytes and weights are still measured), the c4 point of MiMo's strength, the capacity parts of the Qwen and GLM weaknesses, the kernel control in claim 3, and the loads beyond two in claim 6. Where a capacity figure helps the reader, the blog may give reserved pool bytes divided by stored bytes per token, labelled Estimate.

The blog reports every row as held, failed or untested. A failed claim is a result and stays in the article. On the frozen build, reverify V4.1's layer split and short-step fallback (historically 21 full and 19 window-only layers) before reading its rows.

These are hypotheses, not predetermined winners. In particular, **do not start with the claim that V4.1 has constant total decode FLOPs**. The [pinned model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/README.md) bounds deeper decoder indexing through a candidate pool; that does not bound every operation or memory access in the whole model. The [existing implementation check](reports/blog-architecture-h100-v1/v41_prefill_check.md) verifies reduced prefill work, not an optimization-on/off causal speedup. Any such ablation requires a supported single-change configuration and output-correctness checks; it is a separately declared extension, not new implementation work on rented time.

### Speculative comparison

**Deferred on H100: not part of the 15-hour session.** Nothing in this section runs on H100 until a later session is authorized. Its first planned use is on the H200 and B200 servers, at two client loads: see [docs/H200_B200_plan.md](docs/H200_B200_plan.md).

Retain the archive's principle: compare each method against its **own fresh AR baseline** first, then show absolute performance across deployments. Output length, prompts, runtime, precision, target sampling, cache policy and hardware must match within a pair. Draft width is independent of the response budget: a 2K response does not mean a 2K draft.

| Model | Initial candidate from the archived support evidence | Gate on the rerun runtime |
| --- | --- | --- |
| 0731 | Fixed DSpark, requested width 5 | Pin target/draft to the recorded snapshot, verify loaded method and width; no assumed classic MTP |
| V4.1 | Fixed DSpark, requested width 5 | Verify CED path and draft/verification execution together; do not infer method from `mtp.*` weight names |
| MiMo | MTP width 3; DFlash width 7 from its pinned `dflash/` subfolder | Record actual loaded MTP depth/head reuse and DFlash feature taps; revalidate support and output behavior |
| Qwen | Support audit pending | Choose at most one native method for the first screen only after source, checkpoint, correctness and memory validation; otherwise report AR plus unsupported speculation |
| GLM | Support audit pending | Same gate; retain native FP8 and the reasoning setting |

For the initial fixed DSpark arm, carry over the archive's probabilistic draft sampling, standard rejection and adaptive-verification-off policy only after checking their exact meaning on the chosen runtime. Record every resolved setting. Adaptive DSpark, extra widths and runtime/kernel tuning require separately declared cells and fresh matched controls; there is no automatic Cartesian sweep. A documented feature on vLLM's mutable latest page does not prove availability on the pinned runtime.

The speculative screen extends the archive to the same five client loads as the new AR sweep: **1K/2048 and 16K/2048, each at c1,c4,c16,c32,c64**. These are 10 workload/load points per deployment. Let `D` be five AR deployments plus supported speculative arms. Three blocks require `30D` timing cells: **270** if the four archived candidate arms work and Qwen/GLM add none. This is the screen count including AR, not an additional 270 runs on top of reusable baselines. Reuse a new AR cell only when all identities, request lists and block definitions match; otherwise rerun it. If every overlapping AR cell can be reused, the AR matrix and this screen together are **345 timing cells**: the 225 above, which already contain every AR point of the screen, and 120 speculative cells. Record the exact union in the manifest. The 64K-input speculative 2K-output curve is a separately declared extension; AR still measures it at all five loads. c64 is a load probe, not a promised feasible or saturated configuration; a fallback load must be declared and run for every compared deployment while preserving the original point's status.

After screening, select at most one supported method per model with a fixed rule: best paired 2K decode-time gain at c1, subject to task checks and no unexplained regression; ties within uncertainty remain ties. This selects a low-load candidate, not a universal winner; retain every method's full concurrency curve and report where AR becomes preferable. Confirm the selected method on held-out prompts at the same 2,048-token output, at four points: **1K/16K × c1/c16**, against matching AR, in the five held-out blocks that [Serving matrix and repetition](#serving-matrix-and-repetition) requires for a final recommendation. This adds 20 timing cells per selected speculative arm, plus 20 matching AR cells for its model on the same held-out prompts. Do not extend a 2K speedup claim to longer outputs, which this plan does not measure, or choose a method using its held-out score.

Before timed comparisons, run duplicate same-seed AR and each method on the same natural-EOS task set at each tested load. Check output-token agreement where the implementation promises it, task results and truncation; investigate discrepancies against AR's own repeat variability. Forced-length speed and correctness/task behavior are separate results.

Collect proposed, verified, accepted and **actually committed** tokens, verification rounds, draft/target time, fallback frequency and extra memory, with explicit denominators. Report acceptance by generated-position window and domain when instrumentation supports it. Acceptance alone is not speedup, and percentages at different widths do not rank methods. Use actual token usage/engine counters rather than streamed text chunks, which may contain several tokens. Latest [vLLM speculative guidance](https://docs.vllm.ai/en/latest/features/speculative_decoding/) emphasizes workload dependence; [per-request acceptance telemetry](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/) is version-dependent and must be verified before relying on it. Missing counters remain null; do not manufacture a committed-progress count from a proxy.

### Experiments for the blog

**In the 15-hour session:** the HBM-counter attempt and the expert-routing statistics run, each inside its time limit, and the natural-ending check shrinks to 12 prompts per model in the diagnostics launch. The expert-kernel control, the four conditional rows, the 60-task set and the drift reference runs are deferred. The rotating model order and the three block values per point are the session's evidence about drift.

The 225-run AR matrix already covers substantial ground. Prioritize credible explanations and useful operating ranges over adding more contexts, draft widths or checkpoints. **Required** below means part of the proposed publication evidence; a **conditional extension** must be selected, costed and included in the frozen manifest before execution. None is a GPU launch authorization. Their counts are additional to the base matrix unless an identical control is explicitly reused.

| Priority / status | Experiment and bounded scope | Reader-facing result |
| --- | --- | --- |
| 1 / required correctness evidence | Strengthen the existing natural-EOS checks with frozen code unit tests, math answers and a fixed chat instruction-following rubric. Start with at least 20 tasks/domain, increasing the set to exercise the requested concurrency. Keep independent held-out tasks; log per-task scores, output/reasoning lengths, truncation and first-EOS position where observable for AR and each selected speculative method | Whether a speed gain delivers useful answers on the tested tasks; how much forced generation occurs beyond a natural endpoint. A small task set is a regression check, not a general model-quality ranking |
| 1 / required diagnostic attempt | Complete the already planned attention/expert HBM-counter collection: one 16K prefill chunk and 64K decode at actual B1/B8, all five models. Start with three capture conditions/model, a minimal counter set and representative kernels; expand only to establish stated attribution coverage | Measured bytes/step and bytes/committed token for the covered operations. Do not extrapolate a sampled attention kernel to whole-model traffic or call missing counters zero |
| 1 / required diagnostic attempt | Expert-routing statistics for Kan's FFN question, all five models. In a separate instrumented launch, read the router's selected expert IDs on the same frozen prompts: one 16K prefill chunk, and decode from 1K prompts at actual B=1, 8 and 64 over one 2,048-token generation each, balanced across code, math and chat. Routing follows from the weights and the input, not from timing, so this arm makes no speed claim; check that its output tokens agree with the plain launch | Per model and per layer: distinct experts used per step, tokens per expert, load imbalance across experts and across the eight GPUs, padding in the grouped expert kernel, and the share of expert weights touched per step, beside the nominal k of N. Dense FFN layers and shared experts are listed separately. If the runtime has no supported hook, the statistics stay unavailable and the blog keeps nominal sparsity with expert kernel time |
| 1 / conditional mechanism test | Expert-kernel control on one checkpoint, only if the frozen runtime can run the same expert weights at the same precision on two supported kernels. MiMo is the candidate: its experts ran on Marlin in the September build and on `HUMMING` in October. 16K/64K inputs, c1, 2,048 outputs, three paired blocks, plus one prefill and one B=8 decode trace per arm. At most 12 timing cells including both controls; the default-kernel arm may reuse main-matrix cells only with matching identities and block definitions | How much of a model's expert time belongs to the kernel rather than to its expert structure. Check output agreement; keep weights, precision, routing and every non-expert backend fixed. Comparing MiMo with GLM is not this control, because they differ in weights, precision and kernel at once. If no supported switch exists, the cross-model difference stays an observation |
| 1 / conditional mechanism test | V4.1 bounded replay enabled versus disabled **on the same frozen runtime**, only if a supported isolated control exists: 16K/64K inputs, c1, 2,048 outputs, three paired blocks. At most 12 timing cells including both controls; compare TTFT, engine prefill elapsed time and executed token rows/layer | How much prefill time the implementation actually saves. Check output agreement/task behavior, keep KV format and scheduler fixed. Switching runtime versions or enabling prompt logprobs is not an isolated replay ablation. If no suitable control exists, retain trace evidence and leave the causal gain unmeasured |
| 1 / conditional capacity test | For each model, initial 16K/64K prompts and synchronized actual B=1,4,16,32,64, no new admissions during a fixed decode interval of 512 committed tokens per request. Stop ascending a context's batches at the first capacity failure; repeat passing conditions in three blocks. At most 50 candidate conditions / 150 short intervals, with safe memory preflight | Largest **tested** batch that stays live without preemption/recomputation, plus occupied state, reserved pools and latency. This is a capacity bound for that context and interval, not maximum production concurrency. The existing c64 serving run may queue requests and cannot answer this by itself. By October's live-state approximation, only Qwen and GLM are expected to hit a limit below B=64 at 64K; that is an estimate to test. Because the server admits at most 64 sequences, this test cannot rank the three compact-state models against each other; their order comes from the measured state bytes |
| 2 / conditional domain comparison | Separate code/math/chat timing at 1K/2048, c1/c16, all five AR deployments, three blocks: 90 runs. Add 18 runs per selected speculative arm at the same points, with matching AR controls | Whether the aggregate winner is consistent across domains. Mixed-traffic aggregate acceptance does not establish per-domain speculative gains |
| 2 / conditional mixed-traffic check | At c16, mix equal numbers of 1K-input and 64K-input requests, all with 2,048 outputs, using a fixed shuffled schedule. All five AR models, three blocks: 15 mixed runs. The homogeneous controls are the main matrix's 1K/2048/c16 and 64K/2048/c16 cells, reused only with matching request sets/block definitions; otherwise rerun them | How long prefills affect generation already in flight. Report latency and throughput by request class, queueing and preemptions; do not hide a harmed class behind a better aggregate. This remains closed-loop traffic, not an open-loop SLO claim |

Run generated-code task checks in a bounded sandbox with no network access and record the evaluator version. Freeze reference answers, scoring rules and the criteria for investigating a regression before timing; avoid an uncalibrated model judge as the only quality evidence. Use the same task IDs within each AR/spec comparison. Existing repeated natural-EOS checks can supply these results when their identities match; do not add redundant executions simply to fill another table.

Before the session, compute the shape-based estimate of state bytes read and written per step for all five models. It gives Kan's HBM question a labelled answer if no counter can be collected, and it gives each counter an expected size to be checked against. For counters, document metric definitions, rank coverage, kernel coverage, replay mode and clock/cache settings. Nsight Compute can add overhead and serialize work; its profiled elapsed time must not replace authoritative unprofiled serving timing. If replay changes cache behavior, qualify the byte counts accordingly. See the [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/). An unavailable permission, unsupported kernel or missing tool limits the HBM claim and belongs in the handoff; it is not a reason to fabricate a bandwidth ranking.

Add two low-cost checks to the existing runs. First, report **full-run** throughput and a separately defined steady-state interval after ramp-up and before drain; predeclare the occupancy/duration rule and retain both denominators. A finite request list can spend substantial time below its requested concurrency. Second, repeat one fixed 16K/2048/c1 reference-model workload near the start and end of each launch cycle to detect node drift (up to six reference runs over three cycles). Record clocks, temperature, power, CPU load and throttling. Freeze a drift threshold from the pilot, investigate drift and repeat the affected comparison block if necessary; do not normalize model results by an arbitrary correction factor.

The domain, mixed-traffic, capacity, replay and expert-kernel extensions are separately budgeted. If all five are selected, they require **up to 129 extra AR timing runs** (90 domain + 15 mixed + 12 replay + 12 expert-kernel), plus up to six drift checks, diagnostic intervals, speculative pairs and confirmation blocks. The routing-statistics arm adds up to five short launches and no timing runs. Identical controls may reduce this count; report the actual union. Do not silently describe 225 as the cost of the expanded package. Choose the extensions before GPU collection according to the intended claims and available allocation.

The article should then have six main figures: prefill versus context; decode time versus actual context, with the attention path split out; live-state growth and tested capacity; expert sparsity, nominal against realized, beside expert time per routed token; throughput/latency versus client load; and speculative gain versus load/output position. A concise strengths/weaknesses table should point to those figures and state, for each [claim](#claims-the-rerun-must-test), whether it held, failed or stayed untested, and whether its explanation is measured, source-supported or still a hypothesis. After the 15-hour session the first four figures are available, the load figure has two loads, and the speculative figure waits for a later session. Costs can be reported as allocated GPU-seconds per output token (`8 / output_tok_s`); dollars require the actual node price. Power/energy, open-loop tail latency, broad quality leaderboards, prefix reuse and hardware/topology sweeps are not required to make this architecture blog useful.

### Preparation, cost and completion

1. **Prepare locally.** Extend the harness with manifest-driven output lengths, context points and methods. The current `bench/blog_study.py` still hardcodes 256-token point IDs and validation text; changing one generation flag is insufficient. Update request seeds/identities, per-request validity, timeouts, analysis and dry-run counts together. Preserve historical defaults and add no new run to old study directories. Validate short completion, counter reset, unsupported arm and failed-shutdown cases before renting. Because the node time is fixed, finish everything that needs no GPU before the rental starts: the harness changes, the dry-run counts, the shape-based HBM estimate, the routing hook and the counter commands.
2. **Freeze the study.** Record new study ID, checkpoint/draft/runtime pins, dataset hashes, exact matrix union, request counts, warmups, block order, output/EOS policies, capability checks, retry policy and session budget. The tables above are a proposal until that manifest is frozen. The existing article remains completed 256-token evidence.
3. **Budget.** The rental is fixed at 15 hours, and [The 15-hour session](#the-15-hour-session) allocates it from an estimate. The pilot inside each diagnostics launch checks that estimate: every session point must be valid and last at least 60 seconds. If the session falls behind, apply its checkpoint and drop order; do not extend the rental. For the deferred parts, estimate each run as `requests × output_tokens / measured_output_tok_s` using throughput measured at that output length and load; add downloads, startups/JIT, correctness, diagnostics, independent confirmations and retries. Do not extrapolate 2,048-token performance from the old 256-token aggregate rate. Report node-hours and GPU-hours (`8 × node-hours`); monetary cost needs the rental price. No measured total exists before that pilot, and the older 8–10-hour reproduction allowance does not cover the full plan.
4. **Measure in bounded stages.** In the session, follow the order in [The 15-hour session](#the-15-hour-session). In a later session: support/correctness first, then matched AR blocks, speculative screen and declared confirmations. Own and stop every server on success, error, timeout or interruption and verify GPU release. Stop at the budget boundary and leave unrun points pending; do not redefine completion to match what finished.
5. **Publish the answer.** For every model give the workload, throughput and TTFT, decode-only time, attention-path time, HBM bytes (measured or estimated), state bytes and growth, nominal and realized expert sparsity, paired speculative gain/regression, task outcome, uncertainty and explanation confidence. Close each row of the [answer map](#answer-map-for-the-brief) as answered, partly answered or unanswered, and each [claim](#claims-the-rerun-must-test) as held, failed or untested. Distinguish measured, estimated, source-supported, unsupported and unanswered. Include per-request evidence and block summaries; retain raw measurement bytes and hash ledgers. Run the repository audit on a clean clone before publication. Commit/push only when requested.

Completion requires accounting for every frozen cell as valid, failed, unsupported or pending, giving every answer-map row and every claim a stated outcome, and resolving the questions only as far as the evidence permits. A longer output budget alone does not complete an architectural claim.
