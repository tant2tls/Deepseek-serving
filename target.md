# Target: answer Kan's architecture questions on 8×H100, and publish a blog everyone can use

Updated 2026-10-09. **This is the repository's single target file.** It merges the former `blog_target.md` (the architecture-blog plan) with the former three-model `target.md` (the later speculative study, now [section 12](#12-later-study-spec-realtext-h100-v1)). The merge changed no measurement, plan count or model pin, and it authorizes no new GPU work. This full version lives on the `all-data` branch; the `main` branch carries a shorter publication view of it for the five-model blog.

**Two purposes, in this order:**

1. **Measure, to answer Kan's question.** Each part of [the brief](#the-brief) gets an answer backed by evidence collected on 8×H100, or is stated plainly as unanswered.
2. **Publish a blog that is useful to everyone who reads it.** A reader new to inference systems can follow every figure, and a practitioner can trace every number to its run, build and node.

Work that serves neither purpose is later work (section 12) or out of scope.

## Where things stand

| Part | State on 2026-10-09 | Start here |
| --- | --- | --- |
| Measurements for the blog | **Done** for the five main deployments: 0731, V4.1, MiMo and GLM on the first node (`blog-architecture-h100-v1`), and Qwen3.8-Flash-Next in original BF16 on a second node with a MiMo control (`qwen-bf16-h100-v1`). All on vLLM `554340f3…` | [first-node findings](reports/blog-architecture-h100-v1/findings.md) · [Qwen BF16 findings](reports/qwen-bf16-h100-v1/findings.md) |
| The article | [index.html](index.html), refined from the sources selected in [section 11](#11-current-blog-evidence-selection-2026-10-09) | [audience, figures and handoff](#9-blog-figures-and-agent-handoff) |
| Still unanswered | Measured HBM traffic, expert-routing statistics and reusable-prefix capacity; see [what is answered so far](#what-is-answered-so-far) | [first-node handoff](reports/blog-architecture-h100-v1/handoff.md) · [Qwen BF16 handoff](reports/qwen-bf16-h100-v1/handoff.md) |
| Earlier studies (September, vLLM `44af287e…`) | Done; used here only to choose hypotheses | [index of all studies](reports/README.md) |
| Later speculative study `spec-realtext-h100-v1` | **Planned, not measured** | [section 12](#12-later-study-spec-realtext-h100-v1) |

## Standing decisions

**Editorial decision, 2026-10-09:** use the latest completed results in the blog: DeepSeek V4 Flash 0731, V4.1 Flash, MiMo-V2.6-Flash-MOPD and GLM-5.3-Flash from the first node, plus **Qwen3.8-Flash-Next in original BF16** from the second node. Qwen FP8 is **reference only**, excluded from the main figures, rankings and takeaways. [Section 11](#11-current-blog-evidence-selection-2026-10-09) defines the sources, counts and comparison limits. This is an editorial update using existing evidence; no new GPU work is authorized.

**Checkpoint policy update, 2026-10-08:** future Qwen3.8-Flash-Next work uses only the original **BF16** checkpoint `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540`. The completed five-model study and section 10 used Qwen **FP8** and remain unchanged historical evidence. BF16 has its own key (`qwen-38-bf16`) and study: it was measured on 2026-10-08 with this plan's protocol in `qwen-bf16-h100-v1` on a second rented node, with a MiMo node control ([findings](reports/qwen-bf16-h100-v1/findings.md)). No new FP8 runs. Follow [the checkpoint policy](docs/qwen-checkpoint-policy.md).

**Plan status:** sections 1–9 are the plan of study `blog-architecture-h100-v1`, executed on 2026-10-07; section 10 is its addendum for two more models, and section 11 selects the evidence the article uses. The measurements, their status and what is still open are in [reports/blog-architecture-h100-v1/](reports/blog-architecture-h100-v1/): start with [findings.md](reports/blog-architecture-h100-v1/findings.md) and [handoff.md](reports/blog-architecture-h100-v1/handoff.md). Numbers quoted in *this* file from earlier studies are historical, were taken on an older vLLM build, and are used only to choose hypotheses.

## The brief

Kan (SyFI lab) asked:

> We want to teach people architecture of new generation of models. Specifically we want to compare the attention and FFN side. Which model's attention take minimal runtime loading/HBM or runs fastest? For prefix caching, which model is less space consuming? FFN side what are there sparsity differences? Feel free to add more!

**Our task now.** Explain the completed measurements using each model's architecture report and its actual inference path, so that the measurements and the explanation agree and a reader new to these models can follow them. The original protocol below called for one 8×H100 node; the current article combines four deployments from that node with Qwen BF16 from a second node, using the limits in section 11.

- **Main article models:** DeepSeek V4 Flash 0731, DeepSeek V4.1 Flash, MiMo-V2.6-Flash-MOPD, Qwen3.8-Flash-Next, GLM-5.3-Flash. The unqualified Qwen name denotes the original BF16 checkpoint. Sections 1–9 preserve the original three-model measurement plan; section 10 preserves the first-node addendum.
- **Presentation reference:** the [TraceLab post](https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/). It is an example of style, not evidence about these checkpoints.

## The plan on one page

| Kan's question | What we will measure | Experiment | What the reader gets |
| --- | --- | --- | --- |
| What is the architecture of each model? | Layer types, shapes, precision and the kernels the runtime really loads | B0 (section 3) | One side-by-side table and a "one token through the model" diagram |
| Whose attention runs fastest? | Prefill and decode time of the attention path, at several context lengths and batch sizes | B1 + B2 (sections 4–5) | Context curves and a component breakdown |
| Whose attention moves the least data through HBM? | Hardware read/write byte counters on selected kernels | B2 (section 5) | Bytes per token, measured where counters work and labelled estimates elsewhere |
| Which model's cached state takes less space? | Bytes held by one live request's KV/state | B3 (section 6) | "What does one live context cost?" **Prefix reuse itself is not measured; see the boundary below** |
| What are the FFN sparsity differences? | Which experts run, how evenly, and what routing and communication cost | B4 (section 7) | Nominal sparsity next to realized cost |
| "Feel free to add more" | Whether component differences explain user-visible speed | B1 (section 4) | TTFT, TPOT and throughput anchors on public text |

### What is answered so far

| Kan's question | State on 2026-10-09 | Evidence |
| --- | --- | --- |
| What is the architecture of each model? | **Answered** from config, server log and trace on the blog build; the items that could not be verified are listed | [architecture.md](reports/blog-architecture-h100-v1/architecture.md); Qwen BF16: [findings](reports/qwen-bf16-h100-v1/findings.md) section 2 |
| Whose attention runs fastest? | **Answered** for prefill and decode at the measured contexts and batch sizes: diagnostic kernel sums, read beside unprofiled timing | [First-node findings](reports/blog-architecture-h100-v1/findings.md) sections 1–3 and 8; [Qwen BF16 findings](reports/qwen-bf16-h100-v1/findings.md) section 6 |
| Whose attention moves the least data through HBM? | **Unanswered.** No hardware counters were collected (no Nsight Compute on the image); only labelled estimates exist | [First-node findings](reports/blog-architecture-h100-v1/findings.md) section 4; [handoff](reports/blog-architecture-h100-v1/handoff.md) |
| Which model's cached state takes less space? | **Partly answered.** Live KV/state bytes of ordinary requests are measured as an approximation (usage gauge × per-rank pool). Reusable-prefix capacity is **unanswered**: prefix experiments stay deferred | [First-node findings](reports/blog-architecture-h100-v1/findings.md) section 5; [Qwen BF16 findings](reports/qwen-bf16-h100-v1/findings.md) section 7 |
| What are the FFN sparsity differences? | **Partly answered.** Nominal structure and trace-level expert and routing time are recorded. Tokens per expert, coverage and padding are **unsupported** without runtime instrumentation | [First-node findings](reports/blog-architecture-h100-v1/findings.md) section 6 |
| "Feel free to add more" | **Answered:** TTFT, TPOT and throughput on public text, from unprofiled runs | [First-node findings](reports/blog-architecture-h100-v1/findings.md) section 2; [Qwen BF16 findings](reports/qwen-bf16-h100-v1/findings.md) section 3 |

The answers themselves, with their numbers, confidence and limits, live in the findings. This file states the target and its state; it does not restate results.

**The one idea the reader should leave with:** fewer attended positions or fewer active parameters do not automatically make a model faster. A deployment can win one component and still lose overall, because the cost also depends on the work needed to *find* the sparse subset, on the kernel that executes it, on precision, and on communication between GPUs.

**Four evidence labels, never mixed.** Every statement in the blog carries exactly one:

| Label | Meaning | Example |
| --- | --- | --- |
| Architecture fact | What the paper, model card or checkpoint config says | "Top-8 of 256 experts" |
| Executed implementation | What the pinned runtime verifiably loads and runs | "Dense FP8 layers resolve to the Marlin kernel" |
| Measurement | Time or bytes recorded on the node, with its scope | "ms per 8,192-token chunk, rank 0, kernel sum" |
| Estimate | Arithmetic from shapes; never presented as measured | "≈ 14.2B linear weights touched per token" |

Architecture descriptions and config field names do not prove execution. For example, V4.1's paper describes a prefill skip (CED) that the pinned build did not run in the earlier traces.

**Boundaries that hold for the whole blog study:**

- Autoregressive (AR) decoding only, speculation off, **prefix caching off in every launch**.
- Initial ledger: **54 unprofiled timing runs + 18 trace captures + 6 hardware-counter point sets**. Live-KV and routing diagnostics attach to those captures where feasible.
- Outcome of the first session: timing runs, trace captures and live-KV snapshots were collected; the hardware-counter point sets and per-expert routing histograms were **not** (no Nsight Compute on the image, no routing instrumentation in the runtime). Both stay open in the handoff.
- The broader speculative-serving study `spec-realtext-h100-v1` in [section 12](#12-later-study-spec-realtext-h100-v1) stays pending for later. Blog results do not complete or replace it and must not be appended to historical curves.

## Models and immutable identities

One table pins every checkpoint this repository measures or plans to measure. Pin the tokenizer and remote code to the same revision as the weights.

| Key | Checkpoint | Revision | Role in the blog | Arms in the later study (section 12) |
| --- | --- | --- | --- | --- |
| `v4-0731` | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | Main article, first node | AR off; fixed DSpark k5 |
| `v41` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | Main article, first node | AR off; fixed DSpark k5 |
| `mimo-v26` | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | `2479e2d0029eca9a34cc7e7f55a121925f81908e` | Main article, first node; node control on the second | AR off; MTP k3 (layer 0 reused); DFlash k7 |
| `glm-53` | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | Main article, first node; no thinking-off switch ([section 10](#10-addendum-2026-10-07-qwen38-flash-next-and-glm-53-flash)) | Not in the later study |
| `qwen-38-bf16` | `Qwen/Qwen3.8-Flash-Next` (original BF16) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | The main article's Qwen, second node | Not in the later study |
| `qwen-38` | `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` | **Historical reference only**; `serve.sh` and the chain refuse it | Not in the later study |

**Qwen and GLM stay out of section 12.** The 2026-10-08 [checkpoint policy](docs/qwen-checkpoint-policy.md) does not add Qwen to the three-model speculative matrix, and `qwen-bf16-h100-v1` is AR only. The `qwen-38` key and revision remain historical FP8, never a BF16 baseline.

**Two runtime builds, never mixed:**

| vLLM build | Studies | State |
| --- | --- | --- |
| `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`) | `blog-architecture-h100-v1`, `qwen-bf16-h100-v1` | Measured, October 2026 |
| `44af287ebe38d6dc4e102948025f5e3e175aefd6` (`0.30.1rc1.dev223+g44af287eb`) | `v41-vs-0731`, `mimo-v26`; default for `spec-realtext-h100-v1` unless Tan chooses a newer build | Measured, September 2026; the later study is planned |

A number from one build is never a control for the other: kernel backends and V4.1's prefill path differ. State the build beside every number, and the node as well once a result leaves the session it was measured in.

The preview DeepSeek checkpoint and other models are out of scope: the preview's unverified revision is never a 0731 baseline. H200 is an optional, separately requested future study.

## Terms used in this plan

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

## What we already know, and what it does not prove

**Read this section as history.** On the newer build used for the blog study several of these facts changed, most visibly V4.1's prefill path and the expert GEMM backend; the current facts are in [architecture.md](reports/blog-architecture-h100-v1/architecture.md).

These values come from the completed studies ([DeepSeek](reports/v41-vs-0731/report.md), [MiMo](reports/mimo-v26/report.md), [MiMo walkthrough](docs/mimo-v2.6-inference.md)), with the corrections in [update.md](update.md). They used **random-token prompts**, and the traces are **rank-0 kernel sums** of selected chunks. They tell us where to look; they are not blog results. B0 re-verifies every structural entry on the node.

**Structure as recorded earlier (source and trace level, to be re-verified):**

| | V4 Flash 0731 | V4.1 Flash | MiMo-V2.6-Flash-MOPD |
| --- | --- | --- | --- |
| Decoder layers | 43 | 40 | 48 |
| Attention design | Sliding window of 128 plus compressed long-range KV, selected by a learned indexer (top-512), sparse attention kernel | Same family; cross-layer KV reuse and bounded window replay are implemented; the CED prompt skip was **not** executed | 39 sliding-window layers (128 keys plus a sink) and 9 full causal attention layers, FlashAttention-3 |
| KV cache format | `fp8_ds_mla` | `fp8_ds_mla` | `auto`, which resolves to BF16 |
| FFN | MoE, 6 routed + 1 shared expert active | MoE, 6 routed + 1 shared expert active, larger experts | MoE, top-8 of 256, no shared expert; layer 0 is a dense FFN |
| Extra per-layer work | 4-stream mHC residual mixing, indexer, KV compressor | Same, plus Engram on some layers | None of these |
| Dense FP8 kernel observed | DeepGEMM/FlashInfer block-scale | Marlin FP8 | DeepGEMM/FlashInfer block-scale |
| Estimated linear weights touched per token | ≈ 12.2B | ≈ 15.0B | ≈ 14.2B |

**Historical timing anchors (random tokens, prefix caching off, c1):**

| | 0731 | V4.1 | MiMo |
| --- | ---: | ---: | ---: |
| Prefill, µs per prompt token at 16K | 41.6 | 54.5 | 33.2 |
| Decode TPOT p50 from 1K to 260K, ms | 7.4–7.5 | 7.56–7.85 | 5.5–6.3 |
| Prefill kernel sum per 8,192-token chunk at 16K, ms | 293.1 | 399.2 | 216.3 |
| Attention + indexer in that chunk at 16K, ms | 41.9 + 8.9 | 49.5 + 6.1 | 9.1 + none |
| Attention + indexer at about 57K prior context, ms | 53.2 + 40.5 | 50.1 + 28.4 | 38.8 + none |
| Reported AR KV pool, tokens | 1,711,998 | 9,179,728 | 6,906,032 |

**Why these already make a good story, and what the blog must still establish:**

1. **Work done is not time taken.** MiMo is estimated to touch about 16% more linear weights per token than 0731, yet ran faster. The earlier explanation is that it skips the indexer, compressor and mHC work and uses lighter projections. The blog needs fresh, real-text, all-rank evidence for this.
2. **Sparse attention is not free.** DeepSeek's sparse kernel stayed near 42–53 ms per chunk, but its indexer grew with context. MiMo's nine full-attention layers also grew, yet stayed cheaper at the measured contexts. A FLOP-based prediction made before the run was wrong. The blog should show core attention and selection cost separately.
3. **Flat versus rising decode.** DeepSeek's decode time was flat with context; MiMo's rose mildly but started lower. Whether KV reads explain the rise is an estimate until HBM counters are collected.
4. **Communication is a first-class cost.** All-reduce was 44% of MiMo's sampled prefill kernel sum. That share selects a hypothesis; it does not promise a 44% gain.
5. **Token pools are not bytes or capacity.** The three pools use different KV formats and accounting, so they are neither equally expensive tokens nor measured request capacity. B3 replaces them with bytes.
6. **Missing attribution.** V4.1 decode ran inside full CUDA graphs, so its component breakdown is unavailable from the old traces. Unavailable is not zero.
7. **Accounting is not causation.** At fixed lengths, `output rate = input rate × output length / input length` is an identity. It does not prove a prefill bottleneck; phase timing must.

**Strengths and limits recorded in the September studies (vLLM `44af287e…`).** The last column holds the questions the later study (section 12) is meant to answer.

| Model | Current strength | Current limitation and new question |
| --- | --- | --- |
| MiMo | Highest reported uncached throughput; lowest c1 decode latency; DFlash c1 219.2 tok/s on 16K/256 | All-reduce dominates the sampled prefill kernel sum; MTP reuses one head. Does the advantage hold for real text and long output? |
| 0731 | Faster uncached than V4.1; fixed DSpark c1 gives 1.87× over its own AR baseline | Below MiMo's absolute throughput; smaller reported KV pool and slow first repeats. Does its draft agreement pay off at high load? |
| V4.1 | Lower ≤2K TTFT than 0731; largest reported KV pool; advantage in the historical one-touch prefix procedure | MiMo is faster on the tested short uncached prompts too. Dense GEMM/collective cost and inactive CED skipping limit this build. Which supported intervention changes measured cost? |

These observations apply to the saved deployment settings and workloads. Native weight/KV formats and kernel paths differ; equal utilization flags do not mean equal usable KV capacity. The reported pools are not measured maximum concurrent requests. [update.md](update.md) links numerical anchors, raw runs, uncertainty and interpretive corrections, including the distinction between estimated and measured committed tokens per round.

---

## 1. Priorities and scope

| Priority | Question the measurement agent answers | Evidence to collect |
| --- | --- | --- |
| P0 | What does each model actually execute? | Architecture and tensor-shape table, pinned source paths, loaded decoder, layer types, native precision and kernel mapping |
| P1 | Whose attention is faster during prefill and decode, and how does context change that? | Phase-specific latency, complete attention-path breakdown, matched batch/context diagnostics |
| P1 | Which attention path moves fewer bytes through HBM? | Targeted hardware read/write counters; separate weight, KV and temporary-state traffic where attribution permits |
| P1 | How much memory does a live request's attention state occupy? | Logical state calculation and measured physical live-block bytes, including replication and allocator overhead |
| P1 | What do FFN sparsity differences save or cost in practice? | Routed/shared/dense expert structure, actual routing, expert GEMMs, dispatch/combine, padding and exposed communication |
| P2 | Do these component differences explain user-visible performance? | Fresh unprofiled AR controls, TTFT, TPOT, end-to-end latency and throughput on the same public texts |
| Conditional | Can one supported implementation change test a specific explanation? | A separately budgeted single-change ablation with a fresh matched baseline |

**In scope first:** AR decoding with speculation off for all three models.

**Not needed to finish this blog:** DSpark, MTP/DFlash widths, adaptive verification, 2048-token generation, topology sweeps and H200. They belong to later work.

**Prefix-cache boundary.** Kan's brief asks about prefix-caching space. The project instruction still defers every new prefix-cache experiment until Tan explicitly requests one, and this document does not override that. Therefore:

| Allowed now | Not authorized by this document |
| --- | --- |
| Measuring the KV/state bytes of ordinary live requests | Cold/warm comparisons |
| Recording KV occupancy and traffic telemetry | Reuse checks and one/two-touch prewarming |
| Citing existing prefix results as historical context, with their warming caveats | Prefix working-set pressure, or speculation combined with prefix caching |

The blog answers "what does one live context cost in bytes?" and states plainly that retained-prefix capacity and reuse are **unanswered**. Neither live KV bytes nor an advertised token-pool size proves which model retains more reusable prefixes.

## 2. Common deployment and local preparation

**Fixed for every model:**

| Setting | Value |
| --- | --- |
| Checkpoints | The immutable revisions of `v4-0731`, `v41` and `mimo-v26` in [Models and immutable identities](#models-and-immutable-identities); section 10 adds two more models. The older DeepSeek preview is not a substitute for 0731 |
| Runtime | vLLM `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`), the latest nightly on 2026-10-07, chosen by Tan for this study and installed in a separate venv (`VLLM_VENV=/root/vllm-latest`). Tokenizer and remote code are pinned to the checkpoint revisions. The completed studies used `44af287ebe38d6dc4e102948025f5e3e175aefd6`; that venv is kept for reproduction |
| Topology | TP8 + EP on one exclusive 8×H100 80GB SXM node |
| Limits | Memory utilization 0.90, context limit 262144, max sequences 64, explicit 8192-token chunk budget |
| Precision | Native; no re-quantization |
| Requests | Text only, temperature 0, `thinking=false` for DeepSeek and `enable_thinking=false` for MiMo |
| Caching | Prefix caching off |

Record what the runtime *resolves*, not only what was requested: graph modes, quantization, KV format and attention/MoE backends. Record the node as well: GPU topology, clocks/power, CPU model/affinity, RAM/NUMA, driver/CUDA and dependency versions. Keep all of it fixed across models.

**What this comparison can and cannot say.** Native weight and KV formats differ between the models, so conclusions compare *these deployments*. They do not isolate architecture from quantization or kernel quality.

**Runtime change.** The blog study runs on a newer vLLM build than the completed studies. Every blog number is therefore a fresh measurement on the new build, and the historical tables above are context only: a difference between an old and a new number mixes runtime, prompt content (random tokens versus public text) and node, and must not be read as a runtime speedup or regression. Implementation facts recorded for the old build (kernel selection, the missing V4.1 CED skip, CUDA-graph attribution) must be re-verified on the new build before they are repeated.

**Prepare before any GPU is rented.** GPUs are billed by the hour, so everything that can be done locally is done first:

1. **Source worksheet.** Architecture-report and model-card URLs with revisions, checkpoint config and tensor shapes, pinned runtime symbols, and a proposed map from profiler operators to blog components. Paper features are verified against the loaded implementation on the node. In particular, resolve MiMo's text decoder behind the `Omni` wrapper, and check which DeepSeek compression, indexer and CED paths actually execute.
2. **Public inputs.** Balanced code/math/chat texts with licenses or authored provenance, immutable hashes, request IDs and held-out examples. Build meaningful inputs of approximately 1K, 16K and 64K tokens; no repeated filler. Apply exactly one chat template and save the actual token count per model. Two views stay separate: the *same-text* serving comparison (token counts differ by tokenizer) and *exact-token-shape* diagnostics.
3. **Collection.** A study-specific manifest, per-request 256-token validation, phase segmentation, live KV-block accounting, trace classification and a small counter-collection pilot. Do not claim that the existing single-request profiler already provides synchronized B=8 decode or HBM attribution.
4. **Execution.** A new no-prefix, manifest-driven chain that reuses the bounded-readiness and owned-process cleanup patterns from `bench/chain_lib.sh`. Do **not** invoke `chain_mimo.sh` or `chain_dspark.sh`: they hardcode historical study IDs, and the MiMo chain runs prefix stages. Dry-run unique paths, allowed models, pins, counts, no-prefix checks, failure handling and shutdown before renting.

## 3. B0 — Architecture and implementation facts

**Purpose.** Give the reader a correct picture of each model before any timing appears, and give later experiments a shared vocabulary.

Create one side-by-side table with every value linked to its source and tagged with its evidence label. Record:

- **Structure:** total layers and layer types; hidden and head dimensions.
- **Attention:** query/KV heads and TP replication; full, sliding, compressed or sparse attention; window and selection sizes; KV/cache dtype and auxiliary index state; attention projection shapes.
- **FFN:** total, routed and shared experts; top-k; expert width; dense FFN layers.
- **Execution:** weight precision and the runtime backend for each operator class.

The table in "What we already know" above is the starting draft, not the result.

**"Sparsity" means three different things. Keep them apart:**

| Kind | What is sparse | What still has to be paid |
| --- | --- | --- |
| Attention sparsity | Which past positions are read | The selection or compression work needed to find them |
| MoE activation sparsity | Which experts run for a token | Routing, dispatch/combine, shared experts and dense layers |
| Weight sparsity / quantization | Zeros or fewer bits in stored weights | Only counts when actually present and supported. MoE routing does not establish hardware structured sparsity |

**Estimates.** Compute nominal active parameters and linear FLOPs, and label them as estimates. Also report resident weight bytes, including quantization scales and metadata. All experts stay resident in memory even when a token routes to only a few. Never equate active-parameter bytes with measured HBM reads.

## 4. B1 — Fresh serving controls and phase timing

**Purpose.** Produce the trustworthy, unprofiled numbers that everything else is anchored to. Profiled runs explain; only these runs rank.

| Axis | Values |
| --- | --- |
| Models/arms | All three models, AR only, prefix caching off |
| Input buckets | Approximately 1K, 16K, 64K; publish actual rendered token counts |
| Output | Forced 256 tokens per request; natural-EOS checks are collected separately |
| Client concurrency | c1 and c8 |
| Repeats | Three declared repeat blocks |
| Total | **3 models × 3 contexts × 2 loads × 3 repeats = 54 timing runs** |

**Why these points.** c1 exposes per-request cost. c8 tests modest batching while keeping the long-context comparison bounded. This matrix does not establish high-load saturation or maximum capacity.

**Order of work:**

1. A complete 16K comparison at c1 and c8 across all three models: 18 timing runs. This is the first deliverable; it must be complete before anything else expands.
2. The 1K and 64K points: 36 more runs.
3. Rotate model order across the three blocks, and freeze the exact launch and workload order before collection. Nine AR launches cover the full matrix if all six points per model run within each block. Staging separate sessions adds launches and must be budgeted.

**Sizing request lists.** Use a disjoint pilot. Start at `max(16, 4c)` requests, round up for domain balance, and increase until the fastest model has a measurement window of at least 60 seconds. Freeze identical text lists, counts and order across models for each point. If a count proves insufficient, keep that attempt and collect a new matched set for *every* model.

**What to save and report:**

- Output tok/s, requests/s, TTFT, per-request TPOT and end-to-end latency, each with mean, sample SD, median and the individual repeat values.
- Engine prefill/decode intervals where observable, startup/drain behavior and the actual engine batch distribution. Client c8 is not proof of an eight-sequence decode step.
- Before timing, check rendered inputs, output accounting and natural-EOS responses for each model. These are functional checks, not evidence of equal model quality.

**Reading rules.** TTFT includes scheduling as well as first-token work. Throughput proportionality does not prove prefill dominance. Slow first repeats and failures are kept, not replaced.

## 5. B2 — Attention runtime and HBM traffic

**Purpose.** Answer "whose attention is fastest, and whose moves the least data?" without letting the definition of "attention" quietly differ between models.

### 5.1 Define attention the same way for every model

Publish two numbers, both clearly named:

- **Attention core:** the attention kernel itself.
- **Complete attention path:** Q/K/V and output projections, positional operations and norms, KV insertion, compression, indexer/top-k, and the core.

Rules for assigning time:

- Residual, mHC, Engram or other model-specific work goes in its own explicit category where applicable.
- A fused kernel is assigned once. If it spans categories, label its combined scope; do not invent a split.
- FFN and shared-expert projections stay out of attention totals, even though historical reports grouped both under "dense GEMM".

### 5.2 Diagnostic traces

| Diagnostic | Points per model | Total |
| --- | --- | ---: |
| Prefill trace | Single request at 16K and at 64K input | 6 captures |
| Decode trace | Initial KV 1K and 64K, each at actual active B=1 and B=8 | 12 captures |
| **Trace budget** | Six captures per model | **18 captures** |

- **Prefill.** Record the full request phase, then compare representative early and late chunks, each with its actual query-token count and prior KV length. An 8192 scheduler budget does not guarantee that every chunk holds 8192 query tokens.
- **Decode.** Finish prefill first and admit no new prompts during the measured window. Record the exact active B, the context distribution and the generated-token positions. If synchronized batches cannot be held, segment faithful decode-only steps by their actual B. Unmatched states stay pending; they are not relabelled.
- **Graphs.** Use graph-aware tracing where possible. If graph replay hides attribution, as it did for V4.1 decode, run an explicitly separate eager/instrumented diagnostic arm and measure its overhead against default execution.
- **Ranks and overlap.** Collect all-rank timing or disclose rank coverage. Report exposed collective waits and wall-clock spans separately from kernel sums. Overlapping GPU work is not additive latency.

### 5.3 Hardware counters for HBM

After the traces identify the relevant kernels, collect **six initial counter point sets**: each model at a late 64K prefill chunk and at 64K/B=1 decode.

- Cover representative attention-core, projection, indexer/compressor and FFN kernels, including distinct layer types.
- Freeze kernel and rank selections and the actual replay-pass counts before collection. Six point sets are not six cheap timing runs.
- Add B=8 counters only if needed to resolve a batching hypothesis.

Measure HBM read/write bytes, L2 behavior, kernel duration and compute activity where supported. Report:

- Bytes per processed prefill token and per committed decode token, with rank coverage and precision explicit.
- Weight, KV and intermediate bytes separately **only** where kernels or instrumentation support that attribution. Total DRAM counters alone do not identify the source.
- Attained bandwidth, computed with matching byte and time scopes.

Do not call a kernel bandwidth-bound solely because GPU utilization is high or a theoretical byte estimate is large.

**The measurement can disturb what it measures.** Profiler replay, serialization and hardware-cache handling can change execution. Record tool and version, replay mode, pass count, cache/clock policy and overhead, and validate multi-rank compatibility in a short pilot. Use [NVIDIA's profiling guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) to choose a supported collection mode. Hardware L1/L2 cache handling is unrelated to the prohibited application prefix-cache experiments.

**If counters are inaccessible or replay is unreliable:** keep the timings and the labelled byte estimates, and mark the measured HBM comparison **unavailable**. Do not substitute estimates for it.

**What "loading/HBM cost" means here.** Memory traffic during inference. Checkpoint download, disk loading and server startup are recorded for rental budgeting and are not mixed into attention runtime.

## 6. B3 — Live KV/state memory, with prefix caching off

**Purpose.** Answer "what does one live context cost?" in bytes. This is the part of Kan's prefix question that can be answered without a prefix experiment.

At the B1 context/load states, take diagnostic snapshots after prefill and at a fixed decode position, recording the actual live sequences and their lengths. Reuse the B1 workload definitions; no separate cache-reuse requests are needed. If the hooks change execution, pair them with matched unprofiled controls.

**Publish four quantities, and never merge them:**

| # | Quantity | What it is | Why it is not the others |
| --- | --- | --- | --- |
| 1 | Logical state | KV and auxiliary bytes derived from source, per layer type, as a function of context. Includes sliding-window retention, compression and indexer state | An estimate from shapes; ignores block rounding and replication |
| 2 | Physical live allocation | Occupied blocks × actual bytes per block, per cache group and rank. Partial-block waste, metadata and replicated state are included or marked unavailable | What the request really holds right now |
| 3 | Reserved pools | Total engine KV reservation, allocated/unused blocks, and graph/workspace reservations | Reserved at startup whether or not any request uses it |
| 4 | Total device memory | Weights plus runtime pools, workspaces and other allocations, measured independently | Dominated by weights and reservations, not by one request |

A worked illustration of quantity 1, from the earlier MiMo walkthrough and labelled **estimate**: each of its 9 full-attention layers keeps (192 + 128) values per context token at 2 bytes each on a rank, so 9 × 320 × 2 B ≈ 5.8 KB per context token per rank, while its 39 sliding-window layers keep at most 128 tokens regardless of context. B3 produces the equivalent derivation for all three models and then checks it against quantity 2.

**Reading rules:**

- Report per-rank values and the physical sum across the eight GPUs. Replicated bytes are not unique logical state.
- The engine can reserve its KV pool at startup, so a flat `nvidia-smi` reading does not mean request KV is free.
- Derive marginal live-state bytes per token from adjacent context points where valid. Explain nonlinear or block-rounded behavior; do not force one constant slope.
- Record any preemption or recomputation, and any inability to sustain the intended active batch.

**What the figure may claim.** It answers "what does one live context cost?" It must not claim "how many shared prefixes fit?" Maximum request capacity needs a separate experiment.

## 7. B4 — FFN/MoE sparsity and its realized cost

**Purpose.** Show what "only a few experts are active" actually buys, and what it costs to get there.

Reuse the B2 traces and B3 memory metadata; do not launch another full matrix. For prefill and decode separately, report:

| Component | Notes |
| --- | --- |
| Router | Choosing experts for each token |
| Dispatch / permutation | Moving tokens to their experts |
| Routed-expert GEMMs | The expert computation itself |
| Shared experts / dense FFNs | Run for every token; not sparse |
| Combine | Merging expert outputs |
| Communication | TP collectives kept distinct from EP dispatch/combine |

Verify the executed backend. Enabling EP does not by itself imply a particular all-to-all path.

**Routing data to collect:** tokens assigned per expert and per rank, unique experts touched per step, max/mean routing load, actual GEMM shapes, and padding where exposed. Preserve domain labels, but note that mixed-traffic counters cannot establish separate code/math/chat performance. Measure the overhead of routing and communication instrumentation, and keep this diagnostic execution separate from serving timing.

**Why top-k alone misleads:**

- A larger batch can touch many experts even when each token activates few, so the weights read per step grow with B.
- Layer count, shared experts, expert width, quantization and batching can each reverse a ranking based on top-k. Historically, MiMo's 8 active experts and 0731's 6 routed experts showed similar decode MoE time, which is the kind of result the blog should explain rather than assume.
- Show both total model FFN cost and per-layer, per-token cost.
- Uneven routing is not, by itself, a bottleneck. Attribute straggler waits only with aligned rank timelines.

Pair every sparsity estimate with measured expert time and sampled HBM bytes, and state explicitly where coverage is missing.

## 8. Conditional follow-ups and budget discipline

Select additional work only after the initial comparison leaves a specific question open. Declare the points and their cost before collecting.

| Trigger | Bounded proposed extension |
| --- | --- |
| Need to explain large-batch expert/weight reuse | 1K/c64 AR for all three, three repeats = 9 timing runs, plus three actual B=64 decode captures if feasible |
| 64K leaves a context trend unresolved | 128K/c1,c8 AR across all three, three repeats = 18 timing runs, with selected additional diagnostics |
| A supported kernel or communication path may explain a measured cost | One variable changed for one model; two workload points; (baseline + intervention) × two points × three repeats = 12 fresh timing runs, plus correctness/overhead checks |
| A small difference supports a final recommendation | Held-out prompts and at least five independently scheduled blocks at selected points; declare the practical threshold and incremental count before collection |

**Not to be done on rented time:** unsupported CED, new kernels or faithful multi-head MTP development. Topology or precision changes require separate deployment arms with fresh controls. No automatic prefix, speculation or H200 extension.

**Budget.**

- Initial ledger: **54 unprofiled timing runs + 18 trace captures + 6 counter point sets**, with KV/routing diagnostics attached where feasible.
- Additional costs, budgeted separately: functional checks, pilots, warmups, diagnostic controls, counter replays, reruns and relaunches.
- Estimate node-hours from the actual pilot and startup times. Record an explicit allocation and stop point. A run count is never an hourly quote.

**Rental discipline.** Finish complete three-model comparisons first. Measure immediately after readiness, chain bounded work, stop only the owned server on every exit path (success, error or timeout), and verify GPU/process cleanup. Analyze while queued measurements run. No ready-server idle time.

## 9. Blog figures and agent handoff

**Structure of each blog section:** question → measurement → explanation → limitation.

**Audience and editing direction, 2026-10-09:** make the article accessible to university readers with basic computing knowledge. Introduce token, prefill/decode, throughput/latency, live KV state and expert routing where needed. Keep the main narrative focused on what the figures establish and why it matters. Use one concrete example to explain a metric; remove repeated rankings, run-by-run history and configuration lists. Preserve source data and material caveats, with detailed tables/settings in expandable notes or linked reports. The article need not reproduce every finding in the evidence bundle.

For the current article, apply section 11's source selection to every figure below. A five-model throughput chart includes Qwen BF16; a latency ranking spans only the four first-node deployments, with BF16 latency reported separately on its own node. Historical Qwen FP8 figures belong only in a labelled reference appendix.

**Figures and tables to prepare:**

| Output | Content | Guard against misreading |
| --- | --- | --- |
| Architecture diagram/table | One token through each model's attention and FFN | Paper design shown next to verified runtime behavior |
| Context curves | Prefill and decode latency versus actual context at declared batch/load | Shown alongside end-to-end serving anchors |
| Component breakdown | Attention core, projections, selection/compression, FFN, communication, and other/unattributed work | Kernel-sum charts are labelled as such; wall time is shown separately where overlap prevents an additive stack |
| HBM and KV figure | Inference bytes moved and live state bytes, in separate panels | Measured and estimated values are visibly distinguished |
| FFN sparsity figure | Nominal active experts/parameters next to actual expert coverage, GEMM time, routing imbalance and exposed communication | Nominal and realized values are never merged into one bar |
| Findings table | Each model's workload-specific strength and weakness, supporting artifact, explanation confidence and unresolved confounders | Negative and inconclusive findings are reported too |

**Suggested artifacts** under `reports/blog-architecture-h100-v1/`: `architecture.md`, `serving.csv`, `components.csv`, `memory.csv`, `routing.csv`, `findings.md` and `handoff.md`, backed by immutable manifests and raw results under the matching `results/` study. Each row needs model/runtime identity, phase, prompt hash, actual token shape, client c and engine B, rank coverage, units, repeat/attempt and evidence path.

**Using the historical studies.** The [DeepSeek](reports/v41-vs-0731/report.md) and [MiMo](reports/mimo-v26/report.md) results select hypotheses only. Apply the corrections in [update.md](update.md), especially kernel sum versus critical-path time, token pool versus capacity, and throughput accounting versus bottleneck evidence. Three repeats are a screen, not a confidence interval or a production tail-latency claim. Speed does not imply equal task quality.

**Handoff.** The handoff must:

- List every planned point as measured/validated, pending, failed or unsupported.
- Give the next exact commands, the missing instrumentation, the actual rental cost and time, and confirmation that the server was stopped.
- State unanswered questions as unanswered. Missing HBM counters leave the HBM question open even if every timing figure is complete, and the prefix-capacity question stays open until Tan requests that experiment.

**Publication.** Curate small public artifacts with hashes and documented sanitization. Exclude weights, private prompts and host details, credentials and large traces. Validate the evidence bundle with `python tools/audit_references.py` on a clean clone before publishing. Commit and push only when requested.

## 10. Addendum (2026-10-07): Qwen3.8-Flash-Next and GLM-5.3-Flash

**Historical FP8 record.** This completed addendum does not select the checkpoint for future Qwen runs. The 2026-10-08 BF16-only policy above supersedes its Qwen launch choice, while preserving its model pin and all results.

During the first GPU session Tan asked for two more models, measured **with the same setup, after the first three finish**. This widens the blog study only; it does not reopen the historical GLM/Qwen studies in [references/](references/README.md), whose numbers came from other builds and settings and are not controls here.

| Key | Checkpoint | Revision pinned for this study | Thinking off | Parsers |
| --- | --- | --- | --- | --- |
| `qwen-38` | `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` | `enable_thinking=false` | `qwen3` / `qwen3_xml` |
| `glm-53` | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | **not possible**; runs with `reasoning_effort=low` | `glm45` / `glm47` |

**Same as sections 2–7:** runtime, deployment flags, public request lists, request counts, warmup, validity and rerun rules, one diagnostic launch per model (functional checks, disjoint pilot, live-KV snapshots, six traces) and three timing blocks per model. The added ledger is 36 timing runs, 12 trace captures and 12 live-KV snapshots. The frozen record is `plan_addendum_glm_qwen.json` next to `plan.json`.

**What is different, and how to read it:**

- **Order.** The two models run after the first three, alternating with each other across blocks (block 1: Qwen, GLM; block 2: GLM, Qwen; block 3: Qwen, GLM). Model order is therefore not rotated across all five. A slow drift of the node would look like a difference between the two groups; the report says so wherever the groups are compared.
- **Counts.** Request counts were sized on the first three models. If an added model is faster than the fastest of them, its window can be under 60 seconds; the window length is reported.
- **Thinking on GLM.** Its chat template always opens the assistant turn with `<think>` and offers only `reasoning_effort` low/high/max, so the common "thinking off" setting cannot be met. It runs at the lowest effort. With 256 forced output tokens the number of decode steps is the same, but the tokens are reasoning text; this is a declared deviation, and GLM's natural-EOS check is expected to show reasoning content.
- **Flags.** Only the common deployment is used. A model-specific flag is added only if the common launch fails, and is then recorded as a deviation; the historical launches used such flags on older builds.
- **Architecture facts.** Both models must go through the same B0 check (config, server log, trace) before any explanation is written. The historical notes describe GLM as recurrent KDA plus sparse-attention layers and Qwen as 36 linear-attention plus 12 full-attention layers with 512 experts and ten active; treat these as claims to verify on this build.

## 11. Current blog evidence selection (2026-10-09)

**Presentation:** use **Qwen3.8-Flash-Next** in the model list, headings and full chart labels; **Qwen** is the compact label. BF16 is the checkpoint's precision, not a suffix required in the display name. Keep **Qwen FP8** explicit in the historical reference appendix. State consistently that all five models were measured with the **same core serving setup**: vLLM build, 8×H100 configuration, serving limits, request lists/counts and output length. Keep physical-node provenance, the MiMo control, native precision/backend differences and GLM's thinking exception in captions or measurement notes; shared setup does not mean a single physical node.

The current article is [index.html](index.html). All main results use vLLM `0.31.1rc1.dev50+g554340f3d`, commit `554340f3d3259e321be4c07282be7a02a5aeef83`. “Latest” means these completed October measurements, not a moving runtime or a new experiment.

| Article role | Study and model keys | Node | Evidence |
| --- | --- | --- | --- |
| Main comparison: DeepSeek 0731, V4.1, MiMo, GLM | `blog-architecture-h100-v1`: `v4-0731`, `v41`, `mimo-v26`, `glm-53` | First node, 2026-10-07/08 | [findings](reports/blog-architecture-h100-v1/findings.md), [serving](reports/blog-architecture-h100-v1/serving.csv), [components](reports/blog-architecture-h100-v1/components.csv), [memory](reports/blog-architecture-h100-v1/memory.csv) |
| Main Qwen deployment: original BF16 | `qwen-bf16-h100-v1`: `qwen-38-bf16` | Second node, 2026-10-08 | [findings](reports/qwen-bf16-h100-v1/findings.md), [serving](reports/qwen-bf16-h100-v1/serving.csv), [components](reports/qwen-bf16-h100-v1/components.csv), [memory](reports/qwen-bf16-h100-v1/memory.csv) |
| Node control, shown separately | `qwen-bf16-h100-v1`: `mimo-v26` | Second node | [control and comparison](reports/qwen-bf16-h100-v1/comparison.md); one timing block, plus diagnostic traces and snapshots |
| Historical reference appendix only | `blog-architecture-h100-v1`: `qwen-38` (**FP8**) | First node | [historical findings, section 8](reports/blog-architecture-h100-v1/findings.md#8-added-models-qwen38-flash-next-and-glm-53-flash) |

**Counts:** the main article selects 72 first-node timing runs plus 18 BF16 runs = **90 timing runs**, 24 + 6 = **30 trace captures**, and 24 + 6 = **30 live-KV snapshots**. The second-node MiMo control adds 6 timing runs, 6 traces and 6 snapshots, reported separately. Historical Qwen FP8 adds 18 timing runs, 6 traces and 6 snapshots, also separately. The first study's frozen totals remain 90/30/30 and the BF16 study's totals including its control remain 24/12/12; selecting article rows does not rewrite either ledger.

**How to compare and explain:**

- Carry the build and node provenance into captions and source links. Cross-node serving comparisons use **throughput only**, accompanied by the MiMo control (second/first-node throughput 0.983–0.995 on this build). Do not normalize Qwen by the control or attribute differences under about 2% to the model.
- Do not rank BF16 TTFT or eight-client TPOT against first-node values. Keep the main latency explorer on the first node and report BF16 latency in its own table. A standalone BF16 decode observation must say which node it comes from.
- Use BF16 traces, snapshots and startup logs for Qwen's main component/memory figures: **unquantized TRITON** experts, **31.42 GiB** loaded weights and **37.37 GiB** reserved KV per GPU on the second node, vLLM `554340f3…`. Never relabel FP8 measurements or backends as BF16. BF16 all-reduce kernel time includes profiler-induced waiting; trace spans are not unprofiled latency or isolated communication cost.
- Keep the FP8 comparison in a visibly labelled historical reference section. It compares deployments on different nodes, not an isolated precision effect. GLM's measured native FP8 weights and DeepSeek's native KV formats remain part of their deployments; “FP8 reference only” refers to the superseded **Qwen checkpoint**, not every use of FP8.
- Keep GLM's low-reasoning deviation, unrotated order across all five, short windows and slow first blocks visible. Timing, diagnostic kernel sums and approximate live allocation answer different questions. HBM counters, routing statistics, quality, saturation and reusable-prefix capacity remain unmeasured.

Preserve all frozen plans, raw runs, CSVs and curation ledgers. Refresh the article, README and handoff guidance from these selected sources; do not rerun completed points. The later speculative study remains pending and separate.

## 12. Later study: spec-realtext-h100-v1

Planned on 2026-09-30 and still **planned, not measured**. This study is separate from the blog: the blog's results and completion ledger do not complete or replace it, and its matrix, speculative arms and 2048-token workloads are not executed for the blog. The September DeepSeek and MiMo studies are complete; preserve their results in [the DeepSeek report](reports/v41-vs-0731/report.md) and [the MiMo report](reports/mimo-v26/report.md).

**Objective:** establish which deployment performs best for which workload on one 8×H100 80GB SXM node, and explain the evidence for its strengths and weaknesses. Compare absolute throughput, responsiveness, sustained generation, speculative progress/cost and task checks. A valid outcome can be a regression, an inconclusive comparison or an unsupported optimization. Speed alone does not establish equivalent task quality or an intrinsic architecture ranking.

Read [update.md](update.md) for the checked evidence and experiment rationale, [docs/three-model-h100-plan.md](docs/three-model-h100-plan.md) for the concrete run plan, and [docs/reproduce.md](docs/reproduce.md) for environment and curation. These scope rules supersede older experiment lists in historical plans and reports.

### 12.1 Scope

The study covers `v4-0731`, `v41` and `mimo-v26`, with the arms listed in [Models and immutable identities](#models-and-immutable-identities).

Pin target, tokenizer/remote code and draft separately. DSpark and MTP use the target snapshot; MiMo DFlash uses its `dflash/` subfolder. Verify resolved implementation, loaded layers and supported widths against vLLM commit `44af287ebe38d6dc4e102948025f5e3e175aefd6` (`0.30.1rc1.dev223+g44af287eb`). A newer build is a separate runtime arm with new AR controls. The architecture-blog study `blog-architecture-h100-v1` is such an arm: at Tan's request (2026-10-07) it runs on `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`) with its own fresh AR runs (sections 1–11). Its numbers are not controls for this study. Follow [the DeepSeek support audit](docs/speculative-decoding.md) and [the MiMo implementation record](docs/mimo-v2.6-plan.md), not config names alone.

**All new prefix-cache experiments are deferred until Tan explicitly requests them.** No cold/warm comparison, prewarming, reuse check, prefix pressure or speculation-plus-prefix arm. Disable prefix caching in every active run. Ordinary per-request KV allocation, occupancy and traffic remain in scope. Preserve and appropriately qualify existing prefix results.

Preview DeepSeek, GLM, Qwen and other models remain outside this study's scope. The preview's unverified revision is never a 0731 baseline. H200 is an optional separately requested future study, not part of this H100 session or its completion requirements. TP4×DP2/DP-attention and runtime changes are conditional tuned deployments, separate from the common baseline.

### 12.2 Active matrix and priorities

| Stage | Required core | Result needed |
| --- | --- | --- |
| Preparation | Extend the existing harness locally; freeze public real-text prompts, pins, manifest, validation and no-prefix chain | A dry-run manifest and tested collection/analysis before GPU rental |
| E0 | Each model: repeated same-seed AR and planned speculative methods, c1/c64, natural-EOS task checks; validate counters and implementations | Correctness/support record; unavailable telemetry explicitly identified |
| E1 screen | Balanced code/math/chat, approximately 1K input / 2048 output, c1/c64, 7 deployments × 3 repeats | 42 fresh timing runs; absolute and within-model speculative results |
| E1 controls | 16K/2048 and 16K/256 at c1/c64, same prompts between output-length arms | 84 more runs; isolate output-length sensitivity without changing prompt distribution |
| E1 crossover | 1K/2048 at c4/c16, same seven deployments × 3 | 42 more runs; a four-load short-input curve (168 core timing runs total) |
| E3 core | All seven deployments, initial 1K KV, actual B=1/64 decode-only intervals | 14 short diagnostic captures; progress, draft/verification cost and exposed communication |

E0, pilots, warmups, overhead controls, failed attempts and profiles are additional to the 168 timing runs. Stage the work; estimate wall time from a disjoint pilot, and record the session allocation before long sweeps. A partially run stage stays pending rather than becoming a completed study.

**Targeted tier, declare before collection:** E2 supported widths and fixed/adaptive DSpark; E4 real-text long-context sensitivity; separate domain throughput; held-out confirmation and latency validation; a model-specific communication or kernel ablation selected from traces. Do not launch the entire Cartesian product. The run-count and selection rules are in [the GPU plan](docs/three-model-h100-plan.md). No width, alternative kernel, faithful multi-layer MTP or CED implementation is assumed available.

### 12.3 Comparison and evidence rules

- Keep the common deployment at TP8 + EP, memory utilization 0.90, context limit 262,144, max sequences 64, explicit 8,192-token chunk budget, text only, temperature 0 and thinking off. Preserve native precision; record resolved graph, quantization and KV formats. DSpark draft sampling/rejection settings stay pinned. Separate any change to these settings into its own arm. This is the common deployment of [section 2](#2-common-deployment-and-local-preparation), on the build named in 12.1.
- Collect new AR controls for every model with the new workload. Compare the same texts across models and publish token counts; a separate token-count-matched view must be labelled. Context checks include rendered template, requested output and required speculative lookahead. Keep forced 256/2048-token timing separate from natural-EOS quality checks.
- Save immutable manifests, request IDs, prompt hashes, per-request status/tokens/timings, raw counters, launch logs and failures. Validate every completion and per-request output budget, identity and cache-off setting. Reruns use new directories and are paired by manifest identity. Missing telemetry is unavailable, never zero.
- Show mean, sample SD, median, repeat count and paired ratios. Three repeats screen large effects; small gains and final recommendations need held-out confirmation with independently scheduled blocks. Do not treat token samples as independent observations or reuse the old SD heuristic as a significance test. Screen p95 estimates are descriptive only.
- Separate client concurrency from actual engine batch and verification positions. Report actual committed tokens and critical-path wall time where observable; the old `1 + accepted/drafts` is a proxy. Never sum overlapping ranks/kernels or infer a prefill bottleneck from input/output accounting alone.
- Keep authoritative timing unprofiled. Quantify profiler overhead; eager/graph-disabled or patched instrumentation is a separate diagnostic runtime. Record source paths/commit, layers, shapes, fusion and unattributed time. Trace shares suggest interventions; a matched ablation is needed to claim the intervention caused an improvement.
- Evaluate each speculative arm against its own AR baseline first, then compare all models' absolute throughput and latency. Native methods are deployment comparisons, not isolated algorithm comparisons. Use `8/output_tok_s` for allocated GPU-seconds/token; dollar cost needs a supplied hourly price.
- Provide measured throughput/latency Pareto curves. No application SLO is supplied, so do not invent one or claim production goodput from closed-loop load. Open-loop SLO validation and capacity saturation require separate manifests. Task checks bound quality claims to the tested tasks.

### 12.4 Completion and handoff

The bounded H100 core is complete when every core point is measured and validated or explicitly failed/unsupported with evidence, all omitted points are accounted for, and the report contains:

1. Three-model serving curves and absolute/relative results with sample counts, uncertainty, output policy and task outcomes.
2. Support and acceptance tables separating scheduled, verified, accepted and committed tokens; unavailable fields and proxy quantities labelled.
3. Diagnostic phase/component costs with explicit denominators, critical-path versus kernel-sum distinction, trace/source links and unresolved explanations.
4. A strengths/weaknesses table for each model: workload, measured difference, explanation confidence, confounders and recommended or inconclusive operating point.
5. A portable evidence bundle and handoff listing completed/pending/failed/unsupported points, exact commands, pins, durations and next questions. Run `python tools/audit_references.py` on a clean clone before publication.

Missing core runs stay pending. Missing causal telemetry restricts explanations; it does not justify fabricated component timings. Conditional extensions are not core completion gates and cannot be reported as measured. Existing numerical reports remain intact, and the earlier source repository remains untouched.
