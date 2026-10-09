# Target: answer Kan's architecture questions on 8×H100, and publish a blog everyone can use

Updated 2026-10-10. **This is the single target file of the five-model publication branch.** It states what the measurements are for, which checkpoints they cover and what is answered so far. It is the publication view of the full archive's [target.md](https://github.com/tant2tls/Deepseek-serving/blob/4cfa8d5f4d4fc7da3ed6248fda8e5a09004f6bcf/target.md) on the `all-data` branch, which also keeps the complete measurement plan, the frozen study ledgers and the later speculative study.

**Two purposes, in this order:**

1. **Measure, to answer Kan's question.** Each part of [the brief](#the-brief) gets an answer backed by evidence collected on 8×H100, or is stated plainly as unanswered.
2. **Publish a blog that is useful to everyone who reads it.** A reader new to inference systems can follow every figure, and a practitioner can trace every number to its run, build and node.

No new GPU session, prefix-cache experiment or speculative sweep is authorized by this file.

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
| Whose attention moves the least data through HBM? | Hardware read/write counters were planned | **Unanswered.** No counters were collected (no Nsight Compute on the image); byte-traffic statements are estimates | [Experiments and limits](docs/experiments.md) |
| Which model's cached state takes less space? | Bytes held by live requests' KV/state, six snapshots per model | **Partly answered.** Live state is measured as an approximation (usage gauge × per-rank pool). Reusable-prefix capacity is **unanswered**: prefix caching was off | [Live state memory](reports/five-model/results.md#live-state-memory-at-64k-b1) |
| What are the FFN sparsity differences? | Nominal expert structure, and trace-level expert and routing time | **Partly answered.** Routing histograms, expert coverage and padding statistics were unavailable | [Trace components](reports/five-model/results.md#trace-components) |
| "Feel free to add more" | TTFT, TPOT and throughput on public text, from unprofiled runs | **Answered**; across the two nodes only throughput is compared | [Output throughput](reports/five-model/results.md#output-throughput) |

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

## Common setup and boundaries

- **Runtime:** vLLM `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`) for all five. The earlier DeepSeek/MiMo studies used `44af287ebe38d6dc4e102948025f5e3e175aefd6`; a number from one build is never a control for the other.
- **Deployment:** 8×H100 80GB SXM, TP8 plus expert parallel, utilization 0.90, context limit 262144, at most 64 sequences, explicit 8192-token chunk budget, native precision.
- **Workload:** the same public code/math/chat text for every model, approximately 1K/16K/64K input, 256 forced output tokens, one or eight clients, three timing blocks. Autoregressive decoding only: speculation off and prefix caching off in every launch.
- **Two physical nodes:** four deployments on the first node; Qwen and a MiMo control on the second. Compare throughput across nodes and keep Qwen latency separate.
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

- **Measured HBM traffic.** Needs Nsight Compute installed and validated on the node before a rental.
- **Expert-routing statistics** (tokens per expert, coverage, padding). Need runtime instrumentation.
- **Reusable-prefix capacity and reuse.** Deferred until explicitly requested; live state bytes do not establish it.
- **Answer quality, saturation and production tail latency** were not measured.

The later `spec-realtext-h100-v1` study remains planned, not measured, and is outside this branch's reproduction queue. Its plan is [section 12 of the full archive's target.md](https://github.com/tant2tls/Deepseek-serving/blob/4cfa8d5f4d4fc7da3ed6248fda8e5a09004f6bcf/target.md#12-later-study-spec-realtext-h100-v1).
