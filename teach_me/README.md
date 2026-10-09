# Teach me: how to measure a served model with vLLM, and how to set it up anywhere

These lessons show, step by step, how the numbers in the [five-model results](../reports/five-model/results.md) were produced, where in vLLM's own code each quantity is defined, and how to repeat the work on a system that is not ours. Read them in order; each one ends with a "check yourself" list.

They teach the method. For the commands of one model see [reproduce/](../reproduce/README.md); for the rules of the comparison see [docs/experiments.md](../docs/experiments.md).

## From Kan's questions to a measurement

[target.md](../target.md) holds the brief. Each question becomes a number you can collect:

| Kan's question | The number | How it is collected | Lesson |
| --- | --- | --- | --- |
| What is each model's architecture, as executed? | Model class, backends and kernels the runtime loads | Server log lines, then kernels in a trace | [1](01_setup_and_launch.md), [5](05_reading_vllm_source.md) |
| Whose attention runs fastest? | GPU kernel ms per engine step, by component | Profiler trace of one prefill chunk or of decode at a known batch | [3](03_traces_and_components.md) |
| Whose state takes less space? | Bytes held by live requests | KV-usage gauge × per-rank pool, at a known batch | [4](04_live_kv_memory.md) |
| What are the FFN sparsity differences? | Expert and routing kernel ms; expert structure from the config | The same traces, expert columns | [3](03_traces_and_components.md), [5](05_reading_vllm_source.md) |
| "Feel free to add more" | TTFT, TPOT, output tokens/s | `vllm bench serve` on the plain server | [2](02_measure_serving.md) |
| Can numbers from two launches, checkpoints or nodes be compared? | A control ratio | One already-measured model rerun under the new condition | [6](06_new_checkpoint_new_study.md), [7](07_serving_on_a_different_system.md) |

Some parts of the brief are **not** answered by these methods: measured HBM traffic needs hardware counters, expert-routing statistics need runtime instrumentation, and reusable-prefix capacity needs prefix caching, which was off. [target.md](../target.md#still-open) lists them.

## The lessons

| Lesson | You will learn |
| --- | --- |
| [1. Set up the node and launch a server](01_setup_and_launch.md) | What to install, what each launch flag does, how to know the server is really ready |
| [2. Measure serving speed](02_measure_serving.md) | TTFT, TPOT and throughput: how the client computes them and how to run a valid point |
| [3. See where the time goes](03_traces_and_components.md) | Capturing a profiler trace for prefill and for decode at a known batch, and reading it |
| [4. Measure live KV memory](04_live_kv_memory.md) | Turning the KV-usage gauge into bytes per request and per token |
| [5. Read the vLLM source](05_reading_vllm_source.md) | A map of file and line references, and how the V4.1 prefill check was done |
| [6. A new checkpoint is a new study](06_new_checkpoint_new_study.md) | How the original BF16 Qwen checkpoint was measured beside the finished FP8 study: distinct key, same inputs, loaded-dtype check, node control |
| [7. Set up serving on a different system](07_serving_on_a_different_system.md) | What to inspect on a new machine, which settings depend on it, and the control to run before comparing with our numbers |

Two reading paths:

- **I want to measure.** Lessons 1 to 5, then 6 when a second checkpoint or node appears.
- **I have a different machine.** Lesson 1 for the launch, lesson 7 for the system, lesson 6 for the control.

The lessons use the three first models as examples. GLM-5.3-Flash (`glm-53`, no thinking-off switch) and Qwen3.8-Flash-Next were measured with the same core setup. **Qwen has two records:** the historical FP8 checkpoint (`qwen-38`, retired, never launched again) and the original BF16 checkpoint (`qwen-38-bf16`, study `qwen-bf16-h100-v1`, required for all Qwen work from 2026-10-08). Changing a dtype flag does not turn one into the other; lesson 6 and the [checkpoint policy](../reproduce/qwen-38-bf16.md#checkpoint-policy) explain why.

**Everything here refers to one build.** Line numbers are for vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`) installed at `/root/vllm-latest`. Write `$V` for `/root/vllm-latest/lib/python3.12/site-packages/vllm`. On another build, find the same code with `grep -n "<text>" -r $V` using the quoted strings in lesson 5.

## The mental model in five sentences

1. A request has two phases: **prefill** reads the prompt (thousands of tokens per step), **decode** writes the answer (one token per live request per step).
2. The client sees **TTFT** (mostly prefill plus waiting) and **TPOT** (decode step time, which grows when requests share the engine).
3. The server is eight processes ("ranks"), one per GPU; every layer ends with a cross-GPU sum (**all-reduce**), so communication is part of every step.
4. A profiler **trace** lists every GPU kernel inside each engine step; summing kernels by type shows where GPU time goes, but a sum is not elapsed time.
5. Numbers are only comparable when the build, the launch flags, the prompts and the validity rules are the same; always record all four.

## Rules that keep the numbers honest

- Time with the plain server (`off`). Explain with the profiler server (`off-profidle`). Never mix the two in one table.
- A run is valid only if every request finished and returned exactly the forced 256 tokens. Keep invalid runs; rerun into a new directory.
- Client concurrency (`c8`) is not the engine batch. Read the real batch from the trace annotation or the running-requests gauge.
- Stop the server you started, and confirm 0 MiB on all GPUs. The node is billed by the hour.
- Running any of this on rented GPUs needs explicit authorization. The lessons explain completed work; they are not a work queue.

## Adding a lesson

Number it after the last one (`08_<topic>.md`) and add it to both tables above. Follow the shape of the existing lessons: start from something that was observed, show the command, then explain the mechanism, and end with "check yourself". Quote file paths and log lines exactly, and name the build they belong to. Put per-model commands in [reproduce/](../reproduce/README.md) and link to them instead of copying.
