# Teach me: how these serving numbers are measured with vLLM

These lessons show, step by step, how the numbers in [the blog study report](../reports/blog-architecture-h100-v1/) were produced, and where in vLLM's own code each quantity is defined. Read them in order; each one ends with a "check yourself" list.

| Lesson | You will learn |
| --- | --- |
| [1. Set up the node and launch a server](01_setup_and_launch.md) | What to install, what each launch flag does, how to know the server is really ready |
| [2. Measure serving speed](02_measure_serving.md) | TTFT, TPOT and throughput: how the client computes them and how to run a valid point |
| [3. See where the time goes](03_traces_and_components.md) | Capturing a profiler trace for prefill and for decode at a known batch, and reading it |
| [4. Measure live KV memory](04_live_kv_memory.md) | Turning the KV-usage gauge into bytes per request and per token |
| [5. Read the vLLM source](05_reading_vllm_source.md) | A map of file and line references, and how the V4.1 prefill check was done |

The lessons use the three target models as examples. Qwen3.8-Flash-Next (`qwen-38`) and GLM-5.3-Flash (`glm-53`) were measured the same way; only the model key changes, and GLM has no thinking-off switch.

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
