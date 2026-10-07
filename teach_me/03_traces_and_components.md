# Lesson 3: see where the time goes

Timing tells you *how fast*; a trace tells you *what ran*. Traces are diagnostics: they run on the profiler launch (`off-profidle`) and never enter the timing tables.

## How vLLM marks a step

Every engine step is wrapped in an annotation built in `$V/v1/worker/gpu_worker.py:1241`:

```
execute_context_<prefill requests>(<prefill tokens>)_generation_<decode requests>(<decode tokens>)
```

`execute_context_1(8192)_generation_0(0)` is one request prefilling an 8,192-token chunk. `execute_context_0(0)_generation_8(8)` is a pure decode step with eight live sequences. This is how we know the **actual** engine batch, whatever the client concurrency was.

## Capture a prefill trace

```bash
curl -s -X POST localhost:8000/start_profile        # $V/entrypoints/serve/profile/api_router.py:35
# send exactly one request (max_tokens 8, ignore_eos) and wait for the answer
curl -s -X POST localhost:8000/stop_profile
ls results/<study>/<model>/profiles/                # one *.pt.trace.json.gz per rank, written asynchronously
```

`bench/blog_study.py: prefill_capture()` does this with a frozen public prompt and files the eight rank traces under `profiles/prefill16k/`.

## Capture a decode trace at a known batch

A decode-only window needs all B requests past prefill and no new arrivals:

1. Start B streaming requests (`max_tokens` large, `ignore_eos`).
2. Wait until every stream has produced tokens. Now the engine holds B decoding sequences.
3. Read `/metrics` for the live KV snapshot (lesson 4).
4. `POST /start_profile`, let about 48 steps run, `POST /stop_profile`.
5. Close the streams.

This is `decode_capture()` in `bench/blog_study.py`. One trap: with `ignore_eos` a model that passed its natural end keeps emitting special tokens, which produce no text chunks. Count progress on the busiest stream and read B from the annotations.

## Read a trace

```bash
python bench/trace_breakdown.py <rank0 trace.json.gz>     # per step: span, kernel sum, time per category
python bench/blog_report.py components                     # all ranks, all captures -> components.csv
python bench/blog_layers.py <rank0 trace.json.gz>          # per-layer calls of one kernel in the last full chunk
```

The classifier maps kernel names to components by regular expression (`CATS` in `bench/trace_breakdown.py`, extended for this build in `bench/blog_report.py`). After any build change, read `components_unclassified.md`: a large kernel left in `other_elementwise` means the map is out of date. On this build the MoE expert GEMM is `humming<…>`; DeepSeek also uses `humming` for low-rank attention projections, so the two are separated by weight shape.

## How steps are cut out of a decode trace

The first attempt summed the kernels inside the GPU-side `execute_context` span on the main stream. That works for MiMo and 0731, where one span covers the step. V4.1 runs a decode step as several graph segments on several streams, so that rule saw only 7–57% of its step. The fix in `bench/blog_report.py: breakdown(until_next=True)`: there is exactly one **CPU-side** `execute_context` annotation per engine step, so a step owns every kernel from its CPU annotation start to the next one. The check that the cut is right is `span_share_of_step` in `components.csv`: the step length seen in the trace divided by the step time seen by the client in the same window. It should be close to 1 (it is 0.93–1.00 for all twelve decode captures).

The same client-side counts give the profiler's overhead: each decode capture measures one second without the profiler and then the profiled window, at the same batch and context. On this node the profiled step was 0.4–12% longer.

## Three ways to misread a trace

| Mistake | Why it is wrong | What to do |
| --- | --- | --- |
| Treat the kernel sum as latency | Kernels on different streams overlap; idle gaps are not kernels | Report the step `span` next to the sum |
| Read one rank as the whole node | The other seven ranks run at the same time, and a rank can wait for the slowest | Use all ranks; report waits separately |
| Trust a tiny kernel list | The step may be split over streams or hidden in a graph | Compare the trace step length with the client's step time before using the breakdown |

## Check yourself

- Which annotation proves a decode window had exactly eight sequences and no prefill?
- A chunk shows 21 long and 19 very short attention calls. What does that tell you about the model? (Lesson 5.)
- Why can the profiled request be slower than the timed one? (The profiler records every kernel; that overhead is why traces are not timing results.)
