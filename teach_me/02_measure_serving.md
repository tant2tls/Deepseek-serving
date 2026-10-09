# Lesson 2: measure serving speed

## What the client measures

vLLM ships a load generator, `vllm bench serve`. For each streamed request it records timestamps and derives the metrics. In `$V/benchmarks/lib/endpoint_request_func.py` (chat-completions path):

| Line | Code | Meaning |
| --- | --- | --- |
| 438 | `output.ttft = timestamp - st` | **TTFT**: first streamed chunk minus send time |
| 442 | `output.itl.append(timestamp - most_recent_timestamp)` | Inter-token latency: gap between stream chunks |
| 462 | `output.latency = most_recent_timestamp - st` | End-to-end latency |

and in `$V/benchmarks/serve.py`:

| Line | Code | Meaning |
| --- | --- | --- |
| 627–628 | `tpot = (latency - ttft) / (output_len - 1)` | **TPOT**: average time per output token after the first, per request |
| 751 | `output_throughput = sum(actual_output_lens) / dur_s` | **Output tok/s** over the whole run, including ramp-up and drain |

So TTFT contains queueing, tokenization, every prefill chunk and the first decode step. TPOT is a per-request average, not a per-step time: with eight requests in flight, each step serves eight tokens and other requests' prefill chunks interleave with decode steps, so TPOT rises.

## One measured point by hand

```bash
$VLLM_VENV/bin/vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions --base-url http://localhost:8000 \
  --model XiaomiMiMo/MiMo-V2.6-Flash-MOPD --tokenizer-mode auto \
  --dataset-name custom --dataset-path results/blog-architecture-h100-v1/_inputs/16k/timing.jsonl \
  --skip-chat-template --disable-shuffle \
  --request-rate inf --max-concurrency 8 --num-prompts 75 --seed 0 \
  --ignore-eos --temperature 0 \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
  --save-result --save-detailed --result-dir /tmp/demo --result-filename bench.json
```

Why each choice:

- `--dataset-name custom … --disable-shuffle`: our frozen public texts, first *n* lines, same order for every model. Each line is `{"prompt": …, "output_tokens": 256}`.
- `--skip-chat-template`: the client must not render the template, because the server renders it. Rendering twice changes the prompt.
- `--request-rate inf --max-concurrency 8`: closed loop; the client keeps eight requests open.
- `--ignore-eos` with 256 output tokens: every request generates exactly 256 tokens, so models are compared on equal work. Natural stopping is checked separately.
- `--save-detailed`: per-request `ttfts`, `itls`, `output_lens` in `bench.json`.

## The same point through the harness

```bash
bash bench/blog_launch.sh chain-timing timing:mimo-v26:1
```

`bench/blog_study.py` calls `run_matrix.run_point`, which writes for each run:

```
results/blog-architecture-h100-v1/<model>/serving/off/ctx16k_osl256_c8/repeat-1/
  manifest.json          exact command, seed, timestamps
  bench.json             client results, per request
  summary.json           headline numbers + server counter deltas + validity
  blog_validation.json   request IDs and hashes, per-request output lengths, running-requests samples
  telemetry/             /metrics before and after, 1 Hz gauges, nvidia-smi
```

## What makes a run valid

1. Every request completed, none failed.
2. Every request returned exactly 256 tokens (`output_lens` in `bench.json`).
3. Server-counted prompt tokens (`vllm:prompt_tokens_total` delta) are at least 98% of the tokens we expect from the frozen list.
4. Zero preemptions is not required, but any preemption is reported.

## How many requests?

Start at `max(16, 4 × concurrency)`, make it a multiple of 3 (code/math/chat), and raise it until the fastest model runs at least 60 seconds. Get the rates from a **pilot on different prompts**:

```bash
python bench/blog_report.py pilot      # prints requests/s per model and the implied counts
```

Then freeze the counts in `plan.json` and validate before timing:

```bash
$VLLM_VENV/bin/python bench/blog_study.py plan --counts '<json printed by the pilot>'
$VLLM_VENV/bin/python bench/blog_study.py dry-run timing:v4-0731:1 timing:mimo-v26:1 timing:v41:1
```

## Repeats and order

Three repeat blocks, each on a fresh launch, with the model order rotated (block 1: 0731, MiMo, V4.1; block 2: MiMo, V4.1, 0731; block 3: V4.1, 0731, MiMo). Rotation keeps a slow drift of the node from always landing on the same model. Report mean, sample SD, median and the three values; three repeats screen effects, they are not a confidence interval.

## Tables

```bash
python bench/blog_report.py serving    # reports/blog-architecture-h100-v1/serving.csv and serving_tables.md
```

## Check yourself

- A model has lower TTFT but lower tok/s at c8. Is that a contradiction? (No: TTFT is mostly prefill, tok/s over a 256-token answer also depends on decode step time and how steps are shared.)
- Why is `output tok/s = input tok/s × 256 / input length` not evidence of a prefill bottleneck? (It is an identity for fixed lengths; it holds whatever the bottleneck is.)
- Where is the real engine batch recorded? (`running_requests_1hz` in `blog_validation.json`, and exactly in trace annotations, lesson 3.)
