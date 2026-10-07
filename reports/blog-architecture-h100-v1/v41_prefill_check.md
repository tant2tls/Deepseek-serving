# Does vLLM measure DeepSeek V4.1 Flash prefill the way the technical report describes it?

Checked 2026-10-07 on vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`), TP8 + EP, 8×H100. Evidence labels: **report** (model card), **source** (vLLM code read on the node), **log** (server startup log), **trace** (executed GPU kernels), **arithmetic**.

## Short answer

- **The timer is correct; what matters is which layers the build executes.** vLLM times wall clock around whatever it runs. It does not model the architecture.
- **On the old pinned build (`44af287e…`) the answer was "no".** All 40 layers processed every prompt token, so the historical V4.1 prefill numbers in [report.md](../../report.md) (54.5 µs per token, 399 ms kernel sum per chunk) describe a path the report does not intend.
- **On the new build the answer is "mostly yes".** During prefill, layers 0–20 process every prompt token and layers 21–39 process only each request's last 128 tokens. That is **21 full layers, not the report's 20**, and the skip applies only to prefill steps of at least 768 tokens.

## What the technical report says

*Report:* V4.1 Flash is a 40-layer Causal Encoder-Decoder: a 20-layer causal encoder followed by a 20-layer decoder. The decoder's global KV cache is projected from the final encoder hidden states, so the model activates about **8B parameters per token during prefill and 16B during decode**. "SWA Bounded Replay" rebuilds the decoder's sliding-window state by replaying only the most recent window of tokens.

In plain terms: a prompt token only needs the encoder half. The decoder half is needed for the last window of tokens (128) so that generation can start.

## What the new build executes

*Source* (`vllm/models/deepseek_v41/nvidia/model.py`, `decoder_replay_layers.py`, `nvidia/model_state.py`):

- The cut is `max(kv_source_layer_ids)`. The checkpoint config has `kv_source_layer_ids = [2, 8, 14, 20]`, so the cut is layer 20 and the "replay layers" are 21–39.
- In a prefill step the replay layers run on a sub-batch holding `min(query_len, 128)` rows per request; the other rows are skipped.
- The trim happens in eager steps and in PIECEWISE CUDA-graph steps of at least `decoder_replay_trim_threshold = 768` tokens. Smaller steps and FULL-graph steps (ordinary decode) run all 40 layers on the whole batch.
- It is switched off for requests that ask for prompt logprobs, and for some configurations (sequence or prefill-context parallelism, microbatching, a drafter that reads outside the window).

*Log:* `Decoder SWA bounded replay: in eager prefill steps, layers 21-39 run on each request's last 128 tokens only.`

*Trace* (rank 0, last full 8,192-token chunk of one public-text request; [16K](data/v41_prefill16k_attention_calls_rank0.json), [64K](data/v41_prefill64k_attention_calls_rank0.json), expert GEMM calls [16K](data/v41_prefill16k_expert_gemm_calls_rank0.json), [64K](data/v41_prefill64k_expert_gemm_calls_rank0.json)):

| Kernel in one 8,192-token chunk | Calls on the full chunk | Calls on the trimmed batch |
| --- | --- | --- |
| Sparse attention (`sparse_attn_fwd`) | 21, median 1.25 ms (16K) / 1.35 ms (64K) | 19, median 0.024 ms / 0.027 ms |
| Expert GEMM, gate/up (`humming`, 4608×5120) | 21, median 3.33 ms / 3.42 ms | 19, median 0.56 ms / 0.55 ms |
| Expert GEMM, down (`humming`, 5120×2304) | 21, median 1.88 ms / 1.93 ms | 19, median 0.30 ms / 0.30 ms |

So the executed kernels match the source: 21 layers do full-chunk work and 19 layers do window-sized work. The call lists are in time order in the JSON files. Use `python bench/blog_layers.py <rank trace>` to repeat the check on any trace.

## Where the build differs from the report

| Point | Report | New build | Size of the difference |
| --- | --- | --- | --- |
| Layers run on every prompt token | 20 (encoder) | 21 (layers 0–20) | One extra full layer, about 1/20 of the full-chunk layer work (*arithmetic*). Layer 20 is the last KV source, and the runtime runs the whole layer, not only its KV projection |
| Decoder layers during prefill | Replay the last window once | Each prefill step replays its own last 128 tokens per request | A 64K prompt in 8 chunks replays 8 windows; each costs well under 1 ms per layer on rank 0 (*trace*) |
| Short prompts and prompt tails | Not discussed | Steps below 768 tokens run all 40 layers | A 1K prompt gets about 2 full-depth steps' worth less benefit; do not expect a 2× gain at 1K |
| Activated parameters | 8B prefill / 16B decode | Not measured | Parameter counts are estimates; time and bytes are what we measure |

Whether a tighter cut (project layer 20's KV for all tokens, run the rest of layer 20 on the window only) would be valid is a **hypothesis**; it needs a correctness check, not only a source reading.

## How the times are defined

| Quantity | Definition | What it includes |
| --- | --- | --- |
| vLLM `request_prefill_time` | `first_token_ts − scheduled_ts` (`vllm/v1/metrics/stats.py`) | All prefill chunks plus the step that emits the first token, and any time the request waits between its chunks. Not queue time before scheduling |
| Client TTFT (our tables) | First streamed token minus send time | The above plus queueing, tokenization, templating and HTTP |
| Trace chunk span | One `execute_context` annotation on rank 0 | GPU work of one step; kernel sums are not wall time |

All three time the executed path. None of them subtracts the decoder work, and none needs to: after the trim the decoder work is small but real.

## Why this matters for the comparison

The two builds give different answers for the same checkpoint, so the runtime must always be named. On the new build the diagnostic single-request prefill (public text, profiler on, not a timing result) took 0.63 s for 16,309 prompt tokens and 2.33 s for 65,502 on V4.1, against 0.74 s for 15,325 and 2.95 s for 62,699 on MiMo. The unprofiled serving tables are the authority for ranking; these values only show the direction is consistent with the skip being active.

The earlier statement "CED prompt skipping is absent" remains true **for the old pinned build only** and must not be repeated for the new one.
