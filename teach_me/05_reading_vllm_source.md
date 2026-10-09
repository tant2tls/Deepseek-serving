# Lesson 5: read the vLLM source

A config field or a paper feature does not prove that a build executes it. The habit to learn: **find the code, find the log line, then find the kernels in a trace.**

`$V` is `/root/vllm-latest/lib/python3.12/site-packages/vllm` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`). On another build, search for the quoted text: `grep -rn "<text>" $V`.

## Map of the code behind each number

Tip: every server log line carries `[file.py:line]`, so the log itself is an index into the source.

| What | File and line | Text to search for on another build |
| --- | --- | --- |
| Client TTFT | `$V/benchmarks/lib/endpoint_request_func.py:438` | `output.ttft = timestamp - st` |
| Client inter-token latency | `$V/benchmarks/lib/endpoint_request_func.py:442` | `output.itl.append` |
| Client TPOT | `$V/benchmarks/serve.py:627` | `latency_minus_ttft` |
| Output tok/s | `$V/benchmarks/serve.py:751` | `output_throughput=sum(actual_output_lens)` |
| Server prefill time | `$V/v1/metrics/stats.py:588` | `prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts` |
| Server queue / decode time | `$V/v1/metrics/stats.py:584`, `:592` | `queued_time =`, `decode_time =` |
| Prometheus names | `$V/v1/metrics/loggers.py:889`, `:929`, `:949`, `:959` | `vllm:time_to_first_token_seconds`, `vllm:request_prefill_time_seconds` |
| KV usage gauge | `$V/v1/metrics/loggers.py:614`; source `$V/v1/core/block_pool.py:880` | `vllm:kv_cache_usage_perc`, `def get_usage` |
| KV pool size log | `$V/v1/worker/gpu_worker.py:727`, `$V/v1/core/kv_cache_utils.py:2481` | `Available KV cache memory`, `GPU KV cache size` |
| Chunked prefill budget | `$V/config/scheduler.py:309`; scheduler loop `$V/v1/core/sched/scheduler.py:597` | `Chunked prefill is enabled`, `token_budget = self.max_num_scheduled_tokens` |
| Step annotation in traces | `$V/v1/worker/gpu_worker.py:1241` | `"execute_context_"` |
| Profiler endpoints | `$V/entrypoints/serve/profile/api_router.py:35` | `@router.post("/start_profile")` |
| MiMo decoder | `$V/model_executor/models/mimo_v2.py:322` | the server log line `[mimo_v2.py:322] Using FLASH_ATTN_DIFFKV for attention.` names the file and line |
| MiMo MTP depth (one layer reused) | `$V/model_executor/models/mimo_v2_mtp.py:177` | `num_mtp_layers = 1` |
| MXFP4 MoE backend choice | `$V/model_executor/layers/quantization/mxfp4.py:758` | the server log line `[mxfp4.py:758] Using 'HUMMING' Mxfp4 MoE backend.` |
| V4.1 prefill skip: where the cut is set | `$V/models/deepseek_v41/nvidia/model.py:757`–`793` | `Decoder SWA bounded replay` |
| V4.1 prefill skip: when it applies | `$V/models/deepseek_v41/nvidia/model_state.py:437`–`460` | `def _prepare_replay_batch` |
| V4.1 prefill skip: running the trimmed batch | `$V/models/deepseek_v41/decoder_replay_layers.py:85` | `def _run(` |
| V4.1 trim threshold (768 tokens) | `$V/config/compilation.py:585` | `decoder_replay_trim_threshold` |
| When the skip is disabled | `$V/models/deepseek_v41/nvidia/model.py:1156` | `def _decoder_replay_supported` |

Print any of them with context:

```bash
V=/root/vllm-latest/lib/python3.12/site-packages/vllm
sed -n 580,600p $V/v1/metrics/stats.py
```

## Worked example: is V4.1's prefill measured as the report describes?

The question: the technical report says V4.1 is a 20-layer encoder plus a 20-layer decoder and activates half the model during prefill. Does vLLM measure that?

**Step 1: what does the report claim?** Read the model card in the snapshot:

```bash
grep -n "Causal Encoder-Decoder" /workspace/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*/README.md
```

**Step 2: what does the config say?**

```bash
python3 -c "import json,glob; c=json.load(open(glob.glob('/workspace/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*/config.json')[0]))['text_config']; print(c['num_hidden_layers'], c['kv_source_layer_ids'], c['sliding_window'])"
# 40 [2, 8, 14, 20] 128
```

**Step 3: what does the code do with it?** In `model.py:762` the cut is `max(config.kv_source_layer_ids)`, which is 20, so layers 21–39 become "replay layers". In `model_state.py:465` each request keeps `min(query_len, window)` rows for those layers. The condition at `model_state.py:455` says the trim applies in eager steps and in piecewise-graph steps of at least `decoder_replay_trim_threshold` tokens.

**Step 4: does the server say it is on?**

```bash
grep "bounded replay" results/blog-architecture-h100-v1/v41/_server/serve-*.log | head -1
# Decoder SWA bounded replay: in eager prefill steps, layers 21-39 run on each request's last 128 tokens only.
```

**Step 5: do the kernels agree?**

```bash
python bench/blog_layers.py results/blog-architecture-h100-v1/v41/profiles/prefill64k/*rank0*.json.gz --pattern sparse_attn_fwd
# calls 40 … long 21 (median 1.352 ms); short 19 (median 0.027 ms)
```

Twenty-one layers work on the whole 8,192-token chunk and nineteen on a 128-token window.

**Step 6: how is the time taken?** `prefill_time = first_token_ts − scheduled_ts` (`stats.py:588`): a wall-clock interval around whatever ran. The timer is right; the question was always which layers run.

**Conclusion.** On this build prefill follows the report's design with two differences: 21 full layers, not 20, and no skip for steps under 768 tokens. On the old pinned build the same check finds 40 full calls, which is why the old V4.1 prefill numbers were slow. The full write-up is [v41_prefill_check.md](../reports/blog-architecture-h100-v1/v41_prefill_check.md).

## The general recipe

1. State the claim in one sentence, with its source.
2. Find the config fields involved.
3. Find the code that reads those fields; note the conditions around it.
4. Find the startup log line that reports the resolved behavior.
5. Find the kernels in a trace and count or time them.
6. Only then time it, with the plain server.
7. Write down the build. The same checkpoint behaved differently on two builds one week apart.

## Check yourself

- Why is step 5 needed if step 4 already shows a log line? (A log line states intent at startup; conditions such as the 768-token threshold decide what happens in each step.)
- A 1K prompt arrives. Does the skip apply? (Its single prefill step is over 768 tokens, so yes for that step; a prompt under 768 tokens runs all 40 layers.)
