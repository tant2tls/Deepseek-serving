# Lesson 4: measure live KV memory

Four different memory quantities are easy to confuse. Keep them apart:

| Quantity | Where it comes from |
| --- | --- |
| Logical state | Arithmetic from the model config: bytes of keys/values per layer type per token |
| **Physical live allocation** | KV blocks occupied by live requests right now. This lesson |
| Reserved pool | What the engine reserved at startup for KV, used or not |
| Total device memory | `nvidia-smi`: weights + pool + workspaces |

## Where the numbers are

At startup each rank logs its pool (`$V/v1/worker/gpu_worker.py:727`) and the engine logs the token capacity (`$V/v1/core/kv_cache_utils.py:2481`):

```
Model loading took 20.1 GiB memory …
Available KV cache memory: 49.18 GiB
GPU KV cache size: 6,969,972 tokens, Maximum concurrency for 262,144 tokens per request: 26.59x
```

While serving, the gauge `vllm:kv_cache_usage_perc` (`$V/v1/metrics/loggers.py:614`) reports the fraction of KV blocks in use. It comes from the block pool (`$V/v1/core/block_pool.py:880`, `get_usage`): used blocks divided by total blocks. Despite the name it is a 0–1 fraction.

## The measurement

With B requests decoding and nothing else running (the setup of lesson 3):

```bash
curl -s localhost:8000/metrics | grep -E "^vllm:(kv_cache_usage_perc|num_requests_running)"
```

Then, per rank:

```
live bytes per rank = usage fraction × pool bytes per rank
live bytes on the node = 8 × live bytes per rank        (each rank holds its own copy or slice)
bytes per live token = live bytes per rank / (prompt tokens + generated tokens of the live requests)
```

Worked example from the MiMo snapshot at 16K with one request: usage 0.001871 × 49.18 GiB ≈ 94 MiB per rank. `bench/blog_report.py memory` does this for every snapshot and writes `memory.csv`.

## What this does and does not tell you

- It **does** tell you what one live context costs, including block rounding. Compare 1K, 16K and 64K to see whether cost grows linearly.
- It does **not** tell you how many requests fit (no saturation run), and nothing about reusable prefixes (prefix caching is off).
- Token capacities are not comparable across models: the three use different KV formats (`fp8_ds_mla` for DeepSeek, BF16 for MiMo) and different block accounting. Compare **bytes**.
- `nvidia-smi` stays flat while requests come and go, because the pool is reserved at startup. Flat does not mean free.
- The usage fraction assumes every block costs the same bytes. Models with several cache groups (the server logs `kv cache group sizes […]`) make this an approximation; say so next to the number.

## Check yourself

- One request uses 0.19% of the pool and eight use 1.55%. Is that linear? (8 × 0.187% = 1.50%; the extra is generated tokens and block rounding.)
- Why multiply by 8 for the node total? (Under TP every rank keeps KV for its own heads; replicated heads are real bytes even though they are not unique information.)
