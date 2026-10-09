# What was measured

The article asks how five deployments spend time and memory during inference. **Prefill** processes the input prompt; **decode** generates subsequent tokens. **TTFT** is the client's time to first token, including queueing. **TPOT** is a request's average time per output token after the first. **Throughput** is completed output tokens divided by the full benchmark duration, including prefill and scheduling.

All October numbers use vLLM `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`). The five models use the same core serving setup: TP8 plus expert parallel, 8×H100 80GB SXM, utilization 0.90, max context 262144, max sequences 64, explicit chunk budget 8192, native precision, prefix caching off and speculation off.

| Deployment | Source study | Physical node | Valid timing / traces / snapshots |
| --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 | `blog-architecture-h100-v1` | First | 18 / 6 / 6 |
| DeepSeek V4.1 Flash | `blog-architecture-h100-v1` | First | 18 / 6 / 6 |
| MiMo-V2.6-Flash-MOPD | `blog-architecture-h100-v1` | First | 18 / 6 / 6 |
| GLM-5.3-Flash | `blog-architecture-h100-v1` | First | 18 / 6 / 6 |
| Qwen3.8-Flash-Next | `qwen-bf16-h100-v1` | Second | 18 / 6 / 6 |
| MiMo node control (outside the five-model total) | `qwen-bf16-h100-v1` | Second | 6 / 6 / 6 |

These are publication-selection counts. The original frozen plans and completion ledgers have not been redefined; [provenance](provenance.md) explains how the branch selects evidence.

## Serving experiment

The same public text is sent as one user message to every model. Code comes from the node's installed CPython 3.12.3 standard library; math from `EleutherAI/hendrycks_math` at `21a5633873b6a120296cce3e2df9d5550074f4a3`; chat from `OpenAssistant/oasst1` at `fdf72ae0827c1cda404aff25b6603abec9e3399b`. The corpus builder and published input manifests record source hashes and licenses. The code corpus was not pinned to an upstream revision: matching the installed files and corpus hash is a reproduction requirement. The 0731 tokenizer sets approximate length buckets. Each model's own tokenizer and server chat template determine the actual prompt count; equal text does not mean equal tokens.

| Approximate input | Requests at c1 | Requests at c8 | Forced output per request |
| --- | ---: | ---: | ---: |
| 1K | 48 | 204 | 256 |
| 16K | 36 | 75 | 256 |
| 64K | 18 | 33 | 256 |

Here `c` is the client cap on open requests, not a fixed engine batch. Requests are balanced across code, math and chat. Timing uses the first N requests of the frozen timing split, temperature 0 and `ignore_eos`. Warmups and pilots use the disjoint auxiliary split. Held-out prompts were prepared but not used as a confirmation study.

Each model has three timing blocks on separate plain-server launches. Odd blocks visit 16K/c1, 16K/c8, 1K/c1, 1K/c8, 64K/c1, 64K/c8; even blocks reverse this order. The first three models rotated order. GLM ran afterward on the first node. Qwen ran on the second node in the sequence Qwen block 1 → MiMo control → Qwen blocks 2 and 3. There was no rotation across all five models.

A valid run completes all requests with exactly 256 output tokens each, no failed requests and at least 98% of the expected content-token count. Failed attempts and slow first blocks remain in the evidence. Means, sample SDs, medians and individual blocks are available; three blocks do not supply a confidence interval. Some windows fell below the intended 60 seconds because fixed counts were kept comparable.

DeepSeek uses `thinking=false`; MiMo and Qwen use `enable_thinking=false`. GLM's template has no thinking-off switch, so it uses `reasoning_effort=low`. Its forced-length check passes, but the natural-answer thinking-off check does not. This affects token content and prevents an equal-answer-quality interpretation.

## Attention and FFN diagnostics

Each model has prefill captures at 16K and 64K, plus decode captures at 1K and 64K with actual engine batch B=1 and B=8. Prefill comparisons use one 8192-token chunk, with the later 64K chunk having roughly 57K prior context. Profiles come from separate `off-profidle` launches, not timing launches.

The component CSV contains per-rank GPU kernel sums; the tables average eight ranks. Kernels and communication can overlap, so adding durations does not give elapsed latency. Decode steps are cut at the CPU-side `execute_context` annotation, especially for V4.1's multi-stream path. Attention core, indexer, recurrent state, KV preparation, dense projections, experts and communication are separate columns. Dense GEMM is mixed projection work, not an exact attention-projection total.

Qwen's traced 1K/B=1 decode step was about 8.9 ms versus 5.9 ms unprofiled. Its all-reduce durations include rank waiting; they are not isolated communication costs. The second-node MiMo trace control reproduced components within about 3%, with larger all-reduce variation. A 64K/B=1 Qwen stream emitted fewer text chunks than engine steps; use token usage and trace annotations rather than text chunks as the step counter.

Top-k expert counts describe nominal activation sparsity, not realized HBM traffic or speed. Routing histograms, expert coverage and padding statistics were unavailable. Hardware HBM counters were not collected because the image lacked Nsight Compute. Byte-traffic statements are estimates, not measured bandwidth.

## Live memory

Six snapshots per model cover 1K/16K/64K with B=1/8. Live state is approximated as the cache usage gauge multiplied by the per-rank reserved pool. Sum across eight ranks for physical node bytes, including replicated state. Multiple cache groups, recurrent state and block rounding matter, particularly 0731's 256-token blocks at short context.

Live state bytes, reserved KV pool and total device memory are different quantities. None establishes maximum request capacity or reusable-prefix capacity. Prefix caching was off.

## Two physical nodes

The first node used Xeon Platinum 8480+, two NUMA nodes and GPU driver 580.105.08. The second used Xeon Platinum 8592V, four NUMA nodes and driver 580.173.02, with an active CPU frequency governor. Full inventories and dependency lists are retained under each study's `study/_env/`.

MiMo's second/first throughput ratio is 0.983–0.995. Use the unadjusted throughput values beside one another, treating differences below about 2% as unresolved. TTFT and eight-client TPOT medians did not reproduce across nodes: **keep Qwen latency out of first-node rankings**. Short-prompt TTFT was bimodal within some second-node runs; inspect the per-request lists before interpreting a block median.

The studies compare deployments, not architecture alone. Native precision, kernels, tokenizers, node hardware and model order are meaningful limits. They do not establish quality parity, production tail latency, saturation, HBM traffic or prefix-reuse capacity.
