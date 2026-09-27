# DeepSeek V4 Flash reference

Self-contained reference for the [V4 versus V4.1 measurement plan](../../target.md). No access to the previous repository is required.

## Included evidence

[results/mtp-off-image/](results/mtp-off-image/) contains 11 original result JSONs, their 11 benchmark logs, and the [environment manifest](results/mtp-off-image/manifest.txt). Numerical JSONs are unchanged and SHA-256 checked against the source. Logs/manifests are publication exports with any host/path or terminal-formatting sanitization recorded in the [provenance ledger](../provenance.json). These are historical measurements from September 2, 2026, not new runs or V4.1 results.

Provenance: the prior `LLMs_frontier_serving` repository's `deepseek_v4_flash/results/mtp-off-image` arm. The methodology and corrections below are condensed from its `bench.sh`, September 6 root `report.md`, and `fix_bug.md`. Other models, experimental arms, and scripts with external filesystem dependencies are outside this reference bundle.

## Historical setup

- Eight NVIDIA H100 80GB HBM3 GPUs; tensor parallelism 8 and expert parallelism enabled.
- Driver 575.57.08; PyTorch `2.13.0+cu130`; vLLM `0.1.dev20051+g487ecf187`.
- Model `deepseek-ai/DeepSeek-V4-Flash`; recorded precision: MXFP4 experts, FP8 attention/dense weights, FP8 KV cache. An immutable checkpoint revision was not recorded in the manifest.
- MTP off; prefix caching enabled; memory utilization 0.82; maximum model length 262,144; maximum sequences 256.
- Text-only chat-completions requests. Synthetic workload outputs were forced to 256 tokens using `--ignore-eos`. Actual input counts include tokenizer/template effects.

The manifest preserves the original command and resolved cache settings. Requested `--block-size 256` appears alongside a resolved cache metric of `4`; retain both rather than assuming they represent the same cache grouping.

Historical launch configuration, transcribed without the old environment-specific wrapper:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --port 8002 --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-num-seqs 256 --max-cudagraph-capture-size 256 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --gpu-memory-utilization 0.82 --max-model-len 262144 \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --no-enable-flashinfer-autotune --enable-mfu-metrics
```

This records the old runtime's flags; validate compatibility before using it with a new runtime. New V4/V4.1 runs need explicit, matched reasoning settings and model revisions.

## Historical benchmark method

All workloads used `vllm bench serve`, backend `openai-chat`, endpoint `/v1/chat/completions`, unlimited offered request rate, and a client concurrency cap. These were finite request batches, not production arrival traces.

| Axis | Input / output token targets | Concurrency | Requests |
| --- | --- | --- | --- |
| Batch | 16,384 / 256 | 1, 4, 16, 64 | 8, 8, 32, 128 respectively |
| Context | 16,384; 65,536; 131,072; 260,000 / 256 | 8 | 16 per point |
| Prefix | 65,536 shared + 2,048 suffix / 256; 1, 4, 16 distinct prefixes | 8 | 64 per point |

Batch/context used `--dataset-name random`, `--random-input-len`, `--random-output-len`, `--num-prompts`, `--max-concurrency`, and `--ignore-eos`. Seed = `(input_length % 100000) + concurrency * 7919 + output_length * 31`. The original request-count rule was `min(max(8, 2 * concurrency), max(8, floor(8000000 / input_length)))`.

Prefix used `--dataset-name prefix_repetition`, `--prefix-repetition-prefix-len 65536`, `--prefix-repetition-suffix-len 2048`, `--prefix-repetition-num-prefixes N`, `--prefix-repetition-output-len 256`, and seed `4200 + N`.

Metrics were saved with `--percentile-metrics ttft,tpot,itl,e2el --save-result`. Historical JSONs contain means, medians, and p99, not the new plan's p95 or repeat uncertainty. For future runs, warm representative kernel shapes with disjoint prompts and reset or otherwise isolate prefix state before measurement; warmup must not reuse measured prompts. Batch/context shapes can share a seed when their lengths and concurrency match, so unique labels alone do not isolate cache state.

## Retained results

Values below are rounded from the included JSONs. TTFT and TPOT are request medians in milliseconds.

| Workload | Output tokens/s | TTFT | TPOT |
| --- | ---: | ---: | ---: |
| Batch 16K, c1 | 85.0 | 716.8 | 7.9 |
| Batch 16K, c4 | 219.8 | 1,906.4 | 10.7 |
| Batch 16K, c16 | 330.9 | 3,945.8 | 32.4 |
| Batch 16K, c64 | 389.4 | 3,642.9 | 148.5 |
| Context 16,384, c8 | 104.5 | 3,118.7 | 23.3 |
| Context 65,536, c8 | 47.9 | 12,310.8 | 125.2 |
| Context 131,072, c8 | 34.7 | 26,664.8 | 144.2 |
| Context 260,000, c8 | 16.7 | 61,821.8 | 242.3 |
| Prefix, 1 distinct prefix | 198.2 | 1,079.3 | 12.3 |
| Prefix, 4 distinct prefixes | 391.6 | 1,123.5 | 11.6 |
| Prefix, 16 distinct prefixes | 114.6 | 8,638.1 | 29.9 |

## Corrections and requirements for new measurements

- Fresh matched baselines are required. Most historical points have one retained run and small samples. Different builds, startup sessions, utilization settings, or nodes must remain separate arms. The finer-grid and build-bridge arms are not part of this bundle.
- The prior report flags the 16K/context c8 result as unresolved: 104.5 tokens/s here versus 281.1 in a separate build/utilization/session. Rerun batch and context in the same controlled session before explaining the discrepancy. A batch-only runtime bridge does not validate context behavior.
- Prefix results compare sharing patterns, not caching on versus off. Four prefixes performed best here; fewer prefixes did not monotonically improve performance. New experiments need cache-off, cold-fill, and prewarmed controls.
- Historical MTP comparisons used a different, older runtime pair. This bundle supplies no matched MTP baseline; measure off/on anew for both versions.
- `output_throughput = total_output_tokens / duration` includes input processing and scheduling. TTFT includes queueing and prefill; TPOT is a per-request average, distinct from individual inter-token latency. Client concurrency is not instantaneous engine batch size.
- `peak_kv_cache_usage_perc` is a fraction: `0.725` means 72.5% of the engine cache pool, not total GPU memory. High occupancy alone does not prove preemption or a memory bottleneck. Prefix JSONs lack this telemetry.
- The old harness could turn missing telemetry into zero, warn without rejecting cache-contaminated points, accept partial completions, and overwrite manifests while skipping existing results. Saved `cold_run=true` is not independent proof of successful telemetry. Future runs must retain raw scrapes, validate identity and all completions, preserve unknown values, and use immutable run directories.
- Large latency tails require investigation; they do not prove a warmup or queueing cause. Keep slow repeats and failures under the same predeclared acceptance rules instead of selecting the fastest run.
- Actual input plus output tokens must fit the model limit with template headroom. The 260,000 input target was successful on this historical setup; revalidate the boundary for both models.
- Engine FLOP/byte estimates omit substantial V4 operations and are not hardware counters. Use operator traces and hardware counters before asserting a kernel or bandwidth bottleneck.

Efficiency calculations: `GPU-seconds/output token = GPU_count / output_tokens_per_second`; `cost/million output tokens = GPU_count * price_per_GPU_hour * 1000000 / (3600 * output_tokens_per_second)`. State the price assumption and include the measured input workload. These measurements do not establish equal answer quality or cost per successful task.
