# Per-model setup guide (8×H100, vLLM)

One section per model: checkpoint selection, launch configuration, observed logs, requests and lessons from 2026-10-07. The historical observations used vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`) in `/root/vllm-latest`. On another build, re-check the log lines: backends and even the executed layers can change.

**Qwen update, 2026-10-08:** future work uses only the original **BF16** `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Follow the [checkpoint policy](qwen-checkpoint-policy.md). The existing `qwen-38` key selects FP8 and is retained only as history; do not use it for new launches. BF16 harness wiring and runtime validation are pending. Historical FP8 timings, memory and kernel observations below are not BF16 results.

Related: [fresh-node quick setup](reproduce.md) section 0, the [serving skill](../.claude/skills/serving/SKILL.md), the [lessons](../teach_me/README.md), and the measured [architecture table](../reports/blog-architecture-h100-v1/architecture.md).

## Common to all five

```bash
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
```

| Setting | Value | Flag in `bench/serve.sh` |
| --- | --- | --- |
| GPUs | 8, tensor parallel 8 + expert parallel | `--tensor-parallel-size 8 --enable-expert-parallel` |
| Memory | 0.90 of each GPU | `--gpu-memory-utilization 0.90` |
| Context / sequences | 262,144 / 64 | `--max-model-len 262144 --max-num-seqs 64` |
| Prefill chunk | 8,192 tokens | `--max-num-batched-tokens 8192` (pass it as an extra argument) |
| Prefix caching | off | `--no-enable-prefix-caching` (configs `off`, `off-profidle`) |
| Modality | text only | `--language-model-only` |
| Requests | temperature 0, thinking off where possible | `chat_template_kwargs`, see each model |

Launch, check, stop: the commands below apply to supported keys, **excluding the archived Qwen FP8 key**. They do not yet provide a BF16 Qwen launch.

```bash
# launch (KEY is the model key below; use off-profidle + PROFILE_DIR for traces)
PROFILE_DIR=$PWD/results/demo/$KEY/profiles \
  bash bench/serve.sh $KEY off /tmp/serve-$KEY.log --max-num-batched-tokens 8192

# ready when the model id is listed
curl -s localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1

# stop and prove it
PG=$(ps -o pgid= -p $(pgrep -f "bin/[v]llm serv[e]" | head -1) | tr -d ' ')
kill -INT -- -$PG; sleep 20; kill -KILL -- -$PG 2>/dev/null
pgrep -fc "bin/[v]llm serv[e]"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

For measurements use the chain instead; it does all three and cleans up on every exit path:

```bash
bash bench/blog_launch.sh my-run diag:$KEY timing:$KEY:1
```

**Two things that apply to every model's first launch on a fresh node**

- It is slow: weights are read from disk once and kernels are JIT-compiled once. Later launches reuse the page cache and `/tmp` JIT caches.
- It can die with `CUDA error: invalid argument` right after a `flashinfer.jit: Building JIT module …` line, because eight ranks build the same module at once. **Relaunch once**; the module is then cached. The chain does this automatically. It happened to 0731 and GLM, not to the other three.

## Summary table

| Key | Checkpoint | Revision | Size on disk | First launch | Relaunch | Thinking off | Template adds |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `v4-0731` | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | 156 GB | 11 min | 2 min | `{"thinking": false}` | 4 tokens |
| `v41` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | 476 GB | 6 min | 2.5 min | `{"thinking": false}` | 4 tokens |
| `mimo-v26` | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | `2479e2d0029eca9a34cc7e7f55a121925f81908e` | 166 GB | 7.5 min | 2 min | `{"enable_thinking": false}` | 9 tokens |
| `qwen-38` (historical only) | `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` | 173 GB | 19 min | 4 min | `{"enable_thinking": false}` | 12 tokens |
| `qwen-38-bf16` (proposed; not wired) | `Qwen/Qwen3.8-Flash-Next` (original BF16) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Not measured | Not measured | Not measured | Verify `{"enable_thinking": false}` | Pending |
| `glm-53` | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | 306 GB | 8.5 min | 3.5 min | **not available**: `{"reasoning_effort": "low"}` | 12 tokens |

"Template adds" is the server-counted prompt tokens minus the tokens of the user text, measured on one 1K request. The five measured repositories were public; no token was needed. The BF16 Qwen row records a future target, not a sixth measured deployment.

---

## DeepSeek V4 Flash 0731 (`v4-0731`)

**Download**

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 --max-workers 32
```

**What `serve.sh` adds for this model:** `--tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4`.

**Expect in the log**

| Line | Meaning |
| --- | --- |
| `Resolved architecture: DeepseekV4ForCausalLM` | Right model class |
| `Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM` and `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | FP8 dense layers on the DeepGEMM path |
| `Using 'HUMMING' Mxfp4 MoE backend` | 4-bit experts on `HUMMING` |
| `Using DeepSeek's fp8_ds_mla KV cache format`, `Using FP8 indexer cache for Lightning Indexer` | FP8 KV and indexer state |
| `Setting kv cache block size to 256` | 256-token KV blocks: short requests waste part of a block |
| `Model loading took 19.79 GiB`, `Available KV cache memory: 47.43 GiB`, `GPU KV cache size: 1,728,531 tokens` | Weights and pool per GPU |

**Request**

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"thinking": false}}'
```

**Notes**

- Longest first launch of the first three: a "DeepGEMM warmup" over 1,261 kernels takes about 4 minutes the first time.
- First launch on this build crashed once in the JIT build; the relaunch worked.
- The first timing block at a new shape can be slow (1K with eight clients: 552 tok/s, then 622 twice). Keep it; report the median.
- The token capacity in the log understates what fits at long context: a 64K request used 0.56% of the pool.

## DeepSeek V4.1 Flash (`v41`)

**Download** (the largest: 476 GB, about 4 minutes on this node)

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash --revision dba1be0a40aa45a94ad051997016db3960a90277 --max-workers 32
```

**What `serve.sh` adds:** `--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41`.

**Expect in the log**

| Line | Meaning |
| --- | --- |
| `Resolved architecture: DeepseekV41ForCausalLM` | Right model class |
| `Decoder SWA bounded replay: in eager prefill steps, layers 21-39 run on each request's last 128 tokens only.` | **The prefill skip is on.** If this line is missing, prefill runs all 40 layers on every token and will be much slower |
| `Resolved Engram configuration: EngramConfig(cpu_offload=True, …)` and `Engram table offloaded to pinned host memory: … 11.80 GiB per rank` (twice) | Engram tables live in host RAM: about 190 GB of pinned memory for 8 ranks |
| `Using 'HUMMING' Mxfp4 MoE backend`, `DeepSeek V4 expert_dtype resolved to 'fp4'` | 4-bit experts |
| `Using DeepSeek's fp8_ds_mla KV cache format` | FP8 KV (the model card's smallest KV figure assumes FP4) |
| `Setting kv cache block size to 64` | 64-token KV blocks |
| `Model loading took 36.32 GiB`, `Available KV cache memory: 29.14 GiB`, `GPU KV cache size: 9,092,347 tokens` | Largest weights, smallest pool of the three targets |

**Request:** as for 0731 with `"model": "deepseek-ai/DeepSeek-V4.1-Flash"` and `{"thinking": false}`.

**Notes**

- Needs the most host RAM (weights in page cache plus Engram tables). The node had 1.7 TB; check `free -g` on a smaller one.
- The prefill skip does not apply to steps under 768 tokens or to requests asking for prompt logprobs. See the [prefill check](../reports/blog-architecture-h100-v1/v41_prefill_check.md).
- Its decode step is split over several GPU streams. When reading traces, cut steps at the CPU-side annotation ([lesson 3](../teach_me/03_traces_and_components.md)).
- The harmless warnings `Attempted to load weight image_newline …` come from the unused vision part.

## MiMo-V2.6-Flash-MOPD (`mimo-v26`)

**Download**

```bash
hf download XiaomiMiMo/MiMo-V2.6-Flash-MOPD --revision 2479e2d0029eca9a34cc7e7f55a121925f81908e --max-workers 32
```

**What `serve.sh` adds:** `--tokenizer-mode auto --reasoning-parser mimo --tool-call-parser mimo --trust-remote-code --code-revision <same revision>`. The config class comes from the repository's own code, so the code revision is pinned to the weights' revision.

**Expect in the log**

| Line | Meaning |
| --- | --- |
| `Resolved architecture: MiMoV2OmniForCausalLM` | Expected: the config has a vision tower, and the wrapper runs the same text decoder. Multimodal inputs are disabled by `--language-model-only` |
| `Using FLASH_ATTN_DIFFKV for attention`, `Using FlashAttention version 3`, `Diff-KV with sinks: upgrading FlashAttention 3 -> 4` | Attention backend |
| `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | FP8 dense layers on DeepGEMM |
| `Using 'HUMMING' Mxfp4 MoE backend` | MXFP4 experts on `HUMMING` (the old build used Marlin) |
| `Add 6 padding layers, may waste at most 15.38% KV cache memory` | Layout padding of the hybrid KV manager |
| `Model loading took 20.1 GiB`, `Available KV cache memory: 49.18 GiB`, `GPU KV cache size: 6,969,972 tokens` | Weights and pool per GPU |

**Request**

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
```

**Notes**

- The switch is `enable_thinking`, not `thinking`. With the wrong name the request still succeeds and the model thinks.
- With forced-length runs (`ignore_eos`) its streams can stop yielding text after the natural end; the server still generates tokens. Count tokens from usage.
- Its tokenizer gives 1–2% more tokens than DeepSeek's for the same text.
- The model card recommends TP 4 at 0.95 utilization; we use the common TP 8 at 0.90 so the five models are comparable.

## Qwen3.8-Flash-Next: original BF16 for future work

Use `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, for all new Qwen work. Pin its tokenizer and any remote code to that revision. Do not use the `-FP8` repository or quantize the original weights to FP8. `--dtype bfloat16` on an FP8 checkpoint does not select original BF16 weights.

Before an authorized GPU session, prepare a distinct BF16 key and study manifest as described in the [checkpoint policy](qwen-checkpoint-policy.md). The current `qwen-38` key and blog chain remain tied to the completed FP8 study. BF16 support, fit, loaded kernels, KV/state formats, template behavior and performance must be validated separately. This decision does not add Qwen to the later speculative study.

### Archived FP8 setup (`qwen-38`): observed, not for new launches

The completed blog study loaded `Qwen/Qwen3.8-Flash-Next-FP8` at `236dfdf285828023ca3bcd3f37366c58a3469b13`. The following records describe that FP8 deployment only.

**What `serve.sh` added:** `--tokenizer-mode auto --reasoning-parser qwen3 --tool-call-parser qwen3_xml`.

**Observed FP8 log**

| Line | Meaning |
| --- | --- |
| `Resolved architecture: Qwen4ExpForConditionalGeneration` | Right model class |
| `Using FLASHINFER_CUTLASS Fp8 MoE backend` | FP8 experts (not the 4-bit path of the first three) |
| `Initialized PLE embedding … ngram_embedding` | The n-gram embedding module loaded |
| `Setting attention block size to 400 tokens to ensure that attention page size is >= mamba page size` | Recurrent-layer state and attention KV share one page size |
| `kv cache group sizes [262144, 262144, 262144, 262144, 4, 400]` | Recurrent state groups plus one attention group |
| `Model loading took 17.36 GiB`, `Available KV cache memory: 51.44 GiB`, `GPU KV cache size: 4,194,304 tokens` | Smallest weights, largest pool |

**Historical request:** `"model": "Qwen/Qwen3.8-Flash-Next-FP8"` with `{"enable_thinking": false}`.

**Notes**

- **First launch took 19 minutes** (several kernel modules are JIT-built, including `fused_moe_90`). Do not treat a quiet log as a hang before the 30-minute readiness limit; `No available shared memory broadcast block found in 60 seconds` repeats while it compiles.
- Its tokenizer gives about 5% more tokens than DeepSeek's for the same text (17,199 against 16,351 at "16K"). Say so when comparing throughput.
- It is faster than the models the request counts were sized on, so some timing windows were under 60 seconds. Re-size counts from a pilot that includes it.
- Old launches used `--moe-backend triton` and other workarounds on older builds; none was needed here.
- It holds the most KV per live token (about 13 KiB per GPU at 64K). Watch `vllm:kv_cache_usage_perc` under many long requests.

## GLM-5.3-Flash (`glm-53`)

Blog study only. Not part of the later speculative study.

**Download**

```bash
hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --max-workers 32
```

**What `serve.sh` adds:** `--tokenizer-mode auto --reasoning-parser glm45 --tool-call-parser glm47` (two different parser names; they follow the historical launches).

**Expect in the log**

| Line | Meaning |
| --- | --- |
| `Resolved architecture: Glm5NextForConditionalGeneration` | Right model class |
| `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, `Using FLASHINFER_CUTLASS Fp8 MoE backend` | FP8 dense layers and FP8 experts |
| `Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER/KPOOL_TAIL/FLASHINFER_MLA_SPARSE_SM90` | Sparse-attention layers with an indexer |
| `Setting attention block size to 640 tokens …` and `Padding mamba page size by 20.75%` | Recurrent state and attention share a page size; a fifth of each state page is padding |
| `Model loading took 38.8 GiB`, `Available KV cache memory: 28.73 GiB`, `GPU KV cache size: 2,618,281 tokens` | Largest weights, smallest pool of the five |

**Request**

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "zai-org/GLM-5.3-Flash",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

**Notes**

- **Thinking cannot be switched off.** The chat template always opens the assistant turn with `<think>`; the only control is `reasoning_effort` (`low`, `high`, default `max`). At `low`, one of three test prompts still returned reasoning text. The answer is in `content`, the reasoning in `reasoning_content`. Give it enough `max_tokens` for both.
- For forced-length timing this changes what the 256 tokens contain, not how many decode steps run. State it as a deviation whenever GLM is compared with models that run with thinking off.
- The first launch died once in the JIT build (same error as 0731); the relaunch worked.
- Smallest KV pool with about 12 KiB per live token per GPU: eight 64K requests used a fifth of the pool. It will run out of KV first under many long requests.
- The first timing block at one client was slower (16K: 101 tok/s, then 113 twice). Keep it; report the median.
- Old launches used `--block-size 128` and a CUDA-graph capture limit on older builds; neither was needed here.

---

## After any launch: the five-line health check

```bash
L=/tmp/serve-$KEY.log
grep -c "CUDA error\|Traceback" $L                                   # expect 0
grep -m1 "non-default args" $L | grep -o "'enable_prefix_caching': [A-Za-z]*"   # expect False
grep -m1 "Chunked prefill is enabled" $L                             # expect max_num_batched_tokens=8192
grep -m1 "GPU KV cache size" $L                                      # compare with the tables above
curl -s localhost:8000/metrics | grep -E "^vllm:(num_requests_running|kv_cache_usage_perc)"   # expect 0 and 0 when idle
```

The completed study used the following functional check (rendered request, three natural-EOS prompts, one forced 256-token request). Do not reuse it with `qwen-38` for a new Qwen run: it selects FP8 and writes to the historical study. BF16 needs its own prepared key and study path.

```bash
bash bench/blog_launch.sh check-$KEY diag:$KEY --diag-parts functional   # drop the flag to also run pilot, KV snapshots and traces
cat results/blog-architecture-h100-v1/$KEY/_functional/functional.json
```

`"ok": true` means natural answers stopped by themselves with no reasoning text and the forced request returned exactly 256 tokens. GLM reports `false` by design; read its three answers instead.
