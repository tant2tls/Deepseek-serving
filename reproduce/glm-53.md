# GLM-5.3-Flash (`glm-53`)

| | |
| --- | --- |
| Checkpoint | `zai-org/GLM-5.3-Flash` (native FP8 weights) |
| Immutable revision | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |
| Measured | First node, study `blog-architecture-h100-v1`, after the first three models |
| Size on disk | 306 GB |
| First launch / relaunch | 8.5 min / 3.5 min |
| Thinking off | **Not available.** Runs with `{"reasoning_effort": "low"}` |
| Template adds | 12 tokens |
| Architecture and technical report | [docs/models.md](../docs/models.md#glm-53-flash-glm-53) |
| Results | [five-model tables](../reports/five-model/results.md), [raw evidence](../reports/blog-architecture-h100-v1/data/glm-53/), [September archive](../reports/glm-53-september/README.md) (older build, never a control) |

Read the [common setup and known problems](README.md#common-setup) first. They apply to every model. GLM belongs to the blog study only; it is not part of the later speculative study.

## 1. Download

```bash
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --max-workers 32
```

## 2. Launch

```bash
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf
bash bench/serve.sh glm-53 off /tmp/serve-glm-53.log --max-num-batched-tokens 8192
```

For this model `bench/serve.sh` adds `--tokenizer-mode auto --reasoning-parser glm45 --tool-call-parser glm47` (two different parser names; they follow the saved October launch logs). The full command it runs, after setting the [common environment](README.md#common-setup):

```bash
vllm serve zai-org/GLM-5.3-Flash \
  --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --language-model-only \
  --tokenizer-mode auto --reasoning-parser glm45 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 8192
```

## 3. Check the log

| Line | Meaning |
| --- | --- |
| `Resolved architecture: Glm5NextForConditionalGeneration` | Right model class |
| `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, `Using FLASHINFER_CUTLASS Fp8 MoE backend` | FP8 dense layers and FP8 experts |
| `Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER/KPOOL_TAIL/FLASHINFER_MLA_SPARSE_SM90` | Sparse-attention layers with an indexer |
| `Setting attention block size to 640 tokens …` and `Padding mamba page size by 20.75%` | Recurrent state and attention share a page size; a fifth of each state page is padding |
| `Model loading took 38.8 GiB`, `Available KV cache memory: 28.73 GiB`, `GPU KV cache size: 2,618,281 tokens` | Largest weights, smallest pool of the five |

Then run the [health check](README.md#health-check-after-any-launch).

## 4. Send a request

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "zai-org/GLM-5.3-Flash",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

The answer is in `content` and the reasoning in `reasoning_content`. Give it enough `max_tokens` for both.

## 5. Measure

Stop the hand-launched server first. The chain starts its own. It needs the rebuilt inputs and a frozen plan from the [procedure](README.md#3-rebuild-public-inputs).

```bash
export BLOG_STUDY=blog-architecture-h100-v1
bash bench/blog_launch.sh diag-glm-53 diag:glm-53          # functional check, pilot, live KV, traces
# wait for the chain to exit, then:
bash bench/blog_launch.sh timing-glm-53 timing:glm-53:1    # six unprofiled points of block 1
tail -f results/$BLOG_STUDY/_logs/chain.log
```

The functional check reports `"ok": false` for GLM **by design**: its natural-answer thinking-off check cannot pass. Read its three answers and the forced-256 result instead.

## 6. Stop

See [stop the server and prove it](README.md#stop-the-server-and-prove-it).

## Known problems and notes

- **Thinking cannot be switched off.** The chat template always opens the assistant turn with `<think>`; the only control is `reasoning_effort` (`low`, `high`, default `max`). At `low`, one of three test prompts still returned reasoning text.
- **State this as a deviation whenever GLM is compared with models that run with thinking off.** For forced-length timing it changes what the 256 tokens contain, not how many decode steps run. It prevents an equal-answer-quality reading.
- **The first launch died once in the JIT build** (same error as 0731); the relaunch worked.
- **Smallest KV pool, with about 12 KiB per live token per GPU:** eight 64K requests used a fifth of the pool. Maximum concurrency was not measured.
- **The first timing block at one client was slower** (16K: 101 tok/s, then 113 twice). Keep it; report the median.
- **Old launch flags are not needed.** September launches used `--block-size 128` and a CUDA-graph capture limit on older builds; neither was needed here. The [September archive](../reports/glm-53-september/README.md) used another build, utilization 0.82 and no recorded revision, so none of its launchers is a setup recommendation.
