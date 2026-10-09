# MiMo-V2.6-Flash-MOPD (`mimo-v26`)

| | |
| --- | --- |
| Checkpoint | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` |
| Immutable revision | `2479e2d0029eca9a34cc7e7f55a121925f81908e` (weights, tokenizer and remote code) |
| Measured | First node, study `blog-architecture-h100-v1`; node control on the second node, study `qwen-bf16-h100-v1` |
| Size on disk | 166 GB |
| GPU memory | Weights 20.1 GiB per GPU (161 GiB on eight); KV pool reserved 49.18 GiB per GPU; 73.7 GiB in use per GPU |
| Host RAM | Page cache only; nothing pinned was recorded |
| Time on the measured node | Diagnostics step 18.4 min; one timing block 14.4–14.7 min; three blocks 44 min. As the control on the second node: one block 30.2 min, trace launch 13.8 min ([budget](README.md#what-a-reproduction-needs-disk-memory-and-time)) |
| First launch / relaunch | 7.5 min / 2 min |
| Thinking off | `{"enable_thinking": false}` |
| Template adds | 9 tokens |
| Architecture and technical report | [docs/models.md](../docs/models.md#mimo-v26-flash-mopd-mimo-v26), [source walkthrough](../docs/mimo-v2.6-inference.md) |
| Results | [five-model tables](../reports/five-model/results.md), [first-node evidence](../reports/blog-architecture-h100-v1/data/mimo-v26/), [second-node control](../reports/qwen-bf16-h100-v1/data/mimo-v26/) |

Read the [common setup and known problems](README.md#common-setup) first. They apply to every model.

## 1. Download

```bash
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
hf download XiaomiMiMo/MiMo-V2.6-Flash-MOPD --revision 2479e2d0029eca9a34cc7e7f55a121925f81908e --max-workers 32
```

## 2. Launch

```bash
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf
bash bench/serve.sh mimo-v26 off /tmp/serve-mimo-v26.log --max-num-batched-tokens 8192
```

For this model `bench/serve.sh` adds `--tokenizer-mode auto --reasoning-parser mimo --tool-call-parser mimo --trust-remote-code --code-revision <same revision>`. The config class comes from the repository's own code, so the code revision is pinned to the weights' revision. The full command it runs, after setting the [common environment](README.md#common-setup):

```bash
vllm serve XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
  --revision 2479e2d0029eca9a34cc7e7f55a121925f81908e \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --language-model-only \
  --tokenizer-mode auto --reasoning-parser mimo \
  --enable-auto-tool-choice --tool-call-parser mimo \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --trust-remote-code --code-revision 2479e2d0029eca9a34cc7e7f55a121925f81908e \
  --max-num-batched-tokens 8192
```

## 3. Check the log

| Line | Meaning |
| --- | --- |
| `Resolved architecture: MiMoV2OmniForCausalLM` | Expected: the config has a vision tower, and the wrapper runs the same text decoder. Multimodal inputs are disabled by `--language-model-only` |
| `Using FLASH_ATTN_DIFFKV for attention`, `Using FlashAttention version 3`, `Diff-KV with sinks: upgrading FlashAttention 3 -> 4` | Attention backend |
| `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel` | FP8 dense layers on DeepGEMM |
| `Using 'HUMMING' Mxfp4 MoE backend` | MXFP4 experts on `HUMMING` (the old build used Marlin) |
| `Add 6 padding layers, may waste at most 15.38% KV cache memory` | Layout padding of the hybrid KV manager |
| `Model loading took 20.1 GiB`, `Available KV cache memory: 49.18 GiB`, `GPU KV cache size: 6,969,972 tokens` | Weights and pool per GPU |

Then run the [health check](README.md#health-check-after-any-launch).

## 4. Send a request

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
```

The answer should contain no reasoning text and end with `finish_reason` `stop`.

## 5. Measure

Stop the hand-launched server first. The chain starts its own. It needs the rebuilt inputs and a frozen plan from the [procedure](README.md#3-rebuild-public-inputs).

```bash
export BLOG_STUDY=blog-architecture-h100-v1
bash bench/blog_launch.sh diag-mimo-v26 diag:mimo-v26          # functional check, pilot, live KV, traces
# wait for the chain to exit, then:
bash bench/blog_launch.sh timing-mimo-v26 timing:mimo-v26:1    # six unprofiled points of block 1
tail -f results/$BLOG_STUDY/_logs/chain.log
```

**As a node control.** When another model is measured on a different node, MiMo runs one timing block and one trace launch there with the same requests and counts. The commands are in the [Qwen profile of the procedure](README.md#4-diagnostics-plan-and-timing). Do not drop the control when placing new-node evidence beside the first node.

## 6. Stop

See [stop the server and prove it](README.md#stop-the-server-and-prove-it).

## Known problems and notes

- **The switch is `enable_thinking`, not `thinking`.** With the wrong name the request still succeeds and the model thinks.
- **With forced-length runs (`ignore_eos`) its streams can stop yielding text after the natural end**; the server still generates tokens. Count tokens from usage.
- **Its tokenizer gives 1–2% more tokens than DeepSeek's for the same text.** Use the saved actual counts.
- **The model card recommends TP 4 at 0.95 utilization**; we use the common TP 8 at 0.90 so the five models are comparable.
- **Throughput reproduced across the two nodes, latency did not.** Second-node over first-node throughput was 0.983–0.995; TTFT medians and eight-client TPOT medians moved. See the [node control](../reports/five-model/results.md#mimo-node-control).
- **The expert backend changed between builds.** The September build selected Marlin, the October build selects `HUMMING`. Read the backend from the log before comparing expert times across builds.
