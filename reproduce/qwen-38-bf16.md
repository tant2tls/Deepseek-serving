# Qwen3.8-Flash-Next, original BF16 (`qwen-38-bf16`)

| | |
| --- | --- |
| Checkpoint | `Qwen/Qwen3.8-Flash-Next` (the original BF16 weights) |
| Immutable revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` (weights, tokenizer and code) |
| Measured | **Second node**, 2026-10-08, study `qwen-bf16-h100-v1`, with MiMo as node control |
| Size on disk | 336 GiB (360 GB by the hub's count, 131 shards) |
| GPU memory | Weights 31.42 GiB per GPU (251 GiB on eight); KV pool reserved 37.37 GiB per GPU; 74.4 GiB in use per GPU |
| Host RAM | The n-gram embedding is in pinned host memory (size not recorded), besides the page cache |
| Time on the measured node | Download 23 min; diagnostics step 24.9 min; one timing block 21.3–21.5 min; three blocks 64 min ([budget](README.md#what-a-reproduction-needs-disk-memory-and-time)) |
| First launch / relaunch | 9.7 min / 4.3 min |
| Thinking off | `{"enable_thinking": false}` |
| Template adds | 12 tokens |
| Architecture and technical report | [docs/models.md](../docs/models.md#qwen38-flash-next-qwen-38-bf16) |
| Results | [five-model tables](../reports/five-model/results.md), [second-node findings](../reports/qwen-bf16-h100-v1/findings.md), [raw evidence](../reports/qwen-bf16-h100-v1/data/qwen-38-bf16/) |

Read the [common setup and known problems](README.md#common-setup) first. They apply to every model.

## Checkpoint policy

Use **Qwen3.8-Flash-Next**, the original BF16 `Qwen/Qwen3.8-Flash-Next` checkpoint, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, key `qwen-38-bf16`, for all Qwen work. Keep the original weights and native precision; do not convert or substitute weights, and do not add another Qwen deployment to this branch. Qwen is not part of the later speculative study.

Its completed study is `qwen-bf16-h100-v1`, measured on October 8, 2026, on the second 8×H100 node. The server log verifies `dtype=torch.bfloat16`, `quantization=None` and `TRITON Unquantized MoE backend`. Because the node differs, compare its **throughput** with the other four and keep its **latency** separate: see the [comparison limits](../docs/experiments.md#two-physical-nodes).

## 1. Download

```bash
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
hf download Qwen/Qwen3.8-Flash-Next --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --max-workers 32
```

On the 2026-10-08 node this took 23 minutes at about 240 MiB/s. The node's network was the limit: logging in with a token and running a second downloader did not raise the rate. Start the download first and do every other preparation while it runs.

## 2. Launch

```bash
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf
bash bench/serve.sh qwen-38-bf16 off /tmp/serve-qwen-38-bf16.log --max-num-batched-tokens 8192
```

For this model `bench/serve.sh` adds `--tokenizer-mode auto --reasoning-parser qwen3 --tool-call-parser qwen3_xml`. There is no `--dtype` and no quantization flag: precision comes from the checkpoint. The full command it runs, after setting the [common environment](README.md#common-setup):

```bash
vllm serve Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --language-model-only \
  --tokenizer-mode auto --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 8192
```

## 3. Check the log

These lines prove that the BF16 checkpoint is what runs (vLLM `554340f3…`):

| Line | Meaning |
| --- | --- |
| `Resolved architecture: Qwen4ExpForConditionalGeneration` | Right model class |
| `model='Qwen/Qwen3.8-Flash-Next' … revision=de4b8e4d… … dtype=torch.bfloat16 … quantization=None` | Original repository, native precision, no weight quantization |
| `Using TRITON Unquantized MoE backend out of potential backends: ['TRITON', 'BATCHED_TRITON', 'FlashInfer TRTLLM', 'FlashInfer CUTLASS']`, then `Using TritonExperts MoE backend` | Unquantized experts in Triton's `fused_moe_kernel` |
| `Initialized PLE embedding … quantization_method=Qwen4ExpPLEUnquantizedEmbeddingMethod, weight_dtype=torch.bfloat16, weight_device=cpu, pinned=True` | The n-gram embedding is BF16 and lives in pinned host memory, not on the GPUs |
| `Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128)` | Recurrent layers; JIT-built on the first launch |
| `Setting attention block size to 400 tokens …`, `kv cache group sizes [262144, 262144, 262144, 262144, 4, 400]` | Recurrent state groups plus one attention group |
| `Model loading took 31.42 GiB`, `Available KV cache memory: 37.37 GiB`, `GPU KV cache size: 3,046,184 tokens` | Weights and reserved KV pool per GPU |

A quick check after any launch:

```bash
L=/tmp/serve-qwen-38-bf16.log    # chain launches: $(ls results/$BLOG_STUDY/qwen-38-bf16/_server/serve-*.log | tail -1)
grep -m1 -o "dtype=torch.[a-z0-9]*, max_seq_len\|quantization=[A-Za-z0-9]*" $L   # expect bfloat16 and None
grep -m1 "MoE backend" $L                                                       # expect TRITON Unquantized
grep -c -i "fp8" $L                                                             # expect 0
```

A config file that says `bfloat16` is source evidence. These log lines are execution evidence. Only the second kind lets you write "BF16 was measured". Then run the [health check](README.md#health-check-after-any-launch).

## 4. Send a request

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
```

The functional check passed on the measured node: three natural answers stopped by themselves with no reasoning text, and the forced request returned exactly 256 tokens.

## 5. Measure

Stop the hand-launched server first. The chain starts its own. It needs the rebuilt inputs and a frozen plan from the [procedure](README.md#3-rebuild-public-inputs).

```bash
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf BLOG_STUDY=qwen-bf16-h100-v1
bash bench/blog_launch.sh chain-diag diag:qwen-38-bf16          # functional, pilot, live KV, traces
# wait for the chain to exit, then:
bash bench/blog_launch.sh chain-timing timing:qwen-38-bf16:1    # six unprofiled points of one block
tail -f results/$BLOG_STUDY/_logs/chain.log
```

The full second-node sequence, including the MiMo timing block and trace control, is in the [procedure](README.md#4-diagnostics-plan-and-timing). [Lesson 6](../teach_me/06_new_checkpoint_new_study.md) explains why this checkpoint has its own key and study.

## 6. Stop

See [stop the server and prove it](README.md#stop-the-server-and-prove-it).

## Known problems and notes

- **First launch 9.7 minutes, relaunch 4.3 minutes.** The first launch reads 360 GB and JIT-builds the recurrent-layer kernel; `No available shared memory broadcast block found in 60 seconds` repeats meanwhile and is not an error.
- **The MoE backend is the build's automatic choice.** `FlashInfer CUTLASS` and `TRTLLM` are listed as possible unquantized backends. Trying one is a separate single-change arm with its own controls, not part of this deployment.
- **Tokenizer:** about 5% more tokens than DeepSeek's tokenizer for the same text; use the saved actual counts.
- **With forced length, text chunks undercount tokens.** In the 64K, B=1 diagnostic capture the client saw 48 text chunks while the engine ran 143 steps: past its natural end the model emits tokens that produce no text. Use usage counts and trace annotations.
- **Profiling costs more here than on the other models.** Its traced 1K, B=1 decode step was about 8.9 ms against 5.9 ms unprofiled, and its all-reduce durations include rank waiting. Do not read them as isolated communication cost.
- **Short-prompt TTFT was bimodal on the second node.** The per-request TTFT list of the 1K, one-client point switched between two levels (about 190 and 120 ms), so a block median depends on where the switch fell. Inspect the per-request list before interpreting it.
