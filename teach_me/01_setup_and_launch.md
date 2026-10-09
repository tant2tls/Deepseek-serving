# Lesson 1: set up the node and launch a server

## Step 1: look before you install

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
df -h /workspace; ls /workspace/hf/hub 2>/dev/null; ls -d /root/vllm /root/vllm-latest 2>/dev/null
```

A rented node usually starts empty. On 2026-10-07 the vLLM venv existed but no weights did.

## Step 2: runtime

```bash
bash install.sh                       # creates /root/vllm (pinned, old studies) and /root/vllm-latest (blog study)
export VLLM_VENV=/root/vllm-latest    # which runtime the bench scripts use
$VLLM_VENV/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
```

Two venvs exist so the old studies stay reproducible. `bench/serve.sh` and `bench/run_matrix.py` read `VLLM_VENV`.

## Step 3: weights

```bash
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 --max-workers 32
hf download XiaomiMiMo/MiMo-V2.6-Flash-MOPD   --revision 2479e2d0029eca9a34cc7e7f55a121925f81908e --max-workers 32
hf download deepseek-ai/DeepSeek-V4.1-Flash   --revision dba1be0a40aa45a94ad051997016db3960a90277 --max-workers 32
```

Always pass `--revision`: a model ID alone can change under you. About 800 GB in total; it took 10 minutes on this node. The commands for GLM and Qwen, and the launch command, log lines and known problems of each of the five models, are on its page in [reproduce/](../reproduce/README.md).

## Step 4: launch

```bash
PROFILE_DIR=$PWD/results/demo/profiles \
  bash bench/serve.sh mimo-v26 off /tmp/serve-demo.log --max-num-batched-tokens 8192
```

What `bench/serve.sh` passes to `vllm serve`, and why:

| Flag | Meaning |
| --- | --- |
| `--revision <sha>` | Pin the checkpoint |
| `--tensor-parallel-size 8 --enable-expert-parallel` | Split every layer over 8 GPUs and spread MoE experts across them |
| `--gpu-memory-utilization 0.90` | Fraction of each GPU the engine may take; what is left after weights becomes the KV pool |
| `--max-model-len 262144`, `--max-num-seqs 64` | Longest request and most concurrent sequences |
| `--max-num-batched-tokens 8192` | Tokens the scheduler may process per step, so a long prompt is prefilled in 8,192-token chunks |
| `--no-enable-prefix-caching` | Every request pays its full prefill; nothing is reused |
| `--language-model-only` | Text decoder only |
| `--profiler-config …` (config `off-profidle` only) | Loads the torch profiler, idle until `/start_profile` |

The server prints its resolved settings. Read these lines in the log before measuring:

```bash
grep -E "non-default args|Resolved architecture|Chunked prefill|Selected .* for|Using .* backend|KV cache|Model loading took|bounded replay" /tmp/serve-demo.log
```

They tell you what really runs. For example on this build MiMo logs `Using 'HUMMING' Mxfp4 MoE backend` and V4.1 logs `Decoder SWA bounded replay: … layers 21-39 run on each request's last 128 tokens only`.

## Step 5: ready means the model answers

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool | grep '"id"'
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 64, "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}}'
```

Thinking is switched off with `{"thinking": false}` for DeepSeek and `{"enable_thinking": false}` for MiMo. Check that the answer has no reasoning text and `finish_reason` is `stop`.

## Step 6: stop, and prove it

```bash
PG=$(ps -o pgid= -p $(pgrep -f "bin/[v]llm serv[e]" | head -1) | tr -d ' ')
kill -INT -- -$PG; sleep 20; kill -KILL -- -$PG 2>/dev/null
pgrep -fc "bin/[v]llm serv[e]"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Expect `0` processes and `0 MiB` eight times. The chain (`bench/blog_study.py`) does all of steps 4–6 for you and logs the shutdown in `results/<study>/_logs/shutdown.jsonl`.

## Check yourself

- Which log line proves prefix caching is off? (`'enable_prefix_caching': False` in `non-default args`.)
- Why does the first launch of a model take 7–9 minutes and later ones less? (Weights are read from disk once, then served from the page cache; JIT kernels are compiled once.)
- What do you do if the first launch dies with `CUDA error: invalid argument` right after a `flashinfer.jit: Building JIT module` line? (Relaunch once; the module is now cached.)
