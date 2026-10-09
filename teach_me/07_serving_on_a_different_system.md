# Lesson 7: set up serving on a different system

Lessons 1 to 6 used the nodes we rented. You will have another one. This lesson shows what to look at before you trust a launch on a new system, and what to do before you put its numbers next to ours.

## What we saw: "the same GPUs" was not the same system

Both October nodes were sold as "8×H100 80GB". The recorded inventories ([first](../reports/blog-architecture-h100-v1/study/_env/lscpu.txt), [second](../reports/qwen-bf16-h100-v1/study/_env/lscpu.txt)) show what differed:

| | First node | Second node |
| --- | --- | --- |
| GPUs | 8 × H100 80GB HBM3 | 8 × H100 80GB HBM3 |
| CPU | 2 × Xeon Platinum 8480+, 208 threads | 2 × Xeon Platinum 8592V, 256 threads |
| NUMA nodes | 2 | 4 |
| CPU frequency governor | not recorded | `schedutil`, active |
| Host RAM | 1,771 GiB | 1,511 GiB |
| GPU driver | 580.105.08 | 580.173.02 |
| vLLM / torch | `0.31.1rc1.dev50+g554340f3d` / `2.13.0+cu132` | the same |

The software stack and the GPUs matched. Then one model, MiMo, ran the same requests on both:

| MiMo, second node / first node | Result |
| --- | --- |
| Output throughput, six points | 0.983–0.995 |
| TTFT median | 1.02–1.30 |
| TPOT median, eight clients, 16K and 64K | 0.86–0.88 |

Throughput survived the move. The split between "time to first token" and "time between tokens" did not. The control cannot say which host difference caused it; it only says that latency from the two nodes must not be ranked together.

Two more surprises came from the host, not the GPUs. On the first node `/root` was an encrypted FUSE mount, and the eight ranks' concurrent kernel compiles failed there with `FileNotFoundError`. And one model, V4.1, pins about 190 GB of host RAM for its Engram tables, so host memory can decide whether a model starts at all.

**The habit to learn:** a serving system is a stack. Look at every layer, record it, and change one thing at a time.

## Step 1: read the system, layer by layer

| Layer | Look with | What it decides | What we saw |
| --- | --- | --- | --- |
| GPUs | `nvidia-smi --query-gpu=index,name,memory.total --format=csv` | How many ranks, and whether weights plus a KV pool fit | 8 × 81,559 MiB on both nodes |
| GPU links | `nvidia-smi topo -m`, `nvidia-smi nvlink -s` | The cost of the all-reduce that ends every layer | Recorded per node in `_env/` |
| Driver | `nvidia-smi` | Which torch build the installer picks | Two driver versions, one torch build |
| CPU and NUMA | `lscpu`, `numactl -H` | Tokenization, scheduling, and so TTFT | Different CPU, 2 against 4 NUMA nodes |
| CPU frequency | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` | Whether cores change speed between steps | Recorded as active on the second node; not recorded on the first |
| Host RAM | `free -g` | Page cache for weights; tables kept in host memory | V4.1 needs about 190 GB pinned |
| Disk | `df -h /workspace` | Whether the checkpoints fit: 156 to 476 GB each | About 1.4 TB for all five |
| Filesystem of cache directories | `df -T /root /tmp` | Whether concurrent JIT compiles are safe | FUSE home broke them; `/tmp` works |
| Network | Watch the download rate | Time before the first launch | Qwen: 23 minutes at 240 MiB/s |
| Python | `python3 --version`, and its standard-library files | The bytes of the code corpus, so the requests | CPython 3.12.3 |
| vLLM build | `python -c 'import vllm; print(vllm.__version__)'` | Which layers and kernels really run | See step 5 |

One command records most of this in the format we published, so you can diff it:

```bash
export BLOG_STUDY=<your new study id>          # never a completed study; declare it in STUDIES in bench/blog_study.py first
$VLLM_VENV/bin/python bench/blog_study.py env   # writes results/$BLOG_STUDY/_env/
diff results/$BLOG_STUDY/_env/lscpu.txt  reports/blog-architecture-h100-v1/study/_env/lscpu.txt
diff results/$BLOG_STUDY/_env/freeze.txt reports/blog-architecture-h100-v1/study/_env/freeze.txt
```

## Step 2: install the runtime, then check what was resolved

```bash
python3 -m venv /root/vllm-latest
/root/vllm-latest/bin/python -m pip install --upgrade uv
/root/vllm-latest/bin/uv pip install --python /root/vllm-latest/bin/python \
  'vllm==0.31.1rc1.dev50+g554340f3d' pandas \
  --torch-backend=auto --index-strategy unsafe-best-match --prerelease=allow \
  --extra-index-url https://wheels.vllm.ai/554340f3d3259e321be4c07282be7a02a5aeef83
/root/vllm-latest/bin/python -c 'import vllm, torch; print(vllm.__version__, torch.__version__)'
```

`--torch-backend=auto` lets the installer choose the torch build for your driver. You pin vLLM; the rest is resolved on the day you install. So **the command is not the environment**: compare the result with the recorded `freeze.txt`. If it differs, say so and treat it as its own runtime arm.

## Step 3: put the things that depend on the machine in variables

| Variable | Meaning | Default in the scripts |
| --- | --- | --- |
| `VLLM_VENV` | Which runtime to use | `/root/vllm-latest` in `blog_launch.sh`, but the **older** `/root/vllm` in `serve.sh` |
| `HF_HOME` | Where checkpoints are stored | `/workspace/hf` |
| `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE` | Where JIT kernels are compiled | `/tmp/...`, set by `serve.sh` |
| `CUDA_VISIBLE_DEVICES` | Which GPUs the server may use | `0,1,2,3,4,5,6,7`, set by `serve.sh` |
| `BLOG_STUDY` | Which study the results belong to | The completed first-node study |

Export `VLLM_VENV` and `HF_HOME` yourself in every shell. Keep the two cache directories on a local disk: all eight ranks compile the same kernels at the same moment, and a network or FUSE filesystem can show one rank a half-written file.

## Step 4: know what you may change, and what makes it a new study

| If your system differs in | You can adapt | But then |
| --- | --- | --- |
| Paths, user, disk layout | The variables of step 3 | Nothing else changes |
| Host: CPU, NUMA, RAM, driver | Nothing to adapt | Run a node control (step 6) before comparing any latency |
| Python build | Use a matching CPython 3.12.3 for the corpus | If the request hashes differ, your inputs are not ours |
| vLLM build | Install another build in another environment | A new runtime arm: its numbers are never a control for ours |
| Number or type of GPUs | The tensor-parallel size and GPU list in `serve.sh` | A different deployment. This repository measured only TP 8 on H100 and tells you nothing about another shape |

Under tensor parallel, each rank holds a slice of every layer, so the weight bytes per GPU and the KV pool left over both depend on the rank count. The server prints the result at startup (`Model loading took … GiB`, `Available KV cache memory: … GiB`). Read those two lines on your system instead of assuming ours. MiMo's own model card recommends TP 4 at 0.95 utilization; we used TP 8 at 0.90 only so that all five models share one setup.

## Step 5: the first launch tells you what your system really runs

```bash
bash bench/serve.sh mimo-v26 off /tmp/serve-mimo-v26.log --max-num-batched-tokens 8192
grep -E "non-default args|Resolved architecture|Chunked prefill|Selected .* for|Using .* backend|KV cache|Model loading took|bounded replay" /tmp/serve-mimo-v26.log
```

Compare each line with the model's page in [reproduce/](../reproduce/README.md). A different line is a finding, not noise. On our own nodes one checkpoint behaved differently on two builds one week apart: the September build ran all 40 V4.1 layers on every prompt token and picked the Marlin expert kernel for MiMo; the October build skips 19 layers in prefill and picks `HUMMING`.

If the launch dies, look for the pattern before changing flags:

| Symptom | Layer at fault | Fix |
| --- | --- | --- |
| `CUDA error: invalid argument` after `flashinfer.jit: Building JIT module …` | Eight ranks building one module | Relaunch once |
| `FileNotFoundError` on `kernel.ttir` or `kernel.ptx` | Filesystem of the cache directory | Move the caches to local disk |
| `No available shared memory broadcast block found in 60 seconds`, repeated | Nothing: a large model is still loading | Wait |

The full list is in [reproduce/README.md](../reproduce/README.md#known-problems-on-every-model). Host memory is a risk we did not hit: both nodes had about 1.5 TB or more. On a smaller host, check `free -g` before launching V4.1, which pins about 190 GB for Engram tables on top of the page cache.

## Step 6: a node control before any comparison

Your node is a third node. Before you place a number beside ours, rerun **one model that was already measured** with the same requests and counts, and report the ratio:

```bash
bash bench/blog_launch.sh control timing:mimo-v26:1
python bench/blog_report.py serving
```

This needs your study declared in `STUDIES`, the rebuilt inputs and a frozen plan, as in [reproduce/README.md](../reproduce/README.md#3-rebuild-public-inputs).

Read it the way [lesson 6](06_new_checkpoint_new_study.md) does. If throughput is within about 2% and latency is not, compare throughput only. Never divide your results by the ratio to "correct" them: the control tells you what is comparable, it is not a calibration.

## Step 7: write down what differed

For every deviation, record the layer, the old and new value, and whether a control covered it. A table of numbers without this is a table about an unknown system.

## Check yourself

- Two nodes have identical GPUs, driver, vLLM and torch. Which results may still differ, and how would you find out?
- Why does `pip install` with a pinned vLLM version not guarantee the recorded environment?
- You launch with `bash bench/serve.sh v41 off …` in a fresh shell and the `bounded replay` line is missing. Name the most likely cause. (`VLLM_VENV` is unset, so the older build in `/root/vllm` ran.)
- Your node has four GPUs. Can you reuse our KV pool sizes? (No. Read `Available KV cache memory` on your own launch; it is a different deployment and a new study.)
- The dry run says the 16K timing list differs from the reference inputs. Is the tool broken? (No. Your Python standard library differs, so the code corpus does. The check is doing its job.)
