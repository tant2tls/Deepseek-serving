# Findings: Qwen3.8-Flash-Next, original BF16 checkpoint, on 8×H100

Study `qwen-bf16-h100-v1`, measured 2026-10-08 on one rented 8×H100 80GB node. Protocol: [blog_target.md](../../blog_target.md), unchanged. Checkpoint rule: [docs/qwen-checkpoint-policy.md](../../docs/qwen-checkpoint-policy.md). Status of every item, deviations and commands: [handoff.md](handoff.md).

**Read this first**

- **Checkpoint:** `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, the original BF16 weights. Model key `qwen-38-bf16`. The FP8 checkpoint was not run.
- **Build:** vLLM `0.31.1rc1.dev50+g554340f3d` (commit `554340f3d3259e321be4c07282be7a02a5aeef83`), torch `2.13.0+cu132`, the build of `blog-architecture-h100-v1`.
- **Deployment, unchanged:** TP8 + expert parallel, memory utilization 0.90, context limit 262,144, 64 sequences, 8,192-token chunks, native precision, prefix caching off, speculation off, temperature 0, thinking off (`enable_thinking=false`).
- **Workload, unchanged:** the same public code, math and chat request lists as the blog study, hash for hash, with the same counts; 256 forced output tokens. The BF16 tokenizer gives the same token count as the FP8 one on all 543 requests.
- **Another node.** Same GPU model and count as on 2026-10-07, but another CPU (Xeon 8592V, 4 NUMA nodes, frequency scaling active), a newer GPU driver and a few newer Python packages. One MiMo timing block and one set of MiMo traces were run here as **node controls**. Read section 4 before comparing anything across the two nodes.
- **Evidence labels:** *measured* (unprofiled timing, three launch-separated blocks), *trace* (profiler diagnostics, GPU kernel sums, not wall time), *snapshot* (live KV gauge), *log/config*, *estimate*.
- **What "BF16 against FP8" means here:** two deployments. The checkpoint differs, so does the expert kernel the build selects, the KV pool that is left after the weights, the node and the day. It is not a controlled test of precision, and it says nothing about answer quality.

## 1. Answers in one table

| Question | Answer | Evidence |
| --- | --- | --- |
| Does this build load and run the original BF16 checkpoint in the common deployment? | **Yes.** `dtype=torch.bfloat16`, `quantization=None`, unquantized Triton experts, BF16 n-gram embedding, no FP8 path in the log. It fits: 31.42 GiB of weights and a 37.37 GiB KV pool per GPU | log, section 2 |
| Is the measurement valid? | **18 of 18 timing runs valid** over three blocks, functional check passed, pilot 6/6, six traces, six KV snapshots. Controls: MiMo 6 of 6 timing runs valid, six traces, six KV snapshots | [serving.csv](serving.csv) |
| How fast is it? | 154.7 / 793.0 tok/s at 1K with one / eight clients, 125.4 / 385.5 at 16K, 77.5 / 142.5 at 64K. Single-stream decode 5.77–5.85 ms per token | measured, section 3 |
| Is this node comparable with the 2026-10-07 node? | **For throughput and single-client TPOT, within 2%** (MiMo here is 0.983–0.995 of its earlier throughput), and **for GPU kernel time per step, within 3%** (all-reduce aside). **Not for TTFT or eight-client TPOT medians**, which moved by up to +30% and −14% for the same model | measured + trace, section 4 |
| BF16 against the historical FP8 result? | **Faster alone, slower under load.** One client: +6.7% at 1K and +4.1% at 16K, equal at 64K. Eight clients: −11.6% at 1K, −6.6% at 16K, −1.8% at 64K (the last is inside the node difference) | measured, section 5 |
| Why? | BF16 skips input quantization and its dense and expert GEMMs are cheaper at batch 1, so a lone decode step is about 0.5 ms shorter. Its Triton expert kernel grows 2.8× from one to eight sequences where the FP8 kernel grew 1.4× | trace, section 6 |
| What does BF16 cost in memory? | 14.1 GiB more weights per GPU, so a KV pool that is 27% smaller (3.05M against 4.19M tokens). Live KV per token is unchanged (13.1 KiB per GPU at 64K) | log + snapshot, section 7 |

## 2. What was loaded

*Log*, first launch on this node (`data/qwen-38-bf16/_server/`).

| Line in the server log | What it establishes |
| --- | --- |
| `Resolved architecture: Qwen4ExpForConditionalGeneration` | Same served class as the FP8 checkpoint |
| `model='Qwen/Qwen3.8-Flash-Next' … revision=de4b8e4d… tokenizer_revision=de4b8e4d… dtype=torch.bfloat16 … quantization=None … kv_cache_dtype=auto` | Original repository at the pinned revision, native precision, no weight quantization |
| `Using TRITON Unquantized MoE backend out of potential backends: ['TRITON', 'BATCHED_TRITON', 'FlashInfer TRTLLM', 'FlashInfer CUTLASS']`, `Using TritonExperts MoE backend` | The experts run unquantized in Triton's `fused_moe_kernel`. This is the build's automatic choice; FP8 used `FLASHINFER_CUTLASS` |
| `Initialized PLE embedding … Qwen4ExpPLEUnquantizedEmbeddingMethod, weight_dtype=torch.bfloat16, weight_device=cpu, pinned=True` | The n-gram embedding is BF16 and sits in pinned host memory |
| `Using FlashInfer GDN prefill kernel`, `attention block size 400`, `kv cache group sizes [262144, 262144, 262144, 262144, 4, 400]` | Recurrent layers and cache layout as for FP8 |
| `Model loading took 31.42 GiB`, `Available KV cache memory: 37.37 GiB`, `GPU KV cache size: 3,046,184 tokens`, `Maximum concurrency for 262,144 tokens per request: 11.62x` | It fits at utilization 0.90 with the full context limit |

The string `fp8` does not occur in the log. Architecture (config): 48 layers, 36 recurrent and 12 softmax-attention, hidden size 2,560, 512 routed experts with 10 active per token and width 640, as described for the FP8 checkpoint in the [blog study's architecture table](../blog-architecture-h100-v1/architecture.md); the two checkpoints differ in weight format, not in structure.

**Functional check** (`data/qwen-38-bf16/_functional/functional.json`): the rendered request has 1,021 prompt tokens for 1,009 content tokens (template adds 12, as for FP8); three natural-EOS answers (code, math, chat) stop by themselves with no reasoning text; the forced request returns exactly 256 tokens.

First launch 580 s, later launches 255–260 s.

## 3. Serving: the BF16 numbers

*Measured.* Mean ± sample SD over three launch-separated blocks, the three block values in brackets. Three blocks screen effects; they are not confidence intervals. vLLM `554340f3…`, this node.

| Point | Output tok/s | TTFT p50 ms | TPOT p50 ms | End-to-end p50 ms | Window s |
| --- | --- | --- | --- | --- | ---: |
| 1K, c1 | **154.7 ± 2.1** [152.9, 157.0, 154.4] | 147 ± 37 [189, 120, 131] | 5.85 ± 0.00 [5.85, 5.85, 5.85] | 1,639 ± 37 | 78–80 |
| 1K, c8 | **793.0 ± 2.7** [790.0, 793.7, 795.4] | 379 ± 2 [379, 382, 377] | 8.56 ± 0.00 [8.56, 8.56, 8.56] | 2,566 ± 4 | 66–66 |
| 16K, c1 | **125.4 ± 0.2** [125.2, 125.4, 125.6] | 537 ± 11 [548, 536, 527] | 5.85 ± 0.01 [5.84, 5.86, 5.86] | 2,029 ± 8 | 73–74 |
| 16K, c8 | **385.5 ± 2.5** [385.8, 387.8, 382.8] | 1,681 ± 47 [1,642, 1,669, 1,733] | 13.08 ± 0.35 [13.10, 12.73, 13.42] | 5,113 ± 25 | 50–50 |
| 64K, c1 | **77.5 ± 0.7** [76.7, 77.9, 78.0] | 1,808 ± 5 [1,813, 1,805, 1,805] | 5.77 ± 0.00 [5.77, 5.77, 5.77] | 3,281 ± 3 | 59–60 |
| 64K, c8 | **142.5 ± 0.7** [143.0, 141.7, 142.9] | 6,240 ± 256 [6,188, 6,517, 6,014] | 30.31 ± 0.88 [30.46, 29.36, 31.09] | 14,061 ± 70 | 59–60 |

Server-counted prompt tokens per request: 1,075–1,082 at "1K", 17,199–17,268 at "16K", 68,724–69,189 at "64K", identical to the FP8 runs.

Two things to know about these numbers:

- **TTFT at 1K with one client has two modes on this node**, about 190 ms and about 120 ms, and a run switches between them: 34 of 48 requests were in the slow mode in block 1, 8 in block 2 and 20 in block 3. Hence 189, 120 and 131 ms as block medians, and the ± 37. MiMo shows the same pattern here (about 150 ms for its first 15 requests, then about 113 ms) and did not on the first node. It is a property of this node or session, not of the checkpoint; see section 4.
- **Windows** are 50 s at 16K/c8 and 59–60 s at both 64K points, because the counts are the blog study's. The FP8 windows were 46 s at 16K/c8 and 58 s at 64K/c8.

## 4. Is this node comparable with the first one?

*Measured.* MiMo-V2.6-Flash, same build, flags, requests and counts, **one block here** against its three blocks of 2026-10-07. Full table: [comparison.md](comparison.md).

| Point | Output tok/s here | 2026-10-07 mean [min, max] | Here / then | TTFT p50 ms here / then | TPOT p50 ms here / then |
| --- | ---: | --- | ---: | --- | --- |
| 1K, c1 | 158.9 | 160.6 [160.2, 161.0] | 0.989 | 116 / 114 (1.02×) | 5.80 / 5.79 (1.00×) |
| 1K, c8 | 679.1 | 687.7 [684.6, 689.3] | 0.988 | 436 / 409 (1.06×) | 9.95 / 9.93 (1.00×) |
| 16K, c1 | 112.7 | 114.3 [113.5, 114.8] | 0.986 | 815 / 762 (1.07×) | 5.83 / 5.86 (1.00×) |
| 16K, c8 | 249.6 | 250.9 [249.1, 251.8] | 0.995 | 3,121 / 2,878 (1.08×) | 17.71 / 20.16 (0.88×) |
| 64K, c1 | 54.5 | 55.4 [55.3, 55.5] | 0.983 | 3,235 / 3,142 (1.03×) | 5.89 / 6.01 (0.98×) |
| 64K, c8 | 77.5 | 78.5 [77.9, 79.6] | 0.987 | 10,944 / 8,406 (1.30×) | 58.92 / 68.26 (0.86×) |

**What the control says:**

1. **Throughput is 0.983–0.995 of the earlier value at every point**, a consistent shortfall of 0.5–1.7%. Five of the six values lie just below the earlier min–max range, so the shortfall is real but small. Cross-node throughput differences under about 2% cannot be assigned to a model.
2. **Single-client TPOT is the same** (0.98–1.00×). Decode on an idle engine is where the two nodes agree best.
3. **TTFT is higher here** by 2–9% at five points and by 30% at 64K with eight clients, **and eight-client TPOT is lower** by 12–14% at 16K and 64K. Throughput hardly moves, so time has shifted between "waiting for the first token" and "waiting between tokens" in how the scheduler interleaves prefill chunks with decode steps. **Do not compare TTFT medians or eight-client TPOT medians across the two nodes.** Throughput and end-to-end latency are the robust quantities.
4. **Likely cause, not proven:** this node's CPUs run the `schedutil` frequency governor between 800 and 3,900 MHz (most cores sat at 800 MHz when sampled), and the governor cannot be changed from inside the container. The first node exposed no frequency scaling. CPU-side work (tokenizing, scheduling, launching kernels) then runs at a speed that depends on recent load, which fits the two TTFT modes of section 3. No experiment here isolates it.

Limit of the timing control: one block, one model, and it was MiMo's first launch on this node.

### Trace-level control

*Trace.* The same six MiMo captures as on 2026-10-07, taken here on the profiler launch ([addendum](plan_addendum_mimo_trace_control.json)). Mean of 8 ranks, ms per step.

| ms per step, mean of 8 ranks | prefill64k here | prefill64k 10-07 | decode1k_B1 here | decode1k_B1 10-07 | decode64k_B8 here | decode64k_B8 10-07 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 357.6 | 361.3 | 6.14 | 6.04 | 11.14 | 11.22 |
| Attention core | 35.3 | 35.4 | 0.55 | 0.56 | 1.87 | 1.87 |
| Dense GEMM (projections) | 11.88 | 11.90 | 1.09 | 1.09 | 1.10 | 1.10 |
| Residual streams / norm | 8.53 | 8.59 | – | – | – | – |
| MoE expert GEMM | 242.1 | 244.0 | 1.43 | 1.44 | 4.75 | 4.82 |
| GEMM input quantization | 9.32 | 9.36 | 0.17 | 0.17 | 0.20 | 0.20 |
| MoE routing, activation, combine | 4.76 | 4.78 | 0.95 | 0.95 | 0.70 | 0.69 |
| TP all-reduce | 32.1 | 33.7 | 1.17 | 1.06 | 1.72 | 1.74 |
| Other / unclassified | 1.17 | 1.18 | 0.16 | 0.16 | 0.17 | 0.18 |

**Every component other than all-reduce reproduces within 3%**, the step span included (6.14 against 6.04 ms at decode batch 1, 358 against 361 ms for a 64K prefill chunk). All-reduce is within about 10% in five captures and 16% higher in the 16K prefill chunk (36.8 against 31.7 ms). MiMo's KV pool (49.20 against 49.18 GiB per GPU) and live KV per token at 64K (5.75 against 5.69 KiB) reproduce as well. So **at trace level the two nodes are interchangeable**, and the profiler adds the same overhead on both (MiMo, 1K, batch 1: 5.81 ms unprofiled and 6.14 ms traced here; 5.78 and 6.04 ms then). Differences between the Qwen BF16 traces and the Qwen FP8 traces in section 6 therefore belong to the deployments, not to the node. Two of the four MiMo decode windows ran the 5-second cap because the stream had passed its natural end; their step averages cover more steps and are unaffected.

## 5. BF16 beside the FP8 history

*Measured*, two studies. FP8 values are the three blocks of 2026-10-07/08 on the first node; none was repeated. The control column is MiMo's ratio from section 4 at the same point: it shows what the node alone did to another model, and it is **not** applied to the Qwen numbers.

| Point | BF16 tok/s (this node) | FP8 tok/s (2026-10-07/08) | BF16 / FP8 | MiMo control, this node / first node |
| --- | ---: | ---: | ---: | ---: |
| 1K, c1 | 154.7 ± 2.1 | 145.0 ± 0.1 | **1.067** | 0.989 |
| 1K, c8 | 793.0 ± 2.7 | 897.5 ± 3.0 | **0.884** | 0.988 |
| 16K, c1 | 125.4 ± 0.2 | 120.5 ± 0.1 | **1.041** | 0.986 |
| 16K, c8 | 385.5 ± 2.5 | 412.6 ± 1.6 | **0.934** | 0.995 |
| 64K, c1 | 77.5 ± 0.7 | 77.0 ± 0.2 | **1.007** | 0.983 |
| 64K, c8 | 142.5 ± 0.7 | 145.1 ± 0.3 | **0.982** | 0.987 |

| Point | TTFT p50 ms BF16 / FP8 | ratio | MiMo control | TPOT p50 ms BF16 / FP8 | ratio | MiMo control |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 1K, c1 | 147 / 153 | 0.96× | 1.02× | 5.85 / 6.31 | 0.93× | 1.00× |
| 1K, c8 | 379 / 296 | 1.28× | 1.06× | 8.56 / 7.61 | 1.13× | 1.00× |
| 16K, c1 | 537 / 529 | 1.01× | 1.07× | 5.85 / 6.36 | 0.92× | 1.00× |
| 16K, c8 | 1,681 / 1,561 | 1.08× | 1.08× | 13.08 / 12.67 | 1.03× | 0.88× |
| 64K, c1 | 1,808 / 1,724 | 1.05× | 1.03× | 5.77 / 6.36 | 0.91× | 0.98× |
| 64K, c8 | 6,240 / 4,845 | 1.29× | 1.30× | 30.31 / 34.66 | 0.87× | 0.86× |

**What this says:**

- **One client.** BF16 is ahead at 1K (+6.7%) and 16K (+4.1%) although this node is 1–2% slower for the control. The reason is decode: 5.85 ms per token against 6.31 at 1K and 5.77 against 6.36 at 64K, a gain of 7–9% where the control moved by 0–2%. At 64K the two are equal in throughput (77.5 against 77.0 tok/s): prefill takes over the request there, and BF16's TTFT is 5% higher (control: 3%).
- **Eight clients.** BF16 is behind at 1K (−11.6%) and 16K (−6.6%), far outside the control's −1.2% and −0.5%. At 64K the difference (−1.8%) equals the control's (−1.3%) and cannot be assigned to the checkpoint.
- **TTFT and TPOT medians at eight clients** follow the node, not the checkpoint, at 64K: BF16/FP8 is 1.29× and 0.87×, MiMo's own shift is 1.30× and 0.86×. At 1K/c8, however, BF16's TPOT is 13% higher where the control's did not move, and that is where the throughput loss comes from.

### Beside the other models

*Measured*, but **on two nodes**: the first five columns are the blog study, the last is this study. Only throughput is shown, because it is the quantity the control found comparable within about 2%.

| Output tok/s | 0731 | V4.1 | MiMo | GLM | Qwen FP8 | Qwen BF16, other node |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K, c1 | 118.1 | 106.0 | **160.6** | 137.3 | 145.0 | 154.7 |
| 1K, c8 | 598.7 | 562.9 | 687.7 | 705.8 | **897.5** | 793.0 |
| 16K, c1 | 93.4 | 89.3 | 114.3 | 109.4 | 120.5 | **125.4** |
| 16K, c8 | 237.5 | 278.9 | 250.9 | 322.3 | **412.6** | 385.5 |
| 64K, c1 | 48.5 | 55.9 | 55.4 | 66.4 | 77.0 | **77.5** |
| 64K, c8 | 71.5 | 100.5 | 78.5 | 115.7 | **145.1** | 142.5 |

The original BF16 checkpoint keeps Qwen's place in the picture: first or second at every point. It leads at 16K with one client. At 64K with one client it is level with FP8 (0.7% apart, inside the node difference). MiMo is still ahead at 1K with one client, and the FP8 result is still the highest at all three eight-client points. BF16 stays ahead of GLM, V4.1, 0731 and MiMo at every eight-client point by 12% or more, well beyond the node difference. This is not a same-session ranking of six deployments.

## 6. Where the time goes

*Trace*, mean of the 8 ranks, ms of GPU kernel time per engine step; FP8 columns are the 2026-10-07 traces. A kernel sum is not wall time. Full tables: [components_tables.md](components_tables.md).

**Prefill, one 8,192-token chunk**

| ms per step, mean of 8 ranks | prefill16k BF16 | prefill16k FP8 | prefill64k BF16 | prefill64k FP8 |
| --- | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 238.3 | 205.5 | 227.6 | 192.1 |
| Attention core | 29.5 | 29.5 | 30.0 | 30.1 |
| Recurrent state update | 5.10 | 5.09 | 5.10 | 5.09 |
| Indexer / top-k | 4.85 | 4.84 | 12.16 | 12.17 |
| Dense GEMM (projections) | 32.1 | 32.8 | 32.1 | 32.8 |
| Residual streams / norm | 32.1 | 32.1 | 32.1 | 32.1 |
| MoE expert GEMM | 18.69 | 10.49 | 18.74 | 10.43 |
| GEMM input quantization | – | 8.37 | – | 8.37 |
| MoE routing, activation, combine | 10.92 | 7.50 | 10.93 | 7.50 |
| TP all-reduce | 75.2 | 51.5 | 61.2 | 33.0 |
| Other / unclassified | 7.98 | 8.09 | 7.93 | 8.05 |

**Decode, one step**

| ms per step, mean of 8 ranks | decode1k_B1 BF16 | decode1k_B1 FP8 | decode1k_B8 BF16 | decode1k_B8 FP8 | decode64k_B8 BF16 | decode64k_B8 FP8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 8.94 | 6.85 | 9.26 | 7.82 | 9.03 | 8.22 |
| Attention core | 0.06 | 0.06 | 0.13 | 0.13 | 0.13 | 0.13 |
| Recurrent state update | 0.25 | 0.25 | 0.30 | 0.30 | 0.30 | 0.30 |
| Indexer / top-k | 0.12 | 0.12 | 0.14 | 0.15 | 0.35 | 0.34 |
| Dense GEMM (projections) | 2.03 | 2.26 | 2.31 | 2.55 | 2.31 | 2.55 |
| Residual streams / norm | 1.48 | 1.41 | 1.92 | 1.64 | 2.00 | 1.67 |
| MoE expert GEMM | 0.80 | 1.04 | 2.25 | 1.45 | 2.34 | 1.45 |
| GEMM input quantization | – | 0.23 | – | 0.34 | – | 0.34 |
| MoE routing, activation, combine | 1.06 | 0.80 | 1.05 | 0.85 | 1.04 | 0.85 |
| TP all-reduce | 2.89 | 1.02 | 1.75 | 0.85 | 1.32 | 1.00 |
| Other / unclassified | 0.64 | 0.65 | 0.84 | 0.68 | 0.85 | 0.69 |

**How to read it:**

1. **Everything that does not touch quantized weights costs the same on both nodes**: in prefill the attention core, recurrent update, indexer and residual streams agree to within 0.3%, and in decode the first three agree to the rounding shown. The GPUs do the same work at the same speed. This is the trace-level node control for GPU kernels.
2. **BF16 removes the input-quantization step** (8.4 ms per chunk, 0.23–0.34 ms per decode step) and its dense projections are slightly cheaper (cuBLAS BF16 GEMM, 2.03 against 2.26 ms at batch 1).
3. **The expert kernel is the difference, and its sign depends on batch.** At decode batch 1 the Triton BF16 expert GEMM takes 0.80 ms against 1.04 ms for the FP8 kernel. At batch 8 it takes 2.25–2.34 ms against 1.45 ms: it grows 2.8× where the FP8 kernel grew 1.4×. In prefill it takes 18.7 ms per chunk against 10.4 ms. Routing and combine are also 0.2–3.4 ms more expensive on the Triton path.
4. **Adding it up for decode.** Batch 1: expert path (GEMM + input preparation + routing) 1.86 ms against 2.07 ms, projections 2.03 against 2.26 ms, together 0.44 ms less, which matches the measured TPOT gain of 0.46 ms (5.85 against 6.31 ms). Batch 8: expert path 3.30 against 2.64 ms, residual mixing 1.92 against 1.64 ms, projections 0.24 ms less: about 0.7–0.9 ms more, against a measured TPOT that is 0.95 ms higher at 1K with eight clients. The trace explains the direction and most of the size in both cases.
5. **All-reduce time and profiler overhead are larger in the BF16 traces, and the node is not the reason.** All-reduce is 61–75 ms per prefill chunk against 33–52 ms, and 1.3–2.9 ms per decode step against 0.85–1.02 ms. A BF16 decode step at 1K and batch 1 takes 5.88 ms unprofiled and 8.94 ms in the trace; FP8 took 6.33 and 6.85 ms. MiMo's traces on this node match its earlier ones (section 4), so this is a property of the BF16 deployment when traced. An all-reduce kernel waits until every rank has delivered, so it absorbs any skew between ranks. The unprofiled timing shows no matching cost: BF16 decodes faster than FP8 alone, and its single-client TTFT at 16K and 64K is only 1–5% higher. The likeliest reading is that the profiler slows the Triton expert path unevenly across ranks and the wait lands in all-reduce; this was not isolated.
6. **So use each instrument for what it can show.** The traces say which kernels run and how each scales with batch (points 2–4). The timing runs say how long a step takes. Do not read the BF16 all-reduce rows as communication cost.

One capture is unusable for client step time: at 64K with one sequence the client received 48 text chunks while the engine ran 143 steps (forced length past the natural end). Its kernel averages are fine and agree with the 1K capture.

## 7. Memory

*Log and snapshot*, per GPU. Method and caveats as in the blog study: live bytes = usage gauge × pool, an approximation. Table: [memory_tables.md](memory_tables.md).

| Per GPU | Qwen BF16 (this study) | Qwen FP8 (history) |
| --- | ---: | ---: |
| Weights loaded (log) | 31.42 GiB | 17.36 GiB |
| KV pool reserved (log) | 37.37 GiB | 51.44 GiB |
| Pool capacity in tokens (log) | 3,046,184 | 4,194,304 |
| Live KiB per token at 64K, B=1 | 13.14 | 13.16 |
| One live 64K request | 892 MiB | 877 MiB |
| One live 64K request, whole node | 6.97 GiB | 6.85 GiB |
| Eight live 64K requests, share of the pool | 19.0% | 13.6% |

- **The live state is the same.** KV and recurrent state do not depend on the weight format: 13.14 KiB per token per GPU at 64K against 13.16.
- **The pool is what shrinks.** BF16 weights take 14.06 GiB more on every GPU, and at a fixed 0.90 utilization that comes out of the KV pool: 37.37 GiB instead of 51.44 GiB, 27% less. Eight live 64K requests occupy 19.0% of it instead of 13.6%. The log's own figure is 11.62 full-length (262,144-token) requests.
- **Host memory.** The BF16 n-gram embedding is held in pinned host memory, not on the GPUs. Its size was not measured.
- At 1K the per-token figures of the two studies (20.8–24.5 against 27.5–27.8 KiB) are not comparable: the snapshot was taken after more generated tokens here, which spreads the fixed recurrent state over more tokens.

None of this measures maximum concurrency or reusable-prefix capacity.

## 8. Strengths and weaknesses of the BF16 deployment

| | On this build | Explanation and confidence |
| --- | --- | --- |
| Strength | Single-stream decode level with MiMo, the fastest of the first five: 5.77–5.85 ms per token (MiMo 5.79–6.01 on the first node and 5.80–5.89 here; FP8 Qwen 6.31–6.36) | No input quantization, cheaper dense and expert GEMMs at batch 1 (trace, medium). Cross-node, but single-client TPOT is the best-matched quantity (control 0.98–1.00×) |
| Strength | Highest throughput at 16K with one client (125.4 tok/s); level with FP8 at 64K | measured; node difference is under 2% |
| Strength | Original weights: no quantization step between the released checkpoint and what is served | config + log, high |
| Weakness | 12% and 7% lower throughput than FP8 at 1K and 16K with eight clients | Triton expert kernel grows 2.8× from batch 1 to 8 (trace, medium). The backend is the build's choice; alternatives were not tried |
| Weakness | 27% smaller KV pool, 81% more GPU weight memory | log, high |
| Weakness | 336 GiB on disk against 173 GiB; the first launch reads all of it | measured |
| Not shown | Any quality difference between BF16 and FP8 | not measured |

## 9. Limits and what is open

- **No same-node FP8 control**, by rule. BF16-against-FP8 differences smaller than the node control (about 2% in throughput) are not attributable; TTFT and eight-client TPOT medians are not comparable across the nodes at all.
- **One evening, one model plus a control.** Blocks are launch-separated, not day-separated, and nothing was rotated against other models.
- **One trace capture per point**, and the BF16 traces carry a larger profiler overhead than the FP8 or MiMo traces (section 6, point 5).
- **The expert backend was not varied.** Whether `FlashInfer CUTLASS` or `TRTLLM` closes the eight-client gap is the most valuable single-change follow-up; see [handoff.md](handoff.md) section 5.
- **The cause of the TTFT modes on this node is a hypothesis.**
- HBM counters, routing statistics, prefix capacity, 128K behavior and quality remain unmeasured, as in the blog study.
