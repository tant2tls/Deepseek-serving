# Component breakdown (diagnostic traces, idle-profiler launch)

Unit: ms of GPU kernel time per engine step, **mean over the 8 ranks**. A kernel sum is not wall time: kernels on different streams overlap and idle gaps are not kernels. Prefill rows are the last full 8,192-token chunk of one public-text request. Decode rows average the steps of the profiled window at the stated engine batch B. Classifier: `bench/blog_report.py`; leftovers: [components_unclassified.md](components_unclassified.md).


## Prefill: one 8,192-token chunk

| Component | Qwen3.8-Flash-Next (BF16) prefill16k | MiMo-V2.6-Flash prefill16k | Qwen3.8-Flash-Next (BF16) prefill64k | MiMo-V2.6-Flash prefill64k |
| --- | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 238.3 | 334.0 | 227.6 | 357.6 |
| **Kernel sum** | 216.4 | 311.6 | 210.4 | 346.4 |
| Attention core (softmax attention over stored KV) | 29.5 | 4.40 | 30.0 | 35.3 |
| Recurrent (linear) attention state update | 5.10 | – | 5.10 | – |
| Indexer / top-k / candidates | 4.85 | – | 12.16 | – |
| Q/K/V norm, RoPE, KV insert, attention metadata | 0.06 | 1.15 | 0.06 | 1.17 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 32.1 | 11.59 | 32.1 | 11.88 |
| GEMM input quantization | – | 9.21 | – | 9.32 |
| mHC / residual / norm | 32.1 | 8.53 | 32.1 | 8.53 |
| MoE expert GEMM | 18.69 | 234.1 | 18.74 | 242.1 |
| MoE routing, activation, combine | 10.92 | 4.68 | 10.93 | 4.76 |
| TP all-reduce | 75.2 | 36.8 | 61.2 | 32.1 |
| Other / unclassified | 7.98 | 1.20 | 7.93 | 1.17 |
| *Attention path without projections (core + recurrent + indexer + KV work)* | 39.5 | 5.54 | 47.4 | 36.4 |
| *FFN path (expert GEMM + routing/activation/combine)* | 29.6 | 238.8 | 29.7 | 246.9 |

## Decode at engine batch 1

| Component | Qwen3.8-Flash-Next (BF16) decode1k_B1 | MiMo-V2.6-Flash decode1k_B1 | Qwen3.8-Flash-Next (BF16) decode64k_B1 | MiMo-V2.6-Flash decode64k_B1 |
| --- | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 8.94 | 6.14 | 8.66 | 6.26 |
| **Kernel sum** | 9.35 | 5.60 | 9.53 | 5.75 |
| Client step time in the same window | 9.71 | 278.4 | 26.1 | n/a |
| Attention core (softmax attention over stored KV) | 0.06 | 0.55 | 0.06 | 0.83 |
| Recurrent (linear) attention state update | 0.25 | – | 0.25 | – |
| Indexer / top-k / candidates | 0.12 | – | 0.21 | – |
| Q/K/V norm, RoPE, KV insert, attention metadata | 0.01 | 0.08 | 0.01 | 0.08 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 2.03 | 1.09 | 2.02 | 1.08 |
| GEMM input quantization | – | 0.17 | – | 0.17 |
| mHC / residual / norm | 1.48 | – | 1.48 | – |
| MoE expert GEMM | 0.80 | 1.43 | 0.80 | 1.43 |
| MoE routing, activation, combine | 1.06 | 0.95 | 1.05 | 0.95 |
| TP all-reduce | 2.89 | 1.17 | 3.00 | 1.04 |
| Other communication | 0.02 | 0.02 | 0.02 | 0.02 |
| Other / unclassified | 0.64 | 0.16 | 0.63 | 0.16 |
| *Attention path without projections (core + recurrent + indexer + KV work)* | 0.44 | 0.62 | 0.53 | 0.91 |
| *FFN path (expert GEMM + routing/activation/combine)* | 1.86 | 2.39 | 1.85 | 2.38 |

## Decode at engine batch 8

| Component | Qwen3.8-Flash-Next (BF16) decode1k_B8 | MiMo-V2.6-Flash decode1k_B8 | Qwen3.8-Flash-Next (BF16) decode64k_B8 | MiMo-V2.6-Flash decode64k_B8 |
| --- | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 9.26 | 10.17 | 9.03 | 11.14 |
| **Kernel sum** | 10.72 | 9.65 | 10.67 | 10.62 |
| Client step time in the same window | 9.48 | 10.25 | 9.29 | 11.19 |
| Attention core (softmax attention over stored KV) | 0.13 | 0.81 | 0.13 | 1.87 |
| Recurrent (linear) attention state update | 0.30 | – | 0.30 | – |
| Indexer / top-k / candidates | 0.14 | – | 0.35 | – |
| Q/K/V norm, RoPE, KV insert, attention metadata | 0.01 | 0.08 | 0.01 | 0.08 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 2.31 | 1.11 | 2.31 | 1.10 |
| GEMM input quantization | – | 0.20 | – | 0.20 |
| mHC / residual / norm | 1.92 | – | 2.00 | – |
| MoE expert GEMM | 2.25 | 4.80 | 2.34 | 4.75 |
| MoE routing, activation, combine | 1.05 | 0.70 | 1.04 | 0.70 |
| TP all-reduce | 1.75 | 1.75 | 1.32 | 1.72 |
| Other communication | 0.03 | 0.03 | 0.03 | 0.03 |
| Other / unclassified | 0.84 | 0.18 | 0.85 | 0.17 |
| *Attention path without projections (core + recurrent + indexer + KV work)* | 0.58 | 0.89 | 0.78 | 1.95 |
| *FFN path (expert GEMM + routing/activation/combine)* | 3.30 | 5.50 | 3.38 | 5.45 |

Projections are listed under dense GEMM because the kernels do not say which projection they serve; on DeepSeek they are the low-rank Q and grouped O projections, on MiMo the fused QKV, O and the layer-0 FFN. They are therefore **not** included in the attention-path line above, and the complete attention path lies between that line and that line plus dense GEMM.

