# Component breakdown (diagnostic traces, idle-profiler launch)

Unit: ms of GPU kernel time per engine step, **mean over the 8 ranks**. A kernel sum is not wall time: kernels on different streams overlap and idle gaps are not kernels. Prefill rows are the last full 8,192-token chunk of one public-text request. Decode rows average the steps of the profiled window at the stated engine batch B. Classifier: `bench/blog_report.py`; leftovers: [components_unclassified.md](components_unclassified.md).


## Prefill: one 8,192-token chunk

| Component | V4 Flash 0731 prefill16k | V4.1 Flash prefill16k | MiMo-V2.6-Flash prefill16k | Qwen3.8-Flash-Next prefill16k | GLM-5.3-Flash prefill16k | V4 Flash 0731 prefill64k | V4.1 Flash prefill64k | MiMo-V2.6-Flash prefill64k | Qwen3.8-Flash-Next prefill64k | GLM-5.3-Flash prefill64k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 355.1 | 263.0 | 328.6 | 205.5 | 240.2 | 404.7 | 262.3 | 361.3 | 192.1 | 241.6 |
| **Kernel sum** | 336.6 | 244.2 | 306.6 | 190.4 | 210.8 | 395.0 | 263.1 | 350.1 | 179.6 | 230.4 |
| Attention core (softmax attention over stored KV) | 41.0 | 25.3 | 4.41 | 29.5 | 35.2 | 55.3 | 26.6 | 35.4 | 30.1 | 40.1 |
| Recurrent (linear) attention state update | – | – | – | 5.09 | 15.65 | – | – | – | 5.09 | 15.73 |
| Indexer / top-k / candidates | 6.38 | 2.39 | – | 4.84 | 2.03 | 41.1 | 15.50 | – | 12.17 | 13.05 |
| Q/K/V norm, RoPE, KV insert, attention metadata | 14.16 | 5.34 | 1.15 | 0.06 | 0.47 | 14.25 | 5.51 | 1.17 | 0.06 | 0.47 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 39.7 | 29.4 | 11.54 | 32.8 | 27.0 | 40.1 | 30.7 | 11.90 | 32.8 | 27.1 |
| GEMM input quantization | 10.90 | 3.22 | 9.21 | 8.37 | 13.75 | 11.08 | 3.23 | 9.36 | 8.37 | 13.83 |
| mHC / residual / norm | 28.4 | 17.48 | 8.57 | 32.1 | 38.9 | 28.4 | 17.40 | 8.59 | 32.1 | 39.0 |
| MoE expert GEMM | 155.3 | 124.1 | 234.1 | 10.49 | 24.4 | 158.5 | 125.4 | 244.0 | 10.43 | 25.1 |
| MoE routing, activation, combine | 3.11 | 2.51 | 4.68 | 7.50 | 9.27 | 3.12 | 3.43 | 4.78 | 7.50 | 9.37 |
| TP all-reduce | 26.7 | 21.5 | 31.7 | 51.5 | 32.1 | 31.9 | 21.8 | 33.7 | 33.0 | 34.5 |
| Other communication | – | 2.84 | – | – | – | – | 2.87 | – | – | – |
| Engram | – | 2.55 | – | – | – | – | 3.05 | – | – | – |
| Other / unclassified | 11.00 | 7.64 | 1.20 | 8.09 | 12.10 | 11.21 | 7.60 | 1.18 | 8.05 | 12.14 |
| *Attention path without projections (core + recurrent + indexer + KV work)* | 61.5 | 33.0 | 5.56 | 39.5 | 53.3 | 110.7 | 47.6 | 36.5 | 47.4 | 69.4 |
| *FFN path (expert GEMM + routing/activation/combine)* | 158.4 | 126.6 | 238.8 | 17.98 | 33.7 | 161.6 | 128.8 | 248.8 | 17.93 | 34.4 |

## Decode at engine batch 1

| Component | V4 Flash 0731 decode1k_B1 | V4.1 Flash decode1k_B1 | MiMo-V2.6-Flash decode1k_B1 | Qwen3.8-Flash-Next decode1k_B1 | GLM-5.3-Flash decode1k_B1 | V4 Flash 0731 decode64k_B1 | V4.1 Flash decode64k_B1 | MiMo-V2.6-Flash decode64k_B1 | Qwen3.8-Flash-Next decode64k_B1 | GLM-5.3-Flash decode64k_B1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 8.01 | 9.14 | 6.04 | 6.85 | 6.79 | 8.06 | 9.20 | 6.24 | 6.59 | 6.83 |
| **Kernel sum** | 10.50 | 10.44 | 5.54 | 7.86 | 8.17 | 10.52 | 10.52 | 5.73 | 7.66 | 8.20 |
| Client step time in the same window | 8.19 | 9.73 | 6.49 | 7.11 | 6.96 | 8.19 | 9.41 | 6.31 | 6.69 | 6.96 |
| Attention core (softmax attention over stored KV) | 0.97 | 0.91 | 0.56 | 0.06 | 0.19 | 0.98 | 0.90 | 0.83 | 0.06 | 0.19 |
| Recurrent (linear) attention state update | – | – | – | 0.25 | 0.21 | – | – | – | 0.25 | 0.21 |
| Indexer / top-k / candidates | 0.40 | 0.27 | – | 0.12 | 0.10 | 0.42 | 0.29 | – | 0.21 | 0.16 |
| Q/K/V norm, RoPE, KV insert, attention metadata | 0.22 | 0.55 | 0.08 | 0.01 | 0.02 | 0.23 | 0.54 | 0.08 | 0.01 | 0.02 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 3.59 | 2.81 | 1.09 | 2.26 | 2.29 | 3.50 | 2.81 | 1.07 | 2.25 | 2.28 |
| GEMM input quantization | 0.16 | 0.15 | 0.17 | 0.23 | 0.36 | 0.16 | 0.15 | 0.17 | 0.23 | 0.36 |
| mHC / residual / norm | 1.54 | 1.70 | – | 1.41 | 1.89 | 1.53 | 1.72 | – | 1.40 | 1.86 |
| MoE expert GEMM | 1.29 | 1.49 | 1.44 | 1.04 | 0.92 | 1.29 | 1.49 | 1.42 | 1.03 | 0.92 |
| MoE routing, activation, combine | 0.88 | 1.20 | 0.95 | 0.80 | 0.67 | 1.07 | 1.24 | 0.93 | 0.79 | 0.66 |
| TP all-reduce | 0.87 | 0.94 | 1.06 | 1.02 | 1.10 | 0.79 | 0.96 | 1.06 | 0.76 | 1.10 |
| Other communication | 0.02 | 0.07 | 0.02 | 0.02 | 0.02 | 0.02 | 0.07 | 0.02 | 0.02 | 0.02 |
| Engram | – | 0.01 | – | – | – | – | 0.01 | – | – | – |
| Other / unclassified | 0.55 | 0.33 | 0.16 | 0.65 | 0.40 | 0.55 | 0.34 | 0.16 | 0.65 | 0.40 |
| *Attention path without projections (core + recurrent + indexer + KV work)* | 1.60 | 1.72 | 0.64 | 0.43 | 0.53 | 1.62 | 1.73 | 0.91 | 0.53 | 0.59 |
| *FFN path (expert GEMM + routing/activation/combine)* | 2.17 | 2.69 | 2.40 | 1.83 | 1.59 | 2.35 | 2.73 | 2.36 | 1.82 | 1.58 |

## Decode at engine batch 8

| Component | V4 Flash 0731 decode1k_B8 | V4.1 Flash decode1k_B8 | MiMo-V2.6-Flash decode1k_B8 | Qwen3.8-Flash-Next decode1k_B8 | GLM-5.3-Flash decode1k_B8 | V4 Flash 0731 decode64k_B8 | V4.1 Flash decode64k_B8 | MiMo-V2.6-Flash decode64k_B8 | Qwen3.8-Flash-Next decode64k_B8 | GLM-5.3-Flash decode64k_B8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 11.13 | 12.44 | 10.21 | 7.82 | 10.13 | 11.61 | 12.77 | 11.22 | 8.22 | 10.32 |
| **Kernel sum** | 14.53 | 14.50 | 9.70 | 8.97 | 12.04 | 15.03 | 14.81 | 10.71 | 9.34 | 12.25 |
| Client step time in the same window | 11.29 | 12.55 | 10.20 | 7.94 | 10.24 | 11.71 | 12.83 | 11.23 | 8.36 | 10.46 |
| Attention core (softmax attention over stored KV) | 1.01 | 1.01 | 0.80 | 0.13 | 0.20 | 1.11 | 1.01 | 1.87 | 0.13 | 0.20 |
| Recurrent (linear) attention state update | – | – | – | 0.30 | 0.30 | – | – | – | 0.30 | 0.30 |
| Indexer / top-k / candidates | 0.54 | 0.37 | – | 0.15 | 0.07 | 0.68 | 0.55 | – | 0.34 | 0.23 |
| Q/K/V norm, RoPE, KV insert, attention metadata | 0.24 | 0.58 | 0.08 | 0.01 | 0.03 | 0.25 | 0.58 | 0.08 | 0.01 | 0.03 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 3.96 | 3.12 | 1.11 | 2.55 | 2.35 | 3.97 | 3.12 | 1.10 | 2.55 | 2.35 |
| GEMM input quantization | 0.18 | 0.18 | 0.20 | 0.34 | 0.45 | 0.18 | 0.18 | 0.20 | 0.34 | 0.45 |
| mHC / residual / norm | 2.19 | 2.25 | – | 1.64 | 2.58 | 2.18 | 2.25 | – | 1.67 | 2.60 |
| MoE expert GEMM | 3.67 | 4.18 | 4.87 | 1.45 | 2.95 | 3.73 | 4.23 | 4.82 | 1.45 | 2.94 |
| MoE routing, activation, combine | 0.72 | 0.86 | 0.70 | 0.85 | 0.92 | 0.91 | 0.91 | 0.69 | 0.85 | 0.92 |
| TP all-reduce | 1.42 | 1.48 | 1.74 | 0.85 | 1.76 | 1.41 | 1.49 | 1.74 | 1.00 | 1.78 |
| Other communication | 0.02 | 0.08 | 0.02 | 0.03 | 0.02 | 0.02 | 0.08 | 0.02 | 0.03 | 0.02 |
| Engram | – | 0.02 | – | – | – | – | 0.02 | – | – | – |
| Other / unclassified | 0.58 | 0.38 | 0.17 | 0.68 | 0.42 | 0.58 | 0.39 | 0.18 | 0.69 | 0.43 |
| *Attention path without projections (core + recurrent + indexer + KV work)* | 1.79 | 1.97 | 0.88 | 0.58 | 0.59 | 2.04 | 2.14 | 1.95 | 0.77 | 0.76 |
| *FFN path (expert GEMM + routing/activation/combine)* | 4.39 | 5.04 | 5.57 | 2.30 | 3.87 | 4.65 | 5.14 | 5.51 | 2.30 | 3.86 |

Projections are listed under dense GEMM because the kernels do not say which projection they serve; on DeepSeek they are the low-rank Q and grouped O projections, on MiMo the fused QKV, O and the layer-0 FFN. They are therefore **not** included in the attention-path line above, and the complete attention path lies between that line and that line plus dense GEMM.

