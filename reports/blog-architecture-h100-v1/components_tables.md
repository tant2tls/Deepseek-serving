# Component breakdown (diagnostic traces, idle-profiler launch)

Unit: ms of GPU kernel time per engine step, **mean over the 8 ranks**. A kernel sum is not wall time: kernels on different streams overlap and idle gaps are not kernels. Prefill rows are the last full 8,192-token chunk of one public-text request. Decode rows average the steps of the profiled window at the stated engine batch B. Classifier: `bench/blog_report.py`; leftovers: [components_unclassified.md](components_unclassified.md).


## Prefill: one 8,192-token chunk

| Component | V4 Flash 0731 prefill16k | V4.1 Flash prefill16k | MiMo-V2.6-Flash prefill16k | V4 Flash 0731 prefill64k | V4.1 Flash prefill64k | MiMo-V2.6-Flash prefill64k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 355.1 | 263.0 | 328.6 | 404.7 | 262.3 | 361.3 |
| **Kernel sum** | 336.6 | 244.2 | 306.6 | 395.0 | 263.1 | 350.1 |
| Attention core | 41.0 | 25.3 | 4.41 | 55.3 | 26.6 | 35.4 |
| Indexer / top-k / candidates | 6.38 | 2.39 | – | 41.1 | 15.50 | – |
| Q/K/V norm, RoPE, KV insert, attention metadata | 14.16 | 5.34 | 1.15 | 14.25 | 5.51 | 1.17 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 39.7 | 29.4 | 11.54 | 40.1 | 30.7 | 11.90 |
| GEMM input quantization | 10.90 | 3.22 | 9.21 | 11.08 | 3.23 | 9.36 |
| mHC / residual / norm | 28.4 | 17.48 | 8.57 | 28.4 | 17.40 | 8.59 |
| MoE expert GEMM | 155.3 | 124.1 | 234.1 | 158.5 | 125.4 | 244.0 |
| MoE routing, activation, combine | 3.11 | 2.51 | 4.68 | 3.12 | 3.43 | 4.78 |
| TP all-reduce | 26.7 | 21.5 | 31.7 | 31.9 | 21.8 | 33.7 |
| Other communication | – | 2.84 | – | – | 2.87 | – |
| Engram | – | 2.55 | – | – | 3.05 | – |
| Other / unclassified | 11.00 | 7.64 | 1.20 | 11.21 | 7.60 | 1.18 |
| *Attention path without projections (core + indexer + KV work)* | 61.5 | 33.0 | 5.56 | 110.7 | 47.6 | 36.5 |
| *FFN path (expert GEMM + routing/activation/combine)* | 158.4 | 126.6 | 238.8 | 161.6 | 128.8 | 248.8 |

## Decode at engine batch 1

| Component | V4 Flash 0731 decode1k_B1 | V4.1 Flash decode1k_B1 | MiMo-V2.6-Flash decode1k_B1 | V4 Flash 0731 decode64k_B1 | V4.1 Flash decode64k_B1 | MiMo-V2.6-Flash decode64k_B1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 8.01 | 9.14 | 6.04 | 8.06 | 9.20 | 6.24 |
| **Kernel sum** | 10.50 | 10.44 | 5.54 | 10.52 | 10.52 | 5.73 |
| Client step time in the same window | 8.19 | 9.73 | 6.49 | 8.19 | 9.41 | 6.31 |
| Attention core | 0.97 | 0.91 | 0.56 | 0.98 | 0.90 | 0.83 |
| Indexer / top-k / candidates | 0.40 | 0.27 | – | 0.42 | 0.29 | – |
| Q/K/V norm, RoPE, KV insert, attention metadata | 0.22 | 0.55 | 0.08 | 0.23 | 0.54 | 0.08 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 3.59 | 2.81 | 1.09 | 3.50 | 2.81 | 1.07 |
| GEMM input quantization | 0.16 | 0.15 | 0.17 | 0.16 | 0.15 | 0.17 |
| mHC / residual / norm | 1.54 | 1.70 | – | 1.53 | 1.72 | – |
| MoE expert GEMM | 1.29 | 1.49 | 1.44 | 1.29 | 1.49 | 1.42 |
| MoE routing, activation, combine | 0.88 | 1.20 | 0.95 | 1.07 | 1.24 | 0.93 |
| TP all-reduce | 0.87 | 0.94 | 1.06 | 0.79 | 0.96 | 1.06 |
| Other communication | 0.02 | 0.07 | 0.02 | 0.02 | 0.07 | 0.02 |
| Engram | – | 0.01 | – | – | 0.01 | – |
| Other / unclassified | 0.55 | 0.33 | 0.16 | 0.55 | 0.34 | 0.16 |
| *Attention path without projections (core + indexer + KV work)* | 1.60 | 1.72 | 0.64 | 1.62 | 1.73 | 0.91 |
| *FFN path (expert GEMM + routing/activation/combine)* | 2.17 | 2.69 | 2.40 | 2.35 | 2.73 | 2.36 |

## Decode at engine batch 8

| Component | V4 Flash 0731 decode1k_B8 | V4.1 Flash decode1k_B8 | MiMo-V2.6-Flash decode1k_B8 | V4 Flash 0731 decode64k_B8 | V4.1 Flash decode64k_B8 | MiMo-V2.6-Flash decode64k_B8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Step span on the rank** | 11.13 | 12.44 | 10.21 | 11.61 | 12.77 | 11.22 |
| **Kernel sum** | 14.53 | 14.50 | 9.70 | 15.03 | 14.81 | 10.71 |
| Client step time in the same window | 11.29 | 12.55 | 10.20 | 11.71 | 12.83 | 11.23 |
| Attention core | 1.01 | 1.01 | 0.80 | 1.11 | 1.01 | 1.87 |
| Indexer / top-k / candidates | 0.54 | 0.37 | – | 0.68 | 0.55 | – |
| Q/K/V norm, RoPE, KV insert, attention metadata | 0.24 | 0.58 | 0.08 | 0.25 | 0.58 | 0.08 |
| Dense GEMM (attention projections; MiMo also layer-0 FFN) | 3.96 | 3.12 | 1.11 | 3.97 | 3.12 | 1.10 |
| GEMM input quantization | 0.18 | 0.18 | 0.20 | 0.18 | 0.18 | 0.20 |
| mHC / residual / norm | 2.19 | 2.25 | – | 2.18 | 2.25 | – |
| MoE expert GEMM | 3.67 | 4.18 | 4.87 | 3.73 | 4.23 | 4.82 |
| MoE routing, activation, combine | 0.72 | 0.86 | 0.70 | 0.91 | 0.91 | 0.69 |
| TP all-reduce | 1.42 | 1.48 | 1.74 | 1.41 | 1.49 | 1.74 |
| Other communication | 0.02 | 0.08 | 0.02 | 0.02 | 0.08 | 0.02 |
| Engram | – | 0.02 | – | – | 0.02 | – |
| Other / unclassified | 0.58 | 0.38 | 0.17 | 0.58 | 0.39 | 0.18 |
| *Attention path without projections (core + indexer + KV work)* | 1.79 | 1.97 | 0.88 | 2.04 | 2.14 | 1.95 |
| *FFN path (expert GEMM + routing/activation/combine)* | 4.39 | 5.04 | 5.57 | 4.65 | 5.14 | 5.51 |

Projections are listed under dense GEMM because the kernels do not say which projection they serve; on DeepSeek they are the low-rank Q and grouped O projections, on MiMo the fused QKV, O and the layer-0 FFN. They are therefore **not** included in the attention-path line above, and the complete attention path lies between that line and that line plus dense GEMM.

