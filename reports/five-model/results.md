# Five-model measured results

Build: vLLM `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`). 0731, V4.1, MiMo and GLM are first-node measurements; Qwen is second-node. See [protocol and limits](../../docs/experiments.md). All values below are generated from the retained CSVs.

## Output throughput

Output tokens/s, mean ± sample SD across three launch-separated blocks. Qwen uses the original BF16 checkpoint. This is a throughput-only comparison across nodes; values are not rescaled by the control. Differences below about 2% across nodes remain unresolved.

| Input | Clients | 0731 | V4.1 | MiMo | Qwen | GLM |
| --- | --- | --- | --- | --- | --- | --- |
| 1K | 1 | 118.1 ± 0.2 | 106.0 ± 0.1 | 160.6 ± 0.4 | 154.7 ± 2.1 | 137.3 ± 1.8 |
| 1K | 8 | 598.7 ± 40.5 | 562.9 ± 12.6 | 687.7 ± 2.7 | 793.0 ± 2.7 | 705.8 ± 1.7 |
| 16K | 1 | 93.4 ± 0.3 | 89.3 ± 0.2 | 114.3 ± 0.7 | 125.4 ± 0.2 | 109.4 ± 6.9 |
| 16K | 8 | 237.5 ± 2.2 | 278.9 ± 1.2 | 250.9 ± 1.6 | 385.5 ± 2.5 | 322.3 ± 6.1 |
| 64K | 1 | 48.5 ± 0.4 | 55.9 ± 0.4 | 55.4 ± 0.1 | 77.5 ± 0.7 | 66.4 ± 2.1 |
| 64K | 8 | 71.5 ± 1.1 | 100.5 ± 1.1 | 78.5 ± 1.0 | 142.5 ± 0.7 | 115.7 ± 0.2 |

Qwen leads at five of the six measured points; MiMo leads at 1K/c1. This ranks the saved deployments and workloads, not model quality or architecture in isolation.

## MiMo node control

One block on the second node against the mean of three on the first. TTFT and TPOT columns are ratios of block-level per-request medians, not pooled latency percentiles.

| Input | Clients | Throughput ratio | TTFT ratio | TPOT ratio |
| --- | --- | --- | --- | --- |
| 1K | 1 | 0.989 | 1.024 | 1.002 |
| 1K | 8 | 0.988 | 1.064 | 1.002 |
| 16K | 1 | 0.986 | 1.070 | 0.995 |
| 16K | 8 | 0.995 | 1.085 | 0.878 |
| 64K | 1 | 0.983 | 1.030 | 0.981 |
| 64K | 8 | 0.987 | 1.302 | 0.863 |

The control supports cross-node throughput comparison, but not TTFT or eight-client TPOT rankings.

## Latency on the first node

Each cell is mean ± sample SD of the three block medians, in milliseconds.

### TTFT

| Input | Clients | 0731 | V4.1 | MiMo | GLM |
| --- | --- | --- | --- | --- | --- |
| 1K | 1 | 208 ± 3 | 121 ± 1 | 114 ± 2 | 166 ± 1 |
| 1K | 8 | 487 ± 1 | 397 ± 74 | 409 ± 10 | 330 ± 9 |
| 16K | 1 | 759 ± 8 | 559 ± 6 | 762 ± 8 | 527 ± 10 |
| 16K | 8 | 2866 ± 40 | 1815 ± 26 | 2878 ± 86 | 1684 ± 124 |
| 64K | 1 | 3282 ± 39 | 2278 ± 35 | 3142 ± 6 | 2080 ± 12 |
| 64K | 8 | 8950 ± 371 | 5807 ± 240 | 8406 ± 246 | 5410 ± 218 |

### TPOT

| Input | Clients | 0731 | V4.1 | MiMo | GLM |
| --- | --- | --- | --- | --- | --- |
| 1K | 1 | 7.68 ± 0.00 | 8.99 ± 0.00 | 5.79 ± 0.01 | 6.59 ± 0.00 |
| 1K | 8 | 10.81 ± 0.00 | 12.32 ± 0.01 | 9.93 ± 0.00 | 9.91 ± 0.01 |
| 16K | 1 | 7.76 ± 0.00 | 9.04 ± 0.01 | 5.86 ± 0.00 | 6.65 ± 0.01 |
| 16K | 8 | 21.98 ± 0.15 | 20.82 ± 0.26 | 20.16 ± 0.09 | 17.02 ± 0.40 |
| 64K | 1 | 7.81 ± 0.00 | 9.01 ± 0.00 | 6.01 ± 0.01 | 6.62 ± 0.01 |
| 64K | 8 | 73.17 ± 1.68 | 54.90 ± 0.12 | 68.26 ± 0.49 | 46.71 ± 0.67 |

## Qwen latency on the second node

Reported separately. Short-prompt TTFT can switch between two levels within one run. At c8, TPOT includes scheduling interference from other requests and is not an isolated decode step.

| Input | Clients | TTFT ms | TPOT ms |
| --- | --- | --- | --- |
| 1K | 1 | 147 ± 37 | 5.85 ± 0.00 |
| 1K | 8 | 379 ± 2 | 8.56 ± 0.00 |
| 16K | 1 | 537 ± 11 | 5.85 ± 0.01 |
| 16K | 8 | 1681 ± 47 | 13.08 ± 0.35 |
| 64K | 1 | 1808 ± 5 | 5.77 ± 0.00 |
| 64K | 8 | 6240 ± 256 | 30.31 ± 0.88 |

## Live state memory at 64K, B=1

Approximation from usage gauge × per-rank pool. GiB and KiB are binary units. Eight-rank bytes include replication and allocation rounding; these are not measured maximum request capacities.

| Model | Node | Live KiB/token/GPU | Live GiB/eight GPUs | Reserved pool GiB/GPU |
| --- | --- | --- | --- | --- |
| 0731 | First | 4.22 | 2.14 | 47.43 |
| V4.1 | First | 1.85 | 0.93 | 29.14 |
| MiMo | First | 5.69 | 2.77 | 49.18 |
| Qwen | Second | 13.14 | 6.97 | 37.37 |
| GLM | First | 12.01 | 5.77 | 28.73 |

## Trace components

GPU kernel milliseconds per engine step, averaged across all eight ranks. Prefill uses the last full 8192-token chunk. Kernels can overlap; sums are not elapsed latency. Dense GEMM mixes projection work and must not be counted as an exact attention total. Qwen all-reduce includes profiler-related rank waiting; it is not isolated communication cost.

### prefill16k

| Model | Attention core ms | Indexer ms | Recurrent ms | KV preparation ms | Dense GEMM ms | Experts ms | Routing/combine ms | All-reduce ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0731 | 40.99 | 6.38 | 0.00 | 14.16 | 39.71 | 155.27 | 3.11 | 26.73 |
| V4.1 | 25.26 | 2.39 | 0.00 | 5.34 | 29.40 | 124.12 | 2.51 | 21.47 |
| MiMo | 4.41 | 0.00 | 0.00 | 1.15 | 11.54 | 234.10 | 4.68 | 31.70 |
| Qwen | 29.48 | 4.85 | 5.10 | 0.06 | 32.07 | 18.69 | 10.92 | 75.16 |
| GLM | 35.15 | 2.03 | 15.65 | 0.47 | 27.01 | 24.40 | 9.27 | 32.07 |

### prefill64k

| Model | Attention core ms | Indexer ms | Recurrent ms | KV preparation ms | Dense GEMM ms | Experts ms | Routing/combine ms | All-reduce ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0731 | 55.29 | 41.13 | 0.00 | 14.25 | 40.10 | 158.48 | 3.12 | 31.91 |
| V4.1 | 26.59 | 15.50 | 0.00 | 5.51 | 30.70 | 125.42 | 3.43 | 21.85 |
| MiMo | 35.36 | 0.00 | 0.00 | 1.17 | 11.90 | 243.99 | 4.78 | 33.73 |
| Qwen | 30.03 | 12.16 | 5.10 | 0.06 | 32.07 | 18.74 | 10.93 | 61.22 |
| GLM | 40.11 | 13.05 | 15.73 | 0.47 | 27.14 | 25.06 | 9.37 | 34.50 |

### decode1k_B1

| Model | Attention core ms | Indexer ms | Recurrent ms | KV preparation ms | Dense GEMM ms | Experts ms | Routing/combine ms | All-reduce ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0731 | 0.97 | 0.40 | 0.00 | 0.22 | 3.59 | 1.29 | 0.88 | 0.87 |
| V4.1 | 0.91 | 0.27 | 0.00 | 0.55 | 2.81 | 1.49 | 1.20 | 0.94 |
| MiMo | 0.56 | 0.00 | 0.00 | 0.08 | 1.09 | 1.44 | 0.95 | 1.06 |
| Qwen | 0.06 | 0.12 | 0.25 | 0.01 | 2.03 | 0.80 | 1.06 | 2.89 |
| GLM | 0.19 | 0.10 | 0.21 | 0.02 | 2.29 | 0.92 | 0.67 | 1.10 |

### decode1k_B8

| Model | Attention core ms | Indexer ms | Recurrent ms | KV preparation ms | Dense GEMM ms | Experts ms | Routing/combine ms | All-reduce ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0731 | 1.01 | 0.54 | 0.00 | 0.24 | 3.96 | 3.67 | 0.72 | 1.42 |
| V4.1 | 1.01 | 0.37 | 0.00 | 0.58 | 3.12 | 4.18 | 0.86 | 1.48 |
| MiMo | 0.80 | 0.00 | 0.00 | 0.08 | 1.11 | 4.87 | 0.70 | 1.74 |
| Qwen | 0.13 | 0.14 | 0.30 | 0.01 | 2.31 | 2.25 | 1.05 | 1.75 |
| GLM | 0.20 | 0.07 | 0.30 | 0.03 | 2.35 | 2.95 | 0.92 | 1.76 |

### decode64k_B1

| Model | Attention core ms | Indexer ms | Recurrent ms | KV preparation ms | Dense GEMM ms | Experts ms | Routing/combine ms | All-reduce ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0731 | 0.98 | 0.42 | 0.00 | 0.23 | 3.50 | 1.29 | 1.07 | 0.79 |
| V4.1 | 0.90 | 0.29 | 0.00 | 0.54 | 2.81 | 1.49 | 1.24 | 0.96 |
| MiMo | 0.83 | 0.00 | 0.00 | 0.08 | 1.07 | 1.42 | 0.93 | 1.06 |
| Qwen | 0.06 | 0.21 | 0.25 | 0.01 | 2.02 | 0.80 | 1.05 | 3.00 |
| GLM | 0.19 | 0.16 | 0.21 | 0.02 | 2.28 | 0.92 | 0.66 | 1.10 |

### decode64k_B8

| Model | Attention core ms | Indexer ms | Recurrent ms | KV preparation ms | Dense GEMM ms | Experts ms | Routing/combine ms | All-reduce ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0731 | 1.11 | 0.68 | 0.00 | 0.25 | 3.97 | 3.73 | 0.91 | 1.41 |
| V4.1 | 1.01 | 0.55 | 0.00 | 0.58 | 3.12 | 4.23 | 0.91 | 1.49 |
| MiMo | 1.87 | 0.00 | 0.00 | 0.08 | 1.10 | 4.82 | 0.69 | 1.74 |
| Qwen | 0.13 | 0.35 | 0.30 | 0.01 | 2.31 | 2.34 | 1.04 | 1.32 |
| GLM | 0.20 | 0.23 | 0.30 | 0.03 | 2.35 | 2.94 | 0.92 | 1.78 |

HBM byte counters and expert-routing statistics were not collected. The traces identify executed kernels and scaling patterns; they do not isolate causal optimization gains.

## Individual timing blocks

No slow first block is dropped. Full precision and additional metrics are in [serving.csv](serving.csv).

| Model | Input | Clients | Block 1 tok/s | Block 2 tok/s | Block 3 tok/s | Median tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 0731 | 1K | 1 | 118.231 | 118.063 | 117.890 | 118.063 |
| 0731 | 1K | 8 | 551.919 | 621.799 | 622.502 | 621.799 |
| 0731 | 16K | 1 | 93.585 | 93.064 | 93.587 | 93.585 |
| 0731 | 16K | 8 | 238.742 | 234.974 | 238.665 | 238.665 |
| 0731 | 64K | 1 | 48.329 | 48.937 | 48.299 | 48.329 |
| 0731 | 64K | 8 | 70.299 | 71.645 | 72.560 | 71.645 |
| V4.1 | 1K | 1 | 105.985 | 106.049 | 105.938 | 105.985 |
| V4.1 | 1K | 8 | 552.767 | 558.861 | 576.930 | 558.861 |
| V4.1 | 16K | 1 | 89.476 | 89.122 | 89.232 | 89.232 |
| V4.1 | 16K | 8 | 279.426 | 277.495 | 279.727 | 279.426 |
| V4.1 | 64K | 1 | 55.692 | 56.363 | 55.713 | 55.713 |
| V4.1 | 64K | 8 | 99.791 | 101.717 | 99.885 | 99.885 |
| MiMo | 1K | 1 | 160.739 | 160.200 | 160.989 | 160.739 |
| MiMo | 1K | 8 | 689.193 | 684.581 | 689.276 | 689.193 |
| MiMo | 16K | 1 | 114.777 | 113.454 | 114.675 | 114.675 |
| MiMo | 16K | 8 | 251.777 | 249.099 | 251.794 | 251.777 |
| MiMo | 64K | 1 | 55.460 | 55.301 | 55.464 | 55.460 |
| MiMo | 64K | 8 | 77.978 | 79.618 | 77.915 | 77.978 |
| Qwen | 1K | 1 | 152.874 | 157.014 | 154.354 | 154.354 |
| Qwen | 1K | 8 | 790.018 | 793.681 | 795.377 | 793.681 |
| Qwen | 16K | 1 | 125.206 | 125.435 | 125.621 | 125.435 |
| Qwen | 16K | 8 | 385.764 | 387.796 | 382.823 | 385.764 |
| Qwen | 64K | 1 | 76.748 | 77.888 | 78.002 | 77.888 |
| Qwen | 64K | 8 | 143.043 | 141.702 | 142.892 | 142.892 |
| GLM | 1K | 1 | 135.239 | 138.393 | 138.311 | 138.311 |
| GLM | 1K | 8 | 706.685 | 706.962 | 703.872 | 706.685 |
| GLM | 16K | 1 | 101.408 | 113.675 | 113.028 | 113.028 |
| GLM | 16K | 8 | 315.300 | 326.306 | 325.311 | 325.311 |
| GLM | 64K | 1 | 64.044 | 67.461 | 67.727 | 67.461 |
| GLM | 64K | 8 | 115.743 | 115.480 | 115.837 | 115.743 |

## Evidence

- First node: [timing](../blog-architecture-h100-v1/serving.csv), [components](../blog-architecture-h100-v1/components.csv), [memory](../blog-architecture-h100-v1/memory.csv), [raw evidence](../blog-architecture-h100-v1/data/).
- Second node: [timing](../qwen-bf16-h100-v1/serving.csv), [components](../qwen-bf16-h100-v1/components.csv), [memory](../qwen-bf16-h100-v1/memory.csv), [raw evidence](../qwen-bf16-h100-v1/data/).
- [Provenance and sanitization](../../docs/provenance.md); [earlier experiments](../../docs/previous-experiments.md).
