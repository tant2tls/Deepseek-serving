# Retained historical measurements

**Dormant reference: use only if explicitly requested.** All rows are September 2, 2026 observations, not new results. Read the [model guide](README.md) and [shared caveats](../README.md) before comparing. Tables are generated from the linked JSONs; times are request medians in milliseconds. These are generally single retained observations, not repeat means.

## bf16kv

Role: primary. GLM batch/context/prefix base; BF16 KV, utilization 0.82; points span sessions.

[Manifest](results/bf16kv/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [batch_isl16k_c1](results/bf16kv/batch_isl16k_c1.json) | 96.3 | 856.9 | 7.1 | 8 / 8 |
| [batch_isl16k_c2](results/bf16kv/batch_isl16k_c2.json) | 122.2 | 1341.1 | 8.5 | 8 / 8 |
| [batch_isl16k_c4](results/bf16kv/batch_isl16k_c4.json) | 225.3 | 1742.3 | 10.9 | 8 / 8 |
| [batch_isl16k_c8](results/bf16kv/batch_isl16k_c8.json) | 296.9 | 2191.3 | 18.1 | 16 / 16 |
| [batch_isl16k_c16](results/bf16kv/batch_isl16k_c16.json) | 359.2 | 2684.9 | 33.6 | 32 / 32 |
| [batch_isl16k_c32](results/bf16kv/batch_isl16k_c32.json) | 406.4 | 2239.5 | 68.6 | 64 / 64 |
| [batch_isl16k_c48](results/bf16kv/batch_isl16k_c48.json) | 432.6 | 1991.5 | 100.5 | 96 / 96 |
| [batch_isl16k_c64](results/bf16kv/batch_isl16k_c64.json) | 447.1 | 2513.2 | 129.5 | 128 / 128 |
| [ctx_isl16384_c8](results/bf16kv/ctx_isl16384_c8.json) | 295.9 | 2629.3 | 16.7 | 16 / 16 |
| [ctx_isl65536_c8](results/bf16kv/ctx_isl65536_c8.json) | 104.3 | 7212.4 | 48.0 | 16 / 16 |
| [ctx_isl131072_c8](results/bf16kv/ctx_isl131072_c8.json) | 47.2 | 12891.5 | 116.7 | 16 / 16 |
| [ctx_isl260000_c8](results/bf16kv/ctx_isl260000_c8.json) | 26.5 | 38476.4 | 147.6 | 16 / 16 |
| [prefix_p64k_n1](results/bf16kv/prefix_p64k_n1.json) | 265.3 | 1664.0 | 12.6 | 64 / 64 |
| [prefix_p64k_n4](results/bf16kv/prefix_p64k_n4.json) | 365.9 | 1496.7 | 12.9 | 64 / 64 |
| [prefix_p64k_n16](results/bf16kv/prefix_p64k_n16.json) | 199.4 | 1342.1 | 20.8 | 64 / 64 |

## bf16kv-mtp-n1

Role: control. GLM MTP n1 batch; compare bf16kv.

[Manifest](results/bf16kv-mtp-n1/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [batch_isl16k_c1](results/bf16kv-mtp-n1/batch_isl16k_c1.json) | 119.7 | 721.0 | 5.1 | 8 / 8 |
| [batch_isl16k_c4](results/bf16kv-mtp-n1/batch_isl16k_c4.json) | 239.8 | 1273.7 | 11.8 | 8 / 8 |
| [batch_isl16k_c16](results/bf16kv-mtp-n1/batch_isl16k_c16.json) | 344.7 | 2829.2 | 33.0 | 32 / 32 |
| [batch_isl16k_c64](results/bf16kv-mtp-n1/batch_isl16k_c64.json) | 441.9 | 3022.5 | 123.8 | 128 / 128 |

## bf16kv-mtp-n5

Role: control. GLM MTP n5 batch; compare bf16kv.

[Manifest](results/bf16kv-mtp-n5/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [batch_isl16k_c1](results/bf16kv-mtp-n5/batch_isl16k_c1.json) | 120.8 | 733.8 | 4.9 | 8 / 8 |
| [batch_isl16k_c4](results/bf16kv-mtp-n5/batch_isl16k_c4.json) | 230.7 | 1676.4 | 11.7 | 8 / 8 |
| [batch_isl16k_c16](results/bf16kv-mtp-n5/batch_isl16k_c16.json) | 342.1 | 2119.1 | 33.8 | 32 / 32 |
| [batch_isl16k_c64](results/bf16kv-mtp-n5/batch_isl16k_c64.json) | 408.3 | 10723.9 | 101.1 | 128 / 128 |

## bf16kv-mtp-n1-context

Role: control. GLM MTP n1 context; compare bf16kv context.

[Manifest](results/bf16kv-mtp-n1-context/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [ctx_isl16384_c8](results/bf16kv-mtp-n1-context/ctx_isl16384_c8.json) | 314.8 | 2100.9 | 17.2 | 16 / 16 |
| [ctx_isl65536_c8](results/bf16kv-mtp-n1-context/ctx_isl65536_c8.json) | 97.6 | 5426.7 | 57.2 | 16 / 16 |
| [ctx_isl131072_c8](results/bf16kv-mtp-n1-context/ctx_isl131072_c8.json) | 45.3 | 10734.3 | 122.9 | 16 / 16 |
| [ctx_isl260000_c8](results/bf16kv-mtp-n1-context/ctx_isl260000_c8.json) | 22.2 | 41066.7 | 151.3 | 16 / 16 |

## bf16kv-fi618

Role: experimental. Alternate BF16 stack; experimental version-check bypass.

[Manifest](results/bf16kv-fi618/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [batch_isl16k_c1](results/bf16kv-fi618/batch_isl16k_c1.json) | 71.5 | 1015.1 | 9.5 | 8 / 8 |
| [batch_isl16k_c4](results/bf16kv-fi618/batch_isl16k_c4.json) | 147.5 | 2209.3 | 18.5 | 8 / 8 |
| [batch_isl16k_c16](results/bf16kv-fi618/batch_isl16k_c16.json) | 263.0 | 2486.3 | 50.1 | 32 / 32 |
| [batch_isl16k_c64](results/bf16kv-fi618/batch_isl16k_c64.json) | 348.8 | 2537.9 | 167.9 | 128 / 128 |

## fp8kv-fi618

Role: experimental. Alternate FP8; paired dtype control is bf16kv-fi618; numerical parity unverified.

[Manifest](results/fp8kv-fi618/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [batch_isl16k_c1](results/fp8kv-fi618/batch_isl16k_c1.json) | 68.9 | 1098.0 | 9.8 | 8 / 8 |
| [batch_isl16k_c4](results/fp8kv-fi618/batch_isl16k_c4.json) | 128.7 | 2602.6 | 20.7 | 8 / 8 |
| [batch_isl16k_c16](results/fp8kv-fi618/batch_isl16k_c16.json) | 208.8 | 2910.8 | 64.6 | 32 / 32 |
| [batch_isl16k_c64](results/fp8kv-fi618/batch_isl16k_c64.json) | 262.0 | 2958.3 | 227.2 | 128 / 128 |
| [ctx_isl16384_c8](results/fp8kv-fi618/ctx_isl16384_c8.json) | 170.9 | 2876.1 | 35.2 | 16 / 16 |
| [ctx_isl65536_c8](results/fp8kv-fi618/ctx_isl65536_c8.json) | 52.1 | 8047.5 | 106.1 | 16 / 16 |
| [ctx_isl131072_c8](results/fp8kv-fi618/ctx_isl131072_c8.json) | 31.8 | 15949.2 | 187.7 | 16 / 16 |
| [ctx_isl260000_c8](results/fp8kv-fi618/ctx_isl260000_c8.json) | 16.3 | 34809.0 | 349.1 | 16 / 16 |
| [prefix_p64k_n1](results/fp8kv-fi618/prefix_p64k_n1.json) | 234.5 | 1536.2 | 21.3 | 64 / 64 |
| [prefix_p64k_n4](results/fp8kv-fi618/prefix_p64k_n4.json) | 238.4 | 1533.6 | 21.3 | 64 / 64 |
| [prefix_p64k_n16](results/fp8kv-fi618/prefix_p64k_n16.json) | 145.1 | 2084.6 | 25.5 | 64 / 64 |

