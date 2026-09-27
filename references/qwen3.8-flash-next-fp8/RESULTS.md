# Retained historical measurements

**Dormant reference: use only if explicitly requested.** All rows are September 2, 2026 observations, not new results. Read the [model guide](README.md) and [shared caveats](../README.md) before comparing. Tables are generated from the linked JSONs; times are request medians in milliseconds. These are generally single retained observations, not repeat means.

## base-util082

Role: primary. Corrected main batch only; warm compile cache, utilization 0.82.

[Manifest](results/base-util082/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [batch_isl16k_c1](results/base-util082/batch_isl16k_c1.json) | 106.2 | 598.0 | 7.0 | 8 / 8 |
| [batch_isl16k_c4](results/base-util082/batch_isl16k_c4.json) | 256.8 | 1496.2 | 9.6 | 8 / 8 |
| [batch_isl16k_c16](results/base-util082/batch_isl16k_c16.json) | 420.2 | 2917.3 | 26.3 | 32 / 32 |
| [batch_isl16k_c64](results/base-util082/batch_isl16k_c64.json) | 517.7 | 2939.9 | 110.1 | 128 / 128 |

## base

Role: qualified. Context/prefix usable with smaller-pool caveat; batch superseded by base-util082.

[Manifest](results/base/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [batch_isl16k_c1](results/base/batch_isl16k_c1.json) | 107.7 | 602.4 | 7.0 | 8 / 8 |
| [batch_isl16k_c4](results/base/batch_isl16k_c4.json) | 162.2 | 2250.1 | 9.8 | 8 / 8 |
| [batch_isl16k_c16](results/base/batch_isl16k_c16.json) | 371.0 | 3168.0 | 32.2 | 32 / 32 |
| [batch_isl16k_c64](results/base/batch_isl16k_c64.json) | 517.8 | 2942.0 | 110.1 | 128 / 128 |
| [ctx_isl16384_c8](results/base/ctx_isl16384_c8.json) | 322.7 | 2329.9 | 15.6 | 16 / 16 |
| [ctx_isl65536_c8](results/base/ctx_isl65536_c8.json) | 114.5 | 9287.2 | 33.6 | 16 / 16 |
| [ctx_isl131072_c8](results/base/ctx_isl131072_c8.json) | 56.5 | 19835.6 | 63.5 | 16 / 16 |
| [ctx_isl260000_c8](results/base/ctx_isl260000_c8.json) | 25.1 | 21338.2 | 231.3 | 16 / 16 |
| [prefix_p64k_n1](results/base/prefix_p64k_n1.json) | 453.9 | 1470.6 | 10.8 | 64 / 64 |
| [prefix_p64k_n4](results/base/prefix_p64k_n4.json) | 398.1 | 1207.5 | 12.0 | 64 / 64 |
| [prefix_p64k_n16](results/base/prefix_p64k_n16.json) | 262.2 | 2999.3 | 12.1 | 64 / 64 |

## mtp-n1

Role: qualified. Compare original base only with startup/pool caveats; not corrected base-util082.

[Manifest](results/mtp-n1/manifest.txt)

| Point | Output tok/s | TTFT ms | TPOT ms | Completed / planned |
| --- | ---: | ---: | ---: | ---: |
| [batch_isl16k_c1](results/mtp-n1/batch_isl16k_c1.json) | 125.9 | 657.9 | 5.6 | 8 / 8 |
| [batch_isl16k_c4](results/mtp-n1/batch_isl16k_c4.json) | 149.3 | 2305.7 | 16.5 | 8 / 8 |
| [batch_isl16k_c16](results/mtp-n1/batch_isl16k_c16.json) | 322.2 | 1954.3 | 37.0 | 32 / 32 |
| [batch_isl16k_c64](results/mtp-n1/batch_isl16k_c64.json) | 481.5 | 3887.8 | 109.8 | 128 / 128 |

