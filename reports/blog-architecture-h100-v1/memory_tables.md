# Live KV/state memory (prefix caching off)

Snapshot with B requests decoding and nothing else running. Live bytes = KV usage gauge × per-rank KV pool from the server log; an approximation when cache groups differ in block bytes. Tokens = client-counted prompt content + tokens generated at the snapshot (template tokens excluded, under 10 per request).

| Model | Input | Live sequences | Live tokens | KV usage | Live MiB per GPU | Live GiB on 8 GPUs | KiB per token per GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 Flash 0731 | 1k | 1 | 1,853 | 0.067% | 32.5 | 0.25 | 17.96 |
| V4 Flash 0731 | 1k | 8 | 12,825 | 0.520% | 252.3 | 1.97 | 20.15 |
| V4 Flash 0731 | 16k | 1 | 17,145 | 0.183% | 88.9 | 0.69 | 5.31 |
| V4 Flash 0731 | 16k | 8 | 135,363 | 1.460% | 709.2 | 5.54 | 5.37 |
| V4 Flash 0731 | 64k | 1 | 66,343 | 0.563% | 273.4 | 2.14 | 4.22 |
| V4 Flash 0731 | 64k | 8 | 528,416 | 4.483% | 2177.5 | 17.01 | 4.22 |
| V4.1 Flash | 1k | 1 | 1,727 | 0.024% | 7.1 | 0.06 | 4.23 |
| V4.1 Flash | 1k | 8 | 12,270 | 0.181% | 54.0 | 0.42 | 4.50 |
| V4.1 Flash | 16k | 1 | 17,012 | 0.113% | 33.7 | 0.26 | 2.03 |
| V4.1 Flash | 16k | 8 | 134,854 | 0.897% | 267.6 | 2.09 | 2.03 |
| V4.1 Flash | 64k | 1 | 66,218 | 0.400% | 119.4 | 0.93 | 1.85 |
| V4.1 Flash | 64k | 8 | 527,794 | 3.191% | 952.1 | 7.44 | 1.85 |
| MiMo-V2.6-Flash | 1k | 1 | 2,010 | 0.030% | 15.1 | 0.12 | 7.70 |
| MiMo-V2.6-Flash | 1k | 8 | 12,767 | 0.209% | 105.5 | 0.82 | 8.46 |
| MiMo-V2.6-Flash | 16k | 1 | 16,420 | 0.187% | 94.2 | 0.74 | 5.88 |
| MiMo-V2.6-Flash | 16k | 8 | 136,500 | 1.554% | 782.8 | 6.12 | 5.87 |
| MiMo-V2.6-Flash | 64k | 1 | 63,749 | 0.703% | 354.2 | 2.77 | 5.69 |
| MiMo-V2.6-Flash | 64k | 8 | 531,564 | 5.865% | 2953.6 | 23.07 | 5.69 |

## Reserved pools and weights (server log, per GPU)

| Model | Weights loaded | KV pool reserved | Pool capacity in tokens |
| --- | ---: | ---: | ---: |
| V4 Flash 0731 | 19.79 GiB | 47.43 GiB | 1,728,531 |
| V4.1 Flash | 36.32 GiB | 29.14 GiB | 9,092,347 |
| MiMo-V2.6-Flash | 20.10 GiB | 49.18 GiB | 6,969,972 |

Token capacities are not comparable across models (different KV formats and block accounting); compare the byte columns. None of this measures reusable-prefix capacity or maximum concurrency.

