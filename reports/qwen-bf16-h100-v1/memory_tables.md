# Live KV/state memory (prefix caching off)

Snapshot with B requests decoding and nothing else running. Live bytes = KV usage gauge × per-rank KV pool from the server log; an approximation when cache groups differ in block bytes. Tokens = client-counted prompt content + tokens generated at the snapshot (template tokens excluded, under 10 per request).

| Model | Input | Live sequences | Live tokens | KV usage | Live MiB per GPU | Live GiB on 8 GPUs | KiB per token per GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next (BF16) | 1k | 1 | 3,427 | 0.182% | 69.8 | 0.54 | 20.84 |
| Qwen3.8-Flash-Next (BF16) | 1k | 8 | 20,420 | 1.276% | 488.3 | 3.81 | 24.49 |
| Qwen3.8-Flash-Next (BF16) | 16k | 1 | 18,697 | 0.690% | 264.1 | 2.06 | 14.46 |
| Qwen3.8-Flash-Next (BF16) | 16k | 8 | 149,031 | 5.469% | 2092.7 | 16.35 | 14.38 |
| Qwen3.8-Flash-Next (BF16) | 64k | 1 | 69,481 | 2.331% | 891.9 | 6.97 | 13.14 |
| Qwen3.8-Flash-Next (BF16) | 64k | 8 | 563,329 | 18.958% | 7254.8 | 56.68 | 13.19 |
| MiMo-V2.6-Flash | 1k | 1 | 3,487 | 0.046% | 23.2 | 0.18 | 6.81 |
| MiMo-V2.6-Flash | 1k | 8 | 16,409 | 0.267% | 134.5 | 1.05 | 8.39 |
| MiMo-V2.6-Flash | 16k | 1 | 17,657 | 0.201% | 101.1 | 0.79 | 5.86 |
| MiMo-V2.6-Flash | 16k | 8 | 140,731 | 1.610% | 811.0 | 6.34 | 5.90 |
| MiMo-V2.6-Flash | 64k | 1 | 64,302 | 0.717% | 361.3 | 2.82 | 5.75 |
| MiMo-V2.6-Flash | 64k | 8 | 534,380 | 5.915% | 2980.0 | 23.28 | 5.71 |

## Reserved pools and weights (server log, per GPU)

| Model | Weights loaded | KV pool reserved | Pool capacity in tokens |
| --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next (BF16) | 31.42 GiB | 37.37 GiB | 3,046,184 |
| MiMo-V2.6-Flash | 20.10 GiB | 49.20 GiB | 6,972,733 |

Token capacities are not comparable across models (different KV formats and block accounting); compare the byte columns. None of this measures reusable-prefix capacity or maximum concurrency.

