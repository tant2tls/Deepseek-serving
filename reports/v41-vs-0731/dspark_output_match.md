### v41 dspark-fixed-k5 vs off
| Point | Repeat | Prompts | Exact match | Mean common prefix (chars) | Mean ar length |
| --- | --- | --- | --- | --- | --- |
| isl16384_osl256_c1 | repeat-1 | 16 | 1/16 | 121 | 555 |
| isl16384_osl256_c1 | repeat-2 | 16 | 1/16 | 54 | 496 |
| isl16384_osl256_c1 | repeat-3 | 16 | 1/16 | 176 | 640 |
| isl16384_osl256_c16 | repeat-1 | 64 | 1/64 | 58 | 570 |
| isl16384_osl256_c16 | repeat-2 | 64 | 0/64 | 65 | 614 |
| isl16384_osl256_c16 | repeat-3 | 64 | 1/64 | 69 | 597 |
| isl16384_osl256_c4 | repeat-1 | 16 | 0/16 | 80 | 648 |
| isl16384_osl256_c4 | repeat-2 | 16 | 0/16 | 66 | 636 |
| isl16384_osl256_c4 | repeat-3 | 16 | 1/16 | 131 | 585 |
| isl16384_osl256_c64 | repeat-1 | 256 | 7/256 | 66 | 598 |
| isl16384_osl256_c64 | repeat-2 | 256 | 4/256 | 63 | 600 |
| isl16384_osl256_c64 | repeat-3 | 256 | 7/256 | 72 | 571 |

### v41 dspark-adaptive-k5 vs off
| Point | Repeat | Prompts | Exact match | Mean common prefix (chars) | Mean ar length |
| --- | --- | --- | --- | --- | --- |
| isl16384_osl256_c1 | repeat-1 | 16 | 1/16 | 102 | 555 |
| isl16384_osl256_c1 | repeat-2 | 16 | 0/16 | 35 | 496 |
| isl16384_osl256_c1 | repeat-3 | 16 | 1/16 | 126 | 640 |
| isl16384_osl256_c16 | repeat-1 | 64 | 1/64 | 56 | 570 |
| isl16384_osl256_c16 | repeat-2 | 64 | 0/64 | 66 | 614 |
| isl16384_osl256_c16 | repeat-3 | 64 | 2/64 | 82 | 597 |
| isl16384_osl256_c4 | repeat-1 | 16 | 0/16 | 70 | 648 |
| isl16384_osl256_c4 | repeat-2 | 16 | 1/16 | 93 | 636 |
| isl16384_osl256_c4 | repeat-3 | 16 | 1/16 | 135 | 585 |
| isl16384_osl256_c64 | repeat-1 | 256 | 6/256 | 63 | 598 |
| isl16384_osl256_c64 | repeat-2 | 256 | 4/256 | 62 | 600 |
| isl16384_osl256_c64 | repeat-3 | 256 | 4/256 | 61 | 571 |

### v4-0731 dspark-fixed-k5 vs off-profidle
| Point | Repeat | Prompts | Exact match | Mean common prefix (chars) | Mean ar length |
| --- | --- | --- | --- | --- | --- |
| isl16384_osl256_c1 | repeat-1 | 16 | 2/16 | 103 | 515 |
| isl16384_osl256_c1 | repeat-2 | 16 | 2/16 | 84 | 455 |
| isl16384_osl256_c1 | repeat-3 | 16 | 3/16 | 116 | 460 |
| isl16384_osl256_c16 | repeat-1 | 64 | 17/64 | 151 | 464 |
| isl16384_osl256_c16 | repeat-2 | 64 | 14/64 | 133 | 487 |
| isl16384_osl256_c16 | repeat-3 | 64 | 16/64 | 141 | 455 |
| isl16384_osl256_c4 | repeat-1 | 16 | 2/16 | 96 | 464 |
| isl16384_osl256_c4 | repeat-2 | 16 | 5/16 | 170 | 466 |
| isl16384_osl256_c4 | repeat-3 | 16 | 4/16 | 154 | 464 |
| isl16384_osl256_c64 | repeat-1 | 256 | 42/256 | 106 | 467 |
| isl16384_osl256_c64 | repeat-2 | 256 | 51/256 | 122 | 462 |
| isl16384_osl256_c64 | repeat-3 | 256 | 54/256 | 130 | 467 |

### v4-0731 dspark-adaptive-k5 vs off-profidle
| Point | Repeat | Prompts | Exact match | Mean common prefix (chars) | Mean ar length |
| --- | --- | --- | --- | --- | --- |
| isl16384_osl256_c1 | repeat-1 | 16 | 4/16 | 142 | 515 |
| isl16384_osl256_c1 | repeat-2 | 16 | 4/16 | 143 | 455 |
| isl16384_osl256_c1 | repeat-3 | 16 | 4/16 | 144 | 460 |
| isl16384_osl256_c16 | repeat-1 | 64 | 21/64 | 176 | 464 |
| isl16384_osl256_c16 | repeat-2 | 64 | 18/64 | 162 | 487 |
| isl16384_osl256_c16 | repeat-3 | 64 | 13/64 | 120 | 455 |
| isl16384_osl256_c4 | repeat-1 | 16 | 4/16 | 146 | 464 |
| isl16384_osl256_c4 | repeat-2 | 16 | 5/16 | 173 | 466 |
| isl16384_osl256_c4 | repeat-3 | 16 | 5/16 | 173 | 464 |
| isl16384_osl256_c64 | repeat-1 | 256 | 43/256 | 111 | 467 |
| isl16384_osl256_c64 | repeat-2 | 256 | 51/256 | 120 | 462 |
| isl16384_osl256_c64 | repeat-3 | 256 | 54/256 | 126 | 467 |

