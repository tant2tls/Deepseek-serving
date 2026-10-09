# Qwen checkpoint

Use **Qwen3.8-Flash-Next**, the original BF16 `Qwen/Qwen3.8-Flash-Next` checkpoint, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, key `qwen-38-bf16`.

Its completed study is `qwen-bf16-h100-v1`, measured on October 8, 2026, on the second 8×H100 node. The server log verifies `dtype=torch.bfloat16`, `quantization=None` and `TRITON Unquantized MoE backend`. Keep native precision; do not convert or substitute weights. See [setup](model-setup.md) and [comparison limits](experiments.md).
