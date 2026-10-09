# Checkpoints in this branch

The immediate objective is the completed five-model blog comparison. Qwen3.8-Flash-Next denotes the original BF16 checkpoint.

| Key | Checkpoint | Immutable revision |
| --- | --- | --- |
| `v4-0731` | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| `v41` | `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` |
| `mimo-v26` | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | `2479e2d0029eca9a34cc7e7f55a121925f81908e` |
| `qwen-38-bf16` | `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| `glm-53` | `zai-org/GLM-5.3-Flash` | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` |

Pin the tokenizer and remote code with the checkpoint. [Model setup](docs/model-setup.md) documents the actual runtime paths, including the BF16 Qwen expert backend and GLM's thinking exception. The [experiment protocol](docs/experiments.md) defines the measurements. Earlier DeepSeek/MiMo experiments are [preserved separately](docs/previous-experiments.md).

The later `spec-realtext-h100-v1` study remains planned, not measured, and is outside this branch's reproduction queue. No new prefix-cache experiments or GPU sessions are authorized by these documents.
