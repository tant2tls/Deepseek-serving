# The five models: architecture, technical reports and what the runtime executes

One section per checkpoint. Each section keeps two kinds of statement apart, as [target.md](../target.md#evidence-labels-and-terms) requires:

- **Architecture fact**: what the model card, technical report or checkpoint config says.
- **Executed implementation**: what vLLM `554340f3d3259e321be4c07282be7a02a5aeef83` verifiably loaded, read from the server log and traces.

A config field or a report feature does not prove that a build executes it. [Lesson 5](../teach_me/05_reading_vllm_source.md) shows how to check one claim from the report down to the GPU kernels.

Model-card and config links are pinned to the measured revision. **Technical-report, blog and recipe links are the ones the pinned model card points to; they are mutable pages, not pins.** Runtime source: [vLLM at the measured commit](https://github.com/vllm-project/vllm/tree/554340f3d3259e321be4c07282be7a02a5aeef83).

## Side by side

| | 0731 | V4.1 | MiMo | Qwen | GLM |
| --- | --- | --- | --- | --- | --- |
| Layers | 43 | 40 | 48 | 48 | 45 |
| Attention design | Sparse: 128-token window plus selected, compressed long-range state | Related layers share and compress stored state; later layers process less of the prompt | 39 sliding-window layers, 9 full-attention layers | 36 recurrent-state layers, 12 layers that look up stored state | 34 recurrent-state layers, 11 sparse-attention layers |
| Routed experts (config) | 6 of 256 | 6 of 384 | 8 of 256 | 10 of 512 | 8 of 288 |
| Shared experts (config) | 1 | 1 | 0 | 1 | 1 |
| Expert width (config) | 2,048 | 2,304 | 2,048 | 640 | 2,048 |
| Expert precision (config) | FP4 | FP4 | MXFP4 | BF16 | FP8 |
| Expert kernel that ran | `HUMMING` | `HUMMING` | `HUMMING` | Triton, unquantized | FlashInfer CUTLASS FP8 |
| Thinking in the measurements | off | off | off | off | `reasoning_effort=low` |
| How to run it | [v4-0731](../reproduce/v4-0731.md) | [v41](../reproduce/v41.md) | [mimo-v26](../reproduce/mimo-v26.md) | [qwen-38-bf16](../reproduce/qwen-38-bf16.md) | [glm-53](../reproduce/glm-53.md) |

The first six rows are architecture facts from each checkpoint's config, as used in the [article](../index.html). The kernel row is executed implementation. Top-k expert counts describe nominal activation sparsity; they are not measured HBM traffic or speed. Measured times are in the [five-model results](../reports/five-model/results.md#trace-components).

## DeepSeek V4 Flash 0731 (`v4-0731`)

| Source | Link |
| --- | --- |
| Model card, pinned | [README at `7872f01b`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md) |
| Config, pinned | [config.json at `7872f01b`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json) |
| Technical report | [arXiv 2606.19348](https://arxiv.org/abs/2606.19348) |
| vLLM recipe | [recipes.vllm.ai: DeepSeek-V4-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) |

**Architecture facts.** 43 layers. Attention reads a 128-token recent window plus selected, compressed long-range state; an indexer (the log calls it the Lightning Indexer) selects which earlier positions to read. Each MoE layer routes a token to 6 of 256 experts and always runs 1 shared expert. Experts are 2,048 wide and stored in 4-bit precision.

**Executed implementation.** Class `DeepseekV4ForCausalLM`. FP8 dense layers on the DeepGEMM path (`FlashInferFp8DeepGEMMDynamicBlockScaledKernel`). Experts on the `HUMMING` MXFP4 backend. KV state in the `fp8_ds_mla` format with an FP8 indexer cache, in 256-token blocks.

**What to keep in mind.** The search for useful past tokens is extra work: read the indexer column separately from the attention core in the trace tables. DSpark, its speculative method, was off in every October launch; the earlier study is in [speculative-decoding.md](speculative-decoding.md).

## DeepSeek V4.1 Flash (`v41`)

| Source | Link |
| --- | --- |
| Model card, pinned | [README at `dba1be0a`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/README.md) |
| Config, pinned | [config.json at `dba1be0a`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json) |
| Technical report | [DeepSeek_V41_Tech_Report.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) |
| vLLM recipe | [recipes.vllm.ai: DeepSeek-V4.1-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) |

**Architecture facts.** 40 layers, described by the report as a causal encoder-decoder: 20 encoder layers and 20 decoder layers. The config gives `kv_source_layer_ids` `[2, 8, 14, 20]` and `sliding_window` 128: related layers reuse stored attention state, and later layers can process less of the prompt. Each MoE layer routes to 6 of 384 experts plus 1 shared expert; experts are 2,304 wide and 4-bit. The model also carries Engram tables.

**Executed implementation.** Class `DeepseekV41ForCausalLM`. In prefill steps of at least 768 tokens, layers 21 to 39 run on each request's last 128 tokens only, so **21 of 40 layers process the whole chunk** and 19 process a window. This is one layer more than the report's 20, and the skip is off for shorter steps and for requests asking for prompt logprobs. All 40 layers run in ordinary decode. Engram tables are offloaded to pinned host memory (11.80 GiB per rank). Experts on `HUMMING`; KV in `fp8_ds_mla`, 64-token blocks.

**What to keep in mind.** The September build did not execute the prefill skip, so V4.1's old prefill numbers are not comparable. The full check is [v41_prefill_check.md](../reports/blog-architecture-h100-v1/v41_prefill_check.md). Its decode step is split over several GPU streams, which changes how a trace must be cut ([lesson 3](../teach_me/03_traces_and_components.md)).

## MiMo-V2.6-Flash-MOPD (`mimo-v26`)

| Source | Link |
| --- | --- |
| Model card, pinned | [README at `2479e2d0`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/blob/2479e2d0029eca9a34cc7e7f55a121925f81908e/README.md) |
| Config, pinned | [config.json at `2479e2d0`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/blob/2479e2d0029eca9a34cc7e7f55a121925f81908e/config.json) |
| Technical report | [MiMo_V2_6_technical_report.pdf](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/MiMo_V2_6_technical_report.pdf) |
| Product page | [mimo.xiaomi.com/mimo-v2-6](https://mimo.xiaomi.com/mimo-v2-6) |

**Architecture facts.** 48 layers. 39 layers attend to a sliding window of 128 keys with a learned per-head sink; 9 layers (0, 5, 11, 17, 23, 29, 35, 41, 47) use full causal attention. Each MoE layer routes to 8 of 256 experts, with no shared expert; experts are 2,048 wide and MXFP4. Layer 0 uses a dense 16,384-wide FFN instead of MoE.

**Executed implementation.** Class `MiMoV2OmniForCausalLM`: the config has a vision tower, and the wrapper runs the same text decoder with multimodal inputs disabled. Attention backend `FLASH_ATTN_DIFFKV` (FlashAttention 3, upgraded to 4 for sinks). FP8 dense layers on DeepGEMM. Experts on `HUMMING` in October; the September build selected Marlin. KV state is BF16.

**What to keep in mind.** There is no learned search step, but the nine full-attention layers do work that grows with context. A source-level walkthrough on the September build is [mimo-v2.6-inference.md](mimo-v2.6-inference.md); re-check its backends against the October log lines on the [model page](../reproduce/mimo-v26.md).

## Qwen3.8-Flash-Next (`qwen-38-bf16`)

| Source | Link |
| --- | --- |
| Model card, pinned | [README at `de4b8e4d`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md) |
| Config, pinned | [config.json at `de4b8e4d`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) |
| Technical report | [tech_report.pdf](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) |
| Blog post | [qwen.ai: Qwen3.8-Flash-Next](https://qwen.ai/blog?id=qwen3.8-flash-next) |
| vLLM recipe | [recipes.vllm.ai: Qwen3.8-Flash-Next](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) |

**Architecture facts.** 48 layers. 36 layers update a fixed-size recurrent state in place instead of keeping a record for every past token; 12 layers select information from stored state that grows with context. Each MoE layer routes to 10 of 512 experts plus 1 shared expert. Experts are only 640 wide and stored in BF16.

**Executed implementation.** Class `Qwen4ExpForConditionalGeneration`, loaded with `dtype=torch.bfloat16` and `quantization=None`. Unquantized experts run in Triton's `fused_moe_kernel`. The n-gram (PLE) embedding is BF16 in pinned host memory, not on the GPUs. Recurrent layers use the FlashInfer GDN prefill kernel. The KV pool has several cache groups: recurrent-state groups plus one attention group with 400-token blocks.

**What to keep in mind.** This is the **original BF16 checkpoint**, measured on the second node. It selects more experts per token than MiMo, but each is much smaller, so more experts need not mean more expert work. Compare its throughput with the other four and keep its latency separate ([comparison limits](experiments.md#two-physical-nodes)).

## GLM-5.3-Flash (`glm-53`)

| Source | Link |
| --- | --- |
| Model card, pinned | [README at `eb9eb208`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/README.md) |
| Config, pinned | [config.json at `eb9eb208`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json) |
| Technical report | [GLM-5, arXiv 2602.15763](https://arxiv.org/abs/2602.15763) |
| Blog post | [z.ai: GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash) |
| Vendor guide | [docs.z.ai: GLM-5.3-Flash](https://docs.z.ai/guides/llm/glm-5.3-flash) |

**Architecture facts.** 45 layers. 34 layers update a fixed-size recurrent state; 11 layers use sparse attention over past tokens. Each MoE layer routes to 8 of 288 experts plus 1 shared expert; experts are 2,048 wide and FP8. The first three FFNs are dense.

**Executed implementation.** Class `Glm5NextForConditionalGeneration`. FP8 dense layers on DeepGEMM and FP8 experts on `FLASHINFER_CUTLASS`. The sparse-attention layers use an indexer with 64-token blocks; recurrent state and attention share a 640-token page, of which about a fifth is padding.

**What to keep in mind.** The chat template has no thinking-off switch, so GLM runs with `reasoning_effort=low` and is always reported with that deviation. Its experts have the same hidden and intermediate dimensions as MiMo's but a different precision and kernel, and its expert time in a prefill chunk is about a tenth of MiMo's. That observation motivates a controlled backend test; it does not prove what changing MiMo's precision would do.

## Related reading

| Topic | Where |
| --- | --- |
| Speculative decoding in the DeepSeek checkpoints (not used in October) | [speculative-decoding.md](speculative-decoding.md), [DSpark paper](https://arxiv.org/html/2607.05147v1) |
| Presentation reference for the article (style, not evidence) | [TraceLab post](https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/) |
| Where each measured quantity is defined in vLLM | [Lesson 5](../teach_me/05_reading_vllm_source.md) |

## Adding a model

Add a section with the same four parts: pinned sources, architecture facts, executed implementation, what to keep in mind. Fill "executed implementation" only from a server log and trace of your own launch on the stated build. Then add its [reproduce page](../reproduce/README.md#adding-a-model-to-this-folder).
