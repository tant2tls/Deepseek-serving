export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=3600

vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --language-model-only \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v41 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096