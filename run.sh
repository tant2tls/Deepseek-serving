export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# /root is a gocryptfs FUSE mount; concurrent TP-rank Triton compiles race on
# rename visibility there (FileNotFoundError on kernel.ttir/ptx). Use local disk.
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "$TRITON_CACHE_DIR"
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