#!/usr/bin/env bash
# Launch one matched DeepSeek server for the 0731 vs V4.1 study.
# Usage: bench/serve.sh <model-key: v41|v4-0731> <config-id: off|off-prefix|...> <log-path> [extra vllm args]
# Credentials come from the environment (HF_TOKEN); never hardcode them here.
set -euo pipefail

MODEL_KEY=$1; CONFIG_ID=$2; LOG=$3; shift 3

case "$MODEL_KEY" in
  v41)
    MODEL=deepseek-ai/DeepSeek-V4.1-Flash
    REVISION=dba1be0a40aa45a94ad051997016db3960a90277
    PARSER=deepseek_v41 ;;
  v4-0731)
    MODEL=deepseek-ai/DeepSeek-V4-Flash-0731
    REVISION=7872f01b1d1fe23eabc4c98b48bffcef5a386062
    PARSER=deepseek_v4 ;;
  *) echo "unknown model key $MODEL_KEY" >&2; exit 2 ;;
esac

# Prefix caching is off for batch/context baselines; the prefix arm turns it on.
case "$CONFIG_ID" in
  off)        PREFIX_FLAG=--no-enable-prefix-caching ;;
  off-prefix) PREFIX_FLAG=--enable-prefix-caching ;;
  # Same as `off`, plus an idle torch profiler (active only between
  # /start_profile and /stop_profile). Timing runs here need a control rerun.
  off-profidle)
    PREFIX_FLAG=--no-enable-prefix-caching
    PROFILE_DIR=${PROFILE_DIR:?set PROFILE_DIR to an absolute trace directory}
    mkdir -p "$PROFILE_DIR"
    set -- --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}" "$@" ;;
  # DSpark arms (docs/speculative-decoding.md): prefix caching off to match the
  # `ar` concurrency baseline. The draft ships inside the target checkpoint but
  # its revision is resolved separately (default None = main), so pin it here.
  dspark-fixed-k5|dspark-adaptive-k5)
    PREFIX_FLAG=--no-enable-prefix-caching
    [ "$CONFIG_ID" = dspark-adaptive-k5 ] && ADAPTIVE=true || ADAPTIVE=false
    set -- --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"revision\":\"$REVISION\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"enable_adaptive_verification\":$ADAPTIVE}" "$@" ;;
  *) echo "unknown config $CONFIG_ID" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# /root is a gocryptfs FUSE mount; concurrent TP-rank Triton compiles race there.
export TRITON_CACHE_DIR=/tmp/triton_cache
# Same reason for FlashInfer's JIT modules (0731 JIT-builds fp8_blockscale_gemm_90).
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_ws
# Enables /reset_prefix_cache for cache-state control between runs.
export VLLM_SERVER_DEV_MODE=1
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$(dirname "$LOG")"

source /root/vllm/bin/activate
set -x
exec vllm serve "$MODEL" \
  --revision "$REVISION" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --language-model-only \
  --tokenizer-mode "$PARSER" --reasoning-parser "$PARSER" \
  --enable-auto-tool-choice --tool-call-parser "$PARSER" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  "$PREFIX_FLAG" \
  "$@" 2>&1 | tee "$LOG"
