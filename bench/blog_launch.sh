#!/usr/bin/env bash
# Start a blog-protocol chain detached from the calling shell: bench/blog_launch.sh <name> <steps...>
# BLOG_STUDY selects the study (default blog-architecture-h100-v1; BF16 Qwen: qwen-bf16-h100-v1).
# Output goes to results/<study>/_logs/<name>.out; progress to _logs/chain.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_VENV=${VLLM_VENV:-/root/vllm-latest} HF_HOME=${HF_HOME:-/workspace/hf}
name=$1; shift
export BLOG_STUDY=${BLOG_STUDY:-blog-architecture-h100-v1}
out=results/$BLOG_STUDY/_logs/$name.out
mkdir -p "$(dirname "$out")"
setsid nohup "$VLLM_VENV/bin/python" bench/blog_study.py chain "$@" > "$out" 2>&1 < /dev/null &
echo "started pid $! -> $out"
