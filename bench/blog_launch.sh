#!/usr/bin/env bash
# Start a blog-study chain detached from the calling shell: bench/blog_launch.sh <name> <steps...>
# Output goes to results/blog-architecture-h100-v1/_logs/<name>.out; progress to _logs/chain.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_VENV=${VLLM_VENV:-/root/vllm-latest} HF_HOME=${HF_HOME:-/workspace/hf}
name=$1; shift
out=results/blog-architecture-h100-v1/_logs/$name.out
setsid nohup "$VLLM_VENV/bin/python" bench/blog_study.py chain "$@" > "$out" 2>&1 < /dev/null &
echo "started pid $! -> $out"
