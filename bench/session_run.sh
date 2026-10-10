#!/usr/bin/env bash
# Run the stages of the 15-hour rerun session one after another, unattended (target.md#the-15-hour-session).
#   bench/session_run.sh diagfix routing timing   # after the diagnostics chain; waits for any running chain first
# Stages:
#   diagfix  relaunch the diagnostics of a model whose captures are incomplete (once).
#   routing  expert-routing statistics, one `off-routed` launch per model. Gate: the first model must
#            produce routing.json within 20 minutes, and the stage stops at 75 minutes in any case.
#   timing   three blocks, each one plain launch per model, in the frozen rotating order.
# Every chain is started through bench/blog_launch.sh with --publish, so each finished step is pushed.
# HBM counters (stage 4) have no step here: the node refuses GPU performance counters to the
# container, which is recorded in results/<study>/_logs/hbm_counter_attempt.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
export BLOG_STUDY=five-model-rerun-h100-v1
export VLLM_VENV=${VLLM_VENV:-/workspace/vllm-rerun} HF_HOME=${HF_HOME:-/workspace/hf}
S=results/$BLOG_STUDY
say() { echo "[$(date -u +%H:%M:%S)] RUNNER $*" | tee -a "$S/_logs/runner.log"; }
chain_pid() { pgrep -f "bench/blog_study.py chain" | head -1; }
wait_chain() {  # wait_chain [deadline epoch]: returns 1 if the deadline passed and the chain was stopped
  while [ -n "$(chain_pid)" ]; do
    if [ -n "${1:-}" ] && [ "$(date +%s)" -ge "$1" ]; then
      say "deadline reached; stopping the chain (it stops its own server)"
      kill -TERM "$(chain_pid)"; sleep 90; return 1
    fi
    sleep 10
  done
  # never start a launch over a leftover server
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 500 ]; do sleep 5; done
}

for stage in "$@"; do
  wait_chain
  case "$stage" in
    diagfix)  # one more diagnostics launch for a model whose first one did not finish its captures
      redo=()
      for mk in v4-0731 mimo-v26 v41 glm-53 qwen-38-bf16; do
        [ -f "$S/$mk/profiles/prefill128k.manifest.json" ] && [ -f "$S/$mk/_kv/live_128k_B8.json" ] || redo+=("diag:$mk")
      done
      if [ ${#redo[@]} -gt 0 ]; then
        say "diagnostics incomplete, relaunching: ${redo[*]}"
        bash bench/blog_launch.sh diagnostics-redo "${redo[@]}" --publish
        sleep 5; wait_chain
      else
        say "diagnostics complete for all five models"
      fi
      ;;
    routing)
      start=$(date +%s)
      say "stage 3 routing: first model, 20-minute gate"
      bash bench/blog_launch.sh routing-gate routing:v4-0731 --publish
      sleep 5; wait_chain $((start + 1200))
      if [ ! -f "$S/v4-0731/_routing/routing.json" ]; then
        say "stage 3 gate FAILED: no routing statistics on the first model within 20 minutes; stage skipped"
        continue
      fi
      say "stage 3 gate passed; remaining models until $(date -u -d @$((start + 4500)) +%H:%M:%S)"
      bash bench/blog_launch.sh routing-rest routing:mimo-v26 routing:v41 routing:glm-53 routing:qwen-38-bf16 --publish
      sleep 5; wait_chain $((start + 4500)) || say "stage 3 stopped at its 75-minute limit"
      ;;
    timing)
      say "stage 5 timing: three blocks in rotating order"
      steps=(timing:v4-0731:1 timing:mimo-v26:1 timing:v41:1 timing:glm-53:1 timing:qwen-38-bf16:1
             timing:v41:2 timing:glm-53:2 timing:qwen-38-bf16:2 timing:v4-0731:2 timing:mimo-v26:2
             timing:qwen-38-bf16:3 timing:v4-0731:3 timing:mimo-v26:3 timing:v41:3 timing:glm-53:3)
      if ! "$VLLM_VENV/bin/python" bench/blog_study.py dry-run "${steps[@]}" > "$S/_logs/timing-dry-run.out" 2>&1; then
        say "timing dry run FAILED; see _logs/timing-dry-run.out"; exit 1
      fi
      bash bench/blog_launch.sh timing "${steps[@]}" --publish
      sleep 5; wait_chain
      say "stage 5 finished"
      ;;
    *) say "unknown stage $stage"; exit 2 ;;
  esac
done
say "runner finished"
