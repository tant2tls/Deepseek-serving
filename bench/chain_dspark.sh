#!/usr/bin/env bash
# Run the remaining DSpark arms back to back, without idle GPU time between them.
# Waits for any running run_matrix sweep, then for each "<model> <config>":
# stop the ds-serve server, launch the arm, poll /v1/models until ready, sweep c1/4/16/64.
# Aborts (and stops the server) if a launch dies or is not ready within 30 min.
set -uo pipefail
cd "$(dirname "$0")/.."
source /root/vllm/bin/activate
STUDY=results/v41-vs-0731
declare -A NAME=([v41]=deepseek-ai/DeepSeek-V4.1-Flash [v4-0731]=deepseek-ai/DeepSeek-V4-Flash-0731)

stop_server() {
  tmux send-keys -t ds-serve C-c
  for _ in $(seq 120); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 5; done
  sleep 10  # let GPU memory release
}

while pgrep -f "[r]un_matrix.py" >/dev/null; do sleep 15; done
echo "[$(date +%T)] CHAIN previous sweep finished"

for arm in "$@"; do
  read -r MK CFG <<<"$arm"
  stop_server
  LOG=$PWD/$STUDY/$MK/_server/serve-$CFG-$(date +%Y%m%d-%H%M%S).log
  tmux send-keys -t ds-serve "cd $PWD && bash bench/serve.sh $MK $CFG $LOG" Enter
  echo "[$(date +%T)] CHAIN launched $MK $CFG"
  ok=0
  for _ in $(seq 180); do
    sleep 10
    curl -s localhost:8000/v1/models | grep -q "\"${NAME[$MK]}\"" && { ok=1; break; }
    [ $((SECONDS)) -gt 120 ] && ! pgrep -f "[v]llm serve" >/dev/null && [ -s "$LOG" ] && break
  done
  if [ $ok -ne 1 ]; then
    echo "[$(date +%T)] CHAIN FAILED launch $MK $CFG; see $LOG"; stop_server; exit 1
  fi
  echo "[$(date +%T)] CHAIN ready $MK $CFG"
  python bench/run_matrix.py --model "$MK" --config "$CFG" --workloads concurrency \
    --conc-list 1,4,16,64 2>&1 | tee "$STUDY/$MK/_bench_$CFG.log"
  echo "[$(date +%T)] CHAIN sweep done $MK $CFG rc=${PIPESTATUS[0]}"
done
stop_server
echo "[$(date +%T)] CHAIN ALL DONE, server stopped"
