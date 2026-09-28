# Shared helpers for unattended launch chains (source from a chain script).
# GPUs are rented: every step either measures or moves to the next launch.

chain_log() { echo "[$(date +%T)] CHAIN $*"; }

stop_server() {
  tmux send-keys -t ds-serve C-c
  for _ in $(seq 120); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 5; done
  sleep 10  # let GPU memory release
}

wait_sweep() {  # wait for any running run_matrix sweep to finish
  while pgrep -f "[r]un_matrix.py" >/dev/null; do sleep 15; done
}

# launch <model-key> <config> <study> [extra env assignments...]; returns 1 if not ready in 30 min
launch() {
  local mk=$1 cfg=$2 study=$3; shift 3
  local served; served=$(python -c "import sys; sys.path.insert(0,'bench'); from run_matrix import MODELS; print(MODELS['$mk']['model'])")
  local log=$PWD/results/$study/$mk/_server/serve-$cfg-$(date +%Y%m%d-%H%M%S).log
  mkdir -p "$(dirname "$log")"
  stop_server
  tmux send-keys -t ds-serve "cd $PWD && $* bash bench/serve.sh $mk $cfg $log" Enter
  chain_log "launched $mk $cfg"
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt 1800 ]; do
    sleep 10
    curl -s localhost:8000/v1/models | grep -q "\"$served\"" && { chain_log "ready $mk $cfg"; return 0; }
    [ $((SECONDS - t0)) -gt 120 ] && ! pgrep -f "[v]llm serve" >/dev/null && break
  done
  chain_log "FAILED launch $mk $cfg; see $log"
  return 1
}

# trace <model-key> <study> <label> <isl> <max-tokens>: capture, then file the rank traces under profiles/<label>/
trace() {
  local mk=$1 study=$2 label=$3 isl=$4 mt=$5
  local pdir=$PWD/results/$study/$mk/profiles
  python bench/profile_trace.py --model "$mk" --study "$study" --label "$label" --isl "$isl" --max-tokens "$mt" \
    --profile-dir "$pdir" >/dev/null || { chain_log "FAILED trace $label"; return 1; }
  python - "$pdir" "$label" <<'EOF'
import json, sys, shutil
from pathlib import Path
pdir, label = Path(sys.argv[1]), sys.argv[2]
root = pdir.parents[3]
m = json.loads((pdir / f"{label}.manifest.json").read_text())
(pdir / label).mkdir(exist_ok=True)
moved = []
for f in m["files"]:
    src = root / f
    if src.exists():
        dst = pdir / label / src.name; shutil.move(src, dst); moved.append(str(dst.relative_to(root)))
m["files"] = moved
(pdir / f"{label}.manifest.json").write_text(json.dumps(m, indent=2))
EOF
  chain_log "trace $label done"
}
