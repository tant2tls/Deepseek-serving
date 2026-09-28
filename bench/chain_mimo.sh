#!/usr/bin/env bash
# MiMo-V2.6 study (docs/mimo-v2.6-plan.md), run unattended after launch 1's sweep:
# traces on the running off-profidle server -> off-prefix (reuse check, cold/prewarmed)
# -> mtp-k3 -> dflash-k7 (concurrency c1/4/16/64) -> stop server.
# A failed launch is logged and the chain moves to the next arm.
set -uo pipefail
cd "$(dirname "$0")/.."
source /root/vllm/bin/activate
source bench/chain_lib.sh
MK=mimo-v26; STUDY=mimo-v26; R=results/$STUDY/$MK

wait_sweep
chain_log "launch-1 sweep finished"
trace $MK $STUDY prefill16k_decode8 16384 8
trace $MK $STUDY prefill64k_decode4 65536 4

if launch $MK off-prefix $STUDY; then
  python bench/prefix_reuse_check.py --model $MK --study $STUDY 2>&1 | tee -a $R/_prefix_reuse_check.log
  python bench/run_matrix.py --model $MK --study $STUDY --config off --workloads prefix \
    --prefix-states cold,prewarmed 2>&1 | tee $R/_bench_off-prefix.log
  chain_log "sweep done $MK off-prefix"
fi

for cfg in mtp-k3 dflash-k7; do
  if launch $MK $cfg $STUDY; then
    python bench/run_matrix.py --model $MK --study $STUDY --config $cfg --workloads concurrency \
      --conc-list 1,4,16,64 2>&1 | tee $R/_bench_$cfg.log
    chain_log "sweep done $MK $cfg"
  fi
done
stop_server
chain_log "ALL DONE, server stopped"
