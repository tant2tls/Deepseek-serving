#!/usr/bin/env bash
# Write the node record: GPUs, topology, CPU, memory, OS, runtime. Run it on the GPU node, inside the
# allocation when one is used, so the record shows the GPUs that were actually given to the job.
# Usage: bash system_info/collect.sh [out.md]     (default: system_info/node-record.md)
# Host names, GPU UUIDs, user names and paths are removed here: nothing private reaches Git.
set -uo pipefail
OUT=${1:-"$(dirname "$0")/node-record.md"}

# Strip identifiers the record must not carry.
scrub() { sed -E 's/GPU-[0-9a-f]{8}-[0-9a-f-]{27}/GPU-<uuid>/g; s/[A-Za-z0-9.-]*coriander[A-Za-z0-9.-]*/<node>/g; s#/(m-coriander|home|root|workspace|data|tmp)[^ ,"]*#<path>#g'; }
block() { echo '```'; "$@" 2>&1 | scrub; echo '```'; }

{
echo "# Node record"
echo
echo "Generated $(date -u +%Y-%m-%dT%H:%MZ) by \`system_info/collect.sh\`. Measured values only; host names, GPU UUIDs, user names and paths are removed."
echo
echo "## Allocation"
block bash -c 'echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  partition=${SLURM_JOB_PARTITION:-none}  gpus_on_node=${SLURM_GPUS_ON_NODE:-none}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"'
echo
echo "## GPUs"
block nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,pstate --format=csv
echo
echo "## GPU driver and CUDA"
block bash -c 'nvidia-smi | sed -n 3p'
echo
echo "## GPU topology"
block nvidia-smi topo -m
echo
echo "## CPU and memory"
block bash -c 'lscpu | grep -E "^(Model name|Socket|Core|Thread|NUMA node|CPU\(s\)|Vendor)"; echo; free -h'
echo
echo "## Storage"
block bash -c 'df -h --output=target,size,avail,fstype "${HF_HOME:-.}" | tail -1 | sed "s#^[^ ]*#<shared-filesystem>#"'
echo
echo "## OS and kernel"
block bash -c 'grep PRETTY_NAME /etc/os-release; uname -r'
echo
echo "## Runtime"
block bash -c 'PY=${VLLM_PY:-python3}; echo "python: $($PY -V 2>&1)"; $PY -c "import vllm, torch; print(\"vllm\", vllm.__version__); print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)" 2>&1 | tail -2'
echo
echo "## Slurm partitions (limits only)"
block bash -c 'sinfo -o "%P %l %D %G" 2>/dev/null | head -10'
echo
echo "## Hugging Face hub in use"
block bash -c 'echo "HF_HUB_CACHE set: ${HF_HUB_CACHE:+yes}"; echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}"'
} > "$OUT"
echo "wrote $OUT"
