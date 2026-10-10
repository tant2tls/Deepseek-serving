# system_info: the GPU node's environment

What belongs here: the environment each measurement session ran on, written by `collect.sh` on the node itself. Its output shows the GPUs the job was given, the driver, the topology, the CPU and memory, the OS, and the installed runtime.

| File | What it is |
| --- | --- |
| [collect.sh](collect.sh) | Writes the record. Run it on the node, inside the allocation: `VLLM_PY=<python> bash system_info/collect.sh system_info/<session>.md` |
| `node-record.md` or `<session>.md` | One record per allocation, so a later session on other GPUs has its own file |

Rules for this folder:

- Records contain measured values only. Host names, GPU UUIDs, user names and paths are removed by the script; do not add them by hand.
- A record describes one allocation. A session that gets new GPUs writes a new record and never edits an old one.
- No weights, caches, logs or traces go here. Those stay out of Git.
- Runs on the H200 node are described in [docs/H200_B200_plan.md](../docs/H200_B200_plan.md).
