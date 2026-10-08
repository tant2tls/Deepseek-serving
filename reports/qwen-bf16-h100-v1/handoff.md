# Handoff: qwen-bf16-h100-v1

Session of 2026-10-08 on one rented 8×H100 80GB node, authorized by Tan that day ("run Qwen3.8-Flash-Next with the original BF16 checkpoint"). Results: [findings.md](findings.md). Checkpoint rule: [docs/qwen-checkpoint-policy.md](../../docs/qwen-checkpoint-policy.md). Build: vLLM `0.31.1rc1.dev50+g554340f3d` (`554340f3d3259e321be4c07282be7a02a5aeef83`) in `/root/vllm-latest`, the build of `blog-architecture-h100-v1`.

**This node is not the node of 2026-10-07.** Same GPU model and count, same vLLM and torch; different CPU (Xeon Platinum 8592V, 256 threads, 4 NUMA nodes, against 8480+, 208 threads, 2 NUMA nodes), newer GPU driver (580.173.02 against 580.105.08) and a few newer Python dependencies. Records: `study/_env/`. That is why the study carries a MiMo control.

## 1. Status of every planned item

| Item | Planned | State | Evidence |
| --- | ---: | --- | --- |
| Distinct BF16 key and study; FP8 key retired | 1 | **Done** | `bench/serve.sh`, `bench/run_matrix.py`, `STUDIES` in `bench/blog_study.py` |
| Same request lists as the blog study | 9 lists | **Done**: every list and request hash equals the published manifest; BF16 tokenizer counts equal the FP8 counts on 543/543 requests | `study/_inputs/`, dry run |
| Loaded-precision check | 1 | **Done**: `dtype=torch.bfloat16`, `quantization=None`, unquantized Triton MoE backend, BF16 n-gram embedding, no `fp8` in the log | `data/qwen-38-bf16/_server/` |
| Functional check (rendered request, natural EOS, forced 256) | 1 | **Pass** | `data/qwen-38-bf16/_functional/functional.json` |
| Disjoint pilot | 6 runs | **Done**, 6/6 valid | `data/qwen-38-bf16/pilot/` |
| Frozen plan and dry run | 1 | **Done** before timing | [plan.json](plan.json) |
| Unprofiled timing, Qwen BF16 | 18 runs | **Done, 18/18 valid**, three launch-separated blocks | [serving.csv](serving.csv), [serving_tables.md](serving_tables.md) |
| Node control: MiMo, one timing block | 6 runs | **Done, 6/6 valid** | [comparison.md](comparison.md) section 1 |
| Diagnostic traces, Qwen BF16 | 6 | **Done, 6/6**, all 8 ranks | [components.csv](components.csv), [components_tables.md](components_tables.md) |
| Live KV snapshots, Qwen BF16 | 6 | **Done, 6/6** | [memory.csv](memory.csv), [memory_tables.md](memory_tables.md) |
| Trace control: MiMo traces and KV snapshots on this node (addendum) | 6 + 6 | **Done**: functional pass, 6/6 traces, 6/6 KV snapshots; every component within 3% of 2026-10-07 except all-reduce (within about 10%, once +16%) | [plan_addendum_mimo_trace_control.json](plan_addendum_mimo_trace_control.json) |
| FP8 rerun as a direct control | – | **Not run by rule**: no new FP8 runs | policy |
| Hardware HBM counters, routing statistics | – | Not collected, as in the blog study | – |
| Other unquantized MoE backends (`FlashInfer CUTLASS`, `TRTLLM`, `BATCHED_TRITON`) | – | **Not run**: a single-change arm outside the common deployment | section 5 |
| Prefix caching, speculation | – | Off; nothing run | – |

## 2. Failures, deviations and what they affect

| What happened | Effect |
| --- | --- |
| The harness had no BF16 key or second study when the node was rented; it was wired while the weights downloaded | No rented time lost beyond the download itself, but the preparation rule ("finish local preparation before rental") was not met. The wiring is now in the repository |
| Download ran at about 240 MiB/s (360 GB in 23 minutes). A second downloader and a login token did not raise the rate | 24 minutes of rented time before the first launch. The token file was removed from the node after the session |
| A `pgrep -f` pattern matched the calling shell while restarting the download | One download restart, about 1 minute; no data affected. Trap recorded in [docs/reproduce.md](../../docs/reproduce.md) section 0.1 |
| The stopped downloaders left 54 empty `*.incomplete` blobs | None: all 144 files were checked against the hub sizes before the launch, then the leftovers were deleted |
| Timing windows under 60 s at 16K/c8 (about 50 s) and 64K/c8 (59–60 s) | Counts were kept equal to the blog study so that every model sees the same requests; the FP8 windows at these points were 46 s and 58 s |
| In the 64K, B=1 decode capture the client received 48 text chunks while the engine ran 143 steps | The "client step time" of that capture (26 ms profiled, 18 ms unprofiled) is not a step time. Use the trace span and the timing runs. Known forced-length effect |
| Decode steps under the profiler are much slower than unprofiled ones for BF16 (8.9 ms against 5.9 ms at 1K, B=1; FP8 on the first node: 6.9 against 6.3; MiMo on this node: 6.1 against 5.8) | The BF16 all-reduce rows include waiting that the unprofiled timing does not show. The MiMo trace control rules out the node as the cause. See findings section 6, point 5 |
| MiMo's control block was also MiMo's first launch on this node (645 s, JIT builds) | Warmup ran before the timed points as always; single-client TPOT equals the 2026-10-07 value |
| One model plus a control in one evening | Blocks are launch-separated, not day-separated; no rotation against other models |
| The MiMo trace control was added during the session | Declared in an addendum before it ran; diagnostic only; 14 minutes of rented time instead of the estimated 12, because two decode windows ran the 5-second cap |
| The table writer failed on an empty step-time cell (MiMo, 64K, B=1, no text chunks in the window) | Fixed in `bench/blog_report.py` (empty cells are skipped, not zero); the completed study's published tables were not regenerated |
| TTFT at 1K with one client is bimodal on this node (about 190 and 120 ms for Qwen BF16; about 150 and 113 ms for MiMo) | Block medians of that point are 189, 120 and 131 ms. All kept. Throughput is affected by at most 3% between blocks |

## 3. Cost and time (node clock)

| Phase | Wall time |
| --- | --- |
| Login, inventory, download start | 17:48 |
| BF16 download (GPUs idle; harness wiring, inputs, dry-run preparation in parallel) | 17:48–18:12 |
| Diagnostics: first launch 580 s, functional, pilot, KV snapshots, traces | 18:12–18:38 (server up 1,521 s) |
| Timing: Qwen block 1 (server up 1,305 s), MiMo control (1,814 s, first launch), Qwen blocks 2 and 3 | 18:38–20:14 (Qwen blocks 2 and 3: 1,291 s and 1,297 s) |
| MiMo trace control | 20:14–20:28 (server up 838 s) |
| Last server stopped, 0 MiB on all GPUs | 20:28 |

A Qwen BF16 timing step takes about 22 minutes: 4.3 minutes relaunch, about 4.5 minutes of warmup over the three input lengths, six points. Between stages the next chain was started by a waiting script, so the GPUs sat idle for under a minute between diagnostics and timing. No hourly price was supplied; cost is in node-hours: 2.7 from login to the last stop (17:48–20:28), of which 2.24 with a server up (8,066 s over six launches) and 0.4 waiting for the download.

## 4. Exact commands

```bash
export VLLM_VENV=/root/vllm-latest HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 BLOG_STUDY=qwen-bf16-h100-v1
# weights (BF16 Qwen, then MiMo for the control)
hf download Qwen/Qwen3.8-Flash-Next --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --max-workers 32
hf download XiaomiMiMo/MiMo-V2.6-Flash-MOPD --revision 2479e2d0029eca9a34cc7e7f55a121925f81908e --max-workers 32
# public inputs; must reproduce the published lists of blog-architecture-h100-v1
hf download EleutherAI/hendrycks_math --repo-type dataset --revision 21a5633873b6a120296cce3e2df9d5550074f4a3
hf download OpenAssistant/oasst1 --repo-type dataset --revision fdf72ae0827c1cda404aff25b6603abec9e3399b
python3 -m venv /tmp/blog-inputs-venv && /tmp/blog-inputs-venv/bin/pip install pyarrow
/tmp/blog-inputs-venv/bin/python bench/blog_corpus.py && $VLLM_VENV/bin/python bench/blog_inputs.py
$VLLM_VENV/bin/python bench/blog_inputs.py extra qwen-38-bf16
$VLLM_VENV/bin/python bench/blog_study.py env
# what was run
bash bench/blog_launch.sh chain-diag diag:qwen-38-bf16
$VLLM_VENV/bin/python bench/blog_study.py plan --counts '{"16k:c1": 36, "16k:c8": 75, "1k:c1": 48, "1k:c8": 204, "64k:c1": 18, "64k:c8": 33}' --note '<see plan.json>'
$VLLM_VENV/bin/python bench/blog_study.py dry-run timing:qwen-38-bf16:1 timing:mimo-v26:1 timing:qwen-38-bf16:2 timing:qwen-38-bf16:3
bash bench/blog_launch.sh chain-timing timing:qwen-38-bf16:1 timing:mimo-v26:1 timing:qwen-38-bf16:2 timing:qwen-38-bf16:3
bash bench/blog_launch.sh chain-mimo-diag diag:mimo-v26 --diag-parts functional,kv,trace
# tables and publishing
python bench/blog_report.py serving && python bench/blog_report.py memory
python bench/blog_report.py components && python bench/blog_report.py tables
python bench/blog_report.py compare && python bench/blog_report.py publish
python tools/audit_references.py     # on a clean clone
```

Raw runs live in the git-ignored `results/qwen-bf16-h100-v1/`. Curated copies are under `data/<model>/` and `study/` here, each with a `CURATION.json` ledger (lossless gzip, private IPs masked in logs, full traces and public input texts excluded).

## 5. Next steps, in order of value

1. **Expert backend for unquantized experts.** The build chose `TRITON` and lists `FlashInfer CUTLASS`, `FlashInfer TRTLLM` and `BATCHED_TRITON` as alternatives. The BF16 expert path is where this deployment loses to the FP8 history at eight clients (findings section 6). One variable, one model, the three eight-client points plus 1K/c1, baseline and intervention, three blocks: 24 runs, about 1.5 node-hours. Check the selecting flag with `vllm serve --help=all | grep -i moe` first; record it as a separate arm with its own control, never as the common deployment.
2. **Confirmation of the single-client advantage** (BF16 TPOT 5.8 ms against 6.3 ms for FP8 on another node): held-out requests and at least five independently scheduled blocks. A same-node FP8 run would be the direct control and is excluded by the policy unless Tan changes it.
3. **Why TTFT and eight-client TPOT medians moved on this node** while throughput did not (MiMo control: 64K/c8 TTFT +30%, TPOT −14%). Needs per-request scheduling timestamps (`scheduled_ts`, `first_token_ts`) from both nodes; until then do not compare those two medians across nodes.
4. **A BF16 run next to the other four models on one node**, with rotated order, if the blog should show six deployments from one session.
5. HBM counters, routing statistics, 128K and prefix capacity remain as listed in the [blog handoff](../blog-architecture-h100-v1/handoff.md).
