# Reproducing a serving study on one 8×H100 node

This is the procedure that produced [report.md](../report.md) (study `v41-vs-0731`). It also serves as the template for the next model. GPUs are rented by the hour, so the procedure is organized to keep them busy and to stop servers as soon as measurement ends.

## 1. Environment

| Item | Value |
| --- | --- |
| venv | `source /root/vllm/bin/activate`; vLLM `0.30.1rc1.dev223+g44af287eb`, torch `2.13.0+cu132`. Keep it pinned within a study. `pandas` is required for `--dataset-name custom`. |
| Credentials | `HF_TOKEN` from the environment only. Never write it into scripts. |
| Weights | `HF_HOME=/workspace/hf` (shared xet/blob store). Use `du -shL` for the real size. Download with a pinned `--revision <sha>`. |
| JIT caches | `/root` is a gocryptfs FUSE mount, where concurrent TP-rank JIT compiles race. `bench/serve.sh` sets `TRITON_CACHE_DIR=/tmp/triton_cache` and `FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_ws`. If a new model dies in warmup or graph capture with `FileNotFoundError` or `CUDA error: invalid argument`, move that model's JIT cache (for example `DG_JIT_CACHE_DIR`) to `/tmp` first. |
| tmux | `ds-serve` (server), `ds-bench` (harness), `ds-chain` (chained arms), `ds-dl` (downloads) |
| Stopping | `tmux send-keys -t ds-serve C-c`. Never run `pkill -f <pattern>` from a shell whose own command line contains that pattern; use bracket patterns such as `pgrep -f "[v]llm serve"`. |

## 2. Adding a model

1. Check `config.json` `architectures` against `vllm.model_executor.models.registry.ModelRegistry.get_supported_archs()` in the pinned venv.
2. Add one entry each to `MODELS` in `bench/run_matrix.py`, the `case` block in `bench/serve.sh`, and `bench/profile_trace.py`: model ID, pinned revision, tokenizer mode, and reasoning/tool parsers.
3. Confirm that `chat_template_kwargs.thinking=false` (or the model's equivalent) produces non-thinking output, and record the encoded request.
4. Check that the shared prefix prompt files (`results/<study>/_prompts/`) tokenize to the same counts. If they don't, record it and relax the 98% prompt-token check to the measured count.
5. Write a plan JSON (`results/<study>/plan*.json`, copied to `reports/<study>/`) **before** collecting.

## 3. Launch plan (each restart costs 3–10 min)

| Launch | Config | Workloads | Approx. time |
| --- | --- | --- | --- |
| 1 | `off-profidle` (prefix caching off, idle profiler; `PROFILE_DIR` required) | `run_matrix.py --workloads concurrency,context,prefix,isolated --prefix-states cache-off`, then traces with `profile_trace.py` (16K and 64K prefill) | concurrency ~45 min, context ~30, prefix cache-off ~45, isolated ~25, traces ~3 each |
| 2 | `off-prefix` | `run_matrix.py --config off --workloads prefix --prefix-states cold,prewarmed`, then `prefix_reuse_check.py` | ~30 min + 1 min |
| 3+ | one per speculative arm (for example `dspark-fixed-k5`) | concurrency c1/4/16/64 | ~17–25 min per arm plus ~3.5 min launch |

Chain speculative arms with `bench/chain_dspark.sh '<model> <config>' ...`. It waits for any running sweep, stops the server, launches the next arm, polls `/v1/models` (aborting if the process dies or 30 min pass), runs the sweep, and stops the server at the end. Watch `results/<study>/_chain_dspark.log` with a Monitor filter on `CHAIN|valid=False|Traceback|Error`. **Do not hand-roll readiness loops:** a broken `until` condition once left a ready server idle for 1 h 50 min.

The first launch of a new model takes about 8–10 min (JIT, DeepGEMM warmup, graph capture); later launches take about 3.5–5 min. Adding a speculative method changes graph-capture memory and KV capacity; record both from the server log.

## 4. Harness rules

- `run_matrix.py` writes immutable run directories under `results/<study>/<model>/<workload>/<config>/<point>/repeat-N`. Reruns become `repeat-N-rerunK`, and nothing is overwritten.
- A run is valid only with all requests completed, output = n × 256, and server prompt tokens ≥ 98% of the target.
- Speculative counters (`spec_num_drafts`, `spec_num_draft_tokens` = scheduled drafts, `spec_num_accepted_tokens`) are in `summary.json`. Per-position acceptance comes from `telemetry/metrics_{before,after}.prom`.
- **Prewarm caveat:** a model whose SWA prefix reuse needs a second touch (0731-style) must be prewarmed with **two** requests per prefix using distinct suffixes. Verify with `prefix_reuse_check.py` before trusting prewarmed numbers.
- Some models are slower on the first repeat at new shapes (0731: 5–9%, and once 50% on a new DSpark shape). Keep those runs, and report medians next to means.
- Traces: decode under full CUDA graphs is not attributable. After adding a model, check that no large kernel lands in `other_elementwise` in `trace_breakdown.py`.

## 5. Analysis and publishing

```bash
python bench/summarize.py <study> <model> --csv reports/<study>/<model>_off_summary.csv > reports/<study>/<model>_off_tables.md
python bench/compare.py --csv reports/<study>/comparison.csv > reports/<study>/comparison_tables.md
python bench/spec_compare.py --csv reports/<study>/dspark_comparison.csv > reports/<study>/dspark_tables.md
python bench/compare_outputs.py <model> <spec-config> <ar-config>
python bench/curate.py <study> <model>     # lossless gzip, IP/ANSI removed from server logs, prompts/full traces excluded, SHA-256 ledger
```

`compare.py` and `spec_compare.py` contain per-model config maps (`CFG`, `AR`); extend them for a new model pair. CSV writers use `lineterminator="\n"`, and `.gitattributes` marks `reports/**/data/**` as `-text` so hashes survive checkout.

Audit on a clean clone, because `tools/audit_references.py` also scans the git-ignored `results/`:

```bash
git clone --branch <branch> . /tmp/audit && cd /tmp/audit && python tools/audit_references.py
```

Commit on a feature branch, push, and merge into `main` only when the user asks.
