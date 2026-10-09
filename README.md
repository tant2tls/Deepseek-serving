# Five-model serving experiments on 8×H100

This branch (`main`) accompanies the [architecture blog](index.html): **DeepSeek V4 Flash 0731, DeepSeek V4.1 Flash, MiMo-V2.6-Flash-MOPD, Qwen3.8-Flash-Next and GLM-5.3-Flash**. It contains measured results, the evidence behind the figures, per-model setup and reproduction commands. Qwen means the original BF16 checkpoint throughout.

| Start here | Contents |
| --- | --- |
| [Target](target.md) | Kan's questions, the two purposes (measure to answer them; a blog useful to every reader), the five pinned checkpoints and what is answered so far |
| [Model setup](docs/model-setup.md) | Five pinned checkpoints, downloads, launch flags, expected log lines, requests and pitfalls |
| [Reproduce](docs/reproduce.md) | Local table regeneration and the Linux GPU measurement procedure |
| [Experiments and limits](docs/experiments.md) | Workloads, counts, source selection, node control and what was not measured |
| [Five-model numbers](reports/five-model/results.md) | Throughput, separate latency tables, attention/FFN components and live memory |
| [Earlier experiments](docs/previous-experiments.md) | Completed DeepSeek and MiMo studies on the older runtime; their reports are [reports/v41-vs-0731/report.md](reports/v41-vs-0731/report.md) and [reports/mimo-v26/report.md](reports/mimo-v26/report.md) |
| [Provenance](docs/provenance.md) | Exact-byte evidence preservation, derived selections and omitted material |

All five models used the **same core serving setup**: vLLM `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`), 8×H100 80GB SXM, TP8 plus expert parallel, 0.90 utilization, context limit 262144, at most 64 sequences, and an 8192-token prefill chunk budget. The workload uses the same public code/math/chat texts, approximately 1K/16K/64K input, 256 forced output tokens, one or eight clients and three timing blocks. Prefix caching and speculation were off. Native precision differs by model.

Four deployments were measured on the first node on October 7–8, 2026. Qwen was measured on a second node on October 8, with MiMo as a node control. **Compare throughput across nodes; keep Qwen latency separate.** MiMo's control reproduced throughput within about 2%, but TTFT and eight-client TPOT medians moved. GLM used `reasoning_effort=low` because its template has no thinking-off switch.

The main selection contains **90 valid timing runs, 30 trace captures and 30 live-KV snapshots**. The second-node MiMo control adds 6 timing runs, 6 traces and 6 snapshots. This is a selection of completed evidence, not a newly measured study. No GPU session was run to prepare this branch.

Regenerate and check the tables locally with Python 3.10+ (standard library only):

```bash
python tools/build_blog_results.py
python tools/audit_references.py
```

Weights, environments, caches, credentials and full profiler traces are excluded. Raw retained measurements keep their original values and study identities. The full archive, with every study, plan and original frozen ledger, is the `all-data` branch (named `main` until 2026-10-10); see [provenance](docs/provenance.md).
