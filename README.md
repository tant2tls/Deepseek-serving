# Five-model serving experiments on 8×H100

This branch (`main`) accompanies the [architecture blog](index.html): **DeepSeek V4 Flash 0731, DeepSeek V4.1 Flash, MiMo-V2.6-Flash-MOPD, Qwen3.8-Flash-Next and GLM-5.3-Flash**, served with vLLM on eight H100 80GB GPUs. It contains the measured results, the evidence behind every figure, a run guide per model and lessons on the method. Qwen means the original BF16 checkpoint throughout.

## Start here

| I want to | Go to |
| --- | --- |
| Read the article | [index.html](index.html) |
| Know which questions the work answers, and which are still open | [target.md](target.md) |
| Review the planned rerun and the experiments needed for stronger conclusions | [Rerun plan](target.md#five-model-rerun-plan) |
| See the numbers | [reports/five-model/results.md](reports/five-model/results.md) |
| Run one model, or repeat the measurements | [reproduce/](reproduce/README.md) |
| Know the disk, GPU memory, host RAM and hours a reproduction needs | [reproduce/: what a reproduction needs](reproduce/README.md#what-a-reproduction-needs-disk-memory-and-time) |
| Learn how such numbers are measured with vLLM, or set it up on another machine | [teach_me/](teach_me/README.md) |
| Understand the five architectures and find their technical reports | [docs/models.md](docs/models.md) |
| Know how far a comparison can be trusted | [docs/experiments.md](docs/experiments.md) |
| Check where the evidence came from | [docs/provenance.md](docs/provenance.md) |

## Repository map

```
AGENTS.md     rules for anyone, person or agent, who changes this branch
README.md     this map
target.md     the brief, the five pinned checkpoints, what is answered so far
index.html    the article: one self-contained page
install.sh    installs the two recorded vLLM runtimes on a GPU node

bench/        MEASURE  code that launches vLLM and collects numbers; runs on the GPU node
tools/        BUILD    tables, article data and the audit; run on any computer, no GPU
reports/      RESULTS  evidence of every study, its hash ledger, and the generated tables
reproduce/    RUN      one guide per model, the full procedure, known setup problems
teach_me/     LEARN    seven lessons: measuring with vLLM and setting it up on another system
docs/         REFER    models and technical reports, protocol and limits, provenance, earlier studies
```

Every folder has a `README.md` that says what belongs there and how to add to it.

How the pieces connect:

```
bench/  --launch and measure-->  results/<study>/   (raw, on the node, never in Git)
                                      |  review and curate
                                      v
                                reports/<study>/    (evidence, hashed in reports/provenance.json)
                                      |  tools/build_blog_results.py
                                      v
                                reports/five-model/ (generated tables)
                                      |  tools/build_blog_page.py
                                      v
                                index.html          (the article)

tools/audit_references.py checks every arrow above, the layout, the model pins and the links.
```

## The measurements in brief

All five models used the **same core serving setup**: vLLM `554340f3d3259e321be4c07282be7a02a5aeef83` (`0.31.1rc1.dev50+g554340f3d`), 8×H100 80GB SXM, TP8 plus expert parallel, 0.90 utilization, context limit 262144, at most 64 sequences, and an 8192-token prefill chunk budget. The workload uses the same public code/math/chat texts, approximately 1K/16K/64K input, 256 forced output tokens, one or eight clients and three timing blocks. Prefix caching and speculation were off. Native precision differs by model.

Four deployments were measured on the first node on October 7–8, 2026. Qwen was measured on a second node on October 8, with MiMo as a node control. The article displays all five models for each serving metric. MiMo's control reproduced throughput within about 2%, but TTFT and eight-client TPOT medians moved, so cross-node latency differences can reflect the machine as well as the model. GLM used `reasoning_effort=low` because its template has no thinking-off switch.

The main selection contains **90 valid timing runs, 30 trace captures and 30 live-KV snapshots**. The second-node MiMo control adds 6 timing runs, 6 traces and 6 snapshots. This is a selection of completed evidence, not a newly measured study. No GPU session was run to prepare this branch. Earlier September studies on an older build are kept under `reports/` as history; [docs/previous-experiments.md](docs/previous-experiments.md) explains how to read them.

The [new rerun proposal](target.md#five-model-rerun-plan) uses one physical node and the latest vLLM build frozen at preparation time, with 2,048-token main workloads, 8,192-token checks and c1/c4/c16/c32/c64 load curves. Its [answer map](target.md#answer-map-for-the-brief) gives each of Kan's questions one deciding measurement, and its [claims](target.md#claims-the-rerun-must-test) state each model's strength and weakness with the result that would disprove it. It also defines [targeted experiments for the blog](target.md#experiments-for-the-blog), including expert-routing statistics and one [cached-prefix space check](target.md#cached-prefix-space-check) that needs its own approval. This is planning only: the current article and evidence still describe the completed 256-token study.

## Check it locally

Python 3.10+ with the standard library only; no GPU and no network:

```bash
python tools/build_blog_results.py        # regenerate the derived tables
python tools/build_blog_page.py --check   # the article's numbers equal the CSVs
python tools/audit_references.py          # evidence bytes, arithmetic, layout, pins, links
```

## Working in this repository

| To add | Start at | Then |
| --- | --- | --- |
| A model | [bench/README.md](bench/README.md#adding-to-this-folder) | Write `reproduce/<key>.md`, add it to [docs/models.md](docs/models.md) and to the pins in [target.md](target.md) |
| A measurement or study | [bench/README.md](bench/README.md#adding-to-this-folder) | A new study ID and folder under [reports/](reports/README.md#rules-for-this-folder); never a completed study |
| A table, figure or check | [tools/README.md](tools/README.md#adding-a-tool) | Give it `--check` and call it from the audit |
| A lesson | [teach_me/README.md](teach_me/README.md#adding-a-lesson) | Number it after the last one |
| A reference document | [docs/README.md](docs/README.md#where-a-new-document-goes) | One topic per file; link, do not repeat |

Before publishing, run the audit on a clean clone ([how](reproduce/README.md#before-publishing-edits)). It fails if a file lands outside the map above, if a model is pinned differently in two places, if a generated table or the article's data is stale, or if a link or anchor is broken. New GPU work needs explicit authorization; [AGENTS.md](AGENTS.md) has the rules.

Weights, environments, caches, credentials and full profiler traces are excluded. Raw retained measurements keep their original values and study identities. The full archive, with every study, plan and original frozen ledger, is the `all-data` branch (named `main` until 2026-10-10); see [provenance](docs/provenance.md).
