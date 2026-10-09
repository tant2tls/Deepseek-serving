# Bench: the code that launches vLLM and measures it

Everything here runs **on the GPU node**. Tools that only read saved results on a laptop are in [tools/](../tools/README.md). The commands, in order, are in [reproduce/](../reproduce/README.md); the reasons behind them are in the [lessons](../teach_me/README.md).

Nothing in this folder may be run on rented GPUs without explicit authorization. Blog launches have prefix caching and speculation off.

## The October five-model protocol

These scripts produced the numbers in the article. `BLOG_STUDY` selects the study: `blog-architecture-h100-v1` (default, first node) or `qwen-bf16-h100-v1` (second node). Raw output goes to the git-ignored `results/<study>/`; curated evidence goes to `reports/<study>/`.

| Script | What it does | Run it when |
| --- | --- | --- |
| [serve.sh](serve.sh) | Launches one model with the common flags. Holds each key's checkpoint, pinned revision and parsers | Looking at one server by hand; the chain calls it for you |
| [blog_launch.sh](blog_launch.sh) | Starts a chain detached from your shell and returns at once | Every reported launch |
| [blog_study.py](blog_study.py) | The chain. `env` records the node, `plan` freezes request counts, `dry-run` validates pins, inputs and plan, `chain` runs `diag:<key>` and `timing:<key>:<block>` steps. Each step owns and stops its server | Through `blog_launch.sh`, or directly for `env`, `plan`, `dry-run` |
| [run_matrix.py](run_matrix.py) | Runs one `vllm bench serve` point into an immutable directory with telemetry. Holds `MODELS`: tokenizer mode and thinking switch per key | Imported by the chain |
| [blog_corpus.py](blog_corpus.py) | Normalizes the public code, math and chat texts | Once per node, in a scratch environment with `pyarrow` |
| [blog_inputs.py](blog_inputs.py) | Builds the frozen request lists; `extra <key>` adds token counts for a later model | After the corpus |
| [blog_report.py](blog_report.py) | Turns `results/` into tables: `pilot`, `serving`, `memory`, `components`, `tables`, `compare`, `publish` | After each chain finishes |
| [trace_breakdown.py](trace_breakdown.py) | Classifies GPU kernels of one rank trace into components (`CATS`) | Reading one trace; `blog_report.py` extends it for this build |
| [blog_layers.py](blog_layers.py) | Counts how many layers process a whole prefill chunk in one rank trace | Checking a claim such as V4.1's prefill skip |
| [curate.py](curate.py) | Copies reviewed raw runs into `reports/<study>/data/<key>/` and writes `CURATION.json` with source and published hashes | Called by `blog_report.py publish` |

How they connect:

```
blog_corpus.py -> blog_inputs.py -> blog_study.py env / plan / dry-run
blog_launch.sh -> blog_study.py chain -> serve.sh (one owned server per step)
                                      -> run_matrix.py (timing points)
results/<study>/ -> blog_report.py -> reports/<study>/*.csv -> curate.py -> reports/<study>/data/
reports/<study>/*.csv -> tools/build_blog_results.py -> reports/five-model/ and the article
```

## The September studies (historical)

These scripts produced the earlier DeepSeek and MiMo studies on the older build, with random-token prompts, prefix-cache checks and speculative arms. They are kept so the [earlier reports](../docs/previous-experiments.md) remain documented. **Do not run them as the October workflow**: they hardcode historical studies, a `tmux` session and the old runtime, and some include prefix-cache experiments that are not authorized.

| Script | What it did |
| --- | --- |
| [chain_dspark.sh](chain_dspark.sh), [chain_mimo.sh](chain_mimo.sh), [chain_lib.sh](chain_lib.sh) | Unattended launch chains for the DSpark arms and the MiMo study |
| [profile_trace.py](profile_trace.py) | One labelled profiler trace around a single controlled request |
| [prefix_reuse_check.py](prefix_reuse_check.py) | Sequential prefix-cache hit check |
| [quality_smoke.py](quality_smoke.py), [compare_outputs.py](compare_outputs.py) | Four-prompt sanity check; text match of a speculative arm against its autoregressive run |
| [summarize.py](summarize.py), [compare.py](compare.py), [compare_models.py](compare_models.py), [spec_compare.py](spec_compare.py) | Tables for the September reports |

`serve.sh` and `run_matrix.py` serve both periods. Their extra configs (`off-prefix`, `dspark-*`, `mtp-k3`, `dflash-k7`) and workloads belong to September; the October chain accepts only `off` and `off-profidle`.

## Adding to this folder

**A model.** Add its key to the `case` in `serve.sh` (checkpoint, 40-character revision, parsers) and to `MODELS` in `run_matrix.py` (tokenizer mode, thinking switch). The two must agree; the audit compares them. Then write `reproduce/<key>.md` and add the model to [docs/models.md](../docs/models.md).

**A study.** Never write new measurements into a completed study. Add an entry to `STUDIES` in `blog_study.py` with its own ID, allowed keys, blocks and pin files, then run it with `BLOG_STUDY=<new id>`. A node or build change needs a fresh control ([lesson 6](../teach_me/06_new_checkpoint_new_study.md)).

**A measurement.** Put GPU-side collection here and laptop-side analysis in [tools/](../tools/README.md). Record missing values as empty, never zero. Keep timing (config `off`) and profiling (config `off-profidle`) in separate launches and separate tables.

**A trace category.** After a build change, read `components_unclassified.md`. Extend the classifier in `blog_report.py`, not the historical `CATS` list.

Conventions: every script states its usage in its docstring or header; paths are relative to the repository root; credentials come from the environment and never from a file here.
