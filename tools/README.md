# Tools: build the tables, feed the article, check the repository

These run on any computer with Python 3.10+ and nothing else: standard library only, no GPU, no network. They read the saved CSVs under [reports/](../reports/README.md) and never change a measurement. Code that talks to a vLLM server lives in [bench/](../bench/README.md).

```bash
python tools/build_blog_results.py        # regenerate the derived tables
python tools/build_blog_page.py --check   # the article's numbers still equal the CSVs?
python tools/audit_references.py          # everything, before publishing
```

| Tool | Reads | Writes | Use it when |
| --- | --- | --- | --- |
| [build_blog_results.py](build_blog_results.py) | `serving.csv`, `components.csv`, `memory.csv` of the two October studies | `reports/five-model/results.md`, `serving.csv`, `node_control.csv`, and both `findings.md` | A CSV, a model list or a table's wording changed. `--check` only compares |
| [build_blog_page.py](build_blog_page.py) | The same CSVs and the data block inside `index.html` | With no flag, the `<script id="study-data">` block of `index.html`; with `--check`, nothing | Writing or editing the article |
| [audit_references.py](audit_references.py) | The whole repository | Nothing | Before every commit that will be published, on a clean clone |

## How the article is built

`index.html` is one self-contained page: its styles, its script, its text and its data are in the file and it loads nothing from the network, so it opens directly in a browser with no build step.

- **Interactive figures** (serving bars, component table, memory bars) are drawn in the browser from one JSON block, `<script id="study-data">`. `build_blog_page.py` recomputes the `serving`, `components`, `memory` and `source_sha256` parts of that block from the CSVs. It keeps `runtime`, `models`, `componentDefs` and `studies` as written, because names, colors and row labels are presentation choices. To show another model, add it to `models` in the block and rerun the tool.
- **Two static tables** are checked cell by cell against the CSVs: the throughput table (`#throughput-table`, including which cell is marked as the winner) and the Qwen latency table (`#bf16-latency`).
- **Prose, the hero chart and the other static tables** are written by hand. When a number changes, search the page for the old value. Every figure caption should name the build and the node, and say whether it shows elapsed time, a kernel sum, a live-state approximation or an estimate.

A block value is compared with a relative tolerance of 1e-9, so the check does not depend on the Python version. A source hash in the block is accepted if it is the hash of the CSV in this branch or the hash of its source file recorded in `reports/provenance.json`: a row selection keeps the values of its source.

## What the audit checks

| Check | Fails when |
| --- | --- |
| Layout | A tracked file or folder is outside the agreed top level, or a folder has no `README.md` |
| Model pins | `bench/serve.sh` and `bench/run_matrix.py` disagree on a checkpoint or revision; a key has no `reproduce/<key>.md`; a page, `target.md` or `docs/models.md` does not state the pinned revision |
| Evidence bytes | A file in `reports/provenance.json` is missing or its SHA-256 changed; a file under `reports/` is not registered anywhere |
| Curation ledgers | A file named in a `CURATION.json` no longer matches its published hash |
| September GLM archive | An archived file changed, or its 42-point CSV disagrees with the raw JSON |
| Timing arithmetic | A `serving.csv` row disagrees with its raw run, its request count, the forced 256 tokens, the build, or the input hashes |
| Diagnostics | A model lacks its six snapshots or six traces, or a trace lacks one of the eight ranks |
| Generated tables | A file written by `build_blog_results.py` is stale |
| Article data | `build_blog_page.py --check` reports a difference |
| Publication scan | A file exceeds 10 MiB, or contains a credential pattern, a private path or host, or an out-of-scope Qwen deployment |
| Links | A local link points to a missing file, or its `#anchor` does not exist in the target |

A pass means the export is intact and consistent. It does not prove causal control, answer quality or that a fresh GPU run would reproduce the timings.

## Adding a tool

- Keep it standard-library only and runnable from the repository root.
- Give it a docstring that starts with one sentence and shows the command line.
- A tool that writes a file must offer `--check`, and the audit must call that check, so a stale output cannot be published.
- A tool that draws a figure reads the CSVs, never the raw runs, and writes its output next to the page or under `reports/five-model/`. Register a new output in the audit.
- A new top-level file or folder is a layout decision: add it to `FOLDERS` or `ROOT_FILES` in `audit_references.py` and to the map in the root [README](../README.md) in the same commit.
