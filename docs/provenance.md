# Evidence selection and provenance

This branch is a focused publication view of the existing repository at parent commit [`be9ee6abfc251d8145c44a375cd2de8907228513`](https://github.com/tant2tls/Deepseek-serving/tree/be9ee6abfc251d8145c44a375cd2de8907228513). The original branch, reports and frozen ledgers remain in history. The current article layout incorporates the existing local editorial draft; no numerical result was recollected or adjusted.

Retained per-model raw evidence, including numerical JSONs, compressed logs, failure records and `CURATION.json` ledgers, is copied byte-for-byte. [provenance.json](../provenance.json) records the source and exported SHA-256 of every retained evidence file. Existing curation ledgers record the earlier lossless compression and masking of private IPs/terminal formatting. This branch adds no new sanitization to raw records.

The first-node `serving.csv`, `memory.csv` and `components.csv` are **derived row selections** for 0731, V4.1, MiMo and GLM. Retained row strings and values are unchanged. The first-node extra-token map keeps only GLM. The unclassified-kernel report keeps only the selected deployments' sections. These transformations are explicit in the new ledger; they are not substitutions of one checkpoint's numbers for another's.

The second-node CSVs retain Qwen and MiMo control rows unchanged. The five-model generator selects Qwen for the main comparison and reports MiMo's node control separately. The earlier DeepSeek/MiMo study directories and their plans/curation ledgers are retained unchanged.

Shared October plans, chain logs and study-level ledgers that describe deployments outside this branch's selection are omitted from its current tree, rather than edited and presented as frozen originals. Their exact source versions remain at the parent commit:

- [First-node source directory and frozen addenda](https://github.com/tant2tls/Deepseek-serving/tree/be9ee6abfc251d8145c44a375cd2de8907228513/reports/blog-architecture-h100-v1).
- [Second-node frozen plan](https://github.com/tant2tls/Deepseek-serving/blob/be9ee6abfc251d8145c44a375cd2de8907228513/reports/qwen-bf16-h100-v1/plan.json).
- [Second-node diagnostic-control addendum](https://github.com/tant2tls/Deepseek-serving/blob/be9ee6abfc251d8145c44a375cd2de8907228513/reports/qwen-bf16-h100-v1/plan_addendum_mimo_trace_control.json).

The selected [protocol](experiments.md) documents request counts, launch order, validity and meaningful deviations. This selection is not the full first-node study and does not redefine its completed ledger. Non-selected checkpoint data, unrelated future-study planning and mixed historical comparisons are outside this branch's current contents. Git ancestry remains intact.

The [September GLM archive](../references/README.md) is retained separately with 42 result JSONs and its source/export hashes. Its `arms.json`, `results.csv` and `provenance.json` are explicitly selected views of the original reference index. The numerical JSONs, benchmark logs and startup logs remain byte-for-byte identical to the existing export; the reference ledger records the original sanitization. These points are never used in the October figures.

Weights, download/compiler caches, Python environments, credentials, private host details, full torch profiler traces and source-text request bodies are absent. Public input hashes and source metadata are retained. The code corpus was collected from the node's installed CPython 3.12.3 standard library, not an immutable upstream checkout: reproducing the exact distro-patched files is a known requirement. Match the published corpus and request-list hashes rather than assuming another Python 3.12 installation is identical.

Run `python tools/audit_references.py` on a clean clone. The audit checks all retained source/export hashes, original per-model curation hashes, 90 selected timing runs plus six controls against raw benchmarks, input identities, snapshot arithmetic, trace/rank counts, generated tables, local links and targeted credential/path scans. It does not prove scientific validity or promise bit-identical timing on different hardware.
