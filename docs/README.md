# Docs: what was measured, on which models, and how far the evidence goes

Background and rules for the measurements. Commands live in [reproduce/](../reproduce/README.md), numbers in [reports/](../reports/README.md), explanations of the method in [teach_me/](../teach_me/README.md).

## The October five-model comparison

| Document | Read it to learn |
| --- | --- |
| [models.md](models.md) | Each model's architecture, links to its model card and technical report, and what the runtime really executed |
| [experiments.md](experiments.md) | The protocol: workloads, request counts, validity rule, the two nodes, and what was **not** measured |
| [provenance.md](provenance.md) | How evidence was selected and preserved byte-for-byte, and what was left out of Git |

## Planned sessions

| Document | Read it to learn |
| --- | --- |
| [H200_B200_plan.md](H200_B200_plan.md) | How an agent measures on the 4×H200 and the 4×B200 server: the H100 session of [target.md](../target.md#the-15-hour-session) plus speculative decoding, with one checkpoint on disk at a time |

## Earlier studies, older runtime

These used vLLM `44af287ebe38d6dc4e102948025f5e3e175aefd6` in September. They are evidence and history. **A number from them is never a control for an October number**, and none of them is an active work queue.

| Document | Read it to learn |
| --- | --- |
| [previous-experiments.md](previous-experiments.md) | What the September DeepSeek, MiMo and GLM studies measured, and the corrections to read them with |
| [mimo-v2.6-inference.md](mimo-v2.6-inference.md) | A source-level walkthrough of MiMo's prefill and decode path on the September build |
| [mimo-v2.6-plan.md](mimo-v2.6-plan.md) | The plan the September MiMo study followed |
| [speculative-decoding.md](speculative-decoding.md) | DSpark in the DeepSeek checkpoints: which checkpoint supports which method, and how the September arms were set up |

## Where a new document goes

- A fact about a model (architecture, report, executed kernel): a section in [models.md](models.md).
- A rule about how something is measured or compared: [experiments.md](experiments.md).
- A change to which evidence is in the repository: [provenance.md](provenance.md), plus the ledger at `reports/provenance.json`.
- A command or a setup problem: [reproduce/](../reproduce/README.md), not here.
- A report on one study: beside its evidence in `reports/<study>/`, not here.

Keep one topic per file and link instead of repeating. The audit (`python tools/audit_references.py`) fails on a broken local link or anchor.
