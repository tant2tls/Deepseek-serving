# Reports: one folder per study

Every completed study keeps its report, tables, frozen plans and curated run evidence in its own folder here. What the measurements are for, and what is still planned, is in [target.md](../target.md).

| Study folder | What it measured | vLLM build | Node and date | Read first |
| --- | --- | --- | --- | --- |
| [blog-architecture-h100-v1/](blog-architecture-h100-v1/) | Attention, live KV and FFN of DeepSeek V4 Flash 0731, V4.1 Flash, MiMo-V2.6-Flash-MOPD, GLM-5.3-Flash and Qwen3.8-Flash-Next **FP8**. 90/90 valid timing runs, 30 traces, 30 live-KV snapshots | `554340f3…` | First node, 2026-10-07/08 | [findings](blog-architecture-h100-v1/findings.md) · [handoff](blog-architecture-h100-v1/handoff.md) · [architecture](blog-architecture-h100-v1/architecture.md) |
| [qwen-bf16-h100-v1/](qwen-bf16-h100-v1/) | Qwen3.8-Flash-Next in original BF16 with the same protocol, plus MiMo as a node control. 18/18 and 6/6 valid timing runs, 12 traces, 12 live-KV snapshots | `554340f3…` | Second node, 2026-10-08 | [findings](qwen-bf16-h100-v1/findings.md) · [handoff](qwen-bf16-h100-v1/handoff.md) · [BF16 against FP8 and the control](qwen-bf16-h100-v1/comparison.md) |
| [v41-vs-0731/](v41-vs-0731/) | DeepSeek V4.1 Flash against V4 Flash 0731: speculation off (144 valid runs per model), fixed and adaptive DSpark k5 (48/48), prefill traces, historical prefix checks | `44af287e…` | Earlier node, 2026-09-27/28 | [report](v41-vs-0731/report.md) |
| [mimo-v26/](mimo-v26/) | MiMo-V2.6-Flash-MOPD against both DeepSeek models: speculation off (90/90), MTP k3 and DFlash k7 (24/24), traces, historical prefix checks | `44af287e…` | Same earlier node, 2026-09-28 | [report](mimo-v26/report.md) |

[environment-lock.txt](environment-lock.txt) is the dependency lock of the `44af287e…` environment used by `v41-vs-0731` and `mimo-v26`; `install.sh` checks a fresh install against it.

**Reading across folders:**

- The two builds are never controls for each other: kernel backends and V4.1's prefill path differ. State the build beside every number, and the node once a result leaves its own study.
- The blog article selects from the two October studies by the rules in [target.md section 11](../target.md#11-current-blog-evidence-selection-2026-10-09): four first-node deployments plus Qwen BF16 from the second node. The Qwen FP8 rows of `blog-architecture-h100-v1` are historical reference only and are never relabelled BF16.
- The September studies used random-token prompts and select hypotheses only. Read them with the corrections in [update.md](../update.md).
- Each `data/<model>/CURATION.json` records what was compressed, sanitized or excluded. Frozen plans, raw runs, CSVs and curation ledgers are not edited.
- A new measurement gets a new study ID and its own folder; it is never appended to an existing one. The later study `spec-realtext-h100-v1` ([target.md section 12](../target.md#12-later-study-spec-realtext-h100-v1)) is planned, not measured, and has no folder yet.
