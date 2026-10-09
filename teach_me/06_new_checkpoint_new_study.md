# Lesson 6: a new checkpoint of a measured model is a new study

On 2026-10-08 the Qwen rule changed: use the original **BF16** checkpoint `Qwen/Qwen3.8-Flash-Next`, never the `-FP8` one. The FP8 results were already published in `blog-architecture-h100-v1`. This lesson shows how the BF16 checkpoint was measured **without touching the finished study**, and how to tell whether numbers from two rented nodes can sit side by side. The result is study [`qwen-bf16-h100-v1`](../reports/qwen-bf16-h100-v1/findings.md).

## Why not just change the model ID?

Three things would break:

1. **History.** Rerunning under the old key overwrites or mixes FP8 and BF16 points in one curve. A reader can no longer tell which checkpoint produced a number.
2. **Identity.** The FP8 and BF16 repositories have different revisions. A frozen plan names one revision; substituting weights inside it makes the plan false.
3. **Controls.** The new run happens on another day and, here, on another rented node. A difference between "Qwen then" and "Qwen now" mixes the checkpoint with the node.

So the rule is: **new key, new study ID, same protocol, and a control that tells you what the node changed.**

## Step 1: a distinct key, and retire the old one

Two small edits, both in the repository:

```bash
grep -n -A4 "qwen-38-bf16)" bench/serve.sh          # model, pinned revision, parsers
grep -n -A4 '"qwen-38-bf16"' bench/run_matrix.py    # tokenizer mode and thinking switch
```

This branch carries only the BF16 key. The retired FP8 key, and the launcher check that refuses it, are kept as a record in the full archive (the `all-data` branch).

Note what the BF16 entry does **not** contain: no `--dtype`, no quantization flag. Native precision comes from the checkpoint, and you verify it from the log (step 4), not from the command line.

## Step 2: a study that reuses the protocol

`bench/blog_study.py` holds one entry per study in `STUDIES`: which keys may launch, which get a diagnostic launch, the timing blocks, and where the revisions are pinned. Every tool picks the study from one variable:

```bash
export BLOG_STUDY=qwen-bf16-h100-v1     # without it: the completed blog-architecture-h100-v1
```

Results go to `results/$BLOG_STUDY/` and reports to `reports/$BLOG_STUDY/`. Nothing under the old study's directories is written.

## Step 3: prove the inputs are the same

Comparable means the same requests. On the new node the public texts are downloaded again and the request lists rebuilt:

```bash
/tmp/blog-inputs-venv/bin/python bench/blog_corpus.py
$VLLM_VENV/bin/python bench/blog_inputs.py
$VLLM_VENV/bin/python bench/blog_inputs.py extra qwen-38-bf16     # token counts under the new tokenizer
```

The dry run then compares every list hash and every request hash with the **published** manifest of the first study, and refuses to proceed if one differs. On 2026-10-08 all nine lists matched, and the BF16 tokenizer gave the same count as the FP8 one on all 543 requests. If your node has another Python version, the code corpus changes and this check fails. That is the check doing its job.

## Step 4: verify what was loaded, from the log

```bash
L=$(ls results/$BLOG_STUDY/qwen-38-bf16/_server/serve-*.log | tail -1)
grep -m1 -o "dtype=torch.[a-z0-9]*, max_seq_len\|quantization=[A-Za-z0-9]*" $L   # bfloat16, None
grep -m1 "MoE backend" $L        # Using TRITON Unquantized MoE backend ...
grep -m1 "PLE embedding" $L      # ... weight_dtype=torch.bfloat16 ...
grep -c -i fp8 $L                # 0
```

A config file that says `bfloat16` is source evidence. These lines are execution evidence. Only the second kind lets you write "BF16 was measured".

## Step 5: a node control

The two nodes were both "8×H100 80GB", yet `results/$BLOG_STUDY/_env/` showed another CPU model, four NUMA nodes instead of two and a newer GPU driver. Rerunning the FP8 checkpoint would be the direct control, but the policy forbids new FP8 runs. Instead **one model that was already measured (MiMo) runs one timing block** with the same requests and counts:

```bash
bash bench/blog_launch.sh chain-timing timing:qwen-38-bf16:1 timing:mimo-v26:1 \
     timing:qwen-38-bf16:2 timing:qwen-38-bf16:3
python bench/blog_report.py serving && python bench/blog_report.py compare
```

`compare` writes `reports/$BLOG_STUDY/comparison.md`: MiMo here against its three blocks on the first node, point by point, then BF16 Qwen against the historical FP8 Qwen. Read the first table before the second. The control is reported as a ratio. It is never used to rescale a result.

What the control showed on 2026-10-08:

| Quantity (MiMo, second node / first node) | Result | Use across nodes? |
| --- | --- | --- |
| Output throughput, six points | 0.983–0.995 | Yes, with a 2% margin |
| TPOT p50, one client | 0.98–1.00 | Yes |
| TTFT p50 | 1.02–1.30 | **No** |
| TPOT p50, eight clients, 16K and 64K | 0.86–0.88 | **No** |

Throughput survived the change of node; the split between "time to first token" and "time between tokens" did not. Without the control, a 29% higher TTFT at 64K with eight clients would have been blamed on BF16. MiMo moved by 30% at the same point.

Look inside the run as well. The per-request TTFT list of the 1K, one-client point switched between two levels on this node (about 190 and 120 ms), so its median depends on where the switch fell:

```bash
python3 -c "import json;print([round(1e3*x) for x in json.load(open('results/$BLOG_STUDY/qwen-38-bf16/serving/off/ctx1k_osl256_c1/repeat-1/bench.json'))['ttfts']])"
```

**A trace control answers a different question.** The BF16 traces showed twice the all-reduce time of the FP8 traces. Node or checkpoint? One more diagnostic launch of MiMo (declared in an addendum before it ran) gave the answer: MiMo's components here matched its earlier ones within 3%, all-reduce within about 10%. So the node was not the cause.

```bash
bash bench/blog_launch.sh chain-mimo-diag diag:mimo-v26 --diag-parts functional,kv,trace
python bench/blog_report.py components && python bench/blog_report.py tables
```

Kernels that do not depend on the checkpoint's precision (attention core, indexer, recurrent update) are a free extra check: they cost the same in the BF16 and FP8 traces to the second decimal.

## Step 6: say exactly what the comparison is

BF16 against FP8 here is **two deployments**: another checkpoint, another expert kernel (the build picks Triton for unquantized experts and a FlashInfer kernel for FP8), another KV pool size because the weights are larger, another node and day. It is not a controlled test of "precision". Write it that way.

## Check yourself

- Which file would you read to learn which keys a study may launch?
- Why does the dry run compare request hashes with a published file instead of trusting that the same script was used?
- The server log shows `quantization=None`. Which other two lines do you check before writing "BF16 experts were executed"?
- MiMo is 2% slower on the new node at one point. Do you divide the Qwen numbers by 0.98? (No. Report the ratio next to the comparison.)
- BF16 shows a 29% higher TTFT than FP8 at 64K with eight clients. What do you check before writing that down?
- A colleague wants to add BF16 points to the FP8 figure "because it is the same model". What do you answer?
