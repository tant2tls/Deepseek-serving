#!/usr/bin/env python3
"""Tables for a blog-protocol study from results/<study>/ (standard library only).

BLOG_STUDY selects the study: blog-architecture-h100-v1 (default, completed) or qwen-bf16-h100-v1.

  python bench/blog_report.py pilot        # request counts implied by the disjoint pilot
  python bench/blog_report.py serving      # reports/<study>/serving.csv + serving_tables.md
  python bench/blog_report.py memory       # reports/<study>/memory.csv (live KV snapshots + pools)
  python bench/blog_report.py components   # reports/<study>/components.csv (all-rank trace breakdown)
  python bench/blog_report.py tables       # components_tables.md and memory_tables.md from the two CSVs
Missing values are written as empty cells, never zero.
"""
import collections, csv, glob, gzip, json, math, os, re, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_breakdown import CATS, cat as _cat

# Kernels of the 2026-10-07 nightly that the historical classifier (bench/trace_breakdown.py,
# kept unchanged for the published tables) leaves unclassified. Checked first.
EXTRA = [
    ("moe_route_combine", r"preprocessTopkIdKernel|finalizeMoeRoutingKernel|computeExpertOffsets|MoeRouting|moe_"),
    ("gemm_input_prep", r"process_input_kernel|per_token_group_quant"),
    ("moe_route_combine", r"act_and_mul_kernel"),  # SwiGLU activation, as in the historical classifier
    ("indexer_topk", r"topKPerRow|_block_scores_kernel|_mask_candidates_kernel"),
    ("qkv_rope_kvcache", r"get_mla_metadata_kernel|_zero_kv_blocks_kernel|_gather_block_tables|_compute_slot_mappings"),
    ("dense_gemm", r"cublasLt|splitKreduce|gemmk1|sgemm|gemv"),
]
# The `humming` low-bit GEMM serves both MoE experts and (on DeepSeek) low-rank attention
# projections; they are told apart by weight shape (config: hidden x 2*moe_intermediate and back).
MOE_SHAPES = {"mimo-v26": {(4096, 4096), (4096, 2048)}, "v4-0731": {(4096, 4096), (4096, 2048)},
              "v41": {(4608, 5120), (5120, 2304)}}
MODEL = None  # set per trace: MiMo has no attention indexer, so its top-k kernels belong to the MoE router


# Model-specific kernels of the two added models, checked before everything else. Assigned from
# kernel names and the checkpoint config (expert shapes); see architecture.md for the layer types.
MODEL_RULES = {
    # Original BF16 checkpoint. The experts run in Triton's
    # `fused_moe_kernel` (two calls per layer, log: "TRITON Unquantized MoE backend"); dense
    # projections are cuBLAS `nvjet_*` GEMMs, already classified by the shared rules.
    "qwen-38-bf16": [
        ("moe_expert_gemm", r"^fused_moe_kernel"),
        ("moe_route_combine", r"moe_align_block_size|count_and_sort_expert_tokens|moe_sum_vec"),
        ("moe_route_combine", r"topkGating|expandInputRowsKernel|finalizeMoeRouting|doActivationKernel"),
        ("recurrent_attention", r"delta_rule|gated_delta|_causal_conv1d|_fused_post_conv|gdn"),
        ("attention_core", r"_qsa_sparse_paged"),
        ("indexer_topk", r"_qsa_mqa_paged|FilteredTopK|cooperative_topk|_expand_qsa_indices"),
        ("mhc_residual_norm", r"_hc_|HcDownSilu|hyper_connection"),
    ],
    "glm-53": [
        ("moe_expert_gemm", r"fp8_gemm_kernel(_swapAB)?<(4096u, 4096u|4096u, 2048u)"),
        ("gemm_input_prep", r"scale_1x128_kernel"),
        ("moe_route_combine", r"single_group_topk|topkGating|expandInputRowsKernel|finalizeMoeRouting|doActivationKernel"),
        ("recurrent_attention", r"_flash_kda|kda_fwd|delta_rule|gated_delta|_causal_conv1d"),
        ("attention_core", r"MLAPageAttention"),
        ("indexer_topk", r"mqa_logits|topKPerRow|cooperative_topk"),
        ("mhc_residual_norm", r"mhc_|hc_prenorm"),
    ],
}


def cat(name):
    for c, pat in MODEL_RULES.get(MODEL, ()):
        if re.search(pat, name):
            return c
    m = re.match(r"void humming<\w+, Shape<0u, (\d+)u, (\d+)u>", name)
    if m:
        return "moe_expert_gemm" if (int(m[1]), int(m[2])) in MOE_SHAPES.get(MODEL, ()) else "dense_gemm"
    for c, pat in EXTRA:
        if re.search(pat, name):
            return c
    c = _cat(name)
    if MODEL == "mimo-v26" and c == "indexer_topk":
        return "moe_route_combine"
    return c


ROOT = Path(__file__).resolve().parent.parent
BLOG = "blog-architecture-h100-v1"
STUDY = os.environ.get("BLOG_STUDY", BLOG)
SDIR = ROOT / "results" / STUDY
RDIR = ROOT / "reports" / STUDY
RERUN = "five-model-rerun-h100-v1"  # the 15-hour session: 2,048 output tokens, 1 and 16 clients
SESSION = STUDY == RERUN
MODELS = {BLOG: ["v4-0731", "v41", "mimo-v26", "glm-53"],
          "qwen-bf16-h100-v1": ["qwen-38-bf16", "mimo-v26"],
          RERUN: ["v4-0731", "v41", "mimo-v26", "qwen-38-bf16", "glm-53"]}[STUDY]
LOADS = (1, 16) if SESSION else (1, 8)
OSL = 2048 if SESSION else 256
NAME = {"v4-0731": "V4 Flash 0731", "v41": "V4.1 Flash", "mimo-v26": "MiMo-V2.6-Flash",
        "glm-53": "GLM-5.3-Flash",
        "qwen-38-bf16": "Qwen3.8-Flash-Next"}
BUCKETS = ["1k", "16k", "64k"]
CATN = [c for c, _ in CATS] + ["gemm_input_prep", "recurrent_attention", "other_elementwise"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def runs(workload, cfg):
    for mk in MODELS:
        for d in sorted((SDIR / mk / workload / cfg).glob("ctx*/repeat-*")):
            s, v, m = load(d / "summary.json"), load(d / "blog_validation.json"), load(d / "manifest.json")
            if s and v:
                yield mk, d, s, v, m


def wcsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print("wrote", path.relative_to(ROOT), len(rows), "rows")


def pilot():
    rate = collections.defaultdict(dict)
    for mk, d, s, v, _ in runs("pilot", "off-profidle"):
        rate[(v["bucket"], v["concurrency"])][mk] = (s.get("request_throughput"), s.get("valid"), s.get("output_throughput"),
                                                     s.get("median_ttft_ms"), s.get("median_tpot_ms"))
    counts = {}
    for (b, c), by in sorted(rate.items()):
        fastest = max(x[0] for x in by.values() if x[0])
        n = max(16, 4 * c, math.ceil(75 * fastest))  # 60 s rule with margin on the fastest piloted model
        n = 3 * math.ceil(n / 3)
        counts[f"{b}:c{c}"] = n
        print(f"{b} c{c}: n={n}  " + "  ".join(f"{mk}: {x[0]:.3f} req/s valid={x[1]} tok/s={x[2]:.1f} "
                                             f"ttft={x[3]:.0f} tpot={x[4]:.2f}" for mk, x in by.items()))
    print(json.dumps(counts))


def serving():
    rows = []
    for mk, d, s, v, m in runs("serving", "off"):
        rr = v.get("running_requests_1hz") or []
        rows.append(dict(
            study=STUDY, model=mk, runtime=v["runtime"]["vllm"], bucket=v["bucket"], concurrency=v["concurrency"],
            block=v["block"], attempt=d.name, valid=v["valid"], n=v["n"],
            list_sha256=v["list_file_sha256"][:16], duration_s=s.get("duration_s"),
            server_prompt_tokens_per_request=(s["server_prompt_tokens"] / v["n"]) if s.get("server_prompt_tokens") else None,
            output_tok_s=s.get("output_throughput"), requests_s=s.get("request_throughput"),
            ttft_mean_ms=s.get("mean_ttft_ms"), ttft_p50_ms=s.get("median_ttft_ms"),
            tpot_mean_ms=s.get("mean_tpot_ms"), tpot_p50_ms=s.get("median_tpot_ms"),
            e2el_mean_ms=s.get("mean_e2el_ms"), e2el_p50_ms=s.get("median_e2el_ms"),
            running_mean=(sum(rr) / len(rr)) if rr else None, running_max=max(rr) if rr else None,
            peak_kv_usage_frac=s.get("peak_kv_cache_usage_frac"), preemptions=s.get("preemptions"),
            problems="; ".join(v["problems"]), evidence=str(d.relative_to(ROOT))))
    wcsv(RDIR / "serving.csv", rows)
    # One valid attempt per planned block: the last attempt directory of that block that is valid.
    pick = {}
    for r in rows:
        if r["valid"]:
            pick[(r["model"], r["bucket"], r["concurrency"], r["block"])] = r
    global MODELS
    all_models, MODELS = MODELS, [m for m in MODELS if any(k[0] == m for k in pick)]
    out = [f"# Serving controls: AR, prefix caching off, {OSL} forced output tokens\n",
           "Mean ± sample SD over the valid repeat blocks, then median and the individual block values. "
           "Three blocks screen effects; they are not confidence intervals.\n"]
    for metric, label, fmt in (("output_tok_s", "Output tok/s", "{:.1f}"), ("ttft_p50_ms", "TTFT p50 (ms)", "{:.0f}"),
                               ("tpot_p50_ms", "TPOT p50 (ms)", "{:.2f}"), ("e2el_p50_ms", "End-to-end p50 (ms)", "{:.0f}"),
                               ("requests_s", "Requests/s", "{:.3f}")):
        out += [f"\n## {label}\n", "| Input | c | " + " | ".join(NAME[m] for m in MODELS) + " |",
                "| --- | ---: | " + " | ".join("---" for _ in MODELS) + " |"]
        for b in BUCKETS:
            for c in LOADS:
                cells = []
                for mk in MODELS:
                    vals = [r[metric] for (m2, b2, c2, _), r in sorted(pick.items())
                            if (m2, b2, c2) == (mk, b, c) and r[metric] is not None]
                    if not vals:
                        cells.append("pending"); continue
                    if len(vals) == 1:
                        cells.append(f"{fmt.format(vals[0])} (1 block)"); continue
                    sd = st.stdev(vals)
                    cells.append(f"{fmt.format(st.mean(vals))} ± {fmt.format(sd)}; med {fmt.format(st.median(vals))} "
                                 f"[{', '.join(fmt.format(x) for x in vals)}]")
                out.append(f"| {b} | {c} | " + " | ".join(cells) + " |")
    out += ["\n## Actual input length (server-counted prompt tokens per request, mean)\n",
            "| Input | c | " + " | ".join(NAME[m] for m in MODELS) + " |", "| --- | ---: | " + " | ".join("---:" for _ in MODELS) + " |"]
    for b in BUCKETS:
        for c in LOADS:
            cells = []
            for mk in MODELS:
                vals = [r["server_prompt_tokens_per_request"] for (m2, b2, c2, _), r in pick.items()
                        if (m2, b2, c2) == (mk, b, c) and r["server_prompt_tokens_per_request"]]
                cells.append(f"{st.mean(vals):.0f}" if vals else "pending")
            out.append(f"| {b} | {c} | " + " | ".join(cells) + " |")
    (RDIR / "serving_tables.md").write_text("\n".join(out) + "\n")
    MODELS = all_models
    print("wrote", (RDIR / "serving_tables.md").relative_to(ROOT), "valid points:", len(pick))


def session_pilot():
    """Pilot of the session's serving points: validity and the run time each full point implies."""
    plan = load(SDIR / "plan.json") or {}
    rows = []
    for mk, d, s, v, _ in runs("pilot", "off-profidle"):
        full = (plan.get("counts") or {}).get(f"{v['bucket']}:c{v['concurrency']}")
        dur = s.get("duration_s")
        # One client: time scales with requests. Sixteen: with waves of 16 (the 64K pilot holds nine).
        scale = (full / v["n"]) if full and v["concurrency"] == 1 else ((full / 16) if full else None)
        rows.append(dict(study=STUDY, model=mk, runtime=v["runtime"]["vllm"], bucket=v["bucket"],
                         concurrency=v["concurrency"], n=v["n"], valid=v["valid"], duration_s=dur,
                         output_tok_s=s.get("output_throughput"), ttft_p50_ms=s.get("median_ttft_ms"),
                         tpot_p50_ms=s.get("median_tpot_ms"), planned_n=full,
                         estimated_full_run_s=(dur * scale) if dur and scale else None,
                         peak_kv_usage_frac=s.get("peak_kv_cache_usage_frac"), preemptions=s.get("preemptions"),
                         problems="; ".join(v["problems"]), evidence=str(d.relative_to(ROOT))))
    wcsv(RDIR / "pilot.csv", rows)


def session_intervals():
    """Unprofiled decode intervals of the plain timing launches, one row per condition and block."""
    rows = []
    for mk in MODELS:
        for f in sorted((SDIR / mk / "decode_interval" / "off").glob("ctx*/repeat-*/interval.json")):
            k = load(f)
            w = k.get("interval") or {}
            ttft = [x for x in (k.get("progress", {}).get("client_ttft_s") or []) if x is not None]
            rows.append(dict(study=STUDY, model=mk, runtime=(k.get("runtime") or {}).get("vllm"), bucket=k.get("bucket"),
                             engine_B=k.get("B"), block=k.get("block"), attempt=f.parent.name, valid=k.get("valid"),
                             server_prompt_tokens_total=k.get("server_prompt_tokens_total"),
                             interval_tokens=w.get("tokens"), interval_s=w.get("window_s"),
                             ms_per_step=w.get("ms_per_token_step"), engine_steps=w.get("engine_steps"),
                             ms_per_engine_step=w.get("ms_per_engine_step"),
                             tokens_per_engine_step=w.get("tokens_per_engine_step"),
                             time_to_all_decoding_s=k.get("time_to_all_decoding_s"),
                             client_ttft_first_s=min(ttft) if ttft else None,
                             client_ttft_last_s=max(ttft) if ttft else None, kv_usage_frac=k.get("kv_usage_live"),
                             preemptions=k.get("preemptions"), problems="; ".join(k.get("problems") or []),
                             evidence=str(f.parent.relative_to(ROOT))))
    wcsv(RDIR / "decode_intervals.csv", rows)


def session_natural():
    """The twelve natural-ending answers per model: did they stop, how long, how much was reasoning."""
    rows = []
    for mk in MODELS:
        k = load(SDIR / mk / "_functional" / "functional.json")
        for c in (k or {}).get("checks", []):
            u = c.get("usage") or {}
            rows.append(dict(study=STUDY, model=mk, runtime=k["runtime"]["vllm"], kind=c["kind"], domain=c.get("domain"),
                             task=c.get("task"), finish_reason=c.get("finish_reason"),
                             prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                             reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                             answer_chars=len(c.get("text") or ""), reasoning_chars=len(c.get("reasoning") or ""),
                             wall_s=c.get("wall_s"), error=c.get("error"),
                             evidence=f"results/{STUDY}/{mk}/_functional/functional.json"))
    wcsv(RDIR / "natural.csv", rows)


def run_windows(bench, c):
    """Position windows and the steady-state interval of one run, from the client's per-token times.

    Position windows: mean inter-token time over generated positions 2-256, 257-1,024 and 1,025-2,048
    (position 1 is the first token). Steady state, declared before collection: from the moment the
    first c requests have all produced their first token until the first request finishes after the
    last request was sent; both denominators are kept."""
    itls, ttfts, starts, lat = bench["itls"], bench["ttfts"], bench["start_times"], bench["latencies"]
    flat = lambda xs: [float(sum(x)) if isinstance(x, list) else float(x) for x in xs]  # a chunk can carry several tokens
    per = [flat(x) for x in itls]
    win = {}
    for name, a, b in (("pos_2_256", 0, 255), ("pos_257_1024", 255, 1023), ("pos_1025_2048", 1023, 2047)):
        vals = [sum(x[a:b]) / len(x[a:b]) for x in per if len(x[a:b]) > 0]
        win[f"itl_ms_{name}"] = 1e3 * sum(vals) / len(vals) if vals else None
    t0 = min(starts)
    first = [s - t0 + t for s, t in zip(starts, ttfts)]
    end = [s - t0 + l for s, l in zip(starts, lat)]
    order = sorted(range(len(starts)), key=lambda i: starts[i])
    ramp_end = max(first[i] for i in order[:c])
    last_start = max(starts) - t0
    after = [e for e in end if e >= last_start]
    drain = min(after) if after else max(end)
    tokens = 0
    for i, x in enumerate(per):
        t = first[i]
        tokens += ramp_end <= t <= drain
        for d in x:
            t += d
            tokens += ramp_end <= t <= drain
    span = drain - ramp_end
    return dict(**win, steady_start_s=ramp_end, steady_end_s=drain, steady_tokens=tokens,
                steady_output_tok_s=(tokens / span) if span > 0 else None, full_run_s=bench["duration"],
                full_output_tok_s=bench["output_throughput"], steady_share_of_run=(span / bench["duration"]) if span > 0 else None)


def session_windows():
    """Per valid serving run: generated-position windows and steady-state throughput (see run_windows)."""
    rows = []
    for mk, d, s, v, m in runs("serving", "off"):
        if not v["valid"]:
            continue
        try:
            bench = json.loads((d / "bench.json").read_text())
            w = run_windows(bench, v["concurrency"])
        except Exception as e:
            print("windows failed", d, repr(e)); continue
        rows.append(dict(study=STUDY, model=mk, runtime=v["runtime"]["vllm"], bucket=v["bucket"],
                         concurrency=v["concurrency"], block=v["block"], attempt=d.name, n=v["n"], **w,
                         evidence=str(d.relative_to(ROOT))))
    wcsv(RDIR / "serving_windows.csv", rows)


def session_routing():
    """Expert-routing statistics per model and condition (the per-expert counts stay in routing.json)."""
    rows = []
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    for mk in MODELS:
        k = load(SDIR / mk / "_routing" / "routing.json")
        for name, c in ((k or {}).get("conditions") or {}).items():
            routed = c["routed_layers"]
            rows.append(dict(
                study=STUDY, model=mk, runtime=k["runtime"]["vllm"], condition=name, n_experts=c["n_experts"], k=c["k"],
                nominal_share=c["nominal_share"], layers=c["layers"], routed_layers=len(routed), steps=c["steps"],
                tokens_per_step_mean=c["tokens_per_step_mean"], distinct_per_step_mean=c["distinct_per_step_mean"],
                distinct_per_step_min=c["distinct_per_step_min"], distinct_per_step_max=c["distinct_per_step_max"],
                share_of_experts_per_step_mean=c["share_of_experts_per_step_mean"],
                coverage_mean_over_routed_layers=mean([c["coverage_by_layer"][l] for l in routed]),
                tokens_per_expert_max_over_mean=mean(c["tokens_per_expert"]["max_over_mean_by_layer"]),
                tokens_per_expert_cv=mean(c["tokens_per_expert"]["cv_by_layer"]),
                unused_experts_mean=mean(c["tokens_per_expert"]["unused_experts_by_layer"]),
                gpu_load_peak_over_mean_per_step=c["gpu_load_peak_over_mean"]["per_step_mean"],
                gpu_load_peak_over_mean_whole=mean(c["gpu_load_peak_over_mean"]["whole_condition_by_layer"]),
                output_agrees_with_plain_launch=k["agreement_with_plain_launch"]["same_text"],
                evidence=f"results/{STUDY}/{mk}/_routing/routing.json"))
    wcsv(RDIR / "routing.csv", rows)


SUPPORT = (("class", r"Resolved architecture: (\S+)"), ("engine", r"Initializing a V1 LLM engine \((v[^)]+)\)"),
           ("weights_gib_per_gpu", r"Model loading took ([\d.]+) GiB"), ("kv_pool_gib_per_gpu", r"Available KV cache memory: ([\d.]+) GiB"),
           ("kv_pool_tokens", r"GPU KV cache size: ([\d,]+) tokens"), ("max_concurrency_262k", r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
           ("kv_block", r"Setting kv cache block size to (\d+ for .*?) backend"), ("kv_layout", r"Using (\S+) KV cache layout"),
           ("kv_format", r"Using ([^\n]*KV cache format)"), ("kv_group_block_sizes", r"kv cache group sizes (\([^)]*\))"),
           ("kv_lcm_block_size", r"kv lcm block sizes (\d+)"),
           ("hidden_state_cache", r"(Using block size \d+ for hidden-state cache layer[^\n]*)"),
           ("max_model_len_fit", r"(Auto-fit max_model_len[^\n]*)"),
           ("moe_backend", r"Using '?([A-Za-z0-9_ ]+?)'? (?:Mxfp4 |Fp8 |Unquantized |Nvfp4 )?MoE backend"),
           ("moe_backend_line", r"((?:Using|Selected) [^\n]*MoE backend[^\n]*)"), ("experts_class", r"Using (\w*Experts\w*)"),
           ("expert_dtype", r"expert_dtype resolved to '([^']+)'"), ("chunked_prefill", r"Chunked prefill is enabled with (max_num_batched_tokens=\d+)"),
           ("prefix_caching", r"'enable_prefix_caching': (\w+)"), ("allreduce_tp", r"Using (\[[^\]]+\]) all-reduce backends \(in dispatch order\) for group 'tp:0'"),
           ("allreduce_ep", r"Using (\[[^\]]+\]) all-reduce backends \(in dispatch order\) for group 'ep:0'"),
           ("fusions", r"Enabled custom fusions: ([^\n]+)"), ("checkpoint_gib", r"Checkpoint size: ([\d.]+) GiB"),
           ("cudagraph_gib", r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB"),
           ("memory_line", r"(Free memory on device[^\n]*?CUDAGraph memory)"), ("attention_warmup", r"(Warming up [^\n]*attention[^\n]*)"),
           ("engram", r"([^\n]*[Ee]ngram[^\n]*)"), ("quantization", r"([^\n]*(?:quantiz|Quantiz|dtype resolved|kv_cache_dtype|Fp8|FP8)[^\n]*)"),
           ("attention_backend", r"([^\n]*(?:attention backend|Attention backend|attn backend|AttentionBackend)[^\n]*)"))


def session_support():
    """Architecture and support check: what each server log says the build loaded (executed implementation)."""
    rows = []
    for mk in MODELS:
        logs = sorted((SDIR / mk / "_server").glob("serve-off-profidle-*.log"))
        if not logs:
            continue
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", logs[-1].read_text(errors="replace").replace("\r", "\n"))
        text = re.sub(r"^\((?:\w+) pid=\d+\) ", "", text, flags=re.M)
        ready = re.search(r"Application startup complete", text)
        for key, pat in SUPPORT:
            vals = list(dict.fromkeys(m if isinstance(m, str) else m[0] for m in re.findall(pat, text)))
            for v in vals[:6]:
                v = re.sub(r"^(?:INFO|WARNING) \d\d-\d\d \d\d:\d\d:\d\d \[[^\]]+\] ", "", v.strip())
                rows.append(dict(study=STUDY, model=mk, fact=key, value=v[:400], reached_ready=bool(ready),
                                 evidence=f"results/{STUDY}/{mk}/_server/{logs[-1].name}"))
    wcsv(RDIR / "support.csv", rows)


def session_memory():
    """Live state at the eight decode conditions: usage gauge x per-rank pool, tokens counted by the server."""
    pool = pools()
    rows = []
    for mk in MODELS:
        for f in sorted((SDIR / mk / "_kv").glob("live_*.json")):
            k = load(f)
            usage, idle, p = k.get("kv_usage_live"), k.get("kv_usage_idle"), pool.get(mk, {})
            tokens = (k.get("server_prompt_tokens_total") or 0) + (k.get("generated_tokens_total") or 0)
            live = usage * p["kv_cache_gib"] if usage is not None and p.get("kv_cache_gib") else None
            # The same batch a few steps later (start of the unprofiled decode window). Window layers
            # hold a whole prefill chunk until the first decode step frees what lies outside the window,
            # so the reading at the first token can be higher than the state held while decoding.
            man = load(SDIR / mk / "profiles" / f"decode{k['bucket']}_B{k['B']}.manifest.json") or {}
            dec = (man.get("unprofiled_window") or {}).get("kv_usage_start")
            live_dec = dec * p["kv_cache_gib"] if dec is not None and p.get("kv_cache_gib") else None
            rows.append(dict(
                study=STUDY, model=mk, runtime=k["runtime"]["vllm"], bucket=k["bucket"], live_sequences=k["B"],
                capacity_limit=k.get("capacity_limit"), live_tokens_server_prompt_plus_generated=tokens,
                kv_usage_frac_decoding=dec, live_gib_per_rank_decoding=live_dec,
                live_gib_8_ranks_decoding=live_dec * 8 if live_dec is not None else None,
                live_kib_per_token_per_rank_decoding=(live_dec * 2**20 / tokens) if live_dec is not None and tokens else None,
                client_content_tokens=sum(k["client_content_tokens"]), kv_usage_frac=usage, kv_usage_frac_idle=idle,
                pool_gib_per_rank=p.get("kv_cache_gib"), pool_tokens=p.get("kv_cache_tokens"),
                weights_gib_per_rank=p.get("model_weights_gib"), live_gib_per_rank=live,
                live_gib_8_ranks=live * 8 if live is not None else None,
                live_kib_per_token_per_rank=(live * 2**20 / tokens) if live is not None and tokens else None,
                gpu_mem_mib_rank0=k["gpu_mem_mib"][0], gpu_mem_mib_sum=sum(k["gpu_mem_mib"]),
                time_to_all_decoding_s=k.get("time_to_all_decoding_s"), evidence=str(f.relative_to(ROOT))))
    wcsv(RDIR / "memory.csv", rows)


def pools():
    """KV pool facts from each model's newest server log."""
    info = {}
    for mk in MODELS:
        logs = sorted((SDIR / mk / "_server").glob("serve-*.log"))
        rec = {}
        for lg in logs:
            for line in open(lg, errors="replace"):
                for key, pat in (("kv_cache_tokens", r"GPU KV cache size: ([\d,]+) tokens"),
                                 ("kv_cache_gib", r"Available KV cache memory: ([\d.]+) GiB"),
                                 ("model_weights_gib", r"Model loading took ([\d.]+) GiB"),
                                 ("kv_blocks", r"num_gpu_blocks[ =:]+(\d+)")):
                    m = re.search(pat, line)
                    if m:
                        rec[key] = float(m.group(1).replace(",", ""))
        info[mk] = rec
    return info


def memory():
    if SESSION:
        return session_memory()
    pool = pools()
    rows = []
    for mk in MODELS:
        for f in sorted((SDIR / mk / "_kv").glob("live_*.json")):
            k = load(f)
            usage = k["live_gauges"].get("vllm:kv_cache_usage_perc")
            idle = k["idle_gauges"].get("vllm:kv_cache_usage_perc")
            p = pool.get(mk, {})
            tot_tokens = sum(k["client_content_tokens"]) + sum(k["generated_at_snapshot"])
            live_gib = usage * p["kv_cache_gib"] if usage is not None and p.get("kv_cache_gib") else None
            rows.append(dict(
                study=STUDY, model=mk, runtime=k["runtime"]["vllm"], bucket=k["bucket"], live_sequences=k["B"],
                live_tokens_client_content_plus_generated=tot_tokens, kv_usage_frac=usage, kv_usage_frac_idle=idle,
                pool_gib_per_rank=p.get("kv_cache_gib"), pool_tokens=p.get("kv_cache_tokens"),
                live_gib_per_rank=live_gib, live_gib_8_ranks=live_gib * 8 if live_gib is not None else None,
                live_kib_per_token_per_rank=(live_gib * 2**20 / tot_tokens) if live_gib is not None else None,
                gpu_mem_mib_rank0=k["gpu_mem_mib"][0], gpu_mem_mib_sum=sum(k["gpu_mem_mib"]),
                evidence=str(f.relative_to(ROOT))))
    wcsv(RDIR / "memory.csv", rows)
    print(json.dumps(pool, indent=1))


def breakdown(trace, until_next=False):
    """Per-step kernel sums of one rank trace, same rules as bench/trace_breakdown.py.

    until_next=True (decode windows): a step owns every kernel from its span start to the next
    step's span start. V4.1 splits one decode step over several streams and graph segments, so
    counting only kernels inside the main-stream span misses part of the step."""
    op = gzip.open if str(trace).endswith(".gz") else open
    ev = json.load(op(trace))["traceEvents"]
    kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    gs = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context")]
    if not gs:
        return [], kern
    by_tid = collections.defaultdict(float)
    for e in gs:
        by_tid[e["tid"]] += e["dur"]
    tid = max(by_tid, key=by_tid.get) if until_next else max(gs, key=lambda e: e["dur"])["tid"]
    spans = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in gs if e["tid"] == tid)
    if until_next:
        # One CPU-side annotation exists per engine step, whatever the number of GPU segments.
        cpu = sorted((e["ts"], e["name"]) for e in ev
                     if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context"))
        spans = [(cpu[i][0], cpu[i + 1][0], cpu[i][1]) for i in range(len(cpu) - 1)]
        spans = spans[1:] if len(spans) > 4 else spans  # the first step can be cut by profiler start
    rows, i = [], 0
    for s, t, name in spans:
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", name)
        by, cnt, names = collections.Counter(), collections.Counter(), collections.Counter()
        while i < len(kern) and kern[i]["ts"] < s:
            i += 1
        j = i
        while j < len(kern) and kern[j]["ts"] < t:
            c = cat(kern[j]["name"]); by[c] += kern[j]["dur"]; cnt[c] += 1
            if c == "other_elementwise":
                names[kern[j]["name"][:90]] += kern[j]["dur"]
            j += 1
        if j > i and m:
            rows.append(dict(prefill_reqs=int(m[1]), prefill_tokens=int(m[2]), decode_reqs=int(m[3]),
                             decode_tokens=int(m[4]), span_ms=(t - s) / 1e3, kernel_ms=sum(by.values()) / 1e3,
                             n_kernels=j - i, **{f"{c}_ms": by[c] / 1e3 for c in CATN}, _other=names, _t0=s, _t1=t))
        i = j
    return rows, kern


CLASSIFIER = 2  # bump when the rules above change: the session's cached per-trace rows are then rebuilt


def _trace_rows(job):
    """One rank trace of one capture -> (rows, unclassified kernels, kernel inventory of the selected steps)."""
    mk, man, rank, f = job
    global MODEL
    MODEL = mk
    meta = load(man)
    label = meta["label"]
    steps, kern = breakdown(f, until_next=(meta["kind"] == "decode"))
    if meta["kind"] == "prefill":
        sel = [s for s in steps if s["prefill_tokens"]]
        groups = {"first_chunk": sel[:1], "last_full_chunk": [s for s in sel if s["prefill_tokens"] == max(x["prefill_tokens"] for x in sel)][-1:]} if sel else {}
    else:
        sel = [s for s in steps if not s["prefill_tokens"] and s["decode_reqs"] == meta["B"]]
        groups = {"decode_steps": sel}
    rows, other = [], collections.Counter()
    prof, plain = meta.get("profiled_window") or {}, meta.get("unprofiled_window") or {}
    for g, ss in groups.items():
        if not ss:
            continue
        for s in ss:
            other.update(s["_other"])
        avg = lambda k: sum(s[k] for s in ss) / len(ss)
        rows.append(dict(study=STUDY, model=mk, runtime=meta["runtime"]["vllm"], capture=label, kind=meta["kind"],
                         bucket=meta["bucket"], engine_B=meta.get("B", 1), rank_file=Path(f).name, rank_index=rank,
                         group=g, steps_averaged=len(ss), steps_in_trace=len(steps), prefill_tokens=avg("prefill_tokens"),
                         decode_reqs=avg("decode_reqs"), span_ms=avg("span_ms"), kernel_sum_ms=avg("kernel_ms"),
                         n_kernels=avg("n_kernels"), profiled_step_ms=prof.get("ms_per_engine_step"),
                         unprofiled_step_ms=plain.get("ms_per_engine_step"),
                         **{f"{c}_ms": avg(f"{c}_ms") for c in CATN},
                         unit="ms per step, GPU kernel sum on this rank (not wall time); decode span = start of step to start of next step",
                         evidence=str(Path(man).relative_to(ROOT))))
    # Kernel inventory of the whole trace on rank 0: names (with their shape arguments), calls and total time.
    inv = None
    if rank == 0:
        c = collections.defaultdict(lambda: [0, 0.0])
        for e in kern:
            x = c[e["name"][:200]]
            x[0] += 1; x[1] += e["dur"]
        inv = dict(capture=label, model=mk, classifier=CLASSIFIER, steps_in_trace=len(steps),
                   note="every GPU kernel of the rank-0 trace: calls and summed duration in microseconds, with the "
                        "component it was assigned to; step_timeline lists, for the last averaged step of each "
                        "group, every kernel call in time order as [index into names, start in microseconds from "
                        "the step start, duration in microseconds, stream id], so layers can be told apart later",
                   kernels=[dict(name=n, component=cat(n), calls=v[0], us=round(v[1], 1))
                            for n, v in sorted(c.items(), key=lambda kv: -kv[1][1])])
        names, timeline = {}, {}
        for g, ss in groups.items():
            if ss:
                t0, t1 = ss[-1]["_t0"], ss[-1]["_t1"]
                timeline[g] = [[names.setdefault(e["name"][:200], len(names)), round(e["ts"] - t0, 1), round(e["dur"], 1), e.get("tid")]
                               for e in kern if t0 <= e["ts"] < t1]
        inv.update(names=list(names), step_timeline=timeline)
    return rows, other, inv


def session_components():
    """Session: classify traces in parallel and cache the rows per capture, so that a finished model
    is read once and a classifier change rebuilds everything from the traces still on the node."""
    import multiprocessing
    cached, jobs = {}, []
    for mk in MODELS:
        for man in sorted((SDIR / mk / "profiles").glob("*.manifest.json")):
            cache = man.with_name(man.name.replace(".manifest.json", ".components.json"))
            done = load(cache)
            if done and done.get("classifier") == CLASSIFIER:
                cached[(mk, man)] = done
                continue
            label = load(man)["label"]
            files = sorted(f for f in (man.parent / label).iterdir() if f.name.endswith((".json.gz", ".json")))
            jobs += [(mk, man, rank, str(f)) for rank, f in enumerate(files)]
    if jobs:
        with multiprocessing.Pool(min(32, len(jobs))) as pool:
            res = pool.map(_trace_rows, jobs, chunksize=1)
        by = collections.defaultdict(lambda: dict(classifier=CLASSIFIER, rows=[], other=collections.Counter()))
        for (mk, man, rank, f), (rows, other, inv) in zip(jobs, res):
            by[(mk, man)]["rows"] += rows
            by[(mk, man)]["other"].update(other)
            if inv:
                man.with_name(man.name.replace(".manifest.json", ".kernels_rank0.json")).write_text(json.dumps(inv))
        for (mk, man), d in by.items():
            d["other"] = dict(d["other"].most_common(40))
            man.with_name(man.name.replace(".manifest.json", ".components.json")).write_text(json.dumps(d))
            cached[(mk, man)] = d
    rows = [r for key in sorted(cached, key=lambda k: (MODELS.index(k[0]), str(k[1]))) for r in cached[key]["rows"]]
    wcsv(RDIR / "components.csv", rows)
    with open(RDIR / "components_unclassified.md", "w") as f:
        f.write("# Largest kernels left in `other_elementwise` (µs summed over the averaged steps, all ranks)\n")
        for (mk, man), d in sorted(cached.items(), key=lambda kv: (MODELS.index(kv[0][0]), str(kv[0][1]))):
            f.write(f"\n## {mk} {load(man)['label']}\n\n" + "".join(f"- `{n}`: {v}\n" for n, v in list(d["other"].items())[:12]))
    print(f"classified {len(jobs)} new rank traces; {len(cached)} captures in components.csv")


def components():
    if SESSION:
        return session_components()
    rows, other = [], collections.defaultdict(collections.Counter)
    for mk in MODELS:
        for man in sorted((SDIR / mk / "profiles").glob("*.manifest.json")):
            meta = load(man)
            label = meta["label"]
            files = sorted((man.parent / label).iterdir())
            files = [f for f in files if f.name.endswith((".json.gz", ".json"))]
            global MODEL
            MODEL = mk
            for rank, f in enumerate(files):
                try:
                    steps, _ = breakdown(f, until_next=(meta["kind"] == "decode"))
                except Exception as e:
                    print("unreadable", f, repr(e)); continue
                if meta["kind"] == "prefill":
                    sel = [s for s in steps if s["prefill_tokens"]]
                    groups = {"first_chunk": sel[:1], "last_full_chunk": [s for s in sel if s["prefill_tokens"] == max(x["prefill_tokens"] for x in sel)][-1:]} if sel else {}
                else:
                    sel = [s for s in steps if not s["prefill_tokens"] and s["decode_reqs"] == meta["B"]]
                    groups = {"decode_steps": sel}
                for g, ss in groups.items():
                    if not ss:
                        continue
                    for s in ss:
                        other[(mk, label)].update(s["_other"])
                    avg = lambda k: sum(s[k] for s in ss) / len(ss)
                    # Wall time per decode step in the profiled window, from the client's token counts.
                    step_ms = (1e3 * meta["window_s"] / max(meta["tokens_in_window"])
                               if meta["kind"] == "decode" and max(meta["tokens_in_window"]) else None)
                    rows.append(dict(study=STUDY, model=mk, runtime=meta["runtime"]["vllm"], capture=label, kind=meta["kind"],
                                     bucket=meta["bucket"], engine_B=meta.get("B", 1), rank_file=f.name, rank_index=rank,
                                     group=g, steps_averaged=len(ss), prefill_tokens=avg("prefill_tokens"),
                                     decode_reqs=avg("decode_reqs"), span_ms=avg("span_ms"), kernel_sum_ms=avg("kernel_ms"),
                                     n_kernels=avg("n_kernels"), profiled_step_ms=step_ms,
                                     span_share_of_step=(avg("span_ms") / step_ms) if step_ms else None,
                                     **{f"{c}_ms": avg(f"{c}_ms") for c in CATN},
                                     unit="ms per step, GPU kernel sum on this rank (not wall time); decode span = start of step to start of next step",
                                     evidence=str(man.relative_to(ROOT))))
    wcsv(RDIR / "components.csv", rows)
    with open(RDIR / "components_unclassified.md", "w") as f:
        f.write("# Largest kernels left in `other_elementwise` (µs summed over the averaged steps, all ranks)\n")
        for (mk, label), c in sorted(other.items()):
            f.write(f"\n## {mk} {label}\n\n" + "".join(f"- `{n}`: {d}\n" for n, d in c.most_common(12)))




def tables():
    """Readable summaries of components.csv and memory.csv (rank means)."""
    rows = list(csv.DictReader(open(RDIR / "components.csv")))
    global MODELS
    MODELS = [m for m in MODELS if any(r["model"] == m for r in rows)]
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None
    label = {"attention_core": "Attention core (softmax attention over stored KV)",
             "recurrent_attention": "Recurrent (linear) attention state update",
             "indexer_topk": "Indexer / top-k / candidates",
             "qkv_rope_kvcache": "Q/K/V norm, RoPE, KV insert, attention metadata",
             "dense_gemm": "Dense GEMM (attention projections; MiMo also layer-0 FFN)",
             "gemm_input_prep": "GEMM input quantization", "mhc_residual_norm": "mHC / residual / norm",
             "moe_expert_gemm": "MoE expert GEMM", "moe_route_combine": "MoE routing, activation, combine",
             "allreduce": "TP all-reduce", "allgather_other_comm": "Other communication", "engram": "Engram",
             "other_elementwise": "Other / unclassified"}
    order = list(label)
    out = ["# Component breakdown (diagnostic traces, idle-profiler launch)\n",
           "Unit: ms of GPU kernel time per engine step, **mean over the 8 ranks**. A kernel sum is not wall time: "
           "kernels on different streams overlap and idle gaps are not kernels. Prefill rows are the last full "
           "8,192-token chunk of one public-text request. Decode rows average the steps of the profiled window at the "
           "stated engine batch B. Classifier: `bench/blog_report.py`; leftovers: "
           "[components_unclassified.md](components_unclassified.md).\n"]

    def block(title, sel, caps):
        out.append(f"\n## {title}\n")
        hdr = [f"{NAME[m]} {c}" for c in caps for m in MODELS]
        out.append("| Component | " + " | ".join(hdr) + " |")
        out.append("| --- | " + " | ".join("---:" for _ in hdr) + " |")
        cell = {}
        for c in caps:
            for m in MODELS:
                rs = [r for r in rows if r["model"] == m and r["capture"] == c and sel(r)]
                cell[(c, m)] = rs
        def mean(rs, k):  # empty cells (for example no client step time in a window) are skipped, not zero
            vals = [num(r, k) for r in rs if num(r, k) is not None]
            return sum(vals) / len(vals) if vals else None
        fmt = lambda v: "n/a" if v is None else (f"{v:.1f}" if v >= 20 else f"{v:.2f}")
        for key, lab in (("span_ms", "**Step span on the rank**"), ("kernel_sum_ms", "**Kernel sum**")):
            out.append(f"| {lab} | " + " | ".join(fmt(mean(cell[(c, m)], key)) for c in caps for m in MODELS) + " |")
        if caps[0].startswith("decode"):
            out.append("| Client step time in the same window | " + " | ".join(
                fmt(mean(cell[(c, m)], "profiled_step_ms")) for c in caps for m in MODELS) + " |")
        for k in order:
            vals = [mean(cell[(c, m)], f"{k}_ms") for c in caps for m in MODELS]
            if any(v and v >= 0.005 for v in vals):
                out.append(f"| {label[k]} | " + " | ".join(fmt(v) if v and v >= 0.005 else "–" for v in vals) + " |")
        att = lambda rs: sum(mean(rs, f"{k}_ms") or 0 for k in ("attention_core", "recurrent_attention", "indexer_topk", "qkv_rope_kvcache"))
        ffn = lambda rs: sum(mean(rs, f"{k}_ms") or 0 for k in ("moe_expert_gemm", "moe_route_combine"))
        out.append("| *Attention path without projections (core + recurrent + indexer + KV work)* | " + " | ".join(
            fmt(att(cell[(c, m)])) for c in caps for m in MODELS) + " |")
        out.append("| *FFN path (expert GEMM + routing/activation/combine)* | " + " | ".join(
            fmt(ffn(cell[(c, m)])) for c in caps for m in MODELS) + " |")

    block("Prefill: one 8,192-token chunk", lambda r: r["group"] == "last_full_chunk", ["prefill16k", "prefill64k"])
    block("Decode at engine batch 1", lambda r: True, ["decode1k_B1", "decode64k_B1"])
    block("Decode at engine batch 8", lambda r: True, ["decode1k_B8", "decode64k_B8"])
    out.append("\nProjections are listed under dense GEMM because the kernels do not say which projection they serve; "
               "on DeepSeek they are the low-rank Q and grouped O projections, on MiMo the fused QKV, O and the layer-0 FFN. "
               "They are therefore **not** included in the attention-path line above, and the complete attention path lies "
               "between that line and that line plus dense GEMM.\n")
    (RDIR / "components_tables.md").write_text("\n".join(out) + "\n")

    mem = list(csv.DictReader(open(RDIR / "memory.csv")))
    out = ["# Live KV/state memory (prefix caching off)\n",
           "Snapshot with B requests decoding and nothing else running. Live bytes = KV usage gauge × per-rank KV pool "
           "from the server log; an approximation when cache groups differ in block bytes. Tokens = client-counted "
           "prompt content + tokens generated at the snapshot (template tokens excluded, under 10 per request).\n",
           "| Model | Input | Live sequences | Live tokens | KV usage | Live MiB per GPU | Live GiB on 8 GPUs | KiB per token per GPU |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mk in MODELS:
        for b in BUCKETS:
            for r in [x for x in mem if x["model"] == mk and x["bucket"] == b]:
                g = float(r["live_gib_per_rank"])
                out.append(f"| {NAME[mk]} | {b} | {r['live_sequences']} | {int(r['live_tokens_client_content_plus_generated']):,} | "
                           f"{100 * float(r['kv_usage_frac']):.3f}% | {1024 * g:.1f} | {8 * g:.2f} | "
                           f"{float(r['live_kib_per_token_per_rank']):.2f} |")
    p = pools()
    out += ["\n## Reserved pools and weights (server log, per GPU)\n",
            "| Model | Weights loaded | KV pool reserved | Pool capacity in tokens |", "| --- | ---: | ---: | ---: |"]
    for mk in MODELS:
        out.append(f"| {NAME[mk]} | {p[mk].get('model_weights_gib', float('nan')):.2f} GiB | "
                   f"{p[mk].get('kv_cache_gib', float('nan')):.2f} GiB | {int(p[mk].get('kv_cache_tokens', 0)):,} |")
    out.append("\nToken capacities are not comparable across models (different KV formats and block accounting); "
               "compare the byte columns. None of this measures reusable-prefix capacity or maximum concurrency.\n")
    (RDIR / "memory_tables.md").write_text("\n".join(out) + "\n")
    print("wrote components_tables.md and memory_tables.md")


def publish(only=None):
    """Curate small evidence into reports/<study>/ (results/ is git-ignored). Per-model runs go
    through bench/curate.py (lossless gzip, private IPs removed from server logs, traces excluded);
    study-level records are copied with a hash ledger. Public input texts are regenerable and are
    represented by their manifest (hashes, sources, licences, token counts) only."""
    import hashlib, shutil, subprocess
    for mk in (only if only is not None else MODELS):
        if (SDIR / mk).exists():
            subprocess.run([sys.executable, str(ROOT / "bench" / "curate.py"), STUDY, mk], check=True)
    dst = RDIR / "study"
    if dst.exists():
        shutil.rmtree(dst)
    ledger = []
    ip = re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)\d{1,3}(?:\.\d{1,3}){3}\b")
    picks = [SDIR / "plan.json", *sorted(SDIR.glob("plan_addendum_*.json")), SDIR / "_inputs" / "manifest.json",
             SDIR / "_inputs" / "extra_tokens.json", SDIR / "_inputs" / "corpus" / "sources.json",
             SDIR / "_logs" / "chain.log", SDIR / "_logs" / "shutdown.jsonl", SDIR / "_logs" / "timeline.jsonl",
             SDIR / "_logs" / "hbm_counter_attempt.txt",
             *sorted((SDIR / "_env").glob("*"))]
    for src in picks:
        if not src.is_file():
            continue
        raw = src.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        if STUDY == BLOG or src.name != "freeze.txt":  # later studies: 4-part versions (13.4.0.1) are not addresses
            text = ip.sub("<PRIVATE_IP>", text)
        out = text.encode()
        ops = ["private IPv4 -> <PRIVATE_IP>"] if out != raw else ["copied"]
        rel = src.relative_to(SDIR)
        d = dst / rel
        if len(out) > 256 * 1024:
            out = gzip.compress(out, mtime=0); d = d.with_name(d.name + ".gz"); ops.append("gzip (lossless)")
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(out)
        ledger.append(dict(source=str(Path("results") / STUDY / rel), published=str(d.relative_to(ROOT)),
                           source_sha256=hashlib.sha256(raw).hexdigest(),
                           published_sha256=hashlib.sha256(out).hexdigest(), operations=ops))
    (dst / "CURATION.json").write_text(json.dumps(dict(
        study=STUDY, policy="numbers unaltered; gzip is lossless; private IPv4 addresses masked",
        excluded={"_inputs/<bucket>/*.jsonl and _inputs/corpus/*.jsonl": "public texts, regenerable with "
                  "bench/blog_corpus.py and bench/blog_inputs.py from pinned sources; hashes are in manifest.json",
                  "profiles/**/*.pt.trace.json.gz": "full per-rank profiler traces, kept locally",
                  "_logs/download*.log, _logs/*.out": "download progress and duplicate chain output"},
        files=ledger), indent=1))
    print("published study-level files:", len(ledger))


def compare():
    """MiMo node control: one second-node block against three first-node blocks."""
    rd = lambda study: [r for r in csv.DictReader(open(ROOT / "reports" / study / "serving.csv")) if r["valid"] == "True"]
    new, old = rd(STUDY), rd(BLOG)
    metrics = (("output_tok_s", "Output tok/s", "{:.1f}"), ("ttft_p50_ms", "TTFT p50 ms", "{:.0f}"),
               ("tpot_p50_ms", "TPOT p50 ms", "{:.2f}"))

    def vals(rows, mk, b, c, k):
        return [float(r[k]) for r in sorted(rows, key=lambda r: r["block"])
                if (r["model"], r["bucket"], r["concurrency"]) == (mk, b, str(c)) and r[k]]

    out = [f"# {STUDY}: comparison with blog-architecture-h100-v1\n",
           "Generated by `BLOG_STUDY=qwen-bf16-h100-v1 python bench/blog_report.py compare` from the two `serving.csv` "
           "files. Same vLLM build, deployment flags, request lists and counts; **different rented node and day**.\n",
           "\n## 1. Node control: MiMo-V2.6-Flash, one block here against three blocks on 2026-10-07\n",
           "Ratio = this node / mean of the 2026-10-07 blocks. \"Inside\" says whether the value lies within the "
           "min–max of those three blocks.\n",
           "| Input | c | Metric | This node | 2026-10-07 mean [min, max] | Ratio | Inside |", "| --- | ---: | --- | ---: | --- | ---: | --- |"]
    ratios = collections.defaultdict(list)
    for b in BUCKETS:
        for c in (1, 8):
            for k, lab, fmt in metrics:
                n, o = vals(new, "mimo-v26", b, c, k), vals(old, "mimo-v26", b, c, k)
                if not n or not o:
                    out.append(f"| {b} | {c} | {lab} | pending | | | |"); continue
                r = st.mean(n) / st.mean(o)
                ratios[k].append(r)
                out.append(f"| {b} | {c} | {lab} | {fmt.format(st.mean(n))} | {fmt.format(st.mean(o))} "
                           f"[{fmt.format(min(o))}, {fmt.format(max(o))}] | {r:.3f} | "
                           f"{'yes' if min(o) <= st.mean(n) <= max(o) else 'no'} |")
    for k, lab, _ in metrics:
        if ratios[k]:
            out.append(f"\n{lab}: ratio range {min(ratios[k]):.3f}–{max(ratios[k]):.3f} over the six points.")
    (RDIR / "comparison.md").write_text("\n".join(out) + "\n")
    print("wrote", (RDIR / "comparison.md").relative_to(ROOT))


if __name__ == "__main__":
    {"compare": compare, "pilot": session_pilot if SESSION else pilot, "serving": serving, "memory": memory,
     "components": components, "tables": tables, "publish": publish, "intervals": session_intervals,
     "natural": session_natural, "routing": session_routing, "support": session_support,
     "windows": session_windows}[sys.argv[1]]()
