#!/usr/bin/env python3
"""Speculation-off serving matrix for the DeepSeek 0731 vs V4.1 study.

Each measured point runs `vllm bench serve` into an immutable directory:
  results/<study>/<model>/<workload>/<config>/<point>/repeat-<n>/
    manifest.json  requests (bench.json, --save-detailed)  summary.json
    telemetry/metrics_{before,after}.prom  telemetry/poll.jsonl  telemetry/nvsmi.csv
Existing repeat directories are never overwritten; a rerun creates repeat-<n>-rerun<k>.
Missing telemetry is recorded as null, never zero.
"""
import argparse, datetime, json, os, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://localhost:8000"
VENV = "/root/vllm/bin"

MODELS = {
    "v41": dict(model="deepseek-ai/DeepSeek-V4.1-Flash",
                revision="dba1be0a40aa45a94ad051997016db3960a90277",
                tokenizer_mode="deepseek_v41"),
    "v4-0731": dict(model="deepseek-ai/DeepSeek-V4-Flash-0731",
                    revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
                    tokenizer_mode="deepseek_v4"),
}
OSL = 256
# Thinking explicitly disabled; greedy target sampling pinned for every arm.
EXTRA_BODY = {"chat_template_kwargs": {"thinking": False}}
TEMPERATURE = 0.0
CONC = [1, 2, 4, 8, 16, 32, 48, 64]
CTX = [16384, 65536, 131072, 260000]
# Isolated c1 sweep: TTFT = prefill without queueing, TPOT = decode at that context.
ISO = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 260000]
ISO_PROMPTS = 4
PREFIX_N = [1, 4, 16]
PREFIX_LEN, SUFFIX_LEN, PREFIX_REQS = 65536, 2048, 64


def conc_prompts(c):
    return max(16, 4 * c)


def seed_for(isl, c, rep, warm=False):
    s = (isl % 100000) + c * 7919 + OSL * 31 + rep * 104729
    return s + 50_000_000 if warm else s


# ---------------------------------------------------------------- telemetry
def http_get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read().decode()


def http_post(path, timeout=60):
    req = urllib.request.Request(BASE + path, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


PROM_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEnaNinf]+)$')


def prom_sum(text, name):
    """Sum a metric across label sets; None if absent."""
    vals = [float(m.group(3)) for line in text.splitlines()
            if (m := PROM_RE.match(line)) and m.group(1) == name]
    return sum(vals) if vals else None


def prom_max(text, name):
    vals = [float(m.group(3)) for line in text.splitlines()
            if (m := PROM_RE.match(line)) and m.group(1) == name]
    return max(vals) if vals else None


COUNTERS = ["vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total", "vllm:request_success_total"]
GAUGES = ["vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:kv_cache_usage_perc"]


class Poller:
    """1 Hz scrape of scheduler/KV gauges plus nvidia-smi memory/util."""

    def __init__(self, tdir):
        self.tdir, self.stop = tdir, threading.Event()
        self.rows = []

    def run(self):
        with open(self.tdir / "poll.jsonl", "w") as f:
            while not self.stop.is_set():
                row = {"t": time.time()}
                try:
                    txt = http_get("/metrics", timeout=5)
                    for g in GAUGES:
                        row[g] = prom_max(txt, g)
                except Exception as e:  # keep polling; record the gap
                    row["error"] = repr(e)
                f.write(json.dumps(row) + "\n"); f.flush()
                self.rows.append(row)
                self.stop.wait(1.0)

    def __enter__(self):
        self.nvsmi = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm",
             "--format=csv", "-lms", "1000"],
            stdout=open(self.tdir / "nvsmi.csv", "w"), stderr=subprocess.DEVNULL)
        self.th = threading.Thread(target=self.run, daemon=True); self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set(); self.th.join(); self.nvsmi.terminate(); self.nvsmi.wait()

    def peak(self, key):
        v = [r[key] for r in self.rows if r.get(key) is not None]
        return max(v) if v else None


def peak_gpu_mem_mib(tdir):
    try:
        lines = (tdir / "nvsmi.csv").read_text().splitlines()[1:]
        v = [float(l.split(",")[2].strip().split()[0]) for l in lines if l.count(",") >= 5]
        return max(v) if v else None
    except Exception:
        return None


# ---------------------------------------------------------------- runs
def unique_dir(d: Path):
    if not d.exists():
        return d
    k = 1
    while (p := d.with_name(f"{d.name}-rerun{k}")).exists():
        k += 1
    return p


def bench_cmd(mk, out_dir, fname, *, c, n, seed, dataset_args, warm=False):
    m = MODELS[mk]
    cmd = [f"{VENV}/vllm", "bench", "serve",
           "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
           "--base-url", BASE, "--model", m["model"],
           "--tokenizer-mode", m["tokenizer_mode"],
           "--request-rate", "inf", "--max-concurrency", str(c),
           "--num-prompts", str(n), "--seed", str(seed),
           "--ignore-eos", "--temperature", str(TEMPERATURE),
           "--extra-body", json.dumps(EXTRA_BODY),
           "--percentile-metrics", "ttft,tpot,itl,e2el",
           "--metric-percentiles", "50,90,95,99",
           "--disable-tqdm", *dataset_args]
    if not warm:
        cmd += ["--save-result", "--save-detailed",
                "--result-dir", str(out_dir), "--result-filename", fname]
    return cmd


def run_point(mk, study, workload, config, point, rep, *, c, n, seed, dataset_args,
              expect_prompt_tokens=None, reset_cache=False, pre_cmds=()):
    rdir = unique_dir(ROOT / "results" / study / mk / workload / config / point / f"repeat-{rep}")
    tdir = rdir / "telemetry"; tdir.mkdir(parents=True)
    cmd = bench_cmd(mk, rdir, "bench.json", c=c, n=n, seed=seed, dataset_args=dataset_args)
    manifest = dict(study=study, model_key=mk, **MODELS[mk], workload=workload, config=config,
                    point=point, repeat=rep, concurrency=c, num_prompts=n, seed=seed,
                    output_len=OSL, temperature=TEMPERATURE, extra_body=EXTRA_BODY,
                    reset_prefix_cache=reset_cache, pre_commands=[list(p) for p in pre_cmds],
                    command=cmd, started=datetime.datetime.now().isoformat())
    if reset_cache:
        manifest["reset_status"] = http_post("/reset_prefix_cache")
    for pc in pre_cmds:  # e.g. prefix prewarm; runs before the before-scrape
        r = subprocess.run(pc, capture_output=True, text=True)
        (rdir / "prewarm.log").open("a").write(r.stdout + r.stderr)
        if r.returncode:
            manifest["prewarm_error"] = r.returncode
    (tdir / "metrics_before.prom").write_text(before := http_get("/metrics"))
    with Poller(tdir) as poll, open(rdir / "bench.log", "w") as log:
        t0 = time.time()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
    (tdir / "metrics_after.prom").write_text(after := http_get("/metrics"))
    manifest.update(finished=datetime.datetime.now().isoformat(), returncode=rc, wall_s=wall)
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    delta = {}
    for k in COUNTERS:
        a, b = prom_sum(after, k), prom_sum(before, k)
        delta[k] = None if a is None or b is None else a - b
    res = {}
    try:
        res = json.loads((rdir / "bench.json").read_text())
    except Exception as e:
        res = {"error": repr(e)}
    keys = [k for k in res if re.match(r"(mean|median|std|p\d+)_(ttft|tpot|itl|e2el)_ms$", k)]
    summary = dict(
        point=point, repeat=rep, concurrency=c, num_prompts=n,
        completed=res.get("completed"), failed=res.get("failed"),
        duration_s=res.get("duration"),
        total_input_tokens_client=res.get("total_input_tokens"),
        total_output_tokens=res.get("total_output_tokens"),
        output_throughput=res.get("output_throughput"),
        request_throughput=res.get("request_throughput"),
        total_token_throughput=res.get("total_token_throughput"),
        max_concurrent_requests=res.get("max_concurrent_requests"),
        server_prompt_tokens=delta["vllm:prompt_tokens_total"],
        server_generation_tokens=delta["vllm:generation_tokens_total"],
        preemptions=delta["vllm:num_preemptions_total"],
        prefix_cache_queries=delta["vllm:prefix_cache_queries_total"],
        prefix_cache_hits=delta["vllm:prefix_cache_hits_total"],
        peak_running=poll.peak("vllm:num_requests_running"),
        peak_waiting=poll.peak("vllm:num_requests_waiting"),
        peak_kv_cache_usage_frac=poll.peak("vllm:kv_cache_usage_perc"),
        peak_gpu_mem_mib=peak_gpu_mem_mib(tdir),
        gpu_seconds_per_output_token=(8 / res["output_throughput"]) if res.get("output_throughput") else None,
        **{k: res[k] for k in keys})
    problems = []
    if rc != 0: problems.append(f"bench rc={rc}")
    if summary["completed"] != n: problems.append(f"completed {summary['completed']}/{n}")
    if summary["failed"]: problems.append(f"failed={summary['failed']}")
    if summary["total_output_tokens"] != n * OSL:
        problems.append(f"output tokens {summary['total_output_tokens']} != {n * OSL}")
    if expect_prompt_tokens and summary["server_prompt_tokens"] is not None:
        if summary["server_prompt_tokens"] < 0.98 * expect_prompt_tokens:
            problems.append(f"server prompt tokens {summary['server_prompt_tokens']} < expected {expect_prompt_tokens}")
    summary["valid"] = not problems
    summary["problems"] = problems
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{datetime.datetime.now():%H:%M:%S}] {workload}/{point} r{rep}: "
          f"valid={summary['valid']} out_tps={summary['output_throughput']} "
          f"ttft_p50={summary.get('median_ttft_ms')} tpot_p50={summary.get('median_tpot_ms')} "
          f"{problems}", flush=True)
    return summary


def warmup(mk, *, c, n, seed, dataset_args):
    cmd = bench_cmd(mk, None, None, c=c, n=n, seed=seed, dataset_args=dataset_args, warm=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  warmup c={c} n={n} rc={r.returncode}", flush=True)


def random_args(isl):
    return ["--dataset-name", "random", "--random-input-len", str(isl),
            "--random-output-len", str(OSL), "--random-range-ratio", "0"]


# ---------------------------------------------------------------- prefix prompts
def build_prefix_files(mk, n_prefixes, out: Path, seed):
    """Measured file: n_prefixes shared 64K prefixes x (64/n) 2K suffixes, fixed order.
    Prewarm file: each prefix + one disjoint suffix. Same files for every cache state."""
    import numpy as np
    from vllm.tokenizers import get_tokenizer
    from vllm.benchmarks.datasets import gen_prompt_decode_to_target_len
    tok = get_tokenizer(MODELS[mk]["model"], tokenizer_mode=MODELS[mk]["tokenizer_mode"],
                        revision=MODELS[mk]["revision"])
    rng = np.random.default_rng(seed)
    vocab = tok.vocab_size

    def text(n):
        ids = rng.integers(0, vocab, size=n).tolist()
        t, _, _ = gen_prompt_decode_to_target_len(tokenizer=tok, token_sequence=ids,
                                                  target_token_len=n, add_special_tokens=False)
        return t

    prefixes = [text(PREFIX_LEN) for _ in range(n_prefixes)]
    per = PREFIX_REQS // n_prefixes
    measured = [(i, prefixes[i] + text(SUFFIX_LEN)) for i in range(n_prefixes) for _ in range(per)]
    rng.shuffle(measured)
    prewarm = [prefixes[i] + text(SUFFIX_LEN) for i in range(n_prefixes)]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "measured.jsonl", "w") as f:
        for _, p in measured:
            f.write(json.dumps({"prompt": p, "output_tokens": OSL}) + "\n")
    with open(out / "prewarm.jsonl", "w") as f:
        for p in prewarm:
            f.write(json.dumps({"prompt": p, "output_tokens": OSL}) + "\n")
    (out / "order.json").write_text(json.dumps([i for i, _ in measured]))


def custom_args(path):
    return ["--dataset-name", "custom", "--dataset-path", str(path), "--skip-chat-template",
            "--disable-shuffle"]


# ---------------------------------------------------------------- matrix
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--study", default="v41-vs-0731")
    ap.add_argument("--config", default="off")
    ap.add_argument("--workloads", default="concurrency,context")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--conc-list", default=",".join(map(str, CONC)),
                    help="subset of concurrency points (e.g. control reruns)")
    ap.add_argument("--prefix-states", default="cold,prewarmed",
                    help="cache-off runs on the off server; cold/prewarmed on off-prefix")
    a = ap.parse_args()
    mk, study, cfg = a.model, a.study, a.config
    served = json.loads(http_get("/v1/models"))["data"][0]["id"]
    assert served == MODELS[mk]["model"], f"server serves {served}, expected {MODELS[mk]['model']}"
    wl = a.workloads.split(",")

    if "concurrency" in wl:
        print("== concurrency warmup (disjoint seeds)", flush=True)
        for c in (1, 16, 64):
            warmup(mk, c=c, n=c, seed=seed_for(16384, c, 0, warm=True), dataset_args=random_args(16384))
        for rep in range(1, a.repeats + 1):
            conc = [int(x) for x in a.conc_list.split(",")]
            order = conc if rep % 2 else conc[::-1]  # alternate order across repeats
            for c in order:
                n = conc_prompts(c)
                run_point(mk, study, "concurrency", cfg, f"isl16384_osl256_c{c}", rep,
                          c=c, n=n, seed=seed_for(16384, c, rep), dataset_args=random_args(16384),
                          expect_prompt_tokens=n * 16384)
    if "isolated" in wl:
        print("== isolated warmup", flush=True)
        for isl in ISO:
            warmup(mk, c=1, n=1, seed=seed_for(isl, 1, 0, warm=True), dataset_args=random_args(isl))
        for rep in range(1, a.repeats + 1):
            order = ISO if rep % 2 else ISO[::-1]
            for isl in order:
                run_point(mk, study, "isolated", cfg, f"isl{isl}_osl256_c1", rep,
                          c=1, n=ISO_PROMPTS, seed=seed_for(isl, 1, rep) + 7,
                          dataset_args=random_args(isl), expect_prompt_tokens=ISO_PROMPTS * isl)
    if "context" in wl:
        print("== context warmup", flush=True)
        for isl in CTX:
            warmup(mk, c=1, n=1, seed=seed_for(isl, 8, 0, warm=True), dataset_args=random_args(isl))
        for rep in range(1, a.repeats + 1):
            order = CTX if rep % 2 else CTX[::-1]
            for isl in order:
                run_point(mk, study, "context", cfg, f"isl{isl}_osl256_c8", rep,
                          c=8, n=16, seed=seed_for(isl, 8, rep), dataset_args=random_args(isl),
                          expect_prompt_tokens=16 * isl)
    if "prefix" in wl:
        states = a.prefix_states.split(",")
        for rep in range(1, a.repeats + 1):
            for npfx in (PREFIX_N if rep % 2 else PREFIX_N[::-1]):
                pdir = ROOT / "results" / study / "_prompts" / f"prefix_n{npfx}_rep{rep}"
                if not (pdir / "measured.jsonl").exists():
                    build_prefix_files(mk, npfx, pdir, seed=4200 + npfx + 1000 * rep)
                ds = custom_args(pdir / "measured.jsonl")
                expect = PREFIX_REQS * (PREFIX_LEN + SUFFIX_LEN)
                for st in states:
                    pre = ()
                    if st == "prewarmed":
                        pre = (bench_cmd(mk, None, None, c=npfx, n=npfx, seed=0, warm=True,
                                         dataset_args=custom_args(pdir / "prewarm.jsonl")),)
                    run_point(mk, study, "prefix", f"{cfg}-{st}", f"p65536_s2048_n{npfx}_c8", rep,
                              c=8, n=PREFIX_REQS, seed=0, dataset_args=ds, expect_prompt_tokens=expect,
                              reset_cache=(st != "cache-off"), pre_cmds=pre)


if __name__ == "__main__":
    main()
