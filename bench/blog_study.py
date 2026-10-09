#!/usr/bin/env python3
"""Chain for the architecture-blog protocol (target.md, sections 1-11): AR only, prefix caching off.

The study is chosen with BLOG_STUDY (default: the completed blog-architecture-h100-v1).
Study qwen-bf16-h100-v1 (2026-10-08) measures the original BF16 Qwen checkpoint with the same
protocol and inputs, plus one MiMo timing block as a node control; see STUDIES below.

  python bench/blog_study.py env                      # record node/runtime
  python bench/blog_study.py plan --counts '{...}'    # freeze plan.json after the pilots
  python bench/blog_study.py dry-run                  # validate plan, inputs, pins, paths
  python bench/blog_study.py chain diag:v4-0731 timing:v4-0731:1 ...

`diag:<model>` launches the idle-profiler server and runs functional checks, a disjoint
pilot, live-KV snapshots and trace captures. `timing:<model>:<block>` launches the plain
server and runs the six unprofiled points of one repeat block. Each step owns its server:
it is started in its own process group, readiness is bounded, and the group is stopped
and GPU memory checked on every exit path. Historical chains are never invoked.
"""
import argparse, datetime, hashlib, json, os, shutil, signal, subprocess, sys, threading, time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as rm

ROOT = rm.ROOT
BLOG = "blog-architecture-h100-v1"
BLOG_RUNTIME = "0.31.1rc1.dev50+g554340f3d"
# Per study: models that may launch, models with a diagnostic launch, timing blocks, where the
# revisions are pinned, and the frozen inputs it must reproduce byte for byte.
STUDIES = {
    # First three models, then qwen-38 (FP8) and glm-53 by plan_addendum_glm_qwen.json (Tan, 2026-10-07).
    BLOG: dict(allowed=("v4-0731", "v41", "mimo-v26", "qwen-38", "glm-53"),
               diag=("v4-0731", "v41", "mimo-v26", "qwen-38", "glm-53"),
               blocks={"1": ["v4-0731", "mimo-v26", "v41"], "2": ["mimo-v26", "v41", "v4-0731"],
                       "3": ["v41", "v4-0731", "mimo-v26"]},
               pins=("target.md",), inputs_reference=None),
    # Original BF16 Qwen (docs/qwen-checkpoint-policy.md). MiMo runs one timing block as a node
    # control against its three blocks of 2026-10-07, and (plan_addendum_mimo_trace_control.json)
    # one diagnostic launch so trace components can be compared across the two nodes as well.
    "qwen-bf16-h100-v1": dict(allowed=("qwen-38-bf16", "mimo-v26"), diag=("qwen-38-bf16", "mimo-v26"),
                              blocks={"1": ["qwen-38-bf16", "mimo-v26"], "2": ["qwen-38-bf16"],
                                      "3": ["qwen-38-bf16"]},
                              pins=("target.md", "docs/qwen-checkpoint-policy.md"),
                              inputs_reference=f"reports/{BLOG}/study/_inputs/manifest.json"),
}
RETIRED = ("qwen-38",)  # FP8: historical records only, never launched again
STUDY = os.environ.get("BLOG_STUDY", BLOG)
SPEC = STUDIES[STUDY]
SDIR = ROOT / "results" / STUDY
INP = SDIR / "_inputs"
LOGS = SDIR / "_logs"
BASE = rm.BASE
ALLOWED = SPEC["allowed"]
CONFIGS = {"off": "timing", "off-profidle": "diagnostic"}  # both pass --no-enable-prefix-caching
SERVE_ARGS = ["--max-num-batched-tokens", "8192"]  # explicit chunk budget
BUCKETS = ("16k", "1k", "64k")  # 16K first: the priority comparison
LOADS = (1, 8)
READY_TIMEOUT_S = 1800
PILOT_N = {("1k", 1): 6, ("1k", 8): 24, ("16k", 1): 3, ("16k", 8): 12, ("64k", 1): 3, ("64k", 8): 9}
WARM_N = {"1k": 16, "16k": 8, "64k": 3}


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] CHAIN {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "chain.log", "a") as f:
        f.write(line + "\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def runtime():
    venv = os.environ.get("VLLM_VENV", "/root/vllm")
    out = sh(f"{venv}/bin/python -c \"import vllm,torch;print(vllm.__version__);print(torch.__version__)\"").split()
    return dict(venv=venv, vllm=out[-2] if len(out) >= 2 else None, torch=out[-1] if out else None)


def gpu_mem():
    return [int(x) for x in sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()]


def post(path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


def manifest():
    return json.loads((INP / "manifest.json").read_text())


def ctok(req, mk):
    """Client-side content tokens of one manifest request under a model's tokenizer. The frozen
    manifest holds the first three models; later models are in _inputs/extra_tokens.json."""
    if mk in req["content_tokens"]:
        return req["content_tokens"][mk]
    return json.loads((INP / "extra_tokens.json").read_text())[mk][req["request_id"]]


def load_plan():
    plan = json.loads((SDIR / "plan.json").read_text())
    for f in sorted(SDIR.glob("plan_addendum_*.json")):
        add = json.loads(f.read_text())
        for blk, models in add["blocks"].items():
            plan["blocks"][blk] = plan["blocks"][blk] + models
    return plan


def split_rows(bucket, split):
    return [json.loads(l) for l in open(INP / bucket / f"{split}.jsonl")]


def chat_body(mk, prompt, **kw):
    return dict(model=rm.MODELS[mk]["model"], messages=[{"role": "user", "content": prompt}],
                temperature=0, **rm.extra_body(mk), **kw)


# ---------------------------------------------------------------- owned server
class Server:
    def __init__(self, mk, cfg):
        assert mk in ALLOWED, f"model {mk} is outside the study"
        assert mk not in RETIRED, f"{mk} is a retired key (docs/qwen-checkpoint-policy.md)"
        assert cfg in CONFIGS, f"config {cfg} is not a no-prefix AR config"
        self.mk, self.cfg, self.proc = mk, cfg, None
        self.pdir = SDIR / mk / "profiles"
        self.log = SDIR / mk / "_server" / f"serve-{cfg}-{datetime.datetime.now():%Y%m%d-%H%M%S}.log"

    def __enter__(self):
        if any(m > 2000 for m in gpu_mem()) or sh("pgrep -f '[v]llm serve'").strip():
            raise RuntimeError("GPUs are not free; refusing to launch over another server")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.pdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HOME=os.environ.get("HF_HOME", "/workspace/hf"), PROFILE_DIR=str(self.pdir))
        self.t0 = time.time()
        self.proc = subprocess.Popen(["bash", "bench/serve.sh", self.mk, self.cfg, str(self.log), *SERVE_ARGS],
                                     cwd=ROOT, env=env, start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log(f"launched {self.mk} {self.cfg} pgid={self.proc.pid} log={self.log.relative_to(ROOT)}")
        try:
            self.wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def wait_ready(self):
        want = rm.MODELS[self.mk]["model"]
        while time.time() - self.t0 < READY_TIMEOUT_S:
            time.sleep(5)
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode} before readiness")
            try:
                if json.loads(rm.http_get("/v1/models", timeout=5))["data"][0]["id"] == want:
                    self.ready_s = time.time() - self.t0
                    text = self.log.read_text(errors="replace")
                    if "'enable_prefix_caching': False" not in text:
                        raise RuntimeError("server log does not show prefix caching disabled")
                    log(f"ready {self.mk} {self.cfg} after {self.ready_s:.0f}s")
                    return
            except (OSError, ValueError, KeyError, IndexError):
                pass
        raise RuntimeError("readiness timeout")

    def stop(self):
        if self.proc is None:
            return
        # Grace periods are short: results are already on disk when stop() runs and the node is rented.
        def alive():  # poll() reaps the group leader; an unreaped leader would look alive to pgrep
            self.proc.poll()
            return sh(f"pgrep -g {self.proc.pid}").strip()

        for sig, wait in ((signal.SIGINT, 25), (signal.SIGTERM, 10), (signal.SIGKILL, 20)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            t = time.time()
            while time.time() - t < wait and alive():
                time.sleep(1)
            if not alive():
                break
        t = time.time()
        while time.time() - t < 120 and max(gpu_mem()) > 500:
            time.sleep(3)
        rec = dict(model=self.mk, config=self.cfg, stopped=datetime.datetime.now().isoformat(),
                   up_s=round(time.time() - self.t0), gpu_mem_mib=gpu_mem(),
                   leftover=alive().split())
        with open(LOGS / "shutdown.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"stopped {self.mk} {self.cfg} up={rec['up_s']}s gpu_mem_max={max(rec['gpu_mem_mib'])}MiB "
            f"leftover={len(rec['leftover'])}")
        self.proc = None

    def __exit__(self, *a):
        self.stop()


# ---------------------------------------------------------------- timing points
def run_bench(mk, workload, cfg, bucket, c, n, rep, split, timeout):
    """One `vllm bench serve` run through run_matrix.run_point, then per-request validation."""
    man = manifest()["buckets"][bucket]["splits"][split]
    reqs = man["requests"][:n]
    assert len(reqs) == n, f"{bucket}/{split} has fewer than {n} requests"
    point = f"ctx{bucket}_osl256_c{c}"
    pdir = SDIR / mk / workload / cfg / point
    summary = rm.run_point(mk, STUDY, workload, cfg, point, rep, c=c, n=n, seed=0,
                           dataset_args=rm.custom_args(INP / bucket / f"{split}.jsonl"),
                           expect_prompt_tokens=sum(ctok(r, mk) for r in reqs), timeout=timeout)
    rdir = max(pdir.glob(f"repeat-{rep}*"), key=lambda p: p.stat().st_mtime)
    problems = list(summary["problems"])
    out_lens = in_lens = None
    try:
        res = json.loads((rdir / "bench.json").read_text())
        out_lens, in_lens = res.get("output_lens"), res.get("input_lens")
        if out_lens is None or len(out_lens) != n or any(x != rm.OSL for x in out_lens):
            problems.append("per-request output length is not 256 for every request")
    except Exception as e:
        problems.append(f"bench.json unreadable: {e!r}")
    poll = [json.loads(l) for l in open(rdir / "telemetry" / "poll.jsonl")]
    running = [r.get("vllm:num_requests_running") for r in poll if r.get("vllm:num_requests_running") is not None]
    side = dict(study=STUDY, model_key=mk, runtime=runtime(), bucket=bucket, split=split, concurrency=c, n=n,
                block=rep, prefix_caching=False, speculation=False, list_file_sha256=man["file_sha256"],
                request_ids=[r["request_id"] for r in reqs], request_sha256=[r["sha256"] for r in reqs],
                domains=[r["domain"] for r in reqs], client_content_tokens=[ctok(r, mk) for r in reqs],
                output_lens=out_lens, input_lens_client=in_lens,
                running_requests_1hz=running, valid=not problems, problems=problems)
    (rdir / "blog_validation.json").write_text(json.dumps(side))
    log(f"{workload} {mk} {point} block={rep} n={n} valid={side['valid']} dur={summary.get('duration_s')} "
        f"out_tps={summary.get('output_throughput')} ttft_p50={summary.get('median_ttft_ms')} "
        f"tpot_p50={summary.get('median_tpot_ms')} {problems or ''}")
    return summary


def timing_step(mk, block):
    plan = load_plan()
    order = [(b, c) for b in BUCKETS for c in LOADS]
    if block % 2 == 0:
        order.reverse()
    with Server(mk, "off"):
        for b in BUCKETS:  # warm every shape with disjoint requests before the timed points
            rm.warmup(mk, c=min(8, WARM_N[b]), n=WARM_N[b], seed=0,
                      dataset_args=rm.custom_args(INP / b / "aux.jsonl"))
        for b, c in order:
            run_bench(mk, "serving", "off", b, c, plan["counts"][f"{b}:c{c}"], block, "timing",
                      plan["run_timeout_s"])


# ---------------------------------------------------------------- diagnostics
def functional(mk):
    out = SDIR / mk / "_functional"; out.mkdir(parents=True, exist_ok=True)
    rec = dict(model_key=mk, runtime=runtime(), checks=[])
    row = split_rows("1k", "aux")[0]
    meta = manifest()["buckets"]["1k"]["splits"]["aux"]["requests"][0]
    try:  # rendered request: server-side token count of the templated chat prompt
        tk = post("/tokenize", dict(model=rm.MODELS[mk]["model"], messages=[{"role": "user", "content": row["prompt"]}],
                                    add_generation_prompt=True, **rm.extra_body(mk)))
        rec["rendered"] = dict(server_prompt_tokens=tk.get("count"), client_content_tokens=ctok(meta, mk),
                               template_overhead=tk.get("count") - ctok(meta, mk))
    except Exception as e:
        rec["rendered"] = dict(error=repr(e))
    natural = {"code": "Write a Python function that returns the n-th Fibonacci number iteratively. Reply with the code only.",
               "math": "What is 17 * 23? Give the final number on the last line.",
               "chat": "In two sentences, explain why the sky is blue."}
    for d, q in natural.items():  # natural EOS, thinking off
        r = post("/v1/chat/completions", chat_body(mk, q, max_tokens=512))
        m = r["choices"][0]
        rec["checks"].append(dict(kind="natural_eos", domain=d, prompt=q, text=m["message"].get("content"),
                                  reasoning=m["message"].get("reasoning_content") or m["message"].get("reasoning"),
                                  finish_reason=m["finish_reason"], usage=r["usage"]))
    r = post("/v1/chat/completions", chat_body(mk, row["prompt"], max_tokens=rm.OSL, ignore_eos=True))
    rec["checks"].append(dict(kind="forced_256", request_id=meta["request_id"], usage=r["usage"],
                              finish_reason=r["choices"][0]["finish_reason"]))
    nat = [c for c in rec["checks"] if c["kind"] == "natural_eos"]
    rec["ok"] = (all(c["finish_reason"] == "stop" and not c["reasoning"] and c["text"] for c in nat)
                 and rec["checks"][-1]["usage"]["completion_tokens"] == rm.OSL)
    (out / "functional.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log(f"functional {mk} ok={rec['ok']} rendered={rec['rendered']}")
    return rec["ok"]


class Stream(threading.Thread):
    """One streaming chat request; counts content chunks until closed."""

    def __init__(self, mk, prompt):
        super().__init__(daemon=True)
        self.body = chat_body(mk, prompt, max_tokens=4000, ignore_eos=True, stream=True)
        self.n, self.t_first, self.err, self.resp = 0, None, None, None

    def run(self):
        import requests
        try:
            self.resp = requests.post(BASE + "/v1/chat/completions", json=self.body, stream=True, timeout=900)
            for line in self.resp.iter_lines():
                if line.startswith(b"data: {") and b'"content"' in line:
                    self.n += 1
                    self.t_first = self.t_first or time.time()
        except Exception as e:  # closing the response to abort lands here
            self.err = repr(e)

    def close(self):
        if self.resp is not None:
            self.resp.close()


def kv_gauges():
    txt = rm.http_get("/metrics")
    return {g: rm.prom_max(txt, g) for g in rm.GAUGES}


def collect_trace(srv, label, before, meta):
    """Wait for the rank traces written since `before` and file them under profiles/<label>/."""
    new, last = [], -1
    for _ in range(180):
        new = sorted(p for p in srv.pdir.rglob("*") if p.is_file() and p not in before and p.parent == srv.pdir)
        size = sum(p.stat().st_size for p in new)
        if len(new) >= 8 and size == last:
            break
        last = size
        time.sleep(5)
    dst = srv.pdir / label; dst.mkdir(exist_ok=True)
    for p in new:
        shutil.move(str(p), dst / p.name)
    meta.update(label=label, files=sorted(f.name for f in dst.iterdir()), n_rank_files=len(new),
                note="diagnostic profiled capture; not a timing result")
    (srv.pdir / f"{label}.manifest.json").write_text(json.dumps(meta, indent=1))
    log(f"trace {srv.mk} {label} files={len(new)} "
        f"{ {k: meta[k] for k in ('B', 'window_s', 'tokens_in_window', 'request_wall_s') if k in meta} }")


def prefill_capture(srv, bucket):
    mk = srv.mk
    row = split_rows(bucket, "aux")[0]
    meta = manifest()["buckets"][bucket]["splits"]["aux"]["requests"][0]
    before = {p for p in srv.pdir.rglob("*") if p.is_file()}
    post("/start_profile")
    t0 = time.time()
    r = post("/v1/chat/completions", chat_body(mk, row["prompt"], max_tokens=8, ignore_eos=True))
    wall = time.time() - t0
    post("/stop_profile")
    collect_trace(srv, f"prefill{bucket}", before,
                  dict(kind="prefill", model_key=mk, runtime=runtime(), bucket=bucket, request_id=meta["request_id"],
                       request_sha256=meta["sha256"], usage=r["usage"], max_tokens=8, request_wall_s=wall))


def decode_capture(srv, bucket, B, profile):
    """B live sequences, all past prefill, no new admissions: KV snapshot and (optionally) a trace window."""
    mk = srv.mk
    rows = split_rows(bucket, "aux")[:B]
    metas = manifest()["buckets"][bucket]["splits"]["aux"]["requests"][:B]
    idle = kv_gauges()
    streams = [Stream(mk, r["prompt"]) for r in rows]
    for s in streams:
        s.start()
    t0 = time.time()
    try:
        # All B requests must be past prefill. A stream can stop yielding text once the model
        # passes its natural end (ignore_eos keeps generating special tokens), so progress is
        # judged on the busiest stream, and the engine batch is read from the trace itself.
        while min(s.n for s in streams) < 1 or max(s.n for s in streams) < 8:
            if time.time() - t0 > 600 or any(s.err for s in streams):
                raise RuntimeError(f"decode capture did not reach steady decode: {[s.err for s in streams]}")
            time.sleep(0.05)
        snap = dict(kind="live_kv", model_key=mk, runtime=runtime(), bucket=bucket, B=B,
                    request_ids=[m["request_id"] for m in metas],
                    client_content_tokens=[ctok(m, mk) for m in metas],
                    generated_at_snapshot=[s.n for s in streams], idle_gauges=idle, live_gauges=kv_gauges(),
                    gpu_mem_mib=gpu_mem(), time_to_all_decoding_s=time.time() - t0)
        kdir = SDIR / mk / "_kv"; kdir.mkdir(parents=True, exist_ok=True)
        (kdir / f"live_{bucket}_B{B}.json").write_text(json.dumps(snap, indent=1))
        log(f"kv {mk} {bucket} B={B} usage={snap['live_gauges'].get('vllm:kv_cache_usage_perc')} "
            f"running={snap['live_gauges'].get('vllm:num_requests_running')}")
        # Plain (unprofiled) decode window first: step time at this exact B and context.
        n0, t1 = [s.n for s in streams], time.time()
        time.sleep(1.0)
        n1, t2 = [s.n for s in streams], time.time()
        plain = dict(window_s=t2 - t1, tokens=[b - a for a, b in zip(n0, n1)])
        if profile:
            before = {p for p in srv.pdir.rglob("*") if p.is_file()}
            post("/start_profile")
            a, ta = [s.n for s in streams], time.time()
            while max(s.n - x for s, x in zip(streams, a)) < 48 and time.time() - ta < 5:
                time.sleep(0.02)
            b, tb = [s.n for s in streams], time.time()
            post("/stop_profile")
            meta = dict(kind="decode", model_key=mk, runtime=runtime(), bucket=bucket, B=B,
                        request_ids=snap["request_ids"], client_content_tokens=snap["client_content_tokens"],
                        generated_before_window=a, tokens_in_window=[y - x for x, y in zip(a, b)],
                        window_s=tb - ta, unprofiled_window=plain, live_gauges=snap["live_gauges"])
    finally:
        for s in streams:
            s.close()
    t = time.time()  # wait for the aborted requests to leave the engine
    while time.time() - t < 60 and (kv_gauges().get("vllm:num_requests_running") or 0) > 0:
        time.sleep(0.5)
    if profile:
        collect_trace(srv, f"decode{bucket}_B{B}", before, meta)


def diag_step(mk, parts):
    with Server(mk, "off-profidle") as srv:
        if "functional" in parts and not functional(mk):
            log(f"functional check FAILED for {mk}; continuing with diagnostics only")
        if "pilot" in parts:
            for b in BUCKETS:
                for c in LOADS:
                    run_bench(mk, "pilot", "off-profidle", b, c, PILOT_N[(b, c)], 1, "aux", 900)
        if "kv" in parts:
            for b in ("1k", "16k", "64k"):
                for B in (1, 8):
                    try:
                        decode_capture(srv, b, B, profile=("trace" in parts and b != "16k"))
                    except Exception as e:
                        log(f"FAILED decode capture {mk} {b} B={B}: {e!r}")
        if "trace" in parts:
            for b in ("16k", "64k"):
                try:
                    prefill_capture(srv, b)
                except Exception as e:
                    log(f"FAILED prefill capture {mk} {b}: {e!r}")


# ---------------------------------------------------------------- plan / env / dry-run
def write_env():
    out = SDIR / "_env"; out.mkdir(parents=True, exist_ok=True)
    venv = runtime()["venv"]
    for name, cmd in dict(nvidia_smi="nvidia-smi", gpu_query="nvidia-smi --query-gpu=index,name,memory.total,"
                          "clocks.sm,clocks.mem,power.limit,pci.bus_id --format=csv", topology="nvidia-smi topo -m",
                          nvlink="nvidia-smi nvlink -s", lscpu="lscpu", numa="numactl -H", mem="free -g",
                          cpufreq="cd /sys/devices/system/cpu/cpu0/cpufreq 2>/dev/null && grep . scaling_governor "
                                  "scaling_driver scaling_min_freq scaling_max_freq || echo 'no cpufreq interface'",
                          freeze=f"{venv}/bin/python -m pip freeze 2>/dev/null || {venv}/bin/uv pip freeze --python {venv}/bin/python").items():
        (out / f"{name}.txt").write_text(sh(cmd))
    (out / "runtime.json").write_text(json.dumps(dict(runtime(), recorded=datetime.datetime.now().isoformat(),
                                                      serve_args=SERVE_ARGS), indent=1))
    log(f"env recorded: {runtime()}")


def write_plan(counts, note):
    plan = dict(
        study=STUDY, frozen=datetime.datetime.now().isoformat(), runtime=runtime(),
        models={mk: {k: rm.MODELS[mk][k] for k in ("model", "revision")} for mk in ALLOWED},
        inputs_reference=SPEC["inputs_reference"],
        deployment="TP8 + EP, utilization 0.90, max_model_len 262144, max_num_seqs 64, "
                   "max_num_batched_tokens 8192 (explicit), native precision, prefix caching off, speculation off",
        requests="public code/math/chat text, one user message, server-side chat template, temperature 0, "
                 "thinking off, ignore_eos with 256 output tokens",
        inputs_manifest_sha256=hashlib.sha256((INP / "manifest.json").read_bytes()).hexdigest(),
        counts=counts, count_rule="start at max(16, 4c), multiple of 3 for domain balance, raised until the "
                                  "fastest piloted model has a window of at least 60 s; same first-n requests "
                                  "of the timing split for every model",
        blocks=SPEC["blocks"], diagnostic_models=list(SPEC["diag"]),
        point_order="16k c1, 16k c8, 1k c1, 1k c8, 64k c1, 64k c8 in odd blocks; reversed in even blocks",
        warmup="aux split (disjoint from timing) at every bucket after each launch",
        run_timeout_s=1500, ready_timeout_s=READY_TIMEOUT_S,
        validity="all requests complete, every request returns exactly 256 output tokens, server prompt "
                 "tokens at least 98% of the client content tokens, no failed requests",
        rerun_rule="an invalid run is kept; one rerun in a new repeat-N-rerunK directory is paired by "
                   "manifest identity; the fastest attempt is never selected",
        diagnostics="per model: functional checks, disjoint pilot, live-KV snapshots at 1k/16k/64k x B=1/8, "
                    "traces prefill16k, prefill64k, decode1k/64k x B=1/8 on the idle-profiler launch",
        timing_runs=6 * sum(len(v) for v in SPEC["blocks"].values()), trace_captures=6 * len(SPEC["diag"]),
        live_kv_snapshots=6 * len(SPEC["diag"]), note=note)
    (SDIR / "plan.json").write_text(json.dumps(plan, indent=1))
    log(f"plan frozen: {counts}")


def dry_run(steps):
    plan = load_plan()
    man = manifest()
    problems = []
    if hashlib.sha256((INP / "manifest.json").read_bytes()).hexdigest() != plan["inputs_manifest_sha256"]:
        problems.append("inputs manifest changed after the plan was frozen")
    if STUDY != BLOG and runtime()["vllm"] != BLOG_RUNTIME:
        problems.append(f"runtime {runtime()['vllm']} is not the blog build {BLOG_RUNTIME}")
    if SPEC["inputs_reference"]:  # same request lists as the completed blog study, byte for byte
        ref = json.loads((ROOT / SPEC["inputs_reference"]).read_text())
        for b, e in ref["buckets"].items():
            for split, s in e["splits"].items():
                mine = man["buckets"].get(b, {}).get("splits", {}).get(split, {})
                if mine.get("file_sha256") != s["file_sha256"] or \
                        [r["sha256"] for r in mine.get("requests", [])] != [r["sha256"] for r in s["requests"]]:
                    problems.append(f"{b}/{split} differs from the reference inputs")
    serve = (ROOT / "bench" / "serve.sh").read_text()
    for mk in ALLOWED:
        pinned_doc = "".join((ROOT / f).read_text() for f in SPEC["pins"]) + \
            "".join(f.read_text() for f in SDIR.glob("plan_addendum_*.json"))
        if rm.MODELS[mk]["revision"] not in serve or rm.MODELS[mk]["revision"] not in pinned_doc:
            problems.append(f"{mk} revision is not pinned consistently")
    for b in man["buckets"]:
        for split, s in man["buckets"][b]["splits"].items():
            if hashlib.sha256((INP / s["file"]).read_bytes()).hexdigest() != s["file_sha256"]:
                problems.append(f"{s['file']} hash mismatch")
        for c in LOADS:
            n = plan["counts"][f"{b}:c{c}"]
            if n > man["buckets"][b]["splits"]["timing"]["n"] or n % 3:
                problems.append(f"{b}:c{c} count {n} unavailable or not domain balanced")
    seen = set()
    for st in steps:
        kind, mk, *rest = st.split(":")
        if mk not in ALLOWED or mk in RETIRED or kind not in ("diag", "timing") or \
                (kind == "diag" and mk not in SPEC["diag"]):
            problems.append(f"step {st} not allowed")
            continue
        if kind == "timing":
            blk = rest[0]
            if (mk, blk) in seen or mk not in plan["blocks"].get(blk, []):
                problems.append(f"step {st} duplicates or is outside the declared blocks")
            seen.add((mk, blk))
            for b in BUCKETS:
                for c in LOADS:
                    d = SDIR / mk / "serving" / "off" / f"ctx{b}_osl256_c{c}" / f"repeat-{blk}"
                    print(f"  would write {d.relative_to(ROOT)}{' (exists: rerun dir)' if d.exists() else ''}")
    print("configs:", CONFIGS, "serve args:", SERVE_ARGS, "runtime:", runtime())
    print("DRY RUN", "FAILED: " + "; ".join(problems) if problems else "OK", f"({len(steps)} steps)")
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["env", "plan", "dry-run", "chain"])
    ap.add_argument("steps", nargs="*")
    ap.add_argument("--counts"); ap.add_argument("--note", default="")
    ap.add_argument("--diag-parts", default="functional,pilot,kv,trace")
    a = ap.parse_args()
    if a.cmd == "env":
        return write_env()
    if a.cmd == "plan":
        return write_plan(json.loads(a.counts), a.note)
    if a.cmd == "dry-run":
        sys.exit(0 if dry_run(a.steps) else 1)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the finally/exit paths on termination
    for st in a.steps:
        kind, mk, *rest = st.split(":")
        for attempt in (1, 2):
            log(f"step {st} start" + (" (retry after a failed launch)" if attempt == 2 else ""))
            try:
                if kind == "diag":
                    diag_step(mk, a.diag_parts.split(","))
                elif kind == "timing":
                    timing_step(mk, int(rest[0]))
                else:
                    raise SystemExit(f"unknown step {st}")
                log(f"step {st} done")
                break
            except SystemExit:
                raise
            except Exception as e:
                log(f"step {st} FAILED: {e!r}")
                # A first launch can die while the ranks JIT-build one module; one relaunch is allowed.
                if "before readiness" not in repr(e):
                    break
    log("chain finished; no server left running" if not sh("pgrep -f '[v]llm serve'").strip()
        else "chain finished; WARNING a vllm serve process is still present")


if __name__ == "__main__":
    main()
