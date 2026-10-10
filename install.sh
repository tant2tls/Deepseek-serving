python3 -m venv ~/vllm
source ~/vllm/bin/activate
# Install uv.
python -m pip install --upgrade uv

# Pinned runtime used by every study in this repository (reports/v41-vs-0731/report.md, reports/mimo-v26/report.md):
# vLLM commit 44af287ebe38d6dc4e102948025f5e3e175aefd6 = 0.30.1rc1.dev223+g44af287eb.
# Several findings are build-specific (no V4.1 CED prefill path, MiMo MTP uses layer 0 only,
# kernel selection), so reproduce with this exact wheel, not the moving nightly.
VLLM_COMMIT=44af287ebe38d6dc4e102948025f5e3e175aefd6
# --torch-backend=auto on this driver (580.105.08) resolves torch 2.13.0+cu132,
# torchvision 0.28.0+cu132, torchaudio 2.11.0+cpu, exactly as in the lock. Checked 2026-09-28:
# a fresh resolution matched the lock for 200/201 packages (filelock pinned below).
uv pip install "vllm==0.30.1rc1.dev223+g44af287eb" filelock==4.0.4 pandas \
    --torch-backend=auto \
    --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}"
uv pip install nvitop   # monitoring only; its nvidia-ml-py pin conflicts with the lock and is not used by measurements

# Verify against the recorded lock (reports/environment-lock.txt); differences are listed.
uv pip freeze | diff <(grep -v '^#' "$(dirname "$0")/reports/environment-lock.txt") - \
    && echo "environment matches lock" || echo "environment differs from lock (see diff above)"

# Blog study `blog-architecture-h100-v1` (2026-10-07) runs on the then-latest nightly in a SEPARATE
# venv, so the pinned environment above stays intact for the completed studies:
#   vLLM 0.31.1rc1.dev50+g554340f3d = commit 554340f3d3259e321be4c07282be7a02a5aeef83, torch 2.13.0+cu132.
# Select it with `export VLLM_VENV=/root/vllm-latest` (bench/serve.sh and bench/run_matrix.py honour it).
# Results from the two runtimes are not comparable without fresh controls.
BLOG_VLLM_COMMIT=554340f3d3259e321be4c07282be7a02a5aeef83
python3 -m venv ~/vllm-latest
uv pip install --python ~/vllm-latest/bin/python "vllm==0.31.1rc1.dev50+g554340f3d" pandas \
    --torch-backend=auto --index-strategy unsafe-best-match --prerelease=allow \
    --extra-index-url "https://wheels.vllm.ai/${BLOG_VLLM_COMMIT}"
# To move to the newest nightly instead (a new runtime arm; record the resolved commit):
# uv pip install --python ~/vllm-latest/bin/python -U vllm pandas --torch-backend=auto \
#     --index-strategy unsafe-best-match --prerelease=allow --extra-index-url https://wheels.vllm.ai/nightly

# Latest nightly for the five-model rerun and the B200 session (docs/H200_B200_plan.md, target.md#five-model-rerun-plan).
# Resolved on 2026-10-10 from wheels.vllm.ai/nightly: vLLM 0.31.1rc1.dev261+gc41b2639e, commit
# c41b2639e29c3bc01add1d34bef3032a6d9d8aca (committed 2026-10-10 UTC). The only dev build on the index that day.
# Not yet verified on B200 (sm_100): run a smoke test and record the wheel SHA256 and the resolved lock before any measurement.
# The venv is separate from ~/vllm-latest so the completed blog study keeps its runtime.
RERUN_VLLM_COMMIT=c41b2639e29c3bc01add1d34bef3032a6d9d8aca
# CPython 3.12.3 (/usr/bin/python3.12) matches the recorded corpus and older environment; `python3` here is 3.14.
/usr/bin/python3.12 -m venv ~/vllm-b200
uv pip install --python ~/vllm-b200/bin/python "vllm==0.31.1rc1.dev261+gc41b2639e" pandas \
    --torch-backend=auto --index-strategy unsafe-best-match --prerelease=allow \
    --extra-index-url "https://wheels.vllm.ai/${RERUN_VLLM_COMMIT}"
