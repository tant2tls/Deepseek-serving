python3 -m venv ~/vllm
source ~/vllm/bin/activate
# Install uv.
python -m pip install --upgrade uv

# Pinned runtime used by every study in this repository (report.md, report_mimo.md):
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

# For new work on a newer runtime instead (results will not be comparable):
# uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
