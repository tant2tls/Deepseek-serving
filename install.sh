python3 -m venv ~/vllm
source ~/vllm/bin/activate
# Install uv.
python -m pip install --upgrade uv

# Install the prebuilt vLLM nightly and its dependencies.
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
uv pip install nvitop