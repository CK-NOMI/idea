#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the mounted persistent data disk}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$DATA_ROOT/hfl_sd_r118"
mkdir -p "$ROOT"
cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PIP_CACHE_DIR="$ROOT/pip-cache"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_properties(0))'
python -m venv --system-site-packages "$ROOT/venv"
PY="$ROOT/venv/bin/python"
"$PY" -m pip install 'transformers==4.48.3' 'huggingface_hub==0.36.2' 'safetensors==0.8.0'
timeout 35m "$PY" -u "$HERE/prepare.py" --root "$ROOT"
status=0
timeout 15m "$PY" -u "$HERE/latency.py" --root "$ROOT" --out "$ROOT/layerskip_1b_cuda_latency.json" || status=1
timeout 15m "$PY" -u "$HERE/quality.py" --root "$ROOT" --device cuda --out "$ROOT/layerskip_1b_cuda_quality.json" --per-class 6 --exits 3 8 15 || status=1
timeout 10m "$PY" -u "$HERE/anchor_audit.py" --root "$ROOT" --device cuda --out "$ROOT/layerskip_1b_cuda_anchor.json" || status=1
exit "$status"
