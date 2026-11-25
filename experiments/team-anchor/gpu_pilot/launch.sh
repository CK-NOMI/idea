#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the verified persistent data disk mount}"
ROOT="$DATA_ROOT/hfl_sd_gpu_pilot"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT"
export HF_HOME="$ROOT/hf"
export HF_DATASETS_CACHE="$ROOT/hf/datasets"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export TOKENIZERS_PARALLELISM=false
STAGE="${1:-prepare}"
case "$STAGE" in
prepare)
  df -h "$ROOT"
  python -m venv --system-site-packages "$ROOT/venv"
  PY="$ROOT/venv/bin/python"
  "$PY" -m pip install "transformers==4.48.3" "datasets==3.2.0" "huggingface_hub==0.28.1" "accelerate==1.3.0" "safetensors==0.5.2"
  "$PY" -m py_compile "$HERE/pilot.py"
  "$PY" "$HERE/pilot.py" prepare --root "$ROOT" 2>&1 | tee "$ROOT/prepare.log"
  ;;
pilot)
  test -f "$ROOT/prepared.json"
  nvidia-smi
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  timeout --signal=TERM --kill-after=30s 60m "$ROOT/venv/bin/python" -u "$HERE/pilot.py" pilot --root "$ROOT" --seed "${2:-11}" 2>&1 | tee "$ROOT/pilot_seed${2:-11}.log"
  ;;
*) echo "Usage: bash launch.sh prepare|pilot [seed]" >&2;exit 2;;
esac
