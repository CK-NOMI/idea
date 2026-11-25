#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the mounted persistent data disk}"
UPSTREAM="${1:?Provide the pinned facebookresearch/LayerSkip checkout}"
R104="${2:?Provide the project R104 official_pilot.py path}"
ROOT="$DATA_ROOT/hfl_sd_r118"
PY="$ROOT/venv/bin/python"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The upstream split-forward helper uses the pre-4.47 Llama decoder signature.
"$PY" -m pip install --no-deps --target "$ROOT/official_deps_446" \
  'transformers==4.46.3' 'tokenizers==0.20.3' 'colorama==0.4.6'
export PYTHONPATH="$ROOT/official_deps_446${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
counts = {}
with (root / "official_prompts.jsonl").open("w") as f:
    for row in json.loads((root / "prompts.json").read_text()):
        key = (row["split"], row["client"])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= 2:
            item = dict(id=row["id"], split=row["split"], region=str(row["region"]), prompt=row["text"])
            f.write(json.dumps(item) + "\n")
PY
cd "$ROOT"
timeout 25m "$PY" -u "$R104" --upstream "$UPSTREAM" \
  --model-snapshot "$ROOT/layerskip-1b" --prompts "$ROOT/official_prompts.jsonl" \
  --out "$ROOT/official_cuda_latency.jsonl" \
  --exits 3 4 8 --drafts 2 4 8 --tokens 32 --repeats 1
