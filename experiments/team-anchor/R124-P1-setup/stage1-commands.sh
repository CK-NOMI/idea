#!/bin/bash
BASE=/root/rivermind-data/r124-p1
test ! -e "$BASE" || exit 90
mkdir "$BASE" || exit
mkdir "$BASE/tmp" "$BASE/logs" || exit
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/hf-cache" MODELSCOPE_CACHE="$BASE/ms-cache" XDG_CACHE_HOME="$BASE/cache" PIP_NO_CACHE_DIR=1
step() { n="$1"; shift; "$@" >"$BASE/logs/$n.stdout" 2>"$BASE/logs/$n.stderr"; rc=$?; printf '%d\n' "$rc" >"$BASE/logs/$n.exit"; return "$rc"; }
step disk-before df -h "$BASE" || exit
step download-venv python3 -m venv "$BASE/download-venv" || exit
step modelscope-install "$BASE/download-venv/bin/python" -m pip install --no-cache-dir --index-url https://pypi.org/simple modelscope==1.34.0 || exit
step vllm-venv python3 -m venv "$BASE/venv" || exit
step vllm-install "$BASE/venv/bin/python" -m pip install --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu129 'vllm==0.17.1' || exit
step pip-check "$BASE/venv/bin/python" -m pip check || exit
step pip-freeze "$BASE/venv/bin/python" -m pip freeze || exit
step download-pip-freeze "$BASE/download-venv/bin/python" -m pip freeze || exit
step disk-after df -h "$BASE" || exit
