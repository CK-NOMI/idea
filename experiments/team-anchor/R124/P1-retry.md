# R124-P1-retry | Isolated environment bootstrap without ensurepip
Approved recovery of failed P1, source commit 77f6af61823a1d5bda17c81f8c9e6f4c2a7add23. Same authorized server only. The failed /root/rivermind-data/r124-p1 and R120 environment must be retained unchanged; never delete, rename, reuse, or modify them. New base /root/rivermind-data/r124-p1-retry must not exist; stop if it does. No apt, sudo, system changes, TLS bypass, alternate model or server. No LLM inference/training. This supersedes ONLY P1 preparation commands; all scientific model/data/version checks remain unchanged.
Use one bash session, set -e and pipefail, and record exact commands, separate stdout/stderr, numeric exit codes for each step, plus stage exit. If a command fails, stop dependent steps and archive. The original P1 logs and archive are retained; write new logs only to the retry directory. No credentials in logs. 
## 1. Create environments without pip, bootstrap verified pip wheel
```bash
set -e
set -o pipefail
BASE=/root/rivermind-data/r124-p1-retry
test ! -e "$BASE"
test -d /root/rivermind-data/r124-p1
mkdir "$BASE"
mkdir "$BASE/tmp" "$BASE/logs" "$BASE/bootstrap"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/hf-cache" MODELSCOPE_CACHE="$BASE/ms-cache" XDG_CACHE_HOME="$BASE/cache" PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_CONFIG_FILE=/dev/null
unset PYTHONPATH PYTHONHOME
df -h "$BASE"
python3 --version
python3 -m venv --without-pip "$BASE/download-venv"
python3 -m venv --without-pip "$BASE/venv"
# Retrieve only official PyPI release metadata and one fixed wheel.
curl --fail --location --connect-timeout 15 --max-time 60 -o "$BASE/bootstrap/pip-25.3.json" https://pypi.org/pypi/pip/25.3/json
WHEEL_URL=$(python3 - "$BASE/bootstrap/pip-25.3.json" <<'PY'
import json,sys
from urllib.parse import urlparse
j=json.load(open(sys.argv[1]))
xs=[x for x in j['urls'] if x['filename']=='pip-25.3-py3-none-any.whl']
assert len(xs)==1
x=xs[0]
assert x['digests']['sha256']=='9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd'
u=urlparse(x['url'])
assert u.scheme=='https' and u.netloc=='files.pythonhosted.org'
print(x['url'])
PY
)
WHEEL="$BASE/bootstrap/pip-25.3-py3-none-any.whl"
curl --fail --location --connect-timeout 15 --max-time 120 -o "$WHEEL" "$WHEEL_URL"
printf '%s  %s\n' '9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd' "$WHEEL" | sha256sum -c -
# Use the verified wheel as the temporary pip module and install it into each target venv only.
for name in download-venv venv; do
    PY="$BASE/$name/bin/python"
    "$PY" -c 'import sys,pathlib; assert sys.prefix!=sys.base_prefix; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve(); print(sys.executable,sys.prefix,sys.version)' "$BASE/$name"
    PYTHONPATH="$WHEEL" "$PY" -m pip --isolated install --no-index --no-deps --ignore-installed "$WHEEL"
    "$PY" -c 'import pip,sys,pathlib; assert pip.__version__=="25.3"; assert pathlib.Path(pip.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()); print(pip.__version__,pip.__file__)'
    "$PY" -m pip --version
done
# Both pip bootstraps must pass before continuing; no system pip or old R120 pip used.
"$BASE/download-venv/bin/python" -m pip --isolated install --no-cache-dir --index-url https://pypi.org/simple modelscope==1.34.0
"$BASE/venv/bin/python" -m pip --isolated install --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu129 'vllm==0.17.1'
"$BASE/venv/bin/python" -m pip check
"$BASE/download-venv/bin/python" -m pip check
"$BASE/venv/bin/python" -m pip freeze > "$BASE/logs/pip-freeze.txt"
"$BASE/download-venv/bin/python" -m pip freeze > "$BASE/logs/download-pip-freeze.txt"
df -h "$BASE"
```

Use minimum free 15 GiB before any remaining download; stop if lower. No new build/toolchain if wheel installation fails. Use official indices only; no --trusted-host or TLS bypass. Version is a candidate, not proven on RTX3090. Do not override torch or transformers version resolved by vLLM. Before GPU use verify torch 2.10.0 version (allow CUDA suffix), vllm 0.17.1, CUDA runtime and capability, BF16 native support, one visible GPU, and execute only a tiny BF16 matrix multiplication:
```sh
CUDA_VISIBLE_DEVICES=0 "$BASE/venv/bin/python" -c 'import torch,vllm,transformers; print(torch.__version__,torch.version.cuda,vllm.__version__,transformers.__version__); assert torch.__version__.split("+")[0]=="2.10.0" and vllm.__version__=="0.17.1"; assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(),torch.cuda.get_device_capability()); assert torch.cuda.get_device_capability()==(8,6); x=torch.ones((32,32),device="cuda",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert bool((y==32).all()); print("BF16 smoke passed")'
```
## 2. Download public official model
Use ModelScope publisher Qwen/Qwen3.5-0.8B, master only as acquisition reference, not a claimed immutable revision. Record publicly returned snapshot/file metadata before/after when available, source and acquisition time; if metadata lookup fails record it. Download via installed CLI:
```sh
"$BASE/download-venv/bin/modelscope" download --model Qwen/Qwen3.5-0.8B --revision master --local_dir "$BASE/model"
curl --fail --location --connect-timeout 15 --max-time 120 -o "$BASE/gsm8k-test.jsonl" https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl
```
No substitute models (Base, GGUF, quantized, or third-party). No weights uploaded to GitHub. Master snapshot may differ from the pinned HF revision in small files; record every local non-cache artifact SHA256, size, source. 
## 3. Verify artifacts before any LLM inference
```sh
"$BASE/venv/bin/python" - "$BASE" <<'PY'
import hashlib,json,sys
from pathlib import Path
from safetensors import safe_open
b=Path(sys.argv[1]); m=b/'model'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
w=m/'model.safetensors-00001-of-00001.safetensors'
t=m/'tokenizer.json'
assert sha(w)=='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696','official pinned weight mismatch'
assert sha(t)=='5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42','official pinned tokenizer mismatch'
index=json.loads((m/'model.safetensors.index.json').read_text())['weight_map']
assert set(index.values())=={w.name}
with safe_open(str(w),framework='pt',device='cpu') as f:
    keys=set(f.keys())
assert keys==set(index),'index and weight header mismatch'
mtp=sorted(k for k in keys if 'mtp.' in k)
assert mtp,'MTP weights missing'
data=(b/'gsm8k-test.jsonl').read_bytes()
assert hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()=='e4c2ff4942b9a78bd74f04141224c11e28d12dc9'
assert len([json.loads(x) for x in data.splitlines() if x.strip()])==1319
files=[{'path':str(p.relative_to(m)),'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(m.rglob('*')) if p.is_file() and not any(c.startswith('.') for c in p.relative_to(m).parts)]
out={'source':'ModelScope Qwen/Qwen3.5-0.8B master acquired snapshot; pinned HF weight/tokenizer byte matches only','hf_reference':'112d51429c6afaee530e92f97ca2659dbe7cf875','files':files,'mtp_keys':mtp,'gsm8k_sha256':hashlib.sha256(data).hexdigest()}
(b/'logs/artifact-manifest.json').write_text(json.dumps(out,indent=2))
print('weight/tokenizer/data hashes and header verified; MTP tensors',len(mtp))
PY
df -h "$BASE"
```
Run only setup, download, verification and tiny CUDA smoke, NOT model inference or probe.py --approve, no train. The remaining full model/config provenance and vLLM runtime are reviewed before releasing calibration. If hash mismatch or missing index, stop, preserve artifact hashes/error; no retry with a different model. 
Archive sanitized commands, logs, exact numeric exits, pip freeze, artifact manifest, and text copies of small model metadata config.json, generation_config.json (if present), tokenizer_config.json, chat_template.jinja (if present), model.safetensors.index.json and preprocessor_config.json (if present), no weights or private paths/secrets, under experiments/team-anchor/R124-P1-retry/. Provide result commit and readback. If raw.githubusercontent.com unavailable report separately; no substitute dataset. 


Additional retry archive: include the pip release metadata or its SHA256 and the verified wheel hash (not the wheel itself), target pip paths, venv prefixes, stage logs and exits. Archive only to experiments/team-anchor/R124-P1-retry/, never overwrite P1. If pip bootstrap fails, do not improvise with conda/apt/get-pip.py; report exact failure. Do not rerun a failed retry in place. This recovery is based on documented venv --without-pip behavior and official PyPI wheel metadata; not yet executed on the GPU host.
