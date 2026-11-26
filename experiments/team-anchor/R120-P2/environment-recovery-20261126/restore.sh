set -eu
cd /root/rivermind-data
mkdir -p r120
cd r120
git clone https://github.com/CK-NOMI/idea.git repo
git -C repo checkout --detach 18d4a3ec7da6851ca632ff73cd2a3d92674a1fea
/opt/conda/bin/python -m venv --system-site-packages venv
venv/bin/pip install transformers==4.48.3
mkdir -p layerskip-1b
for f in model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json LICENSE; do
curl -fL --retry 4 --connect-timeout 20 -C - -o "layerskip-1b/$f" "https://www.modelscope.cn/api/v1/models/facebook/layerskip-llama3.2-1B/repo?Revision=master&FilePath=$f"
done
venv/bin/python - <<'PY'
import json,hashlib,pathlib,torch,transformers
root=pathlib.Path('/root/rivermind-data/r120')
m=json.loads((root/'repo/experiments/team-anchor/R120-P1/manifest.json').read_text())
expected=dict(m['model_files'],**{'model.safetensors':m['weight_sha256']})
actual={}
for name,h in expected.items():
    got=hashlib.file_digest((root/'layerskip-1b'/name).open('rb'),'sha256').hexdigest()
    actual[name]=got
    assert got==h,(name,got,h)
assert transformers.__version__=='4.48.3'
assert torch.cuda.is_available()
report=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),model_sha256=actual,source_commit='18d4a3ec7da6851ca632ff73cd2a3d92674a1fea')
(root/'setup-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
venv/bin/pip freeze > requirements-recovered.txt
