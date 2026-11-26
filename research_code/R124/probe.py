import argparse, hashlib, json, math, os, random, re, time
from pathlib import Path
from decimal import Decimal

ARMS = {"ar": None, "mtp1": {"method":"qwen3_next_mtp","num_speculative_tokens":1}, "mtp2": {"method":"qwen3_next_mtp","num_speculative_tokens":2}, "lookup4": {"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":8}}
MODEL_REV = "112d51429c6afaee530e92f97ca2659dbe7cf875"
DATA_REV = "3101c7d5072418e28b9008a6636bde82a006892c"

def digest(b): return hashlib.sha256(b).hexdigest()
def parse_answer(text):
    m = re.search(r"####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*$", text.strip())
    return str(Decimal(m.group(1).replace(",", "")).normalize()) if m else None

def choose(raw):
    rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    if len(rows) != 1319: raise ValueError("expected official GSM8K test size 1319")
    order = sorted(range(len(rows)),key=lambda i:digest(("r124-fixed-"+str(i)).encode()))[:14]
    return [{"id":i,"split":"warmup" if j<2 else "cal" if j<6 else "test","question":rows[i]["question"],"answer":parse_answer(rows[i]["answer"])} for j,i in enumerate(order)]

def spec_metrics(llm):
    vals={}
    for m in llm.get_metrics():
        n=getattr(m,"name","").removesuffix("_total")
        if n in ("vllm:spec_decode_num_drafts","vllm:spec_decode_num_draft_tokens","vllm:spec_decode_num_accepted_tokens"):
            if n in vals: raise RuntimeError("multiple metric labels; inspect")
            vals[n]=float(m.value)
    return vals

def delta(a,b):
    if a.keys()!=b.keys(): return None
    d={k:b[k]-a[k] for k in a}
    if any(not math.isfinite(x) or x<0 for x in d.values()): raise ValueError("invalid counter delta")
    return d or None

def self_test():
    assert parse_answer("work\n#### 1,200") == parse_answer("#### 1200.0")
    assert parse_answer("#### 7 and then 9") is None
    raw="\n".join(json.dumps({"question":str(i),"answer":"#### 1"}) for i in range(1319))
    a=choose(raw); assert a==choose(raw) and len({x['id'] for x in a})==14
    assert [sum(x['split']==s for x in a) for s in ('warmup','cal','test')]==[2,4,8]
    assert delta({'x':3},{'x':8})=={'x':5} and delta({}, {}) is None
    try: delta({'x':3},{'x':2})
    except ValueError: pass
    else: raise AssertionError('reset undetected')
    print('4 self-test groups passed (stdlib only)')

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--model');p.add_argument('--data');p.add_argument('--out');p.add_argument('--arm',choices=ARMS);p.add_argument('--session',type=int,choices=(0,1,2));p.add_argument('--split',choices=('cal','test'));p.add_argument('--approve',action='store_true');a=p.parse_args()
    if a.self_test: self_test();return
    if not a.approve or None in (a.model,a.data,a.out,a.arm,a.session,a.split):p.error('explicit approval and all arguments required')
    if not Path(a.model).is_dir(): raise ValueError('local verified model required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    raw=Path(a.data).read_bytes();docs=choose(raw.decode());assert all(d['answer'] is not None for d in docs)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    from vllm import LLM, SamplingParams
    import torch, vllm, transformers
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one visible CUDA GPU required')
    model_path=Path(a.model); index=json.loads((model_path/'model.safetensors.index.json').read_text()); keys=list(index['weight_map'])
    if not any('mtp.' in k for k in keys):raise RuntimeError('no MTP weights listed; inspect, do not substitute')
    config=dict(model=str(model_path),tokenizer=str(model_path),dtype='bfloat16',tensor_parallel_size=1,max_model_len=4096,gpu_memory_utilization=.8,max_num_seqs=1,enable_prefix_caching=False,disable_log_stats=False,trust_remote_code=False,generation_config='vllm',seed=124)
    if ARMS[a.arm] is not None:config['speculative_config']=ARMS[a.arm]
    manifest={'arm':a.arm,'session':a.session,'split':a.split,'model_revision_expected':MODEL_REV,'data_revision_expected':DATA_REV,'data_sha256':digest(raw),'source_sha256':digest(Path(__file__).read_bytes()),'vllm':vllm.__version__,'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'config':config,'docs':docs,'mtp_weight_keys':sum('mtp.' in k for k in keys)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    t=time.perf_counter();llm=LLM(**config);(out/'startup.json').write_text(json.dumps({'seconds':time.perf_counter()-t}))
    tok=llm.get_tokenizer();params=SamplingParams(temperature=0,top_p=1,max_tokens=1024,ignore_eos=False,detokenize=True,seed=124)
    prompts={d['id']:tok.apply_chat_template([{'role':'user','content':d['question']+'\nSolve the problem. End your response with #### followed by the numerical answer.'}],tokenize=False,add_generation_prompt=True) for d in docs}
    active=[d for d in docs if d['split']==a.split];random.Random(124+a.session).shuffle(active)
    with (out/'runs.jsonl').open('x') as f:
        for d in [x for x in docs if x['split']=='warmup']+active:
            before=spec_metrics(llm);t=time.perf_counter();o=llm.generate([prompts[d['id']]],params,use_tqdm=False)[0];elapsed=time.perf_counter()-t;after=spec_metrics(llm);c=o.outputs[0]
            row={'id':d['id'],'split':d['split'],'arm':a.arm,'session':a.session,'seconds':elapsed,'prompt_ids':list(o.prompt_token_ids),'token_ids':list(c.token_ids),'text':c.text,'finish_reason':c.finish_reason,'stop_reason':c.stop_reason,'correct':parse_answer(c.text)==d['answer'],'spec_counters':delta(before,after)}
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    print('completed',a.arm,a.session,a.split)

if __name__=='__main__':main()
