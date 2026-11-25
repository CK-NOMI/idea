"""R120-P1: bounded HF-only oracle timing diagnostic, not a deployable SD."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time
import unittest
from preflight import HFBackend, Toy, verify, file_hash

N, REPS = 64, 3
KS = (1, 2, 4, 8, 16)
SD = tuple(f'sd:{e}:{k}' for e in (3, 4, 8) for k in (2, 4, 8))
DATA_HASH = 'eca6e363a3070ebb6e1610ab273091c0ae6418f27efaf5c39e3060430d81ecf9'
WEIGHT_HASH = 'ba9a27bc1fadc74d7ca576aa618e464aa006b1c56b4a940c668c980a302708ee'
CONFIG_HASH = '0fa5bb77b0c8f1a5a335fd05773ed60522c015b6792324fe91904a757d7f815a'
TOKENIZER_HASH = '6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b'


def select_docs(rows):
    out = []
    for split, count in [('calibration', 1), ('test', 2)]:
        for c in range(4):
            chosen = [r for r in rows if r['split'] == split and r['client'] == c][:count]
            if len(chosen) != count:
                raise ValueError('insufficient documents')
            out.extend(chosen)
    if len({r['id'] for r in out}) != 12 or len({r['text'] for r in out}) != 12:
        raise ValueError('duplicate document')
    return out


def validate_oracle(r, prompt_len, k, ref):
    pos, out = prompt_len - 1, []
    for b in r['blocks']:
        m = min(k, N - len(out) - 1)
        if m < 0 or b['cache_before'] != pos or len(b['draft']) != m:
            return False
        if b['draft'] != ref[len(out):len(out)+m] or len(b['predictions']) != m+1:
            return False
        a = next((i for i in range(m) if b['draft'][i] != b['predictions'][i]), m)
        if a != m or b['accepted'] != a or b['emitted'] != b['draft'][:a]+[b['predictions'][a]]:
            return False
        pos += 1+a
        if b['cache_after'] != pos:
            return False
        out.extend(b['emitted'])
    return out == ref and r['output_ids'] == ref and len(ref) == N


def aggregate(rows, arm, refs, repeats):
    selected = [r for r in rows if r['arm'] == arm]
    keys = [(r['id'], r['rep']) for r in selected]
    expected = {(i, j) for i in refs for j in range(repeats)}
    if len(keys) != len(expected) or set(keys) != expected:
        return None
    if any(not r['valid'] or r['output_ids'] != refs[r['id']] or
           len(r['output_ids']) != N or not math.isfinite(r['seconds']) or r['seconds'] <= 0
           for r in selected):
        return None
    return {i: (sum(r['seconds'] for r in selected if r['id'] == i)/repeats,
                sum(r.get('rounds', 0) for r in selected if r['id'] == i)/repeats) for i in refs}


def compare(base, cand, oracle=False):
    if base is None or cand is None or set(base) != set(cand) or not base:
        return None
    ids = list(base)
    def calc(keys):
        b = sum(base[i][0] for i in keys)
        c = sum(cand[i][0] for i in keys)
        m = sum(cand[i][1] for i in keys)
        return [b/c]+([(b/s-c)/m for s in (1.5, 2.0)] if oracle else [])
    if oracle and any(cand[i][1] <= 0 for i in ids):
        return None
    point = calc(ids)
    rng = random.Random(120)
    samples = [calc(rng.choices(ids, k=len(ids))) for _ in range(5000)]
    def ci(j):
        v = sorted(x[j] for x in samples)
        return [v[125], v[4874]]
    result = dict(speedup=point[0], ci95=ci(0), documents=len(ids))
    if oracle:
        result.update(draft_budget_seconds_per_round_1_5x=point[1], budget_1_5x_ci95=ci(1),
                      draft_budget_seconds_per_round_2x=point[2], budget_2x_ci95=ci(2),
                      mean_committed_per_round=N*len(ids)/sum(cand[i][1] for i in ids))
    return result


class Tests(unittest.TestCase):
    def test_ratio_budget(self):
        a={'a':(1,0),'b':(9,0)}; b={'a':(.1,2),'b':(9,2)}
        self.assertAlmostEqual(compare(a,b)['speedup'],10/9.1)
        self.assertAlmostEqual(compare({'a':(1,0)},{'a':(.9,4)},True)['draft_budget_seconds_per_round_1_5x'], -0.0583333333333333)
    def test_integrity(self):
        refs={'x':[1]*N}; row=dict(id='x',rep=0,arm='a',valid=True,seconds=1,output_ids=refs['x'])
        self.assertIsNotNone(aggregate([row],'a',refs,1))
        for bad in ([], [row,row], [dict(row,valid=False)], [dict(row,output_ids=[2]*N)], [dict(row,seconds=float('nan'))]):
            self.assertIsNone(aggregate(bad,'a',refs,1))
    def test_oracle(self):
        p=[3,5]; ref=verify(p,N,0,Toy())['output_ids']
        for k in KS:
            r=verify(p,N,k,Toy(),ref)
            self.assertTrue(validate_oracle(r,len(p),k,ref))
            broken=copy.deepcopy(r); broken['blocks'][0]['cache_after']+=1
            self.assertFalse(validate_oracle(broken,len(p),k,ref))
            self.assertFalse(validate_oracle(dict(r,blocks=[]),len(p),k,ref))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true')
    p.add_argument('--model',type=Path); p.add_argument('--out',type=Path)
    p.add_argument('--approve-pilot',action='store_true'); args=p.parse_args()
    if args.self_test:
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if not(args.approve_pilot and args.model and args.out): p.error('need --approve-pilot --model --out')
    args.out.mkdir(parents=True,exist_ok=False)
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig
    if transformers.__version__!='4.48.3' or not torch.cuda.is_available(): raise RuntimeError('environment mismatch')
    for name, h in [('model.safetensors',WEIGHT_HASH),('config.json',CONFIG_HASH),('tokenizer.json',TOKENIZER_HASH)]:
        if file_hash(args.model/name)!=h: raise RuntimeError('model file mismatch: '+name)
    if list(args.model.glob('*.safetensors')) != [args.model/'model.safetensors']: raise RuntimeError('unexpected weights')
    data=Path(__file__).resolve().parents[2]/'experiments/team-anchor/R119-P0/results/prompts.json'
    if file_hash(data)!=DATA_HASH: raise RuntimeError('input data hash mismatch')
    docs=select_docs(json.loads(data.read_text()))
    tok=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(args.model),local_files_only=True,torch_dtype=torch.float16,
                 attn_implementation='eager',use_safetensors=True).to('cuda').eval()
    backend=HFBackend(model,torch,DynamicCache)
    prompts={r['id']:tok(r['text'],truncation=True,max_length=64)['input_ids'] for r in docs}
    if any(not ids for ids in prompts.values()): raise RuntimeError('empty input')
    meta=dict(scope='short natural-text HF-only pilot; not official cross-environment or paper benchmark',
              model_files={f.name:file_hash(f) for f in args.model.iterdir() if f.is_file() and f.suffix=='.json'},
              weight_sha256=WEIGHT_HASH, data_sha256=DATA_HASH, source_sha256=file_hash(Path(__file__)),
              preflight_sha256=file_hash(Path(__file__).with_name('preflight.py')),
              torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
              attention_backend=model.config._attn_implementation,N=N,repeats=REPS,
              docs=[dict(id=r['id'],split=r['split'],client=r['client'],input_ids=prompts[r['id']]) for r in docs])
    (args.out/'manifest.json').write_text(json.dumps(meta,indent=2))
    def run(ids, arm, ref=None, audit=False):
        is_manual=arm=='manual' or arm.startswith('oracle:')
        k=int(arm.split(':')[1]) if arm.startswith('oracle:') else 0
        orig=model.generation_config; hooks=[]; depths=[]
        if not is_manual:
            cfg=GenerationConfig(do_sample=False,max_new_tokens=N,eos_token_id=None,pad_token_id=0,use_cache=True)
            if arm.startswith('sd:'):
                _,e,b=arm.split(':'); cfg.assistant_early_exit=int(e); cfg.num_assistant_tokens=int(b)
                cfg.num_assistant_tokens_schedule='constant'; cfg.assistant_confidence_threshold=0.0
            model.generation_config=copy.deepcopy(cfg); model.generation_config.assistant_early_exit=None
            x=torch.tensor([ids],device=model.device); mask=torch.ones_like(x)
            if audit:
                hooks.append(model.register_forward_pre_hook(lambda *_: depths.append(model.config.num_hidden_layers)))
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            with torch.inference_mode():
                if is_manual: result=verify(ids,N,k,backend,ref)
                else: result=dict(output_ids=model.generate(x,attention_mask=mask,generation_config=cfg)[0,len(ids):].tolist())
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            result.update(seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),arm=arm,depths=depths)
            if is_manual: result['rounds']=len(result['blocks'])
            return result
        finally:
            model.generation_config=orig
            for h in hooks: h.remove()
    # Warm up each HF arm; ensure early-exit is actually observed before selection.
    first=prompts[docs[0]['id']]; warm={a:run(first,a,audit=True) for a in ('ar',)+SD}
    eligible=[a for a in SD if int(a.split(':')[1]) in warm[a]['depths'] and model.config.num_hidden_layers in warm[a]['depths']]
    (args.out/'warmup.json').write_text(json.dumps(warm))
    refs={r['id']:run(prompts[r['id']],'ar')['output_ids'] for r in docs}
    if any(len(v)!=N for v in refs.values()): raise RuntimeError('reference length mismatch')
    (args.out/'references.json').write_text(json.dumps(refs))
    for k in KS: run(first,f'oracle:{k}',refs[docs[0]['id']])
    run(first,'manual')
    rows=[]
    with (args.out/'runs.jsonl').open('w') as f:
        def phase(split, arms, repeats):
            for i,d in enumerate(docs):
                if d['split']!=split: continue
                rid=d['id']; ids=prompts[rid]
                for rep in range(repeats):
                    order=list(arms); random.Random(120+i*100+rep).shuffle(order)
                    for a in order:
                        r=run(ids,a,refs[rid]); valid=r['output_ids']==refs[rid] and len(r['output_ids'])==N
                        if a.startswith('oracle:'): valid=valid and validate_oracle(r,len(ids),int(a.split(':')[1]),refs[rid])
                        r.update(id=rid,split=split,rep=rep,valid=valid); rows.append(r)
                        f.write(json.dumps(r,allow_nan=False)+'\n'); f.flush()
        phase('calibration',('ar','manual')+tuple(eligible),1)
        cal={r['id']:refs[r['id']] for r in docs if r['split']=='calibration'}
        stats={a:aggregate([r for r in rows if r['split']=='calibration'],a,cal,1) for a in ('ar','manual')+tuple(eligible)}
        if stats['ar'] is None: raise RuntimeError('unstable calibration AR')
        valid_sd=[a for a in eligible if stats[a] is not None]
        best_sd=min(valid_sd,key=lambda a:sum(t[0] for t in stats[a].values())) if valid_sd else None
        best=min([a for a in stats if stats[a] is not None],key=lambda a:sum(t[0] for t in stats[a].values()))
        lock=dict(best_sd=best_sd,best_baseline=best,eligible_sd=eligible,calibration=stats)
        (args.out/'lock.json').write_text(json.dumps(lock,indent=2))
        arms=('ar','manual')+((best_sd,) if best_sd else ())+tuple(f'oracle:{k}' for k in KS)
        phase('test',arms,REPS)
    test={r['id']:refs[r['id']] for r in docs if r['split']=='test'}
    stats={a:aggregate([r for r in rows if r['split']=='test'],a,test,REPS) for a in arms}
    summary=dict(lock=lock,valid={a:v is not None for a,v in stats.items()},results={})
    for k in KS:
        a=f'oracle:{k}'; summary['results'][a]={b:compare(stats.get(b),stats[a],True) for b in dict.fromkeys(['ar','manual',best]+([best_sd] if best_sd else []))}
    if stats['ar'] is None: summary['results']={}; summary['error']='unstable test AR; no speed conclusions'
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if stats['ar'] is not None else 1


if __name__=='__main__':
    raise SystemExit(main())
