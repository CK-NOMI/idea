"""R120-P3: same-precision FP32 oracle budget pilot; no deployable speed claim."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time
import unittest

N, REPS = 64, 3
KS = (1, 2, 4, 8, 16)
SD = tuple(f'sd:{e}:{k}' for e in (3, 4, 8) for k in (2, 4, 8))


def valid_trace(z, prompt_len, k, ref):
    out = z['output_ids'][:1]
    pos = prompt_len
    if len(ref) != N or out != ref[:1]:
        return False
    for b in z['blocks']:
        m = min(k, N-len(out)-1)
        if m < 0 or b['before'] != pos or b['draft'] != ref[len(out):len(out)+m]:
            return False
        if b['accepted'] != m or len(b['emitted']) != m+1 or b['emitted'] != ref[len(out):len(out)+m+1]:
            return False
        pos += 1+m
        if b['after'] != pos:
            return False
        out.extend(b['emitted'])
    return out == ref == z['output_ids']


def aggregate(rows, arm, refs, repeats):
    a = [r for r in rows if r['arm']==arm]
    expected = {(i,j) for i in refs for j in range(repeats)}
    keys = [(r['id'],r['rep']) for r in a]
    if len(keys)!=len(expected) or set(keys)!=expected:
        return None
    if any(not r['valid'] or r['output_ids']!=refs[r['id']] or len(r['output_ids'])!=N or not math.isfinite(r['seconds']) or r['seconds']<=0 for r in a):
        return None
    return {i:(sum(r['seconds'] for r in a if r['id']==i)/repeats,
               sum(r.get('draft_calls',0) for r in a if r['id']==i)/repeats) for i in refs}


def compare(base, cand, oracle=False):
    if base is None or cand is None or not base or set(base)!=set(cand): return None
    ids=list(base)
    if oracle and any(cand[i][1]<=0 for i in ids): return None
    def calc(keys):
        b=sum(base[i][0] for i in keys); c=sum(cand[i][0] for i in keys); m=sum(cand[i][1] for i in keys)
        return [b/c]+([(b/s-c)/m for s in (1.5,2.)] if oracle else [])
    x=calc(ids); rng=random.Random(1203)
    samples=[calc(rng.choices(ids,k=len(ids))) for _ in range(5000)]
    def ci(j):
        v=sorted(x[j] for x in samples); return [v[125],v[4874]]
    out=dict(speedup=x[0],ci95=ci(0),documents=len(ids))
    if oracle: out.update(draft_budget_seconds_per_nonempty_call_1_5x=x[1],budget_1_5x_ci95=ci(1),draft_budget_seconds_per_nonempty_call_2x=x[2],budget_2x_ci95=ci(2))
    return out


class Tests(unittest.TestCase):
    def test_ratio_budget(self):
        self.assertAlmostEqual(compare({'a':(1,0),'b':(9,0)},{'a':(.1,1),'b':(9,1)})['speedup'],10/9.1)
        self.assertAlmostEqual(compare({'a':(1,0)},{'a':(.9,4)},True)['draft_budget_seconds_per_nonempty_call_1_5x'],-.0583333333333333)
    def test_aggregate(self):
        refs={'a':[1]*N}; r=dict(id='a',arm='ar',rep=0,valid=True,seconds=1,output_ids=refs['a'])
        self.assertIsNotNone(aggregate([r],'ar',refs,1))
        for bad in ([],[r,r],[dict(r,valid=False)],[dict(r,seconds=float('nan'))],[dict(r,output_ids=[2]*N)]):
            self.assertIsNone(aggregate(bad,'ar',refs,1))
    def test_trace(self):
        ref=list(range(N))
        for k in (0,)+KS:
            out=ref[:1]; pos=64; blocks=[]
            while len(out)<N:
                m=min(k,N-len(out)-1); emitted=ref[len(out):len(out)+m+1]
                blocks.append(dict(before=pos,after=pos+m+1,draft=emitted[:-1],accepted=m,emitted=emitted)); pos+=m+1; out+=emitted
            z=dict(output_ids=out,blocks=blocks)
            self.assertTrue(valid_trace(z,64,k,ref)); self.assertFalse(valid_trace(dict(z,blocks=[]),64,k,ref))
            broken=copy.deepcopy(z); broken['blocks'][0]['after']+=1; self.assertFalse(valid_trace(broken,64,k,ref))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--model',type=Path); p.add_argument('--out',type=Path); args=p.parse_args()
    if args.self_test: return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if not(args.model and args.out): p.error('need --model --out')
    args.out.mkdir(parents=True,exist_ok=False)
    import torch, transformers
    from transformers import AutoModelForCausalLM, DynamicCache, GenerationConfig
    from diagnose import Backend, canonical
    from diagnose_v2 import fixed_generate
    from preflight import file_hash
    if transformers.__version__!='4.48.3' or not torch.cuda.is_available(): raise RuntimeError('environment mismatch')
    prev=Path(__file__).resolve().parents[2]/'experiments/team-anchor/R120-P1/manifest.json'
    manifest=json.loads(prev.read_text()); docs=manifest['docs']
    if len(docs)!=12 or len({d['id'] for d in docs})!=12 or sum(d['split']=='calibration' for d in docs)!=4 or sum(d['split']=='test' for d in docs)!=8: raise RuntimeError('unexpected inputs')
    # Reuse the verified checkpoint and inputs; no downloads or tokenization.
    for name,h in dict(manifest['model_files'],**{'model.safetensors':manifest['weight_sha256']}).items():
        if file_hash(args.model/name)!=h: raise RuntimeError('model file mismatch '+name)
    if list(args.model.glob('*.safetensors'))!=[args.model/'model.safetensors']: raise RuntimeError('unexpected weights')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(str(args.model),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager',use_safetensors=True).to('cuda').eval()
    backend=Backend(model,torch,DynamicCache); depth=model.config.num_hidden_layers
    meta=dict(scope='FP32 same-target pilot on reused P1 documents; not independent final test',dtype=str(model.dtype),tf32=torch.backends.cuda.matmul.allow_tf32,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),attention_backend=model.config._attn_implementation,manifest_sha256=file_hash(prev),source_sha256={n:file_hash(Path(__file__).with_name(n)) for n in ('pilot_fp32.py','diagnose.py','diagnose_v2.py','preflight.py')},N=N,repeats=REPS,docs=docs)
    (args.out/'manifest.json').write_text(json.dumps(meta,indent=2))
    def run(d,arm,ref=None,audit=False):
        ids=d['input_ids']; manual=arm=='manual' or arm.startswith('oracle:'); k=int(arm.split(':')[1]) if arm.startswith('oracle:') else 0
        cfg=GenerationConfig(do_sample=False,max_new_tokens=N,eos_token_id=None,pad_token_id=0,use_cache=True)
        if arm.startswith('sd:'):
            _,e,b=arm.split(':'); cfg.assistant_early_exit=int(e); cfg.num_assistant_tokens=int(b); cfg.num_assistant_tokens_schedule='constant'; cfg.assistant_confidence_threshold=0.0
        x=torch.tensor([ids],device=model.device); mask=torch.ones_like(x); depths=[]
        hook=model.register_forward_pre_hook(lambda *_:depths.append(model.config.num_hidden_layers)) if audit else None
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
            with torch.inference_mode():
                z=canonical(ids,N,k,backend,ref) if manual else dict(output_ids=fixed_generate(model,x,mask,cfg)[0,len(ids):].tolist())
            torch.cuda.synchronize(); elapsed=time.perf_counter()-t
            z.update(arm=arm,id=d['id'],split=d['split'],seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),depths=depths)
            if manual: z.update(draft_calls=sum(bool(b['draft']) for b in z['blocks']),target_calls=1+len(z['blocks']))
            z['valid']=len(z['output_ids'])==N and (ref is None or z['output_ids']==ref)
            if manual and ref is not None: z['valid']=z['valid'] and valid_trace(z,len(ids),k,ref)
            return z
        finally:
            if hook: hook.remove()
    first=docs[0]; warm={a:run(first,a,audit=True) for a in ('ar',)+SD}
    eligible=[a for a in SD if int(a.split(':')[1]) in warm[a]['depths'] and depth in warm[a]['depths']]
    (args.out/'warmup.json').write_text(json.dumps(warm))
    references={d['id']:run(d,'ar')['output_ids'] for d in docs}
    (args.out/'references.json').write_text(json.dumps(references))
    if any(len(v)!=N for v in references.values()): raise RuntimeError('reference length mismatch')
    run(first,'manual',references[first['id']])
    for k in KS: run(first,f'oracle:{k}',references[first['id']])
    rows=[]
    with (args.out/'runs.jsonl').open('w') as f:
        def phase(split,arms,reps):
            for i,d in enumerate(docs):
                if d['split']!=split: continue
                for rep in range(reps):
                    order=list(arms); random.Random(1203+i*100+rep).shuffle(order)
                    for arm in order:
                        z=run(d,arm,references[d['id']]); z['rep']=rep; rows.append(z); f.write(json.dumps(z,allow_nan=False)+'\n'); f.flush()
        def stats(split,arms,reps):
            refs={d['id']:references[d['id']] for d in docs if d['split']==split}
            return {a:aggregate([r for r in rows if r['split']==split],a,refs,reps) for a in arms}
        calarms=('ar','manual')+tuple(eligible); phase('calibration',calarms,1); c=stats('calibration',calarms,1)
        if c['ar'] is None: raise RuntimeError('unstable calibration AR')
        valid_sd=[a for a in eligible if c[a] is not None]
        cost=lambda a:sum(v[0] for v in c[a].values())
        best_sd=min(valid_sd,key=cost) if valid_sd else None
        best=min([a for a in c if c[a] is not None],key=cost)
        lock=dict(best_sd=best_sd,best_baseline=best,calibration=c,eligible_sd=eligible)
        (args.out/'lock.json').write_text(json.dumps(lock,indent=2))
        arms=('ar','manual')+((best_sd,) if best_sd else ())+tuple(f'oracle:{k}' for k in KS)
        phase('test',arms,REPS)
    s=stats('test',arms,REPS); summary=dict(lock=lock,valid={a:v is not None for a,v in s.items()},results={})
    if s['ar'] is not None:
        for k in KS:
            a=f'oracle:{k}'; summary['results'][a]={b:compare(s.get(b),s[a],True) for b in dict.fromkeys(['ar','manual',best]+([best_sd] if best_sd else []))}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n'); print(json.dumps(summary,indent=2),flush=True)
    return 0 if s['ar'] is not None else 1


if __name__=='__main__': raise SystemExit(main())
