"""R120-P2: targeted numerical diagnostic, not a speed benchmark."""
import argparse
import copy
import json
from pathlib import Path
import unittest
from preflight import Toy, verify, file_hash

EVENTS = {'train-448': [(45, 0, 482)], 'test-27': [(19, 0, 13452), (19, 4, 690), (42, 16, 85161)], 'test-32': [(41, 2, 1682), (41, 4, 1682), (41, 8, 1682)]}
KS = (0, 1, 2, 4, 8, 16)
N = 64


def chunks(p, n, j, k, style):
    """Half-open input intervals on prompt+reference, ending at a chunk predicting j."""
    target = p+j-1
    if not (p >= 2 and 0 <= j < n and k >= 0 and style in ('full', 'split', 'canonical')):
        raise ValueError('invalid probe')
    if style == 'full':
        return [(0, target+1)]
    end = p-1 if style == 'split' else p
    out = [(0, end)]
    while end <= target:
        new = min(end+k+1, p+n-1)
        out.append((end, new))
        end = new
    return out


def canonical(prompt, n, k, backend, tape=None):
    """Full prompt prefill supplies first output; later cache excludes pending."""
    if not prompt or n < 1 or k < 0 or (k and (tape is None or len(tape) != n)):
        raise ValueError('invalid inputs')
    first, cache = backend.forward(prompt, backend.prefill([]))
    out = [first[-1]]
    pending = out[-1]
    blocks = []
    while len(out) < n:
        old = len(prompt)+len(out)-1
        assert backend.length(cache) == old
        m = min(k, n-len(out)-1)
        draft = list(tape[len(out):len(out)+m]) if m else []
        pred, cache = backend.forward([pending]+draft, cache)
        assert len(pred) == m+1 and backend.length(cache) == old+m+1
        a = next((i for i in range(m) if pred[i] != draft[i]), m)
        emitted = draft[:a]+[pred[a]]
        cache = backend.crop(cache, old+1+a)
        assert backend.length(cache) == old+1+a
        blocks.append(dict(before=old, after=old+1+a, accepted=a, draft=draft, emitted=emitted))
        out.extend(emitted)
        pending = out[-1]
    return dict(output_ids=out, blocks=blocks)


class Tests(unittest.TestCase):
    def test_chunks(self):
        for p in (2, 37, 64):
            for n in (1, 16, 64):
                for j in range(n):
                    for k in KS:
                        for style in ('full','split','canonical'):
                            c = chunks(p,n,j,k,style)
                            self.assertEqual(c[0][0],0)
                            self.assertTrue(all(a < b for a,b in c))
                            self.assertTrue(all(x[1] == y[0] for x,y in zip(c,c[1:])))
                            self.assertTrue(c[-1][0] <= p+j-1 < c[-1][1] <= p+n-1)
    def test_canonical(self):
        for p in ([2,3],[4,7,1]):
            for n in (1,2,7,16):
                ref=verify(p,n,0,Toy())['output_ids']
                for k in KS:
                    for tape in [ref]+[ref[:j]+[(ref[j]+1)%31]+ref[j+1:] for j in range(n)]:
                        self.assertEqual(canonical(p,n,k,Toy(),tape)['output_ids'],ref)


class Backend:
    def __init__(self, model, torch, cache_cls):
        self.model, self.t, self.cls = model, torch, cache_cls
    def prefill(self,prefix):
        assert not prefix
        return self.cls()
    def length(self,c):
        return int(c.get_seq_length())
    def logits(self,ids,c,last=False):
        t=self.t; start=self.length(c)
        inp=t.tensor([ids],device=self.model.device,dtype=t.long)
        pos=t.arange(start,start+len(ids),device=inp.device)
        mask=t.ones((1,start+len(ids)),device=inp.device,dtype=t.long)
        kw={'num_logits_to_keep':1} if last else {}
        r=self.model(input_ids=inp,attention_mask=mask,position_ids=pos[None],cache_position=pos,
                     past_key_values=c,use_cache=True,return_dict=True,**kw)
        return r.logits[0],r.past_key_values
    def forward(self,ids,c):
        # Full-prompt prefill only needs final row, but canonical() expects all
        # rows for subsequent blocks; full logits here deliberately stay explicit.
        v,c=self.logits(ids,c)
        return v.argmax(-1).tolist(),c
    def crop(self,c,n):
        c.crop(n)
        return c


def first_diff(a,b):
    return next((i+1 for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b))+1)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--model',type=Path); p.add_argument('--out',type=Path); args=p.parse_args()
    if args.self_test:
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if not(args.model and args.out): p.error('need --model --out')
    args.out.mkdir(parents=True,exist_ok=False)
    import torch, transformers
    from transformers import AutoModelForCausalLM, DynamicCache, GenerationConfig
    if transformers.__version__!='4.48.3' or not torch.cuda.is_available(): raise RuntimeError('environment mismatch')
    previous=Path(__file__).resolve().parents[2]/'experiments/team-anchor/R120-P1'
    manifest=json.loads((previous/'manifest.json').read_text()); refs=json.loads((previous/'references.json').read_text())
    for name,h in dict(manifest['model_files'],**{'model.safetensors':manifest['weight_sha256']}).items():
        if file_hash(args.model/name)!=h: raise RuntimeError('model file mismatch '+name)
    prompts={r['id']:r['input_ids'] for r in manifest['docs'] if r['id'] in EVENTS}
    if set(prompts)!=set(EVENTS) or any(len(refs[i])!=N for i in EVENTS): raise RuntimeError('unexpected data')
    report=dict(scope='three known failure documents; diagnostic only; FP32 is separate numerical target',source_sha256=file_hash(Path(__file__)),manifest_sha256=file_hash(previous/'manifest.json'),references_sha256=file_hash(previous/'references.json'),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),runs=[])
    def save(): (args.out/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        for dtype in (torch.float16,torch.float32):
            if dtype==torch.float32:
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
            model=AutoModelForCausalLM.from_pretrained(str(args.model),local_files_only=True,torch_dtype=dtype,attn_implementation='eager',use_safetensors=True).to('cuda').eval()
            backend=Backend(model,torch,DynamicCache)
            with torch.inference_mode():
                for rid,prompt in prompts.items():
                    tape=refs[rid]; x=torch.tensor([prompt],device=model.device)
                    cfg=GenerationConfig(do_sample=False,max_new_tokens=N,eos_token_id=None,pad_token_id=0,use_cache=True)
                    g=model.generate(x,attention_mask=torch.ones_like(x),generation_config=cfg,return_dict_in_generate=True,output_scores=True)
                    ar=g.sequences[0,len(prompt):].tolist()
                    if len(ar)!=N: raise RuntimeError('wrong AR length')
                    def scores(v,ids):
                        v=v.float(); val,ix=v.topk(5); return dict(top_ids=ix.tolist(),top_logits=val.tolist(),margin=float(val[0]-val[1]),tracked={str(i):float(v[i]) for i in ids})
                    row=dict(dtype=str(dtype),id=rid,ar_ids=ar,first_diff_from_p1=first_diff(ar,tape),events=[],canonical=[])
                    report['runs'].append(row)
                    for j,k,other in EVENTS[rid]:
                        wanted=[tape[j],other]; event=dict(position=j+1,k=k,expected=tape[j],other=other,probes=[])
                        if ar[:j]==tape[:j]: event['generate']=scores(g.scores[j][0],wanted)
                        stream=prompt+tape
                        for style,width,last in [('split',k,False),('canonical',k,False),('full',0,False),('full',0,True)]:
                            c=DynamicCache(); plan=chunks(len(prompt),N,j,width,style)
                            for a,b in plan: v,c=backend.logits(stream[a:b],c,last=last)
                            index=0 if last else len(prompt)+j-1-plan[-1][0]
                            event['probes'].append(dict(style=style,last_only=last,intervals=plan,**scores(v[index],wanted)))
                            del c,v
                        row['events'].append(event)
                    del g
                    for k in KS:
                        z=canonical(prompt,N,k,backend,ar)
                        z.update(k=k,first_diff=first_diff(z['output_ids'],ar),rejected=sum(len(b['draft'])-b['accepted']>0 for b in z['blocks']))
                        row['canonical'].append(z)
                    # A limited SD check, no tuning or performance claims.
                    row['sd']=[]
                    for e in (3,4):
                        c=copy.deepcopy(cfg); c.assistant_early_exit=e; c.num_assistant_tokens=2; c.num_assistant_tokens_schedule='constant'; c.assistant_confidence_threshold=0.0
                        orig=model.generation_config
                        model.generation_config=copy.deepcopy(c); model.generation_config.assistant_early_exit=None
                        try: s=model.generate(x,attention_mask=torch.ones_like(x),generation_config=c)[0,len(prompt):].tolist()
                        finally: model.generation_config=orig
                        row['sd'].append(dict(exit=e,k=2,output_ids=s,first_diff=first_diff(s,ar)))
                    save()
            del backend,model
            torch.cuda.empty_cache()
        report['completed']=True
    except Exception as e:
        report.update(completed=False,error=repr(e)); raise
    finally: save()
    return 0


if __name__=='__main__': raise SystemExit(main())
