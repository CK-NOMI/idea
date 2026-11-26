"""R120-P4: real frozen-backbone draft feasibility, not an HFL claim."""
import argparse, copy, json, math, random, time, unittest
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

N, K, SEED = 64, 4, 1204


def split_docs(rows, old):
    out = dict(train=[], calibration=[], test=[])
    for c in range(4):
        a = [r for r in rows if r['client']==c and r['split']=='calibration']
        b = [r for r in rows if r['client']==c and r['split']=='test' and r['id'] not in old]
        if len(a)<8 or len(b)<2: raise ValueError('insufficient docs')
        out['train']+=a[:6]; out['calibration']+=a[6:8]; out['test']+=b[:2]
    allrows=sum(out.values(),[])
    if len({r['id'] for r in allrows})!=len(allrows) or len({r['text'] for r in allrows})!=len(allrows): raise ValueError('overlap')
    return out


def lookup(prefix, k):
    seq=list(prefix); out=[]
    for _ in range(k):
        found=None
        for n in range(min(8,len(seq)-1),0,-1):
            for i in range(len(seq)-n-1,-1,-1):
                if seq[i:i+n]==seq[-n:]: found=seq[i+n]; break
            if found is not None: break
        if found is None: break
        out.append(found); seq.append(found)
    return out


class Head(nn.Module):
    def __init__(self,d,v):
        super().__init__(); self.v=v
        self.net=nn.Sequential(nn.Linear(2*d,128),nn.GELU(),nn.Linear(128,K*v))
    def forward(self,h,e):
        x=torch.cat((F.normalize(h.float(),dim=-1),F.normalize(e.float(),dim=-1)),-1)
        return self.net(x).reshape(-1,K,self.v)


def loss_fn(logits,y,survival):
    ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
    if not survival: return ce
    p=logits.softmax(-1).gather(-1,y[...,None]).squeeze(-1)*(y!=0)
    return ce-0.5*p.cumprod(-1).sum(-1).mean()


def training_pairs(h,ids,emb):
    # h[t] is the hidden state predicting the already known pending ids[t].
    return h[:-K],emb[ids[:-K]],torch.stack([ids[i:len(ids)-K+i] for i in range(1,K+1)],1)


def decode(prompt,backend,proposal,k):
    pred,h,c=backend.forward(prompt,backend.empty(),True); out=[pred[-1]]; state=h[-1]; blocks=[]
    while len(out)<N:
        old=len(prompt)+len(out)-1; assert backend.length(c)==old
        m=min(k,N-len(out)-1); d=proposal(prompt+out,state,m) if m else []
        if len(d)>m: raise ValueError('too many draft tokens')
        pred,h,c=backend.forward([out[-1]]+d,c,False)
        assert len(pred)==len(d)+1 and backend.length(c)==old+1+len(d)
        a=next((i for i,x in enumerate(d) if pred[i]!=x),len(d)); z=d[:a]+[pred[a]]
        state=h[a]; c=backend.crop(c,old+1+a); assert backend.length(c)==old+1+a
        out+=z; blocks.append(dict(draft=d,accepted=a,emitted=z,before=old,after=old+1+a))
    return dict(output_ids=out,blocks=blocks)


class Tests(unittest.TestCase):
    def test_pairs_and_loss(self):
        h=torch.arange(64*3).reshape(64,3).float(); ids=torch.arange(64); emb=torch.randn(64,3)
        a,b,y=training_pairs(h,ids,emb); self.assertEqual(y[0].tolist(),[1,2,3,4]); self.assertEqual(y[-1].tolist(),[60,61,62,63]); self.assertTrue(torch.equal(a[0],h[0])); self.assertTrue(torch.equal(b[0],emb[0]))
        model=Head(3,64); z=model(a,b); self.assertEqual(tuple(z.shape),(60,4,64)); l=loss_fn(z,y,True); l.backward(); self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_lookup(self):
        self.assertEqual(lookup([1,2,1,2],4),[1,2,1,2]); self.assertEqual(lookup([1,2,3],4),[])
    def test_decode(self):
        class Toy:
            def empty(self): return []
            def length(self,c): return len(c)
            def crop(self,c,n): return c[:n]
            def forward(self,ids,c,last):
                c=list(c); p=[]; h=[]
                for x in ids: c.append(x); p.append((sum(c)+len(c))%31); h.append(tuple(c))
                return p,h,c
        p=[1,2,3]; ar=decode(p,Toy(),lambda *a:[],0)['output_ids']
        for k in (1,2,4):
            for wrong in (False,True):
                def prop(prefix,state,m):
                    self.assertEqual(tuple(prefix[:-1]),state)
                    d=ar[len(prefix)-len(p):len(prefix)-len(p)+m]; return [(d[0]+1)%31]+d[1:] if wrong and d else d
                self.assertEqual(decode(p,Toy(),prop,k)['output_ids'],ar)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--self-test',action='store_true'); p.add_argument('--model',type=Path); p.add_argument('--out',type=Path); a=p.parse_args()
    if a.self_test: return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if not(a.model and a.out): p.error('need --model --out')
    a.out.mkdir(parents=True,exist_ok=False)
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig
    from diagnose_v2 import fixed_generate
    from preflight import file_hash
    if transformers.__version__!='4.48.3' or not torch.cuda.is_available(): raise RuntimeError('environment mismatch')
    root=Path(__file__).resolve().parents[2]; data=root/'experiments/team-anchor/R119-P0/results/prompts.json'; prev=root/'experiments/team-anchor/R120-P1/manifest.json'; m=json.loads(prev.read_text())
    for name,h in dict(m['model_files'],**{'model.safetensors':m['weight_sha256']}).items():
        if file_hash(a.model/name)!=h: raise RuntimeError('model file mismatch '+name)
    if list(a.model.glob('*.safetensors'))!=[a.model/'model.safetensors']: raise RuntimeError('unexpected weights')
    split=split_docs(json.loads(data.read_text()),{d['id'] for d in m['docs'] if d['split']=='test'})
    torch.manual_seed(SEED); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager',use_safetensors=True).to('cuda').eval(); model.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True); emb=model.get_input_embeddings().weight.detach(); dev=model.device
    docs={r['id']:tok(r['text'],max_length=64,truncation=True)['input_ids'] for r in sum(split.values(),[])}
    meta=dict(scope='actual proposal pilot, no HFL training or novelty claim',torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),dtype=str(model.dtype),data_sha256=file_hash(data),source_sha256=file_hash(Path(__file__)),split={s:[r['id'] for r in rs] for s,rs in split.items()},prompts=docs)
    (a.out/'manifest.json').write_text(json.dumps(meta,indent=2))
    def generate(ids,sd=False,capture=False):
        cfg=GenerationConfig(do_sample=False,max_new_tokens=N,eos_token_id=None,pad_token_id=0,use_cache=True)
        if sd: cfg.assistant_early_exit=3; cfg.num_assistant_tokens=2; cfg.num_assistant_tokens_schedule='constant'; cfg.assistant_confidence_threshold=0.0
        x=torch.tensor([ids],device=dev); hidden=[]
        hook=model.lm_head.register_forward_pre_hook(lambda _,z:hidden.append(z[0][0,-1].detach().clone())) if capture else None
        try: z=fixed_generate(model,x,torch.ones_like(x),cfg)[0,len(ids):].tolist()
        finally:
            if hook: hook.remove()
        if len(z)!=N or (capture and len(hidden)!=N): raise RuntimeError('reference length or hidden count mismatch')
        return z,torch.stack(hidden) if capture else None
    refs={}; train=[]
    with torch.no_grad():
        for r in split['train']:
            z,h=generate(docs[r['id']],capture=True); refs[r['id']]=z; train.append(training_pairs(h,torch.tensor(z,device=dev),emb))
    counts=Counter(t for z in refs.values() for t in z); vocab=[t for t,_ in sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:2047]]; vmap={t:i+1 for i,t in enumerate(vocab)}
    h,e,y=[torch.cat(x) for x in zip(*train)]; y=torch.tensor([[vmap.get(t,0) for t in row] for row in y.tolist()],device=dev); del train
    heads={}; training={}
    for name,surv in [('ce',False),('survival',True)]:
        torch.manual_seed(SEED); head=Head(h.shape[-1],len(vocab)+1).to(dev); opt=torch.optim.AdamW(head.parameters(),lr=.003)
        history=[]
        for epoch in range(20):
            for ix in torch.randperm(len(h),device=dev).split(64):
                opt.zero_grad(); l=loss_fn(head(h[ix],e[ix]),y[ix],surv); l.backward(); opt.step()
            history.append(float(l.detach()))
        heads[name]=head.eval(); training[name]=dict(last_batch_loss=history,parameters=sum(p.numel() for p in head.parameters()))
    (a.out/'training.json').write_text(json.dumps(dict(train_tokens=len(h),vocab=vocab,coverage=float((y!=0).float().mean()),models=training),indent=2)); del h,e,y
    class Backend:
        def empty(self): return DynamicCache()
        def length(self,c): return int(c.get_seq_length())
        def crop(self,c,n): c.crop(n); return c
        def forward(self,ids,c,last):
            start=self.length(c); x=torch.tensor([ids],device=dev); pos=torch.arange(start,start+len(ids),device=dev); hidden=[]
            hook=model.lm_head.register_forward_pre_hook(lambda _,z:hidden.append(z[0][0]))
            try: r=model(input_ids=x,attention_mask=torch.ones((1,start+len(ids)),device=dev,dtype=torch.long),position_ids=pos[None],cache_position=pos,past_key_values=c,use_cache=True,return_dict=True,num_logits_to_keep=1 if last else 0)
            finally: hook.remove()
            return r.logits[0].argmax(-1).tolist(),hidden[0],r.past_key_values
    backend=Backend()
    def run(r,arm,rep):
        ids=docs[r['id']]; torch.cuda.synchronize(); t=time.perf_counter()
        with torch.no_grad():
            if arm in ('ar','sd'): z=dict(output_ids=generate(ids,arm=='sd')[0])
            else:
                name,k=arm.split(':'); k=int(k)
                def propose(prefix,state,m):
                    if name=='lookup': return lookup(prefix,m)
                    classes=heads[name](state[None],emb[prefix[-1]][None])[0,:m].argmax(-1).tolist(); out=[]
                    for i in classes:
                        if not i: break
                        out.append(vocab[i-1])
                    return out
                z=decode(ids,backend,propose,k)
        torch.cuda.synchronize(); z.update(id=r['id'],arm=arm,rep=rep,seconds=time.perf_counter()-t,valid=z['output_ids']==refs[r['id']]); return z
    arms=('ar','sd','lookup:2','lookup:4','ce:2','ce:4','survival:2','survival:4'); results=[]
    with (a.out/'runs.jsonl').open('w') as f:
        for phase in ('calibration','test'):
            for r in split[phase]:
                with torch.no_grad(): refs[r['id']]=generate(docs[r['id']])[0]
            if phase=='calibration':
                for arm in arms: run(split[phase][0],arm,-1)
                active=arms; reps=1
            else: active=('ar','sd')+tuple(locked.values()); reps=3
            for j,r in enumerate(split[phase]):
                for rep in range(reps):
                    order=list(active); random.Random(SEED+j*100+rep).shuffle(order)
                    for arm in order:
                        z=run(r,arm,rep); z['split']=phase; results.append(z); f.write(json.dumps(z,allow_nan=False)+'\n'); f.flush()
            if phase=='calibration':
                locked={}
                for name in ('lookup','ce','survival'):
                    candidates=[arm for arm in active if arm.startswith(name+':') and all(z['valid'] for z in results if z['arm']==arm)]
                    if candidates: locked[name]=min(candidates,key=lambda arm:sum(z['seconds'] for z in results if z['arm']==arm))
                (a.out/'lock.json').write_text(json.dumps(locked))
    (a.out/'references.json').write_text(json.dumps(refs)); summary={}
    for arm in active:
        rr=[z for z in results if z['split']=='test' and z['arm']==arm]; ar=[z for z in results if z['split']=='test' and z['arm']=='ar']; sd=[z for z in results if z['split']=='test' and z['arm']=='sd']; blocks=[b for z in rr for b in z.get('blocks',[])]; valid=len(rr)==24 and all(z['valid'] and math.isfinite(z['seconds']) and z['seconds']>0 for z in rr)
        summary[arm]=dict(n=len(rr),exact=sum(z['valid'] for z in rr),mean_seconds=sum(z['seconds'] for z in rr)/len(rr),speed_vs_ar=sum(z['seconds'] for z in ar)/sum(z['seconds'] for z in rr) if valid and all(z['valid'] for z in ar) else None,speed_vs_sd=sum(z['seconds'] for z in sd)/sum(z['seconds'] for z in rr) if valid and all(z['valid'] for z in sd) else None,accepted=sum(b['accepted'] for b in blocks),proposed=sum(len(b['draft']) for b in blocks),verification_calls=len(blocks),mean_commit=sum(len(b['emitted']) for b in blocks)/len(blocks) if blocks else None)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)); print(json.dumps(summary,indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
