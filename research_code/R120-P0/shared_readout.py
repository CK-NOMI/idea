"""R120-P5: full-vocabulary temporal readout, regional ridge controls, no HFL claim."""
import argparse, json, math, random, time, unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from learned_draft import decode, lookup

N, SEED, RANK = 64, 1205, 128
ARMS = ('ar','sd','manual','lookup','direct','global','region','hybrid_global','hybrid_region')


def unit(x):
    return F.normalize(x.float(), dim=-1)


def features(h,e,R):
    x=torch.cat((unit(h),unit(e)),dim=-1)*math.sqrt(h.shape[-1])
    z=x@R
    return torch.cat((z,torch.ones_like(z[...,:1])),dim=-1)


def stats(x,y):
    x=x.detach().cpu().double(); y=y.detach().cpu().double()
    return x.T@x,x.T@y


def merge(parts):
    return tuple(sum(p[i] for p in parts) for i in (0,1))


def fit(s,reg=10.,prior=None):
    g,b=s; eye=torch.eye(g.shape[0],dtype=g.dtype)
    return torch.linalg.solve(g+reg*eye,b if prior is None else b+reg*prior)


def pairs(h,ids,emb,R):
    # h[t] precedes pending ids[t+1]; predict next hidden state h[t+1].
    if len(h)!=len(ids) or len(ids)<3: raise ValueError('invalid sequence')
    return features(h[:-2],emb[ids[1:-1]],R),unit(h[1:-1]),unit(h[:-2])


def splits(rows,previous,older):
    byid={r['id']:r for r in rows}; used=set(previous['test'])|set(older)
    out={s:[byid[i] for i in previous[s]] for s in ('train','calibration')}; out['test']=[]
    for c in range(4):
        chosen=[r for r in rows if r['client']==c and r['split']=='test' and r['id'] not in used][:2]
        if len(chosen)!=2: raise ValueError('insufficient fresh test docs')
        out['test']+=chosen
    allrows=sum(out.values(),[])
    if len({r['id'] for r in allrows})!=40 or len({r['text'] for r in allrows})!=40: raise ValueError('data overlap')
    return out


class Tests(unittest.TestCase):
    def test_merge_and_prior(self):
        torch.manual_seed(1); x=torch.randn(18,5); y=torch.randn(18,7)
        a=stats(x,y); b=merge([stats(x[:5],y[:5]),stats(x[5:],y[5:])])
        self.assertTrue(torch.allclose(a[0],b[0])); self.assertTrue(torch.allclose(fit(a),fit(b)))
        w=fit(a); self.assertTrue(torch.allclose(fit((torch.zeros_like(a[0]),torch.zeros_like(a[1])),100,w),w))
    def test_pairs(self):
        h=torch.randn(8,4); ids=torch.arange(8); e=torch.randn(8,4); R=torch.randn(8,3)
        x,y,b=pairs(h,ids,e,R)
        self.assertEqual(tuple(x.shape),(6,4)); self.assertTrue(torch.allclose(x[0],features(h[0],e[1],R)))
        self.assertTrue(torch.allclose(y[0],unit(h[1]))); self.assertTrue(torch.allclose(b[-1],unit(h[5])))
    def test_projection(self):
        torch.manual_seed(2); h=torch.randn(20,4); e=torch.randn(20,4); R=torch.randn(8,6)
        x=features(h,e,R); y=unit(h); w=fit(stats(x,y),1.)
        pred=x@w.float(); logits=pred@torch.randn(4,15)
        self.assertEqual(tuple(logits.shape),(20,15)); self.assertTrue(torch.isfinite(logits).all())


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
    root=Path(__file__).resolve().parents[2]; exp=root/'experiments/team-anchor'; data=exp/'R119-P0/results/prompts.json'; p1=exp/'R120-P1/manifest.json'; p4=exp/'R120-P4/manifest.json'; m=json.loads(p1.read_text()); old=json.loads(p4.read_text()); refs4=json.loads((p4.parent/'references.json').read_text())
    for name,h in dict(m['model_files'],**{'model.safetensors':m['weight_sha256']}).items():
        if file_hash(a.model/name)!=h: raise RuntimeError('model file mismatch '+name)
    if list(a.model.glob('*.safetensors'))!=[a.model/'model.safetensors'] or file_hash(data)!=old['data_sha256']: raise RuntimeError('unexpected inputs')
    split=splits(json.loads(data.read_text()),old['split'],[d['id'] for d in m['docs'] if d['split']=='test'])
    torch.manual_seed(SEED); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager',use_safetensors=True).to('cuda').eval(); model.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True); emb=model.get_input_embeddings().weight.detach(); dev=model.device; d=emb.shape[1]
    R=(torch.randn(2*d,RANK,generator=torch.Generator().manual_seed(SEED))/math.sqrt(2*d)).to(dev)
    docs={r['id']:tok(r['text'],max_length=64,truncation=True)['input_ids'] for r in sum(split.values(),[])}
    if any(docs[i]!=old['prompts'][i] for i in old['split']['train']+old['split']['calibration']): raise RuntimeError('prompt drift')
    meta=dict(scope='full vocabulary readout feasibility and regional controls; no physical HFL',torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),dtype=str(model.dtype),source_sha256={n:file_hash(Path(__file__).with_name(n)) for n in ('shared_readout.py','learned_draft.py','diagnose_v2.py','preflight.py')},data_sha256=file_hash(data),prior_manifest_sha256=file_hash(p4),prior_references_sha256=file_hash(p4.parent/'references.json'),seed=SEED,rank=RANK,split={s:[dict(id=r['id'],client=r['client'],region=r['region'],input_ids=docs[r['id']]) for r in rs] for s,rs in split.items()})
    (a.out/'manifest.json').write_text(json.dumps(meta,indent=2))
    direct=[]; local={c:[] for c in range(4)}; flat={r:[] for r in (0,1)}; ntrain=0
    with torch.no_grad():
        for r in split['train']:
            ids=torch.tensor(docs[r['id']]+refs4[r['id']],device=dev)
            h=model.model(input_ids=ids[None],use_cache=False,return_dict=True).last_hidden_state[0]
            x,y,b=pairs(h,ids,emb,R); ntrain+=len(x); direct.append(stats(x,y)); z=stats(x,y-b); local[r['client']].append(z); flat[r['region']].append(z)
    clients={c:merge(v) for c,v in local.items()}; region_clients={r:sorted({z['client'] for z in split['train'] if z['region']==r}) for r in flat}; regions={r:merge([clients[c] for c in cs]) for r,cs in region_clients.items()}
    glob=fit(merge(list(regions.values()))); rd=fit(merge(direct)); rw={r:fit(s,100.,glob) for r,s in regions.items()}
    flat_delta={r:float((rw[r]-fit(merge(flat[r]),100.,glob)).abs().max()) for r in flat}
    if max(flat_delta.values())>1e-8: raise RuntimeError('flat-equivalence drift')
    (a.out/'training.json').write_text(json.dumps(dict(windows=ntrain,rank=RANK,regularization=10,regional_prior_regularization=100,region_clients=region_clients,flat_equivalence_max_abs=flat_delta),indent=2))
    W={'direct':rd.float().to(dev),'global':glob.float().to(dev),**{f'region:{r}':w.float().to(dev) for r,w in rw.items()}}
    class Backend:
        def empty(self): return DynamicCache()
        def length(self,c): return int(c.get_seq_length())
        def crop(self,c,n): c.crop(n); return c
        def forward(self,ids,c,last):
            start=self.length(c); x=torch.tensor([ids],device=dev); pos=torch.arange(start,start+len(ids),device=dev)
            z=model.model(input_ids=x,attention_mask=torch.ones((1,start+len(ids)),device=dev,dtype=torch.long),position_ids=pos[None],cache_position=pos,past_key_values=c,use_cache=True,return_dict=True)
            h=z.last_hidden_state[0]; h=h[-1:] if last else h
            return model.lm_head(h).argmax(-1).tolist(),h,z.past_key_values
    backend=Backend()
    def generate(ids,sd=False):
        cfg=GenerationConfig(do_sample=False,max_new_tokens=N,eos_token_id=None,pad_token_id=0,use_cache=True)
        if sd: cfg.assistant_early_exit=3; cfg.num_assistant_tokens=2; cfg.num_assistant_tokens_schedule='constant'; cfg.assistant_confidence_threshold=0.
        x=torch.tensor([ids],device=dev); return fixed_generate(model,x,torch.ones_like(x),cfg)[0,len(ids):].tolist()
    refs={}; rows=[]
    def run(r,arm,rep):
        ids=docs[r['id']]; torch.cuda.synchronize(); t=time.perf_counter()
        with torch.no_grad():
            if arm in ('ar','sd'): z=dict(output_ids=generate(ids,arm=='sd'))
            else:
                def propose(prefix,state,k):
                    if arm=='manual': return []
                    if arm=='lookup' or arm.startswith('hybrid'):
                        q=lookup(prefix,k)
                        if q or arm=='lookup': return q
                    w=W[f"region:{r['region']}"] if arm in ('region','hybrid_region') else W['direct' if arm=='direct' else 'global']
                    h=features(state,emb[prefix[-1]],R)@w
                    if arm!='direct': h=h+unit(state)
                    return [int(model.lm_head(h).argmax())]
                z=decode(ids,backend,propose,0 if arm=='manual' else 4 if arm=='lookup' or arm.startswith('hybrid') else 1)
        torch.cuda.synchronize(); z.update(id=r['id'],arm=arm,rep=rep,seconds=time.perf_counter()-t,valid=len(z['output_ids'])==N and z['output_ids']==refs[r['id']]); return z
    with (a.out/'runs.jsonl').open('w') as f:
        for phase in ('calibration','test'):
            with torch.no_grad():
                for r in split[phase]: refs[r['id']]=generate(docs[r['id']]); assert len(refs[r['id']])==N
            (a.out/'references.json').write_text(json.dumps(refs))
            if phase=='calibration':
                for arm in ARMS: run(split[phase][0],arm,-1)
            for j,r in enumerate(split[phase]):
                for rep in range(1 if phase=='calibration' else 3):
                    order=list(ARMS); random.Random(SEED+j*100+rep).shuffle(order)
                    for arm in order:
                        z=run(r,arm,rep); z['split']=phase; rows.append(z); f.write(json.dumps(z,allow_nan=False)+'\n'); f.flush()
    summary={}; ar=[r for r in rows if r['split']=='test' and r['arm']=='ar']
    for arm in ARMS:
        rr=[r for r in rows if r['split']=='test' and r['arm']==arm]; bs=[b for r in rr for b in r.get('blocks',[])]; valid=len(rr)==24 and all(r['valid'] for r in rr)
        summary[arm]=dict(n=len(rr),exact=sum(r['valid'] for r in rr),mean_seconds=sum(r['seconds'] for r in rr)/len(rr),speed_vs_ar=sum(r['seconds'] for r in ar)/sum(r['seconds'] for r in rr) if valid and all(r['valid'] for r in ar) else None,accepted=sum(b['accepted'] for b in bs),proposed=sum(len(b['draft']) for b in bs),rounds=len(bs))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n'); print(json.dumps(summary,indent=2)); return 0 if all(r['valid'] for r in ar) else 1


if __name__=='__main__': raise SystemExit(main())
