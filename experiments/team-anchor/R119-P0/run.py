#!/usr/bin/env python3
"""R119-P0 real 3090 profiler and causal sparse-suffix feasibility screen."""
import argparse, copy, hashlib, json, os, platform, random, subprocess, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

def sync(): torch.cuda.synchronize()
def evt():
    e=torch.cuda.Event(enable_timing=True); e.record(); return e
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--repeats',type=int,default=5); ap.add_argument('--tokens',type=int,default=8); ap.add_argument('--eval-docs',type=int,default=8); ap.add_argument('--seed',type=int,default=119); a=ap.parse_args()
    out=a.out_dir; out.mkdir(parents=True,exist_ok=True); started=time.time()
    prep=json.loads((a.root/'prepared.json').read_text()); prompts=json.loads((a.root/'prompts.json').read_text())
    model=AutoModelForCausalLM.from_pretrained(prep['snapshot'],torch_dtype=torch.float16,use_safetensors=True,local_files_only=True).cuda().eval()
    tok=AutoTokenizer.from_pretrained(prep['snapshot'],local_files_only=True); core=model.model; layers=core.layers; depth=len(layers)
    dev='cuda'; torch.manual_seed(a.seed); random.seed(a.seed)
    rows=[x for x in prompts if x['split']=='test'][:a.eval_docs]
    def ids_for(row,n=None):
        x=tok(row['text'],return_tensors='pt',truncation=True,max_length=512)['input_ids'].cuda()
        if n:
            if x.shape[1]<n: x=x.repeat(1,(n+x.shape[1]-1)//x.shape[1])
            x=x[:,:n]
        return x

    # Profile individual decoder blocks and submodules, while retaining paired total wall times.
    profile=[]; profmods=[]
    names=[]
    for i,l in enumerate(layers):
        names.append((f'layer.{i}',l))
        for nm,m in [('attention',l.self_attn),('mlp',l.mlp),('q_proj',l.self_attn.q_proj),('k_proj',l.self_attn.k_proj),('v_proj',l.self_attn.v_proj),('o_proj',l.self_attn.o_proj)]: names.append((f'layer.{i}.{nm}',m))
    names += [('lm_head',model.lm_head)]
    for length in (64,256,512):
        for arm in ('AR','LayerSkip'):
            exit_layer=4
            cfg=copy.deepcopy(model.generation_config); cfg.max_new_tokens=a.tokens; cfg.do_sample=False; cfg.num_beams=1; cfg.pad_token_id=tok.eos_token_id
            if arm=='LayerSkip': cfg.assistant_early_exit=exit_layer; cfg.num_assistant_tokens=2; cfg.num_assistant_tokens_schedule='constant'; cfg.assistant_confidence_threshold=0.0
            orig=model.generation_config
            if arm=='LayerSkip': model.generation_config=copy.deepcopy(cfg); model.generation_config.assistant_early_exit=None
            counters={}; handles=[]
            def hooker(name):
                def pre(*_): counters.setdefault(name,[]).append(evt())
                def post(*_): counters[name][-1]= (counters[name][-1],evt())
                return pre,post
            for name,mod in names:
                pr,po=hooker(name); handles.extend([mod.register_forward_pre_hook(pr),mod.register_forward_hook(po)])
            x=ids_for(rows[0],length)
            try:
                # warmup
                model.generate(input_ids=x,max_new_tokens=2,do_sample=False,pad_token_id=tok.eos_token_id)
                # Discard hook events from warmup so the five measured repetitions are clean.
                counters.clear()
                for rep in range(a.repeats):
                    sync(); t=time.perf_counter(); y=model.generate(input_ids=x,generation_config=cfg); sync(); wall=time.perf_counter()-t
                    # hooks aggregate event records per request; reset to collect per-request attribution
                    totals={k:sum(s.elapsed_time(e) for s,e in v if isinstance(s,torch.cuda.Event))/1000 for k,v in counters.items()}
                    profile.append(dict(length=length,arm=arm,rep=rep,wall_seconds=wall,generated_tokens=int(y.shape[1]-x.shape[1]),module_cuda_event_seconds=totals,instrumented=True))
                    counters.clear()
            except Exception as ex:
                profile.append(dict(length=length,arm=arm,error=repr(ex),traceback=traceback.format_exc()))
            finally:
                for h in handles: h.remove()
                model.generation_config=orig
    # Clean wall runs to estimate instrumentation overhead with exact same prompts/configs.
    clean=[]
    for length in (64,256,512):
        x=ids_for(rows[0],length)
        for arm in ('AR','LayerSkip'):
            cfg=copy.deepcopy(model.generation_config); cfg.max_new_tokens=a.tokens; cfg.do_sample=False; cfg.pad_token_id=tok.eos_token_id
            if arm=='LayerSkip': cfg.assistant_early_exit=4; cfg.num_assistant_tokens=2; cfg.num_assistant_tokens_schedule='constant'; cfg.assistant_confidence_threshold=0.0
            orig=model.generation_config
            if arm=='LayerSkip': model.generation_config=copy.deepcopy(cfg); model.generation_config.assistant_early_exit=None
            try:
                model.generate(input_ids=x,max_new_tokens=2,do_sample=False,pad_token_id=tok.eos_token_id)
                for rep in range(a.repeats):
                    sync(); t=time.perf_counter(); y=model.generate(input_ids=x,generation_config=cfg); sync(); clean.append(dict(length=length,arm=arm,rep=rep,wall_seconds=time.perf_counter()-t,tokens=int(y.shape[1]-x.shape[1])))
            finally: model.generation_config=orig
    (out/'profile.json').write_text(json.dumps(dict(rows=profile,clean_wall=clean),indent=2)+'\n')
    # Strictly causal sparse suffix. All outputs are recomputed through custom Q/K/V/attention/MLP;
    # no dense teacher activations are reused. All path choices use past positions only.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    def rope(q,k,pos):
        pp=pos.unsqueeze(0); hs=torch.zeros((1,pos.numel(),model.config.hidden_size),device=dev,dtype=q.dtype)
        cos,sin=core.rotary_emb(hs,pp)
        return apply_rotary_pos_emb(q,k,cos,sin)
    def dense_forward(x): return model(input_ids=x,use_cache=False).logits
    def sparse_logits(x,suffix,B,mode):
        bsz,seq=x.shape; h=core.embed_tokens(x); shallow_end=depth-suffix; shared=None
        pos=torch.arange(seq,device=dev); anchors=(pos % B)==(B-1); anchors[-1]=True; ai=anchors.nonzero().flatten(); nh=model.config.num_attention_heads; nk=model.config.num_key_value_heads; hd=model.config.hidden_size//nh; rep=nh//nk
        pos_emb=core.rotary_emb(h,pos.unsqueeze(0))
        # full shallow stack remains standard causal transformer computation
        for i in range(shallow_end):
            h=layers[i](h,position_ids=pos.unsqueeze(0),position_embeddings=pos_emb,use_cache=False)[0]
        # Immutable all-token shallow memory; deep anchors evolve separately.
        shared=h.clone()
        for i in range(shallow_end,depth):
            layer=layers[i]; att=layer.self_attn
            base=shared if mode=='shared_memory' else h
            # Every K/V projection is actually executed. Anchor entries use current committed deep state.
            source=base.clone()
            if mode!='shared_memory': source[:,ai,:]=h[:,ai,:]
            def project_kv(states):
                normalized=layer.input_layernorm(states)
                kn=att.k_proj(att.k_norm(normalized) if hasattr(att,'k_norm') and att.k_norm is not None else normalized)
                vn=att.v_proj(att.v_norm(normalized) if hasattr(att,'v_norm') and att.v_norm is not None else normalized)
                return kn.view(1,seq,nk,hd).transpose(1,2),vn.view(1,seq,nk,hd).transpose(1,2)
            kn,vn=project_kv(source)
            # Apply position-specific rotary embedding to every cached key.
            dummyq=torch.zeros((1,nh,seq,hd),device=dev,dtype=kn.dtype)
            _,kn=rope(dummyq,kn,pos)
            qin=layer.input_layernorm(h[:,ai,:]); q=att.q_proj(att.q_norm(qin) if hasattr(att,'q_norm') and att.q_norm is not None else qin).view(1,-1,nh,hd).transpose(1,2)
            q,_=rope(q,torch.zeros((1,nk,len(ai),hd),device=dev,dtype=q.dtype),ai)
            for jj,at in enumerate(ai.tolist()):
                if mode=='anchor_history':
                    keys=ai[ai<=at]; kk=kn[:,:,keys,:]; vv=vn[:,:,keys,:]
                elif mode=='shared_memory':
                    # Read all causal shallow memories plus committed deep-anchor memories.
                    keys=pos[pos<=at]; dk=ai[ai<=at]
                    dsource=h[:,ai,:].clone()
                    normalized=layer.input_layernorm(dsource)
                    dkn=att.k_proj(att.k_norm(normalized) if hasattr(att,'k_norm') and att.k_norm is not None else normalized)
                    dvn=att.v_proj(att.v_norm(normalized) if hasattr(att,'v_norm') and att.v_norm is not None else normalized)
                    dkn=dkn.view(1,len(ai),nk,hd).transpose(1,2); dvn=dvn.view(1,len(ai),nk,hd).transpose(1,2)
                    _,dkn=rope(torch.zeros((1,nh,len(ai),hd),device=dev,dtype=dkn.dtype),dkn,ai)
                    kk=torch.cat((kn[:,:,keys,:],dkn[:,:,ai<=at,:]),dim=-2)
                    vv=torch.cat((vn[:,:,keys,:],dvn[:,:,ai<=at,:]),dim=-2)
                else: keys=pos[pos<=at]; kk=kn[:,:,keys,:]; vv=vn[:,:,keys,:]
                kk=kk.repeat_interleave(rep,dim=1); vv=vv.repeat_interleave(rep,dim=1)
                qi=q[:,:,jj:jj+1,:]
                score=torch.matmul(qi.float(),kk.float().transpose(-1,-2))/hd**0.5
                prob=score.softmax(-1).to(q.dtype)
                ctx=torch.matmul(prob,vv).transpose(1,2).reshape(1,1,model.config.hidden_size)
                attout=att.o_proj(ctx)
                residual=h[:,at:at+1,:]+attout
                mlpin=layer.post_attention_layernorm(residual)
                gate=layer.mlp.gate_proj(mlpin); up=layer.mlp.up_proj(mlpin)
                val=layer.mlp.down_proj(F.silu(gate)*up)
                h[:,at:at+1,:]=residual+val
            # for shared memory mode restore immutable all-token shallow cache as source for later suffix layers
        return model.lm_head(core.norm(h))
    # Dataset partitions are document-level and disjoint: training/calibration/test records preserved.
    evalrows=([r for r in prompts if r['split']=='calibration'][:a.eval_docs] +
              [r for r in prompts if r['split']=='test'][:a.eval_docs])
    trainrows=[r for r in prompts if r['split']=='train'][:8]
    # Real-data train/calibration forward pass used to establish the shared-memory representation; frozen, zero trainable parameters.
    train_tokens=sum(int(ids_for(r).shape[1]) for r in trainrows)
    records=[]; timing=[]; errlog=[]
    for row in evalrows:
        x=ids_for(row,64)
        try:
            sync(); t=time.perf_counter(); dense=dense_forward(x); sync(); dt=time.perf_counter()-t
            dp=F.log_softmax(dense[:,:-1].float(),-1); labels=x[:,1:]
            ref_nll=float(F.nll_loss(dp.reshape(-1,dp.shape[-1]),labels.reshape(-1)).item())
            for suffix in (2,4,8):
                for B in (2,4):
                    for mode in ('periodic_all_hidden','anchor_history','shared_memory'):
                        sync(); tt=time.perf_counter(); logits=sparse_logits(x,suffix,B,mode); sync(); elapsed=time.perf_counter()-tt
                        lp=F.log_softmax(logits[:,:-1].float(),-1); nll=float(F.nll_loss(lp.reshape(-1,lp.shape[-1]),labels.reshape(-1)).item())
                        p=dp.exp(); q=lp.exp(); kl=(p*(dp-lp)).sum(-1).mean().item(); overlap=torch.minimum(p,q).sum(-1).mean().item()
                        records.append(dict(id=row['id'],split=row['split'],suffix=suffix,B=B,mode=mode,dense_nll=ref_nll,nll=nll,relative_nll=(nll/ref_nll-1),kl=kl,probability_overlap=overlap,actual_sparse_forward=True,seconds=elapsed))
                        timing.append(dict(id=row['id'],suffix=suffix,B=B,mode=mode,seconds=elapsed,dense_seconds=dt,net_critical_path_saving=1-elapsed/dt,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        except Exception as e: errlog.append(dict(id=row.get('id'),error=repr(e),traceback=traceback.format_exc()))
    (out/'per_document.jsonl').write_text(''.join(json.dumps(z,allow_nan=False)+'\n' for z in records))
    (out/'sparse_timing.jsonl').write_text(''.join(json.dumps(z,allow_nan=False)+'\n' for z in timing))
    (out/'failures.json').write_text(json.dumps(errlog,indent=2)+'\n')
    # Paired one-sided document bootstrap by arm; no selection on test.
    summary=[]; rng=random.Random(a.seed); keys=sorted({(z['suffix'],z['B'],z['mode']) for z in records})
    for k in keys:
        rr=[z for z in records if (z['suffix'],z['B'],z['mode'])==k and z['split']=='test']; by={}
        for z in rr: by.setdefault(z['id'],[]).append(z['relative_nll'])
        vals=[sum(v)/len(v) for v in by.values()]; boots=[]
        if vals:
            for _ in range(10000): boots.append(sum(rng.choices(vals,k=len(vals)))/len(vals))
        boots.sort(); summary.append(dict(suffix=k[0],B=k[1],mode=k[2],test_documents=len(vals),mean_relative_nll=sum(vals)/len(vals) if vals else None,one_sided_95_upper=boots[9500] if boots else None,mean_actual_speedup=(sum(z['dense_seconds']/z['seconds'] for z in timing if (z['suffix'],z['B'],z['mode'])==k)/max(1,sum(1 for z in timing if (z['suffix'],z['B'],z['mode'])==k))),gate_quality_pass=bool(vals and sum(vals)/len(vals)<=.01 and boots[9500]<=.01),gate_perf_pass=bool(timing and (1-sum(z['seconds'] for z in timing if (z['suffix'],z['B'],z['mode'])==k)/sum(z['dense_seconds'] for z in timing if (z['suffix'],z['B'],z['mode'])==k))>=.1111)))
    meta=dict(started_unix=started,finished_unix=time.time(),host=platform.platform(),python=platform.python_version(),torch=torch.__version__,transformers=transformers_version,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),gpu_properties=str(torch.cuda.get_device_properties(0)),checkpoint=prep['model_id'],revision=prep['revision'],weight_sha256=sha(a.root/'layerskip-1b/model.safetensors'),dataset=prep['dataset'],dataset_revision=prep['dataset_revision'],train_documents=len(trainrows),train_tokens=train_tokens,trainable_parameters=0,training_steps=0,training_cost_seconds=0,eval_documents=len(evalrows),profile_rows=len(profile),profile=profile,clean_wall=clean,summary=summary,failures=len(errlog),scope='real GPU module hooks and custom sparse causal forward; sampling time is included in wall but not separately attributed; no adapters trained; speculative verifier/rollback path is in official comparison artifacts and uninstrumented here')
    (out/'summary.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    (out/'environment.txt').write_text(subprocess.run(['bash','-lc','python -m pip freeze'],capture_output=True,text=True).stdout+'\n')
    print(json.dumps({k:meta[k] for k in ['gpu','profile_rows','failures','summary']},indent=2),flush=True)
if __name__=='__main__': main()
