"""Budget pilot: actual HF LayerSkip greedy latency and teacher-forced anchor quality.
No physical HFL or sparse-kernel speed claim. Run prepare on CPU; run pilot on CUDA.
"""
import argparse, copy, hashlib, json, math, os, platform, random, subprocess, sys, time
from pathlib import Path

def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

def prepare(root, model_id):
    from huggingface_hub import HfApi, snapshot_download
    from datasets import load_dataset
    import torch, transformers
    assert hasattr(transformers.GenerationConfig(), "assistant_early_exit")
    info = HfApi().model_info(model_id)
    snapshot = snapshot_download(model_id, revision=info.sha,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja"])
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    assert cfg.model_type == "llama"
    assert tok("Environment check", return_tensors="pt").input_ids.numel() > 0
    assert list(Path(snapshot).glob("*.safetensors")), "model weights missing"
    # Real public news text; two non-IID logical regions, class-dependent.
    # Same fixed dataset rows are used for all model arms.
    rows = []
    for split, role, per_class in [("train","calibration",4),("test","test",4)]:
        ds = load_dataset("ag_news", split=split)
        count = [0] * 4
        for i, row in enumerate(ds):
            k = int(row["label"])
            if count[k] >= per_class: continue
            if len(tok(row["text"], add_special_tokens=True).input_ids) < 24: continue
            rows.append(dict(id=f"{split}-{i}", split=role, client=k,
                             region=k//2, text=row["text"]))
            count[k] += 1
            if min(count) >= per_class: break
        assert min(count) == per_class
    dump(root/"prompts.json", rows)
    dump(root/"prepared.json", dict(snapshot=str(Path(snapshot).resolve()),
        model_id=model_id, revision=info.sha, torch=torch.__version__,
        transformers=transformers.__version__, n_layers=cfg.num_hidden_layers,
        prompt_sha256=hashlib.sha256((root/"prompts.json").read_bytes()).hexdigest(),
        dataset="ag_news", rows=len(rows), ready=True))
    print("CPU PREPARATION COMPLETE. No GPU test executed.", flush=True)

def pilot(root, tokens, repeats, seed):
    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert torch.cuda.is_available(), "CUDA unavailable; do not use CPU substitute"
    prep = json.loads((root/"prepared.json").read_text())
    raw = (root/"prompts.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prep["prompt_sha256"]
    rows = json.loads(raw)
    assert len({r["id"] for r in rows}) == len(rows)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    snapshot = prep["snapshot"]
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
        torch_dtype=torch.float16, use_safetensors=True).to("cuda").eval()
    n = model.config.num_hidden_layers
    exits = [e for e in [3,4,8] if 0 < e < n]
    assert exits
    modes = [(0,0)] + [(e,k) for e in exits for k in [2,4]]
    meta = dict(prepared=prep, gpu=torch.cuda.get_device_name(),
        device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.version.cuda, seed=seed, tokens=tokens, repeats=repeats,
        git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        scope="batch1 greedy prefill+decode, HF integrated self-speculation")
    run = root/f"run_seed{seed}"
    run.mkdir(exist_ok=False)
    dump(run/"meta.json",meta)
    def sync(): torch.cuda.synchronize()
    @torch.inference_mode()
    def generate(row, mode):
        e,k=mode
        ids=tok(row["text"],return_tensors="pt",truncation=True,max_length=64).to("cuda")
        config=copy.deepcopy(model.generation_config)
        config.do_sample=False
        config.max_new_tokens=tokens
        config.num_beams=1
        config.pad_token_id=tok.eos_token_id
        config.assistant_early_exit=e or None
        if e:
            config.num_assistant_tokens=k
            config.num_assistant_tokens_schedule="constant"
            config.assistant_confidence_threshold=0.0
        sync();torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        result=model.generate(**ids,generation_config=config)
        sync()
        elapsed=time.perf_counter()-t
        out=result[0,ids.input_ids.shape[-1]:].tolist()
        assert out
        return dict(seconds=elapsed, ids=out, tokens=len(out),
                    peak_bytes=torch.cuda.max_memory_allocated())
    for mode in modes: generate(rows[0],mode)
    all_perf=[]
    with (run/"latency.jsonl").open("w") as f:
        for i,row in enumerate(rows):
            for rep in range(repeats):
                order=modes.copy();random.Random(seed+i*100+rep).shuffle(order)
                z={m:generate(row,m) for m in order}
                for e,k in modes:
                    x=dict(id=row["id"],split=row["split"],client=row["client"],region=row["region"],
                           repeat=rep,exit=e,draft=k,greedy_equal=z[e,k]["ids"]==z[0,0]["ids"],**z[e,k])
                    f.write(json.dumps(x)+"\n");all_perf.append(x)
                f.flush()
    # Lock fastest valid global arm on calibration only. Report test separately.
    def stats(mode,split):
        a=[x for x in all_perf if (x["exit"],x["draft"])==mode and x["split"]==split]
        return dict(seconds=sum(x["seconds"] for x in a),tokens=sum(x["tokens"] for x in a),
                    equal=sum(x["greedy_equal"] for x in a),n=len(a))
    valid=[m for m in modes if stats(m,"calibration")["equal"]==stats(m,"calibration")["n"]]
    best=min(valid,key=lambda m:stats(m,"calibration")["seconds"]/stats(m,"calibration")["tokens"])
    base=stats((0,0),"test"); selected=stats(best,"test")
    dump(run/"latency_summary.json",dict(locked_mode=best,base=base,selected=selected,
        speedup_tokens_per_second=(selected["tokens"]/selected["seconds"])/(base["tokens"]/base["seconds"]),
        valid_test=selected["equal"]==selected["n"],
        arms=[dict(mode=m,cal=stats(m,"calibration"),test=stats(m,"test")) for m in modes]))
    # P1: dense teacher features only, no claim that sparse GPU kernels ran.
    # Hook pre-final-RMSNorm hidden to avoid mixing normalized/raw states.
    E=exits[0]
    grids=[0.,.25,.5,.75,1.]
    Bvals=[2,4,8]
    def logits(x):
        return model.lm_head(model.model.norm(x)).float()
    def metrics(p_log, q_log, labels):
        q=q_log.log_softmax(-1)
        return torch.stack([-(q.gather(-1,labels[:,None]).squeeze(-1)).sum(),
            (p_log.exp()*(p_log-q)).sum(),
            torch.minimum(p_log.exp(),q.exp()).sum()])
    def features(row):
        inp=tok(row["text"],return_tensors="pt",truncation=True,max_length=96).input_ids.to("cuda")
        captured=[]
        handle=model.model.norm.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        try:
            y=model.model(input_ids=inp,use_cache=False,output_hidden_states=True,return_dict=True)
        finally: handle.remove()
        assert len(captured)==1
        x=y.hidden_states[E][0,:-1]
        d=captured[0][0,:-1]
        return x,d,inp[0,1:]
    # For each B, offset and gamma, sufficient statistics by client. No training on test.
    cal={}
    test=[]
    with torch.inference_mode():
        for row in [r for r in rows if r["split"]=="calibration"]:
            x,d,labels=features(row)
            ref=logits(d).log_softmax(-1)
            for B in Bvals:
                t=torch.arange(x.shape[0],device=x.device)
                a=(t//B)*B
                residual=d[a]-x[a]
                for j in range(1,B):
                    mask=t%B==j
                    if not mask.any():continue
                    for g in grids:
                        kl=float(metrics(ref[mask],logits(x[mask]+g*residual[mask]),labels[mask])[1])
                        key=(row["client"],B,j,g)
                        cal[key]=cal.get(key,0.)+kl
        def gamma(B,j,clients):
            if j==0:return 1.
            return min(grids,key=lambda g:sum(cal.get((c,B,j,g),0.) for c in clients))
        for row in [r for r in rows if r["split"]=="test"]:
            x,d,labels=features(row); ref=logits(d).log_softmax(-1)
            for B in Bvals:
                t=torch.arange(x.shape[0],device=x.device);a=(t//B)*B
                residual=d[a]-x[a]
                glob=torch.tensor([gamma(B,j,range(4)) for j in range(B)],device=x.device,dtype=x.dtype)[t%B]
                reg=torch.tensor([gamma(B,j,[2*row["region"],2*row["region"]+1]) for j in range(B)],device=x.device,dtype=x.dtype)[t%B]
                states={"teacher":d,"shallow":x,
                    "periodic_sparse":x+(t%B==0).to(x.dtype)[:,None]*residual,
                    "raw_anchor":x+residual,"global_anchor":x+glob[:,None]*residual,
                    "region_anchor":x+reg[:,None]*residual}
                for name,state in states.items():
                    nll,kl,overlap=metrics(ref,logits(state),labels).tolist()
                    test.append(dict(id=row["id"],client=row["client"],region=row["region"],B=B,arm=name,
                                     positions=len(labels),anchors=int((t%B==0).sum()),
                                     nll_sum=nll,kl_sum=kl,teacher_overlap_sum=overlap))
    dump(run/"anchor_rows.json",test)
    summaries=[]
    for B in Bvals:
        for arm in ["teacher","shallow","periodic_sparse","raw_anchor","global_anchor","region_anchor"]:
            a=[r for r in test if r["B"]==B and r["arm"]==arm]
            npos=sum(r["positions"] for r in a)
            summaries.append(dict(B=B,arm=arm,nll=sum(r["nll_sum"] for r in a)/npos,
                kl=sum(r["kl_sum"] for r in a)/npos,teacher_overlap=sum(r["teacher_overlap_sum"] for r in a)/npos,
                query_position_opportunity=1-sum(r["anchors"] for r in a)/npos))
    dump(run/"anchor_summary.json",dict(exit=E,scope="teacher-forced dense feature extraction; no sparse latency, no proposal acceptance or physical HFL claim",rows=summaries))
    print("PILOT COMPLETE",run,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","pilot"])
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--model",default="facebook/layerskip-llama3.2-1B")
    p.add_argument("--tokens",type=int,default=32)
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--seed",type=int,default=11)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    if a.stage=="prepare":prepare(a.root,a.model)
    else:pilot(a.root,a.tokens,a.repeats,a.seed)
if __name__=="__main__":main()
