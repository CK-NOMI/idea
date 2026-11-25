"""P0 smoke benchmark, not HFL or proposed method. Requires CUDA, torch, transformers >=4.57 (verify compatibility), a licensed early-exit checkpoint. Input JSONL: {"id":"public-id","prompt":"public prompt","region":"optional"}; use non-private public inputs. Run with exact HF --revision SHA. No output equality => no speed claim."""
import argparse, copy, hashlib, json, random, time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/layerskip-llama3.2-1B")
    p.add_argument("--revision",required=True)
    p.add_argument("--prompts",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--exits",type=int,nargs="+",default=[2,4,6,8])
    p.add_argument("--tokens",type=int,default=128)
    p.add_argument("--repeats",type=int,default=3)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no silent CPU substitution")
    with open(a.prompts,"rb") as f: source=f.read()
    rows=[json.loads(x) for x in source.splitlines() if x.strip()]
    if not rows or a.tokens<1 or a.repeats<1: raise ValueError("empty prompts or invalid workload")
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.bfloat16).to("cuda").eval()
    n=model.config.num_hidden_layers
    exits=sorted(set(a.exits))
    if any(not 0<x<n for x in exits): raise ValueError("exit must be internal")
    def run(inputs,exit_layer):
        cfg=copy.deepcopy(model.generation_config)
        cfg.do_sample=False
        cfg.max_new_tokens=a.tokens
        cfg.min_new_tokens=0
        cfg.use_cache=True
        if cfg.pad_token_id is None: cfg.pad_token_id=tok.eos_token_id
        kwargs={} if exit_layer==0 else {"assistant_early_exit":exit_layer}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        with torch.inference_mode(): y=model.generate(**inputs,generation_config=cfg,**kwargs)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        ids=y[0,inputs["input_ids"].shape[-1]:].tolist()
        return {"seconds":elapsed,"new_tokens":len(ids),"ids":ids,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
    modes=[0]+exits
    first=tok(rows[0]["prompt"],return_tensors="pt").to("cuda")
    for m in modes: run(first,m)
    meta={"model":a.model,"requested_revision":a.revision,"resolved_revision":getattr(model.config,"_commit_hash",None),"torch":torch.__version__,"transformers":transformers.__version__,"gpu":torch.cuda.get_device_name(),"prompt_sha256":hashlib.sha256(source).hexdigest(),"settings":vars(a),"note":"batch1 greedy prefill+decode; no acceptance trace, training, network, HFL or quality benchmark; exact ID matching required"}
    with open(a.out,"w") as f:
        f.write(json.dumps({"meta":meta})+"\n");f.flush()
        for j,row in enumerate(rows):
            inputs=tok(row["prompt"],return_tensors="pt").to("cuda")
            for rep in range(a.repeats):
                order=modes.copy();random.Random(20261125+j*1000+rep).shuffle(order)
                results={m:run(inputs,m) for m in order}
                for m in modes:
                    z=results[m]
                    f.write(json.dumps({"prompt_id":row.get("id",j),"region":row.get("region"),"repeat":rep,"exit":m,"input_tokens":inputs["input_ids"].shape[-1],"greedy_equal":z["ids"]==results[0]["ids"],**z})+"\n")
                f.flush()
if __name__=="__main__": main()
