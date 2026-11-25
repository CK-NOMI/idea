"""R104 P0 official LayerSkip paired benchmark; no HFL or proposed model claim.
Requires official repository checkout 494752e5fbb0a82989f6cb384841684b1c2ef5c3
and its pinned environment. CUDA and permitted local model snapshot required.
Use public JSONL {id,prompt,region,split}, split=calibration or test.
Do NOT choose configurations from test. This runner measures all arms offline;
the downstream evaluator must lock actions using calibration only.
"""
import argparse, hashlib, json, random, subprocess, sys, time
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream",required=True)
    p.add_argument("--model-snapshot",required=True,help="local immutable HF snapshot path")
    p.add_argument("--prompts",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--exits",type=int,nargs="+",default=[4,8])
    p.add_argument("--drafts",type=int,nargs="+",default=[2,4,6])
    p.add_argument("--tokens",type=int,default=128)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--dtype",choices=["float16","bfloat16"],default="float16")
    a=p.parse_args()
    expected="494752e5fbb0a82989f6cb384841684b1c2ef5c3"
    actual=subprocess.check_output(["git","-C",a.upstream,"rev-parse","HEAD"],text=True).strip()
    if actual!=expected or subprocess.check_output(["git","-C",a.upstream,"status","--porcelain","--untracked-files=no"],text=True).strip(): raise RuntimeError("upstream revision/working tree mismatch")
    sys.path.insert(0,str(Path(a.upstream).resolve()))
    import torch, transformers
    from self_speculation.generator_base import GenerationConfig,HuggingfaceLlamaGenerator
    from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
    from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no silent CPU substitution")
    if not Path(a.model_snapshot).is_dir(): raise ValueError("local snapshot required")
    if a.tokens<1 or a.repeats<1 or min(a.drafts)<1: raise ValueError("invalid token/repeat/draft count")
    raw=Path(a.prompts).read_bytes()
    rows=[json.loads(x) for x in raw.splitlines() if x.strip()]
    if not rows or len({r["id"] for r in rows})!=len(rows): raise ValueError("unique nonempty prompt IDs required")
    if {r["split"] for r in rows}!={"calibration","test"}: raise ValueError("both splits required")
    if any(not r["prompt"] or not r["region"] for r in rows): raise ValueError("empty prompt/region")
    tok=transformers.AutoTokenizer.from_pretrained(a.model_snapshot,local_files_only=True)
    model=transformers.AutoModelForCausalLM.from_pretrained(a.model_snapshot,local_files_only=True,use_safetensors=True,torch_dtype=getattr(torch,a.dtype)).to("cuda").eval()
    n=model.config.num_hidden_layers
    if any(not 0<e<n for e in a.exits): raise ValueError("exit outside model")
    modes=[(0,0)]+[(e,k) for e in sorted(set(a.exits)) for k in sorted(set(a.drafts))]
    engines={False:HuggingfaceLlamaGenerator(tok,model,AutoRegressiveGenerationStrategy()),True:HuggingfaceLlamaGenerator(tok,model,SelfSpeculativeGenerationStrategy())}
    def run(prompt,mode):
        e,k=mode
        cfg=GenerationConfig(max_steps=a.tokens,exit_layer=e if e else -1,num_speculations=k if e else -1,generation_strategy="self_speculative" if e else "autoregressive",sample=False,temperature=1.0,top_k=0,top_p=1.0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        result=engines[bool(e)].generate(prompt,cfg)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        ids=result.generation_strategy_result.predicted_tokens
        if not ids or len(ids)>a.tokens: raise RuntimeError("invalid generated length")
        return dict(seconds=elapsed,new_tokens=len(ids),ids=ids,acceptance=result.generation_strategy_result.acceptance_rate,peak_bytes=torch.cuda.max_memory_allocated())
    for m in modes: run(rows[0]["prompt"],m)
    meta=dict(upstream=actual,model_snapshot=str(Path(a.model_snapshot).resolve()),model_revision=getattr(model.config,"_commit_hash",None),prompt_sha256=hashlib.sha256(raw).hexdigest(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,args=vars(a),scope="batch1 greedy prefill+decode; not full HFL or sampled-kernel proof")
    with open(a.out,"w") as f:
        f.write(json.dumps({"meta":meta})+"\n")
        for j,row in enumerate(rows):
            for rep in range(a.repeats):
                order=modes.copy();random.Random(20261125+j*1000+rep).shuffle(order)
                z={m:run(row["prompt"],m) for m in order}
                for e,k in modes:
                    item=z[(e,k)]
                    f.write(json.dumps(dict(id=row["id"],region=row["region"],split=row["split"],repeat=rep,exit=e,draft=k,greedy_equal=item["ids"]==z[(0,0)]["ids"],**item))+"\n")
                f.flush()
if __name__=="__main__": main()
