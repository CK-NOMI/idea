import torch, json, random, sys
# Minimal transcription of Transformers v4.50.0 NoRepeatNGram index semantics.
# Does not import or run the full LayerSkip model or Transformers package.
def banned(h,n):
    if len(h)+1<n: return []
    suffix=tuple(h[-(n-1):]) if n>1 else ()
    return [h[i+n-1] for i in range(len(h)-n+1) if tuple(h[i:i+n-1])==suffix]
def processor(h,scores,n):
    out=scores.clone()
    out[0,banned(h,n)]=-float("inf")
    return out
def masked(h,z,n):
    return processor(h,z[None,:],n)[0]
def case(h,T,V,n):
    z=torch.arange(V,dtype=torch.float64).repeat(1,T,1)
    try:
        bad=processor(h,z,n)
        return {"banned":banned(h,n),"masked_rows":torch.isneginf(bad).all(-1)[0].nonzero().flatten().tolist(),"masked_entries":int(torch.isneginf(bad).sum())}
    except Exception as e:
        return {"banned":banned(h,n),"exception":type(e).__name__,"message":str(e)}
cases={"out_of_range":case([4],3,8,1),"wrong_axis":case([1],3,8,1)}
z=torch.tensor([5.,1.,4.,0.],dtype=torch.float64)
h=[0,2,1]; inc=[1]
cases["incremental_history"]={"history":h,"incremental":inc,"correct_banned":banned(h,1),"incremental_banned":banned(inc,1),"correct_greedy":int(masked(h,z,1).argmax()),"incremental_greedy":int(masked(inc,z,1).argmax())}
rows=[]
for seed in (11,29,73):
    rng=random.Random(seed)
    for n in (1,2,3):
        for B in (2,4,8):
            num=diff=invalid=0; tvsum=0.; replay_mismatch=0
            for rep in range(80):
                V=32; prefix=[rng.randrange(V) for _ in range(12)]
                H=prefix[:]; draft=[]
                for i in range(B):
                    allow=[v for v in range(V) if v not in banned(H,n)]
                    if not allow: break
                    x=rng.choice(allow);draft.append(x);H.append(x)
                seen={}
                for i in range(len(prefix)-n+1):
                    seen.setdefault(tuple(prefix[i:i+n-1]),set()).add(prefix[i+n-1])
                hh=prefix[:]
                for i in range(len(draft)+1):
                    z=torch.tensor([rng.random()*3 for _ in range(V)],dtype=torch.float64)
                    legal=set(banned(hh,n))
                    suffix=tuple(hh[-(n-1):]) if n>1 else ()
                    current=seen.get(suffix,set()) if len(hh)+1>=n else set()
                    replay_mismatch+=int(current!=legal)
                    true=masked(hh,z,n).softmax(-1)
                    wrong=masked(H,z,n).softmax(-1)
                    if not torch.isfinite(wrong).all(): invalid+=1
                    else:
                        tv=float((true-wrong).abs().sum()/2);tvsum+=tv;diff+=int(tv>1e-12)
                    num+=1
                    if i<len(draft):
                        if len(hh)+1>=n: seen.setdefault(suffix,set()).add(draft[i])
                        hh.append(draft[i])
            rows.append(dict(seed=seed,n=n,B=B,rows=num,different=diff,invalid=invalid,mean_tv=tvsum/(num-invalid),replay_mismatch=replay_mismatch))
result={"env":{"python":sys.version.split()[0],"torch":torch.__version__,"cuda":torch.cuda.is_available()},"cases":cases,"rows":rows}
print(json.dumps(result,indent=2))
