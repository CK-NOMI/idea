import torch,json,time,statistics,platform
# Real CPU projection operations, synthetic weights/hidden vectors; no pretrained model.
torch.set_num_threads(2)
B,V=48,4096
rows=[]
for d in [1536,2048]:
 for seed in [11,29,73]:
  gen=torch.Generator().manual_seed(seed)
  W=torch.randn(V,d,generator=gen)/d**.5
  X=torch.randn(B,d,generator=gen)
  W64,X64=W.double(),X.double()
  ref=X64@W64.T
  def bench(f):
   f();ts=[]
   for _ in range(5):
    t=time.perf_counter();f();ts.append((time.perf_counter()-t)*1000)
   return statistics.median(ts)
  base_ms=bench(lambda:X64@W64.T)
  U=torch.rand(B,V,generator=gen,dtype=torch.float64).clamp(2**-53,1-2**-53)
  G=-torch.log(-torch.log(U))
  for dtype in [torch.bfloat16,torch.float16]:
   # Round operands to the named dtype, explicitly accumulate using FP32 matmul.
   Wq,Xq=W.to(dtype).float(),X.to(dtype).float()
   Wq64,Xq64=Wq.double(),Xq.double()
   wr=(W64-Wq64).norm(dim=1);wn=Wq64.norm(dim=1);w0=W64.norm(dim=1)
   gamma=lambda u:d*u/(1-d*u)
   def approx_bound():
    a=(Xq@Wq.T).double()
    # Cauchy-Schwarz + classical dot product bound, with FP64 guard;
    # conditional on ordinary rounding and accumulation; not formally directed interval arithmetic.
    e=X64.norm(dim=1)[:,None]*wr[None,:]+(X64-Xq64).norm(dim=1)[:,None]*wn[None,:]+gamma(2**-24)*Xq64.norm(dim=1)[:,None]*wn[None,:]+gamma(2**-53)*X64.norm(dim=1)[:,None]*w0[None,:]+1e-10
    return a,e
   a,e=approx_bound(); diff=(ref-a).abs()
   low_ms=bench(lambda:Xq@Wq.T);cert_ms=bench(approx_bound)
   for temp in [0.,.5,1.,2.]:
    noise=G*temp; score=a+noise; truth=ref+noise
    L=score-e;H=score+e
    S=H>=L.max(1,keepdim=True).values
    winner=truth.argmax(1); naive=score.argmax(1)
    # exact mathematical scores for surviving candidates, reuse identical noise.
    def refine():
     idx=S.nonzero(); ri,vi=idx[:,0],idx[:,1]; vals=(W64[vi]*X64[ri]).sum(1)+noise[ri,vi]
     out=torch.full_like(score,-torch.inf);out[ri,vi]=vals;return out.argmax(1)
    repaired=refine(); n=S.sum(1); clear=n==1
    t_ms=bench(refine)
    # probability shift measured at unit temperature, independent of the Gumbel draw.
    p=ref.softmax(1);q=a.softmax(1)
    tv=(p-q).abs().sum(1)/2;kl=(p*(p.log()-q.log())).sum(1)
    rows.append(dict(d=d,seed=seed,precision=str(dtype),temp=temp,B=B,V=V,mean_bound=e.mean().item(),max_actual_error=diff.max().item(),bound_violations=int((diff>e).sum()),raw_mismatch=int((winner!=naive).sum()),refined_mismatch=int((winner!=repaired).sum()),certificate_fraction=clear.double().mean().item(),mean_active=n.double().mean().item(),max_active=int(n.max()),mean_TV=tv.mean().item(),mean_KL=kl.mean().item(),reference_ms=base_ms,low_ms=low_ms,certificate_ms=cert_ms,refine_ms=t_ms))
print(json.dumps(dict(environment=dict(python=platform.python_version(),torch=torch.__version__,device='CPU',threads=torch.get_num_threads()),settings=dict(B=B,V=V,seeds=[11,29,73],dimensions=[1536,2048],warmup=1,repeats=5),rows=rows),indent=2))
