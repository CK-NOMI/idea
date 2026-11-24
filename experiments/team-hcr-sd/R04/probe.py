import torch, time, statistics, json, platform
torch.set_num_threads(1)
def run():
    dtype=torch.float64; d=32; V=32; L=2; rows=[]
    for depth in [2,4,6]:
        N=2**(depth+1)-1
        for seed in [11,29,73]:
            g=torch.Generator().manual_seed(seed)
            W=[[(torch.randn(d,d,generator=g,dtype=dtype)/d**.5).requires_grad_() for _ in range(5)] for _ in range(L)]
            O=(torch.randn(d,V,generator=g,dtype=dtype)/d**.5).requires_grad_()
            X=torch.randn(N,d,generator=g,dtype=dtype)
            mask=torch.zeros(N,N,dtype=torch.bool)
            for i in range(N):
                j=i
                while True:
                    mask[i,j]=True
                    if j==0:break
                    j=(j-1)//2
            def forward(x,m,save=False):
                tape=[]
                for Q,K,Vv,A,B in W:
                    q=x@Q;k=x@K;v=x@Vv
                    a=torch.softmax((q@k.T/d**.5).masked_fill(~m,-torch.inf),dim=1)
                    c=a@v;u=torch.tanh(x@A);y=x+c+u@B
                    if save:tape.append((x,q,k,v,a,c,u))
                    x=y
                return x@O,tape,x
            path=[0]
            for _ in range(depth):path.append(path[-1]*2+1+int(torch.randint(2,(),generator=g)))
            ids=torch.tensor(path); n=len(path)
            labels=torch.randint(V,(n,),generator=g)
            m=torch.ones(n,n,dtype=torch.bool).tril()
            def reference():
                z,_,_=forward(X[ids],m)
                loss=torch.nn.functional.cross_entropy(z,labels)
                grads=torch.autograd.grad(loss,[*sum(W,[]),O])
                return z.detach(),grads
            @torch.no_grad()
            def verify():return forward(X,mask,True)
            @torch.no_grad()
            def compact_back(z,tape,h,ids=ids):
                h=h[ids]; p=torch.softmax(z[ids],dim=-1)
                p[torch.arange(n),labels]-=1;p/=n
                go=h.T@p;dy=p@O.T;allgr=[]
                for (Q,K,Vv,A,B),(x,q,k,v,a,c,u) in zip(reversed(W),reversed(tape)):
                    x=x[ids];q=q[ids];k=k[ids];v=v[ids];a=a[ids][:,ids];u=u[ids]
                    gb=u.T@dy;du=(dy@B.T)*(1-u*u);ga=x.T@du
                    dv=a.T@dy;da=dy@v.T
                    ds=a*(da-(da*a).sum(1,keepdim=True))/d**.5
                    dq=ds@k;dk=ds.T@q
                    allgr.append([x.T@dq,x.T@dk,x.T@dv,ga,gb])
                    dy=dy+du@A.T+dq@Q.T+dk@K.T+dv@Vv.T
                return [*sum(reversed(allgr),[]),go]
            @torch.no_grad()
            def matched_replay():
                zz,tt,hh=forward(X[ids],m,True)
                return compact_back(zz,tt,hh,torch.arange(n))
            z,tape,h=verify();zr,gr=reference();gg=compact_back(z,tape,h)
            err=(torch.cat([(a-b).flatten() for a,b in zip(gr,gg)]).norm()/torch.cat([a.flatten() for a in gr]).norm()).item()
            zerr=(z[ids]-zr).abs().max().item()
            def bench(fn):
                for _ in range(3):fn()
                ts=[]
                for _ in range(20):
                    t=time.perf_counter();fn();ts.append((time.perf_counter()-t)*1000)
                return statistics.median(ts)
            # retained custom tape sizes, not measured peak allocator memory.
            full_bytes=sum(a.numel()*a.element_size() for layer in tape for a in layer)+h.numel()*h.element_size()+z.numel()*z.element_size()
            compact_bytes=8*(L*(6*n*d+n*n)+n*d+n*V)
            rows.append(dict(depth=depth,seed=seed,nodes=N,path=n,relative_grad_error=err,max_logit_error=zerr,
                tape_bytes=full_bytes,compact_bytes=compact_bytes,verify_ms=bench(verify),replay_ms=bench(reference),
                compact_backward_ms=bench(lambda:compact_back(z,tape,h)),matched_replay_ms=bench(matched_replay)))
    return {"env":{"python":platform.python_version(),"torch":torch.__version__,"device":"CPU","threads":torch.get_num_threads()},
            "rows":rows}
if __name__=="__main__":print(json.dumps(run(),indent=2))
