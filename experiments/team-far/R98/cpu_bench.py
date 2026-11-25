import torch, time, json, statistics, random, platform
from itertools import product

def run():
    torch.manual_seed(98)
    rows=[]
    for threads,g,n in product([1,4],[4,16],[1,8]):
        torch.set_num_threads(threads)
        d,v,r=1024,8192,16
        for seed in [17,29,43]:
            torch.manual_seed(seed)
            X=torch.randn(g,n,d); A=torch.randn(r,d)/d**.5; B=torch.randn(g,v,r)/r**.5; W=torch.randn(v,d)/d**.5; G=torch.randn(g,n,v)
            Ae=A.expand(g,-1,-1).contiguous(); At=A.T.contiguous(); Aet=Ae.transpose(1,2); Bt=B.transpose(1,2); Wt=W.T
            def compute(shared,full,back):
                Z=(X.reshape(-1,d)@At).reshape(g,n,r) if shared else torch.bmm(X,Aet)
                Y=torch.bmm(Z,Bt)
                if full: Y=Y+(X.reshape(-1,d)@Wt).reshape(g,n,v)
                if not back: return (Y,)
                dz=torch.bmm(G,B); db=torch.bmm(G.transpose(1,2),Z)
                if shared:
                    da=dz.reshape(-1,r).T@X.reshape(-1,d); dx=(dz.reshape(-1,r)@A).reshape(g,n,d)
                else:
                    da=torch.bmm(dz.transpose(1,2),X).sum(0); dx=torch.bmm(dz,Ae)
                if full: dx=dx+(G.reshape(-1,v)@W).reshape(g,n,d)
                return Y,da,db,dx
            for full,back in product([False,True],[False,True]):
                a,b=compute(False,full,back),compute(True,full,back)
                error=max((x-y).abs().max().item()/max(1.,x.abs().max().item()) for x,y in zip(a,b))
                for _ in range(3): compute(False,full,back);compute(True,full,back)
                samples=[[],[]]; order=random.Random(seed)
                for _ in range(11):
                    for mode in order.sample([0,1],2):
                        t=time.perf_counter_ns();compute(bool(mode),full,back);samples[mode].append((time.perf_counter_ns()-t)/1e6)
                rows.append(dict(threads=threads,g=g,n=n,seed=seed,full=full,back=back,error=error,baseline_ms=statistics.median(samples[0]),shared_ms=statistics.median(samples[1]),raw_ms=samples))
    return dict(torch=torch.__version__,platform=platform.platform(),cuda=torch.cuda.is_available(),d=1024,v=8192,r=16,rows=rows)
if __name__=='__main__': print(json.dumps(run()))
