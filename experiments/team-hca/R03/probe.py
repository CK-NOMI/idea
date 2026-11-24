import numpy as np
import json, sys
def probe():
    rows=[]
    for gauge in [False,True]:
        for drift in [.02,.2,.6]:
            for seed in [11,29,73]:
                rng=np.random.default_rng(seed)
                metrics=[]
                for trial in range(80):
                    d,r,V,n=16,4,8,8
                    A0=np.linalg.qr(rng.normal(size=(d,r)))[0].T
                    B0=rng.normal(size=(V,r))/2
                    C0=B0+.15*rng.normal(size=(V,r))
                    As=[];Bs=[];Cs=[]
                    for i in range(n):
                        A=A0+drift*rng.normal(size=(r,d))/np.sqrt(d)
                        B=B0+drift*rng.normal(size=(V,r))/2
                        C=C0+drift*rng.normal(size=(V,r))/2
                        G=np.linalg.qr(rng.normal(size=(r,r)))[0] if gauge else np.eye(r)
                        As.append(G@A);Bs.append(B@G.T);Cs.append(C@G.T)
                    A,B,C=map(np.array,(As,Bs,Cs))
                    P=(B@A).mean(0);Q=(C@A).mean(0)
                    X=rng.normal(size=(128,d))
                    def prob(W):
                        z=X@W.T;z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
                    def err(P1,Q1):
                        return [np.linalg.norm(P1-P)/np.linalg.norm(P),np.linalg.norm(Q1-Q)/np.linalg.norm(Q),
                                np.minimum(prob(P1),prob(Q1)).sum(1).mean(),
                                np.minimum(prob(P),prob(P1)).sum(1).mean()]
                    naive=(B.mean(0)@A.mean(0),C.mean(0)@A.mean(0))
                    aa=[];bb=[];cc=[]
                    for a,b,c in zip(A,B,C):
                        u,_,vt=np.linalg.svd(A0@a.T)
                        R=u@vt
                        aa.append(R@a);bb.append(b@R.T);cc.append(c@R.T)
                    aligned=(np.mean(bb,0)@np.mean(aa,0),np.mean(cc,0)@np.mean(aa,0))
                    Z=np.vstack((P,Q));u,s,vt=np.linalg.svd(Z,full_matrices=False)
                    Zr=(u[:,:r]*s[:r])@vt[:r]
                    joint=Zr[:V],Zr[V:]
                    def trunc(z):
                        u,s,vt=np.linalg.svd(z,full_matrices=False)
                        return (u[:,:r]*s[:r])@vt[:r]
                    separate=trunc(P),trunc(Q)
                    metrics.append([err(*m) for m in (naive,aligned,joint,separate,(P,Q))])
                rows.append(dict(gauge=gauge,drift=drift,seed=seed,metrics=np.mean(metrics,axis=0).tolist()))
    return rows
if __name__=="__main__":
    print("Python",sys.version.split()[0],"NumPy",np.__version__,file=sys.stderr)
    print(json.dumps(probe(),indent=2))
