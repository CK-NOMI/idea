import numpy as np, json
from numpy.linalg import solve
def sig(z): return 1/(1+np.exp(-z))
def eal(p,q):
    a=np.minimum(p,q);b=np.minimum(1-p,1-q)
    return np.mean(a[:,0]+b[:,0]+b[:,0]*(a[:,1]+b[:,1])+a[:,0]*(a[:,2]+b[:,2]))
def run(seed,scale):
    rng=np.random.default_rng(seed); C=8; E=2; d=8; k=3;n=6
    reg=rng.normal(size=(2,n,3)); data=[]
    for c in range(C):
        qz=rng.normal(size=(n,3));pz=qz+scale*(.7*reg[c//4]+.7*rng.normal(size=qz.shape))
        p=sig(pz);q=sig(qz);t=sig(pz+.35*rng.normal(size=pz.shape))
        S=rng.normal(size=(3*n,d))/np.sqrt(d);D=rng.normal(size=(3*n,k))/np.sqrt(k);Q=.7*S+.7*rng.normal(size=S.shape)/np.sqrt(d)
        J=np.c_[S,D];W=(p*(1-p)).ravel()/(3*n);A=J.T@(W[:,None]*J);b=-J.T@((p-t).ravel()/(3*n))
        def f(x): return eal(sig(pz+(J@x).reshape(n,3)),sig(qz+(Q@x[:d]).reshape(n,3)))
        eye=np.eye(d+k)*1e-5;a=np.array([(f(x)-f(-x))/2e-5 for x in eye])
        data.append((pz,qz,t,S,D,Q,A,b,a))
    def metrics(xs):
        vals=[]
        for (v,u),(pz,qz,t,S,D,Q,*_) in zip(xs,data):
            z=pz+(S@v+D@u).reshape(n,3)
            vals.append((eal(sig(z),sig(qz+(Q@v).reshape(n,3))),np.mean(np.logaddexp(0,z)-t*z),v@v+u@u))
        return np.mean(vals,axis=0)
    initial=metrics([(np.zeros(d),np.zeros(k))]*C)
    rows={}
    for beta in [0,.03,.1,.3]:
        leaves=[]
        for *_,A,b,a in data:
            V=A[d:,d:]+.03*np.eye(k);R=A[:d,d:];l=b+beta*a;K=A[:d,:d]-R@solve(V,R.T);f=l[:d]-R@solve(V,l[d:])
            leaves.append((K,f,V,R,l[d:]))
        for mode in ["global","tree"]:
            if mode=="global":
                v0=solve(sum(z[0] for z in leaves)+.025*C*np.eye(d),sum(z[1] for z in leaves))
                vs=[v0]*C;err=0.
            else:
                msg=[]
                for K,f,*_ in leaves:
                    B=K+.14*np.eye(d);msg.append((K-K@solve(B,K),f-K@solve(B,f),B,K,f))
                em=[]
                for e in range(E):
                    K=sum(z[0] for z in msg[4*e:4*e+4]);f=sum(z[1] for z in msg[4*e:4*e+4]);B=K+.24*np.eye(d)
                    em.append((K-K@solve(B,K),f-K@solve(B,f),B,K,f))
                v0=solve(sum(z[0] for z in em)+.025*C*np.eye(d),sum(z[1] for z in em))
                es=[solve(B,f-K@v0) for _,_,B,K,f in em]
                cs=[solve(B,f-K@(v0+es[c//4])) for c,(_,_,B,K,f) in enumerate(msg)]
                vs=[v0+es[c//4]+cs[c] for c in range(C)]
                H=np.diag(np.repeat([.025*C]+[.24]*E+[.14]*C,d)); h=np.zeros(len(H))
                for c,(K,f,*_) in enumerate(leaves):
                    T=np.zeros((d,len(H)))
                    for i in [0,1+c//4,1+E+c]: T[:,i*d:(i+1)*d]=np.eye(d)
                    H+=T.T@K@T;h+=T.T@f
                err=np.max(np.abs(solve(H,h)-np.concatenate([v0,*es,*cs])))
            xs=[(v,solve(V,l-R.T@v)) for v,(_,_,V,R,l) in zip(vs,leaves)]
            rows[(mode,beta)]=[*metrics(xs),err]
    return initial,rows
out={}
for scale in [.3,1,2]:
    runs=[run(s,scale) for s in range(30)]
    out[str(scale)]={"initial":np.mean([x[0] for x in runs],axis=0).tolist(),"rows":{}}
    for key in runs[0][1]:
        ar=np.array([x[1][key] for x in runs]);out[str(scale)]["rows"][str(key)]=[*ar[:,:3].mean(axis=0),ar[:,3].max()]
print(json.dumps(out,indent=1))
