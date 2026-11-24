import json, time, statistics, platform
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(1)
V,D,N,B,S,L=16,24,128,4,32,2
class LM(nn.Module):
    def __init__(self,w):
        super().__init__(); self.w=w
        self.emb=nn.Embedding(V,D);self.pos=nn.Embedding(N,D)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(D,4,2*D,dropout=0,batch_first=True,norm_first=True) for _ in range(L)])
        self.out=nn.Linear(D,V)
    def forward(self,x,start=0,depth=L):
        n=x.size(1);h=self.emb(x)+self.pos(torch.arange(start,start+n))
        i=torch.arange(n);mask=(i[None,:]>i[:,None]) | (i[:,None]-i[None,:]>=self.w)
        for block in self.blocks[:depth]:h=block(h,src_mask=mask)
        return self.out(h)

def data(g):
    x=torch.randint(V,(B,N+1),generator=g)
    for t in range(1,N+1):
        repeat=torch.rand(B,generator=g)<.7
        x[:,t]=torch.where(repeat,(x[:,t-1]+t//S+1)%V,x[:,t])
    return x[:,:-1],x[:,1:]

def losses(m,x,y,w,mode):
    R=L*(w-1);out=[]
    def seg(a,b,c,d):
        if c<d:out.append(F.cross_entropy(m(x[:,a:b],a)[:,c-a:d-a].reshape(-1,V),y[:,c:d].reshape(-1),reduction='sum')/(B*N))
    if mode=='full':seg(0,N,0,N)
    else:
        for b in range(0,N,S):
            e=b+S;a=max(0,b-R);cut=min(e,b+R)
            if mode=='halo':seg(a,e,b,e)
            elif mode=='truncate':seg(b,e,b,e)
            elif mode=='defer':
                if b==0:seg(b,e,b,e)
                else:seg(b,e,cut,e);seg(a,cut,b,cut)
    return sum(out)

def grad(m,x,y,w,mode):
    m.zero_grad(set_to_none=True);loss=losses(m,x,y,w,mode);loss.backward()
    return float(loss.detach()),torch.cat([p.grad.flatten() for p in m.parameters()])

def run():
    rows=[]
    for w in [4,8,16]:
        for seed in [11,29,73]:
            torch.manual_seed(seed);g=torch.Generator().manual_seed(seed+1000);m=LM(w).double();opt=torch.optim.AdamW(m.parameters(),lr=.003)
            for _ in range(30):
                x,y=data(g);m.zero_grad(set_to_none=True);losses(m,x,y,w,'full').backward();opt.step()
            x,y=data(g);R=L*(w-1);ref,gv=grad(m,x,y,w,'full');metrics={}
            for mode in ['full','halo','defer','truncate']:
                val,gg=grad(m,x,y,w,mode);ts=[]
                for _ in range(5):
                    t=time.perf_counter();grad(m,x,y,w,mode);ts.append((time.perf_counter()-t)*1000)
                metrics[mode]={'loss':val,'rel_grad_error':float((gg-gv).norm()/gv.norm()),'median_ms':statistics.median(ts)}
            match=correct=alln=agree=boundaryn=0;maxerr=0.
            with torch.no_grad():
                z=m(x)
                for b in range(S,N,S):
                    a=b-R; guessed=x[:,a-1:a].clone()
                    for j in range(R):
                        nxt=m(guessed,a-1,depth=1)[:,-1].argmax(-1,keepdim=True);guessed=torch.cat([guessed,nxt],1)
                    h=guessed[:,1:];truth=x[:,a:b];same=h==truth;match+=int(same.all(1).sum());correct+=int(same.sum());alln+=same.numel()
                    zp=m(torch.cat([h,x[:,b:b+S]],1),a)[:,R:R+S];zt=z[:,b:b+S]
                    agree+=int((zp[:,:R].argmax(-1)==zt[:,:R].argmax(-1)).sum());boundaryn+=B*R
                    maxerr=max(maxerr,float((zp[:,R:]-zt[:,R:]).abs().max()))
            rows.append(dict(w=w,seed=seed,R=R,params=sum(p.numel() for p in m.parameters()),metrics=metrics,halo_exact=match/(B*3),halo_token_acc=correct/alln,boundary_greedy_agree=agree/boundaryn,interior_max_logit_err=maxerr))
    return {'env':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU','threads':torch.get_num_threads()},'settings':{'V':V,'D':D,'N':N,'B':B,'S':S,'L':L,'train_steps':30,'timing_repeats':5},'rows':rows}
if __name__=='__main__':print(json.dumps(run(),indent=2))
