import torch, numpy as np, json, pathlib, sysconfig, hashlib, re, time, itertools
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(2)
root=pathlib.Path(sysconfig.get_path('stdlib')); V=27; T=12; B=4
texts=[]; manifest={}
for domain in ['email','json','urllib']:
    files=sorted((root/domain).glob('*.py')); raw=b'\n'.join(p.read_bytes() for p in files); manifest[domain]={'files':len(files),'sha256':hashlib.sha256(raw).hexdigest()}
    x=re.sub('[^a-z]+',' ',raw.decode('utf8',errors='replace').lower()); a=torch.tensor([26 if c==' ' else ord(c)-97 for c in x]); texts.append((a[:int(.8*len(a))],a[int(.8*len(a)):]))
class Model(nn.Module):
    def __init__(self,d,L):
        super().__init__(); self.emb=nn.Embedding(V,d); self.pos=nn.Parameter(torch.zeros(T,d)); self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d,2,2*d,dropout=0,batch_first=True,norm_first=True) for _ in range(L)]); self.norm=nn.LayerNorm(d); self.out=nn.Linear(d,V)
    def forward(self,x,shallow=False):
        h=self.emb(x)+self.pos[:x.shape[1]]; mask=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool).triu(1)
        for j,layer in enumerate(self.blocks):
            h=layer(h,src_mask=mask); 
            if shallow or j==len(self.blocks)-1:
                p=self.out(self.norm(h))
                if shallow:return p
        return p
    def both(self,x):
        h=self.emb(x)+self.pos[:x.shape[1]]; mask=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool).triu(1)
        for j,layer in enumerate(self.blocks):
            h=layer(h,src_mask=mask)
            if j==0:q=self.out(self.norm(h))
        return self.out(self.norm(h)),q

def batch(a):
    idx=torch.randint(len(a)-T,(B,)); y=a[idx[:,None]+torch.arange(T+1)]; return y[:,:-1],y[:,1:]
def grad(m,loss):
    gs=torch.autograd.grad(loss,tuple(m.parameters()),allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(m.parameters(),gs)]).double().numpy()
rows=[]
for L,d in [(2,24),(3,32)]:
 for seed in [11,29,73]:
    torch.manual_seed(seed); m=Model(d,L); opt=torch.optim.AdamW(m.parameters(),lr=.004); start=time.perf_counter()
    for step in range(60):
        x,y=batch(texts[step%3][0]); p,q=m.both(x); loss=F.cross_entropy(p.reshape(-1,V),y.flatten())+.3*F.cross_entropy(q.reshape(-1,V),y.flatten()); opt.zero_grad(); loss.backward(); opt.step()
    train_sec=time.perf_counter()-start; m.eval(); acc=[]; stats={pi:[] for pi in [.25,.5,.75]}; fulltime=[]; shallowtime=[]; cohortgrad=[]
    for cohort in range(8):
        gs=[];hs=[]
        for i in range(4):
            x,y=batch(texts[i%3][1]); label=y[:,-1]
            with torch.no_grad():
                p,q=m.both(x); p=p[:,-1].softmax(-1).double(); q=q[:,-1].softmax(-1).double(); p=p/p.sum(-1,keepdim=True);q=q/q.sum(-1,keepdim=True)
                X=torch.multinomial(q,1).squeeze(-1); a=torch.minimum(torch.ones(B,dtype=torch.double),p.gather(1,X[:,None]).squeeze(1)/q.gather(1,X[:,None]).squeeze(1)); ok=torch.rand(B,dtype=torch.double)<a; r=(p-q).clamp_min(0); Y=X.clone()
                if (~ok).any():Y[~ok]=torch.multinomial(r[~ok],1).squeeze(1)
                acc.extend(ok.tolist()); reward=(Y==label).float()-.05
            t=time.perf_counter(); g=grad(m,-(reward*F.log_softmax(m(x)[:,-1],-1).gather(1,Y[:,None]).squeeze(1)).mean());fulltime.append(time.perf_counter()-t)
            t=time.perf_counter(); h=grad(m,-(reward*F.log_softmax(m(x,True)[:,-1],-1).gather(1,Y[:,None]).squeeze(1)).mean());shallowtime.append(time.perf_counter()-t);gs.append(g/4);hs.append(h/4)
        G=np.sum(gs,axis=0); H=np.sum(hs,axis=0); D=np.array(gs)-np.array(hs); den=np.sum(G*G); ref=G/(np.abs(G)+1e-3)
        for pi in stats:
            ex=np.zeros_like(G); ex2=ex.copy(); exclip=ex.copy(); exstep=ex.copy(); excstep=ex.copy(); neg=0.; mse=0.; fullprob=pi**4
            for mask in itertools.product([0,1],repeat=4):
                I=np.array(mask); pr=pi**I.sum()*(1-pi)**(4-I.sum()); est=H+(I[:,None]*D/pi).sum(0); corr=est**2-((1-pi)*I[:,None]*D**2/pi**2).sum(0); clip=np.maximum(corr,0)
                ex+=pr*est;ex2+=pr*corr;exclip+=pr*clip;neg+=pr*np.mean(corr<0);mse+=pr*np.sum((est-G)**2)
                exstep+=pr*est/(np.abs(est)+1e-3);excstep+=pr*est/(np.sqrt(clip)+1e-3)
            var=((1-pi)/pi)*(D**2).sum(); stats[pi].append(dict(grad_err=float(np.max(np.abs(ex-G))),square_err=float(np.max(np.abs(ex2-G**2))),noise_ratio=float(var/den),clip_excess=float((exclip.sum()-den)/den),negative_frac=float(neg),naive_step_rel=float(np.linalg.norm(exstep-ref)/np.linalg.norm(ref)),clip_step_rel=float(np.linalg.norm(excstep-ref)/np.linalg.norm(ref)),mse_err=float(abs(mse-var))))
    for pi,vals in stats.items():
        row={k:float(np.mean([v[k] for v in vals])) for k in vals[0]};row.update(L=L,d=d,seed=seed,pi=pi,params=sum(p.numel() for p in m.parameters()),accept=float(np.mean(acc)),full_ms=1000*np.median(fulltime),shallow_ms=1000*np.median(shallowtime),train_sec=train_sec);rows.append(row)
result={'torch':torch.__version__,'data':manifest,'lengths':[[len(a),len(b)] for a,b in texts],'rows':rows,'cohorts_per_model':8,'clients':4,'batch':B,'T':T,'pretrain_steps':60,'reference_step_epsilon':.001}
print(json.dumps(result,indent=2))
