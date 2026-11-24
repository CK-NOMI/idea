import torch, json, pathlib, hashlib, platform, re
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(2)
V,T=27,16; seeds=[11,29,73]; corpora=[]; manifest=[]
for name in ['email','json','urllib']:
 paths=sorted((pathlib.Path(re.__file__).parent.parent/name).glob('*.py')); raw=b'\n'.join(p.read_bytes() for p in paths)
 manifest.append(dict(domain=name,files=len(paths),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 s=re.sub('[^a-z]+',' ',raw.decode(errors='replace').lower()).strip(); x=torch.tensor([26 if c==' ' else ord(c)-97 for c in s]); cut=int(.8*len(x));corpora.append((x[:cut],x[cut:]))
class LM(nn.Module):
 def __init__(self,d,L):
  super().__init__();self.e=nn.Embedding(V,d);self.pos=nn.Parameter(torch.zeros(T,d));self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,2,2*d,dropout=0,batch_first=True,norm_first=True) for _ in range(L)]);self.n=nn.LayerNorm(d);self.out=nn.Linear(d,V)
 def forward(self,x):
  z=self.e(x)+self.pos[:x.shape[1]];mask=torch.ones(x.shape[1],x.shape[1],dtype=torch.bool).triu(1)
  for i,l in enumerate(self.layers):
   z=l(z,src_mask=mask)
   if i==0:q=self.out(self.n(z))
  return self.out(self.n(z)),q
def batch(data,B,g):
 ix=torch.randint(len(data)-T-1,(B,),generator=g);a=data[ix[:,None]+torch.arange(T+1)];return a[:,:-1],a[:,1:]
rows=[]
for d,L in [(24,2),(32,3)]:
 for seed in seeds:
  torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+100);m=LM(d,L);opt=torch.optim.AdamW(m.parameters(),lr=.004)
  for step in range(60):
   x,y=batch(corpora[step%3][0],8,rng);p,q=m(x);loss=F.cross_entropy(p.flatten(0,1),y.flatten())+.3*F.cross_entropy(q.flatten(0,1),y.flatten());opt.zero_grad();loss.backward();opt.step()
  m.eval()
  for e,(_,test) in enumerate(corpora):
   with torch.no_grad():
    x,y=batch(test,12,rng);p,q=m(x);p=p.double().softmax(-1).reshape(-1,V);q=q.double().softmax(-1).reshape(-1,V);y=y.flatten();N=len(y)
   D=q[:,:,None]*p[:,None,:];a=(q-p).clamp_min(0);b=(p-q).clamp_min(0);delta=a.sum(-1);G=torch.diag_embed(torch.minimum(p,q))+a[:,:,None]*b[:,None,:]/delta[:,None,None].clamp_min(1e-300)
   DS=D+D.transpose(1,2);GS=G+G.transpose(1,2);idx=torch.arange(N);drow=DS[idx,y];grow=GS[idx,y];keep=torch.arange(V)[None,:]!=y[:,None];drow=drow*keep;grow=grow*keep
   def mean_vec(w):
    z=.5*w.clone();z[idx,y]=-.5*w.sum(-1);return z
   target=mean_vec(drow);naive=mean_vec(grow);norm=target.square().sum(-1);second=.5*drow.sum(-1);var=second-norm
   missing=(drow*(grow==0)).sum(-1);den=drow.sum(-1);ratios={}
   for lam in [.1,.25,.5,1.]:
    mix=(1-lam)*grow+lam*drow;s2=.5*(drow.square()/mix.clamp_min(1e-300)).sum(-1);ratios[str(lam)]=float((s2-norm).mean()/var.mean())
   oracle=.5*den.square()-norm
   rows.append(dict(d=d,L=L,seed=seed,region=e,N=N,params=sum(t.numel() for t in m.parameters()),pNLL=float(-p[idx,y].log().mean()),qNLL=float(-q[idx,y].log().mean()),accept=float(1-delta.mean()),indInformative=float(den.mean()),sdInformative=float(grow.sum(-1).mean()),missingMass=float(missing.mean()),missingFraction=float(missing.sum()/den.sum()),relativeBias=float(((target-naive).square().sum()/target.square().sum()).sqrt()),indVar=float(var.mean()),oracleVarRatio=float(oracle.mean()/var.mean()),mixtureVarRatio=ratios,marginalError=float((G.sum(1)-p).abs().max())))
print(json.dumps(dict(environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.cuda.is_available(),threads=torch.get_num_threads()),manifest=manifest,seeds=seeds,trainLengths=[len(x) for x,y in corpora],testLengths=[len(y) for x,y in corpora],rows=rows),indent=2))
