import torch,torch.nn as nn,torch.nn.functional as F,numpy as np,sysconfig,pathlib,json,itertools,math,hashlib
V,T,B=256,24,4; domains=['email','json','urllib']; torch.set_num_threads(2)
root=pathlib.Path(sysconfig.get_path('stdlib')); corpus={}; manifest={}
for dom in domains:
 files=sorted((root/dom).glob('*.py')); raw=[]
 for f in files:
  b=f.read_bytes();raw.append(b);manifest[str(f.relative_to(root))]=hashlib.sha256(b).hexdigest()
 a=torch.tensor(np.frombuffer(b'\n'.join(raw),dtype=np.uint8).astype(np.int64));k=int(len(a)*.8);corpus[dom]=(a[:k],a[k:])
def batch(dom,part,rng):
 a=corpus[dom][part];i=torch.randint(len(a)-T-1,(B,),generator=rng);x=a[i[:,None]+torch.arange(T+1)];return x[:,:-1],x[:,1:]
class Model(nn.Module):
 def __init__(self,d,L):
  super().__init__();self.e=nn.Embedding(V,d);self.pos=nn.Embedding(T,d);self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,2,2*d,dropout=0,batch_first=True,norm_first=True,activation='gelu') for _ in range(L)]);self.n=nn.LayerNorm(d);self.out=nn.Linear(d,V,bias=False);self.mid=L//2
 def forward(self,x):
  h=self.e(x)+self.pos(torch.arange(x.shape[1]));mask=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),diagonal=1)
  for l in self.layers[:self.mid]:h=l(h,src_mask=mask,is_causal=True)
  q=self.out(self.n(h))
  for l in self.layers[self.mid:]:h=l(h,src_mask=mask,is_causal=True)
  return q,self.out(self.n(h))
def ce(z,y):return F.cross_entropy(z.reshape(-1,V),y.reshape(-1))
def flat(gs,ps):return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gs,ps)]).detach().double().numpy()
def audit(g,c,bd):
 lo=c-bd;hi=c+bd;nz=np.broadcast_to(bd>0,g.shape);lo[nz]=np.nextafter(lo[nz],-np.inf);hi[nz]=np.nextafter(hi[nz],np.inf);remain=np.ones(len(g),bool);m=len(g)//2;curve=[]
 for k in range(len(g)+1):
  a=np.partition(lo,m,axis=0)[m];b=np.partition(hi,m,axis=0)[m];closed=a==b;curve.append(float(closed.mean()))
  if closed.all():return [k,curve,float(np.max(np.abs(a-np.median(g,axis=0))))]
  scores=((lo<=b)&(hi>=a)&(~closed)[None,:]).sum(axis=1);scores[~remain]=-1;i=int(scores.argmax());remain[i]=False;lo[i]=g[i];hi[i]=g[i]
 raise RuntimeError('did not close')
def run():
 rows=[]
 for d,L in [(32,2),(48,4)]:
  for seed in [11,29,73]:
   torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+100);r=np.random.default_rng(seed+700);model=Model(d,L);ps=list(model.parameters());opt=torch.optim.AdamW(ps,lr=.003)
   for step in range(100):
    x,y=batch(domains[step%3],0,rng);q,p=model(x);loss=ce(p,y)+.3*ce(q,y);opt.zero_grad();loss.backward();opt.step()
   blocks=np.cumsum([0]+[p.numel() for p in ps]);D=int(blocks[-1]);n=9
   for cohort in range(3):
    G=[];C=[];overlap=[]
    for i in range(n):
     x,y=batch(domains[i//3],1,rng);q,p=model(x);lq=ce(q,y);lp=ce(p,y);G.append(flat(torch.autograd.grad(lp+.3*lq,ps,retain_graph=True,allow_unused=True),ps));C.append(flat(torch.autograd.grad(1.3*lq,ps,allow_unused=True),ps));overlap.append(float(1-(p.softmax(-1)-q.softmax(-1)).abs().sum(-1).mean().detach()/2))
    g=np.stack(G);c=np.stack(C);diff=np.abs(g-c);med=np.median(g,axis=0);den=float(med@med)+1e-30
    ordered=np.sort(g,axis=0);unique=(ordered[4]!=ordered[3])&(ordered[4]!=ordered[5]);winner=np.argmin(np.abs(g-med),axis=0);needs=unique&(diff[winner,np.arange(D)]>0);counts=np.bincount(winner[needs],minlength=n)
    bd=np.empty_like(diff)
    for a,b in zip(blocks[:-1],blocks[1:]):bd[:,a:b]=diff[:,a:b].max(axis=1)[:,None]
    cert={name:audit(g,c,bounds) for name,bounds in [('client',diff.max(axis=1)[:,None]),('tensor',bd),('coordinate',diff)]}
    mreg=np.median(np.stack([np.median(g[i:i+3],axis=0) for i in [0,3,6]]),axis=0);mh=np.median(c,axis=0);approx=[]
    for k in [3,6]:
     errs=[];ht=[];avg=np.zeros(D)
     for j in range(12):
      ids=r.choice(n,k,replace=False);z=c.copy();z[ids]=g[ids];zh=c.copy();zh[ids]+=(g[ids]-c[ids])*(n/k);a=np.median(z,axis=0);b=np.median(zh,axis=0);avg+=b/12;errs.append(float(np.sum((a-med)**2)/den));ht.append(float(np.sum((b-med)**2)/den))
     approx.append([k,float(np.mean(errs)),float(np.mean(ht)),float(np.sum((avg-med)**2)/den)])
    rows.append(dict(d=d,L=L,seed=seed,cohort=cohort,D=D,overlap=float(np.mean(overlap)),regional=float(np.sum((mreg-med)**2)/den),shallow=float(np.sum((mh-med)**2)/den),unique=float(unique.mean()),need=float(needs.mean()),winner_counts=counts.tolist(),cert=cert,approx=approx))
   print('done',L,seed,flush=True)
 fourier=[]
 for n in [3,5,7,9]:
  a=sum((1 if sum(x)>0 else -1)*math.prod(x) for x in itertools.product([-1,1],repeat=n))/2**n;fourier.append([n,a,2.**-n,a*a*(2**n-1)])
 return dict(env=str(torch.__version__),cuda=torch.cuda.is_available(),manifest=manifest,corpus={k:[len(x) for x in v] for k,v in corpus.items()},rows=rows,fourier=fourier)
if __name__=='__main__':
 out=run();pathlib.Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2))
