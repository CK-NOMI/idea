import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sysconfig, pathlib, hashlib, json, time, statistics, platform
from collections import defaultdict
import copy
torch.set_num_threads(2)
V=256; T=24; B=4; AUX=.3; SEEDS=[11,29,73]; DOM=['email','json','urllib']
root=pathlib.Path(sysconfig.get_path('stdlib')); manifest={}; corp={}
for name in DOM:
 raw=[]
 for f in sorted((root/name).glob('*.py')):
  b=f.read_bytes(); manifest[str(f.relative_to(root))]=hashlib.sha256(b).hexdigest(); raw.append(b)
 data=np.frombuffer(b'\n'.join(raw),dtype=np.uint8).astype(np.int64); split=int(len(data)*.8)
 corp[name]=(torch.tensor(data[:split]),torch.tensor(data[split:]))
def batch(name,part,g):
 a=corp[name][part]; i=torch.randint(len(a)-T-1,(B,),generator=g); x=a[i[:,None]+torch.arange(T+1)];return x[:,:-1],x[:,1:]
class Block(nn.Module):
 def __init__(self,d):
  super().__init__();self.ln=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.n2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
 def forward(self,x):
  b,t,d=x.shape;q,k,v=self.qkv(self.ln(x)).reshape(b,t,3,2,d//2).permute(2,0,3,1,4).unbind(0);a=F.scaled_dot_product_attention(q,k,v,is_causal=True);x=x+self.proj(a.transpose(1,2).reshape(b,t,d));return x+self.ff(self.n2(x))
class Model(nn.Module):
 def __init__(self,d,L):
  super().__init__();self.emb=nn.Embedding(V,d);self.pos=nn.Embedding(T,d);self.layers=nn.ModuleList([Block(d) for _ in range(L)]);self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,V,bias=False);self.mid=L//2
 def forward(self,x,full=True):
  h=self.emb(x)+self.pos(torch.arange(x.shape[1]));
  for layer in self.layers[:self.mid]:h=layer(h)
  q=self.head(self.norm(h));
  if not full:return q
  for layer in self.layers[self.mid:]:h=layer(h)
  return q,self.head(self.norm(h))
def ce(z,y):return F.cross_entropy(z.reshape(-1,V),y.reshape(-1))
def flat(gs,ps):return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gs,ps)]).detach().double().numpy()
def allocate(s,mean=.5,apply=None):
 s=np.maximum(s,1e-12);lo,hi=0.,100/s.min()
 for _ in range(70):
  k=(lo+hi)/2
  if np.clip(k*s,.1,1).mean()<mean:lo=k
  else:hi=k
 return np.clip((lo+hi)/2*(s if apply is None else apply),.1,1)
rows=[]
for d,L in [(32,2),(48,4)]:
 for seed in SEEDS:
  torch.manual_seed(seed);g=torch.Generator().manual_seed(seed+100);m=Model(d,L);ps=list(m.parameters());opt=torch.optim.AdamW(m.parameters(),lr=.003)
  for i in range(160):
   x,y=batch(DOM[i%3],0,g);q,p=m(x);loss=ce(p,y)+AUX*ce(q,y);opt.zero_grad();loss.backward();opt.step()
  deep=np.concatenate([np.full(p.numel(),any(p is z for layer in m.layers[m.mid:] for z in layer.parameters()),dtype=bool) for p in ps]);
  def inspect(name,part,n):
   obs=[]
   for _ in range(n):
    x,y=batch(name,part,g);q,p=m(x);lq=ce(q,y);lp=ce(p,y);gp=flat(torch.autograd.grad(lp+AUX*lq,ps,retain_graph=True,allow_unused=True),ps);gq=flat(torch.autograd.grad((1+AUX)*lq,ps,allow_unused=True),ps)
    with torch.no_grad():
     pq=q.softmax(-1);pp=p.softmax(-1);tv=(pq-pp).abs().sum(-1).mean().item()/2;entropy=-(pq*pq.clamp_min(1e-30).log()).sum(-1).mean().item();agree=(p.argmax(-1)==q.argmax(-1)).float().mean().item()
    diff=gp-gq;obs.append(dict(domain=name,lp=lp.item(),lq=lq.item(),tv=tv,entropy=entropy,agree=agree,gp=gp,gq=gq,delta2=float(diff@diff),deep2=float(gp[deep]@gp[deep])))
   return obs
  cal=sum([inspect(n,0,12) for n in DOM],[]);test=sum([inspect(n,1,16) for n in DOM],[])
  X=lambda a: np.array([[1,o['entropy'],o['lq'],*(float(o['domain']==n) for n in DOM[1:])] for o in a]); target=lambda a:np.array([o['delta2'] for o in a]);xc=X(cal); mean=xc[:,1:3].mean(0);std=xc[:,1:3].std(0)+1e-8;xc[:,1:3]=(xc[:,1:3]-mean)/std;xt=X(test);xt[:,1:3]=(xt[:,1:3]-mean)/std
  reg=np.eye(xc.shape[1])*.5;reg[0,0]=0;w=np.linalg.solve(xc.T@xc+reg,xc.T@np.log(target(cal)+1e-12));sc=np.exp(np.clip(xc@w,-20,20)/2);st=np.exp(np.clip(xt@w,-20,20)/2)
  pi=allocate(sc,apply=st);constant=np.full(len(test),pi.mean());oracle=allocate(np.sqrt(target(test)),pi.mean())
  x,y=batch('json',1,g)
  def run(which):
   if which=='shallow':z=m(x,False);l=(1+AUX)*ce(z,y)
   else:
    q,p=m(x);lq,lp=ce(q,y),ce(p,y);l=lp+AUX*lq if which=='full' else 2*lp+(AUX-1)*lq
   torch.autograd.grad(l,ps,allow_unused=True)
  tm=defaultdict(list)
  for j in range(12):
   for k in (['shallow','full','correction'] if j%2==0 else ['correction','full','shallow']):
    t=time.perf_counter();run(k);dt=time.perf_counter()-t
    if j>=2:tm[k].append(dt*1000)
  times={k:statistics.median(v) for k,v in tm.items()}; gp=np.stack([o['gp'] for o in test]);gq=np.stack([o['gq'] for o in test]);mu=gp.mean(0);basevar=float(np.mean(np.sum((gp-mu)**2,axis=1)));diff=target(test);deep2=np.array([o['deep2'] for o in test]);vfull=basevar*times['full'];
  def metrics(p):
   extra=np.mean((1/p-1)*diff);deepvar=np.mean((1/p-1)*deep2);cost=(1-p.mean())*times['shallow']+p.mean()*times['correction'];return dict(mean_pi=float(p.mean()),extra_var=float(extra),deep_var=float(deepvar),expected_ms=float(cost),cost_var_ratio=float(cost*(basevar+extra)/vfull))
  maxerr=float(np.max(np.abs((1-pi[:,None])*gq+pi[:,None]*(gq+(gp-gq)/pi[:,None])-gp)))
  # Logical HFL continuation: 3 regions, 2 clients each, one local gradient, linear aggregation.
  # No physical concurrency or network. Equal compute proxy budget, actual serial CPU time recorded separately.
  def continuation(method):
   mm=copy.deepcopy(m);pp=list(mm.parameters());oo=torch.optim.SGD(pp,lr=.12);gg=torch.Generator().manual_seed(seed+777);rr=torch.Generator().manual_seed(seed+999)
   ev=torch.Generator().manual_seed(seed+1111)
   eval_batches=[batch(n,1,ev) for n in DOM for _ in range(12)]
   def evaluate():
    with torch.no_grad():
     vals=[ce(mm(x)[1],y).item() for x,y in eval_batches];return float(np.mean(vals))
   before=evaluate(); rounds=30 if method=='full' else int(30*times['full']/metrics(pi)['expected_ms']); count=0;flaterr=0.;t0=time.perf_counter()
   for rd in range(rounds):
    region=[];all_gr=[]
    for j,name in enumerate(DOM):
     local=[]
     for client in range(2):
      x,y=batch(name,0,gg)
      if method=='full':
       q,p=mm(x);loss=ce(p,y)+AUX*ce(q,y);count+=1
      else:
       # Constant policy is the best simple comparator; adaptive policy needs an extra shallow forward to obtain features.
       prob=float(pi.mean())
       if method=='predictive':
        with torch.no_grad():
         z=mm(x,False);pz=z.softmax(-1);f=np.array([1.,float(-(pz*pz.clamp_min(1e-30).log()).sum(-1).mean()),ce(z,y).item(),float(name==DOM[1]),float(name==DOM[2])]);f[1:3]=(f[1:3]-mean)/std;prob=float(allocate(sc,apply=np.array([np.exp(np.clip(f@w,-20,20)/2)]))[0])
       if torch.rand((),generator=rr).item()<prob:
        q,p=mm(x);loss=ce(p,y)/prob+(1+AUX-1/prob)*ce(q,y);count+=1
       else:loss=(1+AUX)*ce(mm(x,False),y)
      gg0=torch.autograd.grad(loss,pp,allow_unused=True);local.append([a.detach() if a is not None else torch.zeros_like(b) for a,b in zip(gg0,pp)]);all_gr.append(local[-1])
     region.append([(a+b)/2 for a,b in zip(*local)])
    oo.zero_grad()
    for k,param in enumerate(pp):
     param.grad=sum(r[k] for r in region)/len(region)
     if rd==0:flaterr=max(flaterr,float((param.grad-sum(r[k] for r in all_gr)/len(all_gr)).abs().max()))
    oo.step()
   dt=time.perf_counter()-t0
   return dict(method=method,rounds=rounds,logical_clients=6,regions=3,full_gradients=count,total_gradients=rounds*6,full_fraction=count/(rounds*6),cpu_seconds=dt,before_nll=before,after_nll=evaluate(),flat_max_difference=flaterr)
  continued=[continuation(k) for k in ['full','constant','predictive']]
  rows.append(dict(d=d,L=L,seed=seed,params=sum(p.numel() for p in ps),train_steps=160,cal_n=len(cal),test_n=len(test),heldout_p_nll=float(np.mean([o['lp'] for o in test])),heldout_q_nll=float(np.mean([o['lq'] for o in test])),mean_accept=float(1-np.mean([o['tv'] for o in test])),greedy_match=float(np.mean([o['agree'] for o in test])),corr_tv_delta2=float(np.corrcoef([o['tv'] for o in test],diff)[0,1]),corr_entropy_delta2=float(np.corrcoef([o['entropy'] for o in test],diff)[0,1]),mean_delta2=float(diff.mean()),mean_deep2=float(deep2.mean()),basevar=basevar,full_ms=times['full'],shallow_ms=times['shallow'],correction_ms=times['correction'],constant=metrics(constant),predictive=metrics(pi),oracle=metrics(oracle),full_iid_cost_var_ratio=1.,shallow_bias_l2=float(np.linalg.norm(gq.mean(0)-mu)),max_expectation_error=maxerr,continued=continued,domain=[{'name':n,'n':sum(o['domain']==n for o in test),'mean_accept':1-float(np.mean([o['tv'] for o in test if o['domain']==n])),'mean_delta2':float(np.mean([o['delta2'] for o in test if o['domain']==n]))} for n in DOM]))
  print(json.dumps(rows[-1]),flush=True)
res=dict(environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.cuda.is_available(),threads=torch.get_num_threads(),source=str(root)),manifest=manifest,corpus={n:[len(a) for a in v] for n,v in corp.items()},rows=rows)
pathlib.Path(__file__).with_name('results.json').write_text(json.dumps(res,indent=2)+'\n')
