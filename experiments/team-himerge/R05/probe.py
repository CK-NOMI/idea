import numpy as np, scipy, json, time, platform
from scipy.optimize import minimize
from scipy.special import expit
rows=[]
for seed in [11,29,73]:
 for drift in [.25,1.,2.]:
  for n in [8,32,128]:
   rng=np.random.default_rng(seed);d=32;R=4
   w=rng.normal(size=d);w/=np.linalg.norm(w);w*=2
   H=rng.normal(size=(R,n,d));H/=np.linalg.norm(H,axis=-1,keepdims=True)
   A=H*np.where(H@w>=0,1,-1)[...,None];AA=A.reshape(-1,d);m=AA@w;b=-.95*m
   X=rng.normal(size=(R,256,d));X/=np.linalg.norm(X,axis=-1,keepdims=True)
   teacher=w+drift*rng.normal(size=(R,d))/np.sqrt(d)*4
   y=expit(np.einsum('rnd,rd->rn',X,teacher))
   def loss(v):
    z=X@v;return float(np.mean(np.logaddexp(0,z)-y*z))
   g=np.mean((expit(X@w)-y)[...,None]*X,axis=1);updates=-32*g;u=updates.mean(0)
   def project(u,A,b):
    start=time.perf_counter();r=minimize(lambda v:.5*np.sum((v-u)**2),np.zeros(d),jac=lambda v:v-u,constraints=[{'type':'ineq','fun':lambda v:A@v-b,'jac':lambda v:A}],method='SLSQP',options={'ftol':1e-11,'maxiter':200});assert r.success,r.message
    return r.x,1000*(time.perf_counter()-start)
   loc=np.mean([project(updates[i],A[i],-.95*(A[i]@w))[0] for i in range(R)],axis=0)
   val=AA@u;mask=val<0;scale=min(1,float(np.min(b[mask]/val[mask])) if mask.any() else 1);ray=u*scale
   qp,ms=project(u,AA,b);null=u-np.linalg.pinv(AA)@(AA@u)
   L0=loss(w);base=L0-loss(w+u)
   metrics={}
   for name,v in [('plain',u),('local',loc),('ray',ray),('globalQP',qp),('null',null)]:
    margins=AA@(w+v);p0=expit(H@w);p1=expit(H@(w+v));metrics[name]={'flip':float(np.mean(margins<=0)),'loss_gain':L0-loss(w+v),'gain_ratio':(L0-loss(w+v))/base,'random_mismatch':float(np.mean(abs(p1-p0))),'update_norm':float(np.linalg.norm(v)),'min_constraint_slack':float(np.min(AA@v-b))}
   rows.append({'seed':seed,'drift':drift,'anchors_per_region':n,'qp_ms':ms,'base_gain':base,'rank':int(np.linalg.matrix_rank(AA)),'metrics':metrics})
print(json.dumps({'env':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'rows':rows}))
