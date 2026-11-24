import numpy as np, sys, json
def probe():
    rows=[]
    V=4; L=128; K=4; B=2; requests=20
    for k in [1,2,4,6]:
        for seed in [11,29,73]:
            rng=np.random.default_rng(seed)
            S=V**k
            # fixed shared sufficient-count table, four clients / two edges, 16 observations per state
            base=rng.normal(size=(S,V))
            clients=[]
            for e in range(2):
                edge=rng.normal(scale=.5,size=(S,V))
                for i in range(2):
                    logits=base+edge+rng.normal(scale=.5,size=(S,V))
                    a=np.exp(logits-logits.max(1,keepdims=True)); a/=a.sum(1,keepdims=True)
                    clients.append(np.array([rng.multinomial(16,row) for row in a]))
            edge_counts=[clients[0]+clients[1],clients[2]+clients[3]]
            counts=edge_counts[0]+edge_counts[1]+.5
            p=counts/counts.sum(1,keepdims=True)
            # shorter-suffix internal table
            C=V**(k-1)
            reduced=counts.reshape(V,C,V).sum(0)
            q=reduced/reduced.sum(1,keepdims=True)
            draft=np.argsort(-q,axis=1,kind="stable")[:,:B]
            kl=np.sum(p*np.log(p/q[np.arange(S)%C]),axis=1).mean()
            def step(s,a):return (s*V+int(a))%S
            def candidates(s):
                nodes=[s];level=[s]
                for d in range(K):
                    level=[step(h,a) for h in level for a in draft[h%C]]
                    nodes.extend(level)
                return nodes
            totals=np.zeros(7,dtype=np.int64)
            for req in range(requests):
                init=int(rng.integers(S)); us=rng.random(L)
                def sample(s,t):return int(np.searchsorted(np.cumsum(p[s]),us[t],side="right"))
                s=init; ar=[]; arc=set()
                for t in range(L):
                    arc.add(s); y=sample(s,t); ar.append(y); s=step(s,y)
                s=init;t=0;out=[];seen=set();tree=dag=rounds=0
                while t<L:
                    nodes=candidates(s);tree+=len(nodes);dag+=len(set(nodes));seen.update(nodes);rounds+=1
                    for j in range(K+1):
                        y=sample(s,t);out.append(y);t+=1
                        match=y in draft[s%C];s=step(s,y)
                        if t==L or j==K or not match:break
                assert ar==out and arc<=seen
                totals+=np.array([L,rounds,tree,dag,len(seen),len(arc),len(seen)-len(arc)])
            rows.append(dict(k=k,seed=seed,states=S,kl=float(kl),tokens=int(totals[0]),rounds=int(totals[1]),tree=int(totals[2]),dag=int(totals[3]),sd_cache=int(totals[4]),ar_cache=int(totals[5]),excess=int(totals[6])))
    return rows
rows=probe()
print(sys.version.split()[0], np.__version__)
for k in [1,2,4,6]:
    rr=[r for r in rows if r['k']==k]
    print(k, {key:round(sum(r[key] for r in rr)/3,6) for key in ['kl','tokens','rounds','tree','dag','sd_cache','ar_cache','excess']})
print(json.dumps(rows,separators=(',',':')))
