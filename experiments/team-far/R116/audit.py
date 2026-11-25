import json, itertools, hashlib, pathlib
P=pathlib.Path('/mnt/data/r112/results.json')
rows=json.loads(P.read_text()); cal=[r for r in rows if r['phase']=='cal']; test=[r for r in rows if r['phase']=='test']; arms=sorted(rows[0]['arms']); fixed=[k for k in arms if len(set(k.strip('()').split(', ')))==1]; assert len(rows)==72 and len(arms)==12 and len(fixed)==5
assert all(len({tuple(v['ids']) for v in r['arms'].values()})==1 for r in rows)
scenarios={'local':([0]*4,float('inf')),'regional':([2,2,20,20],2_500_000),'heterogeneous':([5,20,40,80],2_500_000),'slow_link':([5,20,40,80],100_000),'very_slow_link':([5,20,40,80],50_000)}
results=[]
for name,(rtt,bw) in scenarios.items():
 def cost(r,k):
  a=r['arms'][k];return a['seconds']+a['deep_calls']*rtt[r['client']]/1000+a['bytes_proxy']/bw
 def choose(group,allowed):return min(allowed,key=lambda k:sum(cost(r,k) for r in group))
 def score_cal(groups,allowed):
  choices={}; total=0
  for group in groups:
   a=[r for r in cal if (r['seed'],r['client']) in group]; b=[r for r in test if (r['seed'],r['client']) in group]; k=choose(a,allowed); total+=sum(cost(r,k) for r in b); choices[str(group)]=k
  return total,choices
 seeds=sorted(set(r['seed'] for r in rows)); globalg=[[(s,c) for c in range(4)] for s in seeds]; regionalg=[[(s,c) for c in cs] for s in seeds for cs in [(0,1),(2,3)]]; localg=[[(s,c)] for s in seeds for c in range(4)]
 g,gc=score_cal(globalg,fixed); rg,rc=score_cal(regionalg,arms); lo,lc=score_cal(localg,arms)
 fixed12=sum(cost(r,'(12, 12)') for r in test); best_test_fixed=min(sum(cost(r,k) for r in test) for k in fixed)
 oracle=sum(min(cost(r,k) for k in arms) for r in test)
 # Optimistic per-metric minima may come from different policies. Retain zero-bound on new compute if changing arch.
 separate=sum(min(r['arms'][k]['seconds'] for k in arms)+min(r['arms'][k]['deep_calls'] for k in arms)*rtt[r['client']]/1000+min(r['arms'][k]['bytes_proxy'] for k in arms)/bw for r in test)
 results.append(dict(scenario=name,global_cal=g,regional_cal=rg,local_cal=lo,fixed12=fixed12,best_test_fixed=best_test_fixed,oracle=oracle,separate_metric_lower=separate,speedup_vs_fixed12=fixed12/oracle,speedup_vs_global_cal=g/oracle,relaxation_speedup_vs_fixed12=fixed12/separate,global_choices=gc,region_choices=rc,local_choices=lc))
# Baseline timing comparisons use only historical CPU observations, no new forward.
r110=json.load(open('/mnt/data/r110/kv_results.json')); totals={k:sum(r[k]['seconds'] for r in r110) for k in ('dense_sd','sparse_ar','sparse_sd')}; identical=[r for r in r110 if r['dense_sd']['ids']==r['sparse_sd']['ids']]
# Arithmetic necessary reduction under serialized normalized budget T'=1-f*s+h.
thresholds=[dict(f=f,overhead=h,req_for_1_5=(1-1/1.5+h)/f,req_for_2=(.5+h)/f) for f,h in itertools.product([.25,.4,.5,.6,.7,.8],[0,.03])]
out=dict(source_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),n_rows=len(rows),test_rows=len(test),arms=arms,results=results,r110=dict(n=len(r110),same_dense_sparse_ids=len(identical),totals=totals),thresholds=thresholds)
pathlib.Path('/mnt/data/r116/results.json').write_text(json.dumps(out,indent=2));print(json.dumps([{k:v for k,v in a.items() if not k.endswith('choices')} for a in results],indent=2));print(totals,len(identical))
