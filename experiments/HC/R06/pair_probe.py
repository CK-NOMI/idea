# Execute original source with a small injection to inspect the same held-out gradients.
# Does not modify original probe; executes its training/evaluation, replaces continuation with pair audit only.
from pathlib import Path
p=Path(__file__).with_name('probe.py')
s=p.read_text()
a=s.index('  # Logical HFL continuation:')
b=s.index('  rows.append(dict(d=d,L=L',a)
inject='''  def paired(pairs):
   n=len(test);return float(sum(np.sum(((gp[i]-gq[i])-(gp[j]-gq[j]))**2) for i,j in pairs)/n**2)
  n=len(test);delta=gp-gq;ind=float(np.sum(delta**2)/n**2)
  same=[(i,i+1) for i in range(0,n,2)]
  cross=[(i,16+i) for i in range(8)]+[(8+i,32+i) for i in range(8)]+[(24+i,40+i) for i in range(8)]
  assert len({i for pair in cross for i in pair})==n
  pool=set(range(n));greedy=[];gram=delta@delta.T
  while pool:
   i,j=max(((i,j) for i in pool for j in pool if i<j),key=lambda ij:gram[ij]);greedy.append((i,j));pool.remove(i);pool.remove(j)
  continued=[]
  pair_result=dict(independent=ind,same=paired(same),cross=paired(cross),oracle_greedy=paired(greedy),same_ratio=paired(same)/ind,cross_ratio=paired(cross)/ind,oracle_greedy_ratio=paired(greedy)/ind,n=n)
'''
s=s[:a]+inject+s[b:];s=s.replace('continued=continued,domain=','continued=continued,pair_result=pair_result,domain=')
s=s.replace("with_name('results.json')","with_name('pair_results.json')")
exec(compile(s,str(p),'exec'))
