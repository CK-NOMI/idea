// team-far/R01; node probe.js > results.json
// flatDifference=0 is an algebraic identity for additive counts, not measured latency or a separate simulation.
function probe(){
const seeds=[11,29,73], sigmas=[.25,1,2], lengths=[4,6], epses=[0,.05,.2];
const sum=a=>a.reduce((s,x)=>s+x,0),tv=(a,b)=>sum(a.map((x,i)=>Math.abs(x-b[i])))/2;
let state=0;const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296};
const normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u()),logistic=x=>1/(1+Math.exp(-x));
const fit=counts=>counts[0].map((_,i)=>(sum(counts.map(c=>c[i][1]))+.5)/(sum(counts.map(c=>sum(c[i])))+1));
const prob=(p,b,h)=>b?p[h]:1-p[h];
const lcp=(a,b)=>{let k=0;while(k<a.length&&a[k]===b[k])k++;return k};
let rows=[];
for(const sigma of sigmas)for(const seed of seeds){
 state=seed;const base=[normal(),normal()];let cs=[];
 for(let i=0;i<4;i++){let c=Array.from({length:2},()=>[0,0]);let z=base.map(v=>logistic(v+sigma*normal()));for(let h=0;h<2;h++)for(let j=0;j<256;j++)c[h][Number(u()<z[h])]++;cs.push(c)}
 const q=fit(cs),p=fit(cs.slice(1));
 for(const L of lengths){
 const paths=Array.from({length:2**L},(_,i)=>i.toString(2).padStart(L,"0").split("").map(Number));
 const mass=(p,x)=>x.reduce((m,b,t)=>m*prob(p,b,t?x[t-1]:0),1);
 const Q=paths.map(x=>mass(q,x)),P=paths.map(x=>mass(p,x));
 const K=paths.map(x=>paths.map(y=>{let m=1,shared=true;for(let t=0;t<L;t++){const h=t?y[t-1]:0,px=prob(p,x[t],h),qx=prob(q,x[t],h);if(shared){const a=Math.min(1,px/qx);m*=y[t]===x[t]?a:1-a;shared=y[t]===x[t]}else m*=prob(p,y[t],h)}return m}));
 const worst=Math.max(...K.map(k=>tv(k,P)));
 const metric=mat=>{let marg=P.map((_,j)=>sum(Q.map((v,i)=>v*mat[i][j]))),cond=mat.map(k=>tv(k,P));return {prefix:sum(Q.map((v,i)=>v*sum(mat[i].map((w,j)=>w*lcp(paths[i],paths[j]))))),full:sum(Q.map((v,i)=>v*mat[i][i])),marginalTV:tv(marg,P),jointTV:sum(Q.map((v,i)=>v*cond[i])),worstConditionalTV:Math.max(...cond),maxRowError:Math.max(...mat.map(k=>Math.abs(sum(k)-1)))}};
 const fresh=paths.map(()=>P.slice()),upper=eps=>paths.map((x,i)=>{let k=P.slice(),remain=Math.min(eps,1-k[i]);for(let j of paths.map((_,j)=>j).sort((a,b)=>lcp(x,paths[a])-lcp(x,paths[b]))){if(j===i)continue;let d=Math.min(k[j],remain);k[j]-=d;k[i]+=d;remain-=d;if(remain<1e-14)break}return k});
 const mix=eps=>{const a=worst?Math.min(1,eps/worst):1;return K.map(k=>k.map((v,j)=>(1-a)*P[j]+a*v))};
 rows.push({sigma,seed,L,q,p,flatDifference:0,fresh:metric(fresh),standard:metric(K),bounded:epses.map(e=>({eps:e,mix:metric(mix(e)),oracle:metric(upper(e))}))});
 }}
let summary=[];for(const sigma of sigmas)for(const L of lengths){let rr=rows.filter(r=>r.sigma===sigma&&r.L===L),avg=f=>sum(rr.map(f))/rr.length;summary.push({sigma,L,fresh:avg(r=>r.fresh.prefix),standard:avg(r=>r.standard.prefix),jointTV:avg(r=>r.standard.jointTV),worst:avg(r=>r.standard.worstConditionalTV),mix05:avg(r=>r.bounded[1].mix.prefix),mix20:avg(r=>r.bounded[2].mix.prefix),oracle05:avg(r=>r.bounded[1].oracle.prefix),oracle05MarginalTV:avg(r=>r.bounded[1].oracle.marginalTV),oracle20:avg(r=>r.bounded[2].oracle.prefix),oracle20MarginalTV:avg(r=>r.bounded[2].oracle.marginalTV)})}
return {environment:"V8 CPU; synthetic Markov count training; exact finite path enumeration; no model GPU or measured time",seeds,sigmas,lengths,epses,rows,summary,maxMarginalTV:Math.max(...rows.flatMap(r=>[r.standard.marginalTV,r.fresh.marginalTV,...r.bounded.map(b=>b.mix.marginalTV)])),maxRowError:Math.max(...rows.flatMap(r=>[r.standard.maxRowError,r.fresh.maxRowError,...r.bounded.flatMap(b=>[b.mix.maxRowError,b.oracle.maxRowError])]))}
}
console.log(JSON.stringify(probe(),null,2));
