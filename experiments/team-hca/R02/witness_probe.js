// Finite suffix domain; exact lazy rank sampler with optional old witness.
// Run: node experiments/team-hca/R02/witness_probe.js
// Synthetic CPU experiment, not HFL or GPU wall-clock.
let seed=0; function u(){seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296}
const N=4000,M=4,B=64,V=16;
let results=[];
for(const density of [.2,.5,.8])for(const drift of [.02,.15])for(const s of [11,29,73]){
 seed=s;let queries=0,fp=0,checked=0,fallback=0,bitsBase=0,bitsCert=0,hit=0,dead=0;
 for(let t=0;t<N;t++){
  let old=[],now=[],viable=[],witness=[];
  for(let v=0;v<V;v++){
   let a=Array.from({length:M},()=>Array.from({length:B},()=>u()<density));
   let b=a.map(row=>row.map(x=>u()<drift?!x:x));
   old.push(a);now.push(b);
   let w=Array.from({length:B},(_,i)=>i).find(i=>a.every(r=>r[i]));witness.push(w??-1);
   let good=Array.from({length:B},(_,i)=>i).some(i=>b.every(r=>r[i]));viable.push(good);
   if(b.every(r=>r.some(Boolean))){checked++;if(!good)fp++}
  }
  let rank=Array.from({length:V},(_,v)=>[v,u()]).sort((a,b)=>a[1]-b[1]).map(x=>x[0]);
  let target=rank.find(v=>viable[v]); if(target===undefined)dead++;
  let got;
  for(let v of rank){
   queries++;bitsBase+=M*B;
   if(witness[v]>=0){
    bitsCert+=M*7; // 6-bit witness ID down, 1-bit response up per node; token/state ID common omitted
    if(now[v].every(r=>r[witness[v]])){hit++;got=v;break}
   }
   fallback++;bitsCert+=M*B;
   if(viable[v]){got=v;break}
  }
  if(got!==target)throw Error("mismatch")
 }
 results.push({density,drift,s,localFalse:fp/checked,hit:hit/queries,fallback:fallback/queries,ratio:bitsCert/bitsBase,dead:dead/N,queries:queries/N})
}
let grouped=[];for(const density of [.2,.5,.8])for(const drift of [.02,.15]){let r=results.filter(x=>x.density===density&&x.drift===drift);let g={density,drift};for(let k of ["localFalse","hit","fallback","ratio","dead","queries"])g[k]=r.reduce((a,x)=>a+x[k],0)/r.length;grouped.push(g)}
console.log(JSON.stringify({results,grouped},null,2));
