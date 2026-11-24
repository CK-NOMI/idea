// node experiments/team-hiercouple/R03/probe.js
// Synthetic categorical simulation. No GPU/wall-time measurement.
const N=12000,V=32,n=5,C=5/3,seeds=[11,29,47];
let state;const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296};
const normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
const sum=a=>a.reduce((s,x)=>s+x,0);
const norm=a=>{const z=sum(a);return a.map(x=>x/z)};
const soft=a=>{let m=Math.max(...a);return norm(a.map(x=>Math.exp(x-m)))};
function draw(p){let x=u();for(let i=0;i<p.length;i++){x-=p[i];if(x<0)return i}return p.length-1}
function med(a){return a.slice().sort((x,y)=>x-y)[2]}
function aTheory(t,k){if(t===0)return 0;if(Math.abs(t-1)<1e-10)return k/(k+1);if(t>1)return (1-t**(-k))/(1-t**(-k-1));return t*(1-t**k)/(1-t**(k+1))}
let rows=[];
for(const seed of seeds)for(const [label,sigma] of [['matched',.5],['low',.15],['mid',.6],['high',1.5]]){
 state=seed;const z=Array.from({length:V},normal),P=Array.from({length:4},()=>soft(z.map(x=>x+sigma*normal())));
 // One fixed contaminated but valid probability vector. Not an adaptive Byzantine protocol.
 P.push(soft(z.map((x,i)=>i===0?8:-2)));
 const m=z.map((_,i)=>med(P.map(p=>p[i]))),Z=sum(m),p=norm(m),mean=z.map((_,i)=>sum(P.map(p=>p[i]))/n);
 const q=label==='matched'?p:soft(z),maxAccept=sum(p.map((x,i)=>Math.min(x,q[i])));
 for(const k of [0,-1,1,4,8]){
 let hist=Array(V).fill(0),acc=0,raw=0,unique=0,coins=0,walkSteps=0,repairs=0,mixtureDraws=0;
 for(let t=0;t<N;t++){
  const seen=new Set();
  function query(v){raw++;seen.add(v);return m[v]}
  function envelope(){mixtureDraws++;const v=draw(P[Math.floor(u()*n)]);return [v,u()<query(v)/(C*mean[v])]}
  function target(){while(true){const [v,ok]=envelope();if(ok)return v}}
  function step(v){const w=query(v);while(true){if(u()<w/(w+C*q[v]))return 1;coins++;const [,ok]=envelope();if(ok)return -1}}
  function walk(v,start){let j=start;while(j>0&&j<k+1){walkSteps++;j+=step(v)}return j===k+1}
  let y;
  if(k===0)y=target();
  else if(k===-1){const x=draw(q);for(let v=0;v<V;v++)query(v);
   if(u()<Math.min(1,p[x]/q[x])){y=x;acc++}else{y=draw(norm(p.map((a,i)=>Math.max(0,a-q[i]))));repairs++}}
  else {const x=draw(q);if(walk(x,k)){y=x;acc++}else{do{y=target();repairs++}while(!walk(y,1))}}
  hist[y]++;unique+=seen.size;
 }
 const expected=k===0?null:k===-1?maxAccept:sum(q.map((x,i)=>x*aTheory(p[i]/x,k)));
 rows.push({seed,label,sigma,k,Z,maxAccept,expected,accept:k===0?null:acc/N,TV:sum(hist.map((v,i)=>Math.abs(v/N-p[i])))/2,rawQueries:raw/N,uniqueQueries:unique/N,normalizerCoinCalls:coins/N,walkSteps:walkSteps/N,repairProposals:repairs/N,mixtureDraws:mixtureDraws/N,hist});
 }
}
console.log(JSON.stringify({N,V,n,C,seeds,rows}));