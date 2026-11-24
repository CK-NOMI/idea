// HC/R02 synthetic diagonal-matrix probe. Node.js, no dependencies.
let state=0; function rng(){state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296}
function norm(){return Math.sqrt(-2*Math.log(rng()))*Math.cos(2*Math.PI*rng())}
const d=16,k=4,r=4,E=4,seeds=[11,23,37],N=100,H=64;
const top=(a,b)=>{let ids=Array.from({length:a.length},(_,i)=>i).sort((i,j)=>Math.abs(a[j])-Math.abs(a[i])||i-j).slice(0,b);return a.map((x,i)=>ids.includes(i)?x:0)};
const avg=a=>a[0].map((_,i)=>a.reduce((s,v)=>s+v[i],0)/a.length);
const soft=a=>{let m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),z=e.reduce((s,x)=>s+x,0);return e.map(x=>x/z)};
const tv=(a,b)=>a.reduce((s,x,i)=>s+Math.abs(x-b[i]),0)/2;
const sq=a=>a.reduce((s,x)=>s+x*x,0);
let out=[];
for(const mode of ["aligned","independent","opposed"])for(const seed of seeds){
state=seed;let sums={};for(const b of [4,8,12,"root"])sums[b]={err:0,alpha:0,rootGap:0};
for(let n=0;n<N;n++){
let signal=Array.from({length:12},(_,i)=>i<4?1+.2*rng():0);
let noise=Array.from({length:2},()=>Array.from({length:12},(_,i)=>i<4?0:2+.5*rng()));
let edges=Array.from({length:E},(_,e)=>signal.map((s,i)=>s+(mode==="aligned"?noise[0][i]:mode==="opposed"?(e%2===0?1:-1)*noise[Math.floor(e/2)][i]:i<4?0:2.5*norm())));
let exact=avg(edges),root=top(exact,k),rooterr=sq(root.map((v,i)=>v-exact[i]));
let methods={root};for(const b of [4,8,12])methods[b]=top(avg(edges.map(a=>top(a,b))),k);
let xs=Array.from({length:H},()=>Array.from({length:d},norm));
for(const b of [4,8,12,"root"]){let a=methods[b],err=sq(a.map((v,i)=>v-exact[i]));sums[b].err+=err/Math.max(sq(exact),1e-12);sums[b].rootGap+=err-rooterr;
for(const x of xs){let p=soft(x.map((v,i)=>v*(i<r?1:exact[i-r]))),q=soft(x.map((v,i)=>v*(i<r?1:a[i-r])));sums[b].alpha+=1-tv(p,q)}}
}
out.push({mode,seed,stats:Object.fromEntries(Object.entries(sums).map(([b,v])=>[b,{nmse:v.err/N,alpha:v.alpha/(N*H),excess:v.rootGap/N}]))})
}
console.log(JSON.stringify(out,null,2));
