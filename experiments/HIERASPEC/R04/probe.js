// HIERASPEC/R04: fixed-budget target sample tape; JS CPU, no dependencies
const seeds=[11,29,73],sigmas=[.1,.6,1.5],ks=[0,1,2,4,8],V=64,N=100000;
let st=0;function u(){st=(Math.imul(st,1664525)+1013904223)>>>0;return (st+.5)/4294967296}
function normal(){return Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u())}
const sum=a=>a.reduce((s,x)=>s+x,0);
const soft=a=>{let m=Math.max(...a),v=a.map(x=>Math.exp(x-m)),z=sum(v);return v.map(x=>x/z)};
function cdf(a){let s=0;return a.map(x=>s+=x)}
function draw(c){let x=u()*c[c.length-1],l=0,r=c.length-1;while(l<r){let m=(l+r)>>1;if(x<c[m])r=m;else l=m+1}return l}
function run(){let rows=[];
for(let seed of seeds)for(let sigma of sigmas){
st=seed;let z=Array.from({length:V},normal),p=soft(z),q=soft(z.map(x=>x+sigma*normal())),P=cdf(p),Q=cdf(q),R=cdf(p.map((x,i)=>Math.max(0,x-q[i])));
let delta=1-sum(p.map((x,i)=>Math.min(x,q[i])));
let gumbel=sum(p.map((x,i)=>1/sum(p.map((y,j)=>Math.max(y/x,q[j]/q[i])))));
let methods=[];
for(let k of ks){st=seed+Math.round(sigma*100)*101+k*1009;let counts=Array(V).fill(0),accepted=0,fallback=0,tries=0;
for(let n=0;n<N;n++){let x=draw(Q),y=x;if(u()<Math.min(1,p[x]/q[x]))accepted++;
else{let success=false;for(let j=0;j<k;j++){let z=draw(P);tries++;if(u()<Math.max(0,1-q[z]/p[z])){y=z;success=true;break}}
if(!success){fallback++;y=draw(R)}}counts[y]++}
methods.push({k,accepted,fallback,tries,counts,empiricalTV:sum(counts.map((x,i)=>Math.abs(x/N-p[i])))/2,theoryFallback:delta*(1-delta)**k,theoryTries:1-(1-delta)**k})}
rows.push({seed,sigma,delta,gumbel,maximal:1-delta,methods})}
const merge=[];for(let seed of seeds){st=seed;const v=151936,k=4;let z=Array.from({length:v},normal),g=Array.from({length:k},()=>Array.from({length:v},()=>-Math.log(-Math.log(u()))));let x=12345;
function part(a,b){let m=-Infinity,s=0,best=Array(k).fill(-Infinity),idx=Array(k).fill(-1);for(let i=a;i<b;i++){let old=m;m=Math.max(m,z[i]);s=s*Math.exp(old-m)+Math.exp(z[i]-m);for(let j=0;j<k;j++){let t=z[i]+g[j][i];if(t>best[j]){best[j]=t;idx[j]=i}}}return {m,s,best,idx,xlog:x>=a&&x<b?z[x]:null}}
let dense=part(0,v);
for(let n of [2,4,8]){let a=Array.from({length:n},(_,i)=>part(Math.floor(i*v/n),Math.floor((i+1)*v/n))),m=Math.max(...a.map(t=>t.m)),s=sum(a.map(t=>t.s*Math.exp(t.m-m))),winners=Array.from({length:k},(_,j)=>a.reduce((b,t)=>t.best[j]>b.best[j]?t:b).idx[j]);merge.push({seed,n,lseError:Math.abs(m+Math.log(s)-dense.m-Math.log(dense.s)),sameWinners:JSON.stringify(winners)===JSON.stringify(dense.idx),candidateMatch:a.find(t=>t.xlog!==null).xlog===dense.xlog})}}
return {environment:"JS V8 CPU synthetic categorical; no model weights, GPU, network, wall time",V,N,seeds,sigmas,ks,rows,merge}}
console.log(JSON.stringify(run(),null,2));
