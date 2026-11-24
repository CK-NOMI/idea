function run(rows,M=1024){
let seed=123456789;const u=()=>((seed=Math.imul(seed,1664525)+1013904223>>>0)+.5)/4294967296;
const sum=a=>a.reduce((s,x)=>s+x,0);
function draw(c){let z=u()*c[c.length-1],l=0,h=c.length-1;while(l<h){let m=(l+h)>>1;if(z<c[m])h=m;else l=m+1}return l}
function cdf(a){let s=0;return a.map(x=>s+=x)}
function make(p,q){let K=p.length,a=q.map((v,i)=>v?Math.min(1,p[i]/v):1),lp=p.map((v,i)=>Math.log(v)-Math.log(q[i])),lr=p.map((v,i)=>Math.log(Math.min(1,q[i]/v))),cr=cdf(p.map((v,i)=>Math.max(0,v-q[i]))),cq=cdf(q);
return {sample(N){let x=Array.from({length:N},()=>draw(cq)),y=x.slice(),logratio=sum(y.map(i=>lp[i])),accept=Math.exp(Math.min(0,logratio)),steps=0;
if(u()>=accept){while(true){let pref=1,weights=y.map(i=>{let w=pref*(1-a[i]);pref*=a[i];return w});let first=draw(cdf(weights));if(!(sum(weights)>0))throw Error("invalid outgoing");for(let j=first;j<N;j++)if(j===first||u()>=a[y[j]])y[j]=draw(cr);steps++;if(steps>N)throw Error("cycle");logratio=sum(y.map(i=>lp[i]));let logDoverP=sum(y.map(i=>lr[i]));let stop=logratio>0?(-Math.expm1(-logratio))/(-Math.expm1(logDoverP)):0;if(u()<Math.min(1,stop))break}}
let seen=new Set(x),added=new Set(y.filter(i=>!seen.has(i)));return {x,y,steps,accept,changed:y.reduce((s,v,j)=>s+(v!==x[j]),0),missing:added.size}}}}
let groups={},maxSteps=0,violations=0,draws=0;
for(let r of rows){let {p,q,K,N}=r,m=make(p,q),d=sum(p.map((v,i)=>Math.max(0,v-q[i]))),b=(1-d)**N,lb=sum(p.map((v,i)=>Math.max(0,(1-q[i])**N-(1-v)**N)));let g=groups[K+"_"+N]??=( {K,N,n:0,eq:0,overlapEstimate:0,baseline:0,steps:0,stepsTheory:0,moves:0,movesTheory:0,missing:0,missingTheory:0,pairFirstZ2:0,firstMeanZ2:0});g.n++;
let f=0,ff=0,eq=0,est=0,st=0,mv=0,mi=0;for(let t=0;t<M;t++){let z=m.sample(N);eq+=+(z.changed===0);est+=z.accept;st+=z.steps;mv+=z.changed;mi+=z.missing;f+=+(z.y[0]===0);ff+=+(z.y[0]===0&&z.y[1]===0);maxSteps=Math.max(maxSteps,z.steps);for(let j=0;j<N;j++)if(z.x[j]!==z.y[j]&&!(q[z.x[j]]>p[z.x[j]]&&p[z.y[j]]>q[z.y[j]]))violations++;draws++}g.eq+=eq/M;g.overlapEstimate+=est/M;g.baseline+=b;g.steps+=st/M;g.stepsTheory+=1-b;g.moves+=mv/M;g.movesTheory+=N*d;g.missing+=mi/M;g.missingTheory+=lb;let h=p[0],h2=h*h;g.firstMeanZ2+=(f-M*h)**2/Math.max(1e-12,M*h*(1-h));g.pairFirstZ2+=(ff-M*h2)**2/Math.max(1e-12,M*h2*(1-h2))}
for(let g of Object.values(groups))for(let k of Object.keys(g))if(!["K","N","n"].includes(k))g[k]/=g.n;return {draws,maxSteps,violations,groups:Object.values(groups)}
}
if(require.main===module){const fs=require("fs");console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(process.argv[2],"utf8")).rows,1024),null,2))}
