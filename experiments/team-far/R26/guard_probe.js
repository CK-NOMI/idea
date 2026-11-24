// node guard_probe.js weights.json checkpoint.json input.txt
// R26 finite-step calibration gate; pooled regional stats, no actual network.
const fs=require('node:fs');
function probe(raws,names){
const V=27,d=16,H=4,k=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {h:seq.at(-1),s}}}}
const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],rows=[],audit={finiteGuardViolations:0,projectedFirstOrderViolations:0,exactFlatEquivalent:true},masks=[];
for(let m=0;m<1<<d;m++)if(m.toString(2).replace(/0/g,"").length===8)masks.push(Array.from({length:d},(_,i)=>i).filter(i=>m&(1<<i)));
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]),W=m.W,mom=p=>p.reduce((a,v,i)=>add(a,scale(W[i],v)),zero(d));
function data(name){return [2,4].map(j=>{let pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],{h,s}=m.feature(pre);return {pl:mat(W,h),ql:mat(W,s),y:name.charCodeAt(j)-97}})}
function stats(x,shift,grad){let p=sm(add(x.pl,shift)),q=sm(add(x.ql,shift)),D=0,t=zero(d),mp=grad?mom(p):null,mq=grad?mom(q):null;for(let i=0;i<V;i++)if(p[i]>q[i]){D+=p[i]-q[i];if(grad)t=add(t,sub(scale(sub(W[i],mp),p[i]),scale(sub(W[i],mq),q[i])))}return {L:-Math.log(p[x.y]),D,g:grad?sub(mp,W[x.y]):null,t:grad?t:null}}
let calls=0;
function ev(gs,theta,grad=false){let shift=mat(W,theta);return gs.map(xs=>{let a=xs.map(x=>stats(x,shift,grad));calls+=a.length;return {L:mean(a.map(x=>x.L)),D:mean(a.map(x=>x.D)),g:grad?scale(a.map(x=>x.g).reduce(add),1/a.length):null,t:grad?scale(a.map(x=>x.t).reduce(add),1/a.length):null}})}
const avg=(a,k)=>mean(a.map(x=>x[k])),av=(a,k)=>scale(a.map(x=>x[k]).reduce(add),1/a.length);
for(let seed of [11,29,73]){let groups=Array.from({length:4},(_,r)=>words.filter(x=>x.charCodeAt(0)%4===r).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,48)),train=groups.map(g=>g.slice(0,36).flatMap(data)),cal=train.map(g=>g.slice(0,24)),test=groups.map(g=>g.slice(36).flatMap(data));
for(let method of ["CE","mask","project","guard-global","guard-region"]){calls=0;let theta=zero(d),initial=ev(test,theta),curve=[],accepted=0,viol=0,trainViol=0,trial=0;for(let round=0;round<12;round++){
let old=ev(cal,theta,true),g=av(old,"g"),t=av(old,"t"),top=Array.from({length:d},(_,i)=>i).sort((i,j)=>g[j]**2-g[i]**2).slice(0,8),mask=top;
if(method==="mask"){let best=-Infinity;for(let a of masks){let q=sum(a.map(j=>g[j]*t[j])),u=sum(a.map(j=>g[j]**2));if(q>=0&&u>best){best=u;mask=a}}}
let gg=g.map((x,i)=>mask.includes(i)?x:0),tt=t.map((x,i)=>mask.includes(i)?x:0),direction=scale(gg,-1),norm=dot(tt,tt);
if(["project","guard-global","guard-region"].includes(method)&&dot(gg,tt)<0&&norm>1e-20){direction=add(direction,scale(tt,dot(gg,tt)/norm));if(dot(direction,tt)>1e-10||dot(direction,g)>1e-10)audit.projectedFirstOrderViolations++}
let candidates=[direction],steps=[.6];
if(method.startsWith("guard")){let ratio=Math.sqrt(dot(gg,gg)/(norm+1e-20));candidates=[direction,...[.05,.2,.5,1].map(l=>sub(direction,scale(tt,l*ratio)))];steps=[.6,.3,.15]}
let best=null;for(let v of candidates)for(let eta of steps){let x=add(theta,scale(v,eta)),now=ev(cal,x);trial++;let ok=method==="guard-global"?avg(now,"D")<=avg(old,"D")+1e-12:method==="guard-region"?now.every((z,i)=>z.D<=old[i].D+1e-12):true;ok=ok&&avg(now,"L")<avg(old,"L");if(ok&&(!best||avg(now,"L")<best.L))best={x,L:avg(now,"L"),now}}
if(best){theta=best.x;accepted++;if(avg(best.now,"D")>avg(old,"D")+1e-12)trainViol++;if(method==="guard-global"&&avg(best.now,"D")>avg(old,"D")+1e-12||method==="guard-region"&&best.now.some((z,i)=>z.D>old[i].D+1e-12))audit.finiteGuardViolations++}
let now=ev(test,theta),prev=curve.length?curve.at(-1):{L:avg(initial,"L"),D:avg(initial,"D")};if(avg(now,"D")>prev.D+1e-12)viol++;curve.push({L:avg(now,"L"),D:avg(now,"D"),region:now.map(x=>x.D)})}
rows.push({wi,seed,method,initial:{L:avg(initial,"L"),D:avg(initial,"D"),region:initial.map(x=>x.D)},final:curve.at(-1),accepted,trial,calls,viol,trainViol,curve})}
}}
let summary=[];for(let method of ["CE","mask","project","guard-global","guard-region"]){let a=rows.filter(x=>x.method===method);summary.push({method,nll:mean(a.map(x=>x.final.L)),overlap:1-mean(a.map(x=>x.final.D)),nllImprovement:mean(a.map(x=>x.initial.L-x.final.L)),overlapImprovement:mean(a.map(x=>x.initial.D-x.final.D)),accepted:sum(a.map(x=>x.accepted)),trials:sum(a.map(x=>x.trial)),calls:sum(a.map(x=>x.calls)),testViolationRounds:sum(a.map(x=>x.viol)),trainViolationRounds:sum(a.map(x=>x.trainViol)),allRegionImprovement:a.filter(x=>x.final.region.every((v,i)=>v<=x.initial.region[i])).length})}
return {rows,summary,audit}}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
