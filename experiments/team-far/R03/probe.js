// Forward adapted from CK-NOMI/idea experiments/team-hca/R09/probe.js; originally team-hieracert/R11. node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
 const V=27,d=16,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((x,i)=>x+b[i]),rms=x=>{let s=Math.sqrt(dot(x,x)/d+1e-5);return x.map(t=>t/s)},sm=x=>{let m=Math.max(...x),a=x.map(v=>Math.exp(v-m)),s=sum(a);return a.map(v=>v/s)};
 function model(raw){const vals=Array.from({length:7264},(_,i)=>raw[i]);if(vals.some(x=>!Number.isFinite(x)))throw Error("weights");let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=vals.slice(off,off+m);off+=m;return r});const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));return h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),first;for(let l=0;l<2;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!l)first=seq.at(-1)}let deep=seq.at(-1);return {p:sm(mat(W,deep)),q:sm(mat(W,first)),h2:dot(deep,deep)}}}
 const docs=[...new Set(names.split(/\r?\n/).filter(s=>/^[a-z]{6,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>Math.min(3,Math.floor((s.charCodeAt(0)-97)/7))===c));const rows=[];
 function hash(s,k){let h=k;for(let c of s)h=Math.imul(h^c.charCodeAt(0),16777619);return h>>>0}
 const bin=p=>p<.005?0:p<.02?1:p<.05?2:p<.1?3:p<.25?4:5;
 function makeData(f,seed){let train=[],test=[];for(let c=0;c<4;c++){const b=buckets[c].slice().sort((a,b)=>hash(a,seed)-hash(b,seed)||a.localeCompare(b));for(let j=0;j<96;j++){let name=b[j],h=[26,name.charCodeAt(0)-97,name.charCodeAt(1)-97];for(let t=2;t<6;t++){let {p,q,h2}=f(h),g=name.charCodeAt(t)-97,p2=dot(p,p);for(let y=0;y<V;y++){let w=p[y],a=Math.min(1,q[y]/w),s=h2*(1-2*w+p2);(j<64?train:test).push({w,a,s,r:Number(y===g),c,pb:bin(w),qb:bin(q[y])})}h.push(g)}}}return {train,test}}
 function calc(train,test,budget,scope){
 const keys=[x=>`${scope==='region'?x.c>>1:scope==='client'?x.c:0}/${x.pb}`,x=>`${scope==='region'?x.c>>1:scope==='client'?x.c:0}/${x.pb}/${x.qb}`,x=>`${scope==='region'?x.c>>1:scope==='client'?x.c:0}/${x.pb}/${x.qb}`];
 function policy(mode){let stats=new Map(),tot=[0,0,0,0];function addS(k,w,x){let z=stats.get(k)||[0,0,0,0];z[0]+=w;z[1]+=w*x.s;z[2]+=w*x.s*x.r;z[3]+=w*x.s*x.r*x.r;stats.set(k,z);for(let j=0;j<4;j++)tot[j]+=[w,w*x.s,w*x.s*x.r,w*x.s*x.r*x.r][j]}
 function atoms(x){let k=keys[mode](x);return mode===2?[[k+"/A",x.a],[k+"/R",1-x.a]]:[[k,1]]}
 for(let x of train)for(let [k,a] of atoms(x))if(a>0)addS(k,x.w*a,x);
 function pred(k){let z=stats.get(k)||[0,0,0,0],prior=2;let u=z.map((v,i)=>v+prior*tot[i]/tot[0]),m=u[2]/u[1],e=Math.max(1e-10,(u[3]-2*m*u[2]+m*m*u[1])/u[0]);return {m,e}}
 const cache=new Map();function pp(k){if(!cache.has(k))cache.set(k,pred(k));return cache.get(k)}
 let a=test.map(x=>atoms(x).map(([k,w])=>({...pp(k),w}))),N=sum(test.map(x=>x.w));let lo=0,hi=1000;
 const pi=(e,l)=>Math.max(.02,Math.min(.98,Math.sqrt(e/Math.max(l,1e-20))));
 for(let i=0;i<60;i++){let l=(lo+hi)/2,v=sum(test.map((x,j)=>x.w*sum(a[j].map(k=>k.w*pi(k.e,l)))))/N;if(v>budget)lo=l;else hi=l}
 let varr=0,vrb=0,qry=0,maxMean=0,maxIneq=0,traceVariance=0;
 for(let j=0;j<test.length;j++){let x=test[j],z=a[j].map(k=>({...k,pi:pi(k.e,hi)})),p=sum(z.map(k=>k.w*k.pi)),u=sum(z.map(k=>k.w*(1-k.pi)*k.m))/(1-p);let v=sum(z.map(k=>k.w*(1/k.pi-1)*(x.r-k.m)**2))*x.s,b=(1/p-1)*(x.r-u)**2*x.s;varr+=x.w*v;vrb+=x.w*b;qry+=x.w*p;maxIneq=Math.max(maxIneq,b-v);let m0=u,m1=u+(x.r-u)/p;let g0=sum(z.map(k=>k.w*(1-k.pi)*k.m))/(1-p),g1=sum(z.map(k=>k.w*k.pi*(k.m+(x.r-k.m)/k.pi)))/p;maxMean=Math.max(maxMean,Math.abs(g0-m0),Math.abs(g1-m1),Math.abs((1-p)*m0+p*m1-x.r));traceVariance+=x.w*x.s*(x.r-u)**2*(1/budget-1)}
 return {query:qry/N,variance:varr/N,rbVariance:vrb/N,uniformWithRBPrediction:traceVariance/N,maxMean,maxIneq}}
 return [0,1,2].map(policy)
 }
 for(let wi=0;wi<raws.length;wi++){let f=model(raws[wi]);for(let seed of [11,29,73]){let {train,test}=makeData(f,seed);for(let budget of [.1,.25,.5])for(let scope of ["global","region","client"]){rows.push({wi,seed,budget,scope,nTrain:train.length/27,nTest:test.length/27,methods:calc(train,test,budget,scope)})}}}
 let summaries=[];for(let budget of [.1,.25,.5])for(let scope of ["global","region","client"]){let rr=rows.filter(x=>x.budget===budget&&x.scope===scope),avg=f=>sum(rr.map(f))/rr.length;let p=avg(x=>x.methods[0].variance),pq=avg(x=>x.methods[1].variance),sd=avg(x=>x.methods[2].variance),rb=avg(x=>x.methods[2].rbVariance);summaries.push({budget,scope,p,pq,sd,rb,rbVsSD:rb/sd,rbVsPQ:rb/pq,sdVsPQ:sd/pq})}
 return {environment:"V8 CPU FP64, external two checkpoints same 7264-parameter character Transformer, one-step probabilities; exact weighted calibration/test moment enumeration; proxy correct-next-character reward, not expensive labels; no real rollout, training GPU or timing",rows,summaries,maxConditionalError:Math.max(...rows.flatMap(x=>x.methods.map(m=>m.maxMean))),maxViolation:Math.max(...rows.flatMap(x=>x.methods.map(m=>m.maxIneq))),maxBudgetError:Math.max(...rows.flatMap(x=>x.methods.map(m=>Math.abs(m.query-x.budget))))}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
