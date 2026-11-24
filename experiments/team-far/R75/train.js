function train(raws,names,modelSrc){
const V=26,d=16,H=4,sum=a=>a.reduce((s,v)=>s+v,0),avg=a=>sum(a)/a.length,dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),rms=x=>{let a=1/Math.sqrt(dot(x,x)/d+1e-5);return x.map(v=>v*a)},sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m)),s=sum(v);return v.map(t=>t/s)};let forwards=0;
const model=eval("("+modelSrc+")");
function rng(seed){let a=seed>>>0;return ()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^a>>>15,1|a);t^=t+Math.imul(t^t>>>7,61|t);return (((t^t>>>14)>>>0)+.5)/4294967296}}
function draw(a,r){let u=r();for(let j=0;j<a.length;j++){u-=a[j];if(u<0)return j}return a.length-1}
const words=[...new Set(names.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h},methods=["KL","TV","Hellinger","treeRB","flatRB"];
let rows=[],steps=0,groups=0,scalarCount=0,exactFormalDiff=0;
const zero=()=>Array(V*d).fill(0),copy=w=>w.slice(),mean=ws=>ws[0].map((_,i)=>avg(ws.map(w=>w[i]))),prob=(w,f)=>sm(Array.from({length:V},(_,i)=>dot(w.slice(i*d,(i+1)*d),f)));
function accum(g,c,f){for(let j=0;j<V;j++)for(let k=0;k<d;k++)g[j*d+k]+=c[j]*f[k]}
function change(w,g,lr,anchor){return w.map((v,i)=>v-lr*(g[i]+.002*(v-anchor[i])))}
function sampleBatch(bank,r){return Array.from({length:4},()=>bank[Math.floor(r()*bank.length)])}
function evalset(data,W,D,seed){let ll=0,tv=0,bc=0,count=0,ov=0,r=rng(seed);
for(let t=0;t<40;t++){let a=data.map(bank=>bank[Math.floor(r()*bank.length)]),p=a.map(x=>prob(W,x.deep)),q=a.map(x=>prob(D,x.sh));for(let i=0;i<4;i++){ll-=Math.log(p[i][a[i].y]);tv+=sum(p[i].map((v,j)=>Math.abs(v-q[i][j])))/2;bc+=sum(p[i].map((v,j)=>Math.sqrt(v*q[i][j])));count++}for(let b=0;b<128;b++){let l=0;for(let k=0;k<4;k++)for(let i=0;i<4;i++){let j=draw(q[i],r);l+=Math.log(p[i][j]/q[i][j])}ov+=Math.exp(Math.min(0,l))}}
return {nll:ll/count,tv:tv/count,affinity:bc/count,overlap:ov/(40*128)}}
for(let wi=0;wi<raws.length;wi++){const m=model(raws[wi]),init=m.W.slice(0,V).flat();for(let seed of [11,29,73]){const data=words.slice().sort((a,b)=>hash(a,seed)-hash(b,seed)),trainNames=data.slice(0,640),testNames=data.slice(640,960);for(let hetero of [0,1]){
function bank(ns){let out=[[],[],[],[]];for(let s of ns){let c=hetero?Math.min(3,Math.floor((s.charCodeAt(0)-97)/7)):hash(s,seed+101)%4;for(let t=2;t<Math.min(7,s.length);t++){let f=m.get([26,...s.slice(0,t).split("").map(x=>x.charCodeAt(0)-97)]);out[c].push({deep:f.deep,sh:f.sh,y:s.charCodeAt(t)-97})}}if(out.some(x=>!x.length))throw Error("empty");return out}
let tr=bank(trainNames),te=bank(testNames),initial=evalset(te,init,init,seed+761),formalRef=null;
for(let method of methods){let W=init.slice(),D=init.slice(),rs=rng(seed+wi*421+hetero*987),rr=rng(seed+wi*3333+hetero*1307),trace=[];
for(let round=0;round<10;round++){
let ew=[W.slice(),W.slice()],ed=[D.slice(),D.slice()];
for(let ecycle=0;ecycle<2;ecycle++){let cw=[ew[0].slice(),ew[0].slice(),ew[1].slice(),ew[1].slice()],cd=[ed[0].slice(),ed[0].slice(),ed[1].slice(),ed[1].slice()];
for(let local=0;local<2;local++){let batch=tr.map(b=>sampleBatch(b,rs)),gw=Array.from({length:4},zero),gd=Array.from({length:4},zero);
let a=batch.flatMap((b,i)=>b.map(x=>({i,x,p:prob(cw[i],x.deep),q:prob(cd[i],x.sh)})));
for(let o of a){accum(gw[o.i],o.p.map((v,j)=>(v-+(j===o.x.y))/4),o.x.deep)}
if(method==="treeRB"||method==="flatRB"){
let sets=method==="treeRB"?[a.filter(o=>o.i<2),a.filter(o=>o.i>=2)]:[a];
for(let set of sets){let M=8,coeff=set.map(()=>Array(V).fill(0));let lrs=set.map(o=>o.p.map((v,j)=>Math.log(v/o.q[j])));for(let b=0;b<M;b++){let ix=set.map(o=>draw(o.q,rr)),L=sum(ix.map((j,k)=>lrs[k][j]));for(let k=0;k<set.length;k++){let o=set[k],s=L-lrs[k][ix[k]],mask=lrs[k].map(v=>+(s+v>0)),G=dot(o.q,mask);for(let j=0;j<V;j++)coeff[k][j]+=o.q[j]*(G-mask[j])/M}scalarCount+=set.length}
for(let k=0;k<set.length;k++){let o=set[k];accum(gd[o.i],coeff[k].map(v=>v/4),o.x.sh)}groups++}
}else for(let o of a){let c;if(method==="KL")c=o.q.map((v,j)=>v-o.p[j]);else if(method==="TV"){let m=o.q.map((v,j)=>+(o.p[j]>v)),G=dot(o.q,m);c=o.q.map((v,j)=>v*(G-m[j]))}else{let a=o.q.map((v,j)=>Math.sqrt(v*o.p[j])),s=sum(a);c=o.q.map((v,j)=>v*s-a[j])}accum(gd[o.i],c.map(v=>v/4),o.x.sh)}
for(let i=0;i<4;i++){cw[i]=change(cw[i],gw[i],.03,init);cd[i]=change(cd[i],gd[i],.06,init);steps++}}
ew=[mean(cw.slice(0,2)),mean(cw.slice(2))];ed=[mean(cd.slice(0,2)),mean(cd.slice(2))]}
W=mean(ew);D=mean(ed);if(round===4||round===9)trace.push({round:round+1,...evalset(te,W,D,seed+761)})}
if(formalRef)exactFormalDiff=Math.max(exactFormalDiff,...W.map((v,i)=>Math.abs(v-formalRef[i])));else formalRef=W.slice();
rows.push({wi,seed,hetero,method,initial,trace,final:trace.at(-1),trainSizes:tr.map(x=>x.length),testSizes:te.map(x=>x.length),formalDelta:Math.sqrt(sum(W.map((v,i)=>(v-init[i])**2))),draftDelta:Math.sqrt(sum(D.map((v,i)=>(v-init[i])**2)))})}}}}
let summary=[];for(let hetero of [0,1])for(let method of methods){let a=rows.filter(r=>r.hetero===hetero&&r.method===method);summary.push({hetero,method,n:a.length,initialOverlap:avg(a.map(x=>x.initial.overlap)),overlap:avg(a.map(x=>x.final.overlap)),initialNLL:avg(a.map(x=>x.initial.nll)),nll:avg(a.map(x=>x.final.nll)),tv:avg(a.map(x=>x.final.tv)),formalDelta:avg(a.map(x=>x.formalDelta)),draftDelta:avg(a.map(x=>x.draftDelta))})}
return {rows,summary,forwards,steps,groups,scalarCount,exactFormalDiff}}
const modelSrc="function model(raw){\nlet off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));const E=take(27,d),P=take(16,d),W=take(27,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(64,d),f2:take(d,64)}));if(off!==7264)throw Error(\"shape\");\nlet cache=new Map();\nfunction get(h){let key=h.join(\",\");if(cache.has(key))return cache.get(key);let prev=h.length>1?get(h.slice(0,-1)):null,x=rms(add(E[h.at(-1)],P[h.length-1])),layers=[],sh;\nfor(let j=0;j<2;j++){let b=L[j],z=rms(x),q=mat(b.q,z),k=mat(b.k,z),v=mat(b.v,z),ks=(prev?prev.layers[j].ks:[]).concat([k]),vs=(prev?prev.layers[j].vs:[]).concat([v]),a=[];\nfor(let t=0;t<H;t++){let o=t*4,w=sm(ks.map(k=>dot(q.slice(o,o+4),k.slice(o,o+4))/2));for(let l=o;l<o+4;l++)a.push(sum(w.map((p,i)=>p*vs[i][l])))}\nlet y=add(x,mat(b.o,a));x=add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))));layers.push({ks,vs});if(j===0)sh=x}\nlet p=sm(mat(W.slice(0,V),x)),q=sm(mat(W.slice(0,V),sh)),a={key,p,q,deep:x,sh,layers};cache.set(key,a);forwards++;return a}\nfunction full(h){let seq=h.map((t,i)=>rms(add(E[t],P[i]))),sh;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*4,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+4),x.slice(t,t+4))/2));for(let l=t;l<t+4;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!sh)sh=seq.at(-1)}return {p:sm(mat(W.slice(0,V),seq.at(-1))),q:sm(mat(W.slice(0,V),sh))}}\nreturn {get,full,W,clear:()=>cache.clear()}}\n";
if(require.main===module){const fs=require('fs');console.log(JSON.stringify(train([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'),modelSrc),null,2))}
