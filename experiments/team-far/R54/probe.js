// R54 adapted feature evaluator from R44/gate.js at 499741a35805fce575435e37a64e2e53a4ef0157
function probe(raws,names){
const V=27,d=16,sum=a=>a.reduce((s,v)=>s+v,0),mean=a=>sum(a)/a.length,dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let v=x.map(t=>Math.exp(t-Math.max(...x)));return scale(v,1/sum(v))},copy=A=>A.map(r=>r.slice()),over=(a,b)=>sum(a.map((v,i)=>Math.min(v,b[i]))),hash=(x,s)=>{let h=s>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
let features=0,formalSteps=0,headSteps=0;
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{features++;let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<4;j++){let t=j*4,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+4),x.slice(t,t+4))/2));for(let l=t;l<t+4;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}

const sub=(a,b)=>a.map((x,i)=>x-b[i]),norm=a=>Math.sqrt(dot(a,a)),outer=(a,b)=>a.map(x=>scale(b,x)),madd=(a,b)=>a.map((r,i)=>add(r,b[i])),mscale=(a,s)=>a.map(r=>scale(r,s)),mz=()=>Array.from({length:V},()=>zero(d)),mavg=a=>mscale(a.reduce(madd,mz()),1/a.length),project=(x,B)=>B.reduce((v,b)=>add(v,scale(b,dot(x,b))),zero(d)),projM=(M,B)=>M.map(x=>project(x,B)),maxabs=a=>Math.max(...a.flat().map(Math.abs)),avg=a=>sum(a)/a.length;
let steps=0,gradCalls=0,checks=0,maxInvariant=0,maxFD=0,maxOrth=0,rows=[];
function basis(xs,k,prior,seed,random=false){let out=prior.slice();function remove(x){return sub(x,project(x,out))}
for(let j=0;j<k;j++){let v=remove(Array.from({length:d},(_,i)=>Math.sin((i+1)*(j+2)*(seed+1)+.31*j)));v=scale(v,1/(norm(v)||1));
if(!random)for(let n=0;n<25;n++){let w=remove(xs.reduce((a,x)=>add(a,scale(x,dot(x,v))),zero(d)));if(norm(w)<1e-10)break;v=scale(w,1/norm(w))}
v=remove(v);if(norm(v)<1e-9){for(let i=0;i<d;i++){v=remove(Array.from({length:d},(_,j)=>+(i===j)));if(norm(v)>1e-9)break;}}v=scale(v,1/norm(v));for(let w of out)maxOrth=Math.max(maxOrth,Math.abs(dot(v,w)));out.push(v)}return out.slice(prior.length)}
function mass(p,q){let m=p.map((v,i)=>Math.min(v,q[i]));let gp=zero(V),gq=zero(V);for(let i=0;i<V;i++){let z=p[i]<=q[i]?p:q, g=p[i]<=q[i]?gp:gq;for(let j=0;j<V;j++)g[j]+=z[i]*(+(i===j)-z[j])}return {a:sum(m),m,gp,gq}}
function prob(W,A,visible,s,tied){return [sm(mat(madd(W,A),s.y)),sm(add(mat(W,s.x),mat(A,tied?s.x:project(s.x,visible))))]}
function grad(W,A,vis,cs,tied,beta){let G=mz();for(let s of cs){let [p,q]=prob(W,A,vis,s,tied),m=mass(p,q),ce=p.map((v,i)=>v-+(i===s.label));G=madd(G,madd(outer(sub(ce,scale(m.gp,beta)),s.y),outer(scale(m.gq,-beta),tied?s.x:project(s.x,vis))));gradCalls++}return mscale(G,1/cs.length)}
function evaluate(W,A,vis,cs,tied){let nll=0,eal=0;for(let s of cs){let [p,q]=prob(W,A,vis,s,tied),m=mass(p,q);nll-=Math.log(p[s.label]);let a=m.a;for(let i=0;i<V;i++){let [pp,qq]=prob(W,A,vis,s.fs[i],tied);a+=m.m[i]*over(pp,qq)}eal+=a;}return {nll:nll/cs.length,eal:eal/cs.length}}
const words=[...new Set(names.toLowerCase().split(/\s+/).filter(w=>/^[a-z]{6,14}$/.test(w)))];
for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);let cache=new Map(),F=h=>{let k=h.join(",");if(!cache.has(k))cache.set(k,feature(h));return cache.get(k)};for(let seed of [11,29,73]){
let pools=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,36));
const sample=w=>{let h=[26,...w.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)];return {...F(h),h,label:w.charCodeAt(3)-97}};
let train=pools.map(p=>p.slice(0,18).map(sample)),test=pools.map(p=>p.slice(18,30).map(w=>{let s=sample(w);return {...s,fs:Array.from({length:V},(_,i)=>F(s.h.concat(i)))}}));
for(let eta of [.02,.1,.4]){
let pre=train.map(cs=>{let A=mz();for(let t=0;t<12;t++){A=madd(A,mscale(grad(W,A,[],cs,true,0),-eta));steps++}return A});
for(let rank of [4,8,12]){
for(let method of ["cut","tied","randomCut","taskCut","flatTied"]){
let random=method==="randomCut",tied=method==="tied"||method==="flatTied",beta=method==="taskCut"?0:.3,flat=method==="flatTied";
let r0=rank/2,rg=rank-r0;
let B0=basis(train.flat().map(s=>s.x),r0,[],seed,random);
let Be=[0,1].map(e=>basis(train.slice(2*e,2*e+2).flat().map(s=>s.x),rg,B0,seed+e+13,random));
let vis=Be.map(b=>B0.concat(b)),A=pre.map(copy);
// Identical initial formal matrices within all arms. Decompose only the updates, not initial matrices;
// use W+pre as per-client immutable initial head, and adapt delta from zero. Draft cut projects only new updates.
let Wc=pre.map(a=>madd(W,a));A=Array.from({length:4},mz);
let initial=test.map((cs,c)=>evaluate(Wc[c],A[c],vis[c>>1],cs,tied));
for(let t=0;t<12;t++){let g=train.map((cs,c)=>grad(Wc[c],A[c],vis[c>>1],cs,tied,beta));
let gg=projM(mavg(g),B0),eg=[0,1].map(e=>projM(mavg(g.slice(2*e,2*e+2)),Be[e]));
for(let c=0;c<4;c++){let own=g[c],privateG=madd(own,mscale(projM(own,vis[c>>1]),-1));let delta=mscale(flat?own:madd(madd(gg,eg[c>>1]),privateG),-.08);A[c]=madd(A[c],delta);steps++} }
let final=test.map((cs,c)=>evaluate(Wc[c],A[c],vis[c>>1],cs,tied));
let change=0,privGain=0,predDrift=0;
for(let c=0;c<4;c++){let s=test[c][0],g=grad(Wc[c],A[c],vis[c>>1],train[c],tied,0),priv=mscale(madd(g,mscale(projM(g,vis[c>>1]),-1)),-.08),[p,q]=prob(Wc[c],A[c],vis[c>>1],s,tied),[pn,qn]=prob(Wc[c],madd(A[c],priv),vis[c>>1],s,tied);let dr=sum(q.map((v,i)=>Math.abs(v-qn[i])))/2;predDrift+=dr/4;if(!tied)maxInvariant=Math.max(maxInvariant,dr);privGain+=avg(train[c].map(s=>{let a=prob(Wc[c],A[c],vis[c>>1],s,tied)[0],b=prob(Wc[c],madd(A[c],priv),vis[c>>1],s,tied)[0];return Math.log(b[s.label])-Math.log(a[s.label])}))/4;
if(method==="cut"&&c===0){let eps=1e-5,idx=seed%V,j=seed%d,du=mz();du[idx][j]=eps;let exact=grad(Wc[c],A[c],vis[c>>1],[s],false,.3)[idx][j],obj=a=>{let [p,q]=prob(Wc[c],a,vis[c>>1],s,false);return -Math.log(p[s.label])-.3*over(p,q)};maxFD=Math.max(maxFD,Math.abs(exact-(obj(madd(A[c],du))-obj(madd(A[c],mscale(du,-1))))/(2*eps)));checks++}}
rows.push({wi,seed,eta,rank,method,initialNLL:avg(initial.map(x=>x.nll)),initialEAL:avg(initial.map(x=>x.eal)),nll:avg(final.map(x=>x.nll)),eal:avg(final.map(x=>x.eal)),privateDraftTV:predDrift,privateTrainNLLGain:privGain});
}}}}}
let summary=[];for(let eta of [.02,.1,.4])for(let rank of [4,8,12])for(let method of ["cut","tied","randomCut","taskCut","flatTied"]){let a=rows.filter(x=>x.eta===eta&&x.rank===rank&&x.method===method);summary.push({eta,rank,method,...Object.fromEntries(["initialNLL","initialEAL","nll","eal","privateDraftTV","privateTrainNLLGain"].map(k=>[k,avg(a.map(x=>x[k]))]))})}
return {features,steps,gradCalls,checks,maxInvariant,maxFD,maxOrth,summary,rows};}
const fs=require('node:fs');console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
