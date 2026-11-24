// node probe.js weights.json checkpoint.json input.txt
// Transformer forward adapted from team-far R21/R23; R24 random anchor cohort estimator.
const fs=require('node:fs');
function probe(raws,names){
const G=8,V=27,d=16,H=4,k=4,bs=[2,3,4,6,8],eps=1e-6;
const sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,s)=>a.map(v=>v*s),zero=()=>Array(k).fill(0);
const rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5));
const sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
const sd=m=>Math.sqrt(m*(G-m))/G+eps;
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:(h,shallow=false)=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),q;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!q){q=seq.at(-1);if(shallow)return q}}return {h:seq.at(-1),s:q}}}}
function rng(s){return ()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(a,u){let s=0;for(let i=0;i<a.length;i++){s+=a[i];if(u<s)return i}return a.length-1}
function choose(n,k){if(k<0||k>n)return 0;let v=1;for(let j=1;j<=k;j++)v=v*(n-j+1)/j;return v}
function bin(n,m,p){return choose(n,m)*p**m*(1-p)**(n-m)}
function stats(p,Delta,vp,vm,b,method){
let M=0,S=0,gen=0,back=0;
for(let a=0;a<=b;a++)for(let c=0;c<=G-b;c++){
let pr=bin(b,a,p)*bin(G-b,c,p),m=a+c,mix=a>0&&a<b;
gen+=pr*(mix?G:b);if(!mix)continue;
let w=(G-1)*a*(b-a)/(G*b*(b-1)*sd(m));
let varc=method==="two"?vp+vm:vp/a+vm/(b-a);
M+=pr*w;S+=pr*w*w*(Delta+varc);back+=pr*(method==="two"?2:b);
}return {coef:M,var:S-M*M*Delta,gen,back}}
let theoretical=[],maxMean=0;
for(let p of [.01,.05,.1,.2,.3,.5,.7,.9,.95,.99])for(let v of [0,.1,1,10]){
let base=stats(p,1,v,v,G,"all");
for(let b of bs)for(let method of ["all","two"]){
let s=stats(p,1,v,v,b,method);maxMean=Math.max(maxMean,Math.abs(s.coef-base.coef));
let costs=[];
for(let r of [.25,1,2])costs.push({r,ratio:(s.gen+r*s.back)/(base.gen+r*base.back),varCost:s.var*(s.gen+r*s.back)/(base.var*(base.gen+r*base.back)||1e-99)});
theoretical.push({p,v,b,method,...s,costs})
}}
const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
let words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],rows=[],groupRows=[];
let audit={conditionalMean:0,groups:0,trajectories:0,tokens:0,accepted:0,attempt:0,bonus:0,exactOneMean:maxMean};
let masks={};
for(let b of bs)masks[b]=Array.from({length:1<<G},(_,m)=>m).filter(m=>m.toString(2).replace(/0/g,"").length===b);
for(let wi=0;wi<raws.length;wi++){
let m=model(raws[wi]),W=m.W,A=W.map(w=>w.slice(0,k));
const moment=p=>p.reduce((a,v,i)=>add(a,scale(A[i],v)),zero());
function roll(pre,region,rr,N=4,K=3){
let h=pre.slice(),Y=[],S=zero();
while(Y.length<N){
let hist=h.slice(),draft=[];
for(let j=0;j<Math.min(K,N-Y.length);j++){
let q=sm(mat(W,m.feature(hist,true))),X=cat(q,rr());draft.push({q,X,pre:hist.slice()});hist.push(X)}
let fail=false;
for(let x of draft){
let p=sm(mat(W,m.feature(x.pre).h)),u=rr(),y=x.X;audit.attempt++;
if(u>Math.min(1,p[y]/x.q[y])){let r=p.map((v,i)=>Math.max(0,v-x.q[i]));y=cat(scale(r,1/sum(r)),rr());fail=true}else audit.accepted++;
Y.push(y);h.push(y);S=add(S,sub(A[y],moment(p)));if(fail)break}
if(!fail&&Y.length<N){let p=sm(mat(W,m.feature(h).h)),y=cat(p,rr());Y.push(y);h.push(y);S=add(S,sub(A[y],moment(p)));audit.bonus++}
}audit.trajectories++;audit.tokens+=N;return {r:Number(Y.at(-1)%4===region),S:scale(S,1/N)}}
for(let seed of [11,29,73]){
let groups=Array.from({length:4},(_,r)=>words.filter(x=>x.charCodeAt(0)%4===r).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,24));
let rr=rng(seed+1000*wi);
for(let region=0;region<4;region++){
for(let name of groups[region]){
let pre=[26,...name.slice(0,4).split("").map(c=>c.charCodeAt(0)-97)];
let p=sm(mat(W,m.feature(pre).h)),mu=moment(p),S=A.map(a=>sub(a,mu)),R=p.map((_,i)=>Number(i%4===region)),prob=sum(p.map((v,i)=>v*R[i]));
let plus=zero(),minus=zero();
for(let i=0;i<V;i++)if(R[i])plus=add(plus,scale(S[i],p[i]/prob));else minus=add(minus,scale(S[i],p[i]/(1-prob)));
let vp=sum(p.map((v,i)=>R[i]?v/prob*dot(sub(S[i],plus),sub(S[i],plus)):0));
let vm=sum(p.map((v,i)=>!R[i]?v/(1-prob)*dot(sub(S[i],minus),sub(S[i],minus)):0));
let Delta=dot(sub(plus,minus),sub(plus,minus)),row={wi,seed,region,p:prob,Delta,vp,vm,methods:[]};
for(let b of bs)for(let method of ["all","two"])row.methods.push({b,method,...stats(prob,Delta,vp,vm,b,method)});
rows.push(row)}
for(let name of groups[region].slice(0,8)){
let pre=[26,...name.slice(0,4).split("").map(c=>c.charCodeAt(0)-97)];
let xs=Array.from({length:G},()=>roll(pre,region,rr)),m=sum(xs.map(x=>x.r));
let ref=scale(xs.reduce((a,x)=>add(a,scale(x.S,x.r-m/G)),zero()),1/(G*sd(m)));audit.groups++;
let methods=[];
for(let b of bs){
let ex=zero(),second=0,gen=0,back=0,ml=0;
for(let mask of masks[b]){
let subg=xs.filter((_,i)=>mask&(1<<i)),a=sum(subg.map(x=>x.r)),v=zero();
if(a>0&&a<b){
let s=subg.reduce((v,x)=>add(v,scale(x.S,b*x.r-a)),zero());
v=scale(s,(G-1)/(G*b*(b-1)*sd(m)));gen+=G;back+=b;ml++
}else gen+=b;
ex=add(ex,v);second+=dot(sub(v,ref),sub(v,ref))}
let z=masks[b].length;ex=scale(ex,1/z);audit.conditionalMean=Math.max(audit.conditionalMean,Math.sqrt(dot(sub(ex,ref),sub(ex,ref))));
methods.push({b,gen:gen/z,back:back/z,extra:second/z,mixed:ml/z})
}groupRows.push({wi,seed,region,m,refNorm2:dot(ref,ref),methods})
}
}
}
}
let summary=[];
for(let b of bs)for(let method of ["all","two"]){
let a=rows.map(r=>r.methods.find(x=>x.b===b&&x.method===method)),base=rows.map(r=>r.methods.find(x=>x.b===G&&x.method==="all")),mean=k=>sum(a.map(x=>x[k]))/a.length,ratio=sum(a.map(x=>x.var))/sum(base.map(x=>x.var));
let relative=[];
for(let r of [.25,1,2]){
let cost=sum(a.map(x=>x.gen+r*x.back))/sum(base.map(x=>x.gen+r*x.back));relative.push({r,ratio:cost,varCost:ratio*cost})
}summary.push({b,method,gen:mean("gen"),back:mean("back"),varRatio:ratio,relative})
}
let gs=[];
for(let b of bs){let a=groupRows.map(r=>r.methods.find(x=>x.b===b)),mean=k=>sum(a.map(x=>x[k]))/a.length;gs.push({b,gen:mean("gen"),back:mean("back"),extra:mean("extra"),meanRefNorm2:sum(groupRows.map(x=>x.refNorm2))/groupRows.length})}
return {theoretical,rows,summary,groupRows,groupSummary:gs,audit}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
