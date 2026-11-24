// node probe.js weights.json checkpoint.json input.txt
// Forward adapted from team-far R24; R25 joint probability model and training.
const fs=require('node:fs');
function probe(raws,names){
const V=27,d=16,H=4,k=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i])));for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))})}return seq.at(-1)}}}
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(a,u){let s=0;for(let i=0;i<a.length;i++){s+=a[i];if(u<s)return i}return a.length-1}
const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],rows=[],audit={normError:0,factorError:0,gradError:0,coarseningError:0,mixtureMassError:0,draws:0};
function mixture(x,t,R){let n=d+1+2*V,gate=sm(Array.from({length:R},(_,z)=>dot(t.slice(z*n,z*n+d+1),x.f))),A=[],B=[];for(let z=0;z<R;z++){let off=z*n+d+1;A.push(sm(add(x.log,t.slice(off,off+V))));B.push(sm(add(x.log,t.slice(off+V,off+2*V))))}return {gate,A,B}}
function lossGrad(x,t,method,grad=true){
if(method==="AR"||method==="MTP"){let D=d+1,bs=[x.log,method==="AR"?x.log2:x.log],fs=[x.f,method==="AR"?x.f2:x.f],ys=[x.y,x.z],loss=0,g=zero(t.length);for(let j=0;j<2;j++){let p=sm(bs[j].map((v,i)=>v+dot(t.slice((j*V+i)*D,(j*V+i+1)*D),fs[j])));loss-=Math.log(p[ys[j]]);if(grad)for(let i=0;i<V;i++)for(let l=0;l<D;l++)g[(j*V+i)*D+l]=(p[i]-Number(i===ys[j]))*fs[j][l]}return {loss,g}}
let R=+method,n=d+1+2*V,{gate,A,B}=mixture(x,t,R),raw=gate.map((p,i)=>p*A[i][x.y]*B[i][x.z]),lik=sum(raw),post=scale(raw,1/lik),g=zero(t.length);
if(grad)for(let j=0;j<R;j++){let off=j*n;for(let l=0;l<d+1;l++)g[off+l]=(gate[j]-post[j])*x.f[l];for(let i=0;i<V;i++){g[off+d+1+i]=post[j]*(A[j][i]-Number(i===x.y));g[off+d+1+V+i]=post[j]*(B[j][i]-Number(i===x.z))}}
return {loss:-Math.log(lik),g}}
function joint(x,t,R){let {gate,A,B}=mixture(x,t,R),p=Array.from({length:V},(_,i)=>Array.from({length:V},(_,j)=>sum(gate.map((v,z)=>v*A[z][i]*B[z][j])))),a=p.map(sum),b=Array.from({length:V},(_,j)=>sum(p.map(r=>r[j])));return {p,a,b,gate,A,B}}
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]),W=m.W;
function ctx(name){return [2,4].map(j=>{let pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],y=name.charCodeAt(j)-97,z=name.charCodeAt(j+1)-97,h=m.feature(pre),h2=m.feature(pre.concat(y));return {f:rms(h).concat(1),f2:rms(h2).concat(1),log:mat(W,h),log2:mat(W,h2),y,z,region:name.charCodeAt(0)%4}})}
for(let seed of [11,29,73]){let groups=Array.from({length:4},(_,r)=>words.filter(x=>x.charCodeAt(0)%4===r).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,96)),train=groups.map(g=>g.slice(0,72).flatMap(ctx)),test=groups.flatMap(g=>g.slice(72).flatMap(ctx));
let base=sum(test.map(x=>-Math.log(sm(x.log)[x.y])-Math.log(sm(x.log2)[x.z])))/test.length;
rows.push({wi,seed,method:"frozenAR",topology:"none",initial:base,final:base,params:0,updates:0});
for(let method of ["1","2","4","8","MTP","AR"])for(let topology of ["HFL","flat"]){
let rr=rng(seed+101*wi),dim=method==="MTP"||method==="AR"?2*V*(d+1):+method*(d+1+2*V),t=Array.from({length:dim},()=> (rr()-.5)*.08),dataR=rng(seed+567),lr=.045,curve=[],updates=0;
const evaluate=()=>sum(test.map(x=>lossGrad(x,t,method,false).loss))/test.length,initial=evaluate();
function local(init,client,steps){let v=init.slice();for(let step=0;step<steps;step++){let g=zero(dim);for(let b=0;b<8;b++){let x=train[client][Math.floor(dataR()*train[client].length)];g=add(g,lossGrad(x,v,method).g)}v=add(v,scale(g,-lr/8));updates++}return v}
const avg=vs=>scale(vs.reduce(add),1/vs.length);
for(let round=0;round<16;round++){
if(topology==="HFL"){let ed=[];for(let e=0;e<2;e++){let v=t.slice();for(let j=0;j<2;j++)v=avg([local(v,e*2,2),local(v,e*2+1,2)]);ed.push(v)}t=avg(ed)}else t=avg([0,1,2,3].map(c=>local(t,c,4)));curve.push(evaluate())}
let item={wi,seed,method,topology,params:dim,initial,final:curve.at(-1),curve,updates,uses:updates*8};
if(!["MTP","AR"].includes(method)){
let R=+method,acc=0,mi=0,kl=0,accept2=0,directTV=0,rr2=rng(seed+912),count=0;
for(let x of test){let {p,a,b,gate,A,B}=joint(x,t,R);audit.normError=Math.max(audit.normError,Math.abs(sum(a)-1));let q=a.map(v=>b.map(u=>v*u)),tv=0,info=0;for(let i=0;i<V;i++)for(let j=0;j<V;j++){let v=p[i][j];audit.factorError=Math.max(audit.factorError,Math.abs(v-a[i]*(v/a[i])));tv+=Math.abs(v-q[i][j])/2;info+=v*Math.log(v/q[i][j])}acc+=1-tv;mi+=info;
let merged=gate.map((_,z)=>z%2),mass=[0,0],aa=[zero(V),zero(V)],bb=[zero(V),zero(V)];for(let z=0;z<R;z++){let g=merged[z];mass[g]+=gate[z];aa[g]=add(aa[g],scale(A[z],gate[z]));bb[g]=add(bb[g],scale(B[z],gate[z]))}let qq=Array.from({length:V},(_,i)=>Array.from({length:V},(_,j)=>sum(mass.map((w,g)=>w?aa[g][i]*bb[g][j]/w:0))));let t2=0;for(let i=0;i<V;i++)for(let j=0;j<V;j++)t2+=Math.abs(p[i][j]-qq[i][j])/2;accept2+=1-t2;
if(R>1){let first=0,second=0;for(let z=0;z<R;z++){first+=gate[z];second+=gate[z]*sum(A[z])*sum(B[z])}audit.mixtureMassError=Math.max(audit.mixtureMassError,Math.abs(first-second))}
if(count++<12){let c1=zero(V*V),c2=zero(V*V);for(let s=0;s<256;s++){let z=cat(gate,rr2()),i=cat(A[z],rr2()),j=cat(B[z],rr2());c1[i*V+j]++;let ii=cat(a,rr2()),jj=cat(p[ii].map(v=>v/a[ii]),rr2());c2[ii*V+jj]++;audit.draws+=2}directTV+=sum(c1.map((v,i)=>Math.abs(v-c2[i])))/512}
}
let x=test[0],g=lossGrad(x,t,method).g;for(let i=0;i<Math.min(12,t.length);i++){let j=Math.floor(i*t.length/12),tt=t.slice();tt[j]+=1e-5;let l1=lossGrad(x,tt,method,false).loss;tt[j]-=2e-5;let l2=lossGrad(x,tt,method,false).loss;audit.gradError=Math.max(audit.gradError,Math.abs(g[j]-(l1-l2)/2e-5))}
Object.assign(item,{overlap:acc/test.length,mi:mi/test.length,twoGroupOverlap:accept2/test.length,empiricalPairTV:directTV/Math.min(12,test.length)})}
rows.push(item)
}
}}
let summary=[];for(let method of ["frozenAR","1","2","4","8","MTP","AR"])for(let topology of method==="frozenAR"?["none"]:["HFL","flat"]){let a=rows.filter(x=>x.method===method&&x.topology===topology),mean=k=>sum(a.map(x=>x[k]||0))/a.length;summary.push({method,topology,params:a[0].params,initial:mean("initial"),final:mean("final"),overlap:mean("overlap"),mi:mean("mi"),twoGroupOverlap:mean("twoGroupOverlap"),updates:sum(a.map(x=>x.updates)),uses:sum(a.map(x=>x.uses||0))})}
return {rows,summary,audit,trainPerConfig:576,testPerConfig:192}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
