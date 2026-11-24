function probe(raws,names){
const V=27,d=16,F=64,sum=a=>a.reduce((s,v)=>s+v,0),avg=a=>sum(a)/a.length,zero=n=>Array(n).fill(0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),mul=(a,s)=>a.map(x=>x*s),mat=(A,x)=>A.map(r=>dot(r,x)),rms=x=>mul(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),z=x.map(v=>Math.exp(v-m));return mul(z,1/sum(z))},outer=(a,b)=>a.map(x=>mul(b,x)),madd=(A,B)=>A.map((r,i)=>add(r,B[i])),mscale=(A,s)=>A.map(r=>mul(r,s)),mzero=()=>Array.from({length:d},()=>zero(F)),mmean=a=>mscale(a.reduce(madd,mzero()),1/a.length),over=(p,q)=>sum(p.map((x,i)=>Math.min(x,q[i]))),hash=(x,s)=>{let h=s>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
let featCalls=0,denseChecks=0,maxSparseError=0,fdMax=0,updates=0,exampleGrads=0,gateFits=0,rows=[],traces=[];
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(F,d),f2:take(d,F)}));if(off!==7264)throw Error("shape");
function prep(seq,b){let z=seq.map(rms);return {Q:z.map(x=>mat(b.q,x)),K:z.map(x=>mat(b.k,x)),v:z.map(x=>mat(b.v,x))}}
function att(seq,b,p,i){let a=[];for(let j=0;j<4;j++){let t=j*4,w=sm(p.K.slice(0,i+1).map(x=>dot(p.Q[i].slice(t,t+4),x.slice(t,t+4))/2));for(let l=t;l<t+4;l++)a.push(sum(w.map((s,k)=>s*p.v[k][l])))}let y=add(seq[i],mat(b.o,a));return {t:y,f:mat(b.f1,rms(y)).map(x=>Math.max(0,x))}}
function shallow(h){let s=h.map((t,i)=>rms(add(E[t],P[i]))),b=layers[0],p=prep(s,b);return s.map((_,i)=>{let a=att(s,b,p,i);return add(a.t,mat(b.f2,a.f))})}
function feature(h){featCalls++;let seq=shallow(h),b=layers[1],z=seq.map(rms),i=seq.length-1,p={Q:Array.from({length:seq.length},()=>null),K:z.map(x=>mat(b.k,x)),v:z.map(x=>mat(b.v,x))};p.Q[i]=mat(b.q,z[i]);let a=att(seq,b,p,i),x=seq[i],q=sm(mat(W,x));if(denseChecks<180){let all=prep(seq,b),dense=seq.map((_,i)=>{let s=att(seq,b,all,i);return add(s.t,mat(b.f2,s.f))});maxSparseError=Math.max(maxSparseError,...sub(dense[i],add(a.t,mat(b.f2,a.f))).map(Math.abs));denseChecks++}return {x,t:a.t,f:a.f,z:[1,...rms(x),-sum(q.map(v=>v*Math.log(v))),Math.max(...q)]}}
function block(s,bank,i){let b=layers[1],z=bank.map(rms),Q=mat(b.q,rms(s)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x)),a=[];for(let j=0;j<4;j++){let t=j*4,w=sm(K.slice(0,i+1).map(x=>dot(Q.slice(t,t+4),x.slice(t,t+4))/2));for(let l=t;l<t+4;l++)a.push(sum(w.map((s,k)=>s*v[k][l])))}let y=add(s,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(x=>Math.max(0,x))))}function deltaBank(x,a){let b=layers[1],i=x.length-1,Q=mat(b.q,rms(a[i])),K=x.map(z=>mat(b.k,rms(z))),Vv=x.map(z=>mat(b.v,rms(z))),K1=a.map(z=>mat(b.k,rms(z))),V1=a.map(z=>mat(b.v,rms(z))),heads=[];
for(let h=0;h<4;h++){let j=h*4,q=Q.slice(j,j+4),s=K.map(k=>dot(q,k.slice(j,j+4))/2),t=K1.map(k=>dot(q,k.slice(j,j+4))/2),m=Math.max(...s,...t),u=s.map(v=>Math.exp(v-m)),v=t.map(v=>Math.exp(v-m)),N=zero(4),D=sum(u),dn=[],dd=[];for(let k=0;k<x.length;k++){let p=mul(Vv[k].slice(j,j+4),u[k]),r=mul(V1[k].slice(j,j+4),v[k]);N=add(N,p);dn.push(sub(r,p));dd.push(v[k]-u[k])}heads.push({N,D,dn,dd})}
return mask=>{let o=[];for(let h of heads){let N=h.N.slice(),D=h.D;for(let k of mask){N=add(N,h.dn[k]);D+=h.dd[k]}o.push(...mul(N,1/D))}let y=add(a[i],mat(b.o,o));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))}}
return {W,shallow,block,deltaBank}}


const ent=p=>-sum(p.map(x=>x*Math.log(x))),norm=x=>Math.sqrt(dot(x,x)),md=(a,b)=>Math.max(...sub(a,b).map(Math.abs)),pairs=n=>Array.from({length:n},(_,i)=>Array.from({length:n-i-1},(_,j)=>[i,i+j+1])).flat(),Pdim=41;
let checks=0,error=0,bankCalls=0,evals=0,fitCount=0,nonSub=0,negative=0,totalInteract=0,maxInteraction=0;
function ridge(data,key,lam){let M=Array.from({length:Pdim},()=>zero(Pdim+1));for(let s of data){for(let i=0;i<Pdim;i++){M[i][Pdim]+=s.z[i]*s[key];for(let j=0;j<Pdim;j++)M[i][j]+=s.z[i]*s.z[j]}}for(let i=0;i<Pdim;i++)M[i][i]+=lam;for(let i=0;i<Pdim;i++){let k=M[i][i];for(let j=i;j<=Pdim;j++)M[i][j]/=k;for(let k=0;k<Pdim;k++)if(k!==i){let u=M[k][i];for(let j=i;j<=Pdim;j++)M[k][j]-=u*M[i][j]}}fitCount++;return M.map(x=>x[Pdim])}

let words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],stat={trainRoots:0,trainingChildren:0,exactRootScores:0};
for(let wi=0;wi<raws.length;wi++){
let mo=model(raws[wi]),W=mo.W,cache=new Map();
function get(h){let key=h.join(",");if(!cache.has(key)){let x=mo.shallow(h),i=x.length-1,a=x.map((v,j)=>mo.block(v,x.slice(0,j+1),j)),q=sm(mat(W,x[i])),bank=mo.deltaBank(x,a),pc=new Map(),features=x.slice(0,i).map((v,j)=>[1,...rms(v),...rms(x[i]),(i-j)/i,dot(rms(v),rms(x[i]))/d,ent(sm(mat(W,v))),ent(q)]);cache.set(key,{x,a,q,features,pc,bank});featCalls++;bankCalls++}return cache.get(key)}
function p(s,ids){let k=ids.slice().sort((a,b)=>a-b).join(",");if(!s.pc.has(k)){let y=s.bank(ids);if(checks<600){let bank=s.x.map((v,i)=>ids.includes(i)?s.a[i]:v),ref=mo.block(s.a.at(-1),bank,bank.length-1);error=Math.max(error,md(y,ref));checks++}s.pc.set(k,sm(mat(W,y)));evals++}return s.pc.get(k)}
function pairZ(s,ij){let [i,j]=ij,a=s.features[i],b=s.features[j];return [...mul(add(a,b),.5),Math.abs(i-j)/(s.x.length-1),dot(rms(s.x[i]),rms(s.x[j]))/d,Math.abs(a[35]-b[35]),a[34]*b[34]]}
function entPick(s){return s.features.map((z,i)=>({i,v:z[35]})).sort((a,b)=>b.v-a.v||a.i-b.i).slice(0,2).map(a=>a.i)}
function choose(s,coef){return pairs(s.features.length).map(ij=>({ij,v:dot(pairZ(s,ij),coef)})).sort((a,b)=>b.v-a.v)[0].ij}
function childA(h,policy){return Array.from({length:V},(_,i)=>{let s=get(h.concat(i));return over(p(s,policy(s)),s.q)})}
function score(s,lab,ij,ca,beta){let pp=p(s,ij),a=sum(pp.map((v,i)=>Math.min(v,s.q[i])*(1+ca[i])));return {j:Math.log(pp[lab])+beta*a,nll:-Math.log(pp[lab]),a}}
for(let seed of [11,29,73]){
let pools=Array.from({length:4},(_,c)=>words.filter(w=>Math.min(3,Math.floor((w.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,40));
let sample=w=>({h:[26,...w.slice(0,5).split("").map(x=>x.charCodeAt(0)-97)],lab:w.charCodeAt(5)-97});
let train=pools.map(v=>v.slice(0,20).map(sample)),cal=pools.map(v=>v.slice(20,28).map(sample)),test=pools.map(v=>v.slice(28,40).map(sample));
let trainData=train.map(cs=>cs.flatMap(t=>{
let s=get(t.h),ca=childA(t.h,entPick);stat.trainRoots++;stat.trainingChildren+=V;
return pairs(s.features.length).map(ij=>{let v=score(s,t.lab,ij,ca,1);stat.exactRootScores++;return {z:pairZ(s,ij),ce:-v.nll,a:v.a}})
}));
for(let beta of [0,.3,1]){
let data=trainData.map(cs=>cs.map(s=>({...s,g:s.ce+beta*s.a}))),glob=ridge(data.flat(),"g",3),edge=[ridge(data[0].concat(data[1]),"g",3),ridge(data[2].concat(data[3]),"g",3)],loc=data.map(x=>ridge(x,"g",3));
let shrink=(a,b,l)=>add(mul(a,l),mul(b,1-l)),coefs=loc.map((x,c)=>[glob,edge[c>>1],x,shrink(edge[c>>1],glob,.5),shrink(x,edge[c>>1],.5)]);
let selected=cal.map((cs,c)=>{let v=coefs[c].map(coef=>avg(cs.map(t=>{let s=get(t.h),ca=childA(t.h,entPick);return score(s,t.lab,choose(s,coef),ca,beta).j})));return v.indexOf(Math.max(...v))});
for(let c=0;c<4;c++){
let totals={},cs=test[c];
for(let t of cs){let s=get(t.h),ca=childA(t.h,entPick),oracle=Math.max(...pairs(s.features.length).map(ij=>score(s,t.lab,ij,ca,beta).j));
for(let method of ["entropy","global","edge","local","calibrated"]){
let pol=method==="entropy"?entPick:s=>choose(s,method==="global"?glob:method==="edge"?edge[c>>1]:method==="local"?loc[c]:coefs[c][selected[c]]),ij=pol(s),base=score(s,t.lab,ij,ca,beta),actual=score(s,t.lab,ij,childA(t.h,pol),beta),a=totals[method]||{nll:0,eal:0,obj:0,regret:0};a.nll+=actual.nll;a.eal+=actual.a;a.obj+=actual.j;a.regret+=oracle-base.j;totals[method]=a
}}
let arms=Object.fromEntries(Object.entries(totals).map(([k,v])=>[k,Object.fromEntries(Object.entries(v).map(([j,x])=>[j,x/cs.length]))]));
rows.push({wi,seed,beta,c,selected:selected[c],arms});
}
}
}
}
let summary=[];for(let beta of [0,.3,1]){let a=rows.filter(x=>x.beta===beta);summary.push({beta,n:a.length,selected:Array.from({length:5},(_,i)=>a.filter(x=>x.selected===i).length),arms:Object.fromEntries(Object.keys(a[0].arms).map(k=>[k,Object.fromEntries(["nll","eal","obj","regret"].map(j=>[j,avg(a.map(x=>x.arms[k][j]))]))]))})}
return {featCalls,bankCalls,evals,fitCount,checks,error,stat,summary,rows};
}
const fs=require('node:fs');console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
