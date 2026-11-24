// Derived from team-far/R52/probe.js at cf7864ef729d17992018b1d091c5e28c78d7edbb; node probe.js weights.json checkpoint.json input.txt
function probe(raws,names){
const V=27,d=16,sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),zero=n=>Array(n).fill(0),add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),mul=(a,b)=>a.map(x=>x*b),avg=a=>mul(a.reduce(add,zero(a[0].length)),1/a.length),mat=(a,x)=>a.map(r=>dot(r,x)),rms=x=>mul(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),z=x.map(t=>Math.exp(t-m));return mul(z,1/sum(z))},hash=(x,s)=>{let h=s>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
let features=0,fdMax=0,unbiasMax=0,fitSteps=0;
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return h=>{features++;let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<4;j++){let t=j*4,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+4),x.slice(t,t+4))/2));for(let l=t;l<t+4;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {p:mat(W,seq.at(-1)),q:mat(W,s)}}}
function mass(p,q){const m=p.map((x,i)=>Math.min(x,q[i]));const g=m.map((x,i)=>mul((p[i]<=q[i]?p:q).map((v,j)=>+(j===i)-v),x));return {m,g,a:sum(m),G:g.reduce(add,zero(V))}}
function calc(s,pb,qb,theta=zero(V)){
const ms=s.fs.map(f=>mass(sm(add(add(f.p,pb),theta)),sm(add(add(f.q,qb),theta))));
const rt=ms[0],q=sm(add(add(s.fs[0].q,qb),theta));
const a=ms.slice(1).map(x=>x.a),G=ms.slice(1).map(x=>x.G);
const root=rt.G,val=rt.a+dot(rt.m,a);
const t=rt.g.map((g,i)=>mul(g,1/q[i])),S=rt.g.map((g,i)=>mul(add(mul(g,a[i]),mul(G[i],rt.m[i])),1/q[i]));
const mu=S.map((g,i)=>mul(g,q[i])).reduce(add,zero(V));
const T=t.map(g=>sub(g,root)),Z=S.map(g=>sub(g,mu));
return {q,T,Z,root,mu,val,truth:add(root,mu),a,m:rt.m,ce:sm(add(add(s.fs[0].p,pb),theta)).map((x,i)=>x-+(i===s.lab))}
}
function moment(cs){let a=0,b=0,over=0;for(let c of cs){over+=dot(c.q,c.a);for(let i=0;i<V;i++){a+=c.q[i]*dot(c.T[i],c.Z[i]);b+=c.q[i]*dot(c.T[i],c.T[i]);}}return {num:a,den:b,n:cs.length,over};}
function baseline(cs){let m=moment(cs);return m.num/Math.max(m.den,1e-20)}
function varAt(c,b){return sum(c.q.map((w,i)=>{const v=sub(c.Z[i],mul(c.T[i],b));return w*dot(v,v)}))}
function rawVar(c){return sum(c.q.map((w,i)=>w*dot(add(c.Z[i],c.T[i]),add(c.Z[i],c.T[i]))))}

function norm(a){return Math.sqrt(dot(a,a))}
function vecs(c,b){return c.q.map((w,i)=>mul(sub(add(c.Z[i],c.mu),mul(add(c.T[i],c.root),b)),w))}
function normalize(a){return mul(a,1/sum(a))}
function mix(q,r,t){return q.map((v,i)=>(1-t)*v+t*r[i])}
function fit(cs,b){const num=zero(V),den=zero(V);for(const c of cs){const v=vecs(c,b);for(let i=0;i<V;i++){num[i]+=norm(v[i]);den[i]+=c.q[i]}}return num.map((x,i)=>(x+1e-8)/(den[i]+1e-8))}
function pred(c,s){return mix(c.q,normalize(c.q.map((v,i)=>v*s[i])),.5)}
function metric(c,b,r){let v=vecs(c,b),mu=v.reduce(add,zero(V));let z=v.map((w,i)=>sub(mul(w,1/r[i]),mu));let vv=sum(r.map((w,i)=>w*dot(z[i],z[i]))),delta=sum(r.map((w,i)=>Math.abs(w-c.q[i])))/2;let err=z.map((w,i)=>mul(w,r[i])).reduce(add,zero(V));unbiasMax=Math.max(unbiasMax,...err.map(Math.abs));let mismatch=1-sum(c.q.map(w=>w*w));return {v:vv,delta,iid:varAt(c,b)*(1-Math.min(1,delta/mismatch)/2)}}
function audit(c,b,sg,se,sl){let v=vecs(c,b),rstar=normalize(v.map(norm)),r={q:c.q,mass:mix(c.q,normalize(c.m),.5),global:pred(c,sg),edge:pred(c,se),local:pred(c,sl),oracle:mix(c.q,rstar,.9)};return Object.fromEntries(Object.entries(r).map(([k,p])=>[k,metric(c,b,p)]))}

const words=[...new Set(names.toLowerCase().split(/\s+/).filter(w=>/^[a-z]{6,14}$/.test(w)))],rows=[];
for(let wi=0;wi<raws.length;wi++){const feat=model(raws[wi]),cache=new Map();const F=h=>{const key=h.join(",");if(!cache.has(key))cache.set(key,feat(h));return cache.get(key)};
for(let seed of [11,29,73]){
const pools=Array.from({length:4},(_,c)=>words.filter(w=>Math.min(3,Math.floor((w.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,40));
const ex=w=>{let h=[26,...w.slice(0,3).split("").map(x=>x.charCodeAt(0)-97)];return {h,lab:w.charCodeAt(3)-97,f:F(h)}};
const clients=pools.map(p=>p.slice(0,16).map(ex));
const tree=w=>{let e=ex(w);return {...e,fs:[e.f,...Array.from({length:V},(_,i)=>F(e.h.concat(i)))]}};
const cal=pools.map(p=>p.slice(16,28).map(tree)),test=pools.map(p=>p.slice(28,40).map(tree));
for(let eta of [.05,.2,.6]){
const pb=clients.map(cs=>{let b=zero(V);for(let it=0;it<24;it++){let g=zero(V);for(let s of cs){const p=sm(add(s.f.p,b));g=add(g,p.map((v,j)=>+(j===s.lab)-v));}b=add(b,mul(g,eta/cs.length));fitSteps++}return b});
const qb=[0,1].map(e=>{let b=avg(pb.slice(e*2,e*2+2));for(let it=0;it<16;it++){let g=zero(V);for(let c=e*2;c<e*2+2;c++)for(let s of clients[c])g=add(g,sub(sm(add(s.f.p,pb[c])),sm(add(s.f.q,b))));b=add(b,mul(g,.2/32));fitSteps++}return b});
const C=cal.map((cs,c)=>cs.map(s=>calc(s,pb[c],qb[c>>1]))),T=test.map((cs,c)=>cs.map(s=>calc(s,pb[c],qb[c>>1])));
const gB=baseline(C.flat()),eB=[baseline(C[0].concat(C[1])),baseline(C[2].concat(C[3]))],lB=C.map(baseline);
const naiveMean=moment(C.flat()).over/C.flat().length; const sg=fit(C.flat(),gB),se=[fit(C[0].concat(C[1]),gB),fit(C[2].concat(C[3]),gB)],sl=C.map(cs=>fit(cs,gB));
for(let c=0;c<4;c++){
let vs=T[c].map((t,j)=>{let own=baseline([t]);let raw=rawVar(t),root=varAt(t,0),glob=varAt(t,gB),edge=varAt(t,eB[c>>1]),local=varAt(t,lB[c]),heur=varAt(t,naiveMean),oracle=varAt(t,own);
let err=zero(V);for(let i=0;i<V;i++)err=add(err,mul(sub(t.Z[i],mul(t.T[i],eB[c>>1])),t.q[i]));unbiasMax=Math.max(unbiasMax,...err.map(Math.abs));
if(j===0){let h=1e-5;for(let k=0;k<V;k++){let u=zero(V),v=zero(V);u[k]=h;v[k]=-h;let fd=(calc(test[c][j],pb[c],qb[c>>1],u).val-calc(test[c][j],pb[c],qb[c>>1],v).val)/(2*h);fdMax=Math.max(fdMax,Math.abs(fd-t.truth[k]));}}
return {raw,root,glob,edge,local,heur,oracle,eal:t.val, audits:audit(t,gB,sg,se[c>>1],sl[c]), joint:sub(t.ce,mul(t.truth,.3))}});
let jm=avg(vs.map(v=>v.joint)),between=sum(vs.map(v=>{let x=sub(v.joint,jm);return dot(x,x)}))/vs.length;let arms=Object.fromEntries(Object.keys(vs[0].audits).map(k=>[k,Object.fromEntries(["v","delta","iid"].map(j=>[j,sum(vs.map(v=>v.audits[k][j]))/vs.length]))]));rows.push({wi,seed,eta,c,between,arms});
}
}}}
let summary=[.05,.2,.6].map(eta=>{let a=rows.filter(r=>r.eta===eta),between=sum(a.map(r=>r.between))/a.length;let arms=Object.fromEntries(Object.keys(a[0].arms).map(k=>{let v=sum(a.map(r=>r.arms[k].v))/a.length,delta=sum(a.map(r=>r.arms[k].delta))/a.length,iid=sum(a.map(r=>r.arms[k].iid))/a.length;return [k,{v,delta,iid,total:between+.09*v,costProxy:(between+.09*v)*(1+delta),iidCostProxy:(between+.09*iid)*(1+delta)}]}));return {eta,between,arms,edgeWinsGlobal:a.filter(r=>r.arms.edge.v<r.arms.global.v).length,n:a.length}});return {features,fitSteps,fdMax,unbiasMax,summary,rows};}
const fs=require('node:fs');console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
