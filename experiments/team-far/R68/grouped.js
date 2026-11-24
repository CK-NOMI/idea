function run(raws,names){
const V=27,d=16,H=4,k=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let shallow,lastF;let seq=h.map((t,i)=>rms(add(E[t],P[i])));for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));let f=mat(b.f1,rms(y)).map(v=>Math.max(0,v));if(i===seq.length-1)lastF=f;return add(y,mat(b.f2,f))});if(!shallow)shallow=seq.at(-1).slice()}return {deep:seq.at(-1),shallow,f:lastF}}}}



const avg=a=>sum(a)/a.length,hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h},words=[...new Set(names.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],D=d*4*d;
let rows=[],forwards=0,updates=0,maxGrad=0,maxParam=0,maxFD=0;
function fitMap(data,K){let f=Array.from({length:V},()=>Array(V).fill(.01));for(let n of data)for(let i=1;i<n.length;i++)f[n.charCodeAt(i-1)-97][n.charCodeAt(i)-97]++;let g=f.slice(0,26).map((v,i)=>({ids:[i],v}));let ent=v=>{let n=sum(v);return -sum(v.map(x=>x*Math.log(x/n)))};while(g.length>K-1){let best=Infinity,bi=0,bj=1;for(let i=0;i<g.length;i++)for(let j=i+1;j<g.length;j++){let a=g[i].v,b=g[j].v,c=add(a,b),z=ent(c)-ent(a)-ent(b);if(z<best){best=z;bi=i;bj=j}}g[bi]={ids:g[bi].ids.concat(g[bj].ids),v:add(g[bi].v,g[bj].v)};g.splice(bj,1)}let out=zero(V);g.forEach((a,i)=>a.ids.forEach(x=>out[x]=i));out[26]=K-1;return out}
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]),W=m.W;for(let seed of [11,29,73]){
let clients=Array.from({length:4},(_,r)=>words.filter(x=>{let v=x.charCodeAt(x.length-1)-97;return (v>=13?2:0)+v%2===r}).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,96));let tr=clients.map(a=>a.slice(0,64)),te=clients.map(a=>a.slice(64,96));
for(let K of [4,8,16])for(let A of [1,2]){let g=fitMap(tr.flat(),K),ids=Array.from({length:K},()=>[]);g.forEach((c,i)=>ids[c].push(i));let reps=ids.map(a=>a[0]),cache=new Map();
function feature(h){let k=h.join(",");if(!cache.has(k)){let a=m.feature(h);if(!a.f||a.f.length!==64)throw Error("feature");cache.set(k,a);forwards++}return cache.get(k)}
function examples(a){let o=[];for(let n of a){let x=[...n].map(c=>c.charCodeAt(0)-97),anchor=[26,...x.slice(0,A)];for(let j=0;j<4;j++){let h=x.slice(A,A+j),y=x[A+j],raw=anchor.concat(h),coarse=anchor.concat(h.map(x=>reps[g[x]]));o.push({raw:raw.join(","),coarse:coarse.join(","),y,c:g[y],f:feature(coarse),r:feature(raw)})}}return o}
let train=tr.map(examples),test=te.map(examples);
function logits(a,t){let z=a.deep.map((x,i)=>x+dot(t.slice(i*64,(i+1)*64),a.f));return sm(mat(W,z))}
function group(a,key){let map=new Map();for(let x of a){let k=x[key];if(!map.has(k))map.set(k,{f:key==="raw"?x.r:x.f,n:0,counts:zero(K)});let b=map.get(k);b.n++;b.counts[x.c]++}return [...map.values()]}
function calc(b,t,grad){let p=logits(b.f,t),mass=zero(K);p.forEach((x,i)=>mass[g[i]]+=x);let loss=-sum(b.counts.map((x,i)=>x*Math.log(mass[i])));if(grad){let r=p.map((x,i)=>x*(b.n-b.counts[g[i]]/mass[g[i]])),u=Array.from({length:d},(_,j)=>sum(r.map((x,i)=>x*W[i][j])));for(let j=0;j<d;j++)for(let k=0;k<64;k++)grad[j*64+k]+=u[j]*b.f.f[k]}return loss}
function single(a){return a.map(x=>({f:x.f,n:1,counts:zero(K).map((_,i)=>+(i===x.c))}))}
function agg(groups,t){let grad=zero(D),loss=0;for(let b of groups)loss+=calc(b,t,grad);return {grad,loss}}
let t=zero(D),u=zero(D),cnt={n:0,rawLocal:0,coarseLocal:0,rawEdge:0,coarseEdge:0,rawGlobal:0,coarseGlobal:0,bytesCount:0,bytesGrad:0};
function evalT(t){let a=test.flat(),loss=0,teacher=0,classLoss=0;for(let x of a){let p=logits(x.f,t),c=sum(ids[x.c].map(i=>p[i])),ep=sm(ids[x.c].map(i=>dot(W[i],x.r.shallow)));loss-=Math.log(c*ep[ids[x.c].indexOf(x.y)]);classLoss-=Math.log(c);teacher-=Math.log(sm(mat(W,x.r.deep))[x.y])}return {nll:loss/a.length,teacher:teacher/a.length,classNLL:classLoss/a.length}}
let initial=evalT(t);
for(let step=0;step<8;step++){let batch=train.map(a=>a.slice(step*32,(step+1)*32)),all=batch.flat(),groups=group(all,"coarse"),a=agg(single(all),t),b=agg(groups,u);maxGrad=Math.max(maxGrad,...a.grad.map((x,i)=>Math.abs(x-b.grad[i])));if(step===0){for(let j of [0,33,100,401,767,1023]){let v=u.slice();v[j]+=1e-5;let l1=agg(groups,v).loss;v[j]-=2e-5;let l2=agg(groups,v).loss;maxFD=Math.max(maxFD,Math.abs(b.grad[j]-(l1-l2)/2e-5))}}
t=add(t,scale(a.grad,-.05/all.length));u=add(u,scale(b.grad,-.05/all.length));maxParam=Math.max(maxParam,...t.map((x,i)=>Math.abs(x-u[i])));updates++;
cnt.n+=all.length;for(let x of batch){cnt.rawLocal+=group(x,"raw").length;cnt.coarseLocal+=group(x,"coarse").length;for(let k of group(x,"coarse"))cnt.bytesCount+=4*(K+2);cnt.bytesGrad+=4*D}
for(let e=0;e<2;e++){let x=batch[2*e].concat(batch[2*e+1]);cnt.rawEdge+=group(x,"raw").length;cnt.coarseEdge+=group(x,"coarse").length}cnt.rawGlobal+=group(all,"raw").length;cnt.coarseGlobal+=groups.length}
rows.push({wi,seed,K,A,initial,final:evalT(u),cnt})}
}}
let summary=[];for(let K of [4,8,16])for(let A of [1,2]){let a=rows.filter(x=>x.K===K&&x.A===A);summary.push({K,A,n:a.length,initial:avg(a.map(x=>x.initial.nll)),final:avg(a.map(x=>x.final.nll)),teacher:avg(a.map(x=>x.final.teacher)),...Object.fromEntries(["rawLocal","coarseLocal","rawEdge","coarseEdge","rawGlobal","coarseGlobal"].map(k=>[k,avg(a.map(x=>x.cnt[k]/x.cnt.n))])),countBytesRatio:avg(a.map(x=>x.cnt.bytesCount/x.cnt.bytesGrad))})}
return {rows,summary,forwards,updates,maxGrad,maxParam,maxFD};
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("fs");console.log(JSON.stringify(run([JSON.parse(fs.readFileSync(process.argv[2],"utf8")),JSON.parse(fs.readFileSync(process.argv[3],"utf8")).weights],fs.readFileSync(process.argv[4],"utf8")),null,2))}
