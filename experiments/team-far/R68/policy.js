function run(raws,names){
const V=27,d=16,H=4,k=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let shallow,lastF;let seq=h.map((t,i)=>rms(add(E[t],P[i])));for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));let f=mat(b.f1,rms(y)).map(v=>Math.max(0,v));if(i===seq.length-1)lastF=f;return add(y,mat(b.f2,f))});if(!shallow)shallow=seq.at(-1).slice()}return {deep:seq.at(-1),shallow,f:lastF}}}}




const avg=a=>sum(a)/a.length,hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h},words=[...new Set(names.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],D=1024;
function fitMap(data,K){let f=Array.from({length:V},()=>Array(V).fill(.01));for(let n of data)for(let i=1;i<n.length;i++)f[n.charCodeAt(i-1)-97][n.charCodeAt(i)-97]++;let g=f.slice(0,26).map((v,i)=>({ids:[i],v}));let ent=v=>{let n=sum(v);return -sum(v.map(x=>x*Math.log(x/n)))};while(g.length>K-1){let best=Infinity,bi=0,bj=1;for(let i=0;i<g.length;i++)for(let j=i+1;j<g.length;j++){let a=g[i].v,b=g[j].v,c=add(a,b),z=ent(c)-ent(a)-ent(b);if(z<best){best=z;bi=i;bj=j}}g[bi]={ids:g[bi].ids.concat(g[bj].ids),v:add(g[bi].v,g[bj].v)};g.splice(bj,1)}let out=zero(V);g.forEach((a,i)=>a.ids.forEach(x=>out[x]=i));out[26]=K-1;return out}

function rng(seed){let s=seed;return ()=>((s=Math.imul(s,1664525)+1013904223>>>0)/4294967296)}
function sample(p,u){let v=u*sum(p);for(let i=0;i<p.length;i++){v-=p[i];if(v<0)return i}return p.findLastIndex(x=>x>0)}
let rows=[],forwards=0,rollouts=0,gradientChecks=0,maxGrad=0,maxNorm=0;
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]),W=m.W;for(let seed of [11,29,73]){let data=words.slice().sort((a,b)=>hash(a,seed)-hash(b,seed));for(let K of [4,8,16]){let g=fitMap(data.slice(0,256),K),ids=Array.from({length:K},()=>[]);g.forEach((c,i)=>ids[c].push(i));let reps=ids.map(a=>a[0]);for(let N of [8,32]){
let r=rng(seed+wi*17+K*43+N),tot={tokens:0,round:0,verifyRows:0,uniqueVerify:0,uniqueCommitted:0,rawUnique:0,localClass:0,edgeClass:0,globalClass:0,retained:0};
for(let name of data.slice(256,264)){let anchor=[26,...name.slice(0,2).split("").map(x=>x.charCodeAt(0)-97)],bank=new Map(),emitBank=new Map(),vk=new Set(),ck=new Set(),rk=new Set(),all=[],local=Array.from({length:4},()=>new Set()),edge=Array.from({length:2},()=>new Set());
function dist(h){let key=h.join(",");if(bank.has(key))return bank.get(key);let f=m.feature(anchor.concat(h.map(c=>reps[c])));forwards++;let raw=sm(mat(W,f.deep)),sh=sm(mat(W,f.shallow)),p=zero(K),q=zero(K);raw.forEach((v,i)=>p[g[i]]+=v);sh.forEach((v,i)=>q[g[i]]+=v);maxNorm=Math.max(maxNorm,Math.abs(sum(p)-1),Math.abs(sum(q)-1));let a={p,q,raw,f};bank.set(key,a);return a}
function emit(h,c){let key=h.join(",");if(!emitBank.has(key)){emitBank.set(key,m.feature(anchor.concat(h)).shallow);forwards++}let a=emitBank.get(key);return ids[c][sample(sm(ids[c].map(i=>dot(W[i],a))),r())]}
for(let j=0;j<N;j++){let h=[],fine=[],path=[],cl=j%4;rollouts++;while(h.length<4){let k=4-h.length-1,xs=[],z=h.slice();for(let l=0;l<k;l++){let a=dist(z),x=sample(a.q,r());xs.push(x);z.push(x)}
tot.round++;let v=h.slice();for(let l=0;l<=k;l++){vk.add(v.join(","));dist(v);tot.verifyRows++;if(l<k)v.push(xs[l])}
let ok=true;for(let x of xs){let a=dist(h),c=x;if(r()>=Math.min(1,a.p[x]/a.q[x])){c=sample(a.p.map((p,i)=>Math.max(0,p-a.q[i])),r());ok=false}let key=h.join(",");path.push({key,a,c});ck.add(key);rk.add(fine.join(","));local[cl].add(key);edge[cl>>1].add(key);fine.push(emit(fine,c));h.push(c);if(!ok)break}
if(ok&&h.length<4){let a=dist(h),c=sample(a.p,r()),key=h.join(",");path.push({key,a,c});ck.add(key);rk.add(fine.join(","));local[cl].add(key);edge[cl>>1].add(key);fine.push(emit(fine,c));h.push(c)}}
let reward=+(fine.filter(x=>[0,4,8,14,20].includes(x)).length>=2);for(let p of path)all.push({...p,reward});tot.tokens+=4}
function grad(a,counts,n,out){let rr=a.raw.map((p,i)=>p*(counts[g[i]]/a.p[g[i]]-n)),u=Array.from({length:d},(_,j)=>sum(rr.map((x,i)=>x*W[i][j])));for(let j=0;j<d;j++)for(let k=0;k<64;k++)out[j*64+k]+=u[j]*a.f.f[k]}
let raw=zero(D),grouped=zero(D),groups=new Map();for(let x of all){let c=zero(K);c[x.c]=x.reward;grad(x.a,c,x.reward,raw);if(!groups.has(x.key))groups.set(x.key,{a:x.a,n:0,c:zero(K)});let b=groups.get(x.key);b.n+=x.reward;b.c[x.c]+=x.reward}
for(let b of groups.values())grad(b.a,b.c,b.n,grouped);gradientChecks++;maxGrad=Math.max(maxGrad,...raw.map((x,i)=>Math.abs(x-grouped[i])));tot.uniqueVerify+=vk.size;tot.uniqueCommitted+=ck.size;tot.rawUnique+=rk.size;tot.localClass+=sum(local.map(x=>x.size));tot.edgeClass+=sum(edge.map(x=>x.size));tot.globalClass+=groups.size;tot.retained+=[...ck].filter(x=>vk.has(x)).length}
rows.push({wi,seed,K,N,...tot})}
}}}
let summary=[];for(let K of [4,8,16])for(let N of [8,32]){let a=rows.filter(x=>x.K===K&&x.N===N);summary.push({K,N,n:a.length,...Object.fromEntries(["round","verifyRows","uniqueVerify","uniqueCommitted","rawUnique","localClass","edgeClass","globalClass"].map(k=>[k,avg(a.map(x=>x[k]/x.tokens))])),extraVsRaw:avg(a.map(x=>1-x.uniqueCommitted/x.rawUnique)),extraVerifyVsClassAR:avg(a.map(x=>x.uniqueVerify/x.uniqueCommitted-1)),coverage:avg(a.map(x=>x.retained/x.uniqueCommitted))})}
return {rows,summary,forwards,rollouts,gradientChecks,maxGrad,maxNorm};
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("fs");console.log(JSON.stringify(run([JSON.parse(fs.readFileSync(process.argv[2],"utf8")),JSON.parse(fs.readFileSync(process.argv[3],"utf8")).weights],fs.readFileSync(process.argv[4],"utf8")),null,2))}
