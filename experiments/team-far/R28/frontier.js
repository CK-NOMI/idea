// node frontier.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
const V=27,d=16,H=4,k=4,N=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v*b[i]*0+v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {h:seq.at(-1),s}}}}
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(a,u){let s=0;for(let i=0;i<a.length;i++){s+=a[i];if(u<s)return i}return a.length-1}
const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
function conv(a,b){let c=zero(a.length+b.length-1);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)c[i+j]+=a[i]*b[j];return c}
const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],rows=[],audit={normalization:0,localLaw:0,naiveMarginalTV:0,maxNaiveJointVoteTV:0,cohorts:0,draws:0,actualFirstDrafts:0,actualSecondDrafts:0};
function verify(x,p,q,rr){let alpha=q[x]===0?1:Math.min(1,p[x]/q[x]);if(rr()<=alpha)return {y:x,ok:true};let r=p.map((v,i)=>Math.max(0,v-q[i]));return {y:cat(scale(r,1/sum(r)),rr()),ok:false}}
function kernel(x,p,q){let alpha=q[x]===0?1:Math.min(1,p[x]/q[x]),r=p.map((v,i)=>Math.max(0,v-q[i])),s=sum(r);return r.map((v,i)=>Number(i===x)*alpha+(s?(1-alpha)*v/s:0))}
function posterior(q,bit,f0,f1){let z=sum(q.map((v,i)=>v*(bit[i]?f1:f0)));return q.map((v,i)=>v*(bit[i]?f1:f0)/z)}
const event=(s,mode,v)=>mode==="count"?s===v:Number(s>=2)===v;
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]),W=m.W,mom=p=>p.reduce((a,v,i)=>add(a,scale(W[i],v)),zero(d));
for(let seed of [11,29,73]){let groups=Array.from({length:N},(_,r)=>words.filter(x=>x.charCodeAt(0)%N===r).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,32));
let make=(name)=>{let j=3,pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],{h,s}=m.feature(pre);return {pre,h,s,y:name.charCodeAt(j)-97}};
let train=groups.map(g=>g.slice(0,16).map(make)),g=scale(train.flat().map(x=>sub(mom(sm(mat(W,x.h))),W[x.y])).reduce(add),1/(16*N)),shift=mat(W,scale(g,-1.5));
let test=groups.map(g=>g.slice(16).map(make)),rr=rng(seed+1000*wi);
for(let c=0;c<16;c++){let xs=test.map(a=>a[c]).map(x=>{let p=sm(mat(W,x.h)),p1=sm(add(mat(W,x.h),shift)),q=sm(mat(W,x.s)),bit=p.map((v,i)=>Number(p1[i]>v)),a=dot(q,bit),next=Array.from({length:V},(_,i)=>{let {h,s}=m.feature(x.pre.concat(i)),l=mat(W,h);return {p:[sm(l),sm(add(l,shift))],q:sm(mat(W,s))}});return {p:[p,p1],q,bit,a,next,y:x.y}}),suff=Array(N+1);suff[N]=[1];for(let i=N-1;i>=0;i--)suff[i]=conv([1-xs[i].a,xs[i].a],suff[i+1]);
let votePatterns=Array.from({length:1<<N},(_,mask)=>{let bits=xs.map((_,i)=>Number(!!(mask&(1<<i)))),w=bits.reduce((v,b,i)=>v*(b?xs[i].a:1-xs[i].a),1);return {bits,w,s:sum(bits)}});
audit.cohorts++;
rows.push({wi,seed,c,xs});
}}} return rows;}

function frontier(cohorts){
const sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),ov=(p,q,w)=>sum(p.map((v,i)=>Math.min(v,q[i])*(w?w[i]:1))),norm=a=>{let z=sum(a);return z?a.map(x=>x/z):a};
let rows=[],maxerr=0,maxDependenceError=0,maxJointError=0;for(let {wi,seed,c,xs} of cohorts){let N=xs.length,M=1<<N,full=M-1,pat=Array.from({length:M},(_,k)=>xs.map((_,i)=>(k>>i)&1)),count=pat.map(sum),prior=pat.map(b=>b.reduce((w,x,i)=>w*(x?xs[i].a:1-xs[i].a),1));
for(let mode of ["threshold","count"]){let rec={wi,seed,c,mode,first:{reveal:0,parallel:0,adaptive:0,canonical:0,fresh:0},block:{reveal:0,parallel:0,adaptive:0,canonical:0,fresh:0},rounds:{canonical:0,adaptive:0,penalty025:0,penalty100:0,penalty025Block:0,penalty100Block:0},informative:0,events:0};
for(let J of mode==="threshold"?[0,1]:Array.from({length:N+1},(_,i)=>i)){let match=k=>mode==="threshold"?Number(count[k]>=N/2)===J:count[k]===J,prob=sum(prior.filter((_,k)=>match(k)));if(prob<1e-14)continue;let v=mode==="threshold"?J:Number(J>=N/2),Q=prior.map((w,k)=>match(k)?w/prob:0),marg=xs.map((_,i)=>dot(Q,pat.map(b=>b[i]))),qc=xs.map(x=>[0,1].map(b=>norm(x.q.map((w,j)=>x.bit[j]===b?w:0))));
let independent=[];for(let mask=0;mask<M;mask++){let ids=xs.map((_,i)=>i).filter(i=>mask&(1<<i)),dist=Array(M).fill(0);for(let k=0;k<M;k++)dist[k&mask]+=Q[k];let e=0;for(let k=0;k<M;k++)if((k&mask)===k){let p=ids.reduce((z,i)=>z*(pat[k][i]?marg[i]:1-marg[i]),1);e=Math.max(e,Math.abs(dist[k]-p))}if(e<1e-11)independent.push(mask)}
let weights=xs.map(x=>x.next.map(z=>1+ov(z.p[v],z.q)));
for(let kind of ["first","block"]){let w=kind==="block"?weights:xs.map(x=>x.q.map(()=>1)),qj=xs.map((x,i)=>qc[i][0].map((z,j)=>(1-marg[i])*z+marg[i]*qc[i][1][j])),reveal=xs.map((x,i)=>(1-marg[i])*ov(qc[i][0],x.p[v],w[i])+marg[i]*ov(qc[i][1],x.p[v],w[i])),gain=xs.map((x,i)=>ov(qj[i],x.p[v],w[i])-reveal[i]);
let score=mask=>sum(gain.filter((_,i)=>mask&(1<<i))),choice=independent.reduce((a,b)=>score(b)>score(a)+1e-12?b:a,0),local=sum(reveal)+score(choice),fresh=sum(xs.map((x,i)=>ov(x.q,x.p[v],w[i])));if(kind==="first")rec.informative+=prob*count[choice];
function kernel(x,p,q){let a=q[x]>0?Math.min(1,p[x]/q[x]):0,r=p.map((v,j)=>Math.max(0,v-q[j])),z=sum(r);return r.map((v,j)=>Number(x===j)*a+(z?(1-a)*v/z:0))}
let outB=Array(M).fill(0),targetB=pat.map(b=>b.reduce((z,t,i)=>{let a=dot(xs[i].p[v],xs[i].bit);return z*(t?a:1-a)},1));
let output=xs.map((x,i)=>[0,1].map(b=>{let qq=choice&(1<<i)?qj[i]:qc[i][b];let y=Array(x.q.length).fill(0);for(let t=0;t<x.q.length;t++)if(qc[i][b][t]){let r=kernel(t,x.p[v],qq);for(let j=0;j<y.length;j++)y[j]+=qc[i][b][t]*r[j]}return y}));
for(let k=0;k<M;k++)if(Q[k])for(let l=0;l<M;l++)outB[l]+=Q[k]*pat[l].reduce((z,t,i)=>{let a=dot(output[i][pat[k][i]],xs[i].bit);return z*(t?a:1-a)},1);
maxJointError=Math.max(maxJointError,sum(outB.map((z,k)=>Math.abs(z-targetB[k])))/2);
for(let i=0;i<N;i++){let pp=xs[i].p[v];maxerr=Math.max(maxerr,...pp.map((z,j)=>Math.abs(z-((1-marg[i])*output[i][0][j]+marg[i]*output[i][1][j]))))}
function solve(penalty,canonical){let memo=new Map();function dp(mask,values){if(mask===full)return {r:0,d:0};let key=mask+","+values;if(memo.has(key))return memo.get(key);let left=xs.map((_,i)=>i).filter(i=>!(mask&(1<<i))),valid=Q.map((z,k)=>(k&mask)===values?z:0),total=sum(valid);if(total<1e-15)return {r:0,d:0};let a=left.map(i=>dot(valid,pat.map(b=>b[i]))/total);let factor=true;for(let k=0;k<M;k++)if((k&mask)===values){let z=left.reduce((s,i,j)=>s*(pat[k][i]?a[j]:1-a[j]),1);if(Math.abs(valid[k]/total-z)>1e-11)factor=false}
let val=(i,b)=>ov(qc[i][0].map((z,j)=>(1-b)*z+b*qc[i][1][j]),xs[i].p[v],w[i]);
if(factor){let o={r:sum(left.map((i,j)=>val(i,a[j]))),d:1};memo.set(key,o);return o}
let best=null;for(let j=0;j<(canonical?1:left.length);j++){let i=left[j],p=a[j],l=dp(mask|(1<<i),values),h=dp(mask|(1<<i),values|(1<<i)),o={r:val(i,p)+(1-p)*l.r+p*h.r,d:1+(1-p)*l.d+p*h.d};if(!best||o.r-penalty*o.d>best.r-penalty*best.d+1e-12||(Math.abs(o.r-penalty*o.d-best.r+penalty*best.d)<1e-12&&o.d<best.d))best=o}memo.set(key,best);return best}return dp(0,0)}
let ca=solve(0,true),ad=solve(0,false);for(let [k,z] of Object.entries({reveal:sum(reveal),parallel:local,adaptive:ad.r,canonical:ca.r,fresh}))rec[kind][k]+=prob*z/N;
if(kind==="block"){let p25=solve(.25,false),p100=solve(1,false);rec.rounds.canonical+=prob*ca.d;rec.rounds.adaptive+=prob*ad.d;rec.rounds.penalty025+=prob*p25.d;rec.rounds.penalty100+=prob*p100.d;rec.rounds.penalty025Block+=prob*p25.r/N;rec.rounds.penalty100Block+=prob*p100.r/N}
}rec.events++}
rows.push(rec)}}
let summary=["threshold","count"].map(mode=>{let a=rows.filter(r=>r.mode===mode),avg=k=>sum(a.map(r=>r[k]))/a.length,agg=kind=>Object.fromEntries(Object.keys(a[0][kind]).map(k=>[k,sum(a.map(r=>r[kind][k]))/a.length]));return {mode,n:a.length,events:sum(a.map(r=>r.events)),first:agg("first"),block:agg("block"),rounds:agg("rounds"),informative:avg("informative")}});return {rows,summary,maxerr,maxJointError};
}
console.log(JSON.stringify(frontier(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
