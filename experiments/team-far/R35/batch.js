// node batch.js weights.json checkpoint.json input.txt
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
let make=(name)=>{let j=3,pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],{h,s}=m.feature(pre);return {pre,h,s,y:name.charCodeAt(j)-97,y2:name.charCodeAt(j+1)-97,ref:name.slice(j,j+4).split('').map(c=>c.charCodeAt(0)-97)}};
let train=groups.map(g=>g.slice(0,16).map(make)),g=scale(train.flat().map(x=>sub(mom(sm(mat(W,x.h))),W[x.y])).reduce(add),1/(16*N)),shift=mat(W,scale(g,-1.5));
let test=groups.map(g=>g.slice(16).map(make)),rr=rng(seed+1000*wi);
for(let c=0;c<16;c++){let xs=test.map(a=>a[c]).map(x=>{let p=sm(mat(W,x.h)),p1=sm(add(mat(W,x.h),shift)),q=sm(mat(W,x.s)),bit=p.map((v,i)=>Number(p1[i]>v)),a=dot(q,bit),next=Array.from({length:V},(_,i)=>{let {h,s}=m.feature(x.pre.concat(i)),l=mat(W,h);return {p:[sm(l),sm(add(l,shift))],q:sm(mat(W,s))}});let path=x.ref.map((v,j)=>{let {h,s}=m.feature(x.pre.concat(x.ref.slice(0,j)));return {p:sm(mat(W,h)),q:sm(mat(W,s)),v}});return {p:[p,p1],q,bit,a,next,y:x.y,y2:x.y2,path}}),suff=Array(N+1);suff[N]=[1];for(let i=N-1;i>=0;i--)suff[i]=conv([1-xs[i].a,xs[i].a],suff[i+1]);
let votePatterns=Array.from({length:1<<N},(_,mask)=>{let bits=xs.map((_,i)=>Number(!!(mask&(1<<i)))),w=bits.reduce((v,b,i)=>v*(b?xs[i].a:1-xs[i].a),1);return {bits,w,s:sum(bits)}});
audit.cohorts++;
rows.push({wi,seed,c,xs});
}}} return rows;}











function batchStudy(cohorts){
const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
function bin(n,p){let a=[1];for(let i=0;i<n;i++){let b=Array(a.length+1).fill(0);a.forEach((x,j)=>{b[j]+=x*(1-p);b[j+1]+=x*p});a=b}return a}
function solve(prob,N,a,policy=null){let L=prob.length,C=Array.from({length:L+1},()=>Array(N+1).fill(0)),K=C.map(x=>x.slice());
for(let t=L-1;t>=0;t--)for(let m=1;m<=N;m++){let best=Infinity,r=1,pr=1;for(let k=1;k<=L-t;k++){pr*=prob[t+k-1];if(policy&&k!==Math.min(policy(t,m),L-t))continue;let z=a+k+sum(bin(m,pr).map((x,j)=>x*C[t+k][j]));if(z<best){best=z;r=k}}C[t][m]=best;K[t][m]=r}return {cost:C[0][N],K}}
let rows=[],summary=[];
for(let {wi,seed,c,xs} of cohorts)for(let [ix,x] of xs.entries())for(let N of [4,8,16])for(let a of [.25,1,4,16]){let p=x.path.map(z=>z.p[z.v]),q=x.path.map(z=>z.q[z.v]),oracle=solve(p,N,a),draft=solve(q,N,a),evalq=solve(p,N,a,(t,m)=>draft.K[t][m]),fixed=Array.from({length:p.length},(_,j)=>solve(p,N,a,()=>j+1).cost);rows.push({wi,seed,c,ix,N,a,oracle:oracle.cost,q:evalq.cost,fixed,qWithOverhead:evalq.cost+2,full:fixed.at(-1),one:fixed[0]})}
for(let N of [4,8,16])for(let a of [.25,1,4,16]){let r=rows.filter(x=>x.N===N&&x.a===a),fixed=r[0].fixed.map((_,j)=>mean(r.map(x=>x.fixed[j]))),best=Math.min(...fixed),q=mean(r.map(x=>x.q)),o=mean(r.map(x=>x.oracle));summary.push({N,a,fixed,bestFixed:best,bestK:fixed.indexOf(best)+1,q,oracle:o,qSaving:1-q/best,oracleSaving:1-o/best,qWithOverhead:q+2,maxAllowableQOverhead:best-q})}return {summary,rows}}
console.log(JSON.stringify(batchStudy(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));