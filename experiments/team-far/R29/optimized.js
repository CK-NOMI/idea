// node optimized.js weights.json checkpoint.json input.txt
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




function optimized(cohorts){
const sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),ov=(p,q,w)=>sum(p.map((x,i)=>Math.min(x,q[i])*(w?w[i]:1))),norm=a=>{let z=sum(a);return z?a.map(x=>x/z):a},conv=(a,b)=>{let z=Array(a.length+b.length-1).fill(0);a.forEach((x,i)=>b.forEach((y,j)=>z[i+j]+=x*y));return z};
function tree(a,lo=0,hi=a.length){if(hi-lo===1)return {lo,hi,A:[1-a[lo],a[lo]]};let mid=(lo+hi)>>1,L=tree(a,lo,mid),R=tree(a,mid,hi);return {lo,hi,L,R,A:conv(L.A,R.A)}}
function split(t,k){return norm(t.L.A.map((x,i)=>x*(t.R.A[k-i]||0)))}
function couple(a,b){let l=0,r=0,g=a.map(()=>b.map(()=>0));for(let i=0;i<a.length;i++){let hi=l+a[i];r=0;for(let j=0;j<b.length;j++){let hj=r+b[j];g[i][j]=Math.max(0,Math.min(hi,hj)-Math.max(l,r));r=hj}l=hi}return g}
function dist(t,mask,k,l){if(t.hi-t.lo===1)return [[l<<t.lo,1]];let s=0;for(let i=t.L.lo;i<t.L.hi;i++)s+=(mask>>i)&1;let c=norm(couple(split(t,k),split(t,l))[s]),out=[];for(let j=0;j<c.length;j++)if(c[j]>1e-14){for(let [x,p] of dist(t.L,mask,s,j))for(let [y,q] of dist(t.R,mask,k-s,l-j))out.push([x|y,c[j]*p*q])}return out}
function allocate(p,qc,a,W,w,bit){let seg=[],t=p.map(()=>0);
for(let j=0;j<p.length;j++){let b=bit[j],q=qc[b][j],breaks=[0,p[j],Math.max(0,Math.min(p[j],a*q)),Math.max(0,Math.min(p[j],p[j]-(1-a)*q))].sort((x,y)=>x-y);
for(let k=0;k<breaks.length-1;k++){let l=breaks[k],h=breaks[k+1];if(h<=l)continue;let m=(l+h)/2,s=w[j]*((m/a<q?W[b][1]/a:0)-((p[j]-m)/(1-a)<q?W[b][0]/(1-a):0));seg.push({j,l,h,s})}}
seg.sort((x,y)=>y.s-x.s||x.l-y.l);let rem=a;
for(let {j,l,h,s} of seg){let x=Math.min(rem,h-l);t[j]+=x;rem-=x;if(rem<1e-15)break}
return [p.map((x,j)=>(x-t[j])/(1-a)),t.map(x=>x/a)]}
function getScore(qc,r,W,w){let z=0;for(let b=0;b<2;b++)for(let c=0;c<2;c++)z+=W[b][c]*ov(r[c],qc[b],w);return z}
let rows=[],audit={maxMarginalError:0,maxRNormalization:0,maxBelowStandard:0,maxUpperViolation:0,maxTargetVoteTV:0};
for(let {wi,seed,c,xs} of cohorts){let N=xs.length,M=1<<N,pat=Array.from({length:M},(_,k)=>xs.map((_,i)=>(k>>i)&1)),cnt=pat.map(sum),a=xs.map(x=>x.a),prior=pat.map(b=>b.reduce((w,x,i)=>w*(x?a[i]:1-a[i]),1)),t=tree(a);
for(let mode of ["threshold","count"]){let rec={wi,seed,c,mode,first:{independent:0,standard:0,optimized:0,upper:0},block:{independent:0,standard:0,optimized:0,upper:0},events:0};
for(let J of mode==="threshold"?[0,1]:[0,1,2,3,4]){let match=k=>mode==="threshold"?Number(k>=N/2)===J:k===J,PJ=sum(prior.filter((_,i)=>match(cnt[i])));if(PJ<1e-14)continue;let v=mode==="threshold"?J:Number(J>=2),source=norm(t.A.map((x,k)=>match(k)?x:0)),g=couple(source,t.A),W=xs.map(()=>[[0,0],[0,0]]),out=Array(M).fill(0);
for(let m=0;m<M;m++)if(match(cnt[m])&&prior[m]>1e-15){let k=cnt[m];for(let l=0;l<=N;l++)if(g[k][l]>1e-15){let pr=prior[m]/PJ*g[k][l]/source[k];for(let [n,r] of dist(t,m,k,l)){let prob=pr*r;out[n]+=prob;for(let i=0;i<N;i++)W[i][pat[m][i]][pat[n][i]]+=prob}}}
audit.maxTargetVoteTV=Math.max(audit.maxTargetVoteTV,sum(out.map((z,i)=>Math.abs(z-prior[i])))/2);
for(let i=0;i<N;i++){let x=xs[i],p=x.p[v],q=x.q,qc=[0,1].map(b=>norm(q.map((z,j)=>x.bit[j]===b?z:0))),res=p.map((z,j)=>Math.max(0,z-q[j])),z=sum(res),rstd=qc.map(c=>{let rr=p.map(()=>0);for(let j=0;j<q.length;j++)if(c[j]){let ac=Math.min(1,p[j]/q[j]);rr[j]+=c[j]*ac;if(z)for(let k=0;k<q.length;k++)rr[k]+=c[j]*(1-ac)*res[k]/z}return rr}),am=sum(W[i][1]),marg=q.map((_,j)=>(1-am)*qc[0][j]+am*qc[1][j]);
for(let kind of ["first","block"]){let w=kind==="first"?q.map(()=>1):x.next.map(z=>1+ov(z.p[v],z.q)),r=allocate(p,qc,a[i],W[i],w,x.bit),val=getScore(qc,r,W[i],w),std=getScore(qc,rstd,W[i],w),ind=getScore(qc,[p,p],W[i],w),upper=ov(p,marg,w);
audit.maxMarginalError=Math.max(audit.maxMarginalError,...p.map((z,j)=>Math.abs(z-(1-a[i])*r[0][j]-a[i]*r[1][j])));audit.maxRNormalization=Math.max(audit.maxRNormalization,...r.map(a=>Math.abs(sum(a)-1)));audit.maxBelowStandard=Math.max(audit.maxBelowStandard,std-val);audit.maxUpperViolation=Math.max(audit.maxUpperViolation,val-upper);
for(let [k,z] of Object.entries({independent:ind,standard:std,optimized:val,upper}))rec[kind][k]+=PJ*z/N}}
rec.events++}
rows.push(rec)}}
let summary=["threshold","count"].map(mode=>{let a=rows.filter(r=>r.mode===mode);return {mode,n:a.length,...Object.fromEntries(["first","block"].map(kind=>[kind,Object.fromEntries(Object.keys(a[0][kind]).map(k=>[k,sum(a.map(r=>r[kind][k]))/a.length]))]))}});
return {rows,summary,audit}}
console.log(JSON.stringify(optimized(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
