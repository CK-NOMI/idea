// node scalable.js weights.json checkpoint.json input.txt
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





const sum=a=>a.reduce((s,x)=>s+x,0),ov=(p,q,w)=>sum(p.map((x,i)=>Math.min(x,q[i])*(w?w[i]:1)));
function allocate(p,qc,a,W,w,bit){let seg=[],t=p.map(()=>0);
for(let j=0;j<p.length;j++){let b=bit[j],q=qc[b][j],breaks=[0,p[j],Math.max(0,Math.min(p[j],a*q)),Math.max(0,Math.min(p[j],p[j]-(1-a)*q))].sort((x,y)=>x-y);
for(let k=0;k<breaks.length-1;k++){let l=breaks[k],h=breaks[k+1];if(h<=l)continue;let m=(l+h)/2,s=w[j]*((m/a<q?W[b][1]/a:0)-((p[j]-m)/(1-a)<q?W[b][0]/(1-a):0));seg.push({j,l,h,s})}}
seg.sort((x,y)=>y.s-x.s||x.l-y.l);let rem=a;
for(let {j,l,h,s} of seg){let x=Math.min(rem,h-l);t[j]+=x;rem-=x;if(rem<1e-15)break}
return [p.map((x,j)=>(x-t[j])/(1-a)),t.map(x=>x/a)]}
function getScore(qc,r,W,w){let z=0;for(let b=0;b<2;b++)for(let c=0;c<2;c++)z+=W[b][c]*ov(r[c],qc[b],w);return z}

function scalable(cohorts){
const sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),ov=(p,q,w)=>sum(p.map((x,i)=>Math.min(x,q[i])*(w?w[i]:1))),norm=a=>{let z=sum(a);return z?a.map(x=>x/z):a},conv=(a,b)=>{let z=Array(a.length+b.length-1).fill(0);a.forEach((x,i)=>b.forEach((y,j)=>z[i+j]+=x*y));return z};
function tree(a,lo=0,hi=a.length){if(hi-lo===1)return {lo,hi,A:[1-a[lo],a[lo]]};let mid=(lo+hi)>>1,L=tree(a,lo,mid),R=tree(a,mid,hi);return {lo,hi,L,R,A:conv(L.A,R.A)}}
function split(t,k){return norm(t.L.A.map((x,i)=>x*(t.R.A[k-i]||0)))}
function pairs(a,b){let i=0,j=0,u=a[0],v=b[0],out=[];while(i<a.length&&j<b.length){let z=Math.min(u,v);if(z>0)out.push([i,j,z]);u-=z;v-=z;if(u<=0){i++;u=a[i]||0}if(v<=0){j++;v=b[j]||0}}return out}
function transport(t,root,W,audit){if(!t.L){let m=[[0,0],[0,0]];for(let [k,l,p] of root)m[k][l]+=p;W[t.lo]=m;return}let n=t.L.hi-t.L.lo+1,m=t.R.hi-t.R.lo+1,L=Array.from({length:n},()=>Array(n).fill(0)),R=Array.from({length:m},()=>Array(m).fill(0)),cache=t.A.map((_,i)=>split(t,i));
for(let [k,l,p] of root){for(let [i,j,z] of pairs(cache[k],cache[l])){let r=k-i,s=l-j;if(r<0||r>=m||s<0||s>=m){audit.bad++;audit.badRangeMass+=p*z;continue}L[i][j]+=p*z;R[r][s]+=p*z;audit.work++;if((l-k)*(j-i)<0||(l-k)*(s-r)<0){audit.bad++;audit.badMonotonicMass+=p*z;audit.maxBadMonotonicMass=Math.max(audit.maxBadMonotonicMass,p*z)}}}
let list=M=>M.flatMap((r,i)=>r.map((x,j)=>[i,j,x]).filter(t=>t[2]>0));transport(t.L,list(L),W,audit);transport(t.R,list(R),W,audit)}
let rows=[],audit={bad:0,badRangeMass:0,badMonotonicMass:0,maxBadMonotonicMass:0,work:0,maxVoteMarginal:0,maxCountError:0,maxOutputMarginal:0,maxNorm:0};
const config=new Map();for(let c of cohorts){let key=c.wi+","+c.seed;if(!config.has(key))config.set(key,[]);config.get(key).push(...c.xs)}
for(let [key,all] of config)for(let N of [4,16,64])for(let start=0;start<all.length;start+=N){let xs=all.slice(start,start+N),a=xs.map(x=>x.a),t=tree(a),T=Math.round(sum(a)),pref=[[1]],suff=Array(N+1);suff[N]=[1];for(let i=0;i<N;i++)pref.push(conv(pref[i],[1-a[i],a[i]]));for(let i=N-1;i>=0;i--)suff[i]=conv([1-a[i],a[i]],suff[i+1]);
let rec={key,N,start,T,first:0,block:0,seqFirst:0,seqBlock:0,freshFirst:0,freshBlock:0,upperBlock:0,meanChanged:0,rootLower:0,zeroInfoBlock:0};
for(let J of [0,1]){let event=k=>Number(k>=T)===J,source=t.A.map((x,k)=>event(k)?x:0),PJ=sum(source);if(PJ===0)continue;let root=pairs(norm(source),t.A),W=Array(N);transport(t,root,W,audit);let countCost=sum(root.map(([k,l,p])=>Math.abs(k-l)*p)),leafCost=sum(W.map(m=>m[0][1]+m[1][0]));audit.maxCountError=Math.max(audit.maxCountError,Math.abs(countCost-leafCost));rec.meanChanged+=PJ*leafCost;rec.rootLower+=PJ*countCost;
for(let i=0;i<N;i++){let x=xs[i],p=x.p[J],q=x.q,qc=[0,1].map(b=>norm(q.map((z,j)=>x.bit[j]===b?z:0))),m=W[i],am=sum(m[1]),w=x.next.map(z=>1+ov(z.p[J],z.q)),r=allocate(p,qc,a[i],m,w,x.bit),r1=allocate(p,qc,a[i],m,q.map(()=>1),x.bit);audit.maxVoteMarginal=Math.max(audit.maxVoteMarginal,Math.abs(m[0][1]+m[1][1]-a[i]));
audit.maxOutputMarginal=Math.max(audit.maxOutputMarginal,...p.map((z,j)=>Math.abs(z-(1-a[i])*r[0][j]-a[i]*r[1][j])));audit.maxNorm=Math.max(audit.maxNorm,...r.map(x=>Math.abs(sum(x)-1)));
rec.first+=PJ*getScore(qc,r1,m,q.map(()=>1))/N;rec.block+=PJ*getScore(qc,r,m,w)/N;rec.freshFirst+=PJ*ov(p,q)/N;rec.freshBlock+=PJ*ov(p,q,w)/N;rec.upperBlock+=PJ*ov(p,qc[0].map((z,j)=>(1-am)*z+am*qc[1][j]),w)/N;rec.zeroInfoBlock+=PJ*getScore(qc,[p,p],m,w)/N;
for(let k=0;k<pref[i].length;k++){let e0=sum(suff[i+1].map((z,l)=>event(k+l)?z:0)),e1=sum(suff[i+1].map((z,l)=>event(k+1+l)?z:0)),den=(1-a[i])*e0+a[i]*e1;if(den===0)continue;let u=a[i]*e1/den,qp=qc[0].map((z,j)=>(1-u)*z+u*qc[1][j]),weight=pref[i][k]*den;rec.seqFirst+=weight*ov(p,qp)/N;rec.seqBlock+=weight*ov(p,qp,w)/N}
}}
rows.push(rec)}
let summary=[4,16,64].map(N=>{let r=rows.filter(x=>x.N===N);return {N,groups:r.length,...Object.fromEntries(["first","block","seqFirst","seqBlock","freshFirst","freshBlock","upperBlock","meanChanged","rootLower","zeroInfoBlock"].map(k=>[k,sum(r.map(x=>x[k]))/r.length]))}});
return {rows,summary,audit}}
console.log(JSON.stringify(scalable(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
