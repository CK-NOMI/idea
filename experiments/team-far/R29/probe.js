// node probe.js weights.json checkpoint.json input.txt
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



function study(cohorts){
const sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),ov=(p,q,w)=>sum(p.map((x,i)=>Math.min(x,q[i])*(w?w[i]:1))),norm=a=>{let z=sum(a);return z?a.map(x=>x/z):a},conv=(a,b)=>{let z=Array(a.length+b.length-1).fill(0);a.forEach((x,i)=>b.forEach((y,j)=>z[i+j]+=x*y));return z};
function tree(a,b,lo=0,hi=a.length){if(hi-lo===1)return {lo,hi,A:[1-a[lo],a[lo]],B:[1-b[lo],b[lo]]};let mid=(lo+hi)>>1,L=tree(a,b,lo,mid),R=tree(a,b,mid,hi);return {lo,hi,L,R,A:conv(L.A,R.A),B:conv(L.B,R.B)}}
function split(t,k,key){return norm(t.L[key].map((x,i)=>x*(t.R[key][k-i]||0)))}
function couple(a,b){let x=0,y=0,l=0,r=0,g=a.map(()=>b.map(()=>0));for(let i=0;i<a.length;i++){let hi=l+a[i];r=0;for(let j=0;j<b.length;j++){let hj=r+b[j];g[i][j]=Math.max(0,Math.min(hi,hj)-Math.max(l,r));r=hj}l=hi}return g}
function conditional(a,b,i){return norm(couple(a,b)[i])}
function distribution(t,mask,k,l){if(t.hi-t.lo===1)return [[l<<t.lo,1]];let s=0;for(let i=t.L.lo;i<t.L.hi;i++)s+=(mask>>i)&1;let c=conditional(split(t,k,"A"),split(t,l,"B"),s),out=[];for(let j=0;j<c.length;j++)if(c[j]>1e-14){let L=distribution(t.L,mask,s,j),R=distribution(t.R,mask,k-s,l-j);for(let [x,p] of L)for(let [y,q] of R)out.push([x|y,c[j]*p*q])}return out}
let rows=[],audit={maxTargetTV:0,maxBridgeTV:0,maxMonotonicViolation:0,maxCountCostError:0,maxCountCouplingMarginal:0};
for(let {wi,seed,c,xs} of cohorts){let N=xs.length,M=1<<N,pat=Array.from({length:M},(_,k)=>xs.map((_,i)=>(k>>i)&1)),cnt=pat.map(sum),a=xs.map(x=>x.a),prior=pat.map(b=>b.reduce((w,x,i)=>w*(x?a[i]:1-a[i]),1));
for(let mode of ["threshold","count"]){let row={wi,seed,c,mode,bridgeFirst:0,bridgeBlock:0,directFirst:0,directBlock:0,baselineFirst:0,baselineBlock:0,changed:0,lower:0,events:0};
for(let J of mode==="threshold"?[0,1]:[0,1,2,3,4]){let match=k=>mode==="threshold"?Number(k>=N/2)===J:k===J,PJ=sum(prior.filter((_,i)=>match(cnt[i])));if(PJ<1e-14)continue;let v=mode==="threshold"?J:Number(J>=N/2),b=xs.map(x=>dot(x.p[v],x.bit)),qt=tree(a,a),pt=tree(a,b),source=norm(qt.A.map((x,k)=>match(k)?x:0)),w=xs.map(x=>x.next.map(z=>1+ov(z.p[v],z.q))),qcond=xs.map(x=>[0,1].map(d=>norm(x.q.map((z,i)=>x.bit[i]===d?z:0)))),pcond=xs.map(x=>[0,1].map(d=>norm(x.p[v].map((z,i)=>x.bit[i]===d?z:0))));
let sourceQ=prior.map((z,i)=>match(cnt[i])?z/PJ:0),targetP=pat.map(t=>t.reduce((z,d,i)=>z*(d?b[i]:1-b[i]),1));
for(let [type,t] of [["bridge",qt],["direct",pt]]){let g=couple(source,t.B),out=Array(M).fill(0),first=0,block=0,changed=0,lower=0;
for(let k=0;k<=N;k++)for(let l=0;l<=N;l++)lower+=g[k][l]*Math.abs(k-l);
for(let m=0;m<M;m++)if(sourceQ[m]>1e-15){let k=cnt[m];for(let l=0;l<=N;l++)if(g[k][l]>1e-15){let pr=sourceQ[m]*g[k][l]/source[k];for(let [n,r] of distribution(t,m,k,l)){let prob=pr*r;out[n]+=prob;let d=0;for(let i=0;i<N;i++){let old=pat[m][i],now=pat[n][i];if(old!==now){d++;continue}let prop=qcond[i][old],p=type==="bridge"?xs[i].p[v]:pcond[i][now];let diag=prop.map((z,j)=>type==="bridge"?z*Math.min(1,xs[i].q[j]?p[j]/xs[i].q[j]:0):Math.min(z,p[j]));first+=prob*sum(diag)/N;block+=prob*dot(diag,w[i])/N}changed+=prob*d;if(type==="bridge"&&d!==Math.abs(k-l))audit.maxMonotonicViolation=Math.max(audit.maxMonotonicViolation,prob)}}}
let tv=sum(out.map((x,i)=>Math.abs(x-(type==="bridge"?prior[i]:targetP[i]))))/2;
if(type==="bridge"){audit.maxBridgeTV=Math.max(audit.maxBridgeTV,tv);audit.maxCountCostError=Math.max(audit.maxCountCostError,Math.abs(changed-lower));row.changed+=PJ*changed;row.lower+=PJ*lower}else audit.maxTargetTV=Math.max(audit.maxTargetTV,tv);
row[type+"First"]+=PJ*first;row[type+"Block"]+=PJ*block}
row.baselineFirst+=PJ*sum(xs.map(x=>ov(x.p[v],x.q)))/N;row.baselineBlock+=PJ*sum(xs.map((x,i)=>ov(x.p[v],x.q,w[i])))/N;row.events++}
rows.push(row)}}
let summary=["threshold","count"].map(mode=>{let r=rows.filter(x=>x.mode===mode);return {mode,n:r.length,...Object.fromEntries(["bridgeFirst","bridgeBlock","directFirst","directBlock","baselineFirst","baselineBlock","changed","lower"].map(k=>[k,sum(r.map(x=>x[k]))/r.length]))}});
function rng(seed){let z=seed>>>0;return ()=>((z=(Math.imul(z,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(a,u){let s=0;for(let i=0;i<a.length;i++){s+=a[i];if(u<s)return i}return a.length-1}
function sampleTree(t,B,k,l,r){if(!t.L)return [l];let s=sum(B.slice(t.L.lo,t.L.hi)),j=cat(conditional(split(t,k,"A"),split(t,l,"B"),s),r());return sampleTree(t.L,B,s,j,r).concat(sampleTree(t.R,B,k-s,l-j,r))}
let scaling=[];for(let N of [4,16,64,128])for(let seed of [11,29,73]){let r=rng(seed+N),a=Array.from({length:N},()=>.05+.9*r()),t=tree(a,a),T=Math.round(sum(a)),dist=t.A,srcs=[norm(dist.map((x,k)=>k<T?x:0)),norm(dist.map((x,k)=>k>=T?x:0))],target=Array(N).fill(0),pair=0,change=0,lb=0,bad=0,draws=1024;
for(let rep=0;rep<draws;rep++){let B=a.map(x=>Number(r()<x)),k=sum(B),J=Number(k>=T),l=cat(conditional(srcs[J],dist,k),r()),C=sampleTree(t,B,k,l,r),d=sum(B.map((x,i)=>Number(x!==C[i])));change+=d;lb+=Math.abs(k-l);bad+=Number(d!==Math.abs(k-l)||sum(C)!==l);C.forEach((x,i)=>target[i]+=x);pair+=C[0]*C[1]}
scaling.push({N,seed,draws,meanChanged:change/draws,lower:lb/draws,bad,maxMarginalError:Math.max(...target.map((x,i)=>Math.abs(x/draws-a[i]))),pairError:Math.abs(pair/draws-a[0]*a[1])})}
return {summary,rows,audit,scaling}
}
console.log(JSON.stringify(study(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
