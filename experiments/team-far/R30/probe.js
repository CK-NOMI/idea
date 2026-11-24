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





const sum=a=>a.reduce((s,x)=>s+x,0),ov=(p,q,w)=>sum(p.map((x,i)=>Math.min(x,q[i])*(w?w[i]:1)));
function allocate(p,qc,a,W,w,bit){let seg=[],t=p.map(()=>0);
for(let j=0;j<p.length;j++){let b=bit[j],q=qc[b][j],breaks=[0,p[j],Math.max(0,Math.min(p[j],a*q)),Math.max(0,Math.min(p[j],p[j]-(1-a)*q))].sort((x,y)=>x-y);
for(let k=0;k<breaks.length-1;k++){let l=breaks[k],h=breaks[k+1];if(h<=l)continue;let m=(l+h)/2,s=w[j]*((m/a<q?W[b][1]/a:0)-((p[j]-m)/(1-a)<q?W[b][0]/(1-a):0));seg.push({j,l,h,s})}}
seg.sort((x,y)=>y.s-x.s||x.l-y.l);let rem=a;
for(let {j,l,h,s} of seg){let x=Math.min(rem,h-l);t[j]+=x;rem-=x;if(rem<1e-15)break}
return [p.map((x,j)=>(x-t[j])/(1-a)),t.map(x=>x/a)]}
function getScore(qc,r,W,w){let z=0;for(let b=0;b<2;b++)for(let c=0;c<2;c++)z+=W[b][c]*ov(r[c],qc[b],w);return z}


const norm=a=>{let z=sum(a);return z?a.map(x=>x/z):a},conv=(a,b)=>{let z=Array(a.length+b.length-1).fill(0);a.forEach((x,i)=>b.forEach((y,j)=>z[i+j]+=x*y));return z};
function tree(a,lo=0,hi=a.length){if(hi-lo===1)return {lo,hi,A:[1-a[lo],a[lo]]};let mid=(lo+hi)>>1,L=tree(a,lo,mid),R=tree(a,mid,hi);return {lo,hi,L,R,A:conv(L.A,R.A)}}
function split(t,k){return norm(t.L.A.map((x,i)=>x*(t.R.A[k-i]||0)))}
function pairs(a,b){let i=0,j=0,u=a[0],v=b[0],out=[];while(i<a.length&&j<b.length){let z=Math.min(u,v);if(z>0)out.push([i,j,z]);u-=z;v-=z;if(u<=0){i++;u=a[i]||0}if(v<=0){j++;v=b[j]||0}}return out}
function transport(t,root,W,audit){if(!t.L){let m=[[0,0],[0,0]];for(let [k,l,p] of root)m[k][l]+=p;W[t.lo]=m;return}let n=t.L.hi-t.L.lo+1,m=t.R.hi-t.R.lo+1,L=Array.from({length:n},()=>Array(n).fill(0)),R=Array.from({length:m},()=>Array(m).fill(0)),cache=t.A.map((_,i)=>split(t,i));
for(let [k,l,p] of root){for(let [i,j,z] of pairs(cache[k],cache[l])){let r=k-i,s=l-j;if(r<0||r>=m||s<0||s>=m){audit.bad++;audit.badRangeMass+=p*z;continue}L[i][j]+=p*z;R[r][s]+=p*z;audit.work++;if((l-k)*(j-i)<0||(l-k)*(s-r)<0){audit.bad++;audit.badMonotonicMass+=p*z;audit.maxBadMonotonicMass=Math.max(audit.maxBadMonotonicMass,p*z)}}}
let list=M=>M.flatMap((r,i)=>r.map((x,j)=>[i,j,x]).filter(t=>t[2]>0));transport(t.L,list(L),W,audit);transport(t.R,list(R),W,audit)}

function compressed(t,root,audit){
const N=t.hi-t.lo,A=t.A,source=Array(N+1).fill(0),mn=source.slice();for(let [k,l,p] of root){source[k]+=p;mn[Math.min(k,l)]+=p}
let W=Array(N);
function walk(u,h,g){if(!u.L){let a=u.A[1],s=a*h[1],m=a*g[1];W[u.lo]=[[1-s-a+m,a-m],[s-m,m]];return}
let L=u.L.A,R=u.R.A,hl=L.map(()=>0),hr=R.map(()=>0),gl=hl.slice(),gr=hr.slice();
for(let i=0;i<L.length;i++)for(let j=0;j<R.length;j++){let v=h[i+j],w=g[i+j];hl[i]+=v*R[j];hr[j]+=v*L[i];gl[i]+=w*R[j];gr[j]+=w*L[i];audit.reversePairs++}
walk(u.L,hl,gl);walk(u.R,hr,gr)}
walk(t,A.map((x,i)=>x?source[i]/x:0),A.map((x,i)=>x?mn[i]/x:0));return W}
function allocateLinear(p,qc,a,W,bit){let groups=new Map(),t=p.map(()=>0);
for(let j=0;j<p.length;j++){let b=bit[j],q=qc[b][j],breaks=[0,p[j],Math.max(0,Math.min(p[j],a*q)),Math.max(0,Math.min(p[j],p[j]-(1-a)*q))].sort((x,y)=>x-y);
for(let k=0;k<breaks.length-1;k++){let l=breaks[k],h=breaks[k+1];if(h<=l)continue;let m=(l+h)/2,s=(m/a<q?W[b][1]/a:0)-((p[j]-m)/(1-a)<q?W[b][0]/(1-a):0);if(!groups.has(s))groups.set(s,[]);groups.get(s).push({j,l,h})}}
let rem=a;for(let s of [...groups.keys()].sort((x,y)=>y-x)){for(let {j,l,h} of groups.get(s)){let x=Math.min(rem,h-l);t[j]+=x;rem-=x;if(rem<1e-15)break}if(rem<1e-15)break}
return [p.map((x,j)=>(x-t[j])/(1-a)),t.map(x=>x/a)]}
function run(cohorts){
let rows=[],audit={maxWError:0,maxLinearError:0,maxMixError:0,maxRegretViolation:0,maxNegativeW:0,reversePairs:0,bad:0,badRangeMass:0,badMonotonicMass:0,maxBadMonotonicMass:0,work:0};const config=new Map();for(let c of cohorts){let key=c.wi+","+c.seed;if(!config.has(key))config.set(key,[]);config.get(key).push(...c.xs)}
for(let [key,all] of config)for(let N of [4,16,64])for(let start=0;start<all.length;start+=N){let xs=all.slice(start,start+N),a=xs.map(x=>x.a),t=tree(a),T=Math.round(sum(a));
for(let mode of (N===4?["threshold","count","parity"]:["threshold"])){let rec={key,N,start,mode,first:0,blockOptimal:0,blockCheap:0,fresh:0,regret:0,changed:0,events:0,oldWork:0,newWork:0};
for(let J of (mode==="count"?Array.from({length:N+1},(_,i)=>i):[0,1])){let event=k=>mode==="count"?k===J:mode==="parity"?k%2===J:Number(k>=T)===J,source=t.A.map((x,k)=>event(k)?x:0),PJ=sum(source);if(PJ<1e-14)continue;let v=mode==="count"?Number(J>=T):J,root=pairs(norm(source),t.A),oldW=Array(N),before=audit.work,rev=audit.reversePairs;transport(t,root,oldW,audit);let W=compressed(t,root,audit);rec.oldWork+=audit.work-before;rec.newWork+=audit.reversePairs-rev;rec.events++;
for(let i=0;i<N;i++){let x=xs[i],p=x.p[v],q=x.q,qc=[0,1].map(b=>norm(q.map((z,j)=>x.bit[j]===b?z:0))),m=W[i],w=x.next.map(z=>1+ov(z.p[v],z.q)),one=q.map(()=>1),r=allocate(p,qc,a[i],m,w,x.bit),r1=allocate(p,qc,a[i],m,one,x.bit),rl=allocateLinear(p,qc,a[i],m,x.bit),opt=getScore(qc,r,m,w),cheap=getScore(qc,rl,m,w),f=getScore(qc,rl,m,one),diff=opt-cheap;
audit.maxWError=Math.max(audit.maxWError,...m.flat().map((x,j)=>Math.abs(x-oldW[i].flat()[j])));audit.maxNegativeW=Math.max(audit.maxNegativeW,-Math.min(...m.flat()));audit.maxLinearError=Math.max(audit.maxLinearError,Math.abs(f-getScore(qc,r1,m,one)));audit.maxMixError=Math.max(audit.maxMixError,...p.map((z,j)=>Math.abs(z-(1-a[i])*rl[0][j]-a[i]*rl[1][j])));audit.maxRegretViolation=Math.max(audit.maxRegretViolation,-diff,diff-(Math.max(...w)-Math.min(...w))*f);
rec.first+=PJ*f/N;rec.blockOptimal+=PJ*opt/N;rec.blockCheap+=PJ*cheap/N;rec.fresh+=PJ*ov(p,q,w)/N;rec.regret+=PJ*diff/N;rec.changed+=PJ*(m[0][1]+m[1][0])}
}rows.push(rec)}}
let summary=[];for(let N of [4,16,64])for(let mode of ["threshold","count","parity"]){let r=rows.filter(x=>x.N===N&&x.mode===mode);if(!r.length)continue;summary.push({N,mode,groups:r.length,...Object.fromEntries(["first","blockOptimal","blockCheap","fresh","regret","changed"].map(k=>[k,sum(r.map(x=>x[k]))/r.length])),oldWork:sum(r.map(x=>x.oldWork)),newWork:sum(r.map(x=>x.newWork)),events:sum(r.map(x=>x.events))})}
let scale=[];for(let N of [16,64,128,256,512,1024]){let a=Array.from({length:N},(_,i)=>.05+.9*((Math.imul(i+13,2654435761)>>>0)/4294967296)),t=tree(a),T=Math.round(sum(a)),source=t.A.map((x,k)=>k>=T?x:0),root=pairs(norm(source),t.A),rev=audit.reversePairs,W=compressed(t,root,audit),err=Math.max(...W.map((m,i)=>Math.abs(m[0][1]+m[1][1]-a[i]))),cost=sum(W.map(m=>m[0][1]+m[1][0])),lower=sum(root.map(([k,l,p])=>Math.abs(k-l)*p));scale.push({N,reversePairs:audit.reversePairs-rev,marginalError:err,costError:Math.abs(cost-lower),negative:Math.max(0,-Math.min(...W.flat(2))),zeroRootMass:t.A.filter(x=>x===0).length})}
return {summary,rows,audit,scale}}
console.log(JSON.stringify(run(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));