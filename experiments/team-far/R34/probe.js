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
let make=(name)=>{let j=3,pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],{h,s}=m.feature(pre);return {pre,h,s,y:name.charCodeAt(j)-97,y2:name.charCodeAt(j+1)-97}};
let train=groups.map(g=>g.slice(0,16).map(make)),g=scale(train.flat().map(x=>sub(mom(sm(mat(W,x.h))),W[x.y])).reduce(add),1/(16*N)),shift=mat(W,scale(g,-1.5));
let test=groups.map(g=>g.slice(16).map(make)),rr=rng(seed+1000*wi);
for(let c=0;c<16;c++){let xs=test.map(a=>a[c]).map(x=>{let p=sm(mat(W,x.h)),p1=sm(add(mat(W,x.h),shift)),q=sm(mat(W,x.s)),bit=p.map((v,i)=>Number(p1[i]>v)),a=dot(q,bit),next=Array.from({length:V},(_,i)=>{let {h,s}=m.feature(x.pre.concat(i)),l=mat(W,h);return {p:[sm(l),sm(add(l,shift))],q:sm(mat(W,s))}});return {p:[p,p1],q,bit,a,next,y:x.y,y2:x.y2}}),suff=Array(N+1);suff[N]=[1];for(let i=N-1;i>=0;i--)suff[i]=conv([1-xs[i].a,xs[i].a],suff[i+1]);
let votePatterns=Array.from({length:1<<N},(_,mask)=>{let bits=xs.map((_,i)=>Number(!!(mask&(1<<i)))),w=bits.reduce((v,b,i)=>v*(b?xs[i].a:1-xs[i].a),1);return {bits,w,s:sum(bits)}});
audit.cohorts++;
rows.push({wi,seed,c,xs});
}}} return rows;}









function study(cohorts){
const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length,conv=(a,b)=>{let r=Array(a.length+b.length-1).fill(0);a.forEach((x,i)=>b.forEach((y,j)=>r[i+j]+=x*y));return r},bin=(n,p)=>{let a=[1];for(let i=0;i<n;i++)a=conv(a,[1-p,p]);return a};
function norm(a){let s=sum(a);return a.map(x=>x/s)}
function pairs(a,b){let i=0,j=0,u=a[0],v=b[0],out=[];while(i<a.length&&j<b.length){let z=Math.min(u,v);if(z>0)out.push([i,j,z]);u-=z;v-=z;if(u<=0){i++;u=a[i]||0}if(v<=0){j++;v=b[j]||0}}return out}
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(a,r){let s=0,u=r();for(let i=0;i<a.length;i++){s+=a[i];if(u<s)return i}return a.length-1}
function metrics(N,p,q,O0,O1){
let A=bin(N,p),Q=bin(N,q),Z=sum(A.slice(1,-1)),B=norm(A.map((v,k)=>k>0&&k<N?v:0)),G=pairs(A,B),H=pairs(Q,B),L=sum(B.map((v,k)=>k*v)),lo=Math.max(L/p,(N-L)/(1-p)),base=N/Z,up=sum(G.map(([k,l,w])=>Math.max(0,l-k)*w)),down=sum(G.map(([k,l,w])=>Math.max(0,k-l)*w));
let mixRepair=N+up/p+down/(1-p),naiveRepair=N+A[0]*L/p+A[N]*(N-L)/(1-p);
let qUp=sum(H.map(([k,l,w])=>Math.max(0,l-k)*w)),qDown=sum(H.map(([k,l,w])=>Math.max(0,k-l)*w)),keep1=sum(H.map(([k,l,w])=>Math.min(k,l)*w)),keep0=N-qUp-qDown-keep1;
let agree=(keep0*O0+keep1*O1)/N;
let freshUpper=O0*(N-L)/N+O1*L/N; // oracle drafts already drawn from the target reward classes with no cost for conditioning
return {p,q,Z,base,lo,mixRepair,naiveRepair,up,down,retention:(N-up-down)/N,oracleQRetention:agree,oracleFreshClassRetention:freshUpper,qChanges:qUp+qDown,oracleQClassChangesCost:qUp/p+qDown/(1-p),upperSaving:1-lo/base,repairSaving:1-mixRepair/base,naiveSaving:1-naiveRepair/base,distributionError:Math.max(...B.map((v,l)=>Math.abs(v-sum(G.filter(x=>x[1]===l).map(x=>x[2]))))),sourceError:Math.max(...A.map((v,k)=>Math.abs(v-sum(G.filter(x=>x[0]===k).map(x=>x[2])))))}
}
let rows=[],audit={maxDistributionError:0,maxSourceError:0,maxLowerViolation:0,maxUniversalViolation:0,groups:0,baseRollouts:0,repairRollouts:0,maxCountTV:0,meanCountTV:0,invalid:0},mcRows=[];
for(let {wi,seed,c,xs} of cohorts)for(let [ix,x] of xs.entries())for(let horizon of [1,2]){
let pp=[],qq=[],label=horizon===1?x.y:27*x.y+x.y2;if(horizon===1){pp=x.p[0];qq=x.q}else for(let i=0;i<27;i++)for(let j=0;j<27;j++){pp.push(x.p[0][i]*x.next[i].p[0][j]);qq.push(x.q[i]*x.next[i].q[j])}
let p=pp[label],q=qq[label],O0=sum(pp.map((v,i)=>i===label?0:Math.min(v/(1-p),qq[i]/(1-q))));
for(let N of [4,8,16,32]){let a=metrics(N,p,q,O0,1);rows.push({wi,seed,c,ix,horizon,N,...a});audit.maxDistributionError=Math.max(audit.maxDistributionError,a.distributionError);audit.maxSourceError=Math.max(audit.maxSourceError,a.sourceError);audit.maxLowerViolation=Math.max(audit.maxLowerViolation,a.lo-a.mixRepair);audit.maxUniversalViolation=Math.max(audit.maxUniversalViolation,a.upperSaving-2**(1-N));
if(N<=16&&ix===0&&c%2===0){let rr=rng(321+wi*1000+seed*17+c*101+N*11+horizon),A=bin(N,p),B=norm(A.map((v,k)=>k>0&&k<N?v:0)),G=pairs(A,B),cntR=Array(N+1).fill(0),cntB=Array(N+1).fill(0),totB=0,totR=0,flips=0,R=128;
function draw(){let i=cat(pp,rr);return Number(i===label)}
function drawGroup(){return Array.from({length:N},()=>draw())}
for(let rep=0;rep<R;rep++){let b,k;do{b=drawGroup();totB+=N;k=sum(b)}while(k===0||k===N);cntB[k]++;
let z=drawGroup(),s=sum(z);totR+=N;let L=cat(Array.from({length:N+1},(_,j)=>sum(G.filter(a=>a[0]===s&&a[1]===j).map(a=>a[2]))/A[s]),rr);let change=L>s?0:1,eligible=z.map((v,i)=>v===change?i:-1).filter(i=>i!==-1);for(let t=0;t<Math.abs(s-L);t++){let idx=Math.floor(rr()*eligible.length),i=eligible.splice(idx,1)[0],v;do{v=draw();totR++}while(v===change);z[i]=v;flips++}let K=sum(z);cntR[K]++;if(K!==L||K===0||K===N)audit.invalid++}
let tv=cnt=>sum(cnt.map((v,i)=>Math.abs(v/R-B[i])))/2;let row={wi,seed,c,N,horizon,R,base:totB/R,repair:totR/R,expectedBase:a.base,expectedRepair:a.mixRepair,tvBase:tv(cntB),tvRepair:tv(cntR),flips:flips/R};mcRows.push(row);audit.groups+=R;audit.baseRollouts+=totB;audit.repairRollouts+=totR;audit.maxCountTV=Math.max(audit.maxCountTV,row.tvRepair);audit.meanCountTV+=row.tvRepair}
}}
audit.meanCountTV/=mcRows.length;
let summary=[];for(let h of [1,2])for(let N of [4,8,16,32]){let r=rows.filter(x=>x.horizon===h&&x.N===N),m=mcRows.filter(x=>x.horizon===h&&x.N===N);summary.push({h,N,n:r.length,...Object.fromEntries(["p","q","Z","base","lo","mixRepair","naiveRepair","retention","oracleQRetention","oracleFreshClassRetention","qChanges","oracleQClassChangesCost","upperSaving","repairSaving","naiveSaving"].map(k=>[k,mean(r.map(x=>x[k]))])),mc:m.length?{n:m.length,base:mean(m.map(x=>x.base)),repair:mean(m.map(x=>x.repair)),expectedBase:mean(m.map(x=>x.expectedBase)),expectedRepair:mean(m.map(x=>x.expectedRepair)),tvRepair:mean(m.map(x=>x.tvRepair))}:null})}
let stress=[];for(let p of [.001,.01,.05,.1,.25,.5,.75,.95,.99,.999])for(let N of [4,8,16,32]){let x=metrics(N,p,p,1,1);stress.push({p,N,base:x.base,lower:x.lo,repair:x.mixRepair,upperSaving:x.upperSaving,repairSaving:x.repairSaving})}
return {summary,audit,rows,mcRows,stress}}
console.log(JSON.stringify(study(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
