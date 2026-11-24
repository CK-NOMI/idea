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
let make=(name)=>{let j=3,pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],{h,s}=m.feature(pre);return {pre,h,s,y:name.charCodeAt(j)-97,y2:name.charCodeAt(j+1)-97,ref:name.slice(j,j+4).split('').map(c=>c.charCodeAt(0)-97)}};
let train=groups.map(g=>g.slice(0,16).map(make)),g=scale(train.flat().map(x=>sub(mom(sm(mat(W,x.h))),W[x.y])).reduce(add),1/(16*N)),shift=mat(W,scale(g,-1.5));
let test=groups.map(g=>g.slice(16).map(make)),rr=rng(seed+1000*wi);
for(let c=0;c<16;c++){let xs=test.map(a=>a[c]).map(x=>{let p=sm(mat(W,x.h)),p1=sm(add(mat(W,x.h),shift)),q=sm(mat(W,x.s)),bit=p.map((v,i)=>Number(p1[i]>v)),a=dot(q,bit),next=Array.from({length:V},(_,i)=>{let {h,s}=m.feature(x.pre.concat(i)),l=mat(W,h);return {p:[sm(l),sm(add(l,shift))],q:sm(mat(W,s))}});let path=x.ref.map((v,j)=>{let {h,s}=m.feature(x.pre.concat(x.ref.slice(0,j)));return {p:sm(mat(W,h)),q:sm(mat(W,s)),v}});return {p:[p,p1],q,bit,a,next,y:x.y,y2:x.y2,path}}),suff=Array(N+1);suff[N]=[1];for(let i=N-1;i>=0;i--)suff[i]=conv([1-xs[i].a,xs[i].a],suff[i+1]);
let votePatterns=Array.from({length:1<<N},(_,mask)=>{let bits=xs.map((_,i)=>Number(!!(mask&(1<<i)))),w=bits.reduce((v,b,i)=>v*(b?xs[i].a:1-xs[i].a),1);return {bits,w,s:sum(bits)}});
audit.cohorts++;
rows.push({wi,seed,c,xs});
}}} return rows;}










function study(cohorts){
const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
function rng(seed){let z=seed>>>0;return ()=>((z=(Math.imul(z,1664525)+1013904223)>>>0)+.5)/4294967296}
function calc(path,N){let L=path.length,s=[1];for(let x of path)s.push(s.at(-1)*x.p[x.v]);let p=s[L],Z=1-p**N-(1-p)**N,oneOther=1-(1-p)**(N-1),suffix=N*sum(s.slice(0,-1).map((v,t)=>(v-s[t+1])*(L-t-1)))*oneOther,naive=N*L,stop=N*sum(s.slice(0,-1))+suffix,shared=sum(s.slice(0,-1).map(v=>1-(1-v)**N))+suffix,full=L+suffix;return {L,N,p,Z,naive,stop,shared,full,suffix,saveStop:1-stop/naive,saveShared:1-shared/naive,saveAgainstShared:1-full/shared,ref:shared-suffix,s}}
let rows=[],audit={groups:0,maxLawErr:0,invalid:0,maxCostErr:0},samples=[];
for(let {wi,seed,c,xs} of cohorts)for(let [ix,x] of xs.entries())for(let L of [2,4])for(let N of [4,8,16]){let a=calc(x.path.slice(0,L),N);rows.push({wi,seed,c,ix,...a});let s=a.s,diff=s.slice(0,-1).map((v,t)=>v-s[t+1]);audit.maxLawErr=Math.max(audit.maxLawErr,Math.abs(sum(diff)+s[L]-1));if(ix===0&&c%2===0){let rr=rng(7+wi*10000+seed*173+c*23+N*17+L),R=2048,cost=[0,0,0,0],ad=0,labels=Array(N+1).fill(0);for(let i=0;i<R;i++){let u=Array.from({length:N},()=>rr()),t=u.map(v=>{for(let j=1;j<=L;j++)if(v>=s[j])return j;return L+1}),k=t.filter(v=>v===L+1).length,mixed=k>0&&k<N;labels[k]++;ad+=Number(mixed);let tail=mixed?sum(t.map(v=>Math.max(0,L-v))):0;cost[0]+=N*L;cost[1]+=sum(t.map(v=>Math.min(v,L)))+tail;cost[2]+=Math.min(L,Math.max(...t))+tail;cost[3]+=L+tail;if(mixed&&(!k||k===N))audit.invalid++}audit.groups+=R;samples.push({wi,seed,c,L,N,R,ad:ad/R,expectedAd:a.Z,mc:cost.map(v=>v/R),exact:[a.naive,a.stop,a.shared,a.full]})}}
let summary=[];for(let L of [2,4])for(let N of [4,8,16]){let r=rows.filter(x=>x.L===L&&x.N===N),m=samples.filter(x=>x.L===L&&x.N===N);summary.push({L,N,n:r.length,...Object.fromEntries(["p","Z","naive","stop","shared","full","suffix","saveStop","saveShared","saveAgainstShared","ref"].map(k=>[k,mean(r.map(x=>x[k]))])),mc:{n:m.length,ad:mean(m.map(x=>x.ad)),expectedAd:mean(m.map(x=>x.expectedAd)),cost:costMean(m,"mc"),expected:costMean(m,"exact")}})}function costMean(a,k){return a[0][k].map((_,i)=>mean(a.map(x=>x[k][i])))}
return {summary,audit,rows,samples}}
console.log(JSON.stringify(study(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));