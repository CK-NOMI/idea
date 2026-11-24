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
const sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),zero=n=>Array(n).fill(0),norm=a=>{let z=sum(a);return z?a.map(v=>v/z):a},V=27;
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(p,r){let s=0,u=r();for(let i=0;i<p.length;i++){s+=p[i];if(u<s)return i}return p.length-1}
function draw(p,q,r){let x=cat(q,r),a=r()<=Math.min(1,p[x]/q[x]);return [x,a?x:cat(norm(p.map((v,i)=>Math.max(0,v-q[i]))),r)]}
function conv(a,b){let out=zero(a.length+b.length-1);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)out[i+j]+=a[i]*b[j];return out}
function d(N,k){return Math.sqrt((N-1-k)/(k+1))+Math.sqrt(k/(N-k))}
function coeff(z){let N=z.length,out=zero(N);for(let i=0;i<N;i++){let f=[1];for(let j=0;j<N;j++)if(j!==i)f=conv(f,[1-z[j],z[j]]);out[i]=sum(f.map((x,k)=>x*d(N,k)))}return out}
function adv(N,k,r){if(k===0||k===N)return 0;return (r-k/N)/Math.sqrt(k/N*(1-k/N))}
function probability(p,y){return p[y]}
let rows=[],audit={maxPotentialGradError:0,maxDPEnumeration:0,maxJointRewardError:0,unitCVWorse:0,oracleCVWorse:0,contexts:0,draws:0};
const config=new Map();for(let c of cohorts){let key=c.wi+","+c.seed;if(!config.has(key))config.set(key,[]);config.get(key).push(...c.xs)}
const methods=["group","sampledCoeff","unitCV","optimalCV","oracleCoeff","rawReward"];
for(let [key,all] of config)for(let N of [4,8,16])for(let start=0;start<all.length;start+=N){let xs=all.slice(start,start+N),ps=xs.map(z=>z.p[0]),qs=xs.map(z=>z.q),ys=xs.map(z=>z.y),pi=ps.map((p,i)=>p[ys[i]]),mu=qs.map((q,i)=>q[ys[i]]),beta=[],unitRatio=[],optimalRatio=[];
for(let i=0;i<N;i++){let p=ps[i],q=qs[i],a=pi[i],b=mu[i],joint=Math.min(a,b),cov=joint-a*b,vr=a*(1-a),vq=b*(1-b);let t=cov/vq;beta.push(t);let u=vr+vq-2*cov,v=vr-cov*cov/vq;unitRatio.push(u/vr);optimalRatio.push(v/vr);audit.unitCVWorse+=Number(u>vr+1e-12);audit.oracleCVWorse+=Number(v>vr+1e-12);audit.contexts++}
let co=coeff(pi),truth=zero(V);for(let i=0;i<N;i++)for(let j=0;j<V;j++)truth[j]+=co[i]*pi[i]*(Number(j===ys[i])-ps[i][j])/N;
let Phi=[0];for(let k=0;k<N;k++)Phi.push(Phi[k]+d(N,k)/N);
let potential=z=>{let f=[1];for(let v of z)f=conv(f,[1-v,v]);return dot(f,Phi)};
for(let i=0;i<N;i++){let z=pi.slice(),e=1e-5;z[i]+=e;let v=potential(z);z[i]-=2*e;let g=(v-potential(z))/(2*e);audit.maxPotentialGradError=Math.max(audit.maxPotentialGradError,Math.abs(g-co[i]/N))}
if(N<=8){let en=zero(N);for(let m=0;m<(1<<N);m++){let b=pi.map((_,i)=>(m>>i)&1),k=sum(b),pr=b.reduce((a,x,i)=>a*(x?pi[i]:1-pi[i]),1);for(let i=0;i<N;i++)en[i]+=pr*adv(N,k,b[i])*(b[i]/pi[i]-(1-b[i])/(1-pi[i]))}audit.maxDPEnumeration=Math.max(audit.maxDPEnumeration,...en.map((z,i)=>Math.abs(z-co[i])))}
let rr=rng(1234+Number(key.split(",")[0])*1000+Number(key.split(",")[1])*100+N*10+start),stats=Object.fromEntries(methods.map(m=>[m,{sum:zero(V),sq:0,err:0}])),R=1024;
for(let rep=0;rep<R;rep++){let pairs=xs.map((_,i)=>draw(ps[i],qs[i],rr)),r=pairs.map((a,i)=>Number(a[1]===ys[i])),s=pairs.map((a,i)=>Number(a[0]===ys[i])),k=sum(r),z1=r.map((a,i)=>a-s[i]+mu[i]),z2=r.map((a,i)=>a-beta[i]*(s[i]-mu[i])),cf=coeff(r),c1=coeff(z1),c2=coeff(z2);
for(let method of methods){let g=zero(V);for(let i=0;i<N;i++){let w=method==="group"?adv(N,k,r[i]):method==="rawReward"?r[i]-mu[i]:(r[i]-mu[i])*(method==="sampledCoeff"?cf[i]:method==="unitCV"?c1[i]:method==="optimalCV"?c2[i]:co[i]);for(let j=0;j<V;j++)g[j]+=w*(Number(j===pairs[i][1])-ps[i][j])/N}
let st=stats[method];for(let j=0;j<V;j++)st.sum[j]+=g[j]/R;st.sq+=dot(g,g)/R;st.err+=sum(g.map((z,j)=>(z-truth[j])**2))/R}audit.draws++}
let vals=Object.fromEntries(methods.map(m=>{let st=stats[m];return [m,{variance:st.sq-dot(st.sum,st.sum),mse:st.err,meanError:Math.sqrt(sum(st.sum.map((z,i)=>(z-truth[i])**2)))}]}));
rows.push({key,N,start,meanPi:sum(pi)/N,meanMu:sum(mu)/N,unitRatio:sum(unitRatio)/N,optimalRatio:sum(optimalRatio)/N,truthNorm:Math.sqrt(dot(truth,truth)),vals})}
let summary=[4,8,16].map(N=>{let r=rows.filter(x=>x.N===N);let avg=k=>sum(r.map(x=>x[k]))/r.length,vals=Object.fromEntries(methods.map(m=>[m,Object.fromEntries(["variance","mse","meanError"].map(k=>[k,sum(r.map(x=>x.vals[m][k]))/r.length]))]));return {N,groups:r.length,meanPi:avg("meanPi"),meanMu:avg("meanMu"),unitRatio:avg("unitRatio"),optimalRatio:avg("optimalRatio"),truthNorm:avg("truthNorm"),vals}});return {summary,audit,rows}}
console.log(JSON.stringify(study(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
