// node two_token.js weights.json checkpoint.json input.txt
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








function study2(cohorts){
const sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),zero=n=>Array(n).fill(0),norm=a=>{let z=sum(a);return z?a.map(v=>v/z):a},V=27;
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(p,r){let s=0,u=r();for(let i=0;i<p.length;i++){s+=p[i];if(u<s)return i}return p.length-1}
function draw(p,q,r){let x=cat(q,r),a=r()<=Math.min(1,p[x]/q[x]);return [x,a?x:cat(norm(p.map((v,i)=>Math.max(0,v-q[i]))),r)]}
function conv(a,b){let out=zero(a.length+b.length-1);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)out[i+j]+=a[i]*b[j];return out}
function d(N,k){return Math.sqrt((N-1-k)/(k+1))+Math.sqrt(k/(N-k))}
function coeff(z){let N=z.length,out=zero(N);for(let i=0;i<N;i++){let f=[1];for(let j=0;j<N;j++)if(j!==i)f=conv(f,[1-z[j],z[j]]);out[i]=sum(f.map((x,k)=>x*d(N,k)))}return out}
let rows=[],audit={draws:0,draftOnly:0,formalTokens:0,firstMismatch:0};
const config=new Map();for(let c of cohorts){let key=c.wi+","+c.seed;if(!config.has(key))config.set(key,[]);config.get(key).push(...c.xs)}
const ms=[0,1,4,16],methods=["sampled","oracleMean",...ms.slice(1).map(m=>"M"+m),"oracleCoeff"];
for(let [key,all] of config)for(let N of [8,16])for(let start=0;start<all.length;start+=N){let xs=all.slice(start,start+N),ps=xs.map(z=>z.p[0]),qs=xs.map(z=>z.q),ys=xs.map(z=>z.y),pi=xs.map((z,i)=>ps[i][z.y]*z.next[z.y].p[0][z.y2]),mu=xs.map((z,i)=>qs[i][z.y]*z.next[z.y].q[z.y2]),cov=xs.map((z,i)=>Math.min(ps[i][z.y],qs[i][z.y])*Math.min(z.next[z.y].p[0][z.y2],z.next[z.y].q[z.y2])-pi[i]*mu[i]),base=pi.map(a=>a*(1-a)),b=mu.map(a=>a*(1-a)),co=coeff(pi),truth=zero(V);
for(let i=0;i<N;i++)for(let j=0;j<V;j++)truth[j]+=co[i]*pi[i]*(Number(j===xs[i].y)+Number(j===xs[i].y2)-ps[i][j]-xs[i].next[xs[i].y].p[0][j])/N;
let rr=rng(833+Number(key.split(",")[0])*1000+Number(key.split(",")[1])*100+N*10+start),stats=Object.fromEntries(methods.map(m=>[m,{sum:zero(V),sq:0,err:0}])),R=512;
function draftReward(z){let x=cat(z.q,rr),y=cat(z.next[x].q,rr);audit.draftOnly++;return Number(x===z.y&&y===z.y2)}
for(let rep=0;rep<R;rep++){let r=[],s=[],score=[],extra=[];
for(let i=0;i<N;i++){let z=xs[i],[x,y]=draw(ps[i],qs[i],rr),a,c;if(x===y)[a,c]=draw(z.next[x].p[0],z.next[x].q,rr);else{a=cat(z.next[x].q,rr);c=cat(z.next[y].p[0],rr);audit.firstMismatch++}r.push(Number(y===z.y&&c===z.y2));s.push(Number(x===z.y&&a===z.y2));score.push(ps[i].map((v,j)=>Number(j===y)+Number(j===c)-v-z.next[y].p[0][j]));let v=0,e={};for(let j=1;j<=16;j++){v+=draftReward(z);if(ms.includes(j))e[j]=v/j}extra.push(e);audit.formalTokens+=2}
let allC={sampled:coeff(r),oracleCoeff:co,oracleMean:coeff(r.map((v,i)=>v-(cov[i]/b[i])*(s[i]-mu[i])))};
for(let M of ms.slice(1))allC["M"+M]=coeff(r.map((v,i)=>v-(cov[i]/(b[i]*(1+1/M)))*(s[i]-extra[i][M])));
for(let method of methods){let g=zero(V);for(let i=0;i<N;i++){let w=(r[i]-mu[i])*allC[method][i];for(let j=0;j<V;j++)g[j]+=w*score[i][j]/N}let st=stats[method];for(let j=0;j<V;j++)st.sum[j]+=g[j]/R;st.sq+=dot(g,g)/R;st.err+=sum(g.map((z,j)=>(z-truth[j])**2))/R}audit.draws++}
let vals=Object.fromEntries(methods.map(m=>{let st=stats[m];return [m,{variance:st.sq-dot(st.sum,st.sum),mse:st.err,meanError:Math.sqrt(sum(st.sum.map((z,i)=>(z-truth[i])**2)))}]}));let rewardRatios=Object.fromEntries(ms.map(M=>[M,sum(base.map((v,i)=>1-cov[i]**2/(v*b[i]*(1+(M?1/M:0)))))/N]));
rows.push({key,N,start,pi:sum(pi)/N,mu:sum(mu)/N,vals,rewardRatios})}
let summary=[8,16].map(N=>{let r=rows.filter(x=>x.N===N),vals=Object.fromEntries(methods.map(m=>[m,Object.fromEntries(["variance","mse","meanError"].map(k=>[k,sum(r.map(x=>x.vals[m][k]))/r.length]))]));return {N,groups:r.length,pi:sum(r.map(x=>x.pi))/r.length,mu:sum(r.map(x=>x.mu))/r.length,vals,rewardRatios:Object.fromEntries(ms.map(M=>[M,sum(r.map(x=>x.rewardRatios[M]))/r.length])),breakEven:Object.fromEntries(ms.slice(1).map(M=>[M,(vals.sampled.variance/vals["M"+M].variance-1)/M]))}});return {summary,audit,rows}}
console.log(JSON.stringify(study2(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
