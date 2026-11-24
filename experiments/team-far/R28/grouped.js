// node grouped.js weights.json checkpoint.json input.txt
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


function grouped(cohorts){
const sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),ov=(p,q,w)=>sum(p.map((v,i)=>Math.min(v,q[i])*(w?w[i]:1))),norm=a=>{let z=sum(a);return z?a.map(x=>x/z):a};
let rows=[];for(let {wi,seed,c,xs} of cohorts){let N=xs.length,M=1<<N,full=M-1,pat=Array.from({length:M},(_,k)=>xs.map((_,i)=>(k>>i)&1)),count=pat.map(sum),prior=pat.map(b=>b.reduce((w,x,i)=>w*(x?xs[i].a:1-xs[i].a),1));
for(let mode of ["threshold","count"]){let rec={wi,seed,c,mode,first:Array(N).fill(0),block:Array(N).fill(0),rounds:Array(N).fill(0),freshFirst:0,freshBlock:0,events:0};
for(let J of mode==="threshold"?[0,1]:Array.from({length:N+1},(_,i)=>i)){let match=k=>mode==="threshold"?Number(count[k]>=N/2)===J:count[k]===J,prob=sum(prior.filter((_,k)=>match(k)));if(prob<1e-14)continue;let v=mode==="threshold"?J:Number(J>=N/2),Q=prior.map((w,k)=>match(k)?w/prob:0),qc=xs.map(x=>[0,1].map(b=>norm(x.q.map((w,j)=>x.bit[j]===b?w:0)))),weights=xs.map(x=>x.next.map(z=>1+ov(z.p[v],z.q)));
for(let kind of ["first","block"]){let w=kind==="block"?weights:xs.map(x=>x.q.map(()=>1)),memo=new Map(),states=new Map();
function state(mask,values){let key=mask+","+values;if(states.has(key))return states.get(key);let valid=Q.map((z,k)=>(k&mask)===values?z:0),total=sum(valid);if(total<1e-15)return null;valid=valid.map(z=>z/total);let left=full^mask,ids=xs.map((_,i)=>i).filter(i=>left&(1<<i)),a=xs.map((_,i)=>dot(valid,pat.map(b=>b[i]))),base=xs.map((x,i)=>(1-a[i])*ov(qc[i][0],x.p[v],w[i])+a[i]*ov(qc[i][1],x.p[v],w[i])),gain=xs.map((x,i)=>ov(qc[i][0].map((z,j)=>(1-a[i])*z+a[i]*qc[i][1][j]),x.p[v],w[i])-base[i]);
let independent=[];for(let I=left;;I=(I-1)&left){let dst=Array(M).fill(0);for(let k=0;k<M;k++)dst[k&I]+=valid[k];let err=0;for(let k=I;;k=(k-1)&I){let z=ids.filter(i=>I&(1<<i)).reduce((s,i)=>s*(pat[k][i]?a[i]:1-a[i]),1);err=Math.max(err,Math.abs(dst[k]-z));if(!k)break}if(err<1e-11)independent.push(I);if(!I)break}
let actions=[];for(let S=left;S;S=(S-1)&left){let b=sum(base.filter((_,i)=>S&(1<<i))),best=0;for(let I of independent)if((I&S)===I)best=Math.max(best,sum(gain.filter((_,i)=>I&(1<<i))));let dst=Array(M).fill(0);for(let k=0;k<M;k++)dst[k&S]+=valid[k];actions.push({S,r:b+best,dst})}
let st={left,actions};states.set(key,st);return st}
function dp(mask,values,k){if(mask===full)return {r:0,d:0};if(!k)return {r:-Infinity,d:Infinity};let key=[mask,values,k].join(",");if(memo.has(key))return memo.get(key);let st=state(mask,values);if(!st)return {r:0,d:0};let best={r:-Infinity,d:Infinity};for(let {S,r,dst} of st.actions){if(k===1&&S!==st.left)continue;let z={r,d:1};for(let b=0;b<M;b++)if(dst[b]>1e-15){let v=dp(mask|S,values|b,k-1);z.r+=dst[b]*v.r;z.d+=dst[b]*v.d}if(z.r>best.r+1e-12||(Math.abs(z.r-best.r)<1e-12&&z.d<best.d))best=z}memo.set(key,best);return best}
for(let k=1;k<=N;k++){let z=dp(0,0,k);rec[kind][k-1]+=prob*z.r/N;if(kind==="block")rec.rounds[k-1]+=prob*z.d}
rec[kind==="block"?"freshBlock":"freshFirst"]+=prob*sum(xs.map((x,i)=>ov(x.q,x.p[v],w[i])))/N;
}rec.events++}rows.push(rec)}}
const summary=["threshold","count"].map(mode=>{let a=rows.filter(r=>r.mode===mode);return {mode,n:a.length,first:a[0].first.map((_,i)=>sum(a.map(r=>r.first[i]))/a.length),block:a[0].block.map((_,i)=>sum(a.map(r=>r.block[i]))/a.length),rounds:a[0].rounds.map((_,i)=>sum(a.map(r=>r.rounds[i]))/a.length),freshFirst:sum(a.map(r=>r.freshFirst))/a.length,freshBlock:sum(a.map(r=>r.freshBlock))/a.length}});return {rows,summary};
}
console.log(JSON.stringify(grouped(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
