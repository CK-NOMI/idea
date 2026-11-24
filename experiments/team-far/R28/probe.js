// R28: run node probe.js weights.json checkpoint.json input.txt
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
const sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),ov=(p,q,w)=>sum(p.map((v,i)=>Math.min(v,q[i])*(w?w[i]:1))),norm=a=>{let z=sum(a);return z?a.map(x=>x/z):a},perms=a=>a.length?a.flatMap((v,i)=>perms(a.filter((_,j)=>j!==i)).map(p=>[v,...p])):[[]];
function flow(supply,demand,reward){let n=supply.length,m=demand.length,K=n+m+2,src=K-2,dst=K-1,g=Array.from({length:K},()=>[]);function edge(i,j,cap,c){let x={to:j,rev:g[j].length,cap,c},y={to:i,rev:g[i].length,cap:0,c:-c};g[i].push(x);g[j].push(y);return x}
for(let i=0;i<n;i++)edge(src,i,supply[i],0);for(let j=0;j<m;j++)edge(n+j,dst,demand[j],0);
let edges=Array.from({length:n},(_,i)=>Array.from({length:m},(_,j)=>edge(i,n+j,1,8-reward[i][j])));
let total=0,it=0;while(total<1-1e-12){if(++it>10000)throw Error("flow limit");let d=Array(K).fill(Infinity),pr=Array(K);d[src]=0;for(let k=0;k<K-1;k++){let change=false;for(let i=0;i<K;i++)if(Number.isFinite(d[i]))for(let j=0;j<g[i].length;j++){let e=g[i][j];if(e.cap>1e-14&&d[e.to]>d[i]+e.c+1e-12){d[e.to]=d[i]+e.c;pr[e.to]=[i,j];change=true}}if(!change)break}if(!pr[dst])throw Error("no path "+total);let f=1-total;for(let j=dst;j!==src;){let [i,k]=pr[j];f=Math.min(f,g[i][k].cap);j=i}for(let j=dst;j!==src;){let [i,k]=pr[j],e=g[i][k];e.cap-=f;g[j][e.rev].cap+=f;j=i}total+=f;}
let mat=edges.map((row,i)=>row.map(e=>g[e.to][e.rev].cap));let err=Math.max(...mat.map((r,i)=>Math.abs(sum(r)-supply[i])),...demand.map((x,j)=>Math.abs(sum(mat.map(r=>r[j]))-x)));let val=sum(mat.map((r,i)=>dot(r,reward[i])));return {mat,err,val,it}}
let rows=[],maxerr=0;for(let {wi,seed,c,xs} of cohorts){let N=xs.length,V=xs[0].q.length,order=perms(xs.map((_,i)=>i)),pat=Array.from({length:1<<N},(_,k)=>xs.map((_,i)=>(k>>i)&1)),count=pat.map(sum),prior=pat.map(b=>b.reduce((w,x,i)=>w*(x?xs[i].a:1-xs[i].a),1));
for(let mode of ["threshold","count"]){let rec={wi,seed,c,mode,first:{seq:0,best:0,ot:0,hybrid:0,fresh:0},block:{seq:0,best:0,ot:0,hybrid:0,fresh:0},maxerr:0,events:0,otWinsFirst:0,otWinsBlock:0,iterations:0};
for(let J of mode==="threshold"?[0,1]:Array.from({length:N+1},(_,i)=>i)){let match=k=>mode==="threshold"?Number(count[k]>=N/2)===J:count[k]===J,prob=sum(prior.filter((_,k)=>match(k)));if(prob<1e-14)continue;let v=mode==="threshold"?J:Number(J>=N/2),Q=prior.map((w,k)=>match(k)?w/prob:0),P=pat.map(b=>b.reduce((w,x,i)=>{let a=dot(xs[i].p[v],xs[i].bit);return w*(x?a:1-a)},1));
let weights=xs.map(x=>x.next.map(z=>1+ov(z.p[v],z.q))),qc=xs.map(x=>[0,1].map(b=>norm(x.q.map((w,j)=>x.bit[j]===b?w:0)))),pc=xs.map(x=>[0,1].map(b=>norm(x.p[v].map((w,j)=>x.bit[j]===b?w:0))));
for(let kind of ["first","block"]){let w=kind==="block"?weights:xs.map(x=>x.q.map(()=>1));let cost=xs.map((x,i)=>[0,1].map(b=>ov(qc[i][b],pc[i][b],w[i])));let reward=pat.map(b=>pat.map(t=>sum(b.map((z,i)=>z===t[i]?cost[i][z]:0))));
let f=flow(Q,P,reward);maxerr=Math.max(maxerr,f.err);rec.maxerr=Math.max(rec.maxerr,f.err);rec.iterations+=f.it;
function seq(o){let val=0;for(let k=0;k<pat.length;k++){if(!Q[k])continue;let prefix=[];for(let i of o){let mass=0,one=0;for(let j=0;j<pat.length;j++){if(prefix.every(h=>pat[j][h]===pat[k][h])){mass+=Q[j];one+=Q[j]*pat[j][i]}}let a=one/mass,qq=qc[i][0].map((z,j)=>(1-a)*z+a*qc[i][1][j]);val+=Q[k]*ov(qq,xs[i].p[v],w[i]);prefix.push(i)}}return val/N}
let seq0=seq(order[0]),best=Math.max(...order.map(seq)),ot=f.val/N,fresh=sum(xs.map((x,i)=>ov(x.q,x.p[v],w[i])))/N;
let a=rec[kind];for(let [k,val] of Object.entries({seq:seq0,best,ot,hybrid:Math.max(best,ot),fresh}))a[k]+=prob*val;
if(ot>best+1e-10)rec[kind==="first"?"otWinsFirst":"otWinsBlock"]++;}
rec.events++;}
rows.push(rec)}}
let summary=["threshold","count"].map(mode=>{let a=rows.filter(r=>r.mode===mode);return {mode,n:a.length,events:sum(a.map(r=>r.events)),winsFirst:sum(a.map(r=>r.otWinsFirst)),winsBlock:sum(a.map(r=>r.otWinsBlock)),first:Object.fromEntries(Object.keys(a[0].first).map(k=>[k,sum(a.map(r=>r.first[k]))/a.length])),block:Object.fromEntries(Object.keys(a[0].block).map(k=>[k,sum(a.map(r=>r.block[k]))/a.length]))}});
return {rows,summary,maxerr};
}
console.log(JSON.stringify(study(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
