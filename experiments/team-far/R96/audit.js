const fs=require('fs');
function run(input){
const V=26,D=8,B=4,H=16,arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),sum=a=>a.reduce((s,v)=>s+v,0),mean=a=>sum(a)/a.length,hash=(s,k)=>{let h=k;for(let c of s)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h},rng=s=>()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+.5)/4294967296},soft=z=>{let m=Math.max(...z),p=z.map(x=>Math.exp(x-m)),a=sum(p);return p.map(x=>x/a)},draw=(p,r)=>{let u=r();for(let i=0;i<p.length;i++){u-=p[i];if(u<=0)return i}return p.length-1};
const names=[...new Set(input.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],ids=s=>[...s.slice(0,7)].map(c=>c.charCodeAt(0)-97);
let fits=0,updates=0,examples=0,gradChecks=0,maxGradRel=0,maxStateErr=0,maxKernel=0,rollouts=0;let rows=[],calCalls=0,pathDiff=0;
function init(L,r){return {E:arr(V*D,()=>.4*(r()-.5)),W:arr(V*D,()=>.4*(r()-.5)),b:arr(V,()=>0),A:arr(L*D,()=>.8+.2*r())};}
function state(w,anchor,h,L,shallow=false){let s=arr(D,d=>w.E[anchor*D+d]);for(let l=1;l<=Math.min(h.length,shallow?1:L);l++){let x=h[h.length-l];for(let d=0;d<D;d++)s[d]+=w.A[(l-1)*D+d]*w.E[x*D+d];}return s;}
function probability(w,s){let z=s.map(Math.tanh),p=soft(arr(V,y=>w.b[y]+sum(z.map((v,d)=>v*w.W[y*D+d]))));return {z,p};}
function forward(w,a,h,L,shallow=false){return probability(w,state(w,a,h,L,shallow));}
function zeros(w){return Object.fromEntries(Object.entries(w).map(([k,v])=>[k,v.map(()=>0)]));}
function grad(w,data,L){let g=zeros(w),n=0,loss=0;for(let s of data)for(let t=1;t<s.length;t++){let h=s.slice(1,t),y=s[t],{z,p}=forward(w,s[0],h,L),adj=arr(D,()=>0);loss-=Math.log(p[y]);n++;for(let j=0;j<V;j++){let e=p[j]-(j===y);g.b[j]+=e;for(let d=0;d<D;d++){g.W[j*D+d]+=e*z[d];adj[d]+=e*w.W[j*D+d];}}for(let d=0;d<D;d++){let u=adj[d]*(1-z[d]*z[d]);g.E[s[0]*D+d]+=u;for(let l=1;l<=Math.min(h.length,L);l++){let x=h[h.length-l],k=(l-1)*D+d;g.A[k]+=u*w.E[x*D+d];g.E[x*D+d]+=u*w.A[k];}}}for(let k in g)g[k]=g[k].map(x=>x/n);return {g,loss:loss/n,n};}
function nll(w,bank,L){return grad(w,bank,L).loss;}
function audit(w,s,L){let {g}=grad(w,[s],L);for(let k of Object.keys(w))for(let j of [0,Math.floor(w[k].length/2),w[k].length-1]){let old=w[k][j],ep=1e-5;w[k][j]=old+ep;let a=nll(w,[s],L);w[k][j]=old-ep;let b=nll(w,[s],L);w[k][j]=old;maxGradRel=Math.max(maxGradRel,Math.abs((a-b)/(2*ep)-g[k][j]));gradChecks++;}}

function calibrate(w,bank,L,seed){
let cnt=arr(B*V*V,()=>0),den=arr(B*V,()=>0),calls=0;
for(let k=0;k<bank.length;k++){let r=rng(seed+k*31),h=[],cand=[],surv=1;
for(let j=0;j<B;j++){let q=forward(w,bank[k][0],h,L,true).p,p=forward(w,bank[k][0],h,L).p,x=draw(q,r),alpha=Math.min(1,p[x]/q[x]),d=sum(p.map((v,i)=>Math.max(0,v-q[i])));calls+=2;den[j*V+x]++;
if(d>1e-15)for(let y=0;y<V;y++)cnt[(j*V+x)*V+y]+=surv*(1-alpha)*Math.max(0,p[y]-q[y])/d;
surv*=alpha;h.push(x);}
}return {cnt,den,calls};}
function decode(w,anchor,L,mode,seed,stats,budget){
let r=rng(seed),out=[],cache=new Map(),pc=0,qc=0,waves=0,rej=0,hit=0,repair=0,prefetch=0,used=0,lookups=0,rankOps=0;
const key=h=>h.slice(-L).join(",");
function get(h){let k=key(h);lookups++;if(cache.has(k)){hit++;return {p:cache.get(k),fresh:false};}let p=forward(w,anchor,h,L).p;cache.set(k,p);pc++;return {p,fresh:true};}
while(out.length<H){
if(mode==="AR"){let z=get(out);waves+=+z.fresh;out.push(draw(z.p,r));continue;}
let old=out.slice(),cand=[],n=Math.min(B,H-out.length),start=out.length;
for(let j=0;j<n;j++){let q=forward(w,anchor,old,L,true).p,x=draw(q,r);qc++;cand.push({q,x,h:old.slice()});old.push(x);}
let extra=new Set(),fresh=false,branch=[];
for(let c of cand){let z=get(c.h);c.p=z.p;fresh=fresh||z.fresh;}
if(budget){
let surv=1;
for(let j=0;j<n-1;j++){let c=cand[j],d=sum(c.p.map((v,i)=>Math.max(0,v-c.q[i]))),alpha=Math.min(1,c.p[c.x]/c.q[c.x]);
for(let y=0;y<V;y++)if(y!==c.x){let weight=mode==="oracle"?(d>1e-15?surv*(1-alpha)*Math.max(0,c.p[y]-c.q[y])/d:0):mode==="qrank"?c.q[y]: (stats.cnt[(j*V+c.x)*V+y]+.001*c.q[y])/(stats.den[j*V+c.x]+.001);
for(let t=j+1;t<Math.min(n,j+L+1);t++){let h=cand[t].h.slice();h[start+j]=y;let k=key(h);if(!cache.has(k))branch.push({h,k,w:weight});rankOps++;}}
surv*=alpha;}
let m=new Map();for(let z of branch){if(m.has(z.k))m.get(z.k).w+=z.w;else m.set(z.k,z);}
let chosen=[...m.values()].sort((a,b)=>b.w-a.w||a.k.localeCompare(b.k)).slice(0,budget);
let nFresh=0;for(let z of chosen){let v=get(z.h);if(v.fresh){prefetch++;extra.add(z.k);nFresh++;}}
if(mode==="oracle")waves+=+(nFresh>0);else fresh=fresh||nFresh>0;
}
waves+=+fresh;
for(let c of cand){let k=key(out),p;
if(k===key(c.h))p=c.p;else{if(extra.has(k))used++;let z=get(out);p=z.p;waves+=+z.fresh;repair+=+z.fresh;}
let y=c.x,ok=r()<Math.min(1,p[y]/c.q[y]);if(!ok){let z=p.map((v,i)=>Math.max(0,v-c.q[i])),d=sum(z);y=draw(z.map(v=>v/d),r);rej++;}
out.push(y);if(!ok&&mode==="prefix")break;
}
}
rollouts++;return {pc,qc,waves,rej,hit,repair,prefetch,used,lookups,rankOps,tokens:out.length,path:out.join(",")};}
for(let seed of [17,29,43])for(let layout of ["hash","prefix","last"])for(let L of [2,4]){
let bank=arr(4,c=>names.filter(s=>(layout==="hash"?hash(s,seed)%4:layout==="prefix"?(s.charCodeAt(0)-97)%4:(s.charCodeAt(s.length-1)-97)%4)===c).sort((a,b)=>hash(a,seed+13)-hash(b,seed+13)).slice(0,224).map(ids));
let tr=bank.map(a=>a.slice(0,128)),te=bank.map(a=>a.slice(160,224)),cal=bank.map(a=>a.slice(128,160));if(te.some(a=>a.length!==64))throw Error("small split");
let initial=init(L,rng(seed*100+L));let w=arr(2,()=>JSON.parse(JSON.stringify(initial)));fits++;
for(let t=0;t<72;t++){let local=arr(4,c=>grad(w[c>>1],tr[c].slice((t%8)*16,(t%8+1)*16),L));updates+=4;examples+=sum(local.map(x=>x.n));for(let e=0;e<2;e++)for(let k in w[e])for(let j=0;j<w[e][k].length;j++){let gl=mean(local.map(x=>x.g[k][j])),ge=(local[2*e].g[k][j]+local[2*e+1].g[k][j])/2;w[e][k][j]-=.65*(k==="A"?ge:gl);}}

let st=arr(2,e=>calibrate(w[e],[...cal[2*e],...cal[2*e+1]],L,seed*1000+e));calCalls+=sum(st.map(x=>x.calls));
let global={cnt:st[0].cnt.map((v,i)=>v+st[1].cnt[i]),den:st[0].den.map((v,i)=>v+st[1].den[i])};
for(let e=0;e<2;e++){audit(w[e],te[2*e][0],L);let test=[...te[2*e],...te[2*e+1]],loss=nll(w[e],test,L),base=nll(initial,test,L),reference=[];
for(let [mode,budget] of [["AR",0],["prefix",0],["continue",0],...["qrank","global","region","oracle"].flatMap(m=>[1,2,4,8].map(k=>[m,k]))]){
let z=[];for(let k=0;k<32;k++){let v=decode(w[e],test[k][0],L,mode,seed*100000+e*1000+k*17+L,mode==="region"?st[e]:global,budget);
if(mode==="continue")reference[k]=v.path;else if(budget&&v.path!==reference[k])pathDiff++;z.push(v);}
let cnt=Object.fromEntries(Object.keys(z[0]).filter(k=>k!=="path").map(key=>[key,sum(z.map(x=>x[key]))]));rows.push({seed,layout,L,e,mode,budget,loss,base,...cnt});}}}
let summary=[];for(let L of [2,4])for(let [mode,budget] of [["AR",0],["prefix",0],["continue",0],...["qrank","global","region","oracle"].flatMap(m=>[1,2,4,8].map(k=>[m,k]))]){
let a=rows.filter(x=>x.L===L&&x.mode===mode&&x.budget===budget),tokens=sum(a.map(x=>x.tokens));summary.push({L,mode,budget,n:a.length,loss:mean(a.map(x=>x.loss)),tokens,...Object.fromEntries(["pc","qc","waves","rej","hit","repair","prefetch","used","lookups","rankOps"].map(k=>[k,sum(a.map(x=>x[k]))/tokens]))});}
return {spec:"R95 model same architecture/training, reexecuted; disjoint 32 calibration/64 test names per client; 32 test anchors per edge (first client only), 16 generated tokens, B4 no bonus, same committed sample path for all continuation variants. qrank uses q(y); global/region count first-rejection correction mass by position and proposal token with Laplace-like q smoothing. Counts from calibration teacher and q forward, global pools counts across edges. Branches single substitution, dedup last-L keys, score sums approximate occupancy, no claim exact marginal. Oracle uses true initial target rows and incurs separate prefetch wave. Important implementation computes initial p for all modes before ranking, but qrank/global/region ranking does not use p and can in principle run before same batch; wave is proxy. rankOps counts candidate constructions, actual sorting/cache overhead excluded from linear proxy, calibration calls separate. No SD-aware formal training, GPU, physical HFL, CI.",names:names.length,fits,updates,examples,gradChecks,maxGradRel,calCalls,pathDiff,rollouts,summary,rows};
}

console.log(JSON.stringify(run(fs.readFileSync(process.argv[2],'utf8')),null,2));
