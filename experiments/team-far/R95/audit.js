const fs=require('fs');
function run(input){
const V=26,D=8,B=4,H=16,arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),sum=a=>a.reduce((s,v)=>s+v,0),mean=a=>sum(a)/a.length,hash=(s,k)=>{let h=k;for(let c of s)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h},rng=s=>()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+.5)/4294967296},soft=z=>{let m=Math.max(...z),p=z.map(x=>Math.exp(x-m)),a=sum(p);return p.map(x=>x/a)},draw=(p,r)=>{let u=r();for(let i=0;i<p.length;i++){u-=p[i];if(u<=0)return i}return p.length-1};
const names=[...new Set(input.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],ids=s=>[...s.slice(0,7)].map(c=>c.charCodeAt(0)-97);
let fits=0,updates=0,examples=0,gradChecks=0,maxGradRel=0,maxStateErr=0,maxKernel=0,rollouts=0;let rows=[];
function init(L,r){return {E:arr(V*D,()=>.4*(r()-.5)),W:arr(V*D,()=>.4*(r()-.5)),b:arr(V,()=>0),A:arr(L*D,()=>.8+.2*r())};}
function state(w,anchor,h,L,shallow=false){let s=arr(D,d=>w.E[anchor*D+d]);for(let l=1;l<=Math.min(h.length,shallow?1:L);l++){let x=h[h.length-l];for(let d=0;d<D;d++)s[d]+=w.A[(l-1)*D+d]*w.E[x*D+d];}return s;}
function probability(w,s){let z=s.map(Math.tanh),p=soft(arr(V,y=>w.b[y]+sum(z.map((v,d)=>v*w.W[y*D+d]))));return {z,p};}
function forward(w,a,h,L,shallow=false){return probability(w,state(w,a,h,L,shallow));}
function zeros(w){return Object.fromEntries(Object.entries(w).map(([k,v])=>[k,v.map(()=>0)]));}
function grad(w,data,L){let g=zeros(w),n=0,loss=0;for(let s of data)for(let t=1;t<s.length;t++){let h=s.slice(1,t),y=s[t],{z,p}=forward(w,s[0],h,L),adj=arr(D,()=>0);loss-=Math.log(p[y]);n++;for(let j=0;j<V;j++){let e=p[j]-(j===y);g.b[j]+=e;for(let d=0;d<D;d++){g.W[j*D+d]+=e*z[d];adj[d]+=e*w.W[j*D+d];}}for(let d=0;d<D;d++){let u=adj[d]*(1-z[d]*z[d]);g.E[s[0]*D+d]+=u;for(let l=1;l<=Math.min(h.length,L);l++){let x=h[h.length-l],k=(l-1)*D+d;g.A[k]+=u*w.E[x*D+d];g.E[x*D+d]+=u*w.A[k];}}}for(let k in g)g[k]=g[k].map(x=>x/n);return {g,loss:loss/n,n};}
function nll(w,bank,L){return grad(w,bank,L).loss;}
function audit(w,s,L){let {g}=grad(w,[s],L);for(let k of Object.keys(w))for(let j of [0,Math.floor(w[k].length/2),w[k].length-1]){let old=w[k][j],ep=1e-5;w[k][j]=old+ep;let a=nll(w,[s],L);w[k][j]=old-ep;let b=nll(w,[s],L);w[k][j]=old;maxGradRel=Math.max(maxGradRel,Math.abs((a-b)/(2*ep)-g[k][j]));gradChecks++;}}
function decode(w,anchor,L,mode,seed){let r=rng(seed),out=[],cache=new Map(),pc=0,qc=0,waves=0,rej=0,hit=0,stale=0,repair=0,changed=0;
const key=h=>h.slice(-L).join(",");
function get(h){let k=key(h);if(cache.has(k)){hit++;return {p:cache.get(k),fresh:false};}let p=forward(w,anchor,h,L).p;cache.set(k,p);pc++;return {p,fresh:true};}
function proposal(h){qc++;return forward(w,anchor,h,L,true).p;}
while(out.length<H){
if(mode==="AR"){let z=get(out);waves+=+z.fresh;out.push(draw(z.p,r));continue;}
let old=out.slice(),cand=[],n=Math.min(B,H-out.length);
for(let j=0;j<n;j++){let q=proposal(old),x=draw(q,r);cand.push({q,x,h:old.slice(),s:state(w,anchor,old,L)});old.push(x);}
let fresh=false;for(let c of cand){let z=get(c.h);c.p=z.p;fresh=fresh||z.fresh;}waves+=+fresh;
for(let c of cand){
let same=key(out)===key(c.h),p;
if(same){p=c.p;}else{stale++;let z=get(out);p=z.p;waves+=+z.fresh;repair+=+z.fresh;
// exact correction of affine causal state from old suffix; audit against direct state
let s=c.s.slice();for(let l=1;l<=Math.min(out.length,L);l++){let x=out[out.length-l],y=c.h[c.h.length-l];if(x!==y)for(let d=0;d<D;d++)s[d]+=w.A[(l-1)*D+d]*(w.E[x*D+d]-w.E[y*D+d]);}
let a=state(w,anchor,out,L);maxStateErr=Math.max(maxStateErr,...s.map((v,i)=>Math.abs(v-a[i])));let pp=probability(w,s).p;maxStateErr=Math.max(maxStateErr,...pp.map((v,i)=>Math.abs(v-p[i])));}
let d=sum(p.map((v,i)=>Math.max(0,v-c.q[i])));for(let i=0;i<V;i++){let marginal=Math.min(p[i],c.q[i])+(d>1e-15?d*Math.max(0,p[i]-c.q[i])/d:0);maxKernel=Math.max(maxKernel,Math.abs(marginal-p[i]));}
let y=c.x,ok=r()<Math.min(1,p[y]/c.q[y]);if(!ok){let z=p.map((v,i)=>Math.max(0,v-c.q[i]));y=draw(z.map(v=>v/d),r);rej++;changed++;}
out.push(y);if(!ok&&mode==="prefix")break;
}
}
rollouts++;return {pc,qc,waves,rej,hit,stale,repair,changed,tokens:out.length};}
for(let seed of [17,29,43])for(let layout of ["hash","prefix","last"])for(let L of [2,4]){
let bank=arr(4,c=>names.filter(s=>(layout==="hash"?hash(s,seed)%4:layout==="prefix"?(s.charCodeAt(0)-97)%4:(s.charCodeAt(s.length-1)-97)%4)===c).sort((a,b)=>hash(a,seed+13)-hash(b,seed+13)).slice(0,192).map(ids));
let tr=bank.map(a=>a.slice(0,128)),te=bank.map(a=>a.slice(128,192));if(te.some(a=>a.length!==64))throw Error("small split");
let initial=init(L,rng(seed*100+L));let w=arr(2,()=>JSON.parse(JSON.stringify(initial)));fits++;
for(let t=0;t<72;t++){let local=arr(4,c=>grad(w[c>>1],tr[c].slice((t%8)*16,(t%8+1)*16),L));updates+=4;examples+=sum(local.map(x=>x.n));for(let e=0;e<2;e++)for(let k in w[e])for(let j=0;j<w[e][k].length;j++){let gl=mean(local.map(x=>x.g[k][j])),ge=(local[2*e].g[k][j]+local[2*e+1].g[k][j])/2;w[e][k][j]-=.65*(k==="A"?ge:gl);}}
for(let e=0;e<2;e++){audit(w[e],te[2*e][0],L);let test=[...te[2*e],...te[2*e+1]],loss=nll(w[e],test,L),base=nll(initial,test,L);for(let mode of ["AR","prefix","continue"]){let z=[];for(let k=0;k<64;k++)z.push(decode(w[e],test[k][0],L,mode,seed*100000+e*1000+k*17+L));let cnt=Object.fromEntries(Object.keys(z[0]).map(key=>[key,sum(z.map(x=>x[key]))]));rows.push({seed,layout,L,e,mode,loss,base,...cnt});}}}
let summary=[];for(let L of [2,4])for(let mode of ["AR","prefix","continue"]){let a=rows.filter(x=>x.L===L&&x.mode===mode),tokens=sum(a.map(x=>x.tokens));summary.push({L,mode,n:a.length,loss:mean(a.map(x=>x.loss)),base:mean(a.map(x=>x.base)),tokens,...Object.fromEntries(["pc","qc","waves","rej","hit","stale","repair"].map(k=>[k,sum(a.map(x=>x[k]))/tokens]))});}
// exhaustive binary B3 old-proposal causal kernel: sum all draft and formal histories
let exact=0,exCases=0;for(let seed of [1,2,3,4,5,6,7,8]){let r=rng(seed),Q=arr(3,i=>arr(2**i,()=>{let x=.05+.9*r();return [x,1-x]})),P=arr(3,i=>arr(2**i,()=>{let x=.05+.9*r();return [x,1-x]})),out=arr(8,()=>0);function go(i,d,y,m){if(i===3){out[y]+=m;return;}let q=Q[i][d],p=P[i][y],delta=Math.abs(q[0]-p[0]);for(let x=0;x<2;x++)for(let v=0;v<2;v++){let a=x===v?Math.min(q[x],p[x]):delta>1e-14?Math.max(0,q[x]-p[x])*Math.max(0,p[v]-q[v])/delta:0;if(a)go(i+1,2*d+x,2*y+v,m*a);}}go(0,0,0,1);for(let y=0;y<8;y++){let m=1,h=0;for(let i=0;i<3;i++){let x=(y>>(2-i))&1;m*=P[i][h][x];h=2*h+x;}exact=Math.max(exact,Math.abs(m-out[y]));exCases++;}}
return {spec:"External microGPT input.txt, fixed 736d88e..., distinct lowercase names len7–14, first 7 chars, fixed anchor + up to L recent raw token embeddings; tanh nonlinear readout. 4 logical clients, 2 edge lag vectors, common E/W/b, synchronous per-client gradients at same snapshot (not multi-step local FedAvg, not physical HFL); 3 seeds and 3 synthetic layouts, 72 updates; two memory lengths not two pretrained scales. Inner q drops lags>1. Fixed horizon16, B4 no bonus, no EOS; AR, prefix SD, no-discard old-proposal continuation use same target and same memoized last-L keys per sequence, but not same paths. Initial batch costs one conceptual wave if any miss, repair miss one additional wave, cache lookup/actual kernel/communication ignored. No actual parallel GPU, no real wall time; all probabilities calculated in JS, model is tiny new finite-memory neural LM, not a Transformer. No SD-driven training, no proof G2. State delta audit computes direct state too, counts exclude audit. Binary exact enumeration independent synthetic kernels. Strong flat with same delegated edge params can replicate.",names:names.length,fits,updates,examples,gradChecks,maxGradRel,maxStateErr,maxKernel,rollouts,exact,exCases,summary,rows};
}
console.log(JSON.stringify(run(fs.readFileSync(process.argv[2],'utf8')),null,2));
