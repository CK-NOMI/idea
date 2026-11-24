function experiment(){
const N=8,D=12,R=160, seeds=Array.from({length:30},(_,i)=>11+19*i);
const sum=a=>a.reduce((x,y)=>x+y,0), dot=(a,b)=>sum(a.map((x,i)=>x*b[i])), norm=a=>Math.sqrt(dot(a,a)), add=(a,b)=>a.map((x,i)=>x+b[i]), sub=(a,b)=>a.map((x,i)=>x-b[i]), mul=(a,s)=>a.map(x=>x*s), zero=()=>Array(D).fill(0), mean=a=>mul(a.reduce(add,zero()),1/a.length);
const combos=[];for(let a=0;a<4;a++)for(let b=a+1;b<4;b++)for(let c=4;c<8;c++)for(let d=c+1;d<8;d++){let s=[a,b,c,d];if(a===0)combos.push([s,Array.from({length:8},(_,i)=>i).filter(i=>!s.includes(i))])}
const modes=["rawUniform","rawTime","cacheUniform","cacheTime","cacheJoint","predictTime","predictJoint","predictFull","oracle"];
let rows=[],maxBias=0,maxBound=0,gradientCalls=0,updates=0;
for(let seed of seeds)for(let kind of ["isotropic","anisotropic","noisy"])for(let layout of [0,1]){
let state=seed;const rand=()=>((state=Math.imul(state,1664525)+1013904223>>>0)/4294967296), unif=()=>2*rand()-1;
let A=Array.from({length:N},(_,i)=>Array.from({length:D},()=>2*unif()+(i<4?-1:1))), H=Array.from({length:N},(_,i)=>{let s=.3+2*rand();return Array.from({length:D},()=>kind==="isotropic"?s:.15+2.6*rand())});
let star=Array.from({length:D},(_,j)=>sum(H.map((h,i)=>h[j]*A[i][j]))/sum(H.map(h=>h[j])));
let w0=Array(D).fill(3), gap=w=>sum(H.map(h=>sum(h.map((x,j)=>x*(w[j]-star[j])**2))))/(2*N),target=.01*gap(w0);
let ar=[2,3,6,10,2.5,4,7,9],sd=ar.map((x,i)=>x*(layout===0?[.8,.8,.5,.35,.8,.8,.5,.35][i]:[.4,.45,.8,.85,.4,.45,.8,.85][i]));
let T=s=>.5+Math.max(...s.map(i=>sd[i])), pairT=p=>(T(p[0])+T(p[1]))/2;
let noise=Array.from({length:R+1},()=>Array.from({length:N},()=>Array.from({length:D},()=>kind==="noisy"?.15*unif():0))), coins=Array.from({length:R},()=>[rand(),rand()]);
function trueG(i,w){return w.map((x,j)=>H[i][j]*(x-A[i][j]))}
function grad(i,w,t){gradientCalls++;return add(trueG(i,w),noise[t][i])}
for(let mode of modes){
let w=w0.slice(), cache=Array.from({length:N},(_,i)=>grad(i,w,0)), anchors=Array.from({length:N},()=>w.slice()), slopes=Array(N).fill(1), seen=Array(N).fill(false), elapsed=.5+Math.max(...sd), first=null, firstR=null, vSum=0, scheduleChange=0, gSum=0, trace=[], warm=elapsed;
for(let t=0;t<R;t++){
let pred=mode.startsWith("predict")||mode==="oracle", cached=!mode.startsWith("raw");
let edgeSlope=[0,1].map(e=>{let ids=Array.from({length:4},(_,k)=>4*e+k).filter(i=>seen[i]);return ids.length?sum(ids.map(i=>slopes[i]))/ids.length:1});
let B=cache.map((g,i)=>cached?(pred?add(g,mul(sub(w,anchors[i]),edgeSlope[i>>2])):g):zero());
let rhat=cache.map((g,i)=>cached?mul(sub(w,anchors[i]),slopes[i]-(pred?edgeSlope[i>>2]:0)):g);
let trueResidual=Array.from({length:N},(_,i)=>sub(trueG(i,w),B[i]));
function imbalance(S,r,e=null){let z=zero();for(let i=0;i<N;i++)if(e===null||(i>>2)===e)z=add(z,mul(r[i],(S.includes(i)?1:-1)/8));return z}
let base=mean(B), full=mean(Array.from({length:N},(_,i)=>trueG(i,w))), bestT=combos.reduce((a,b)=>pairT(a)<=pairT(b)?a:b);
let bfloor=.0001+.04*sum(rhat.map(x=>dot(x,x)))/N+(kind==="noisy"?.0225*D/12:0);
function score(p){let S=p[0];if(mode==="oracle")return (bfloor+dot(imbalance(S,trueResidual),imbalance(S,trueResidual)))*pairT(p);
if(mode==="predictFull")return (bfloor+dot(imbalance(S,rhat),imbalance(S,rhat)))*pairT(p);
let bound=norm(imbalance(S,rhat,0))+norm(imbalance(S,rhat,1));let val=dot(imbalance(S,rhat),imbalance(S,rhat));maxBound=Math.max(maxBound,val-bound*bound);return (bfloor+bound*bound)*pairT(p)}
let pair=mode.endsWith("Uniform")?combos[Math.floor(coins[t][0]*combos.length)]:mode.endsWith("Time")?bestT:combos.reduce((a,b)=>score(a)<=score(b)?a:b);
if(pair!==bestT)scheduleChange++;
let S=pair[+(coins[t][1]>=.5)];
let exactA=add(base,mean(pair[0].map(i=>trueResidual[i]))),exactB=add(base,mean(pair[1].map(i=>trueResidual[i])));
maxBias=Math.max(maxBias,norm(sub(mul(add(exactA,exactB),.5),full)));
vSum+=dot(imbalance(S,trueResidual),imbalance(S,trueResidual));
let gs=S.map(i=>grad(i,w,t+1));let g=add(base,mean(gs.map((x,j)=>sub(x,B[S[j]]))));
for(let j=0;j<S.length;j++){let i=S[j],dw=sub(w,anchors[i]),dg=sub(gs[j],cache[i]),den=dot(dw,dw);if(den>1e-8){let s=Math.min(3,Math.max(.05,dot(dw,dg)/den));slopes[i]=seen[i]?.7*slopes[i]+.3*s:s;seen[i]=true}cache[i]=gs[j];anchors[i]=w.slice()}
w=sub(w,mul(g,.12));elapsed+=T(S);updates++;gSum+=norm(g);
if(first===null&&gap(w)<=target){first=elapsed;firstR=t+1}
if([0,9,29,79,159].includes(t))trace.push({round:t+1,gap:gap(w),cost:elapsed});
}
rows.push({seed,kind,layout,mode,first,firstR,finalGap:gap(w),time:elapsed,warm,variance:vSum/R,nonTime:scheduleChange,trace})
}
}
const avg=a=>sum(a)/a.length;
let summary=[];for(let kind of ["isotropic","anisotropic","noisy"])for(let layout of [0,1]){
let rr=rows.filter(r=>r.kind===kind&&r.layout===layout);let arms={};for(let m of modes){let a=rr.filter(x=>x.mode===m),base=rr.filter(x=>x.mode==="predictTime");let paired=a.map((x,i)=>[x,base[i]]).filter(([x,y])=>x.first!==null&&y.first!==null);arms[m]={hit:a.filter(x=>x.first!==null).length,avgHit:avg(a.filter(x=>x.first!==null).map(x=>x.first)),avgRounds:avg(a.filter(x=>x.first!==null).map(x=>x.firstR)),avgGap:avg(a.map(x=>x.finalGap)),avgV:avg(a.map(x=>x.variance)),avgNonTime:avg(a.map(x=>x.nonTime)),pairedRatio:paired.length?avg(paired.map(([x,y])=>x.first/y.first)):null,wins:paired.filter(([x,y])=>x.first<y.first).length}}summary.push({kind,layout,arms})}
return {N,D,R,seeds,gradientCalls,updates,maxBias,maxBound,summary,rows};
}
console.log(JSON.stringify(experiment(),null,2));
