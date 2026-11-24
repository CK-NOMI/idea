// FCKG: synthetic exact moments and MC. Node.js, no dependencies.
function probe(){
const seeds=[11,29,73], sigmas=[.1,.5,1.5], V=32,N=160;let state=0;
const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296};
const normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
const sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),sq=a=>dot(a,a);
const soft=z=>{let m=Math.max(...z),a=z.map(x=>Math.exp(x-m)),s=sum(a);return a.map(x=>x/s)};
function stats(p,q,f){
 let n=p.length,mu=dot(p,f),gp=p.map((v,i)=>v*(f[i]-mu)),gq=q.map((v,i)=>v-p[i]),delta=sum(p.map((v,i)=>Math.max(0,v-q[i]))),pp=sq(p);
 let wp=p.map((v,i)=>v*(1-2*v+pp)),b=dot(wp,f)/sum(wp);
 let vp=dot(p,f.map(v=>(v-mu)**2))-sq(gp);
 let vscore=sum(wp.map((w,i)=>w*(f[i]-b)**2))-sq(gp);
 let vq=2*delta-sq(gq),vi=2*(1-dot(p,q))-sq(gq),vh=1-pp;
 // exact E <teacher pair,student pair> minus mean inner product; Z~p independently.
 let ec=0,ei=0,eh=0,active=0,activeI=0,zeroQ=0,zeroBoth=0;
 for(let y=0;y<n;y++){
   let m=p.map((v,j)=>.5*((j===y?f[y]-mu:0)-f[y]*v+v*f[j]));
   let my=m[y],mp=dot(m,p),mq=dot(m,q);
   eh+=p[y]*(mq-my);
   for(let x=0;x<n;x++){
     let w=(x===y?Math.min(p[y],q[x]):0)+(delta>0?Math.max(0,q[x]-p[x])*Math.max(0,p[y]-q[y])/delta:0);
     ec+=w*(m[x]-my);ei+=q[x]*p[y]*(m[x]-my);
     // Number of distinct nonzero token rows across two paths, assuming f has no accidental ties.
     // Teacher term zero if f(y)==f(z); student term zero if x==y.
     for(let z=0;z<n;z++){
       let ids=new Set();if(x!==y){ids.add(x);ids.add(y)}if(f[y]!==f[z]){ids.add(y);ids.add(z)}
       let w2=w*p[z],wi=q[x]*p[y]*p[z];active+=w2*ids.size;activeI+=wi*ids.size;
       if(x===y){zeroQ+=w2;if(f[y]===f[z])zeroBoth+=w2}
     }
   }
 }
 let cov=ec-dot(gp,gq),covI=ei-dot(gp,gq),covH=eh-dot(gp,gq);
 return {delta,gp2:sq(gp),gq2:sq(gq),vp,vscore,vq,vi,vh,cov,covI,covH,active,activeI,zeroQ,zeroBoth,
   joint0:vp+vq,jointHalf:vp+vq+cov,indHalf:vp+vi+covI,hardHalf:vp+vh+covH,
   teacherRejectLower:sq(gp)*(1/Math.max(delta,1e-300)-1)};
}
let rows=[];
for(let sigma of sigmas)for(let seed of seeds){
 state=seed;let sums={};
 for(let t=0;t<N;t++){let z=Array.from({length:V},normal),zq=z.map(v=>v+sigma*normal()),p=soft(z),q=soft(zq);let s=stats(p,q,z.map((v,i)=>v-zq[i]));for(let k in s)sums[k]=(sums[k]??0)+s[k]/N}
 rows.push({sigma,seed,...sums})
}
const stress=[];
for(const eps of [.1,.01,.001,.0001]){
 // q0 may underflow; calculate log ratio analytically instead.
 let M=1/eps,p=[eps,1-eps],q0=eps*Math.exp(-M),q=[q0,1-q0],f=[M,Math.log1p(-eps)-Math.log1p(-q0)];
 let s=stats(p,q,f);stress.push({eps,M,...s});
}
// MC at fixed external context. Independent extra p sample Z. No gradients through samples.
let mc=[];for(let seed of seeds){
 state=seed;const p=[.4,.3,.2,.1],q=[.25,.35,.1,.3],f=p.map((v,i)=>Math.log(v/q[i])),n=4,T=200000,mu=dot(p,f);
 const delta=sum(p.map((v,i)=>Math.max(0,v-q[i]))),r=p.map((v,i)=>Math.max(0,v-q[i])/delta);
 const draw=a=>{let t=u();for(let i=0;i<a.length;i++){t-=a[i];if(t<0)return i}return a.length-1};
 let sp=Array(n).fill(0),sqv=Array(n).fill(0),hist=Array(n).fill(0),acc=0,teacherOnAccept=0,teacherNonzero=0;
 for(let t=0;t<T;t++){
   let x=draw(q),a=u()<Math.min(1,p[x]/q[x]),y=a?x:draw(r),z=draw(p);hist[y]++;acc+=Number(a);
   let c=.5*(f[y]-f[z]);sp[y]+=c;sp[z]-=c;sqv[x]++;sqv[y]--;
   if(y!==z&&c!==0){teacherNonzero++;teacherOnAccept+=Number(a)}
 }
 mc.push({seed,T,accepted:acc,hist,teacherOnAccept,teacherNonzero,gp:sp.map(x=>x/T),gq:sqv.map(x=>x/T),trueGp:p.map((v,i)=>v*(f[i]-mu)),trueGq:q.map((v,i)=>v-p[i])})
}
return {environment:"JS V8 CPU; exact finite probability moments + synthetic MC; no real model/optimizer/network or wall time",V,N,seeds,sigmas,rows,stress,mc};
}
console.log(JSON.stringify(probe(),null,2));
