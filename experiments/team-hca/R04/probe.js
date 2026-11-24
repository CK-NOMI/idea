// HPRS synthetic CPU verification; node experiments/team-hca/R04/probe.js
function probe(){
let st=0;
const U=()=>{st=(Math.imul(1664525,st)+1013904223)>>>0;return (st+.5)/4294967296};
const N=()=>Math.sqrt(-2*Math.log(U()))*Math.cos(2*Math.PI*U());
const sum=a=>a.reduce((s,x)=>s+x,0);
const soft=a=>{let m=Math.max(...a),b=a.map(x=>Math.exp(x-m)),z=sum(b);return b.map(x=>x/z)};
function hull(lines,lo){
 lines=lines.slice().sort((a,b)=>a.m-b.m||b.b-a.b);
 let h=[];for(const l of lines){
  if(h.length&&l.m===h.at(-1).m)continue;
  let t=-Infinity;
  while(h.length){let k=h.at(-1);t=(k.b-l.b)/(l.m-k.m);if(t>k.t)break;h.pop()}
  h.push({...l,t:h.length?t:-Infinity});
 }
 return h.filter((l,i)=>i===h.length-1||h[i+1].t>lo);
}
const best=(lines,z)=>{let k=-1,v=0;for(const l of lines){let x=l.b+z*l.m;if(x>v){v=x;k=l.i}}return k};
let rows=[],tileRows=[],samplingRows=[];
for(const V of [1024,8192,32768])for(const sig of [.1,.5,1.5])for(const seed of [11,29,73]){
 st=seed;let count=0,full=0,root=0,err=0,alpha=0,cap8=0,cap16=0,bytes=0,tail=0;
 let T=48,S=4;
 for(let t=0;t<T;t++){
  let base=Array.from({length:V},()=>N()),p=soft(base),q=soft(base.map(x=>x+sig*N()));
  let z=V,a=p.map(x=>x*z),lines=a.map((x,i)=>{let e=-Math.log(U());return {m:-q[i]/e,b:x/e,i}});
  let hh=[],overflow8=false,overflow16=false;
  for(let s=0;s<S;s++){
   let l=lines.slice(s*V/S,(s+1)*V/S),lo=sum(a.slice(s*V/S,(s+1)*V/S));
   let h=hull(l,lo);hh.push(...h);count+=h.length;full+=l.length;
   overflow8 ||= h.length>8;overflow16 ||= h.length>16;
  }
  let k=best(lines,z),j=best(hh,z);err+=Number(k!==j);
  let delta=sum(p.map((x,i)=>Math.max(0,x-q[i])));alpha+=1-delta;
  root+=hh.length;cap8+=Number(overflow8);cap16+=Number(overflow16);
  bytes+=20*hh.length+8*S;tail+=delta;
 }
 rows.push({V,sig,seed,trials:T,meanHull:count/(T*S),meanTotalHull:root/T,cap8:cap8/T,cap16:cap16/T,meanBytes:bytes/T,accept:alpha/T,mismatch:err});
}
for(const sig of [.1,.5,1.5])for(const seed of [11,29,73]){
 st=seed;let total=0,peak=0,maxPeak=0,overflow8=0,overflow16=0,mismatch=0,weightedFallback16=0;
 const V=32768,T=24,S=4,B=128;
 for(let t=0;t<T;t++){
  const base=Array.from({length:V},()=>N()),a=base.map(x=>Math.exp(x)),z=sum(a),q=soft(base.map(x=>x+sig*N()));
  const lines=a.map((x,i)=>{let e=-Math.log(U());return {m:-q[i]/e,b:x/e,i}});
  let all=[],ov8=false,ov16=false;
  for(let s=0;s<S;s++){
   let h=[],mass=0,shardPeak=0;
   for(let j=s*V/S;j<(s+1)*V/S;j+=B){
    let tm=sum(a.slice(j,j+B)),tile=hull(lines.slice(j,j+B),tm);
    mass+=tm;h=hull(h.concat(tile),mass);shardPeak=Math.max(shardPeak,h.length);
   }
   all.push(...h);total+=h.length;peak+=shardPeak;maxPeak=Math.max(maxPeak,shardPeak);ov8 ||=shardPeak>8;ov16 ||=shardPeak>16;
  }
  mismatch+=Number(best(all,z)!==best(lines,z));
  overflow8+=Number(ov8);overflow16+=Number(ov16);
  const delta=sum(a.map((x,i)=>Math.max(0,x/z-q[i])));
  weightedFallback16+=Number(ov16)*delta;
 }
 tileRows.push({V,sig,seed,trials:T,tile:B,meanFinal:total/(T*S),meanPrefixPeak:peak/(T*S),maxPrefixPeak:maxPeak,cap8:overflow8/T,cap16:overflow16/T,weightedFallback16:weightedFallback16/T,mismatch});
}
const p=[.35,.25,.25,.15],q=[.15,.35,.1,.4],draw=w=>{let t=U();for(let i=0;i<w.length;i++){t-=w[i];if(t<0)return i}return w.length-1};
for(const seed of [11,29,73]){
 st=seed;let cnt=[0,0,0,0],res=[0,0,0,0],accept=0,mismatch=0;
 const T=100000;
 for(let t=0;t<T;t++){
  let x=draw(q),u=U();
  const a=p.map(x=>3.7*x),z=sum(a);
  const lines=a.map((x,i)=>{let e=-Math.log(U());return {m:-q[i]/e,b:x/e,i}});
  let h=hull(lines.slice(0,2),a[0]+a[1]).concat(hull(lines.slice(2),a[2]+a[3]));
  let k=best(h,z);
  mismatch+=Number(k!==best(lines,z));
  if(u<Math.min(1,p[x]/q[x]))accept++;else{x=k;res[x]++}
  cnt[x]++;
 }
 samplingRows.push({seed,T,accept,cnt,res,mismatch});
}
return {environment:"JS V8 CPU; no actual GPU, network or model training; synthetic LCG/Box-Muller",rows,tileRows,samplingRows};
}
console.log(JSON.stringify(probe(),null,2));
