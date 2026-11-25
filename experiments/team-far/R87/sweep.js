function run(){
const seeds=[17,29,43],Es=[8,32,64],widths=[.1,.3,.7],modes=["independent","common","cluster"],n=256,k=2;let rows=[],cases=0,viol=0,oracleMismatch=0,counter=0;
function rng(s){return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296}}
function sweep(a,b,i,w){
 let count=0,events=[];
 for(let j=0;j<a.length;j++)if(j!==i){
   const c=a[j]-a[i],d=b[j]-b[i], left=c-d*w;
   if(left>0||left===0&&j<i)count++;
   if(d!==0){const x=-c/d;if(x>-w&&x<w)events.push({x,d,j});}
 }
 // Random continuous endpoints are non-tied; audit explicit interior crossings, with deterministic ties.
 let max=count;events.sort((u,v)=>u.x-v.x);
 for(let p=0;p<events.length;){let q=p+1;while(q<events.length&&events[q].x===events[p].x)q++;
  let at=count,after=count;
  for(let h=p;h<q;h++){let {d,j}=events[h];if(d>0){after++;if(j<i)at++;}else{after--;if(j>i)at--;}}
  max=Math.max(max,at,after);count=after;p=q;
 }
 // Check right endpoint; left endpoint ties at crossing are conservatively handled separately.
 let right=0;for(let j=0;j<a.length;j++)if(j!==i){const z=(a[j]-a[i])+(b[j]-b[i])*w;if(z>0||z===0&&j<i)right++;}
 return Math.max(max,right);
}
function oracle(a,b,i,w){
 const ev=[-w,w];for(let j=0;j<a.length;j++)if(j!==i&&b[j]!==b[i]){let z=-(a[j]-a[i])/(b[j]-b[i]);if(z>-w&&z<w)ev.push(z);}
 ev.sort((u,v)=>u-v);const pts=ev.concat(ev.slice(1).map((x,j)=>(x+ev[j])/2));let mx=0;
 for(const x of pts){let c=0;for(let j=0;j<a.length;j++)if(i!==j){let f=(a[j]-a[i])+(b[j]-b[i])*x;if(f>1e-11||Math.abs(f)<=1e-11&&j<i)c++;}mx=Math.max(mx,c);}
 return mx;
}
for(let E of Es)for(let w of widths)for(let mode of modes)for(let seed of seeds){
const r=rng(seed+1009*Es.indexOf(E)+101*widths.indexOf(w)+13*modes.indexOf(mode)),rnd=()=>2*r()-1;
let s={pair:0,sweep:0,full:0,pairFull:0,oracle:0,nominalHit:0,extraCases:0};
for(let t=0;t<n;t++){
 let a=Array.from({length:E},rnd),c=rnd()*3,cl=Array.from({length:4},()=>rnd()*1.5),b=Array.from({length:E},(_,j)=>rnd()*(mode==="independent"?1:.3)+(mode==="common"?c:mode==="cluster"?cl[j%4]:0));
 let nom=a.map((v,i)=>i).sort((i,j)=>a[j]-a[i]||i-j).slice(0,k),z=rnd()*w,scores=a.map((v,i)=>v+b[i]*z),truth=scores.map((v,i)=>i).sort((i,j)=>scores[j]-scores[i]||i-j).slice(0,k);
 let p=0,u=0,o=0;for(let i of nom){
 let possible=0;for(let j=0;j<E;j++)if(j!==i&&a[i]-a[j]<=w*Math.abs(b[i]-b[j])+1e-12)possible++;
 let sw=sweep(a,b,i,w),or=oracle(a,b,i,w);
 if(sw!==or)oracleMismatch++;if(sw<k&&!truth.includes(i))viol++;
 p+=Number(possible<k);u+=Number(sw<k);o+=Number(or<k);
 }
 s.pair+=p/k;s.sweep+=u/k;s.oracle+=o/k;s.full+=Number(u===k);s.pairFull+=Number(p===k);s.extraCases+=Number(u>p);s.nominalHit+=nom.filter(i=>truth.includes(i)).length/k;cases++;
}
rows.push({E,w,mode,seed,...Object.fromEntries(Object.entries(s).map(([key,v])=>[key,v/n]))});
}
const ex={a:[0,-.1,-.1],b:[0,1,-1],i:0,w:1,k:2};counter=sweep(ex.a,ex.b,0,1);
const summary=[];for(let E of Es)for(let w of widths)for(let mode of modes){const a=rows.filter(x=>x.E===E&&x.w===w&&x.mode===mode);const o={E,w,mode};for(let key of ["pair","sweep","oracle","full","pairFull","extraCases","nominalHit"])o[key]=a.reduce((s,x)=>s+x[key],0)/a.length;summary.push(o)}
return {spec:{seeds,Es,widths,modes,n,k,description:"synthetic one scalar uncertainty, continuous random coefficients, no endpoint ties in generated cases, exact rank sweep vs direct endpoint/interior evaluation; no model training or SD execution"},cases,viol,oracleMismatch,counterexample:{...ex,maxOutrank:counter,pairPossible:2},summary,rows};
}
console.log(JSON.stringify(run(),null,2));
