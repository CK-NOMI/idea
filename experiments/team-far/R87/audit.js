function run(){
const seeds=[17,29,43], Es=[8,32,64], widths=[.1,.3,.7], modes=["independent","common","cluster"], n=128, rdim=3,k=2;let rows=[],maxErr=0,maxGradErr=0,viol=0,stableMismatch=0,caseCount=0,forward=0,gradCount=0;
function rng(s){return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296}}
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0), top=(s,k)=>s.map((v,i)=>i).sort((a,b)=>s[b]-s[a]||a-b).slice(0,k), equal=(a,b)=>a.length===b.length&&a.every(x=>b.includes(x));
for(let E of Es)for(let width of widths)for(let mode of modes)for(let seed of seeds){
const random=rng(seed+1009*Es.indexOf(E)+101*widths.indexOf(width)+13*modes.indexOf(mode)), rnd=()=>2*random()-1;
let sums={box:0,center:0,pair:0,boxSet:0,centerSet:0,pairSet:0,staleHit:0,staleWaste:0,mandatory:0,oracleSet:0,zero:0,pairChecks:0,overlap:0};
for(let t=0;t<n;t++){
const a=Array.from({length:E},()=>rnd()), common=Array.from({length:rdim},()=>rnd()*3), cluster=Array.from({length:4},()=>Array.from({length:rdim},()=>rnd()*1.5));
const B=Array.from({length:E},(_,e)=>Array.from({length:rdim},(_,j)=>rnd()*(mode==="independent"?1:.3)+(mode==="common"?common[j]:mode==="cluster"?cluster[e%4][j]:0)));
const center=Array.from({length:rdim},(_,j)=>B.reduce((s,x)=>s+x[j],0)/E);
const radius=B.map(b=>width*b.reduce((s,v)=>s+Math.abs(v),0)), centerRadius=B.map(b=>width*b.reduce((s,v,j)=>s+Math.abs(v-center[j]),0));
const nominal=top(a,k), actualZ=Array.from({length:rdim},()=>rnd()*width), score=a.map((v,i)=>v+dot(B[i],actualZ)), actual=top(score,k);
const verts=Array.from({length:1<<rdim},(_,mask)=>top(a.map((v,i)=>v+dot(B[i],Array.from({length:rdim},(_,j)=>(mask>>j&1?1:-1)*width))),k));
const stable=verts.every(x=>equal(x,nominal));sums.oracleSet+=Number(stable);
const rule=(kind)=>nominal.filter(i=>{let count=0;for(let j=0;j<E;j++)if(i!==j){let bound=kind==="box"?radius[i]+radius[j]:kind==="center"?centerRadius[i]+centerRadius[j]:width*B[i].reduce((s,v,h)=>s+Math.abs(v-B[j][h]),0);if(kind==="pair")sums.pairChecks++;if(a[i]-a[j]<=bound+1e-12)count++;}return count<k;});
const box=rule("box"), cen=rule("center"), pair=rule("pair"); 
const setCert=nominal.every(i=>Array.from({length:E},(_,j)=>j).filter(j=>!nominal.includes(j)).every(j=>a[i]-a[j]>width*B[i].reduce((s,v,h)=>s+Math.abs(v-B[j][h]),0)+1e-12));
if(setCert!==stable)stableMismatch++;
for(const [name,arr] of [["box",box],["center",cen],["pair",pair]]){sums[name]+=arr.length/k;sums[name+"Set"]+=Number(arr.length===k);if(arr.some(i=>!actual.includes(i)))viol++;}
const hit=nominal.filter(i=>actual.includes(i)).length;sums.staleHit+=hit/k;sums.staleWaste+=(k-hit)/k;sums.mandatory+=pair.length;sums.zero+=Number(pair.length===0);sums.overlap+=Number(setCert);
 // Actual toy expert forward/gradient: 4 input, 3 hidden tanh, scalar expert output. Expert and input fixed before router completion.
const x=Array.from({length:4},rnd), W=Array.from({length:E},()=>Array.from({length:3},()=>Array.from({length:4},rnd))), V=Array.from({length:E},()=>Array.from({length:3},rnd)), y=rnd();
const compute=i=>{forward++;const h=W[i].map(w=>Math.tanh(dot(w,x)));return {h,o:dot(V[i],h)}};
const cached=new Map(pair.map(i=>[i,compute(i)]));
const ref=actual.map(i=>compute(i));const used=actual.map(i=>cached.has(i)?cached.get(i):compute(i));const m=Math.max(...actual.map(i=>score[i]));let alpha=actual.map(i=>Math.exp(score[i]-m));const Z=alpha.reduce((s,v)=>s+v,0);alpha=alpha.map(v=>v/Z);
const calc=(f)=>{let o=alpha.reduce((s,v,i)=>s+v*f[i].o,0), d=o-y;let g=[];for(let j=0;j<actual.length;j++){const i=actual[j],h=f[j].h;g.push(d*alpha[j]*(f[j].o-o));for(let a=0;a<3;a++){g.push(d*alpha[j]*h[a]);for(let b=0;b<4;b++)g.push(d*alpha[j]*V[i][a]*(1-h[a]*h[a])*x[b]);}}return {o,g}};
const u=calc(ref),v=calc(used);maxErr=Math.max(maxErr,Math.abs(u.o-v.o));maxGradErr=Math.max(maxGradErr,...u.g.map((z,i)=>Math.abs(z-v.g[i])));gradCount+=u.g.length;caseCount++;
}
rows.push({E,width,mode,seed,...Object.fromEntries(Object.entries(sums).map(([key,val])=>[key,val/n]))});
}
const summary=[];for(let E of Es)for(let width of widths)for(let mode of modes){const a=rows.filter(x=>x.E===E&&x.width===width&&x.mode===mode);let o={E,width,mode};for(let key of ["box","center","pair","pairSet","oracleSet","overlap","staleHit","staleWaste","zero","pairChecks"])o[key]=a.reduce((s,v)=>s+v[key],0)/a.length;summary.push(o)}
return {spec:{seeds,Es,widths,modes,n,rdim,k,description:"synthetic affine router uncertainty, fixed toy expert parameters and fixed inputs, no SD rollout, SGD, LLM or network. Pair mandatory sufficient, full set cross-boundary exact strict-margin. Vertices certify full-set stability. Floating-point tolerance audit."},caseCount,forward,gradCount,viol,stableMismatch,maxErr,maxGradErr,summary,rows};
}
console.log(JSON.stringify(run(),null,2));
