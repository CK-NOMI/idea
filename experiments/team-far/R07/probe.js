// Forward adapted from FAR/R06, FAR/R05, HCA/R09 and FEIA/R11; node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
 const V=27,d=16,L=6,cap=2,sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),rms=x=>x.map(t=>t/Math.sqrt(dot(x,x)/d+1e-5)),arg=a=>a.reduce((j,v,i)=>v>a[j]?i:j,0),sm=a=>{let m=Math.max(...a),v=a.map(x=>Math.exp(x-m)),s=sum(v);return v.map(x=>x/s)};
 function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296}}
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=raw.slice(off,off+m);off+=m;return r});const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),first;for(let l=0;l<2;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!l)first=seq.at(-1)}return {first,deep:seq.at(-1)}}}}
 const docs=[...new Set(names.split(/\r?\n/).filter(s=>/^[a-z]{8,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>(s.charCodeAt(0)-97)%4===c)),train=buckets.map(b=>b.slice(0,Math.floor(b.length*.8))),test=buckets.map(b=>b.slice(-16)),enc=s=>s.split("").map(c=>c.charCodeAt(0)-97),rows=[];
 const avgW=ws=>ws[0].map((r,i)=>r.map((_,j)=>sum(ws.map(w=>w[i][j]))/ws.length));
 function audit(M,W,dirs,owners,name,eps,noise,mode){
  const K=dirs.length,h0=[26,...enc(name).slice(0,2)],gold=enc(name).slice(2,8),cache=new Map(),matDirs=new Map(),cnt={base:0,draft:0,direction:0,visits:0,materialized:0,screened:0,coarse:0,zeroPairs:0,formal:2*K*L,anchorNodes:0,anchorBound:0,localActive:0,greedyAudits:0,actualBaseUnchanged:0,lateFalsePositive:0};
  function get(h){let k=h.join(","),x=cache.get(k);if(!x){let f=M.feature(h0.concat(h)),z=mat(W,f.deep);x={f,z,a:add(z,noise[h.length]),done:new Map()};cache.set(k,x);cnt.base++}return x}
  function getRow(j,v){if(!matDirs.has(j)){matDirs.set(j,new Map());cnt.materialized++}let m=matDirs.get(j);if(!m.has(v)){let r=rng((dirs[j].seed^Math.imul(v+1,2654435761))>>>0);m.set(v,Array.from({length:d},()=>r()<.5?-1:1))}return m.get(v)}
  function rawRow(j,v){let r=rng((dirs[j].seed^Math.imul(v+1,2654435761))>>>0);return Array.from({length:d},()=>r()<.5?-1:1)}
  function verify(j,h){let x=get(h),y=x.done.get(j);if(y)return y;cnt.visits++;let b=eps*sum(x.f.deep.map(Math.abs)),outside=x.a.map((v,i)=>owners[i]===j?-Infinity:v),w=arg(outside),yy=[w,w],best=[outside[w],outside[w]];for(const v of dirs[j].ids)if(x.a[v]+b>=outside[w]-1e-12){let dv=eps*dot(getRow(j,v),x.f.deep);cnt.direction++;for(let s=0;s<2;s++){let a=x.a[v]+(s===0?1:-1)*dv;if(a>best[s]||(a===best[s]&&v<yy[s])){best[s]=a;yy[s]=v}}}
   for(let s=0;s<2;s++){let ref=arg(x.a.map((a,v)=>a+(owners[v]===j?(s===0?1:-1)*eps*dot(rawRow(j,v),x.f.deep):0)));if(ref!==yy[s])throw Error("verify mismatch");let zg=x.z.map((a,v)=>a+(owners[v]===j?(s===0?1:-1)*eps*dot(rawRow(j,v),x.f.deep):0)),outsideG=x.z.map((v,i)=>owners[i]===j?-Infinity:v),wg=arg(outsideG),bg=outsideG[wg],gg=wg;for(let v of dirs[j].ids)if(x.z[v]+b>=outsideG[wg]-1e-12){let a=zg[v];if(a>bg||(a===bg&&v<gg)){bg=a;gg=v}}if(gg!==arg(zg))throw Error("greedy mismatch");cnt.greedyAudits++}
   x.done.set(j,yy);return yy
  }
  let anchor=[],path=[],out=Array.from({length:K},()=>[[],[]]);
  if(mode!=="sharedAR"){
    let t=0;
    while(t<L){
     let h=anchor.slice(),k=mode==="anchorSD"?Math.min(cap,L-t):1,draft=[];
     if(mode==="anchorSD"){for(let a=0;a<k;a++){let x=get(h);let q=arg(add(mat(W,x.f.first),noise[h.length]));draft.push(q);h=h.concat(q);cnt.draft++}}
     for(let a=0;a<k;a++){let x=get(anchor),w=arg(x.a);path.push(x);anchor.push(w);t++;if(mode==="anchorSD"&&w!==draft[a])break}
    }
    cnt.anchorNodes=L;
    const first=Array(K).fill(L),coarse=Array(K).fill(false);
    for(let t=0;t<L;t++){
     const x=path[t],w=anchor[t],o=owners[w],a=x.a,b=eps*sum(x.f.deep.map(Math.abs)),p=sm(x.z),pB=dirs.map(z=>sum(z.ids.map(v=>p[v]))),c=Math.exp(b),c2=Math.exp(2*b);
     cnt.anchorBound+=sum(pB.map(x=>c*x/(1+(c-1)*x)))-1 +1-sum(p.map(x=>x/(1+(c2-1)*(1-x))));
     let risky=Array(K).fill(false),crude=Array(K).fill(false);
     for(let v=0;v<V;v++)if(v!==w){let j=owners[v],gap=a[w]-a[v];if(gap<=2*b+1e-12){crude[j]=true;crude[o]=true}if(j===o){if(gap<=2*b+1e-12)risky[o]=true}else if(gap<=b+1e-12){risky[j]=true;risky[o]=true}}
     for(let j=0;j<K;j++){if(crude[j])coarse[j]=true;if(risky[j]){first[j]=Math.min(first[j],t);cnt.localActive++}}
    }
    for(let j=0;j<K;j++){if(!coarse[j])cnt.coarse++;if(first[j]===L){cnt.screened++;out[j]=[anchor.slice(),anchor.slice()]}else{for(let s=0;s<2;s++){let h=anchor.slice(0,first[j]);for(let t=first[j];t<L;t++)h.push(verify(j,h)[s]);out[j][s]=h}}}
  }else{
    for(let j=0;j<K;j++)for(let s=0;s<2;s++)for(let t=0;t<L;t++)out[j][s].push(verify(j,out[j][s])[s])
  }
  let coeff=[],refCache=new Map();
  for(let j=0;j<K;j++){for(let s=0;s<2;s++){let h=[];for(let t=0;t<L;t++){let key=h.join(","),z=refCache.get(key);if(!z){let f=M.feature(h0.concat(h)).deep;z={f,z:mat(W,f)};refCache.set(key,z)}let y=arg(z.z.map((a,v)=>a+noise[t][v]+(owners[v]===j?(s===0?1:-1)*eps*dot(rawRow(j,v),z.f):0)));if(y!==out[j][s][t])throw Error("full sequence mismatch");h.push(y)}}
   let a=out[j][0],b=out[j][1];let diff=(sum(a.map((v,t)=>Number(v===gold[t])))-sum(b.map((v,t)=>Number(v===gold[t]))))/(L*2*eps);coeff.push(diff);cnt.zeroPairs+=Number(a.join(",")===b.join(","));if(mode!=="sharedAR"&&a.join(",")===anchor.join(",")&&b.join(",")===anchor.join(","))cnt.actualBaseUnchanged++;
  }
  if(mode!=="sharedAR"){cnt.lateFalsePositive=cnt.actualBaseUnchanged-cnt.screened;if(cnt.lateFalsePositive<0)throw Error("unsafe certificate")}
  return {coeff,cnt,out}
 }
 function loss(M,W){let v=0,n=0;for(let b of test)for(let name of b){let a=enc(name),h=[26,...a.slice(0,2)];for(let t=2;t<8;t++){v-=Math.log(sm(mat(W,M.feature(h).deep))[a[t]]);n++;h.push(a[t])}}return v/n}
 for(let wi=0;wi<raws.length;wi++){let M=model(raws[wi]);for(let seed of [11,29,73])for(let K of [4,8,16])for(let eps of [.005,.04]){
  let random=rng(seed+K*17),W=M.W.map(r=>r.slice()),init=loss(M,W),counts={anchorSD:{},anchorAR:{},sharedAR:{}},updates=0,maxCoeff=0,maxUpdate=0;
  function addCounts(mode,c){for(let k in c)counts[mode][k]=(counts[mode][k]||0)+c[k]}
  for(let round=0;round<2;round++){let edges=[];for(let e=0;e<2;e++){let w=W;for(let step=0;step<2;step++){let locals=[];for(let client=e*2;client<e*2+2;client++){
   let perm=Array.from({length:V},(_,v)=>v);for(let i=V-1;i>0;i--){let j=Math.floor(random()*(i+1));[perm[i],perm[j]]=[perm[j],perm[i]]}let owners=Array(V);
   let dirs=Array.from({length:K},(_,j)=>{let ids=perm.filter((_,i)=>i%K===j);for(let v of ids)owners[v]=j;return {ids,seed:Math.floor(random()*4294967296)>>>0}});
   const name=train[client][Math.floor(random()*train[client].length)],g=Array.from({length:L},()=>Array.from({length:V},()=>-Math.log(-Math.log(random()))));
   let sd=audit(M,w,dirs,owners,name,eps,g,"anchorSD"),ar=audit(M,w,dirs,owners,name,eps,g,"anchorAR"),sr=audit(M,w,dirs,owners,name,eps,g,"sharedAR");
   if(JSON.stringify(sd.out)!==JSON.stringify(ar.out)||JSON.stringify(sd.out)!==JSON.stringify(sr.out))throw Error("mode mismatch");
   for(let j=0;j<K;j++)maxCoeff=Math.max(maxCoeff,Math.abs(sd.coeff[j]-sr.coeff[j]),Math.abs(sd.coeff[j]-ar.coeff[j]));
   addCounts("anchorSD",sd.cnt);addCounts("anchorAR",ar.cnt);addCounts("sharedAR",sr.cnt);
   const stepW=co=>w.map((r,v)=>{let j=owners[v],rr=rng((dirs[j].seed^Math.imul(v+1,2654435761))>>>0);return r.map(z=>z+.0005*co[j]*(rr()<.5?-1:1))});let a=stepW(sd.coeff),b=stepW(sr.coeff);for(let v=0;v<V;v++)for(let k=0;k<d;k++)maxUpdate=Math.max(maxUpdate,Math.abs(a[v][k]-b[v][k]));locals.push(a);updates++}
   w=avgW(locals)}edges.push(w)}W=avgW(edges)}
  rows.push({wi,seed,K,eps,updates,init,final:loss(M,W),maxCoeff,maxUpdate,counts})
 }}
 const summary=[];for(let K of [4,8,16])for(let eps of [.005,.04]){let a=rows.filter(x=>x.K===K&&x.eps===eps),total={};for(let m of ["anchorSD","anchorAR","sharedAR"]){total[m]={};for(let k in a[0].counts[m])total[m][k]=sum(a.map(x=>x.counts[m][k]));total[m].headRows=V*(total[m].base+total[m].draft)+total[m].direction}
 const n=sum(a.map(x=>x.updates)),s=total.anchorSD,ar=total.anchorAR,b=total.sharedAR;summary.push({K,eps,updates:n,init:sum(a.map(x=>x.init))/a.length,final:sum(a.map(x=>x.final))/a.length,counts:total,certified:s.screened/(n*K),coarseCertified:s.coarse/(n*K),actualUnchanged:s.actualBaseUnchanged/(n*K),zeroPair:s.zeroPairs/(n*K),directionVisitRatio:s.visits/b.visits,materializedRatio:s.materialized/b.materialized,headVsSharedAR:s.headRows/b.headRows,headVsAnchorAR:s.headRows/ar.headRows,anchorARvsSharedAR:ar.headRows/b.headRows,meanLocalActive:s.localActive/(n*L),meanConditionalBound:s.anchorBound/(n*L)})}
 return {environment:"V8 CPU FP64 external same-architecture microGPT; 6-token paired rollout, 2 cloud rounds/2 local regional synchronizations; actual anchor SD and two AR controls plus dense reference checks; not GPU or physical HFL",rows,summary,maxCoeff:Math.max(...rows.map(r=>r.maxCoeff)),maxUpdate:Math.max(...rows.map(r=>r.maxUpdate)),note:"Rademacher signs generated lazily by fixed seed per direction and row; update arithmetic audit separately materializes signs for all rows even if coefficient zero, so counted materializations only during verification, not actual total RNG cost. Head rows include shallow draft, base and perturbation rows; no timing. Greedy checks cover evaluated prefixes; certificate proof applies with g=0. No expensive external reward query executed or skipped."}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
