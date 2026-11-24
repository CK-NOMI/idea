// Forward adapted from FAR/R04, team-hca/R09 and FEIA R11; node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names) {
 const V=27,d=16,L=4,cap=2,sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),rms=x=>x.map(t=>t/Math.sqrt(dot(x,x)/d+1e-5)),arg=a=>a.reduce((j,v,i)=>v>a[j]?i:j,0),sm=a=>{let m=Math.max(...a),v=a.map(x=>Math.exp(x-m)),s=sum(v);return v.map(x=>x/s)};
 function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296}}
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=raw.slice(off,off+m);off+=m;return r});const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),first;for(let l=0;l<2;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!l)first=seq.at(-1)}return {first,deep:seq.at(-1)}}}}
 const docs=[...new Set(names.split(/\r?\n/).filter(s=>/^[a-z]{6,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>(s.charCodeAt(0)-97)%4===c)),train=buckets.map(b=>b.slice(0,Math.floor(b.length*.8))),test=buckets.map(b=>b.slice(-32)),rows=[];
 const avgW=ws=>ws[0].map((r,i)=>r.map((_,j)=>sum(ws.map(w=>w[i][j]))/ws.length)),encode=s=>s.split("").map(c=>c.charCodeAt(0)-97),reward=(name,Y)=>sum(Y.map((v,i)=>Number(v===name[i+2])))/L;
 function verify(W,U,h,eps,g,signs,counts){
  let z=mat(W,h),a=add(z,g),w=arg(a),b=eps*sum(h.map(Math.abs)),dw=eps*dot(U[w],h),lower=Math.min(...signs.map(s=>a[w]+s*dw)),ids=[];for(let i=0;i<V;i++)if(a[i]+b>=lower-1e-12)ids.push(i);
  let values=ids.map(i=>i===w?dw:eps*dot(U[i],h)),wins=signs.map(s=>{let j=arg(ids.map((i,k)=>a[i]+s*values[k]));return ids[j]});
  const full=mat(U,h).map(v=>eps*v),ref=signs.map(s=>arg(a.map((v,i)=>v+s*full[i])));if(wins.some((v,i)=>v!==ref[i]))throw Error("lazy mismatch");
  const wg=arg(z),dg=eps*dot(U[wg],h),lg=Math.min(...signs.map(s=>z[wg]+s*dg)),ig=z.map((v,i)=>i).filter(i=>z[i]+b>=lg-1e-12);
  const gv=signs.map(s=>ig[arg(ig.map(i=>z[i]+s*eps*dot(U[i],h)))]),gr=signs.map(s=>arg(z.map((v,i)=>v+s*full[i])));if(gv.some((v,i)=>v!==gr[i]))throw Error("greedy mismatch");
  if(!ids.includes(w))throw Error("base winner");
  counts.calls++;counts.lazyRows+=ids.length;counts.fullRows+=V;counts.greedyRows+=ig.length;counts.signs+=signs.length;counts.bound+=Math.min(V,Math.exp(Math.min(700,2*b)));counts.b+=b;counts.maxRows=Math.max(counts.maxRows,ids.length);counts.maxB=Math.max(counts.maxB,b);return wins
 }
 function pair(M,W,U,eps,name,random,counts){
  let data=encode(name),prefix=[26,...data.slice(0,2)],gs=Array.from({length:L},()=>Array.from({length:V},()=>-Math.log(-Math.log(random())))),out=[[],[]],t=0;
  // speculative blocks only while both perturbation paths have the same prefix.
  while(t<L){
   if(out[0].join(",")===out[1].join(",")){
    let h=prefix.concat(out[0]),k=Math.min(cap,L-t),draft=[],features=[];counts.blocks++;
    for(let j=0;j<k;j++){let f=M.feature(h),x=arg(add(mat(W,f.first),gs[t+j]));features.push(f.deep);draft.push(x);h=h.concat(x);counts.draftRows+=V}
    const results=features.map((f,j)=>verify(W,U,f,eps,gs[t+j],[1,-1],counts));
    for(let j=0;j<k;j++){let a=results[j];out[0].push(a[0]);out[1].push(a[1]);t++;counts.committedCommon++;counts.attempts++;if(a[0]===draft[j]&&a[1]===draft[j]){counts.bothAccept++;continue}break}
   } else {
    for(let s=0;s<2;s++){let f=M.feature(prefix.concat(out[s])).deep;out[s].push(verify(W,U,f,eps,gs[t],[s===0?1:-1],counts)[0])}t++;
   }
  }
  let refs=[[],[]];
  for(let j=0;j<L;j++){if(refs[0].join(",")===refs[1].join(","))counts.arCalls++;else counts.arCalls+=2;for(let s=0;s<2;s++){let h=M.feature(prefix.concat(refs[s])).deep,z=mat(W,h),u=mat(U,h),sg=s===0?1:-1;refs[s].push(arg(z.map((v,i)=>v+gs[j][i]+sg*eps*u[i])))}}
  if(out.some((o,s)=>o.join(",")!==refs[s].join(",")))throw Error("sequence mismatch");
  counts.pairs++;counts.tokens+=2*L;for(let j=0;j<L;j++)counts.tokenMatch+=Number(out[0][j]===out[1][j]);
  return (reward(data,out[0])-reward(data,out[1]))/(2*eps)
 }
 function evalLoss(M,W){let loss=0,n=0;for(const bucket of test)for(const name of bucket){let a=encode(name),h=[26,...a.slice(0,2)];for(let t=2;t<6;t++){let p=sm(mat(W,M.feature(h).deep));loss-=Math.log(p[a[t]]);n++;h.push(a[t])}}return loss/n}
 for(let wi=0;wi<raws.length;wi++){let M=model(raws[wi]);for(let seed of [11,29,73])for(let eps of [.005,.02,.08]){
  let random=rng(seed),W=M.W.map(r=>r.slice()),counts={calls:0,lazyRows:0,fullRows:0,greedyRows:0,signs:0,bound:0,b:0,maxRows:0,maxB:0,blocks:0,draftRows:0,committedCommon:0,attempts:0,bothAccept:0,arCalls:0,pairs:0,tokens:0,tokenMatch:0},initial=evalLoss(M,W),updates=0;
  for(let round=0;round<2;round++){let edges=[];for(let e=0;e<2;e++){let w=W;for(let k=0;k<2;k++){let locals=[];for(let c=2*e;c<2*e+2;c++){let U=Array.from({length:V},()=>Array.from({length:d},()=>random()<.5?-1:1)),diff=0;for(let j=0;j<4;j++){let name=train[c][Math.floor(random()*train[c].length)];diff+=pair(M,w,U,eps,name,random,counts)/4}locals.push(w.map((r,i)=>r.map((v,j)=>v+.001*diff*U[i][j])));updates++}w=avgW(locals)}edges.push(w)}W=avgW(edges)}
  let final=evalLoss(M,W);rows.push({wi,seed,eps,updates,initial,final,...counts,rowsRatio:counts.lazyRows/counts.fullRows,greedyRatio:counts.greedyRows/counts.fullRows,relativeHeadMAC:(counts.fullRows+counts.lazyRows)/(2*counts.fullRows),vsARDenseHeadMAC:(counts.fullRows+counts.lazyRows)/(2*counts.arCalls*V),meanB:counts.b/counts.calls,meanBound:counts.bound/counts.calls,meanK:counts.lazyRows/counts.calls,tokenAgreement:counts.tokenMatch/counts.tokens*2,acceptBoth:counts.bothAccept/counts.attempts})
 }}
 let summary=[];for(let eps of [.005,.02,.08]){let a=rows.filter(x=>x.eps===eps),avg=f=>sum(a.map(f))/a.length,o={eps};for(let k of ["initial","final","rowsRatio","greedyRatio","relativeHeadMAC","vsARDenseHeadMAC","meanB","meanBound","meanK","tokenAgreement","acceptBoth"])o[k]=avg(x=>x[k]);for(let k of ["updates","pairs","tokens","calls","blocks","arCalls","fullRows","lazyRows","greedyRows","draftRows"])o[k]=sum(a.map(x=>x[k]));summary.push(o)}
 return {environment:"V8 FP64, actual microGPT two checkpoint same size, actual multi-token Gumbel paired self-spec generation with dense AR and greedy audits, logical HFL 2 rounds, no GPU or physical network, head MAC proxy excludes trunk and draft",rows,summary,bucketSizes:buckets.map(b=>b.length),trainingSizes:train.map(b=>b.length),testSizes:test.map(b=>b.length),note:"All sequence, sparse dense winner and greedy comparisons passed; all target logits were recomputed in audit, counts represent method not audit overhead."}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
