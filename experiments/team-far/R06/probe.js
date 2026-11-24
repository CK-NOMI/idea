// Forward adapted from FAR/R05, HCA/R09, FEIA/R11; node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
 const V=27,d=16,L=4,cap=2,sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),rms=x=>x.map(t=>t/Math.sqrt(dot(x,x)/d+1e-5)),arg=a=>a.reduce((j,v,i)=>v>a[j]?i:j,0),sm=a=>{let m=Math.max(...a),v=a.map(x=>Math.exp(x-m)),s=sum(v);return v.map(x=>x/s)};
 function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296}}
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=raw.slice(off,off+m);off+=m;return r});const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),first;for(let l=0;l<2;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!l)first=seq.at(-1)}return {first,deep:seq.at(-1)}}}}
 const docs=[...new Set(names.split(/\r?\n/).filter(s=>/^[a-z]{6,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>(s.charCodeAt(0)-97)%4===c)),train=buckets.map(b=>b.slice(0,Math.floor(b.length*.8))),test=buckets.map(b=>b.slice(-16)),enc=s=>s.split("").map(c=>c.charCodeAt(0)-97),rows=[];
 const avgW=ws=>ws[0].map((r,i)=>r.map((_,j)=>sum(ws.map(w=>w[i][j]))/ws.length));
 function audit(M,W,dirs,name,eps,noise,mode){
  const K=dirs.length,h0=[26,...enc(name).slice(0,2)],gold=enc(name).slice(2,6),states=Array.from({length:2*K},()=>[]),cache=new Map(),cnt={base:0,draft:0,direction:0,uncertain:0,calls:0,zeroPairs:0,greedyAudits:0,formal:2*K*L};
  const key=(i,h)=>`${mode==="independentAR"?i:0}:${h.join(",")}`;
  function get(i,h){let k=key(i,h),x=cache.get(k);if(!x){let f=M.feature(h0.concat(h));x={f,z:mat(W,f.deep),a:add(mat(W,f.deep),noise[i][h.length]),done:new Map()};cache.set(k,x);cnt.base++}return x}
  function verify(i,h){let j=i>>1,s=i%2===0?1:-1,k=mode==="independentAR"?i:j,x=get(i,h);let y=x.done.get(k);if(y)return y[i%2];const U=dirs[j],ids=U.ids,b=eps*sum(x.f.deep.map(Math.abs)),others=x.a.map((v,vv)=>U.set.has(vv)?-Infinity:v),w=arg(others),outside=others[w];let yy=[w,w],best=[outside,outside],touched=0;
   for(const v of ids)if(x.a[v]+b>=outside-1e-12){let delta=eps*dot(U.u[v],x.f.deep);touched++;for(let ss=0;ss<2;ss++){let a=x.a[v]+(ss===0?1:-1)*delta;if(a>best[ss]||(a===best[ss]&&v<yy[ss])){best[ss]=a;yy[ss]=v}}}
   cnt.direction+=touched;cnt.uncertain+=Number(touched>0);cnt.calls++;
   let ref=arg(x.a.map((a,v)=>a+(U.set.has(v)?s*eps*dot(U.u[v],x.f.deep):0)));if(ref!==yy[i%2])throw Error("local exactness");
   // dense greedy audit at same prefix
   let og=x.z.map((a,v)=>U.set.has(v)?-Infinity:a),wg=arg(og),bg=og[wg],gg=wg;for(const v of ids)if(x.z[v]+b>=bg-1e-12){let a=x.z[v]+s*eps*dot(U.u[v],x.f.deep);if(a>bg||(a===bg&&v<gg)){bg=a;gg=v}}
   let gr=arg(x.z.map((a,v)=>a+(U.set.has(v)?s*eps*dot(U.u[v],x.f.deep):0)));if(gg!==gr)throw Error("greedy exactness");cnt.greedyAudits++;
   x.done.set(k,yy);return yy[i%2]
  }
  // SD prefetch a single shared causal draft chain at cache miss, only exact base rows; lazy directional evaluation for committed prefixes.
  for(let t=0;t<L;t++){
   let groups=new Map();for(let i=0;i<2*K;i++){let k=key(i,states[i]);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(i)}
   for(const members of groups.values()){let i=members[0],h=states[i].slice();if(mode==="sharedSD"&&!cache.has(key(i,h))){for(let j=0;j<Math.min(cap,L-t);j++){let x=get(i,h);let draft=arg(add(mat(W,x.f.first),noise[i][h.length]));cnt.draft++;h=h.concat(draft)}}for(const ii of members)states[ii].push(verify(ii,states[ii]))}
  }
  // Independent dense AR reference on exact same noise stream and directions, cache feature by full prefix for auditing only.
  const rc=new Map();for(let i=0;i<2*K;i++){let h=[],U=dirs[i>>1],s=i%2===0?1:-1;for(let t=0;t<L;t++){let k=h.join(","),f=rc.get(k);if(!f){f=M.feature(h0.concat(h)).deep;rc.set(k,f)}let y=arg(mat(W,f).map((z,v)=>z+noise[i][t][v]+(U.set.has(v)?s*eps*dot(U.u[v],f):0)));if(y!==states[i][t])throw Error("sequence mismatch");h.push(y)}}
  let coeff=[];for(let j=0;j<K;j++){let a=states[2*j],b=states[2*j+1];let diff=(sum(a.map((v,t)=>Number(v===gold[t])))-sum(b.map((v,t)=>Number(v===gold[t]))))/(L*2*eps);coeff.push(diff);cnt.zeroPairs+=Number(a.join(",")===b.join(","))}
  return {coeff,cnt,states}
 }
 function loss(M,W){let v=0,n=0;for(let b of test)for(let name of b){let a=enc(name),h=[26,...a.slice(0,2)];for(let t=2;t<6;t++){v-=Math.log(sm(mat(W,M.feature(h).deep))[a[t]]);n++;h.push(a[t])}}return v/n}
 for(let wi=0;wi<raws.length;wi++){let M=model(raws[wi]);for(let seed of [11,29,73])for(let K of [2,4,8])for(let eps of [.005,.04]){
  let random=rng(seed+K*17),W=M.W.map(r=>r.slice()),init=loss(M,W),counts={sharedSD:{},sharedAR:{},pairIndependentAR:{}},updates=0,maxCoeff=0,coeffMom={shared:0,ind:0},overlap=0;
  function addCounts(mode,c){for(let k in c)counts[mode][k]=(counts[mode][k]||0)+c[k]}
  for(let round=0;round<2;round++){let edges=[];for(let e=0;e<2;e++){let w=W;for(let step=0;step<2;step++){let locals=[];for(let client=e*2;client<e*2+2;client++){
   let perm=Array.from({length:V},(_,v)=>v);for(let i=V-1;i>0;i--){let j=Math.floor(random()*(i+1));[perm[i],perm[j]]=[perm[j],perm[i]]}
   let dirs=Array.from({length:K},(_,j)=>{let ids=perm.filter((_,i)=>i%K===j),set=new Set(ids),u=Array.from({length:V},(_,v)=>Array.from({length:d},()=>set.has(v)?(random()<.5?-1:1):0));return {ids,set,u}});
   for(let j=0;j<K;j++)for(let jj=0;jj<j;jj++)overlap=Math.max(overlap,Math.abs(sum(dirs[j].u.map((r,v)=>dot(r,dirs[jj].u[v])))));
   const name=train[client][Math.floor(random()*train[client].length)],makeNoise=()=>Array.from({length:L},()=>Array.from({length:V},()=>-Math.log(-Math.log(random())))),g=makeNoise(),shared=Array.from({length:2*K},()=>g),ind=Array.from({length:K},()=>makeNoise()).flatMap(z=>[z,z]);
   let sd=audit(M,w,dirs,name,eps,shared,"sharedSD"),ar=audit(M,w,dirs,name,eps,shared,"sharedAR");
   if(JSON.stringify(sd.states)!==JSON.stringify(ar.states))throw Error("SD vs AR");
   // separate AR by direction, still shared +/- noise within each pair; distinct keys prevent cross-direction feature sharing.
   let ic={};let c=[];for(let j=0;j<K;j++){let r=audit(M,w,[dirs[j]],name,eps,[ind[2*j],ind[2*j+1]],"sharedAR");c.push(r.coeff[0]);for(let k in r.cnt)ic[k]=(ic[k]||0)+r.cnt[k]}
   addCounts("sharedSD",sd.cnt);addCounts("sharedAR",ar.cnt);addCounts("pairIndependentAR",ic);
   for(let j=0;j<K;j++)maxCoeff=Math.max(maxCoeff,Math.abs(sd.coeff[j]-ar.coeff[j]));
   // squared norm (orthogonal directions) is sum of coefficient squares times direction norms; empirical second moments not MSE.
   coeffMom.shared+=sum(sd.coeff.map((v,j)=>v*v*dirs[j].ids.length*d));coeffMom.ind+=sum(c.map((v,j)=>v*v*dirs[j].ids.length*d));
   locals.push(w.map((r,v)=>r.map((z,k)=>z+.0005*sum(dirs.map((u,j)=>u.u[v][k]*sd.coeff[j])))));updates++}
   w=avgW(locals)}edges.push(w)}W=avgW(edges)}
  rows.push({wi,seed,K,eps,updates,init,final:loss(M,W),maxCoeff,overlap,coeffMom,counts})
 }}
 const summary=[];for(let K of [2,4,8])for(let eps of [.005,.04]){let a=rows.filter(x=>x.K===K&&x.eps===eps),total={};for(let m of ["sharedSD","sharedAR","pairIndependentAR"]){total[m]={};for(let k in a[0].counts[m])total[m][k]=sum(a.map(x=>x.counts[m][k]));total[m].headRows=V*(total[m].base+total[m].draft)+total[m].direction}
 summary.push({K,eps,updates:sum(a.map(x=>x.updates)),init:sum(a.map(x=>x.init))/a.length,final:sum(a.map(x=>x.final))/a.length,counts:total,sharedSDvsIndependent:total.sharedSD.headRows/total.pairIndependentAR.headRows,sharedARvsIndependent:total.sharedAR.headRows/total.pairIndependentAR.headRows,SDvsSharedAR:total.sharedSD.headRows/total.sharedAR.headRows,sharedSecondMoment:sum(a.map(x=>x.coeffMom.shared)),indSecondMoment:sum(a.map(x=>x.coeffMom.ind))})}
 return {environment:"V8 CPU FP64 external same-architecture microGPT; true 4-token paired rollout, 2 cloud rounds/2 local regional synchronizations, deterministic reference checks; not GPU or physical HFL",rows,summary,maxCoeff:Math.max(...rows.map(r=>r.maxCoeff)),maxOrthogonalityError:Math.max(...rows.map(r=>r.overlap)),note:"Sparse directions have disjoint rows; sharedSD prefetch counts speculative base rows; evaluates direction certificate only on actually visited rows. All sequence and greedy audits passed. Head row count proxy; no timing. Pair-independent vs shared noise uses different random draws and is not quality-matched training comparison."}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
