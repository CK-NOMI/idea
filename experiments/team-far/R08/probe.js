// Forward adapted from FAR/R07, FAR/R05, HCA/R09 and FEIA/R11; node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
 const V=27,d=16,sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),rms=x=>x.map(t=>t/Math.sqrt(dot(x,x)/d+1e-5)),sm=a=>{let m=Math.max(...a),v=a.map(x=>Math.exp(x-m)),s=sum(v);return v.map(x=>x/s)};
 function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296}}
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=raw.slice(off,off+m);off+=m;return r});const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));return h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i])));for(let l=0;l<2;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))})}let hlast=seq.at(-1);return {z:mat(W,hlast),h:hlast,p:sm(mat(W,hlast))}}}
 const docs=[...new Set(names.split(/\r?\n/).filter(s=>/^[a-z]{6,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>(s.charCodeAt(0)-97)%4===c)),enc=s=>s.split("").map(c=>c.charCodeAt(0)-97);
 let rows=[],maxMeanErr=0,maxDomViolation=0,maxHazardViolation=0,maxBoundViolation=0;
 function caseRun(fs,dirs,eps,gold){
   let K=dirs.length, norm=dirs.map(x=>Math.sqrt(x.ids.length*d)),base=fs.map(x=>x.p);
   const D=dirs.map((dir,j)=>fs.map(x=>{let shift=Array(V).fill(0);for(let v of dir.ids)shift[v]=eps*dot(dir.u[v],x.h);let plus=sm(x.z.map((v,i)=>v+shift[i])),minus=sm(x.z.map((v,i)=>v-shift[i])),m=plus.map((v,i)=>Math.min(v,minus[i])),delta=sum(plus.map((v,i)=>Math.max(0,v-minus[i]))),B=Math.min(1,2*Math.exp(eps*sum(x.h.map(Math.abs)))*Math.expm1(eps*sum(x.h.map(Math.abs)))*sum(dir.ids.map(v=>x.p[v])));maxDomViolation=Math.max(maxDomViolation,...m.map((v,i)=>v-x.p[i]));maxHazardViolation=Math.max(maxHazardViolation,delta-B);return {plus,minus,m,delta,B,ap:plus.map((v,i)=>delta>0?Math.max(0,v-minus[i])/delta:0),am:minus.map((v,i)=>delta>0?Math.max(0,v-plus[i])/delta:0)}}));
   const means=Array(K).fill(0),direct=Array(K).fill(0),hyMean=Array(K).fill(0),lrMean=Array(K).fill(0),sec={paired:0,envelope:0,uniform:0,oracleHazard:0,hybrid:0,lr:0,lrOracle:0},cost={envelope:0,uniform:0,oracleHazard:0,hybrid:0,lr:4*V,pairedReward:0},active={envelope:0,paired:0},lrA=Array(K).fill(0),lrB=Array(K).fill(0),lrC=Array(K).fill(0);
   // Conditional contrast moments, independent suffix draws after first divergence.
   function moment(j,t,x,hybrid=false){
     let p=D[j][t?x+1:0];if(!p.delta)return {mu:0,m2:0};
     if(t){let a=p.ap[gold[1]],b=p.am[gold[1]],mu=(a-b)/2;return {mu,m2:(a+b-2*a*b)/4}}
     let pa=p.ap,ma=p.am,ap=0,am=0,ap2=0,am2=0;for(let v=0;v<V;v++){let u=Number(v===gold[0]),pp=(hybrid?D[j][v+1].minus:D[j][v+1].plus)[gold[1]],pm=D[j][v+1].minus[gold[1]];ap+=pa[v]*(u+pp)/2;am+=ma[v]*(u+pm)/2;ap2+=pa[v]*(u*u+(2*u+1)*pp)/4;am2+=ma[v]*(u*u+(2*u+1)*pm)/4}return {mu:ap-am,m2:Math.max(0,ap2+am2-2*ap*am)}
   }
   const rootMom=dirs.map((_,j)=>moment(j,0,0)),hyRoot=dirs.map((_,j)=>moment(j,0,0,true));
   for(let j=0;j<K;j++){let jp=0,jm=0;for(let x=0;x<V;x++){jp+=D[j][0].plus[x]*(Number(x===gold[0])+D[j][x+1].plus[gold[1]])/2;jm+=D[j][0].minus[x]*(Number(x===gold[0])+D[j][x+1].minus[gold[1]])/2}direct[j]=(jp-jm)/(2*eps)}
   for(let x=0;x<V;x++){
     let w=base[0][x],items=[];
     for(let j=0;j<K;j++){let root=D[j][0],next=D[j][x+1],a=root.m[x]/w,h=root.plus[x]/w;
       for(let t=0;t<2;t++){let p=t?next:root,W=t?a:1,H=t?h:1,mo=t?moment(j,1,x):rootMom[j],hm=t?mo:hyRoot[j],k=norm[j]*W*p.delta/(2*eps),kh=norm[j]*H*p.delta/(2*eps),e=norm[j]*p.B;let rowCost=2*V+(t?2*dirs[j].ids.length:3*dirs[j].ids.length+V*(2-root.ap[x]-root.am[x]));items.push({j,t,W,H,p,mo,hm,k,kh,e,rowCost});means[j]+=w*W*p.delta*mo.mu/(2*eps);hyMean[j]+=w*H*p.delta*hm.mu/(2*eps);sec.paired+=w*W*p.delta*mo.m2*norm[j]**2/(4*eps*eps);active.paired+=w*W*p.delta;cost.pairedReward+=2*w*W*p.delta;
       }
     }
     let E=sum(items.map(z=>z.e)),A=sum(items.map(z=>z.k));let condEnvelope=0;
     for(let z of items){let pi=E?z.e/E:0,pu=1/items.length,po=A?z.k/A:0; if(pi>0){sec.envelope+=w*z.k*z.k*z.mo.m2/pi;sec.hybrid+=w*z.kh*z.kh*z.hm.m2/pi;cost.envelope+=w*pi*z.rowCost;cost.hybrid+=w*pi*z.rowCost;active.envelope+=w*pi*Number(z.k>0);condEnvelope+=z.k*z.k*z.mo.m2/pi}sec.uniform+=w*z.k*z.k*z.mo.m2/pu;cost.uniform+=w*pu*z.rowCost;if(po){sec.oracleHazard+=w*z.k*z.k*z.mo.m2/po;cost.oracleHazard+=w*po*z.rowCost}}
     maxBoundViolation=Math.max(maxBoundViolation,condEnvelope-(E/(2*eps))**2);
     for(let y=0;y<V;y++){let by=base[x+1][y],prob=w*by,R=(Number(x===gold[0])+Number(y===gold[1]))/2;for(let j=0;j<K;j++){let u=(D[j][0].plus[x]*D[j][x+1].plus[y]-D[j][0].minus[x]*D[j][x+1].minus[y])/(prob*2*eps),s=prob*u*u*norm[j]**2;lrMean[j]+=prob*u*R;sec.lr+=s*R*R;lrA[j]+=s;lrB[j]+=s*R;lrC[j]+=s*R*R}}
   }
   sec.lrOracle=sum(lrC.map((v,j)=>v-(lrA[j]>0?lrB[j]**2/lrA[j]:0)));
   const meanSq=sum(direct.map((v,j)=>v*v*norm[j]**2));for(let j=0;j<K;j++)maxMeanErr=Math.max(maxMeanErr,Math.abs(means[j]-direct[j]),Math.abs(hyMean[j]-direct[j]),Math.abs(lrMean[j]-direct[j]));
   let variances=Object.fromEntries(Object.entries(sec).map(([k,v])=>[k,Math.max(0,v-meanSq)]));
   return {meanSq,variances,cost,active,mean:direct,maxBoundViolation,maxHazardViolation};
 }
 for(let wi=0;wi<raws.length;wi++){let f=model(raws[wi]);for(let seed of [11,29,73])for(let K of [4,8]){
  let random=rng(seed+17*K),cases=[];for(let c=0;c<4;c++){
   let name=buckets[c][Math.floor(random()*buckets[c].length)],chars=enc(name),h=[26,...chars.slice(0,2)],fs=[f(h),...Array.from({length:V},(_,v)=>f(h.concat(v)))],perm=Array.from({length:V},(_,v)=>v);for(let i=V-1;i>0;i--){let j=Math.floor(random()*(i+1));[perm[i],perm[j]]=[perm[j],perm[i]]}
   let dirs=Array.from({length:K},(_,j)=>{let ids=perm.filter((_,i)=>i%K===j),s=new Set(ids),u=Array.from({length:V},(_,v)=>Array.from({length:d},()=>s.has(v)?(random()<.5?-1:1):0));return {ids,u}});
   cases.push({fs,dirs,gold:chars.slice(2,4)})
  }
  for(let eps of [.001,.005,.04]){let a=cases.map(c=>caseRun(c.fs,c.dirs,eps,c.gold)),o={wi,seed,K,eps,n:a.length,meanSq:sum(a.map(v=>v.meanSq))/a.length,variances:{},cost:{},active:{}};for(let k of Object.keys(a[0].variances))o.variances[k]=sum(a.map(v=>v.variances[k]))/a.length;for(let k of Object.keys(a[0].cost))o.cost[k]=sum(a.map(v=>v.cost[k]))/a.length;for(let k of Object.keys(a[0].active))o.active[k]=sum(a.map(v=>v.active[k]))/a.length;rows.push(o)}
 }}
 const summary=[];for(let K of [4,8])for(let eps of [.001,.005,.04]){let a=rows.filter(x=>x.K===K&&x.eps===eps),o={K,eps,n:sum(a.map(v=>v.n)),meanSq:sum(a.map(v=>v.meanSq))/a.length,variances:{},cost:{},active:{}};for(let group of ["variances","cost","active"])for(let k of Object.keys(a[0][group]))o[group][k]=sum(a.map(v=>v[group][k]))/a.length;summary.push(o)}
 return {environment:"V8 CPU FP64 external same-size 2-layer microGPT; exact 2-token probability/conditional moment enumeration, not training or sampled SD or GPU; reward correct character fraction",rows,summary,maxMeanErr,maxDomViolation,maxHazardViolation,maxBoundViolation}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
