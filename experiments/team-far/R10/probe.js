// CPU operator and hypothetical byte sensitivity; node probe.js source_metrics.json
const fs=require('node:fs');
function probe(metrics){
 const sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),zeros=(n,d)=>Array.from({length:n},()=>Array(d).fill(0)),diff=(a,b)=>Math.max(...a.flatMap((r,i)=>r.map((v,j)=>Math.abs(v-b[i][j]))));
 function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296}}
 const math=[];
 for(let seed of [11,29,73])for(let m of [16,64,256])for(let n of [1,4,16]){
  let random=rng(seed+m+n),d=8,heads=4,kv=2,scale=1/Math.sqrt(d),vec=()=>Array.from({length:d},()=>2*random()-1),pk=Array.from({length:kv},()=>Array.from({length:m},vec)),pv=Array.from({length:kv},()=>Array.from({length:m},vec)),baseK=Array.from({length:kv},()=>zeros(m,d)),baseV=Array.from({length:kv},()=>zeros(m,d)),regionK=Array.from({length:2},()=>Array.from({length:kv},()=>zeros(m,d))),regionV=Array.from({length:2},()=>Array.from({length:kv},()=>zeros(m,d))),errors={o:0,q:0,k:0,v:0,sk:0,sv:0,fd:0};
  for(let edge=0;edge<2;edge++)for(let client=0;client<2;client++)for(let t=0;t<n;t++)for(let h=0;h<heads;h++){
   let j=Math.floor(h/2),q=vec(),g=vec().map(v=>v/4),sK=Array.from({length:t+1},vec),sV=Array.from({length:t+1},vec),K=pk[j].concat(sK),V=pv[j].concat(sV);
   let z=K.map(k=>dot(k,q)*scale),max=Math.max(...z),e=z.map(v=>Math.exp(v-max)),den=sum(e),a=e.map(v=>v/den),o=Array.from({length:d},(_,k)=>sum(a.map((w,i)=>w*V[i][k]))),D=dot(g,o);
   function part(K,V){let zs=K.map(k=>dot(q,k)*scale),mx=Math.max(...zs),es=zs.map(v=>Math.exp(v-mx)),den=sum(es),l=mx+Math.log(den);return {l,o:Array.from({length:d},(_,k)=>sum(es.map((w,i)=>w*V[i][k]))/den)}}
   let p=part(pk[j],pv[j]),s=part(sK,sV),mx=Math.max(p.l,s.l),L=mx+Math.log(Math.exp(p.l-mx)+Math.exp(s.l-mx)),op=p.o.map((v,k)=>Math.exp(p.l-L)*v+Math.exp(s.l-L)*s.o[k]),dp=dot(g,op);
   errors.o=Math.max(errors.o,...o.map((v,k)=>Math.abs(v-op[k])));
   function backward(K,V,weights,normal){let dq=Array(d).fill(0),dk=zeros(K.length,d),dv=zeros(K.length,d);for(let i=0;i<K.length;i++){let w=weights[i],dz=w*(dot(g,V[i])-normal);for(let k=0;k<d;k++){dq[k]+=scale*dz*K[i][k];dk[i][k]=scale*dz*q[k];dv[i][k]=w*g[k]}}return {dq,dk,dv}}
   let dense=backward(K,V,a,D),rem=backward(pk[j],pv[j],pk[j].map(k=>Math.exp(dot(k,q)*scale-L)),dp),loc=backward(sK,sV,sK.map(k=>Math.exp(dot(k,q)*scale-L)),dp);
   errors.q=Math.max(errors.q,...dense.dq.map((v,k)=>Math.abs(v-rem.dq[k]-loc.dq[k])));errors.k=Math.max(errors.k,diff(dense.dk.slice(0,m),rem.dk));errors.v=Math.max(errors.v,diff(dense.dv.slice(0,m),rem.dv));errors.sk=Math.max(errors.sk,diff(dense.dk.slice(m),loc.dk));errors.sv=Math.max(errors.sv,diff(dense.dv.slice(m),loc.dv));
   for(let i=0;i<m;i++)for(let k=0;k<d;k++){baseK[j][i][k]+=dense.dk[i][k];baseV[j][i][k]+=dense.dv[i][k];regionK[edge][j][i][k]+=rem.dk[i][k];regionV[edge][j][i][k]+=rem.dv[i][k]}
   if(t===0&&h===0){const loss=(q,K,V)=>{let z=K.map(k=>dot(k,q)*scale),mx=Math.max(...z),a=z.map(v=>Math.exp(v-mx)),s=sum(a);return sum(a.map((w,i)=>w*dot(g,V[i])))/s},eps=1e-5,fd=(arr,i,fn)=>{let v=arr[i];arr[i]=v+eps;let a=fn();arr[i]=v-eps;let b=fn();arr[i]=v;return(a-b)/(2*eps)};errors.fd=Math.max(errors.fd,Math.abs(fd(q,0,()=>loss(q,K,V))-dense.dq[0]),Math.abs(fd(K[0],0,()=>loss(q,K,V))-dense.dk[0][0]),Math.abs(fd(V[0],0,()=>loss(q,K,V))-dense.dv[0][0]))}
  }
  for(let j=0;j<kv;j++){errors.k=Math.max(errors.k,diff(baseK[j],regionK[0][j].map((r,i)=>r.map((v,k)=>v+regionK[1][j][i][k]))));errors.v=Math.max(errors.v,diff(baseV[j],regionV[0][j].map((r,i)=>r.map((v,k)=>v+regionV[1][j][i][k]))))}
  math.push({seed,m,n,errors})
 }
 const data=metrics.methods.MTP2.rows.filter(x=>x.matches_baseline===true),Hq=28,Hkv=4,d=128,Dq=Hq*d,Dkv=Hkv*d,bytes=2,L=28;
 const scenarios=[];
 for(let share of [.25,.5,1])for(let cohort of [1,4,8])for(let over of [1,1.5,2]){
  let rep=0,remote=0,wins=0,hybrid=0,mem=0;
  for(const x of data){let m=Math.floor(x.prompt_tokens*share),nt=x.generated_tokens*cohort,nv=Math.ceil(x.generated_tokens*over)*cohort,bc=4*m*Dkv*bytes*L,br=(2*Dq*(nt+nv)+Hq*(nv+2*nt))*bytes*L;rep+=bc;remote+=br;hybrid+=Math.min(bc,br);mem+=2*m*Dkv*bytes*L;if(br<bc)wins++}
  scenarios.push({share,cohort,over,n:data.length,remoteToRep:remote/rep,oracleSwitchToRep:hybrid/rep,remoteWins:wins,meanResidentKV_MiB:mem/data.length/1048576,meanRepBytes:rep/data.length,meanRemoteBytes:remote/data.length})
 }
 return {environment:"V8 FP64 deterministic attention arithmetic; actual public log replay for prompt/output lengths only, no GPU, network, HFL training or measured latency",source:metrics.source,logRows:data.length,model:{Hq,Hkv,d,Dq,Dkv,L,wireBytes:bytes},math,scenarios,maxErrors:Object.fromEntries(Object.keys(math[0].errors).map(k=>[k,Math.max(...math.map(x=>x.errors[k]))])),note:"All remote scenario byte counts are hypothetical. Repeated cohorts assume same prompt length and one common prefix; no proof any trace prompt has actual shared public prefix. Treat full prompt as optimistic upper-bound m, not actual common prefix. Scalars are counted at BF16-sized 2 bytes solely for model; real LSE/adjoint normalizer typically FP32. No rollout or model update ran."}
}
console.log(JSON.stringify(probe(JSON.parse(fs.readFileSync(process.argv[2],'utf8'))),null,2));
