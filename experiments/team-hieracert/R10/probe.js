// node experiments/team-hieracert/R10/probe.js
const input=require('./source_metrics.json');
function analyze(input){
 const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
 const ks=[1,2,4,8,16,32],costs=[.05,.1,.25],groups=[],methods=[...new Set(input.rows.map(r=>r.method))];
 for(const method of methods){
  const raw=input.rows.filter(r=>r.method===method);
  const records=raw.flatMap(r=>r.D.map((d,i)=>({d,a:r.A[i],task:r.task})));
  if(records.some(r=>!Number.isInteger(r.d)||!Number.isInteger(r.a)||r.a<0||r.d<r.a))throw Error("invalid diagnostic lengths");
  const N=records.length,allD=sum(records.map(r=>r.d)),allA=sum(records.map(r=>r.a)),fails=records.filter(r=>r.a<r.d).length;
  const naive=allA/allD,mle=allA/(allA+fails);
  const lower=k=>mean(records.map(r=>Math.min(r.a,k)));
  const upper=k=>mean(records.map(r=>r.a===r.d?k:Math.min(r.a,k)));
  const survival=[];let KM=1;
  for(let j=1;j<=32;j++){
    const knownSuccess=records.filter(r=>r.a>=j).length;
    const knownFailure=records.filter(r=>r.a<j&&r.a<r.d).length;
    const atRisk=records.filter(r=>r.d>=j&&r.a>=j-1).length;
    const events=records.filter(r=>r.a===j-1&&r.a<r.d).length;
    const unknown=N-knownSuccess-knownFailure;
    if(atRisk)KM*=1-events/atRisk;else KM=null;
    survival.push({j,knownSuccess,knownFailure,unknown,atRisk,events,lo:knownSuccess/N,hi:1-knownFailure/N,km:KM});
  }
  const caps=ks.map(k=>({k,lo:lower(k),hi:upper(k),unknownWidth:upper(k)-lower(k),naive:sum(Array.from({length:k},(_,j)=>naive**(j+1))),iidMLE:sum(Array.from({length:k},(_,j)=>mle**(j+1))),km:survival.slice(0,k).every(x=>x.km!==null)?sum(survival.slice(0,k).map(x=>x.km)):null}));
  const policy=costs.map(c=>{const scores=[{k:0,lo:1,hi:1,naive:1,iidMLE:1},...caps.map(x=>({k:x.k,lo:(1+x.lo)/(1+c*x.k),hi:(1+x.hi)/(1+c*x.k),naive:(1+x.naive)/(1+c*x.k),iidMLE:(1+x.iidMLE)/(1+c*x.k)}))];const best=key=>scores.reduce((a,b)=>a[key]>=b[key]?a:b);return {c,robust:best("lo"),optimistic:best("hi"),naive:best("naive"),iidMLE:best("iidMLE")}});
  groups.push({method,N,allD,allA,fails,naive,mle,fullAccept:N-fails,diagnosticMismatch:raw.filter(r=>r.N!==r.D.length).length,caps,survival,policy});
 }
 // Same observed D,A and two indistinguishable latent worlds, different optimal length.
 const toy={n:100,D:1,A:1,Flo:1,Fhi:8,cap:8,lo:1,hi:8,scoreShort:2/1.1,scoreLongLo:2/1.8,scoreLongHi:9/1.8};
 return {evidence:"Actual V8 numerical analysis of third-party published diagnostic length records. No model/GPU/HFL execution. Bounds apply to same block starts with prefix-consistent chain and no terminal truncation; not full-policy rollout or observed wall time.",source:input,groups,toy};
}
console.log(JSON.stringify(analyze(input),null,2));
