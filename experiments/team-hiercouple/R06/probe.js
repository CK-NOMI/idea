// node experiments/team-hiercouple/R06/probe.js
// This is a trace-anchored hypothetical two-worker queue and synthetic gradient check, not RL or HFL execution.
function run(source){
 const trace=source.methods.MTP1.rows.filter(r=>r.matches_baseline);
 const avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
 const meanAR=avg(trace.map(x=>x.baseline_wall_time_s));
 let state=0;const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296};
 const seeds=[11,29,73],G=8,N=8,eps=1e-6,d=16;let rows=[],error=0,groupsChecked=0;
 function queue(jobs,genEnd){let t=0;for(const j of jobs.sort((a,b)=>a.release-b.release||a.id-b.id))t=Math.max(t,j.release)+j.cost;return Math.max(t,genEnd)}
 function schedule(groups,mode,gen,ratio,policy,extra=0){
  const order=[];for(let x=0;x<G*N;x++){const g=mode==="contiguous"?Math.floor(x/N):x%G,i=mode==="contiguous"?x%N:Math.floor(x/G);order.push(groups[g][i]);}
  let time=0,arr=new Map();for(const x of order){time+=gen==="AR"?x.ar:x.sd;arr.set(x.id,time)}
  let jobs=[],nonzero=0,firstGate=0,firstComplete=0,extraJobs=0;
  for(const group of groups){
   let a=group.slice().sort((x,y)=>arr.get(x.id)-arr.get(y.id));
   let full=arr.get(a[N-1].id),first=a[0].r,idx=a.findIndex(x=>x.r!==first),mixed=idx>=0;
   if(mixed){nonzero++;firstGate+=arr.get(a[idx].id);firstComplete+=full}
   if(!mixed&&policy!=="eager")continue;
   const gate=mixed?arr.get(a[idx].id):full;
   for(const x of a){
    let release=policy==="group"?full:policy==="gated"?Math.max(gate,arr.get(x.id)):arr.get(x.id);
    jobs.push({id:x.id,release,cost:ratio*meanAR*x.tokens/128+(policy==="gated"?extra:0)});
    if(!mixed)extraJobs++;
   }
  }
  return {makespan:queue(jobs,time),genEnd:time,nonzero,jobs:jobs.length,extraJobs,meanGateLead:nonzero?(firstComplete-firstGate)/nonzero:0};
 }
 for(const seed of seeds)for(const p of [.1,.5,.9])for(const ratio of [.25,1,2]){
  state=seed;let sums={};const add=(k,v)=>sums[k]=(sums[k]||0)+v;
  for(let rep=0;rep<100;rep++){
   const groups=Array.from({length:G},(_,g)=>Array.from({length:N},(_,i)=>{
    const t=trace[Math.floor(u()*trace.length)],r=Number(u()<p),rho=.5+u();
    return {id:g*N+i,ar:t.baseline_wall_time_s,sd:t.method_wall_time_s,tokens:t.generated_tokens,r,rho,score:Array.from({length:d},()=>2*u()-1)};
   }));
   for(const group of groups){
    let m=group.reduce((s,x)=>s+x.r,0),mu=m/N,den=Math.sqrt(mu*(1-mu))+eps;
    let pos=Array(d).fill(0),neg=pos.slice(),full=pos.slice();
    for(const x of group){
     const A=(x.r-mu)/den;
     // clipped policy objective (maximize), off-boundary derivative d rho/d theta = rho*score.
     const active=A>0?x.rho<1.2:A<0?x.rho>.8:false;
     const unit=x.r?x.rho<1.2:x.rho>.8;
     for(let j=0;j<d;j++){
      full[j]+=active?A*x.rho*x.score[j]/N:0;
      if(unit)(x.r?pos:neg)[j]+=x.rho*x.score[j];
     }
    }
    for(let j=0;j<d;j++)error=Math.max(error,Math.abs(full[j]-((1-mu)*pos[j]-mu*neg[j])/(N*den)));
    groupsChecked++;
   }
   for(const gen of ["AR","MTP1"]){
    let g0=schedule(groups,"contiguous",gen,ratio,"group"),g1=schedule(groups,"roundrobin",gen,ratio,"group");
    let s0=schedule(groups,"contiguous",gen,ratio,"gated"),s1=schedule(groups,"roundrobin",gen,ratio,"gated"),e0=schedule(groups,"contiguous",gen,ratio,"eager");
    let h0=schedule(groups,"contiguous",gen,ratio,"gated",.02);
    const gb=Math.min(g0.makespan,g1.makespan),sb=Math.min(s0.makespan,s1.makespan);
    for(const [k,v] of Object.entries({groupContiguous:g0.makespan,groupRoundrobin:g1.makespan,gatedContiguous:s0.makespan,gatedRoundrobin:s1.makespan,gatedWith20ms:h0.makespan,eager:e0.makespan,groupBest:gb,gatedBest:sb,relativeSaving:1-sb/gb,sameScheduleSaving:1-s0.makespan/g0.makespan,gateLead:s0.meanGateLead,eagerExtraJobs:e0.extraJobs,mixed:s0.nonzero,genTime:g0.genEnd}))add(gen+"_"+k,v);
   }
  }
  rows.push({seed,p,ratio,...Object.fromEntries(Object.entries(sums).map(([k,v])=>[k,v/100]))});
 }
 return {evidence:"Public MTP1 paired service times reused from HIERASPEC/R05; random synthetic groups/rewards/gradient vectors, hypothetical serial generator and trainer overlap; no physical RL/HFL/SD or wall-time benchmark",source:source.source,traceRows:trace.length,meanAR,meanMTP:avg(trace.map(x=>x.method_wall_time_s)),settings:{seeds,G,N,d,replicates:100,rewardP:[.1,.5,.9],trainingCostRatio:[.25,1,2],eps},groupsChecked,maxGradientError:error,rows}
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("node:fs"),path=require("node:path");console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(path.join(__dirname,"../../HIERASPEC/R05/source_metrics.json"),"utf8"))),null,2))}
