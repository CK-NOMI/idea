// R05: deterministic audit of public benchmark metrics; no model execution.
// Run: node experiments/HIERASPEC/R05/probe.js
const fs=require("node:fs"),path=require("node:path");
const data=JSON.parse(fs.readFileSync(path.join(__dirname,"source_metrics.json"),"utf8"));
function analyze(data){
 const sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
 const quant=(a,q)=>a.slice().sort((x,y)=>x-y)[Math.floor((a.length-1)*q)];
 const names=Object.keys(data.methods),maps=Object.fromEntries(names.map(n=>[n,new Map(data.methods[n].rows.map(r=>[r.prompt_id,r]))]));
 const ids=[...maps.Medusa.keys()].filter(id=>names.every(n=>maps[n].has(id)));
 const row=(n,id)=>maps[n].get(id),ratio=(n,id)=>n==="AR"?1:row(n,id).method_wall_time_s/row(n,id).baseline_wall_time_s;
 const alpha=(n,id)=>row(n,id).accepted_draft_tokens/row(n,id).proposed_draft_tokens;
 const group=id=>{let x=row("Medusa",id).prompt_tokens;return x<512?0:x<768?1:2};
 const stats={};
 for(const n of names){let a=data.methods[n].rows;
  const B=sum(a.map(x=>x.baseline_wall_time_s)),T=sum(a.map(x=>x.method_wall_time_s)),D=sum(a.filter(x=>!x.matches_baseline).map(x=>x.baseline_wall_time_s));
  stats[n]={n:a.length,match:a.filter(x=>x.matches_baseline).length,tokens:sum(a.map(x=>x.generated_tokens)),accept:sum(a.map(x=>x.accepted_draft_tokens))/sum(a.map(x=>x.proposed_draft_tokens)),baseSeconds:B,methodSeconds:T,speedup:B/T,slower:a.filter(x=>x.method_wall_time_s>x.baseline_wall_time_s).length,targetForwards:sum(a.map(x=>x.target_forwards)),specSteps:sum(a.map(x=>x.speculation_steps)),draftForwards:sum(a.map(x=>x.draft_forwards)),p95Normalized:quant(ids.map(id=>ratio(n,id)),.95),failedBaselineSeconds:D,idealRerunSeconds:T+D,idealRerunSpeedup:B/(T+D),shadowARSpeedup:B/(B+T)};
 }
 const eligible=names.filter(n=>stats[n].match===stats[n].n),arms=["AR",...eligible];
 const strata=[0,1,2].map(g=>{let b=ids.filter(id=>group(id)===g);return {group:g,n:b.length,arms:Object.fromEntries(names.map(n=>[n,{meanNormalized:mean(b.map(id=>ratio(n,id))),mismatches:b.filter(id=>!row(n,id).matches_baseline).length}]))}});
 const best=(a,loss)=>a.reduce((u,v)=>loss(v)<loss(u)?v:u);
 const oracle={eligible:mean(ids.map(id=>Math.min(...arms.map(n=>ratio(n,id))))),allUnconstrained:mean(ids.map(id=>Math.min(1,...names.map(n=>ratio(n,id))))),freeDetectorRerun:mean(ids.map(id=>Math.min(1,...names.map(n=>ratio(n,id)+(row(n,id).matches_baseline?0:1)))))};
 let state=0;function rand(){state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296}
 const splits=[],bootstrap=[];
 for(let seed of [11,29,73]){
  state=seed;let train=[],test=[];
  for(let g of [0,1,2]){let a=ids.filter(id=>group(id)===g);for(let i=a.length-1;i>0;i--){let j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j],a[i]]}let k=Math.floor(a.length*.7);train.push(...a.slice(0,k));test.push(...a.slice(k))}
  const emp=(a,b)=>mean(b.map(id=>ratio(a,id)));
  const global=best(arms,n=>emp(n,train)),regional=[0,1,2].map(g=>best(arms,n=>emp(n,train.filter(id=>group(id)===g))));
  // Same-information flat sums exactly equal the hierarchical grouped sums.
  const acceptance=best(eligible,n=>-sum(train.map(id=>row(n,id).accepted_draft_tokens))/sum(train.map(id=>row(n,id).proposed_draft_tokens)));
  splits.push({seed,nTrain:train.length,nTest:test.length,global,regional,acceptance,globalMean:emp(global,test),regionalMean:mean(test.map(id=>ratio(regional[group(id)],id))),acceptanceMean:emp(acceptance,test),oracleEligible:mean(test.map(id=>Math.min(...arms.map(n=>ratio(n,id)))))});
  let reps=Object.fromEntries(names.map(n=>[n,[]]));
  for(let b=0;b<2000;b++){let a=Array.from({length:ids.length},()=>ids[Math.floor(rand()*ids.length)]);for(let n of names){reps[n].push(sum(a.map(id=>row(n,id).baseline_wall_time_s))/sum(a.map(id=>row(n,id).method_wall_time_s)))}}
  bootstrap.push({seed,replicates:2000,intervals:Object.fromEntries(names.map(n=>[n,[quant(reps[n],.025),quant(reps[n],.975)]]))});
 }
 return {source:data.source,methods:stats,pairedIds:ids.length,sameGeneratedLength:ids.filter(id=>new Set(names.map(n=>row(n,id).generated_tokens)).size===1).length,allMatched:ids.filter(id=>names.every(n=>row(n,id).matches_baseline)).length,observationallyEligible:eligible,mtp2VsMedusaAcceptanceRankReversal:ids.filter(id=>alpha("MTP2",id)>alpha("Medusa",id)&&ratio("MTP2",id)>ratio("Medusa",id)).length,strata,oracle,splits,bootstrap,zeroFailuresUpper95:1-Math.pow(.05,1/100)};
}
console.log(JSON.stringify(analyze(data),null,2));
