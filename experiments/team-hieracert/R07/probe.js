// Run node experiments/team-hieracert/R07/probe.js from repo root.
const path=require('node:path');
const hc=['chat','code','summarization'].flatMap(x=>require(path.join(__dirname,'../../HC/R05',x+'.json')).pairs);
const hs=require(path.join(__dirname,'../../HIERASPEC/R05/source_metrics.json'));
function analyze(input){
 const sum=a=>a.reduce((s,x)=>s+x,0), median=a=>{let b=a.slice().sort((x,y)=>x-y);return (b[Math.floor((b.length-1)/2)]+b[Math.ceil((b.length-1)/2)])/2};
 const sets={HC_all:input.hc.map(p=>[p.ar/1000,p.sd/1000]),HC_matched:input.hc.filter(p=>p.equalText&&p.arTokens===p.sdTokens).map(p=>[p.ar/1000,p.sd/1000]),MTP1_all:input.mtp.map(p=>[p.baseline_wall_time_s,p.method_wall_time_s]),Medusa_all:input.medusa.map(p=>[p.baseline_wall_time_s,p.method_wall_time_s]),Medusa_matched:input.medusa.filter(p=>p.matches_baseline).map(p=>[p.baseline_wall_time_s,p.method_wall_time_s])};
 const rows=[];let maxIdentityError=0;
 for(const [name,data] of Object.entries(sets))for(const bScale of [.5,.75,1,1.25])for(const reduce of [.25,.5]){
  const B=median(data.map(x=>x[0]))*bScale,B1=B*(1-reduce);
  let t00=0,t10=0,t01=0,t11=0,interaction=0,positive=0,negative=0,bothPassOnly=0;
  for(const [A,A1] of data){
   const a=Math.max(A,B),h=Math.max(A,B1),s=Math.max(A1,B),hs=Math.max(A1,B1);
   const I=h+s-a-hs;
   const overlap=Math.max(0,Math.min(Math.max(A,A1),B)-Math.max(Math.min(A,A1),B1))*Math.sign(A-A1);
   maxIdentityError=Math.max(maxIdentityError,Math.abs(I-overlap));
   t00+=a;t10+=h;t01+=s;t11+=hs;interaction+=I;
   positive+=Number(I>1e-10);negative+=Number(I<-1e-10);
   bothPassOnly+=Number(h>.85*a&&s>.85*a&&hs<=.85*a);
  }
  rows.push({name,n:data.length,bScale,reduce,B,B1,t00,t10,t01,t11,interaction,interactionPct:100*interaction/t00,Hsaving:100*(1-t10/t00),SDsaving:100*(1-t01/t00),jointSaving:100*(1-t11/t00),positive,negative,bothPassOnly});
 }
 const example=[100,100,100,60];
 return {evidence:"Archived public inference timings plus hypothetical parallel non-generation path; arithmetic replay, not HFL execution. Matched subsets selected after output comparison, not deployable routing.",source:{HC:"experiments/HC/R05/{chat,code,summarization}.json",HIERASPEC:"experiments/HIERASPEC/R05/source_metrics.json"},example:{T00:example[0],T10:example[1],T01:example[2],T11:example[3],interaction:40},maxIdentityError,rows};
}
console.log(JSON.stringify(analyze({hc,mtp:hs.methods.MTP1.rows,medusa:hs.methods.Medusa.rows}),null,2));
