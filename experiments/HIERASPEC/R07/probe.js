// HIERASPEC/R07. Node.js CPU; source data from R05 public benchmark metrics.
// node experiments/HIERASPEC/R07/probe.js
const fs=require("node:fs"),path=require("node:path");
const data=JSON.parse(fs.readFileSync(path.join(__dirname,"../R05/source_metrics.json"),"utf8"));
function run(data){
 const seeds=[11,29,73],sizes=[1024,65536,1048576],slacks=[0,.05,.5],scales=[.25,1,2],header=16,rate=100e6/8;
 let st=0;const u=()=>{st=(Math.imul(1664525,st)+1013904223)>>>0;return(st+.5)/4294967296};
 const sum=a=>a.reduce((s,x)=>s+x,0),rows=[];
 // 100 independent request windows: actual MTP1 timing/counts, hypothetical unrelated update release and packet sizes.
 const records=data.methods.MTP1.rows.map(x=>({D:4*Math.max(0,x.target_forwards-x.accepted_draft_tokens)+header*x.speculation_steps,AR:4*x.generated_tokens+header,t:x.method_wall_time_s,b:x.baseline_wall_time_s}));
 for(const seed of seeds)for(const U of sizes)for(const slack of slacks)for(const scale of scales){
  st=seed;
  let plain=0,coded=0,oracle=0,paired=0,netSaved=0,wait=0,deadlineMissIfForced=0,arPlain=0,arOracle=0;
  for(const r of records){
   const releaseUp=r.b*scale*(.8+.4*u()),releaseDown=r.t,delay=Math.abs(releaseUp-releaseDown);
   const benefit=Math.min(U,r.D),extra=header;
   // two ingress transmissions and relay broadcast(s), equal serial rates. Extra combined header on coded broadcast.
   plain+=2*(U+r.D);oracle+=2*(U+r.D)-Math.max(0,benefit-extra);
   if(delay<=slack&&benefit>extra){paired++;coded+=2*(U+r.D)-benefit+extra;wait+=delay;netSaved+=(benefit-extra)/rate;}
   else coded+=2*(U+r.D);
   if(delay>slack)deadlineMissIfForced++;
   arPlain+=2*(U+r.AR);arOracle+=2*(U+r.AR)-Math.max(0,Math.min(U,r.AR)-extra);
  }
  rows.push({seed,U,slack,scale,paired,plain,coded,oracle,netSaved,wait,deadlineMissIfForced,arPlain,arOracle});
 }
 // XOR exact round trip with unequal lengths, finite byte arrays, synthetic RNG.
 const checks=[];for(const seed of seeds){st=seed;let errors=0;for(let i=0;i<100;i++){let n=1+Math.floor(u()*2048),m=1+Math.floor(u()*2048),A=Array.from({length:n},()=>Math.floor(u()*256)),B=Array.from({length:m},()=>Math.floor(u()*256));let C=Array.from({length:Math.max(n,m)},(_,j)=>(A[j]??0)^(B[j]??0));for(let j=0;j<n;j++)errors+=Number((C[j]^(B[j]??0))!==A[j]);for(let j=0;j<m;j++)errors+=Number((C[j]^(A[j]??0))!==B[j]);}checks.push({seed,checks:100,byteErrors:errors})}
 return {evidence:"V8 actual byte-count arithmetic, source log replay, XOR checks. Artificial relay topology, message sizing, readiness, delay; no GPU/network/HFL/wall time.",source:data.source,n:records.length,recordsSummary:{Dmean:sum(records.map(x=>x.D))/records.length,ARmean:sum(records.map(x=>x.AR))/records.length,Dmin:Math.min(...records.map(x=>x.D)),Dmax:Math.max(...records.map(x=>x.D))},seeds,sizes,slacks,scales,header,rate,rows,checks};
}
console.log(JSON.stringify(run(data),null,2));
