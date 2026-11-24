// node experiments/team-hca/R07/probe.js
function audit(src){
 const sum=a=>a.reduce((s,v)=>s+v,0);
 const rows=src.rows.map(r=>{let D=sum(r.draft),A=sum(r.accepted);return {...r,D,A,consistency:r.N===r.draft.length&&r.N===r.accepted.length,delta:D-A,
 // Payload size proxy: MsgPack default 9 bytes per float64, each verified vector 32256 floats.
 verifiedVectorBytes:9*32256*D,sentCounterBytes:9*32256*r.sent,verifPerToken:r.N/r.L}});
 const rtts=[.015,.04,.065,.1,.2],methods=[...new Set(rows.map(r=>r.method))],group=[];
 for(const method of methods){let a=rows.filter(r=>r.method===method),T=sum(a.map(r=>r.T)),L=sum(a.map(r=>r.L)),N=sum(a.map(r=>r.N)),D=sum(a.map(r=>r.D)),A=sum(a.map(r=>r.A)),sent=sum(a.map(r=>r.sent)),generated=sum(a.map(r=>r.generated));group.push({method,n:a.length,T,L,N,D,A,sent,generated,weightedTPOT:T/L,verifiedBytes:D*9*32256,sentBytes:sent*9*32256,serializedQuota:D*9*32256/2500000,localVsMandatoryRemote:rtts.map(R=>({R,extra:N*R,relativeDrop:N*R/(T+N*R),remoteT:T+N*R})),breakEvenRTT15:0.15*T/(0.85*N),counterexampleEqualDelegateDrop:0})}
 return {rows,group,assumptions:["Extra RTT before each verification; fully exposed, serial, no overlap; not measured in logs.","Pre-authorized delegation and HFL run identical state transition, so no extra RTT in either; not a measured timing gain.","Probability bytes proxy only; sender counters not packet capture; payload repeats/cancel/metadata and actual quotas not observed."]};
}
console.log(JSON.stringify(audit(require('./source_metrics.json')),null,2));
