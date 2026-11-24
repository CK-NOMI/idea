function audit(){
let seed=9181;const u=()=>((seed=Math.imul(seed,1664525)+1013904223>>>0)+.5)/4294967296,sum=a=>a.reduce((s,v)=>s+v,0);
let maxMeanErr=0,maxBoundViolation=0,wins=0,ratios=[];
for(let t=0;t<120;t++){
let n=4,K=3,p=[],q=[],bc=[],tilt=[],r=[];
for(let i=0;i<n;i++){let a=Array.from({length:K},()=>.02+u()),b=Array.from({length:K},()=>.02+u()),sa=sum(a),sb=sum(b);a=a.map(v=>v/sa);b=b.map(v=>v/sb);p.push(a);q.push(b);let c=a.map((v,j)=>Math.sqrt(v*b[j])),s=sum(c);bc.push(s);tilt.push(c.map(v=>v/s));r.push(a.map((v,j)=>Math.log(v/b[j])))}
let B=bc.reduce((a,b)=>a*b,1),mQ=0,mT=0,sQ=0,sT=0;for(let a=0;a<K;a++)for(let b=0;b<K;b++)for(let c=0;c<K;c++){let S=r[1][a]+r[2][b]+r[3][c],wQ=q[1][a]*q[2][b]*q[3][c],wT=tilt[1][a]*tilt[2][b]*tilt[3][c],gQ=0,gT=0;
for(let j=0;j<K;j++)if(S+r[0][j]>0){let score=+(j===0)-q[0][0];gQ-=q[0][j]*score;gT-=B*tilt[0][j]*Math.exp(-(S+r[0][j])/2)*score}
mQ+=wQ*gQ;mT+=wT*gT;sQ+=wQ*gQ*gQ;sT+=wT*gT*gT}
maxMeanErr=Math.max(maxMeanErr,Math.abs(mQ-mT));maxBoundViolation=Math.max(maxBoundViolation,Math.abs(mT)-B*Math.max(q[0][0],1-q[0][0]));let vQ=sQ-mQ*mQ,vT=sT-mT*mT;if(vQ>1e-15){ratios.push(vT/vQ);if(vT<vQ)wins++}}
return {cases:120,maxMeanErr,maxBoundViolation,wins,valid:ratios.length,meanVarianceRatio:sum(ratios)/ratios.length,maxVarianceRatio:Math.max(...ratios)}}
if(require.main===module)console.log(JSON.stringify(audit(),null,2));
