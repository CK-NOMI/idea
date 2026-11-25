function run(src){
const data=src.rows.filter(x=>x.K===27&&x.N===8), V=27,sum=a=>a.reduce((s,v)=>s+v,0),mean=a=>sum(a)/a.length,arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),kl=(p,q)=>sum(p.map((v,i)=>v*Math.log(v/q[i]))),tv=(p,q)=>sum(p.map((v,i)=>Math.abs(v-q[i])))/2, soft=z=>{let m=Math.max(...z),p=z.map(x=>Math.exp(x-m)),s=sum(p);return p.map(x=>x/s)},seeds=[...new Set(data.map(x=>x.seed))],ws=[...new Set(data.map(x=>x.wi))],ks=[4,8,12,27],ridge=[0,.01,.1],rows=[];
const reg=x=>(x.name.charCodeAt(0)-97)&1;
function mask(bank,k,kind){let val=arr(V,i=>mean(bank.map(({p,q})=>kind==="abs"?Math.abs(p[i]-q[i]):p[i]*Math.log(p[i]/q[i])-p[i]+q[i])));return arr(V,i=>i).sort((a,b)=>val[b]-val[a]||a-b).slice(0,k)}
function features(q){return [1,...q.map(x=>Math.log(x)*.25)]}
function predict(q,S,W){let f=features(q),h=S.map((i,j)=>Math.log(q[i])+sum(W[j].map((w,k)=>w*f[k]))),s=soft(h),m=sum(S.map(i=>q[i])),p=q.slice();S.forEach((i,j)=>p[i]=m*s[j]);return {p,s,f,m}}
function train(bank,S,lam){let W=arr(S.length,()=>arr(V+1,()=>0));for(let t=0;t<160;t++){let g=arr(S.length,()=>arr(V+1,()=>0));for(let {p,q} of bank){let a=predict(q,S,W),rS=sum(S.map(i=>p[i]));for(let j=0;j<S.length;j++)for(let k=0;k<V+1;k++)g[j][k]+=(rS*a.s[j]-p[S[j]])*a.f[k]/bank.length}for(let j=0;j<S.length;j++)for(let k=0;k<V+1;k++)W[j][k]-=.35*(g[j][k]+lam*W[j][k]);}return W}
function evaluate(bank,S,W){return mean(bank.map(x=>kl(x.p,predict(x.q,S,W).p)))}
let fits=0,gradBatches=0,maxSum=0,maxKernel=0,maxOutside=0,positive=0;
for(let wi of ws)for(let seed of seeds)for(let k of ks){let trainBank=data.filter(x=>x.wi===wi&&x.seed!==seed),test=data.filter(x=>x.wi===wi&&x.seed===seed);for(let arm of ["global","regional","absolute","dense"]){if(arm==="dense"&&k!==27)continue;for(let e=0;e<2;e++){let tr=trainBank.filter(x=>reg(x)===e),te=test.filter(x=>reg(x)===e);if(!tr.length||!te.length)continue;
let source=arm==="regional"?tr:trainBank,S=mask(source,arm==="dense"?27:k,arm==="absolute"?"abs":"div");let pooled=arm==="regional"?tr:trainBank;
// Fit all fixed ridge candidates and use leave-one-out context split within training bank, separate held-out seed.
let cal=pooled.filter((_,i)=>i%4===0),fit=pooled.filter((_,i)=>i%4!==0);if(!fit.length||!cal.length)continue;
let cand=ridge.map(l=>{let W=train(fit,S,l);fits++;gradBatches+=160;return {l,D:evaluate(cal,S,W)}}).sort((a,b)=>a.D-b.D)[0];let W=train(pooled,S,cand.l);fits++;gradBatches+=160;
for(let x of te){let a=predict(x.q,S,W),p=a.p,q=x.q,r=x.p,T=tv(p,q),delta=p.map((v,i)=>Math.max(0,v-q[i])),mass=sum(delta);maxSum=Math.max(maxSum,Math.abs(sum(p)-1));for(let i=0;i<V;i++){if(!S.includes(i))maxOutside=Math.max(maxOutside,Math.abs(p[i]-q[i]));let rec=Math.min(p[i],q[i])+(T>1e-14?T*delta[i]/mass:0);maxKernel=Math.max(maxKernel,Math.abs(rec-p[i]));}positive+=delta.filter(v=>v>1e-12).length;
let oracle=q.slice(),rs=sum(S.map(i=>r[i]));for(let i of S)oracle[i]=a.m*r[i]/rs;
rows.push({wi,seed,e,k,arm,name:x.name,lambda:cand.l,D:kl(r,p),T,base:kl(r,q),baseTV:tv(r,q),fixedMaskOracle:kl(r,oracle),mask:S,positive:delta.filter(v=>v>1e-12).length});}}}}
const summary=[];for(let k of ks)for(let arm of ["global","regional","absolute","dense"]){let a=rows.filter(x=>x.k===k&&x.arm===arm);if(a.length)summary.push({k,arm,n:a.length,D:mean(a.map(x=>x.D)),T:mean(a.map(x=>x.T)),base:mean(a.map(x=>x.base)),fixedMaskOracle:mean(a.map(x=>x.fixedMaskOracle)),within005:a.filter(x=>x.D<=.005).length,positive:mean(a.map(x=>x.positive))})}
return {spec:{data:"R69 K27 N8 48 historical contexts; 2 same-size checkpoints, held-out hash seed, region artificial first letter parity, log-q feature linear residual; fixed q not newly trained, teacher r not ground truth. Static support selected on train bank; ridge chosen within train; dense head same input but higher parameter count; no actual multi-token sampling, GPU, physical HFL, or timing. Reused names across seed banks possible, not independently pretraining-held-out."},fits,gradBatches,maxSum,maxKernel,maxOutside,positive,summary,rows}
}
console.log(JSON.stringify(run(JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'))),null,2));
