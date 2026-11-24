// R05: exact label correction for fixed-feature square-loss SGD.
// Run: node experiments/team-hieracert/R05/probe.js
function probe(){
 const seeds=[11,29,73], dims=[16,64], lengths=[32,128], deltas=[.01,.05,.2], trials=12, eta=.2;
 let state=0; const rand=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296};
 const normal=()=>Math.sqrt(-2*Math.log(rand()))*Math.cos(2*Math.PI*rand());
 const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
 const norm=a=>Math.sqrt(dot(a,a));
 function step(w,x,y){const f=eta*(y-dot(w,x));return w.map((a,i)=>a+f*x[i])}
 function replay(w,X,Y,start=0){for(let t=start;t<X.length;t++)w=step(w,X[t],Y[t]);return w}
 function cache(X,d){
  let M=Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>Number(i===j))), C=Array(X.length);
  for(let t=X.length-1;t>=0;t--){
   const x=X[t],u=M.map(r=>dot(r,x)); C[t]=u.map(v=>eta*v);
   M=M.map((r,i)=>r.map((v,j)=>v-eta*u[i]*x[j]));
  }
  return C;
 }
 let rows=[];
 for(const d of dims)for(const T of lengths)for(const delta of deltas)for(const seed of seeds){
  state=seed;let sums={changes:0,rollbackSteps:0,lateColumns:0,naiveError:0,cacheError:0,recurrenceError:0,rollbackError:0,noMismatch:0},maxError=0;
  for(let trial=0;trial<trials;trial++){
   const X=Array.from({length:T},()=>{let x=Array.from({length:d},normal),n=norm(x);return x.map(v=>v/n)});
   const Y=Array.from({length:T},()=>Number(rand()<.5)), Yhat=Y.map(y=>rand()<delta?1-y:y);
   const w0=Array(d).fill(0), trueW=replay(w0,X,Y), checkpoints=[w0];
   for(let t=0;t<T;t++)checkpoints.push(step(checkpoints[t],X[t],Yhat[t]));
   const spec=checkpoints[T], changed=Y.map((y,t)=>y!==Yhat[t]?t:-1).filter(t=>t>=0), first=changed[0]??T;
   const rolled=replay(checkpoints[first],X,Y,first);
   const C=cache(X,d), corrected=spec.slice(),naive=spec.slice();
   for(const j of changed)for(let i=0;i<d;i++){corrected[i]+=C[j][i]*(Y[j]-Yhat[j]);naive[i]+=eta*X[j][i]*(Y[j]-Yhat[j]);}
   let D=Array(d).fill(0);for(let t=first;t<T;t++)D=step(D,X[t],Y[t]-Yhat[t]);
   const recur=spec.map((v,i)=>v+D[i]), err=v=>norm(v.map((x,i)=>x-trueW[i]));
   let e=err(corrected);maxError=Math.max(maxError,e,err(recur),err(rolled));
   if(maxError>1e-10)throw Error("exact check failed");
   const vals={changes:changed.length,rollbackSteps:T-first,lateColumns:changed.length,naiveError:err(naive),cacheError:e,recurrenceError:err(recur),rollbackError:err(rolled),noMismatch:Number(first===T)};
   for(const k in vals)sums[k]+=vals[k]/trials;
  }
  rows.push({d,T,delta,seed,...sums,maxError});
 }
 const sig=x=>1/(1+Math.exp(-x));
 let spec=-.5-sig(-.5), truth=.5-sig(.5);
 return {environment:"V8 CPU arithmetic; synthetic fixed-feature SGD; no wall-clock/GPU/real HFL or SD",seeds,dims,lengths,deltas,trials,eta,rows,logisticCounterexample:{spec,truth,naive:spec+1,error:Math.abs(spec+1-truth)}};
}
console.log(JSON.stringify(probe(),null,2));
