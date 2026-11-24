// HIERASPEC/R08: formal supervised gradient control variates on real-text count models.
// node probe.js wallpaper.txt alice.txt
function run(wall,alice){
 const V=27,D=V*V,seeds=[11,29,73],smooths=[1,20,100],N=8192;
 const clean=t=>{let a=t.indexOf("*** START OF THIS PROJECT GUTENBERG"),b=t.indexOf("*** END OF THIS PROJECT GUTENBERG");if(a<0||b<0)throw Error("markers");return t.slice(t.indexOf("\n",a),b).toLowerCase().replace(/[^a-z]+/g," ").trim()};
 let w=clean(wall),a=clean(alice),mid=Math.floor(a.length/2);const id=c=>c===" "?26:c.charCodeAt(0)-97;
 const regions=[w,a.slice(0,mid),a.slice(mid)].map(x=>Array.from(x,id)),train=regions.map(x=>x.slice(0,Math.floor(.8*x.length))),test=regions.map(x=>x.slice(Math.floor(.8*x.length)));
 const sum=x=>x.reduce((s,v)=>s+v,0),dot=(x,y)=>sum(x.map((v,i)=>v*y[i])),norm=x=>{let z=sum(x);return x.map(v=>v/z)},zero=n=>Array(n).fill(0);
 function count(xs,k){let c=zero(V**(k+1));for(let x of xs)for(let t=k;t<x.length;t++){let j=0;for(let s=t-k;s<=t;s++)j=j*V+x[s];c[j]++}return c}
 let gb=count(train,1),gt=count(train,2),loc=train.map(x=>count([x],1)),g=Array.from({length:V},(_,i)=>norm(gb.slice(i*V,(i+1)*V).map(v=>v+.5))),p=Array.from({length:D},(_,i)=>norm(gt.slice(i*V,(i+1)*V).map((v,j)=>v+5*g[i%V][j])));
 let rows=[],maxBias=0,maxVarError=0;let st=0;const u=()=>{st=(Math.imul(st,1664525)+1013904223)>>>0;return(st+.5)/4294967296};
 const draw=x=>{let r=u()*sum(x);for(let j=0;j<x.length;j++){r-=x[j];if(r<0)return j}return x.length-1};
 for(let e=0;e<3;e++)for(let smooth of smooths){
 let q=Array.from({length:V},(_,i)=>norm(loc[e].slice(i*V,(i+1)*V).map((v,j)=>v+smooth*g[i][j]))),freq=count([test[e]],1); // use actual trigram context frequencies below
 let f=zero(D);for(let t=2;t<test[e].length;t++)f[test[e][t-2]*V+test[e][t-1]]++;
 let n=sum(f),m={alpha:0,plain:0,independentCV:0,coupled:0,optimal:0,stratified:0,rb:0,correctionRows:0,beta:0,denseQRows:0,labelReuseConditionalBias2:0};let cache=[];
 for(let h=0;h<D;h++)if(f[h]){
  const P=p[h],Q=q[h%V],A=sum(P.map((v,j)=>Math.min(v,Q[j]))),d=1-A,pp=dot(P,P),qq=dot(Q,Q),pq=dot(P,Q),diff2=pp+qq-2*pq,cv=1-pp,C=A-pq,B=1-qq,beta=C/B;
  const pos=P.map((v,j)=>Math.max(0,v-Q[j])),neg=Q.map((v,j)=>Math.max(0,v-P[j])),R=d>1e-12?norm(pos):P;
  let rb=0,second=0,mean=zero(V);
  for(let x=0;x<V;x++){let accept=Math.min(1,P[x]/Q[x]),c=R.map((v,j)=>(1-accept)*v+(j===x?accept:0)),v=c.map((v,j)=>Q[j]+v-(j===x?1:0));rb+=Q[x]*sum(v.map((v,j)=>(v-P[j])**2));for(let j=0;j<V;j++){mean[j]+=Q[x]*v[j];second+=Q[x]*c[j]*(j===x?0:2)}}
  maxBias=Math.max(maxBias,...mean.map((v,j)=>Math.abs(v-P[j])));maxVarError=Math.max(maxVarError,Math.abs(second-2*d));
  const z={alpha:A,plain:cv,independentCV:cv+B,coupled:2*d-diff2,optimal:cv-C*C/B,stratified:2*d*d-diff2,rb,correctionRows:2*d,beta,denseQRows:Q.filter(x=>x>0).length,labelReuseConditionalBias2:cv};
  for(const k in m)m[k]+=f[h]/n*z[k];cache[h]={P,Q,A,d,R,pos,neg,beta};
 }
 const mc=[];for(let seed of seeds){st=seed;let err={plain:0,coupled:0,optimal:0,stratified:0,rb:0},accepted=0,naiveBiasConditional=0;for(let i=0;i<N;i++){
 let t=2+Math.floor(u()*n),h=test[e][t-2]*V+test[e][t-1],z=cache[h],{P,Q,A,d,R,pos,neg,beta}=z,x=draw(Q),a=Math.min(1,P[x]/Q[x]),y=u()<a?x:draw(R);accepted+=+(x===y);let j=draw(pos),k=draw(neg);
 for(let v=0;v<V;v++){let Y=+(v===y),X=+(v===x),base=Q[v]-P[v];err.plain+=(Y-P[v])**2;err.coupled+=(base+Y-X)**2;err.optimal+=(Y-P[v]-beta*(X-Q[v]))**2;err.stratified+=(base+d*(+(v===j)-+(v===k)))**2;err.rb+=(base+(1-a)*R[v]+a*X-X)**2}}
 for(let k in err)err[k]/=N;mc.push({seed,N,accepted:accepted/N,...err})}
 rows.push({region:e,smooth,n,...m,mc});
 }
 // Reward-weighted own-label counterexample p=q=(.5,.5), reward R(Y)=Y, scalar Bernoulli logit.
 return {env:"V8 CPU actual public corpus count fitting, exact conditional moments and MC; not neural pretraining/HFL/GPU",V,train:train.map(x=>x.length),test:test.map(x=>x.length),N,seeds,smooths,rows,maxBias,maxVarError,policyCounterexample:{trueGradient:.25,reusedOwnSampleGradient:0,reusedIdenticalDraftGradient:0}};
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("node:fs");console.log(JSON.stringify(run(fs.readFileSync(process.argv[2],"utf8"),fs.readFileSync(process.argv[3],"utf8")),null,2))}
