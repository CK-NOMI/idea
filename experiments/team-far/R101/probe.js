function probe(){
const V=5,M=8,N=256,arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),sum=a=>a.reduce((s,v)=>s+v,0),mean=a=>sum(a)/a.length,sm=x=>{let a=x.map(v=>Math.exp(v-Math.max(...x))),z=sum(a);return a.map(v=>v/z)},argmax=(a,inds=arr(a.length,i=>i))=>inds.reduce((i,j)=>a[j]>a[i]?j:i),kl=(p,q)=>sum(p.map((x,i)=>x*Math.log(x/q[i])));function rng(seed){let x=seed>>>0;return ()=>((x=Math.imul(1664525,x)+1013904223>>>0)+.5)/4294967296}
function sample(p,u){let s=0;for(let i=0;i<p.length;i++){s+=p[i];if(u<s)return i}return p.length-1}
let rows=[],viol=0,maxW=0;
for(let seed of [11,29,73,101,131,197])for(let spread of [.2,1,2])for(let c=0;c<4;c++){
let r=rng(seed*1000+Math.round(spread*100)*10+c),noise=()=>2*r()-1,base=arr(V,()=>arr(V,()=>2*noise())),edge=arr(V,()=>arr(V,()=>noise())),p0=base.map((a,s)=>sm(a.map((v,i)=>v+.25*edge[s][i])));
let ps=arr(M,j=>base.map((a,s)=>sm(a.map((v,i)=>v+.25*edge[s][i]+.20*noise()*(j/7))))),qs=arr(M,j=>base.map((a,s)=>sm(a.map((v,i)=>v+spread*(.4*edge[s][i]+noise())*(.4+j/7)))));
let quality=ps.map(p=>mean(p.map((a,s)=>kl(p0[s],a)))),valid=arr(M,i=>i).filter(i=>quality[i]<=.008);if(!valid.length)throw Error("no valid");let A=ps.map((p,j)=>p.map((a,s)=>a.map((v,i)=>Math.min(v,qs[j][s][i]))));
for(let K of [2,4,8]){
const truth=A.map(a=>{let f=arr(V,()=>0);for(let k=0;k<K;k++)f=a.map(row=>sum(row.map((v,i)=>v*(1+f[i]))));return mean(f)});
let e={direct:arr(M,()=>0),fixed:arr(M,()=>0),step:arr(M,()=>0),mix:arr(M,()=>0)};
let count={direct:0,fixed:0,step:0,mix:0};
for(let n=0;n<N;n++){
const j=n%M;let init=Math.floor(r()*V),us=arr(K,()=>r());
for(let arm of ["direct","fixed","step","mix"]){
let state=init,aa=arr(M,()=>1),qq=arr(M,()=>1),b=1;
for(let k=0;k<K;k++){
let beh=arm==="fixed"?qs[0][state]:arm==="step"?arr(V,i=>mean(qs.map(q=>q[state][i]))):qs[j][state];let x=sample(beh,us[k]);b*=beh[x];for(let m=0;m<M;m++){aa[m]*=A[m][state][x];qq[m]*=qs[m][state][x];}
let denom=arm==="mix"?mean(qq):b;
if(arm==="direct"){e.direct[j]+=aa[j]/denom/(N/M);count.direct++}else{for(let m=0;m<M;m++){let w=aa[m]/denom;e[arm][m]+=w/N;if(arm==="mix"){maxW=Math.max(maxW,w);if(w>M+1e-10)viol++}}count[arm]++}
state=x;
}
}}
const best=argmax(truth,valid);let metrics={};for(let arm of Object.keys(e)){let pick=argmax(e[arm],valid);metrics[arm]={mse:mean(e[arm].map((v,j)=>(v-truth[j])**2)),regret:truth[best]-truth[pick],correct:pick===best?1:0,pick,estimate:e[arm]}}
rows.push({seed,spread,c,K,truth,quality,valid,best,metrics,count});
}}
let summary=[];for(let K of [2,4,8])for(let spread of [.2,1,2]){let a=rows.filter(x=>x.K===K&&x.spread===spread);summary.push({K,spread,n:a.length,...Object.fromEntries(["direct","fixed","step","mix"].map(m=>[m,{rmse:Math.sqrt(mean(a.map(x=>x.metrics[m].mse))),regret:mean(a.map(x=>x.metrics[m].regret)),selection:mean(a.map(x=>x.metrics[m].correct))}]))})}
return {spec:{V,M,N,seeds:[11,29,73,101,131,197],clients:4,spreads:[.2,1,2],K:[2,4,8],sameUniforms:true,qualityGuard:.008},viol,maxW,summary,rows};
}
console.log(JSON.stringify(probe(),null,2));
