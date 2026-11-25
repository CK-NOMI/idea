function jointProbe(){
const arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mean=a=>sum(a)/a.length,sm=x=>{let max=Math.max(...x),a=x.map(v=>Math.exp(v-max)),z=sum(a);return a.map(v=>v/z)},kl=(p,q)=>sum(p.map((v,i)=>v*Math.log(v/q[i]))),ov=(p,q)=>sum(p.map((v,i)=>Math.min(v,q[i]))),norm=x=>Math.sqrt(dot(x,x)/x.length+1e-5);
function rng(seed){let x=seed>>>0;return ()=>((x=Math.imul(1664525,x)+1013904223>>>0)+.5)/4294967296}
function run(seed,G,M,mode,scale,d=64,V=256){
let r=rng(seed+G*111+M*971),noise=()=>2*r()-1,shuffle=a=>{a=a.slice();for(let i=a.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a};
let W=arr(V,()=>arr(d,()=>noise()/Math.sqrt(d))),gamma=arr(d,()=>.8+.4*r()),U=W.map(w=>w.map((v,i)=>v*gamma[i])),y=arr(d,()=>noise()),group=arr(G,()=>arr(d,()=>noise()*scale/Math.sqrt(G))),ref=arr(G,i=>i<G/2?1:0),masks=[ref],seen=new Set([ref.join("")]);for(let j=1;j<M;j++){let a;for(let t=0;t<200;t++){a=mode==="swap"?ref.slice():arr(G,()=>0);if(mode==="swap"){let on=Math.floor(r()*G/2),off=G/2+Math.floor(r()*G/2);a[on]=0;a[off]=1;}else for(let i of shuffle(arr(G,i=>i)).slice(0,G/2))a[i]=1;if(!seen.has(a.join("")))break;}if(seen.has(a.join("")))break;seen.add(a.join(""));masks.push(a);}
let h0=arr(d,i=>y[i]+sum(group.map((g,j)=>ref[j]*g[i]))),p0=sm(U.map(w=>dot(w,h0)/norm(h0))),q=sm(U.map(w=>dot(w,y)/norm(y))),P=masks.slice(1).map(a=>{let h=arr(d,i=>h0[i]+sum(a.map((v,g)=>(v-ref[g])*group[g][i])));return sm(U.map(w=>dot(w,h)/norm(h)));}),n=P.length,truth=[{k:0,a:ov(p0,q)},...P.map(p=>({k:kl(p0,p),a:ov(p,q)}))],eta=.005,ids=arr(n,i=>i),best=truth.reduce((b,x,i)=>x.k<=eta&&x.a>truth[b].a?i:b,0),tol=1e-10,random=shuffle(arr(V,i=>i)),metrics={},viol=0;
function ber(t,s){return t*Math.log(t/s)+(1-t)*Math.log((1-t)/(1-s))}
function mass(t,budget,target){
if(ber(t,target)<=budget)return target;
let a=Math.min(t,target),b=Math.max(t,target);for(let i=0;i<48;i++){let m=(a+b)/2;if(target<t){if(ber(t,m)>budget)a=m;else b=m;}else{if(ber(t,m)>budget)b=m;else a=m;}}
return target<t?a:b; // outward root bracket makes overlap bound conservative
}
for(let orderName of ["p0","q","uniform","joint"])for(let rule of ["quality","separate","combined"]){
let order=orderName==="uniform"?random:arr(V,i=>i).sort((a,b)=>(orderName==="p0"?p0[b]-p0[a]:orderName==="q"?q[b]-q[a]:p0[b]+q[b]-p0[a]-q[a])),J=ids.slice(),prev=0,rows=0,qp=0,ap=0,selected=0,score=truth[0].a,minGap=1,history=[];
for(let end of [16,32,64,128,256]){if(!J.length)break;rows+=J.length*(end-prev);let I=order.slice(0,end),Pr=I.map(i=>p0[i]),Q=I.map(i=>q[i]),t=sum(Pr),u=sum(Q),Pc=Pr.map(x=>x/t),Qc=Q.map(x=>x/u),next=[];
for(let j of J){let raw=I.map(i=>P[j][i]),s=sum(raw),p=raw.map(v=>v/s),L=t*kl(Pc,p),sep=1-u*(1-ov(p,Qc));if(end===V){if(truth[j+1].k<=eta&&truth[j+1].a>score){selected=j+1;score=truth[j+1].a;}continue;}
if(L>truth[j+1].k+tol||sep<truth[j+1].a-tol)viol++;
if(L>eta+tol){qp++;continue;}
let sb=mass(t,Math.max(0,eta-L)+tol,u),comb=sum(p.map((v,i)=>Math.min(sb*v,Q[i])))+Math.min(1-sb,1-u);if(truth[j+1].k<=eta&&comb<truth[j+1].a-tol)viol++;
let ub=rule==="combined"?Math.min(sep,comb):sep;minGap=Math.min(minGap,comb-sep);
if(rule!=="quality"&&ub<score-tol)ap++;else next.push(j);}
history.push({end,before:J.length,after:next.length});J=next;prev=end;}
metrics[orderName+"_"+rule]={rows,qp,ap,selected,match:selected===best,minGap,history};}
return {seed,G,M:masks.length,mode,scale,n,metrics,viol};}
let rows=[];for(let seed of [11,29,73])for(let G of [8,16,32])for(let M of [8,32,64])for(let mode of ["swap","random"])for(let scale of [.4,1.2])rows.push(run(seed,G,M,mode,scale));let summary=Object.fromEntries(Object.keys(rows[0].metrics).map(m=>{let a=rows.map(x=>x.metrics[m]);return [m,{meanRows:mean(a.map((x,i)=>x.rows/(rows[i].n*256))),qualityPrunes:sum(a.map(x=>x.qp)),acceptPrunes:sum(a.map(x=>x.ap)),matches:sum(a.map(x=>+x.match)),largestTightening:Math.min(...a.map(x=>x.minGap))}]}));return {summary,rows,viol:sum(rows.map(x=>x.viol))};}
console.log(JSON.stringify(jointProbe(),null,2));
