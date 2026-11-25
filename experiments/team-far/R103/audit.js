function audit(){
const arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mean=a=>sum(a)/a.length,sm=x=>{let max=Math.max(...x),a=x.map(v=>Math.exp(v-max)),z=sum(a);return a.map(v=>v/z)},kl=(p,q)=>sum(p.map((v,i)=>v*Math.log(v/q[i]))),ov=(p,q)=>sum(p.map((v,i)=>Math.min(v,q[i]))),norm=x=>Math.sqrt(dot(x,x)/x.length+1e-5);
function rng(seed){let x=seed>>>0;return ()=>((x=Math.imul(1664525,x)+1013904223>>>0)+.5)/4294967296}
function basis(rows){let B=[],P=[],C=[];for(let row of rows){let x=row.slice(),c=Array(B.length).fill(0);for(let i=0;i<B.length;i++){let v=x[P[i]];c[i]=v;for(let j=0;j<x.length;j++)x[j]-=v*B[i][j];}let p=x.findIndex(v=>Math.abs(v)>1e-9);if(p>=0){let v=x[p];P.push(p);B.push(x.map(z=>z/v));c.push(v);}C.push(c);}return {B,C:C.map(c=>arr(B.length,i=>c[i]||0))}}
function run(seed,G,M,mode,scale,d=64,V=256){
let r=rng(seed+G*111+M*971),noise=()=>2*r()-1,shuffle=a=>{a=a.slice();for(let i=a.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a};
let W=arr(V,()=>arr(d,()=>noise()/Math.sqrt(d))),gamma=arr(d,()=>.8+.4*r()),U=W.map(w=>w.map((v,i)=>v*gamma[i])),y=arr(d,()=>noise()),group=arr(G,()=>arr(d,()=>noise()*scale/Math.sqrt(G))),ref=arr(G,i=>i<G/2?1:0),masks=[ref],seen=new Set([ref.join("")]);for(let j=1;j<M;j++){let a;for(let t=0;t<200;t++){a=mode==="swap"?ref.slice():arr(G,()=>0);if(mode==="swap"){let on=Math.floor(r()*G/2),off=G/2+Math.floor(r()*G/2);a[on]=0;a[off]=1;}else for(let i of shuffle(arr(G,i=>i)).slice(0,G/2))a[i]=1;if(!seen.has(a.join("")))break;}if(seen.has(a.join("")))break;seen.add(a.join(""));masks.push(a);}
let h0=arr(d,i=>y[i]+sum(group.map((g,j)=>ref[j]*g[i]))),base=U.map(w=>dot(w,h0)),p0=sm(base.map(x=>x/norm(h0))),q=sm(U.map(w=>dot(w,y)/norm(y))),diff=masks.slice(1).map(a=>a.map((v,i)=>v-ref[i])),n=diff.length,H=diff.map(a=>arr(d,i=>h0[i]+sum(a.map((v,g)=>v*group[g][i])))),nr=H.map(norm),truth=[{k:0,a:ov(p0,q)},...H.map((h,j)=>{let p=sm(U.map(w=>dot(w,h)/nr[j]));return {k:kl(p0,p),a:ov(p,q)}})],eta=.005,ids=arr(n,i=>i),best=truth.reduce((b,x,i)=>x.k<=eta&&x.a>truth[b].a?i:b,0);
let tolerance=1e-10,maxError=0,violations=0;
function block(J,I){
let a=J.map(j=>diff[j]),B=basis(a),nz=x=>x.reduce((s,row)=>s+row.filter(v=>Math.abs(v)>1e-9).length,0),active=arr(G,g=>g).filter(g=>a.some(row=>row[g])),parent=arr(G,g=>g),root=g=>parent[g]===g?g:root(parent[g]);
for(let row of a){let s=active.filter(g=>row[g]);for(let g of s.slice(1))parent[root(g)]=root(s[0]);}
let comps=[...new Set(active.map(root))].map(k=>active.filter(g=>root(g)===k)),anchors=comps.map(c=>c.reduce((b,g)=>a.filter(row=>row[g]).length>a.filter(row=>row[b]).length?g:b,c[0])),others=active.filter(g=>!anchors.includes(g)),anchor=g=>anchors[comps.findIndex(c=>c.includes(g))];
let costs=[J.length*I.length*d,active.length*I.length*d+I.length*nz(a),B.B.length*I.length*d+I.length*nz(B.C)+d*nz(B.B),others.length*I.length*d+I.length*sum(a.map(row=>others.filter(g=>row[g]).length))+d*others.length];let method=costs.indexOf(Math.min(...costs)),vec,coef;
if(method===1){vec=active.map(g=>group[g]);coef=a.map(row=>active.map(g=>row[g]));}if(method===2){vec=B.B.map(row=>arr(d,i=>sum(row.map((v,g)=>v*group[g][i]))));coef=B.C;}if(method===3){vec=others.map(g=>group[g].map((v,i)=>v-group[anchor(g)][i]));coef=a.map(row=>others.map(g=>row[g]));}
let output;
if(method===0)output=J.map(j=>I.map(i=>dot(U[i],H[j])/nr[j]));else{let P=I.map(i=>vec.map(v=>dot(U[i],v)));output=J.map((j,k)=>I.map((i,l)=>(base[i]+dot(coef[k],P[l]))/nr[j]));}
return {output,cost:costs[method],method};
}
let baseline=block(ids,arr(V,i=>i)).cost,random=shuffle(arr(V,i=>i)),metrics={};
for(let orderName of ["p0","q","uniform","joint"])for(let rule of ["quality","joint"]){
let order=orderName==="uniform"?random:arr(V,i=>i).sort((a,b)=>(orderName==="p0"?p0[b]-p0[a]:orderName==="q"?q[b]-q[a]:p0[b]+q[b]-p0[a]-q[a])),J=ids.slice(),z=arr(n,()=>[]),prev=0,mac=0,rows=0,qualityPrunes=0,acceptPrunes=0,methodCount=[0,0,0,0],history=[],selected=0,score=truth[0].a;
for(let end of [16,32,64,128,256]){if(!J.length)break;let I=order.slice(prev,end),t=block(J,I);mac+=t.cost;methodCount[t.method]++;rows+=J.length*I.length;let P=order.slice(0,end).map(i=>p0[i]),Q=order.slice(0,end).map(i=>q[i]),massP=sum(P),massQ=sum(Q),Pc=P.map(x=>x/massP),Qc=Q.map(x=>x/massQ),next=[];
for(let [k,j] of J.entries()){z[j].push(...t.output[k]);let p=sm(z[j]),L=massP*kl(Pc,p),Ubound=1-massQ*(1-ov(p,Qc));if(L>truth[j+1].k+tolerance||Ubound<truth[j+1].a-tolerance)violations++;if(end===V){maxError=Math.max(maxError,Math.abs(L-truth[j+1].k),Math.abs(Ubound-truth[j+1].a));if(L<=eta&&Ubound>score){selected=j+1;score=Ubound;}}else if(L>eta+tolerance)qualityPrunes++;else if(rule==="joint"&&Ubound<score-tolerance)acceptPrunes++;else next.push(j);}
history.push({end,before:J.length,after:next.length,mac:t.cost,method:t.method});J=next;prev=end;}
metrics[orderName+"_"+rule]={selected,score,mac,ratio:mac/baseline,rows,qualityPrunes,acceptPrunes,methodCount,history,regret:truth[best].a-truth[selected].a,match:selected===best};}
return {seed,G,M:masks.length,mode,scale,n,baseline,best,feasible:truth.filter(x=>x.k<=eta).length,metrics,maxError,violations};
}
let rows=[];for(let seed of [11,29,73])for(let G of [8,16,32])for(let M of [8,32,64])for(let mode of ["swap","random"])for(let scale of [.4,1.2])rows.push(run(seed,G,M,mode,scale));
let summary=Object.fromEntries(Object.keys(rows[0].metrics).map(m=>{let a=rows.map(x=>x.metrics[m]);return [m,{meanRatio:mean(a.map(x=>x.ratio)),weightedRatio:sum(a.map(x=>x.mac))/sum(rows.map(x=>x.baseline)),meanRows:mean(a.map((x,i)=>x.rows/(rows[i].n*256))),qualityPrunes:sum(a.map(x=>x.qualityPrunes)),acceptPrunes:sum(a.map(x=>x.acceptPrunes)),matches:sum(a.map(x=>+x.match)),wins:sum(a.map(x=>+(x.ratio<1))),maxRatio:Math.max(...a.map(x=>x.ratio))}]}));return {summary,rows,maxError:Math.max(...rows.map(x=>x.maxError)),violations:sum(rows.map(x=>x.violations))};
}
console.log(JSON.stringify(audit(),null,2));
