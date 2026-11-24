function run(input){
const sum=a=>a.reduce((s,x)=>s+x,0),avg=a=>sum(a)/a.length,zero=n=>Array(n).fill(0),key=x=>x.join(",");
let rows=[],maxAudit=0,maxMonotone=0,edgeCount=0,terminalPairs=0,checks=0,drawChecks=0;
function rng(seed){let s=seed;return ()=>((s=Math.imul(s,1664525)+1013904223>>>0)+.5)/4294967296}
function draw(a,u){u*=sum(a);for(let i=0;i<a.length;i++){u-=a[i];if(u<0)return i}return a.findLastIndex(v=>v>0)}
function states(K,N){let out=[];function rec(a,n){if(a.length===K-1){out.push(a.concat(n));return}for(let i=0;i<=n;i++)rec(a.concat(i),n-i)}rec([],N);return out}
function masses(xs,p,N){let f=[0];for(let i=1;i<=N;i++)f.push(f[i-1]+Math.log(i));return xs.map(x=>Math.exp(f[N]-sum(x.map(n=>f[n]))+sum(x.map((n,i)=>n*Math.log(p[i])))))}
function monotone(p,q,N,xs,P,Q){let K=p.length,idx=new Map(xs.map((x,i)=>[key(x),i])),def=p.map((v,i)=>v>q[i]),res=p.map((v,i)=>Math.max(0,v-q[i])),delta=sum(res),T=q.map((v,i)=>q.map((_,j)=>i===j?Math.min(1,p[i]/v):delta?Math.max(0,1-p[i]/v)*res[j]/delta:0)),edges=xs.map(()=>[]),incoming=zero(xs.length),outgoing=zero(xs.length),diag=zero(xs.length);
for(let i=0;i<xs.length;i++){let dist=new Map([[key(zero(K)),{x:zero(K),w:1}]]);for(let j=0;j<K;j++)for(let n=0;n<xs[i][j];n++){let d=new Map();for(let a of dist.values())for(let k=0;k<K;k++)if(T[j][k]>0){let x=a.x.slice();x[k]++;let h=key(x),v=a.w*T[j][k];if(!d.has(h))d.set(h,{x,w:0});d.get(h).w+=v}dist=d}
for(let a of dist.values()){let j=idx.get(key(a.x)),v=a.w*Q[i];if(i===j)diag[i]+=v;else{edges[i].push([j,v]);incoming[j]+=v;outgoing[i]+=v;edgeCount++}}}
let order=xs.map((x,i)=>[i,sum(x.map((v,j)=>def[j]?v:0))]).sort((a,b)=>a[1]-b[1]).map(a=>a[0]),bags=xs.map(()=>new Map()),joint=xs.map((_,i)=>new Map([[i,Math.min(P[i],Q[i])]]));
function inc(m,k,v){m.set(k,(m.get(k)||0)+v)}
for(let i of order){let b=bags[i],s=outgoing[i]-incoming[i];if(s>0)inc(b,i,s);let total=sum([...b.values()]);if(!total)continue;let sink=Math.max(0,-s),den=sink+outgoing[i];if(!den)continue;for(let [origin,v] of b){if(sink>0){inc(joint[origin],i,v*sink/den);terminalPairs++}for(let [j,w] of edges[i])inc(bags[j],origin,v*w/den)}}
let col=zero(xs.length),eq=0,missing=0,moves=0,sourceErr=0,mono=0;for(let i=0;i<xs.length;i++){sourceErr=Math.max(sourceErr,Math.abs(sum([...joint[i].values()])-Q[i]));for(let [j,w] of joint[i]){col[j]+=w;if(i===j)eq+=w;missing+=w*sum(xs[j].map((n,k)=>+(n>0&&xs[i][k]===0)));moves+=w*sum(xs[j].map((n,k)=>Math.max(0,n-xs[i][k])));if(xs[j].some((v,k)=>def[k]?v<xs[i][k]:v>xs[i][k]))mono+=w}}
let lb=sum(p.map((v,i)=>Math.max(0,(1-q[i])**N-(1-v)**N))),overlap=sum(P.map((v,i)=>Math.min(v,Q[i]))),err=Math.max(sourceErr,...col.map((v,i)=>Math.abs(v-P[i])),Math.abs(eq-overlap),Math.abs(missing-lb),Math.abs(moves-N*delta));maxAudit=Math.max(maxAudit,err);maxMonotone=Math.max(maxMonotone,mono);checks++;
let rr=rng(101+Math.round(1e6*p[0])+N),mcEq=0,mcMiss=0,mcMove=0;for(let t=0;t<256;t++){let x=zero(K);for(let n=0;n<N;n++)x[draw(q,rr())]++;let i=idx.get(key(x)),es=[...joint[i]],j=es[draw(es.map(x=>x[1]),rr())][0],y=xs[j];mcEq+=+(i===j);mcMiss+=sum(y.map((v,k)=>+(v>0&&x[k]===0)));mcMove+=sum(y.map((v,k)=>Math.max(0,v-x[k])));drawChecks++}
return {eq,missing,moves,mono,err,mcEq:mcEq/256,mcMiss:mcMiss/256,mcMove:mcMove/256,indEq:sum(diag)}}
for(let r of input.rows.filter(x=>x.K===2||(x.K===4&&x.N<=32))){let {p,q,N,K}=r,xs=states(K,N),P=masses(xs,p,N),Q=masses(xs,q,N),delta=sum(p.map((v,i)=>Math.max(0,v-q[i]))),ov=sum(P.map((v,i)=>Math.min(v,Q[i]))),lb=sum(p.map((v,i)=>Math.max(0,(1-q[i])**N-(1-v)**N))),ind=(1-delta)**N;
let exact=(K===4&&N===8)?monotone(p,q,N,xs,P,Q):null;
rows.push({wi:r.wi,seed:r.seed,name:r.name,K,N,states:xs.length,overlap:ov,ind,lb,moves:N*delta,exact})}
let summary=[];for(let K of [2,4])for(let N of [8,32,64]){let a=rows.filter(x=>x.K===K&&x.N===N);if(!a.length)continue;let s={K,N,n:a.length,overlap:avg(a.map(x=>x.overlap)),ind:avg(a.map(x=>x.ind)),lb:avg(a.map(x=>x.lb)),moves:avg(a.map(x=>x.moves))};if(a[0].exact)s.exact=Object.fromEntries(Object.keys(a[0].exact).map(k=>[k,avg(a.map(x=>x.exact[k]))]));summary.push(s)}
return {rows,summary,maxAudit,maxMonotone,edgeCount,terminalPairs,checks,drawChecks}
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("fs");console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(process.argv[2],"utf8"))),null,2))}
