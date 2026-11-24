function run(){
const seeds=[11,29,73,101,137,173],N=8,L0=3,K=3,sum=a=>a.reduce((s,x)=>s+x,0),avg=a=>sum(a)/a.length,z=()=>[0,0,0],add=(a,b)=>a.map((x,j)=>x+b[j]),sub=(a,b)=>a.map((x,j)=>x-b[j]),mul=(a,b)=>a.map(x=>x*b),dot=(a,b)=>sum(a.map((x,j)=>x*b[j])),sig=x=>1/(1+Math.exp(-x));
let rows=[],leaves=0,rollouts=0,maxMass=0,maxMean=0,maxResidual=0;
for(let seed of seeds)for(let het of [.2,1,2])for(let task of ["count","parity"]){
let state=seed;const rnd=()=>((state=Math.imul(state,1664525)+1013904223>>>0)/4294967296);
let cs=Array.from({length:N},(_,i)=>({w:[(i<4?-1:1)*het*.65+(rnd()-.5)*het,.9+(rnd()-.5)*het,.8+(rnd()-.5)*het],L:[8,8,9,10,8,9,10,12][i],speed:[1,1.2,1.7,2.1,1,1.4,1.9,2.7][i],threshold:i%2?.55:.45}));
function feat(h){return [1,h.length?2*Number(h[h.length-1])-1:0,h.length?2*sum([...h].map(Number))/h.length-1:0]}
function p(h,c,sh=false){let f=feat(h);return sig(c.w[0]+c.w[1]*f[1]+(sh?0:c.w[2]*f[2]))}
function grad(h,c){let v=z();for(let j=0;j<h.length;j++){let a=h.slice(0,j);v=add(v,mul(feat(a),Number(h[j])-p(a,c)))}return mul(v,task==="parity"?+(sum([...h].map(Number))%2===0):+(sum([...h].map(Number))>=Math.ceil(c.threshold*c.L)))}
function momentTree(c){let m=new Map();function rec(h){if(h.length===c.L){let v=grad(h,c),a={m:v,s:dot(v,v)};m.set(h,a);leaves++;return a}let t=p(h,c),a=rec(h+"0"),b=rec(h+"1"),r={m:add(mul(a.m,1-t),mul(b.m,t)),s:(1-t)*a.s+t*b.s};m.set(h,r);return r}rec("");return m}
function enumerate(h,k,c,sh,cb,w=1,path=""){if(k===0){cb(path,w);return}let q=p(h+path,c,sh);enumerate(h,k-1,c,sh,cb,w*(1-q),path+"0");enumerate(h,k-1,c,sh,cb,w*q,path+"1")}
function outcomes(h,x,c){let r=[],a=1,y=h;for(let j=0;j<x.length;j++){let b=p(y,c),q=p(y,c,true),t=x[j]==="1"?b/q:(1-b)/(1-q);t=Math.min(1,t);if(a*(1-t)>0)r.push([y+(x[j]==="1"?"0":"1"),a*(1-t)]);a*=t;y+=x[j]}let b=p(y,c);r.push([y+"0",a*(1-b)],[y+"1",a*b]);return r}
function tables(c){let tree=momentTree(c),tabs=[new Map(),new Map(),new Map()],root=tree.get("");function push(tab,key,w,m,s){if(!tab.has(key))tab.set(key,{w:0,m:z(),s:0});let a=tab.get(key);a.w+=w;a.m=add(a.m,mul(m,w));a.s+=s*w}
enumerate("",L0,c,false,(h,wh)=>{let t=tree.get(h);push(tabs[0],h,wh,t.m,t.s);enumerate(h,K,c,true,(x,wx)=>{let out=outcomes(h,x,c);maxMass=Math.max(maxMass,Math.abs(sum(out.map(a=>a[1]))-1));let m=z(),s=0;for(let [y,w] of out){let a=tree.get(y);m=add(m,mul(a.m,w));s+=a.s*w;push(tabs[2],y,wh*wx*w,a.m,a.s)}push(tabs[1],h+"|"+x,wh*wx,m,s)})});
for(let tab of tabs){let m=z(),mass=0,s=0;for(let a of tab.values()){mass+=a.w;m=add(m,a.m);s+=a.s;if(a.w>0){a.m=mul(a.m,1/a.w);a.s/=a.w}}maxMass=Math.max(maxMass,Math.abs(mass-1));maxMean=Math.max(maxMean,...sub(m,root.m).map(Math.abs));maxResidual=Math.max(maxResidual,Math.abs(s-root.s))}
return {tabs,root}}
function block(h,k,c){let x="",y=h;for(let j=0;j<k;j++)x+=rnd()<p(h+x,c,true)?"1":"0";for(let j=0;j<k;j++){let b=p(y,c),q=p(y,c,true),t=x[j]==="1"?b/q:(1-b)/(1-q);if(rnd()<Math.min(1,t))y+=x[j];else{return {x,y:y+(b>q?"1":"0")}}}if(y.length<c.L)y+=rnd()<p(y,c)?"1":"0";return {x,y}}
function seg(h,end,c){let t=0;while(h.length<end){let k=Math.min(K,end-h.length);let b=block(h,k,c);h=b.y.slice(0,end);t+=c.speed*(1+.13*k)}return {h,t}}
function draw(c){let a=seg("",L0,c),b=block(a.h,K,c),d=seg(b.y,c.L,c);rollouts++;return {keys:[a.h,a.h+"|"+b.x,b.y],cost:[a.t,a.t+c.speed*.13*K,a.t+c.speed*(1+.13*K)],full:a.t+c.speed*(1+.13*K)+d.t,g:grad(d.h,c)}}
let ts=cs.map(tables),pilot=cs.map(c=>Array.from({length:256},()=>draw(c))), cal=Array.from({length:128},()=>cs.map(draw)),test=Array.from({length:128},()=>cs.map(draw));
let fits=pilot.map(a=>[0,1,2].map(j=>{let m=new Map();for(let x of a){let k=x.keys[j];if(!m.has(k))m.set(k,{n:0,s:z()});let b=m.get(k);b.n++;b.s=add(b.s,x.g)}return m}));
let rootPred=pilot.map(a=>mul(a.map(x=>x.g).reduce(add,z()),1/a.length));
let region=[0,1].map(e=>[0,1,2].map(j=>{let map=new Map();for(let i=4*e;i<4*e+4;i++)for(let [k,a] of fits[i][j]){if(!map.has(k))map.set(k,{n:0,s:z()});let b=map.get(k);b.n+=a.n;b.s=add(b.s,a.s)}return map}));
let global=[0,1,2].map(j=>{let m=new Map();for(let e=0;e<2;e++)for(let [k,a] of region[e][j]){if(!m.has(k))m.set(k,{n:0,s:z()});let b=m.get(k);b.n+=a.n;b.s=add(b.s,a.s)}return m});
function fit(i,j,k,kind){let prior=j===1?fit(i,0,k.split("|")[0],kind):rootPred[i];
if(kind==="regional"||kind==="global"){let a=(kind==="regional"?region[i>>2][j]:global[j]).get(k);if(a){let b=fits[i][j].get(k),n=a.n-(b?b.n:0),s=b?sub(a.s,b.s):a.s;if(n>0)prior=mul(add(s,mul(prior,12)),1/(n+12))}}
let a=fits[i][j].get(k);return a?mul(add(a.s,mul(prior,8)),1/(a.n+8)):prior}
let fullV=sum(ts.map(t=>t.root.s-dot(t.root.m,t.root.m)))/N**2;
function stats(j,kind){return ts.map((t,i)=>{let cond=0,expl=0,err=0,em=z();for(let [k,a] of t.tabs[j]){if(a.w===0)continue;let m=kind==="oracle"?a.m:fit(i,j,k,kind),e=sub(a.m,m);cond+=a.w*(a.s-dot(a.m,a.m));expl+=a.w*dot(sub(a.m,t.root.m),sub(a.m,t.root.m));err+=a.w*dot(e,e);em=add(em,mul(e,a.w))}return {cond,expl,err,em}})}
function variance(st,pi){let v=0;for(let i=0;i<N;i++){let a=st[i];v+=a.expl+a.cond/pi[i]+(1/pi[i]-1)*a.err;for(let k=i+1;k<N;k++)v+=2*(1/Math.max(pi[i],pi[k])-1)*dot(a.em,st[k].em)}return v/N**2}
let rank=cs.map((_,i)=>i).sort((i,k)=>cs[i].speed*cs[i].L-cs[k].speed*cs[k].L);
let policies=[Array(N).fill(1)];for(let n=1;n<=N;n++)for(let pi of [.2,.4,.6,.8]){let a=Array(N).fill(1);for(let k=N-n;k<N;k++)a[rank[k]]=pi;policies.push(a)}
function cost(data,j,pi){return avg(data.map(d=>{let grid=[0,...new Set(pi),1].sort((a,b)=>a-b),t=0;for(let k=1;k<grid.length;k++){let u=(grid[k]+grid[k-1])/2;t+=(grid[k]-grid[k-1])*Math.max(...d.map((x,i)=>u<pi[i]?x.full:x.cost[j]))}return t}))}
let baseline=avg(test.map(d=>Math.max(...d.map(x=>x.full))));
let calG=cal.map(d=>mul(d.map(x=>x.g).reduce(add,z()),1/N)),calMean=mul(calG.reduce(add,z()),1/calG.length);
let baseVar=avg(calG.map(x=>dot(sub(x,calMean),sub(x,calMean))));
for(let j=0;j<3;j++)for(let kind of ["local","regional","global"]){
let st=stats(j,kind),rs=cal.map(d=>d.map((x,i)=>sub(x.g,fit(i,j,x.keys[j],kind))));
let gram=Array.from({length:N},(_,i)=>Array.from({length:N},(_,k)=>avg(rs.map(r=>dot(r[i],r[k])))/N**2));
function calV(pi){let v=baseVar;for(let i=0;i<N;i++){v+=(1/pi[i]-1)*gram[i][i];for(let k=i+1;k<N;k++)v+=2*(1/Math.max(pi[i],pi[k])-1)*gram[i][k]}return v}
let best=policies[0],bestJ=Infinity;for(let pi of policies){let q=cost(cal,j,pi)*calV(pi);if(q<bestJ){bestJ=q;best=pi}}
let v=variance(st,best),c=cost(test,j,best),ratio=c*v/(baseline*fullV);
rows.push({seed,het,task,stage:j,kind,expl:sum(st.map(x=>x.expl))/N**2/fullV,cond:sum(st.map(x=>x.cond))/N**2/fullV,predError:sum(st.map(x=>x.err))/N**2/fullV,ratio,cost:c/baseline,varRatio:v/fullV,pi:best,headroom:Math.max(0,(1-ratio)/(v/fullV)),ratioWith2Pct:ratio+.02*v/fullV})
}
}
let summary=[];for(let task of ["count","parity"])for(let j=0;j<3;j++)for(let kind of ["local","regional","global"]){let a=rows.filter(x=>x.task===task&&x.stage===j&&x.kind===kind);summary.push({task,stage:j,kind,n:a.length,expl:avg(a.map(x=>x.expl)),predError:avg(a.map(x=>x.predError)),ratio:avg(a.map(x=>x.ratio)),cost:avg(a.map(x=>x.cost)),varRatio:avg(a.map(x=>x.varRatio)),headroom:avg(a.map(x=>x.headroom)),ratioWith2Pct:avg(a.map(x=>x.ratioWith2Pct)),good:a.filter(x=>x.ratio<=.85).length,wins:a.filter(x=>x.ratio<.999999).length})}
return {seeds,L0,K,leaves,rollouts,maxMass,maxMean,maxResidual,summary,rows};
}
console.log(JSON.stringify(run(),null,2));