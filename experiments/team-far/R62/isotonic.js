function run(){
const seeds=[11,29,73,101,137,173,211,251,293,337,379,419],N=8,sum=a=>a.reduce((s,x)=>s+x,0),avg=a=>sum(a)/a.length,z=()=>[0,0,0],add=(a,b)=>a.map((x,j)=>x+b[j]),sub=(a,b)=>a.map((x,j)=>x-b[j]),mul=(a,b)=>a.map(x=>x*b),dot=(a,b)=>sum(a.map((x,j)=>x*b[j])),sig=x=>1/(1+Math.exp(-x));
let rows=[],leaves=0,rollouts=0;
function iso(v,t,lambda){
let b=[];for(let i=0;i<v.length;i++){b.push({ids:[i],v:v[i],c:t[i]});while(b.length>1){let x=b[b.length-2],y=b[b.length-1];if(x.v/x.c>=y.v/y.c)break;b.splice(b.length-2,2,{ids:[...x.ids,...y.ids],v:x.v+y.v,c:x.c+y.c})}}
let p=[];for(let x of b)for(let i of x.ids)p[i]=Math.max(.05,Math.min(1,Math.sqrt(x.v/(lambda*x.c))));return p
}
for(let seed of seeds)for(let het of [.2,1,2])for(let task of ["count","parity"]){
let state=seed;const rnd=()=>((state=Math.imul(state,1664525)+1013904223>>>0)/4294967296);
let cs=Array.from({length:N},(_,i)=>({w:[(i<4?-1:1)*het*.65+(rnd()-.5)*het,.9+(rnd()-.5)*het,.8+(rnd()-.5)*het],L:[8,8,9,10,8,9,10,12][i],speed:[1,1.2,1.7,2.1,1,1.4,1.9,2.7][i],threshold:i%2?.55:.45}));
function feat(h){return [1,h.length?2*h[h.length-1]-1:0,h.length?2*sum(h)/h.length-1:0]}
function p(h,c,sh=false){let f=feat(h);return sig(c.w[0]+c.w[1]*f[1]+(sh?0:c.w[2]*f[2]))}
function grad(h,c){let v=z();for(let j=0;j<h.length;j++){let a=h.slice(0,j);v=add(v,mul(feat(a),h[j]-p(a,c)))}return mul(v,task==="parity"?+(sum(h)%2===0):+(sum(h)>=Math.ceil(c.threshold*c.L)))}
function seg(h,end,c,mode){let y=h.slice(),t=0;while(y.length<end){if(mode==="AR"){y.push(+(rnd()<p(y,c)));t+=c.speed;continue}let k=Math.min(3,end-y.length),a=y.slice(),xs=[],qs=[];for(let j=0;j<k;j++){let q=p(a,c,true),x=+(rnd()<q);xs.push(x);qs.push(q);a.push(x)}t+=c.speed*(1+.13*k);let good=true;for(let j=0;j<k;j++){let b=p(y,c),q=qs[j],x=xs[j],r=x?b/q:(1-b)/(1-q);if(rnd()<Math.min(1,r))y.push(x);else{y.push(+(b>q));good=false;break}}if(good&&y.length<end)y.push(+(rnd()<p(y,c)))}return {h:y,t}}
for(let cut of [2,4,6,7]){
let key=h=>sum(h.slice(0,cut).map((x,j)=>x*(1<<j)));
let mm=cs.map(c=>{let a=Array.from({length:2**cut},()=>({mass:0,m:z(),s:0}));function rec(h,w){if(h.length===c.L){let v=grad(h,c),x=a[key(h)];x.mass+=w;x.m=add(x.m,mul(v,w));x.s+=w*dot(v,v);leaves++;return}let q=p(h,c);rec([...h,0],w*(1-q));rec([...h,1],w*q)}rec([],1);let W0=0,W1=0,mu=z();for(let x of a){x.m=mul(x.m,1/x.mass);x.s/=x.mass;mu=add(mu,mul(x.m,x.mass));W0+=x.mass*dot(x.m,x.m);W1+=x.mass*(x.s-dot(x.m,x.m))}return {a,W0:W0-dot(mu,mu),W1}});
let w0=sum(mm.map(x=>x.W0))/N**2,v=mm.map(x=>x.W1/N**2),fullV=w0+sum(v);
for(let mode of ["AR","SD"]){
let train=[],test=[];for(let t=0;t<512;t++){let d=cs.map(c=>{let a=seg([],cut,c,mode),b=seg(a.h,c.L,c,mode);rollouts++;return [a.t,a.t+b.t]});(t<256?train:test).push(d)}
function cost(d,pi){let times=[0,...new Set(pi),1].sort((a,b)=>a-b),s=0;for(let j=1;j<times.length;j++){let u=(times[j-1]+times[j])/2;s+=(times[j]-times[j-1])*Math.max(...d.map((x,i)=>u<pi[i]?x[1]:x[0]))}return s}
let C=(data,pi)=>avg(data.map(d=>cost(d,pi))),V=pi=>w0+sum(v.map((x,i)=>x/pi[i])),one=Array(N).fill(1),base=C(test,one),rank=cs.map((_,i)=>i).sort((i,j)=>avg(train.map(x=>x[i][1]))-avg(train.map(x=>x[j][1]))),B=avg(train.map(d=>Math.max(...d.map(x=>x[0])))),prev=B;
let delta=rank.map(i=>{let T=Math.max(prev,avg(train.map(d=>d[i][1]))),d=T-prev;prev=T;return Math.max(1e-8,d)});
let uniform=[one,...Array.from({length:19},(_,i)=>Array(N).fill((i+1)*.05))];
let nested=[...uniform];for(let lambda of Array.from({length:81},(_,i)=>10**(-5+i*.125))){let ordered=iso(rank.map(i=>v[i]),delta,lambda),pi=one.slice();rank.forEach((i,j)=>pi[i]=ordered[j]);nested.push(pi)}
let best=arr=>arr.reduce((a,b)=>C(train,a)*V(a)<=C(train,b)*V(b)?a:b);
let u=best(uniform),q=best(nested),ratio=pi=>C(test,pi)*V(pi)/(base*fullV);
rows.push({seed,het,task,cut,mode,base,fullV,w0,w1:sum(v),uniform:u,nested:q,uniformRatio:ratio(u),nestedRatio:ratio(q),nestedCost:C(test,q)/base,nestedVariance:V(q)/fullV});
}
}
}
let summary=[];for(let task of ["count","parity"])for(let cut of [2,4,6,7])for(let mode of ["AR","SD"]){let a=rows.filter(x=>x.task===task&&x.cut===cut&&x.mode===mode);summary.push({task,cut,mode,uniform:avg(a.map(x=>x.uniformRatio)),nested:avg(a.map(x=>x.nestedRatio)),cost:avg(a.map(x=>x.nestedCost)),variance:avg(a.map(x=>x.nestedVariance)),wins:a.filter(x=>x.nestedRatio<1).length,good:a.filter(x=>x.nestedRatio<=.85).length})}
return {seeds,rollouts,leaves,summary,rows};
}
console.log(JSON.stringify(run(),null,2));