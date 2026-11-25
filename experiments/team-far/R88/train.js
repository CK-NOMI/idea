function train(){
const V=3,E=2,steps=20,seeds=[17,29,43],drifts=[.2,.7,1.4],modes=["aligned","opposed","mixed"],betas=[.05,.2,.8],lambda=1;
function rng(s){return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296}}
const soft=z=>{let m=Math.max(...z),a=z.map(x=>Math.exp(x-m)),s=a.reduce((u,v)=>u+v,0);return a.map(x=>x/s)};
const arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)), dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
function vertices(ps){let vals=arr(V,j=>[0,1,...ps.map(p=>p[j])]),out=[],seen=new Set();for(let r=0;r<V;r++){let j=(r+1)%V,k=(r+2)%V;for(let u of vals[j])for(let v of vals[k]){let w=1-u-v;if(w>=-1e-12&&w<=1+1e-12){let q=[];q[j]=u;q[k]=v;q[r]=Math.max(0,w);let key=q.map(x=>x.toFixed(12)).join(",");if(!seen.has(key)){seen.add(key);out.push(q)}}}}return out}
function metrics(p,c,q,child){let A=0,C=0;for(let e=0;e<E;e++)for(let x=0;x<V;x++){let a=Math.min(p[e][0][x],q[x]);A+=a/2;C+=q[x]*c[e][0][x]/2;for(let y=0;y<V;y++){A+=a*Math.min(p[e][x+1][y],child[x][y])/2;C+=q[x]*child[x][y]*c[e][x+1][y]/2}}return {A,C,U:A-lambda*C}}
function opt(p,c,lam){let roots=vertices(p.map(t=>t[0])),ch=arr(V,x=>vertices(p.map(t=>t[x+1]))),best=null,bv=-Infinity;for(let q of roots){let child=[],val=0;for(let x=0;x<V;x++){let a=p.map(t=>Math.min(t[0][x],q[x])/2),root=a[0]+a[1]-lam*q[x]*(c[0][0][x]+c[1][0][x])/2,bestC=-Infinity,qbest;for(let z of ch[x]){let v=0;for(let e=0;e<E;e++)for(let y=0;y<V;y++)v+=a[e]*Math.min(p[e][x+1][y],z[y])-lam*q[x]*z[y]*c[e][x+1][y]/2;if(v>bestC+1e-13){bestC=v;qbest=z}}child.push(qbest);val+=root+bestC}if(val>bv+1e-13){bv=val;best={q,child,...metrics(p,c,q,child)}}}return best}
function ce(p,t){let s=0;for(let e=0;e<E;e++)for(let h=0;h<=V;h++)for(let y=0;y<V;y++)s-=t[e][h][y]*Math.log(p[e][h][y])/(E*(V+1));return s}
function cost(p,c){let s=0;for(let e=0;e<E;e++)for(let x=0;x<V;x++)s+=p[e][0][x]*(c[e][0][x]+dot(p[e][x+1],c[e][x+1]))/2;return s}
function gradient(p,c,target,q,mode,beta){return arr(E,e=>{let gp=arr(V+1,h=>arr(V,y=>-target[e][h][y]/p[e][h][y]/(V+1)));if(mode==="eal"||mode==="joint"){const f=(a,b)=>a<b-1e-10?1:a>b+1e-10?0:.5;for(let x=0;x<V;x++){let a=Math.min(p[e][0][x],q.q[x]),t=1+dot(p[e][x+1].map((v,y)=>Math.min(v,q.child[x][y])),[1,1,1]);gp[0][x]-=beta*f(p[e][0][x],q.q[x])*t;for(let y=0;y<V;y++)gp[x+1][y]-=beta*a*f(p[e][x+1][y],q.child[x][y]);}}else if(mode==="direct"){for(let x=0;x<V;x++){gp[0][x]+=beta*(c[e][0][x]+dot(p[e][x+1],c[e][x+1]));for(let y=0;y<V;y++)gp[x+1][y]+=beta*p[e][0][x]*c[e][x+1][y];}}return gp.map((g,h)=>{let m=dot(g,p[e][h]);return g.map((v,y)=>p[e][h][y]*(v-m))})})}
let rows=[],updates=0,oracleCalls=0,checks=0,maxFD=0;
for(let drift of drifts)for(let mode of modes)for(let seed of seeds){let rand=rng(seed+1009*drifts.indexOf(drift)+37*modes.indexOf(mode)),rnd=()=>2*rand()-1,base=arr(V+1,()=>arr(V,rnd)),trueP=arr(E,()=>base.map(z=>soft(z.map(v=>v+drift*rnd())))),initial=arr(E,()=>base.map(z=>z.map(v=>v+.4*rnd()))),route=arr(E,()=>arr(V,()=>Math.floor(rand()*4)));if(mode==="aligned")route[1]=route[0].slice();if(mode==="opposed")route[1]=route[0].map(x=>3-x);let resident=mode==="aligned"?[0,0]:[0,3],c=arr(E,e=>[route[e].map(v=>+(v!==resident[e])),...route[e].map(i=>route[e].map(j=>+(j!==resident[e]&&j!==i)))]);
// 2 clients per edge; disjoint generated sample batches; aggregate counts locally. Same training data for all arms.
let local=arr(E,e=>arr(2,()=>trueP[e].map(p=>{let n=[0,0,0];for(let i=0;i<64;i++){let u=rand(),j=0;while(j<V-1&&u>=p[j]){u-=p[j];j++}n[j]++}return n.map(v=>v/64)}))),target=arr(E,e=>arr(V+1,h=>arr(V,y=>(local[e][0][h][y]+local[e][1][h][y])/2)));
let initialP=initial.map(z=>z.map(soft));let init=opt(initialP,c,lambda);oracleCalls++;
for(let beta of betas)for(let arm of ["CE","eal","joint","direct"]){
let z=initial.map(t=>t.map(v=>v.slice())),p;
for(let t=0;t<steps;t++){p=z.map(v=>v.map(soft));let q=null;if(arm==="eal"||arm==="joint"){q=opt(p,c,arm==="eal"?0:lambda);oracleCalls++}let g=gradient(p,c,target,q,arm,beta);
// hierarchical shared+regional parameterization: global mean step eta .3, regional centered step eta .15.
for(let h=0;h<=V;h++)for(let y=0;y<V;y++){let m=(g[0][h][y]+g[1][h][y])/2;for(let e=0;e<E;e++)z[e][h][y]-=.3*m+.15*(g[e][h][y]-m);}
updates++;
}
p=z.map(v=>v.map(soft));let q=opt(p,c,lambda);oracleCalls++;
rows.push({drift,mode,seed,beta,arm,NLL:ce(p,trueP),trainNLL:ce(p,target),A:q.A,C:q.C,U:q.U,initialNLL:ce(initialP,trueP),initialU:init.U,initialA:init.A,initialC:init.C});
}
}
let summary=[];for(let beta of betas)for(let arm of ["CE","eal","joint","direct"]){let s=rows.filter(r=>r.beta===beta&&r.arm===arm);summary.push({beta,arm,...Object.fromEntries(["NLL","trainNLL","A","C","U","initialNLL","initialU"].map(k=>[k,s.reduce((u,v)=>u+v[k],0)/s.length]))})}
return {spec:{seeds,drifts,modes,betas,steps,lambda,description:"actual synthetic categorical logit model SGD, four logical clients/two regions, 64 fixed draws/client/context, global mean plus regional centered update at each step; population teacher NLL known synthetic oracle, no independent model training seeds, no actual transformer/SD/physical HFL; fixed expert costs; exact full-information shared proposal reoptimized for eal/joint, tie uses .5 super/subgradient for fixed q, not gradient of optimized value"},updates,oracleCalls,rows,summary};
}
console.log(JSON.stringify(train(),null,2));
