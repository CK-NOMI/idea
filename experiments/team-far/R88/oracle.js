function run(){
const V=3,E=2,seeds=[17,29,43],drifts=[.2,.7,1.4],costModes=["aligned","opposed","mixed"],lams=[0,.25,.5,1],N=24;
function rng(s){return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296}}
const soft=z=>{let m=Math.max(...z),a=z.map(x=>Math.exp(x-m)),s=a.reduce((u,v)=>u+v,0);return a.map(x=>x/s)};
function vertices(ps){let vals=Array.from({length:V},(_,j)=>[0,1,...ps.map(p=>p[j])]);let out=[],seen=new Set();for(let r=0;r<V;r++){let j=(r+1)%V,k=(r+2)%V;for(let u of vals[j])for(let v of vals[k]){let w=1-u-v;if(w>=-1e-12&&w<=1+1e-12){let q=[];q[j]=u;q[k]=v;q[r]=Math.max(0,w);let key=q.map(x=>x.toFixed(12)).join(",");if(!seen.has(key)){seen.add(key);out.push(q)}}}}return out}
function metrics(p,c,q,child,weights){let A=0,C=0;for(let e=0;e<E;e++)for(let x=0;x<V;x++){let a=Math.min(p[e][0][x],q[x]);A+=weights[e]*a;C+=weights[e]*q[x]*c[e][0][x];for(let y=0;y<V;y++){A+=weights[e]*a*Math.min(p[e][x+1][y],child[x][y]);C+=weights[e]*q[x]*child[x][y]*c[e][x+1][y]}}return {A,C}}
function opt(p,c,lam,weights,mode="joint"){let roots=vertices(p.map(t=>t[0])),ch=Array.from({length:V},(_,x)=>vertices(p.map(t=>t[x+1])));let best=null,bv=-Infinity;
if(mode==="mean"){roots=[Array.from({length:V},(_,j)=>p.reduce((s,t,e)=>s+weights[e]*t[0][j],0))];ch=ch.map((_,x)=>[Array.from({length:V},(_,j)=>p.reduce((s,t,e)=>s+weights[e]*t[x+1][j],0))]);}
for(let q of roots){let child=[];let val=0,rootVal=0;for(let x=0;x<V;x++){let a=p.map((t,e)=>weights[e]*Math.min(t[0][x],q[x]));let root=a.reduce((s,v)=>s+v,0)-lam*q[x]*weights.reduce((s,w,e)=>s+w*c[e][0][x],0);rootVal+=root;let bestC=-Infinity,qbest;
for(let z of ch[x]){let v=0;for(let e=0;e<E;e++)for(let y=0;y<V;y++)v+=a[e]*Math.min(p[e][x+1][y],z[y])-lam*q[x]*weights[e]*z[y]*c[e][x+1][y];if(v>bestC+1e-13){bestC=v;qbest=z}}child.push(qbest);val+=root+bestC;}
let crit=mode==="myopic"?rootVal:val;if(crit>bv+1e-13){bv=crit;best={q,child,...metrics(p,c,q,child,weights)}}}
return best}
function make(random,drift,mode){let rnd=()=>2*random()-1;let p=Array.from({length:E},()=>[]),z0=Array.from({length:V+1},()=>Array.from({length:V},rnd));for(let e=0;e<E;e++)for(let h=0;h<=V;h++)p[e].push(soft(z0[h].map(v=>v+drift*rnd())));
let r=Array.from({length:E},()=>Array.from({length:V},()=>Math.floor(random()*4)));if(mode==="aligned")r[1]=r[0].slice();if(mode==="opposed")r[1]=r[0].map(x=>3-x);
// each region initial resident expert differs. Cost counts additional distinct nonresident experts in hypothetical draft path. Each next token prefix routes to its mapped expert.
let resident=mode==="aligned"?[0,0]:[0,3];let c=Array.from({length:E},(_,e)=>[r[e].map(v=>+(v!==resident[e])),...r[e].map(i=>r[e].map(j=>+(j!==resident[e]&&j!==i)))]);
return {p,c,z0};}
let rows=[],checks=0,maxGap=0,viol=0,gridChecks=0;
for(let drift of drifts)for(let mode of costModes)for(let seed of seeds){let rand=rng(seed+1009*drifts.indexOf(drift)+37*costModes.indexOf(mode));let sum={};for(let l of lams)for(let a of ["joint","eal","myopic","mean","independent"])sum[[l,a]]={A:0,C:0,U:0};for(let t=0;t<N;t++){
let {p,c}=make(rand,drift,mode),w=[.5,.5],eal=opt(p,c,0,w),mean=opt(p,c,0,w,"mean");
for(let l of lams){let joint=opt(p,c,l,w),myopic=opt(p,c,l,w,"myopic"),a0=opt(p,c,l,[1,0]),a1=opt(p,c,l,[0,1]),ind={A:(a0.A+a1.A)/2,C:(a0.C+a1.C)/2};let vals={joint,eal,myopic,mean,independent:ind};for(let key in vals){let m=vals[key],s=sum[[l,key]];s.A+=m.A;s.C+=m.C;s.U+=m.A-l*m.C;if(key!=="independent"&&m.A-l*m.C>joint.A-l*joint.C+1e-10)viol++}checks++;
// coarse grid audit: 21 root grid; exact optimal conditional child vertices. Compare grid against claimed vertex optimum.
if(t===0){for(let u=0;u<=5;u++)for(let v=0;v<=5-u;v++){let q=[u/5,v/5,1-(u+v)/5],val=0;for(let x=0;x<V;x++){let a=p.map((t,e)=>w[e]*Math.min(t[0][x],q[x]));let root=a.reduce((s,v)=>s+v,0)-l*q[x]*w.reduce((s,w,e)=>s+w*c[e][0][x],0);let m=-Infinity;for(let z of vertices(p.map(t=>t[x+1]))){let v=0;for(let e=0;e<E;e++)for(let y=0;y<V;y++)v+=a[e]*Math.min(p[e][x+1][y],z[y])-l*q[x]*w[e]*z[y]*c[e][x+1][y];m=Math.max(m,v)}val+=root+m}maxGap=Math.max(maxGap,val-(joint.A-l*joint.C));gridChecks++;}}
}}
for(let l of lams)for(let a of ["joint","eal","myopic","mean","independent"]){let v=sum[[l,a]];rows.push({drift,mode,seed,lambda:l,arm:a,A:v.A/N,C:v.C/N,U:v.U/N})}}
let summary=[];for(let l of lams)for(let a of ["joint","eal","myopic","mean","independent"]){let s=rows.filter(x=>x.lambda===l&&x.arm===a);summary.push({lambda:l,arm:a,...Object.fromEntries(["A","C","U"].map(k=>[k,s.reduce((u,v)=>u+v[k],0)/s.length]))})}
return {spec:{V,E,seeds,drifts,costModes,lams,N,K:2,notes:"synthetic conditional categorical distributions and deterministic expert maps. Shared fully expressive q oracle, no actual SD rollout, Transformer, training, network or GPU. Cost distinct newly loaded experts relative to fixed resident expert, expected over entire draft regardless of rejection. Mandatory root/bonus omitted."},checks,viol,gridChecks,maxGap,summary,rows};
}
console.log(JSON.stringify(run(),null,2));
