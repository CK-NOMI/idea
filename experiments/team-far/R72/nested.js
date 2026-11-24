function run(input,M=2048){
const sum=a=>a.reduce((s,v)=>s+v,0),avg=a=>sum(a)/a.length,rows=input.rows.filter(x=>x.K===27&&x.N===32);
function rng(seed){let x=seed>>>0;return ()=>((x=Math.imul(x,1664525)+1013904223>>>0)+.5)/4294967296}
function cdf(a){let s=0;return a.map(v=>s+=v)}
function draw(c,u){let z=u*c[c.length-1],l=0,h=c.length-1;while(l<h){let m=(l+h)>>1;if(z<c[m])h=m;else l=m+1}return l}
let violations=0,maxHops=0,rowsOut=[];
for(let wi of [0,1])for(let seed of [11,29,73]){let data=rows.filter(x=>x.wi===wi&&x.seed===seed);if(data.length!==8)throw Error("unexpected table");for(let scenario of ["withinHomogeneous","mixed","driftCluster"])for(let N of [16,32,64]){
let R=4,n=N/R,leaves=Array.from({length:N},(_,i)=>{let e=Math.floor(i/n),j=i%n,idx=scenario==="withinHomogeneous"?e:scenario==="mixed"?(e+j)%8:(e*2+j%2);let {p,q}=data[idx],delta=sum(p.map((v,k)=>Math.max(0,v-q[k])));return {i,ids:[i],p,q,log:p.map((v,k)=>Math.log(v/q[k])),res:cdf(p.map((v,k)=>Math.max(0,v-q[k]))),qc:cdf(q),delta}});
let regions=Array.from({length:R},(_,e)=>({ids:Array.from({length:n},(_,j)=>e*n+j),ch:leaves.slice(e*n,(e+1)*n),e})),root={ids:leaves.map(x=>x.i),ch:regions},star={ids:root.ids,ch:leaves};
const ratio=(node,y)=>sum(node.ids.map(i=>leaves[i].log[y[i]]));
let hops=0,rootCalls=0;
function force(node,y,rr){let before=ratio(node,y);if(!(before<1e-9))violations++;
if(!node.ch){y[node.i]=draw(node.res,rr());if(!(ratio(node,y)>-1e-9))violations++;return}
let count=0;while(true){let logs=node.ch.map(c=>ratio(c,y)),pref=1,w=logs.map(l=>{let a=Math.exp(Math.min(0,l)),v=pref*(1-a);pref*=a;return v});if(!(sum(w)>0))throw Error("no outgoing");let first=draw(cdf(w),rr());for(let j=first;j<node.ch.length;j++)if(j===first||rr()>=Math.exp(Math.min(0,logs[j])))force(node.ch[j],y,rr);
hops++;count++;if(node===root)rootCalls++;if(count>node.ch.length)throw Error("loop");let after=node.ch.map(c=>ratio(c,y)),l=sum(after),logDoverP=sum(after.map(v=>Math.min(0,-v))),stop=l>0?-Math.expm1(-l)/-Math.expm1(logDoverP):0;if(rr()<Math.min(1,stop))break}
maxHops=Math.max(maxHops,count);if(!(ratio(node,y)>-1e-9))violations++}
function apply(node,y,rr){if(rr()>=Math.exp(Math.min(0,ratio(node,y))))force(node,y,rr)}
let methods=["independent","regional","star","nested"],acc=Object.fromEntries(methods.map(m=>[m,{all:0,edges:0,moves:0,move2:0,maxRegion:0,tail8:0,hops:0,rootHops:0,first:0,pair:0,hist:Array(N+1).fill(0)}]));let theory={root:0,edge:0,leaf:sum(leaves.map(l=>l.delta))};let rs=rng(seed+wi*1000+N*23+["withinHomogeneous","mixed","driftCluster"].indexOf(scenario)*5000),streams=methods.map((m,i)=>rng(seed+wi*100000+N*71+i*100003+["withinHomogeneous","mixed","driftCluster"].indexOf(scenario)*3001));
for(let t=0;t<M;t++){let x=leaves.map(l=>draw(l.qc,rs()));theory.root+=1-Math.exp(Math.min(0,ratio(root,x)));theory.edge+=sum(regions.map(e=>1-Math.exp(Math.min(0,ratio(e,x)))));
for(let k=0;k<methods.length;k++){let m=methods[k],y=x.slice(),rr=streams[k],a=acc[m];hops=0;rootCalls=0;if(m==="independent")for(let l of leaves)apply(l,y,rr);else if(m==="regional")for(let e of regions)apply(e,y,rr);else apply(m==="star"?star:root,y,rr);
let cs=regions.map(e=>sum(e.ids.map(i=>+(x[i]!==y[i])))),moves=sum(cs),mx=Math.max(...cs);a.all+=+(moves>0);a.edges+=sum(cs.map(v=>+(v>0)));a.moves+=moves;a.move2+=moves*moves;a.maxRegion+=mx;a.tail8+=+(moves>=8);a.hops+=hops;a.rootHops+=rootCalls;a.first+=+(y[0]===0);a.pair+=+(y[0]===0&&y[n]===0);a.hist[moves]++}}
theory.root/=M;theory.edge/=M;
for(let m of methods){let a=acc[m];for(let k of Object.keys(a))if(k!=="hist")a[k]/=M;a.var=a.move2-a.moves*a.moves;a.cond=a.moves/a.all;a.leafTheory=theory.leaf;a.firstZ2=(a.first-leaves[0].p[0])**2*M/(leaves[0].p[0]*(1-leaves[0].p[0]));let p2=leaves[0].p[0]*leaves[n].p[0];a.pairZ2=(a.pair-p2)**2*M/(p2*(1-p2));rowsOut.push({wi,seed,scenario,N,method:m,theory,...a})}
}}
let summary=[];for(let N of [16,32,64])for(let m of ["independent","regional","star","nested"]){let a=rowsOut.filter(x=>x.N===N&&x.method===m),b={N,method:m,n:a.length};for(let k of ["all","edges","moves","var","maxRegion","tail8","hops","rootHops","cond","leafTheory","firstZ2","pairZ2"])b[k]=avg(a.map(x=>x[k]));b.rootBound=avg(a.map(x=>x.theory.root));b.edgeBound=avg(a.map(x=>x.theory.edge));summary.push(b)}
return {M,configs:rowsOut.length/4,draws:rowsOut.length*M,violations,maxHops,summary,rows:rowsOut}}
if(require.main===module){const fs=require("fs");console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(process.argv[2],"utf8"))),null,2))}
