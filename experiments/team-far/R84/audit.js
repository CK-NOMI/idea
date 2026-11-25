function run(raw){
const sum=a=>a.reduce((s,x)=>s+x,0),avg=a=>sum(a)/a.length,zero=()=>Array(27).fill(0),pos=x=>Math.max(0,x);
function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296}}
function draw(p,u){let s=0;for(let i=0;i<p.length;i++){s+=p[i];if(u<s)return i}return p.length-1}
function soft(p,b){let a=p.map((x,i)=>x*Math.exp(b[i])),s=sum(a);return a.map(x=>x/s)}
function make(r,P,Q){let p=soft(r.p,P),q=soft(r.q,Q),res=p.map((v,i)=>pos(v-q[i])),d=sum(res);return {p,q,res:res.map(x=>d?x/d:0),d,lab:r.name.charCodeAt(3)-97}}
function lane(r,k,m,rand){let x=draw(r.q,rand()),q=r.q[x],don=pos(q-r.p[x]),t=m===1?don:Math.min(q/m,pos(don-k*q/m)),R=rand()<t*m/q,y=R?draw(r.res,rand()):x,next=0;if(m>1){let v=r.p[y],qy=r.q[y],rec=pos(v-qy),diag=j=>qy/m-Math.min(qy/m,pos(qy-v-j*qy/m));let l=R?0:rec+sum(Array.from({length:k},(_,j)=>diag(j))),len=R?rec:diag(k);next=Math.min(m-1,Math.floor(m*(l+rand()*len)/v));}return {y,R:+R,next}}
function schedule(items,repairs,deadline,kind){let uploads=[],todo=items.slice(),T=deadline;for(let t=0;t<=T;t++){let ready=todo.filter(x=>x.release<=t),last=todo.length&&todo.every(x=>x.release<=t),go=kind==="early"?ready.length&&(repairs[t]||t===T):kind==="final"?last&&(repairs[t]||t===T):kind==="separate"?t===T:0;if(go&&ready.length){uploads.push({t,items:ready,free:!!repairs[t]&&kind!=="separate"});todo=todo.filter(x=>x.release>t)}}if(todo.length)throw Error("deadline");let g=zero();for(let u of uploads){let a=zero();for(let x of u.items)for(let i=0;i<27;i++)a[i]+=x.g[i];for(let i=0;i<27;i++)g[i]+=a[i]}return {paid:uploads.filter(x=>!x.free).length,msg:uploads.length,g,wait:sum(uploads.flatMap(u=>u.items.map(x=>u.t-x.release)))/items.length}}
function greedy(intervals,free,T){let u=intervals.slice(),paid=0;for(let t=0;t<=T;t++)if(free[t]||u.some(x=>x[1]===t)){if(!free[t])paid++;u=u.filter(x=>x[0]>t||x[1]<t)}return paid}
function brute(intervals,free,T){let a=[];for(let t=0;t<=T;t++)if(!free[t])a.push(t);let best=T+1;for(let mask=0;mask<(1<<a.length);mask++){let n=0,s=free.slice();for(let i=0;i<a.length;i++)if(mask>>i&1){s[a[i]]=1;n++}if(n>=best)continue;if(intervals.every(([r,d])=>s.slice(r,d+1).some(Boolean)))best=n}return best}
let aRand=rng(181),bad=0;for(let i=0;i<600;i++){let intervals=Array.from({length:6},()=>{let a=Math.floor(aRand()*7),b=a+Math.floor(aRand()*(7-a));return[a,b]}),free=Array.from({length:7},()=>aRand()<.25);if(greedy(intervals,free,6)!==brute(intervals,free,6))bad++}
let rows=raw.rows.filter(x=>x.K===27&&x.N===32),results=[],maxGrad=0,maxParam=0,nUpdates=0,nTokens=0;
for(let wi of [0,1])for(let sd of [11,29,73])for(let seed of [7,19,41])for(let lag of [0,1,3])for(let slack of [0,2,4])for(let m of [1,4]){
 let b=rows.filter(x=>x.wi===wi&&x.seed===sd),P=zero(),Q=zero(),Pr=zero(),Qr=zero(),rand=rng(seed+wi*1001+sd*23+lag*123+slack*717),rounds=40,stats={};
 for(let method of ["separate","early","final"])stats[method]={paid:0,msg:0,wait:0};let repairs=0,formalCorrect=0,repairLeaf=0,probBefore=avg(b.map(r=>r.p[r.name.charCodeAt(3)-97]));
 for(let round=0;round<rounds;round++){
  let r=b.map(x=>make(x,P,Q)),deadline=7+lag+slack,allG=zero(),allRef=zero(),qGrad=zero();
  for(let e=0;e<2;e++){
   let events=Array(deadline+1).fill(0),items=[];
   for(let t=0;t<=deadline;t++){let state=Math.floor(rand()*m),rep=0,g=zero();for(let i=0;i<4;i++){let j=e*4+(i+t)%4,x=r[j],z=lane(x,state,m,rand);state=z.next;rep|=z.R;repairLeaf+=z.R;nTokens++;if(t<=7){let reward=+(z.y===x.lab);formalCorrect+=reward;for(let v=0;v<27;v++){g[v]+=reward*(+(z.y===v)-x.p[v])/64;qGrad[v]+=(x.q[v]-x.p[v])/64}}}
    events[t]=rep;repairs+=rep;if(t<=7)items.push({release:t+lag,g});
   }
   let ref=zero();for(let item of items)for(let v=0;v<27;v++)ref[v]+=item.g[v];
   for(let method of ["separate","early","final"]){let z=schedule(items,events,deadline,method);for(let k of ["paid","msg","wait"])stats[method][k]+=z[k];maxGrad=Math.max(maxGrad,...z.g.map((x,i)=>Math.abs(x-ref[i])));if(method==="final")for(let v=0;v<27;v++)allG[v]+=z.g[v]}
   for(let v=0;v<27;v++)allRef[v]+=ref[v];
  }
  for(let v=0;v<27;v++){P[v]+=.15*allG[v];Pr[v]+=.15*allRef[v];Q[v]-=.15*qGrad[v];Qr[v]-=.15*qGrad[v]}maxParam=Math.max(maxParam,...P.map((x,i)=>Math.abs(x-Pr[i])),...Q.map((x,i)=>Math.abs(x-Qr[i])));nUpdates++;
 }
 let post=avg(b.map(x=>soft(x.p,P)[x.name.charCodeAt(3)-97]));
 results.push({wi,sd,seed,lag,slack,m,rounds,repairs,repairLeaf,formalCorrect,probBefore,post,stats});
}
let summary=[];for(let lag of [0,1,3])for(let slack of [0,2,4])for(let m of [1,4]){let a=results.filter(x=>x.lag===lag&&x.slack===slack&&x.m===m),norm=2*40*a.length,rep=sum(a.map(x=>x.repairs))/norm,row={lag,slack,m,rep,probBefore:avg(a.map(x=>x.probBefore)),post:avg(a.map(x=>x.post))};for(let method of ["separate","early","final"]){let z=Object.fromEntries(["paid","msg","wait"].map(k=>[k,sum(a.map(x=>x.stats[method][k]))/norm]));row[method]={...z,starts:rep+z.paid,cost025:rep+z.paid+.25*z.msg,cost1:rep+z.paid+z.msg};}summary.push(row)}
return {audit:{bad,maxGrad,maxParam,nUpdates,nTokens},summary,results};
}
const fs=require('fs'); console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(process.argv[2],'utf8'))),null,2));
