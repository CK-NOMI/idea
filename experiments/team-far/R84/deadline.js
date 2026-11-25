function runDP(raw){
 const sum=a=>a.reduce((s,x)=>s+x,0),avg=a=>sum(a)/a.length;
 function rate(rs,m){let f=Array.from({length:m+1},(_,j)=>j/m);for(let {p,q} of rs){let old=f;f=Array.from({length:m+1},(_,j)=>sum(q.map((v,i)=>{if(!v)return 0;let x=Math.min(m,j*p[i]/v),k=Math.min(m-1,Math.floor(x));return v*(old[k]+(x-k)*(old[k+1]-old[k]))})))}return 1-f[m]}
 function solve(r,k,ready,policy){let L=r.length,D=1,V=0,act=[],cost=0,coord=0;for(let t=L-1;t>=0;t--){let a=r[t];let done=policy==="all4"?1:policy==="all1"?0:(a[1]+k<a[0]?1:0),pend=policy==="adaptive"?(a[1]+k+(1-a[1])*D<a[0]+(1-a[0])*D?1:0):done;act[t]={done,pend,D};let v=a[done]+k*done;D=a[pend]+k*pend+(1-a[pend])*D-v;V+=v}
 let alive=1,base=0,choices=0;
 for(let t=0;t<L;t++){let z=act[t];let a=t<ready?0:alive,sel1=z.done,sel2=z.pend;let r0=r[t][sel1],r1=r[t][sel2];base+=(1-a)*r0+a*r1;choices+=(1-a)*sel1+a*sel2;if(t>=ready)alive*=1-r1;}
 let pay=alive;cost=base+pay+k*choices;
 let formula=V+(ready<L? (r[ready][act[ready].pend]+k*act[ready].pend+(1-r[ready][act[ready].pend])*act[ready].D-(r[ready][act[ready].done]+k*act[ready].done)):1);
 return {cost,base,pay,choices,formula,act};
 }
 const rows=raw.rows.filter(x=>x.K===27&&x.N===32),out=[];let maxDP=0,worse=0;
 for(let wi of [0,1])for(let seed of [11,29,73]){let bank=rows.filter(x=>x.wi===wi&&x.seed===seed);for(let edge of [0,1]){let rs=bank.slice(edge*4,edge*4+4),r4=Array.from({length:4},(_,j)=>{let a=Array.from({length:4},(_,i)=>rs[(i+j)%4]);return [rate(a,1),rate(a,4)]});for(let lag of [0,1,3])for(let slack of [0,2,4])for(let k of [0,.01,.03,.05,.08,.12]){
 let ready=7+lag,L=ready+slack+1,r=Array.from({length:L},(_,t)=>r4[t%4]),a=solve(r,k,ready,"adaptive"),b=solve(r,k,ready,"all1"),c=solve(r,k,ready,"all4"),d=solve(r,k,ready,"myopic");maxDP=Math.max(maxDP,Math.abs(a.cost-a.formula));worse+=+(a.cost>Math.min(b.cost,c.cost,d.cost)+1e-12);out.push({wi,seed,edge,lag,slack,k,adaptive:a,all1:b,all4:c,myopic:d});
 }}}
 let summary=[];for(let slack of [0,2,4])for(let k of [0,.01,.03,.05,.08,.12]){let a=out.filter(r=>r.slack===slack&&r.k===k);let row={slack,k};for(let mode of ["adaptive","all1","all4","myopic"])row[mode]=Object.fromEntries(["cost","base","pay","choices"].map(key=>[key,avg(a.map(r=>r[mode][key]))]));summary.push(row)}
 return {audit:{maxDP,worse,cases:out.length},summary,rows:out};
}
const fs=require('fs'); console.log(JSON.stringify(runDP(JSON.parse(fs.readFileSync(process.argv[2],'utf8'))),null,2));
