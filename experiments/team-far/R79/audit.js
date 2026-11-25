function run(raw){
 const sum=a=>a.reduce((s,x)=>s+x,0),pos=x=>Math.max(0,x),tv=(a,b)=>sum(a.map((x,i)=>Math.abs(x-b[i])))/2;
 let error=0,flagError=0;
 function direction(a,b){
  const da=tv(a.p,a.q),db=tv(b.p,b.q),don=b.q.map((v,i)=>pos(v-b.p[i])),rec=b.p.map((v,i)=>pos(v-b.q[i]));let overlap=0;
  for(let y=0;y<a.p.length;y++){let py=a.p[y];if(!py)continue;let s=pos(py-a.q[y])/py;
   const t1=don.map((d,x)=>Math.min(d,s*b.q[x])),t0=don.map((d,x)=>d-t1[x]);overlap+=py*sum(t1);
   let col=Array(b.p.length).fill(0);
   for(const [w,t] of [[s,t1],[1-s,t0]])for(let x=0;x<b.q.length;x++){let row=0;for(let z=0;z<b.p.length;z++){let v=x===z?w*b.q[x]-t[x]:db?t[x]*rec[z]/db:0;row+=v;col[z]+=v;error=Math.max(error,-v)}error=Math.max(error,Math.abs(row-w*b.q[x]));}
   for(let z=0;z<col.length;z++)error=Math.max(error,Math.abs(col[z]-b.p[z]));
  }
  function flag(tau,audit=false){let A=sum(a.p.map((v,y)=>Math.min(tau*v,pos(v-a.q[y])))),t1=don.map((d,x)=>Math.min(d,tau*b.q[x])),B=sum(t1);
   if(audit){let col=Array(b.p.length).fill(0);for(const [w,t] of [[tau,t1],[1-tau,don.map((d,x)=>d-t1[x])]])for(let x=0;x<b.q.length;x++){let row=0;for(let z=0;z<b.p.length;z++){let v=x===z?w*b.q[x]-t[x]:db?t[x]*rec[z]/db:0;row+=v;col[z]+=v;flagError=Math.max(flagError,-v)}flagError=Math.max(flagError,Math.abs(row-w*b.q[x]));}for(let z=0;z<col.length;z++)flagError=Math.max(flagError,Math.abs(col[z]-b.p[z]));
   for(let y=0;y<a.p.length;y++){let v=a.p[y],r=pos(v-a.q[y]),f=Math.min(r,tau*v);flagError=Math.max(flagError,-f,-(tau*v-f),-(r-f),-(v-tau*v-r+f));}}
   return da+db-A*B/tau-(da-A)*(db-B)/(1-tau);
  }
  let fixed=flag(.5,true),best=fixed,tau=.5;for(let j=1;j<100;j++){let v=flag(j/100);if(v<best){best=v;tau=j/100;}}flag(tau,true);
  return {cost:da+db-overlap,fixed,best,tau};
 }
 function pair(a,b){let da=tv(a.p,a.q),db=tv(b.p,b.q);let full=0;for(let i=0;i<a.p.length;i++)for(let j=0;j<b.p.length;j++)full+=Math.abs(a.p[i]*b.p[j]-a.q[i]*b.q[j])/2;let independent=da+db-da*db,ab=direction(a,b),ba=direction(b,a);return {independent,ab:ab.cost,ba:ba.cost,best:Math.min(ab.cost,ba.cost),full,fixedAB:ab.fixed,fixedBA:ba.fixed,flagAB:ab.best,flagBA:ba.best,flagBest:Math.min(ab.best,ba.best),tauAB:ab.tau,tauBA:ba.tau,delta:da+db}}
 let rows=[];for(let wi of [0,1])for(let seed of [11,29,73]){let bank=raw.rows.filter(x=>x.K===27&&x.N===32&&x.wi===wi&&x.seed===seed);for(let i=0;i<bank.length;i++)for(let j=i+1;j<bank.length;j++)rows.push({wi,seed,i,j,...pair(bank[i],bank[j])});}
 const avg=k=>sum(rows.map(r=>r[k]))/rows.length;
 const means=Object.fromEntries(["independent","ab","ba","best","full","fixedAB","fixedBA","flagAB","flagBA","flagBest","delta"].map(k=>[k,avg(k)]));
 const counter=pair({p:[.25,.75],q:[.75,.25]},{p:[.25,.75],q:[.75,.25]});
 return {cases:rows.length,means,error,flagError,counter,strict:rows.filter(r=>r.best>r.full+1e-10).length,improved:rows.filter(r=>r.best<r.independent-1e-10).length,flagImproved:rows.filter(r=>r.flagBest<r.independent-1e-10).length,ordering:rows.filter(r=>Math.abs(r.ab-r.ba)>1e-10).length,rows};
}
const fs=require('fs'); console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(process.argv[2],'utf8'))),null,2));
