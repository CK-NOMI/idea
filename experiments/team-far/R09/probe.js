// Forward adapted from FAR/R08 and HCA/R09/FEIA. node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
 const V=27,d=16,D=V*d,sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),rms=x=>x.map(t=>t/Math.sqrt(dot(x,x)/d+1e-5)),sm=a=>{let m=Math.max(...a),v=a.map(x=>Math.exp(x-m)),s=sum(v);return v.map(x=>x/s)};
 function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296}}
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=raw.slice(off,off+m);off+=m;return r});const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));return h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),first;for(let l=0;l<2;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!l)first=seq.at(-1)}let hlast=seq.at(-1);return {h:hlast,p:sm(mat(W,hlast)),q:sm(mat(W,first))}}}
 const docs=[...new Set(names.split(/\r?\n/).filter(s=>/^[a-z]{6,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>(s.charCodeAt(0)-97)%4===c)),enc=s=>s.split("").map(c=>c.charCodeAt(0)-97),train=buckets.map(x=>x.slice(0,Math.floor(.8*x.length))),test=buckets.map(x=>x.slice(Math.floor(.8*x.length)));
 const mkCounts=ss=>{let all=[Array(V).fill(0),Array(V).fill(0)],mp=new Map();for(let s of ss){let z=enc(s),k=s.slice(0,2);if(!mp.has(k))mp.set(k,[Array(V).fill(0),Array(V).fill(0)]);for(let i=0;i<2;i++){all[i][z[i+2]]++;mp.get(k)[i][z[i+2]]++}}return {all,mp}};
 const stats={global:mkCounts(train.flat()),region:[mkCounts(train[0].concat(train[1])),mkCounts(train[2].concat(train[3]))]};
 function predict(stats,k){return stats.all.map((a,i)=>{let n=sum(a),q=a.map(x=>(x+.5)/(n+.5*V)),c=stats.mp.get(k)?.[i]||Array(V).fill(0),s=sum(c);return c.map((x,j)=>(x+8*q[j])/(s+8))})}
 const top=(p,k)=>Array.from({length:V},(_,i)=>i).sort((i,j)=>p[j]-p[i]||i-j).slice(0,k);
 const score=(f,y)=>Array.from({length:D},(_,i)=>((Math.floor(i/d)===y?1:0)-f.p[Math.floor(i/d)])*f.h[i%d]);
 const corr=(f,vals)=>{let mu=dot(f.p,vals);return Array.from({length:D},(_,i)=>f.p[Math.floor(i/d)]*(vals[Math.floor(i/d)]-mu)*f.h[i%d])};
 const norm2=a=>dot(a,a);
 function evaluate(f,name,client,random){
  let z=enc(name),h=[26,...z.slice(0,2)],fs=[f(h),...Array.from({length:V},(_,v)=>f(h.concat(v)))],gold=z.slice(2,4),s0=Array.from({length:V},(_,x)=>score(fs[0],x)),s1=fs.slice(1).map(a=>Array.from({length:V},(_,x)=>score(a,x))),configs=[],mu=new Float64Array(D),sq={raw:0,constant:0,sketch4:0,sketch8:0},means={},maxError=0;
  let wt=0,wr=0;
  for(let x=0;x<V;x++)for(let y=0;y<V;y++){let p=fs[0].p[x]*fs[x+1].p[y],r=(Number(x===gold[0])+Number(y===gold[1]))/2;let n=0;for(let i=0;i<D;i++){let g=s0[x][i]+s1[x][y][i];mu[i]+=p*r*g;n+=g*g}wt+=p*n;wr+=p*r*n}
  let base=wr/wt,mu2=0;for(let v of mu)mu2+=v*v;
  // Predicted rewards use only earlier training names. Oracle state baseline separately uses held-out gold.
  for(let source of ["global","region","oracle"]){
   let pred=source==="oracle"?gold.map(v=>Array.from({length:V},(_,i)=>Number(i===v))):predict(source==="global"?stats.global:stats.region[client>>1],name.slice(0,2));
   let q0=fs[0].q,q1=fs.slice(1).map(x=>x.q),p0=fs[0].p;
   let b0=(dot(p0,pred[0])+dot(q0,pred[1]))/2,b1=fs.slice(1).map((x,i)=>(pred[0][i]+dot(x.p,pred[1]))/2);
   if(source==="oracle")b0=sum(fs.slice(1).map((f,i)=>p0[i]*(pred[0][i]+dot(f.p,pred[1]))/2));
   let bAct0=Array.from({length:V},(_,i)=>(pred[0][i]+dot(source==="oracle"?fs[i+1].p:q1[i],pred[1]))/2);
   for(let k of source==="oracle"?[0,27]:[0,1,4,8,27]){
    let ids0=new Set(top(q0,k)),c0=bAct0.map((v,i)=>ids0.has(i)?v-b0:0),c1=fs.slice(1).map((f,i)=>{let ids=new Set(top(f.q,k));return Array.from({length:V},(_,y)=>ids.has(y)?(pred[0][i]+pred[1][y])/2-b1[i]:0)}),C0=corr(fs[0],c0),C1=fs.slice(1).map((f,i)=>corr(f,c1[i]));let key=`${source}_${k}`;
    configs.push({key,b0,b1,c0,c1,C0,C1});sq[key]=0;means[key]=new Float64Array(D);
   }
  }
  // Row support sizes for the Rademacher projection comparison, same random partition as R08.
  let perm=Array.from({length:V},(_,v)=>v);for(let i=V-1;i>0;i--){let j=Math.floor(random()*(i+1));[perm[i],perm[j]]=[perm[j],perm[i]]}
  const sizes=K=>{let owners=Array(V),ns=Array(K).fill(0);perm.forEach((v,i)=>{owners[v]=i%K;ns[i%K]+=d});return owners.map(j=>ns[j])},n4=sizes(4),n8=sizes(8);
  for(let x=0;x<V;x++)for(let y=0;y<V;y++){
   let p=fs[0].p[x]*fs[x+1].p[y],r=(Number(x===gold[0])+Number(y===gold[1]))/2,a=s0[x],b=s1[x][y];
   for(let i=0;i<D;i++){let s=a[i]+b[i],u=r*s,u2=(r-base)*s;sq.raw+=p*u*u;sq.constant+=p*u2*u2;sq.sketch4+=p*u*u*n4[Math.floor(i/d)];sq.sketch8+=p*u*u*n8[Math.floor(i/d)]}
   for(let c of configs){let n=0,m=means[c.key],v0=r-c.b0-c.c0[x],v1=r-c.b1[x]-c.c1[x][y];for(let i=0;i<D;i++){let g=v0*a[i]+v1*b[i]+c.C0[i]+c.C1[x][i];n+=g*g;m[i]+=p*g}sq[c.key]+=p*n}
  }
  for(let m of Object.values(means))for(let i=0;i<D;i++)maxError=Math.max(maxError,Math.abs(mu[i]-m[i]));
  return {meanSq:mu2,variance:Object.fromEntries(Object.entries(sq).map(([k,v])=>[k,Math.max(0,v-mu2)])),maxError,base};
 }
 let rows=[];
 for(let wi=0;wi<raws.length;wi++){let f=model(raws[wi]);for(let seed of [11,29,73]){let rand=rng(seed);for(let c=0;c<4;c++){let name=test[c][Math.floor(rand()*test[c].length)];rows.push({wi,seed,c,...evaluate(f,name,c,rand)})}}}
 const variance={};for(let k of Object.keys(rows[0].variance))variance[k]=sum(rows.map(x=>x.variance[k]))/rows.length;
 return {environment:"V8 CPU FP64 external microGPT two same-size checkpoints, exact 2-token path enumeration, no actual training, SD or GPU",rows,summary:{n:rows.length,variance,meanSq:sum(rows.map(x=>x.meanSq))/rows.length,maxMeanError:Math.max(...rows.map(x=>x.maxError)),trainCounts:train.map(x=>x.length),testCounts:test.map(x=>x.length)},notes:"Oracle baseline uses held-out gold and full conditional p, not deployable. Raw score and random-direction projected score target the true first-order gradient in the epsilon→0 limit, not R08 fixed-direction finite differences. Sketch moments analytically averaged over independent Rademacher signs; not measured variance from Monte Carlo. Learned predictors use prior names only; candidate top-k deterministic shallow q, no actual decoding/extra GPU calls."}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
