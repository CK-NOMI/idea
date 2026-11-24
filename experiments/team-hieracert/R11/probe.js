// node probe.js weights.json input.txt
const fs=require("node:fs");
function run(raw,names){
const V=27,d=16,BOS=26,L=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((x,i)=>x+b[i]),rms=x=>{let s=Math.sqrt(dot(x,x)/d+1e-5);return x.map(t=>t/s)},sm=x=>{let m=Math.max(...x),a=x.map(v=>Math.exp(v-m)),s=sum(a);return a.map(v=>v/s)},arg=x=>x.indexOf(Math.max(...x)),tv=(a,b)=>sum(a.map((v,i)=>Math.abs(v-b[i])))/2;
let st=0;const U=()=>{st=(Math.imul(st,1664525)+1013904223)>>>0;return(st+.5)/4294967296};const draw=p=>{let u=U();for(let i=0;i<p.length;i++){u-=p[i];if(u<0)return i}return p.length-1};const randn=()=>Math.sqrt(-2*Math.log(U()))*Math.cos(2*Math.PI*U());
const vals=Array.from({length:7264},(_,i)=>raw[String(i)]);if(vals.some(x=>!Number.isFinite(x)))throw Error("checkpoint numeric schema");
let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=vals.slice(off,off+m);off+=m;return r});
const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");
const copy=w=>w.map(r=>r.slice()),zero=()=>Array.from({length:V},()=>Array(d).fill(0)),avg=ws=>ws[0].map((r,i)=>r.map((_,j)=>sum(ws.map(w=>w[i][j]))/ws.length));
function features(h,depth=2){let seq=h.map((t,i)=>rms(add(E[t],P[i]))),first=null;
 for(let l=0;l<depth;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(l===0)first=seq.map(x=>x.slice());}
 return {shallow:first,deep:seq};
}
const docs=names.split(/\r?\n/).map(s=>s.trim()).filter(s=>/^[a-z]{6,14}$/.test(s));
const region=s=>Math.min(3,Math.floor((s.charCodeAt(0)-97)/7));const buckets=Array.from({length:4},(_,c)=>docs.filter(x=>region(x)===c));const evalSet=buckets.map(x=>x.slice(-40)),trainSet=buckets.map(x=>x.slice(0,-40));
const pre=s=>[BOS,s.charCodeAt(0)-97,s.charCodeAt(1)-97],labels=s=>s.slice(2,6).split("").map(c=>c.charCodeAt(0)-97);
function evalModel(w,heads){let n=0,nll=0,qnll=0,over=0;for(let c=0;c<4;c++)for(const name of evalSet[c]){let h=pre(name);for(const y of labels(name)){let f=features(h),p=sm(mat(w,f.deep.at(-1))),q=sm(mat(heads[c],f.shallow.at(-1)));nll-=Math.log(p[y]);qnll-=Math.log(q[y]);over+=1-tv(p,q);n++;h.push(y)}}return {n,nll:nll/n,qnll:qnll/n,overlap:over/n}}
function rollout(w,qw,h,mode,cap,greedy=false){let out=[],tape=[],cost={verifies:0,deepPositions:0,shallowPositions:0,proposed:0,attempted:0,accepted:0};let full=h.slice();
 function commit(x,f,p){out.push(x);full.push(x);tape.push({x,deep:f.deep,shallow:f.shallow,p})}
 while(out.length<L){
  if(mode==="AR"){let f=features(full),p=sm(mat(w,f.deep.at(-1)));cost.verifies++;cost.deepPositions+=full.length;commit(greedy?arg(p):draw(p),{deep:f.deep.at(-1),shallow:f.shallow.at(-1)},p);continue}
  let tmp=full.slice(),xs=[],qs=[],n=Math.min(cap,L-out.length);
  for(let j=0;j<n;j++){let f=features(tmp,1);cost.shallowPositions+=tmp.length;let q=sm(mat(qw,f.shallow.at(-1))),x=greedy?arg(q):draw(q);xs.push(x);qs.push(q);tmp.push(x);cost.proposed++}
  const f=features(tmp);cost.verifies++;cost.deepPositions+=tmp.length;let base=full.length-1,ok=true;
  for(let j=0;j<n;j++){let p=sm(mat(w,f.deep[base+j])),q=qs[j],x=xs[j];cost.attempted++;let accept=greedy?x===arg(p):U()<Math.min(1,p[x]/q[x]);if(accept){cost.accepted++;commit(x,{deep:f.deep[base+j],shallow:f.shallow[base+j]},p)}else{let r=p.map((v,i)=>Math.max(0,v-q[i])),z=sum(r);if(!greedy&&z<=0)throw Error("empty residual");let y=greedy?arg(p):draw(r.map(v=>v/z));commit(y,{deep:f.deep[base+j],shallow:f.shallow[base+j]},p);ok=false;break}}
  if(ok&&out.length<L){let p=sm(mat(w,f.deep[base+n]));commit(greedy?arg(p):draw(p),{deep:f.deep[base+n],shallow:f.shallow[base+n]},p)}
 }return {out,tape,cost}}
function update(w,qw,batch,mode,shared){let gp=zero(),gq=zero(),count=0,reward=0,cs={verifies:0,deepPositions:0,shallowPositions:0,proposed:0,attempted:0,accepted:0};
 for(const name of batch){const gold=labels(name),rolls=Array.from({length:3},()=>rollout(w,qw,pre(name),mode,3));const R=rolls.map(r=>sum(r.out.map((x,j)=>Number(x===gold[j])))/L),mean=sum(R)/3;
 for(let i=0;i<3;i++){const r=rolls[i],adv=R[i]-mean;reward+=R[i];for(const k in cs)cs[k]+=r.cost[k];for(const f of r.tape){const q=sm(mat(qw,f.shallow));for(let v=0;v<V;v++)for(let j=0;j<d;j++){gp[v][j]+=adv*(f.p[v]-Number(v===f.x))*f.deep[j];gq[v][j]+=(q[v]-f.p[v])*f.shallow[j]}count++}}
 }
 const next=w.map((r,v)=>r.map((x,j)=>x-.08*(gp[v][j]+(shared?.25*gq[v][j]:0))/count));const qnext=shared?next:qw.map((r,v)=>r.map((x,j)=>x-.08*gq[v][j]/count));return {w:next,q:qnext,reward:reward/(batch.length*3),cost:cs,count};
}
let initial=evalModel(W,Array(4).fill(W));st=101;const randomHead=zero().map(r=>r.map(()=>.08*randn()));const randomNll=evalModel(randomHead,Array(4).fill(randomHead)).nll;
let checks=0,greedyMismatch=0,causalMax=0;for(let c=0;c<4;c++)for(const s of evalSet[c].slice(0,10)){let h=pre(s);const ar=rollout(W,W,h,"AR",3,true),sd=rollout(W,W,h,"SD",3,true);checks++;if(ar.out.join(",")!==sd.out.join(","))greedyMismatch++;let x=h.concat(labels(s)),f=features(x);for(let j=0;j<x.length;j++){let g=features(x.slice(0,j+1));causalMax=Math.max(causalMax,...f.deep[j].map((v,k)=>Math.abs(v-g.deep.at(-1)[k])))}}
const rows=[];
for(const seed of [11,29,73])for(const variant of ["AR-global","SD-global","SD-regional","SD-shared"]){
 st=seed;let w=copy(W),heads=Array.from({length:4},()=>copy(W)),shared=variant==="SD-shared",mode=variant.startsWith("AR")?"AR":"SD",tot={verifies:0,deepPositions:0,shallowPositions:0,proposed:0,attempted:0,accepted:0},reward=0,steps=0;
 for(let round=0;round<4;round++){let ew=[],eq=[];for(let e=0;e<2;e++){let wp=copy(w),qp=copy(heads[2*e]);for(let local=0;local<2;local++){let clients=[];for(let c=2*e;c<2*e+2;c++){let batch=Array.from({length:3},()=>trainSet[c][Math.floor(U()*trainSet[c].length)]);let a=update(wp,shared?wp:qp,batch,mode,shared);clients.push(a);reward+=a.reward;steps++;for(const k in tot)tot[k]+=a.cost[k]}wp=avg(clients.map(a=>a.w));qp=avg(clients.map(a=>a.q))}ew.push(wp);eq.push(qp)}
 w=avg(ew);heads=shared?Array(4).fill(w):variant==="SD-regional"?[eq[0],eq[0],eq[1],eq[1]]:Array(4).fill(avg(eq));}
 rows.push({seed,variant,...evalModel(w,heads),reward:reward/steps,steps,...tot,formalDelta:Math.sqrt(sum(w.flatMap((r,i)=>r.map((v,j)=>(v-W[i][j])**2))))});
}
return {environment:"V8 CPU; externally published numeric character checkpoint mapped by source shapes; actual forward/SD/SGD of output heads, frozen transformer backbone; no GPU or physical HFL",params:off,docs:docs.length,bucketLengths:buckets.map(x=>x.length),evalSize:evalSet.map(x=>x.length),initial,randomHeadNll:randomNll,checks,greedyMismatch,causalMax,rows};
}
console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(process.argv[2],"utf8")),fs.readFileSync(process.argv[3],"utf8")),null,2));
