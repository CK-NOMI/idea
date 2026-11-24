// node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
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

const avgG=(gs)=>avg(gs),scale=(g,k)=>g.map(r=>r.map(x=>k*x)),plus=(a,b)=>a.map((r,i)=>r.map((x,j)=>x+b[i][j])),inner=(a,b)=>sum(a.map((r,i)=>dot(r,b[i]))),norm=g=>Math.sqrt(inner(g,g)),kl=(p,q)=>sum(p.map((x,i)=>x*Math.log(x/q[i])));
const evalF=evalSet.flatMap(set=>set.flatMap(name=>{let h=pre(name),ret=[];for(const y of labels(name)){let f=features(h);ret.push({deep:f.deep.at(-1),shallow:f.shallow.at(-1),y});h.push(y)}return ret}));
const basis=xs=>{let B=[];for(let x of xs){let v=x.slice();for(let pass=0;pass<2;pass++)for(const b of B){let z=dot(v,b);v=v.map((x,i)=>x-z*b[i])}let n=Math.sqrt(dot(v,v));if(n>1e-8)B.push(v.map(x=>x/n));}return B};
const proj=(g,B)=>g.map(r=>{let v=r.slice();for(const b of B){let z=dot(v,b);v=v.map((x,i)=>x-z*b[i])}return v});
const pc=(g,p)=>{let n=inner(p,p),v=inner(g,p);return v<0&&n>1e-30?plus(g,scale(p,-v/n)):g};
const metric=(w,q,w0,fs)=>{let pkl=0,qkl=0,over=0,overRef=0,nll=0,maxTV=0;for(const f of fs){let p0=sm(mat(w0,f.deep)),p=sm(mat(w,f.deep)),z=sm(mat(q,f.shallow));pkl+=kl(p0,p);qkl+=kl(p0,z);over+=1-tv(p,z);overRef+=1-tv(p0,z);maxTV=Math.max(maxTV,tv(p0,p));if(f.y!==undefined)nll-=Math.log(p[f.y]);}return {pkl:pkl/fs.length,qkl:qkl/fs.length,over:over/fs.length,overRef:overRef/fs.length,nll:nll/fs.length,maxTV}};
const rows=[];let causalError=0,total={accepted:0,attempted:0,proposed:0,verifies:0,deepPositions:0,shallowPositions:0};
for(const seed of [11,29,73]){
 st=seed;const tapes=[],gps=[];
 // one fixed cohort, all methods share the exact same committed SD paths
 for(let c=0;c<4;c++){const gp=zero(),fset=[];for(let i=0;i<4;i++){const name=trainSet[c][Math.floor(U()*trainSet[c].length)],gold=labels(name),h=pre(name);const rs=Array.from({length:3},()=>rollout(W,W,h,"SD",3));const rewards=rs.map(r=>sum(r.out.map((x,j)=>Number(x===gold[j])))/4),mean=sum(rewards)/3;
 for(let i=0;i<3;i++){const r=rs[i];for(const k in total)total[k]+=r.cost[k];let prefix=h.slice();for(const f of r.tape){const check=features(prefix);causalError=Math.max(causalError,...check.deep.at(-1).map((v,j)=>Math.abs(v-f.deep[j])));prefix.push(f.x);
 for(let v=0;v<V;v++)for(let j=0;j<d;j++)gp[v][j]+=(rewards[i]-mean)*(f.p[v]-Number(v===f.x))*f.deep[j];fset.push(f);}}}
 tapes.push(fset);gps.push(scale(gp,1/fset.length));}
 const pg=avgG(gps),w0=plus(W,scale(pg,-.08));
 const gs=tapes.map(fs=>{let g=zero();for(const f of fs){const p=sm(mat(w0,f.deep)),q=sm(mat(w0,f.shallow));for(let v=0;v<V;v++)for(let j=0;j<d;j++)g[v][j]+=(q[v]-p[v])*f.shallow[j]}return scale(g,1/fs.length)});
 const g=avgG(gs),rg=[avgG(gs.slice(0,2)),avgG(gs.slice(2))],rp=[avgG(gps.slice(0,2)),avgG(gps.slice(2))],rf=[tapes[0].concat(tapes[1]),tapes[2].concat(tapes[3])],all=rf.flat(),anchors=rf.map(fs=>fs.slice(0,4)),B=anchors.map(fs=>basis(fs.map(f=>f.deep))),BU=basis(anchors.flat().map(f=>f.deep)),BF=basis(all.map(f=>f.deep));
 const local=avgG(rg.map((r,i)=>proj(r,B[i]))),global=proj(g,BU),full=proj(g,BF);
 for(const [method,gg,indep] of [["none",zero(),false],["shared",g,false],["independent",g,true],["local-PC",avgG(rg.map((r,i)=>pc(r,rp[i]))),false],["global-PC",pc(g,pg),false],["local-null4",local,false],["union-null8",global,false],["full-null",full,false]]){
 const update=scale(gg,-.25),w=indep?w0:plus(w0,update),q=plus(w0,update);rows.push({seed,method,ranks:[B[0].length,B[1].length,BU.length,BF.length],gradRetain:norm(gg)/norm(g),draftDirectional:inner(g,gg)/inner(g,g),formalDirectional:inner(pg,gg),eval:metric(w,q,w0,evalF),anchor:metric(w,q,w0,anchors.flat()),train:metric(w,q,w0,all),localAnchorTV:rg.map((r,i)=>metric(plus(w0,scale(proj(r,B[i]),-.25)),w0,w0,anchors[i]).maxTV)});
 }
}
return {rows,total,causalError,params:off,evalN:evalF.length,docs:docs.length,buckets:buckets.map(x=>x.length)};
}

const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));const text=fs.readFileSync(process.argv[4],'utf8');console.log(JSON.stringify({models:[{id:'weights',...run(read(process.argv[2]),text)},{id:'checkpoint',...run(read(process.argv[3]).weights,text)}]},null,2));
