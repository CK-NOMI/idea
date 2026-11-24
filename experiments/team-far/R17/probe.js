// node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
const V=27,d=16,H=4,k=4,BOS=26,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error('shape');return h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),q;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let h=0;h<H;h++){let j=h*k,w=sm(K.slice(0,i+1).map(t=>dot(Q[i].slice(j,j+k),t.slice(j,j+k))/2));for(let l=j;l<j+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!q)q=sm(mat(W,seq.at(-1)))}return {p:sm(mat(W,seq.at(-1))),q}}}


const clamp=(x,l,h)=>Math.min(h,Math.max(l,x));
function product(p,q){return q.map(x=>p.map(y=>x*y))}
function sd(p,q){let d=sum(p.map((x,i)=>Math.max(0,x-q[i])));return q.map((x,i)=>p.map((y,j)=>Number(i===j)*Math.min(x,p[i])+(d>1e-14?Math.max(0,x-p[i])*Math.max(0,y-q[j])/d:0)))}
function mix(A,B,t){return A.map((r,i)=>r.map((v,j)=>(1-t)*v+t*B[i][j]))}
function alloc(p,q,mu,D){
 let T=1-D,n=p.length,lo=p.map((x,i)=>Math.max(x,q[i])),hi=p.map((x,i)=>x+q[i]-Math.max(0,(x+q[i]-T)/2)),pp=dot(p,p),w=p.map((x,i)=>x*Math.sqrt(mu[i]*(1-mu[i])*(1-2*x+pp)));
 let l=0,h=1;let at=c=>w.map((x,i)=>clamp(c*x,lo[i],hi[i]));while(sum(at(h))<2-D-1e-13&&h<1e12)h*=2;
 for(let k=0;k<80;k++){let m=(l+h)/2;if(sum(at(m))<2-D)l=m;else h=m}
 return at((l+h)/2);
}
function flow(p,q,pi){
 let n=p.length,N=2*n+2,S=2*n,T=S+1,diag=p.map((v,i)=>Math.max(0,v+q[i]-pi[i])),a=q.map((v,i)=>Math.max(0,v-diag[i])),b=p.map((v,i)=>Math.max(0,v-diag[i]));
 let g=Array.from({length:N},()=>[]);function edge(u,v,cap){let e={v,cap,rev:g[v].length},r={v:u,cap:0,rev:g[u].length};g[u].push(e);g[v].push(r);return e}
 let es=Array.from({length:n},()=>Array(n).fill(null));for(let i=0;i<n;i++){edge(S,i,a[i]);edge(n+i,T,b[i]);for(let j=0;j<n;j++)if(i!==j)es[i][j]=edge(i,n+j,1)}
 for(;;){let lev=Array(N).fill(-1),Q=[S];lev[S]=0;for(let i=0;i<Q.length;i++)for(let e of g[Q[i]])if(e.cap>1e-14&&lev[e.v]<0){lev[e.v]=lev[Q[i]]+1;Q.push(e.v)}if(lev[T]<0)break;let it=Array(N).fill(0);function dfs(u,f){if(u===T)return f;for(;it[u]<g[u].length;it[u]++){let e=g[u][it[u]];if(e.cap<=1e-14||lev[e.v]!==lev[u]+1)continue;let v=dfs(e.v,Math.min(f,e.cap));if(v>1e-14){e.cap-=v;g[e.v][e.rev].cap+=v;return v}}return 0}while(dfs(S,1)>1e-14){}}
 return p.map((_,i)=>p.map((_,j)=>i===j?diag[i]:g[n+j][es[i][j].rev].cap))
}
function check(A,p,q,pi,D){return Math.max(...q.map((v,i)=>Math.abs(sum(A[i])-v)),...p.map((v,i)=>Math.abs(sum(A.map(r=>r[i]))-v)),...pi.map((v,i)=>Math.abs(p[i]+q[i]-A[i][i]-v)),Math.abs(sum(A.map((r,i)=>r[i]))-D))}
function variance(A,p,q,mu,m){
 let n=p.length,pp=dot(p,p),pi=p.map((v,i)=>v+q[i]-A[i][i]),err=mu.map((v,i)=>v-m[i]),res=p.map((v,i)=>v*(err[i]-dot(p,err))),v=-dot(res,res),mean=Array(n).fill(0);
 for(let i=0;i<n;i++){let w=p[i]/pi[i];v+=pi[i]*w*w*(mu[i]*(1-mu[i])+err[i]*err[i])*(1-2*p[i]+pp);for(let j=0;j<n;j++)if(i!==j)v+=(A[i][j]+A[j][i])*w*p[j]/pi[j]*err[i]*err[j]*(pp-p[i]-p[j]);for(let j=0;j<n;j++)mean[j]+=pi[i]*w*err[i]*(Number(i===j)-p[j])+p[i]*m[i]*(Number(i===j)-p[j])}
 let trueG=p.map((v,i)=>v*(mu[i]-dot(p,mu)));return {v,cost:sum(pi),bias:Math.max(...mean.map((v,i)=>Math.abs(v-trueG[i]))),pi}
}

function sparse(p,q,di){
 const n=p.length,a=q.map((x,i)=>Math.max(0,x-di[i])),b=p.map((x,i)=>Math.max(0,x-di[i])),E=[],diag=Array(n).fill(0);
 let i=0,j=0,ar=a[0],br=b[0];
 while(i<n&&j<n){let z=Math.min(ar,br);if(z>0){if(i===j)diag[i]+=z;else E.push([i,j,z])}ar-=z;br-=z;if(ar<=0){i++;ar=a[i]||0}if(br<=0){j++;br=b[j]||0}}
 let D=sum(diag),k=0;for(i=1;i<n;i++)if(diag[i]>diag[k])k=i;
 let kind="none",repairs=0;
 if(D>0&&diag[k]<=D/2){
  kind="rotate";let ids=[];for(i=0;i<n;i++)if(diag[i]>0)ids.push(i);
  let r=0,c=0,off=D/2;while(c<ids.length-1&&off>=diag[ids[c]]){off-=diag[ids[c]];c++}
  let u=diag[ids[0]],v=diag[ids[c]]-off;
  while(r<ids.length){let z=Math.min(u,v);if(z>0){if(ids[r]===ids[c]){if(z>1e-12)throw Error("rotation diagonal");}else{E.push([ids[r],ids[c],z]);repairs++}}u-=z;v-=z;if(u<=0){r++;u=r<ids.length?diag[ids[r]]:0}if(v<=0){c=(c+1)%ids.length;v=diag[ids[c]]}}
 }else if(D>0){
  kind="dominant";let remaining=diag[k],base=E.length;
  for(i=0;i<n;i++)if(i!==k&&diag[i]>0){let z=diag[i];E.push([k,i,z],[i,k,z]);remaining-=z;repairs++}
  for(i=0;i<base&&remaining>0;i++){let [r,c,v]=E[i];if(r===k||c===k||v===0)continue;let z=Math.min(v,remaining);E[i][2]-=z;E.push([r,k,z],[k,c,z]);remaining-=z;repairs++}
  if(remaining>1e-10)throw Error("infeasible "+remaining);
 }
 for(i=0;i<n;i++)if(di[i]>0)E.push([i,i,di[i]]);
 return {edges:E.filter(e=>e[2]>0),kind,repairs}
}
function auditSparse(p,q,di,r){let rows=Array(p.length).fill(0),cols=rows.slice(),d=rows.slice(),neg=0;for(let [i,j,v] of r.edges){rows[i]+=v;cols[j]+=v;if(i===j)d[i]+=v;neg=Math.max(neg,-v)}
 let e=0,l1=0;for(let i=0;i<p.length;i++){e=Math.max(e,Math.abs(rows[i]-q[i]),Math.abs(cols[i]-p[i]),Math.abs(d[i]-di[i]));l1+=Math.abs(rows[i]-q[i])+Math.abs(cols[i]-p[i])+Math.abs(d[i]-di[i])}return {e,l1,neg,edges:r.edges.length,kind:r.kind,repairs:r.repairs}}
function denseSparse(n,r){let A=Array.from({length:n},()=>Array(n).fill(0));for(let [i,j,v] of r.edges)A[i][j]+=v;return A}
let docs=[...new Set(names.split(/\r?\n/).map(s=>s.trim()).filter(s=>/^[a-z]{6,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>Math.min(3,Math.floor((s.charCodeAt(0)-97)/7))===c)),rows=[];
for(let wi=0;wi<raws.length;wi++){let f=model(raws[wi]);for(let c=0;c<4;c++)for(let s of buckets[c].slice(-16)){let h=[BOS,...s.slice(0,3).split('').map(x=>x.charCodeAt(0)-97)],{p,q}=f(h),y=s.charCodeAt(3)-97,A=sd(p,q),B=product(p,q),Dmax=sum(A.map((r,i)=>r[i])),Dind=dot(p,q);
for(let type of ["sparse","dense","constant"]){let mu=p.map((_,i)=>type==="sparse"?(i===y?.85:.05):type==="dense"?.1+.8*(1-Math.abs(i-y)/26):.5);
for(let t of [.25,.5,1]){let D=(1-t)*Dmax+t*Dind,pi=alloc(p,q,mu,D),di=p.map((x,i)=>x+q[i]-pi[i]),S=sparse(p,q,di),C=denseSparse(p.length,S),F=flow(p,q,pi),a=auditSparse(p,q,di,S);let r={wi,c,type,t,...a};
for(let cal of [1,.5,0]){let m=mu.map(v=>.5+cal*(v-.5)),vs=variance(C,p,q,mu,m),vf=variance(F,p,q,mu,m);r["c"+cal]={sparse:vs.v,dense:vf.v,bias:vs.bias,delta:vs.v-vf.v}}
rows.push(r)}}}}
let groups=["sparse","dense","constant"].map(type=>{let a=rows.filter(r=>r.type===type),g={type,n:a.length,edges:sum(a.map(r=>r.edges))/a.length};for(let k of ["c1","c0.5","c0"]){g[k]={meanSparse:sum(a.map(r=>r[k].sparse))/a.length,meanDense:sum(a.map(r=>r[k].dense))/a.length,maxAbsDifference:Math.max(...a.map(r=>Math.abs(r[k].delta))),win:a.filter(r=>r[k].delta< -1e-12).length/a.length}}return g});
function rng(seed){let x=seed;return ()=>{x=(Math.imul(x,1664525)+1013904223)>>>0;return (x+.5)/4294967296}}
function normalize(x){let z=sum(x);return x.map(v=>v/z)}
let large=[];for(let n of [2,3,27,1024,32768,128000])for(let kind of ["uniform","zipf","dominant","zeros"])for(let seed of [11,29,73]){
let U=rng(seed),p=normalize(Array.from({length:n},(_,i)=>kind==="uniform"?U()+.01:kind==="zipf"?(U()+.5)/(i+1)**1.1:kind==="dominant"?(i===0?4*n:U()+.01):(i%3===0?U()+.01:0))),q=normalize(Array.from({length:n},(_,i)=>kind==="uniform"?U()+.01:kind==="zipf"?(U()+.5)/(i+1)**1.1:kind==="dominant"?(i===0?3*n:U()+.01):(i%3===1?U()+.01:0)));
for(let t of [0,.5,1]){let di=p.map((v,i)=>(1-t)*Math.min(v,q[i])+t*v*q[i]),S=sparse(p,q,di),a=auditSparse(p,q,di,S);large.push({n,kind,seed,t,...a,ratio:a.edges/n})}}
return {rows,groups,large,maxMarginalError:Math.max(...rows.map(r=>r.e)),maxBias:Math.max(...rows.flatMap(r=>[r.c1.bias,r["c0.5"].bias,r.c0.bias])),maxOracleDiff:Math.max(...rows.map(r=>Math.abs(r.c1.delta))),maxLargeError:Math.max(...large.map(r=>r.e)),maxLargeL1:Math.max(...large.map(r=>r.l1)),maxLargeRatio:Math.max(...large.map(r=>r.ratio)),cases:large.reduce((a,r)=>(a[r.kind]=(a[r.kind]||0)+1,a),{}),branches:large.reduce((a,r)=>(a[r.kind+"_"+r.n]=(a[r.kind+"_"+r.n]||0)+1,a),{})}
}

const source={"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}],"prior":"experiments/team-far/R16/probe.js"};
console.log(JSON.stringify({source,...probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))},null,2));
