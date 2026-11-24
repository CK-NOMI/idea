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
let docs=[...new Set(names.split(/\r?\n/).map(s=>s.trim()).filter(s=>/^[a-z]{6,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>Math.min(3,Math.floor((s.charCodeAt(0)-97)/7))===c)),rows=[];
for(let wi=0;wi<raws.length;wi++){let f=model(raws[wi]);for(let c=0;c<4;c++)for(let s of buckets[c].slice(-16)){let h=[BOS,...s.slice(0,3).split('').map(x=>x.charCodeAt(0)-97)],{p,q}=f(h),y=s.charCodeAt(3)-97,A=sd(p,q),B=product(p,q),Dmax=sum(A.map((r,i)=>r[i])),Dind=dot(p,q);
for(let type of ["sparse","dense","constant"]){let mu=p.map((_,i)=>type==="sparse"?(i===y?.85:.05):type==="dense"?.1+.8*(1-Math.abs(i-y)/26):.5);
for(let t of [.25,.5,1]){let D=(1-t)*Dmax+t*Dind,M=mix(A,B,t),r={wi,c,type,t,D,cost:2-D,sdD:Dmax};
 for(let calibration of [1,.5,0]){let m=mu.map(v=>.5+calibration*(v-.5)),pi=alloc(p,q,m,D),C=flow(p,q,pi),v=variance(C,p,q,mu,m),base=variance(M,p,q,mu,m),max=variance(A,p,q,mu,m),pp=variance(product(p,p),p,p,mu,m);let oracle=calibration===1?sum(p.map((x,i)=>x*x*mu[i]*(1-mu[i])*(1-2*x+dot(p,p))/pi[i])):null;
 r["c"+calibration]={v:v.v,base:base.v,sd:max.v,pp:pp.v,cv:v.v*v.cost,basecv:base.v*base.cost,sdcv:max.v*max.cost,ppcv:pp.v*pp.cost,error:check(C,p,q,pi,D),bias:v.bias,oracleError:oracle===null?0:Math.abs(oracle-v.v),allocErr:Math.abs(sum(pi)-(2-D))}}
 rows.push(r)}}}}
let groups=["sparse","dense","constant"].flatMap(type=>[.25,.5,1].map(t=>{let a=rows.filter(x=>x.type===type&&x.t===t),g={type,t,n:a.length,D:sum(a.map(x=>x.D))/a.length,cost:sum(a.map(x=>x.cost))/a.length};for(let k of ["c1","c0.5","c0"]){let r=a.map(x=>x[k]);g[k]={...Object.fromEntries(["v","base","sd","pp","cv","basecv","sdcv","ppcv"].map(z=>[z,sum(r.map(x=>x[z]))/r.length])),winMix:r.filter(x=>x.v<x.base-1e-12).length/r.length,winPP:r.filter(x=>x.cv<x.ppcv-1e-12).length/r.length}}return g}));
let flat=rows.flatMap(r=>["c1","c0.5","c0"].map(k=>r[k]));return {rows,groups,maxFlowError:Math.max(...flat.map(r=>r.error)),maxBias:Math.max(...flat.map(r=>r.bias)),maxOracleError:Math.max(...flat.map(r=>r.oracleError)),maxAllocError:Math.max(...flat.map(r=>r.allocErr)),maxOracleViolation:Math.max(0,...rows.map(r=>r.c1.v-r.c1.base)),buckets:buckets.map(x=>x.length)}
}

const source={"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}],"forwardSource":"experiments/team-far/R15/probe.js"};
console.log(JSON.stringify({source,...probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))},null,2));
