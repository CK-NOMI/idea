// node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
const V=27,d=16,H=4,k=4,BOS=26,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error('shape');return h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),q;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let h=0;h<H;h++){let j=h*k,w=sm(K.slice(0,i+1).map(t=>dot(Q[i].slice(j,j+k),t.slice(j,j+k))/2));for(let l=j;l<j+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!q)q=sm(mat(W,seq.at(-1)))}return {p:sm(mat(W,seq.at(-1))),q}}}

function dotScore(p,i,j){return Number(i===j)-p[i]-p[j]+sum(p.map(x=>x*x))}
function audit(p,q,mu,mode){
 let n=p.length,delta=sum(p.map((v,i)=>Math.max(0,v-q[i]))),G=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>mode==="ar"?Number(i===j)*p[i]:mode==="sd"?(Number(i===j)*Math.min(q[i],p[i])+(delta>1e-15?Math.max(0,q[i]-p[i])*Math.max(0,p[j]-q[j])/delta:0)):mode==="pp"?p[i]*p[j]:q[i]*p[j]));
 let inc=p.map((_,i)=>sum(G[i])+sum(G.map(row=>row[i]))-G[i][i]),w=p.map((v,i)=>inc[i]>0?v/inc[i]:0);
 let trueG=p.map((v,i)=>v*(mu[i]-dot(p,mu))),est=Array(n).fill(0),second=0,mcost=0,accept=0;
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){let t=G[i][j];if(t===0)continue;let ids=i===j?[i]:[i,j];mcost+=t*ids.length;accept+=t*Number(i===j);
 for(let a of ids){for(let z=0;z<n;z++)est[z]+=t*w[a]*mu[a]*(Number(z===a)-p[z]);for(let b of ids)second+=t*w[a]*w[b]*(a===b?mu[a]:mu[a]*mu[b])*dotScore(p,a,b)}}
 let variance=Math.max(0,second-dot(trueG,trueG)),err=Math.max(...est.map((v,i)=>Math.abs(v-trueG[i])));
 return {variance,cost:mcost,cv:variance*mcost,accept,err,incErr:mode==="sd"?Math.max(...inc.map((v,i)=>Math.abs(v-Math.max(p[i],q[i])))):0}
}
let docs=[...new Set(names.split(/\r?\n/).map(s=>s.trim()).filter(s=>/^[a-z]{6,14}$/.test(s)))],buckets=Array.from({length:4},(_,c)=>docs.filter(s=>Math.min(3,Math.floor((s.charCodeAt(0)-97)/7))===c)),rows=[];
for(let wi=0;wi<raws.length;wi++){let f=model(raws[wi]);for(let c=0;c<4;c++)for(let s of buckets[c].slice(-32)){let h=[BOS,...s.slice(0,3).split('').map(x=>x.charCodeAt(0)-97)],{p,q}=f(h),y=s.charCodeAt(3)-97;for(let type of ["sparse","dense","constant"]){let mu=p.map((_,i)=>type==="sparse"?(i===y?.85:.05):type==="dense"?.1+.8*(1-Math.abs(i-y)/26):.5);let r={wi,c,type};for(let mode of ["ar","sd","pq","pp"])r[mode]=audit(p,q,mu,mode);rows.push(r)}}}
let groups=["sparse","dense","constant"].map(type=>{let a=rows.filter(x=>x.type===type),g={type,n:a.length};for(let m of ["ar","sd","pq","pp"])g[m]={...Object.fromEntries(["variance","cost","cv","accept"].map(k=>[k,sum(a.map(x=>x[m][k]))/a.length])),winAr:a.filter(x=>x[m].cv<x.ar.cv-1e-12).length/a.length};return g});
return {rows,groups,maxError:Math.max(...rows.flatMap(x=>["ar","sd","pq","pp"].map(m=>x[m].err))),maxInclusionError:Math.max(...rows.map(x=>x.sd.incErr)),buckets:buckets.map(x=>x.length)}
}

const source={"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}],"forwardSource":"experiments/team-far/R13/probe.js"};
console.log(JSON.stringify({source,...probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))},null,2));
