function transportProbe(raws,names){
const V=27,d=16,sum=a=>a.reduce((s,v)=>s+v,0),mean=a=>sum(a)/a.length,dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let v=x.map(t=>Math.exp(t-Math.max(...x)));return scale(v,1/sum(v))},copy=A=>A.map(r=>r.slice()),over=(a,b)=>sum(a.map((v,i)=>Math.min(v,b[i]))),hash=(x,s)=>{let h=s>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
let features=0,formalSteps=0,headSteps=0;
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{features++;let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<4;j++){let t=j*4,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+4),x.slice(t,t+4))/2));for(let l=t;l<t+4;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}
function fit(W,examples,steps,lr){let U=copy(W);for(let t=0;t<steps;t++){let g=Array.from({length:V},()=>zero(d));let z=0;for(let {x,p,w} of examples){let q=sm(mat(U,x));z+=w;for(let i=0;i<V;i++)for(let j=0;j<d;j++)g[i][j]+=w*(p[i]-q[i])*x[j];}for(let i=0;i<V;i++)for(let j=0;j<d;j++)U[i][j]+=lr*g[i][j]/z;}return U}
function ridge(data,k,lambda){let n=data[0].z.length;let a=Array.from({length:n},()=>zero(n+1));for(let o of data)for(let i=0;i<n;i++){a[i][n]+=o.z[i]*o.g[k];for(let j=0;j<n;j++)a[i][j]+=o.z[i]*o.z[j]}for(let i=0;i<n;i++)a[i][i]+=lambda;for(let i=0;i<n;i++){let b=a[i][i];for(let j=i;j<=n;j++)a[i][j]/=b;for(let k=0;k<n;k++)if(k!==i){let c=a[k][i];for(let j=i;j<=n;j++)a[k][j]-=c*a[i][j]}}return a.map(r=>r[n]);}

const avg=as=>as[0].map((r,i)=>r.map((_,j)=>mean(as.map(a=>a[i][j])))),sub=(a,b)=>a.map((r,i)=>add(r,scale(b[i],-1))),plus=(a,b)=>a.map((r,i)=>add(r,b[i])),diag=(R,b)=>R.map(r=>r.map((v,i)=>v*b[i])),mul=(A,B)=>A.map(r=>Array.from({length:d},(_,j)=>sum(r.map((v,i)=>v*B[i][j]))));
let words=[...new Set(names.toLowerCase().split(/\s+/).filter(w=>/^[a-z]{6,14}$/.test(w)))],rows=[],transportSteps=0;
for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);for(let seed of [11,29,73]){
let pools=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,40));
const sample=w=>{let h=[26,...w.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)];return {...feature(h),h,label:w.charCodeAt(3)-97}};
let clients=pools.map(p=>p.slice(0,24).map(sample)),tree=w=>{let s=sample(w);return {...s,fs:[s,...Array.from({length:V},(_,i)=>feature(s.h.concat(i)))]}};
let train=pools.flatMap(p=>p.slice(24,26).map(tree)),test=pools.flatMap(p=>p.slice(26,28).map(tree));
let stats=train.flatMap(s=>s.fs.map(f=>({z:f.x,g:f.y}))),B=Array.from({length:d},(_,i)=>ridge(stats,i,.1));
for(let eta of [.02,.1,.4]){
let ex=clients.map(cs=>cs.map(s=>({x:s.y,p:Array.from({length:V},(_,j)=>+(j===s.label)),w:1})));
let P0=ex.map(cs=>fit(W,cs,10,eta)),P1=ex.map((cs,e)=>fit(P0[e],cs,10,eta));formalSteps+=80;
const prep=(data,P)=>data.map(s=>({...s,p:P.map(w=>s.fs.map(t=>sm(mat(w,t.y))))}));let D=prep(train,P0),D1=prep(train,P1),T=prep(test,P1),c0=avg(P0),c1=avg(P1);
const trainHead=(data,es,init)=>{let egs=[];for(let s of data)for(let e of es)for(let j=0;j<=V;j++)egs.push({x:s.fs[j].x,p:s.p[e][j],w:j===0?1:s.p[e][0][j-1]});headSteps+=6;return fit(init,egs,6,.15);};
let U=trainHead(D,[0,1,2,3],c0),local=P0.map((p,e)=>trainHead(D,[e],p)),fresh=P1.map((p,e)=>trainHead(D1,[e],local[e]));
let R=P0.map(p=>sub(p,c0));
function trainDiag(mode){let b=Array(d).fill(1);for(let step=0;step<6;step++){let grad=zero(d);for(let s of D)for(let e=0;e<4;e++){let p=s.p[e],H=plus(U,diag(R[e],b)),q=s.fs.map(f=>sm(mat(H,f.x))),g=q.map(()=>zero(V));if(mode==="window"){for(let i=0;i<V;i++){let m=Math.min(p[0][i],q[0][i]),a=over(p[i+1],q[i+1]);g[0][i]=q[0][i]<p[0][i]?1+a:0;for(let j=0;j<V;j++)g[i+1][j]=q[i+1][j]<p[i+1][j]?m:0;}}else for(let j=0;j<=V;j++)for(let i=0;i<V;i++)g[j][i]=(j===0?1:p[0][j-1])*p[j][i]/q[j][i];
for(let j=0;j<=V;j++){let t=dot(q[j],g[j]),v=q[j].map((v,i)=>v*(g[j][i]-t));for(let k=0;k<d;k++){let a=0;for(let i=0;i<V;i++)a+=v[i]*R[e][i][k];grad[k]+=s.fs[j].x[k]*a;}}}
for(let k=0;k<d;k++)b[k]+=.8*(grad[k]/(D.length*4)-.001*(b[k]-1));transportSteps++;}return b;}
let bW=trainDiag("window"),bK=trainDiag("kl"),base=plus(U,sub(c1,c0)),res=P1.map(p=>sub(p,c1));
const score=heads=>{let a=0;for(let s of T)for(let e=0;e<4;e++){let q=s.fs.map(f=>sm(mat(heads[e],f.x))),p=s.p[e];for(let j=0;j<V;j++)a+=Math.min(p[0][j],q[0][j])*(1+over(p[j+1],q[j+1]));}return a/(4*T.length);};
let m={sharedStale:score(P1.map(()=>U)),tied:score(P1),staleLocal:score(local),freshLocal:score(fresh),identity:score(res.map(r=>plus(base,r))),ordinaryBridge:score(res.map(r=>plus(base,mul(r,B)))),tunedLens:score(P1.map(r=>mul(r,B))),diagKL:score(res.map(r=>plus(base,diag(r,bK)))),diagWindow:score(res.map(r=>plus(base,diag(r,bW))))};rows.push({wi,seed,eta,m,bW,bK});}}}
let summary=[.02,.1,.4].map(eta=>{let r=rows.filter(o=>o.eta===eta);return {eta,...Object.fromEntries(Object.keys(r[0].m).map(k=>[k,mean(r.map(o=>o.m[k]))])),winKL:r.filter(o=>o.m.diagWindow>o.m.diagKL).length,winFresh:r.filter(o=>o.m.diagWindow>o.m.freshLocal).length}});return {features,formalSteps,headSteps,transportSteps,summary,rows};}

const fs=require('node:fs');console.log(JSON.stringify(transportProbe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
