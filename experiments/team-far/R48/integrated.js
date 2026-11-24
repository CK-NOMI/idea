function integratedProbe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}


 const zeros=()=>Array.from({length:V},()=>zero(d)),ma=(A,B)=>A.map((r,i)=>add(r,B[i])),ms=(A,t)=>A.map(r=>scale(r,t)),outer=(a,b)=>a.map(x=>scale(b,x)),sub=(A,B)=>ma(A,ms(B,-1)),tv=(a,b)=>.5*sum(a.map((x,i)=>Math.abs(x-b[i]))),over=(a,b)=>1-tv(a,b),mix=(a,b)=>scale(add(a,b),.5),gradprob=(p,g)=>{let t=dot(p,g);return p.map((v,i)=>v*(g[i]-t))},copy=A=>A.map(r=>r.slice());


 const norm=A=>Math.sqrt(sum(A.flat().map(v=>v*v))),avg=A=>ms(A.reduce(ma),1/A.length);
 let words=[...new Set(names.toLowerCase().split(/\s+/).filter(w=>/^[a-z]{6,14}$/.test(w)))],rows=[],evals=0,formalSteps=0,auxSteps=0,checks=0,maxBoundViolation=0,maxFD=0;
 for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);for(let seed of [11,29,73]){
 let pools=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,48));
 function sample(w){let h=[26,...w.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)],f=feature(h);evals++;return {...f,h,label:w.charCodeAt(3)-97}}
 let clients=pools.map(p=>p.slice(0,24).map(sample)),guard=pools.map(p=>p.slice(24,30).map(sample)),testLabels=pools.map(p=>p.slice(34,44).map(sample));
 let getTree=w=>{let s=sample(w);s.fs=[s,...Array.from({length:V},(_,i)=>feature(s.h.concat(i)))];evals+=V;return s};
 let cal=pools.flatMap((p,c)=>p.slice(30,32).map(w=>({...getTree(w),c}))),test=pools.flatMap((p,c)=>p.slice(32,34).map(w=>({...getTree(w),c})));
 function ce(F,cs){let g=zeros();for(let s of cs){let p=sm(mat(F,s.y));p[s.label]-=1;g=ma(g,outer(p,s.y));}return ms(g,1/cs.length);}
 function update(F,g,lr){let cloud=avg(g),edge=[avg(g.slice(0,2)),avg(g.slice(2,4))];return F.map((w,e)=>ma(w,ms(ma(ma(ms(cloud,.25),ms(edge[e>>1],.25)),ms(g[e],.5)),lr)));}
 function nll(F,data){return data.map((cs,c)=>-mean(cs.map(s=>Math.log(sm(mat(F[c],s.y))[s.label]))));}
 function route(F,s,mode,gate){let e=s.c,g=e>>1,edge=avg(F.slice(2*g,2*g+2)),p=s.fs.map(f=>sm(mat(F[e],f.y))),owners=s.fs.map((f,j)=>mode==="local"||j>0&&(mode==="fullchild"||mode==="adaptive"&&gate[j-1]===g||mode==="static0"&&g===0||mode==="static1"&&g===1)?[e]:[2*g,2*g+1]),q=s.fs.map((f,j)=>sm(mat(owners[j].length===1?F[e]:edge,f.x))),m=p.map((r,i)=>r.map((v,j)=>Math.min(v,q[i][j])));return {p,q,m,owners};}
 function gateFit(F){let scores=Array.from({length:V},()=>[0,0]);for(let s of cal){let a=route(F,s,"shared",[]),b=route(F,s,"fullchild",[]);for(let x=0;x<V;x++)scores[x][s.c>>1]+=a.m[0][x]*(sum(b.m[x+1])-sum(a.m[x+1]));}return scores.map(r=>Math.max(...r)<=0?-1:r[0]>=r[1]?0:1);}
 function anchor(F,data,mode,gate){return data.map(s=>route(F,s,mode,gate).m);}
 function stat(F,data,mode,gate,anc=null,kind="direct",need=false,detach=false){let G=F.map(()=>zeros()),A=0,A0=0,H0=0;
 for(let n=0;n<data.length;n++){let s=data[n],{p,q,m,owners}=route(F,s,mode,gate),a=anc?anc[n]:m,z=m.map(()=>zero(V));for(let i=0;i<V;i++){let child=sum(m[i+1]),oldchild=sum(a[i+1]),w=a[0][i]*(1+oldchild);A+=m[0][i]*(1+child);A0+=w;H0+=w*Math.log(m[0][i]/a[0][i]);z[0][i]=kind==="mm"?w/m[0][i]:1+child;for(let j=0;j<V;j++){let ww=a[0][i]*a[i+1][j];H0+=ww*Math.log(m[i+1][j]/a[i+1][j]);z[i+1][j]=kind==="mm"?ww/m[i+1][j]:m[0][i];}}
 if(need)for(let j=0;j<=V;j++){let gp=z[j].map((v,i)=>p[j][i]<=q[j][i]?v:0),gq=z[j].map((v,i)=>p[j][i]>q[j][i]?v:0);G[s.c]=ma(G[s.c],outer(gradprob(p[j],gp),s.fs[j].y));let qg=outer(gradprob(q[j],gq),s.fs[j].x);if(detach&&owners[j].length>1){G[s.c]=ma(G[s.c],qg);}else for(let e of owners[j])G[e]=ma(G[e],ms(qg,1/owners[j].length));}}
 return {A:A/data.length,bound:A0/data.length*Math.exp(H0/A0),G:G.map(g=>ms(g,4/data.length))};}
 for(let eta of [.02,.1,.4]){
 let F=[copy(W),copy(W),copy(W),copy(W)];for(let t=0;t<12;t++){F=update(F,F.map((w,e)=>ms(ce(w,clients[e]),-1)),eta);formalSteps+=4;}
 let ref=F.map(copy),gate=gateFit(ref),rn=nll(ref,guard),tn=nll(ref,testLabels),modes={};
 const pack=(F,mode,g,it)=>{let t=nll(F,testLabels);return {eal:stat(F,test,mode,g).A,cal:stat(F,cal,mode,g).A,nll:mean(t),maxTestDelta:Math.max(...t.map((v,i)=>v-tn[i])),steps:it};};
 for(let mode of ["shared","static0","static1","adaptive","fullchild","local"])modes[mode]=pack(ref,mode,gate,0);
 let ac=anchor(ref,cal,"adaptive",gate),gg=stat(ref,cal,"adaptive",gate,ac,"direct",true).G,ix=0,jx=0,kx=0,eps=1e-5,B=ref.map(copy),C=ref.map(copy);B[ix][jx][kx]+=eps;C[ix][jx][kx]-=eps;maxFD=Math.max(maxFD,Math.abs(gg[ix][jx][kx]/4-(stat(B,cal,"adaptive",gate).A-stat(C,cal,"adaptive",gate).A)/(2*eps)));
 for(let [mode,kind,detach] of [["adaptive","direct",false],["adaptive","mm",false],["adaptive","direct",true],["local","direct",false],["local","mm",false]]){
 let name=mode+"_"+kind+(detach?"_localgrad":""),anc=anchor(ref,cal,mode,gate),base=stat(ref,cal,mode,gate,anc).A,cur=ref.map(copy),best=modes[mode],bestA=base;
 for(let it=1;it<=4;it++){let g=stat(cur,cal,mode,gate,anc,kind,true,detach).G,nn=Math.sqrt(mean(g.map(x=>norm(x)**2)))||1;cur=update(cur,g.map(x=>ms(x,1/nn)),.01);auxSteps++;let v=stat(cur,cal,mode,gate,anc),ng=nll(cur,guard);checks++;maxBoundViolation=Math.max(maxBoundViolation,v.bound-v.A);
 if(ng.every((x,i)=>x<=rn[i]+.005)&&v.A>bestA&&(kind!=="mm"||v.bound>=base-1e-12)){best=pack(cur,mode,gate,it);bestA=v.A;}}
 modes[name]=best;}
 rows.push({wi,seed,eta,gate,modes});
 }}}
 let summary=[.02,.1,.4].map(eta=>{let r=rows.filter(o=>o.eta===eta),modes={};for(let k of Object.keys(r[0].modes)){let a=r.map(o=>o.modes[k]);modes[k]={eal:mean(a.map(x=>x.eal)),nll:mean(a.map(x=>x.nll)),cal:mean(a.map(x=>x.cal)),accepted:a.filter(x=>x.steps>0).length,testViolations:a.filter(x=>x.maxTestDelta>.005+1e-12).length};}return {eta,modes,adaptiveWins:r.filter(o=>o.modes.adaptive_mm.eal>o.modes.local_direct.eal).length};});
 return {evals,formalSteps,auxSteps,checks,maxBoundViolation,maxFD,summary,rows};}

const fs=require('node:fs');console.log(JSON.stringify(integratedProbe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
