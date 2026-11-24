function scmProbe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}


 const zeros=()=>Array.from({length:V},()=>zero(d)),ma=(A,B)=>A.map((r,i)=>add(r,B[i])),ms=(A,t)=>A.map(r=>scale(r,t)),outer=(a,b)=>a.map(x=>scale(b,x)),sub=(A,B)=>ma(A,ms(B,-1)),tv=(a,b)=>.5*sum(a.map((x,i)=>Math.abs(x-b[i]))),over=(a,b)=>1-tv(a,b),mix=(a,b)=>scale(add(a,b),.5),gradprob=(p,g)=>{let t=dot(p,g);return p.map((v,i)=>v*(g[i]-t))},copy=A=>A.map(r=>r.slice());

 const norm=A=>Math.sqrt(sum(A.flat().map(v=>v*v)));
 let words=[...new Set(names.toLowerCase().split(/\s+/).filter(w=>/^[a-z]{6,14}$/.test(w)))],rows=[],evals=0,formalSteps=0,auxSteps=0,checks=0,violations=0,worst=0,tangentError=0;
 for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);for(let seed of [11,29,73]){
 let pools=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,48));
 function sample(w){let h=[26,...w.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)],f=feature(h);evals++;return {...f,h,label:w.charCodeAt(3)-97}}
 let clients=pools.map(p=>p.slice(0,24).map(sample)),guard=pools.map(p=>p.slice(24,30).map(sample)),testLabels=pools.map(p=>p.slice(34,44).map(sample));
 let getTree=w=>{let s=sample(w);s.fs=[s,...Array.from({length:V},(_,i)=>feature(s.h.concat(i)))];evals+=V;return s};
 let cal=pools.flatMap(p=>p.slice(30,31).map(getTree)),test=pools.flatMap(p=>p.slice(31,33).map(getTree));
 function ce(F,cs){let g=zeros();for(let s of cs){let p=sm(mat(F,s.y));p[s.label]-=1;g=ma(g,outer(p,s.y));}return ms(g,1/cs.length);}
 function update(F,g,lr=.1){let gg=ms(ma(g[0],g[1]),.5);return F.map((w,e)=>ma(w,ms(ma(ms(gg,.5),ms(g[e],.5)),lr)));}
 function nll(F,data){return data.map((cs,c)=>-mean(cs.map(s=>Math.log(sm(mat(F[c>>1],s.y))[s.label]))));}
 function probs(F,s){let p=s.fs.map(f=>sm(mat(F,f.y))),q=s.fs.map(f=>sm(mat(F,f.x))),m=p.map((r,i)=>r.map((v,j)=>Math.min(v,q[i][j])));return {p,q,m};}
 function anchor(F,data){return F.map(w=>data.map(s=>probs(w,s).m));}
 function stats(F,data,anc,mode,grad=false){let G=F.map(()=>zeros()),As=[0,0],S=[0,0],A0=[0,0];
 for(let e=0;e<2;e++)for(let n=0;n<data.length;n++){let s=data[n],{p,q,m}=probs(F[e],s),a=anc[e][n],z=m.map(()=>zero(V)),root=0,total=0,base=0,loggain=0;
 for(let i=0;i<V;i++){let child=sum(m[i+1]),oldchild=sum(a[i+1]),w=a[0][i]*(1+oldchild);root+=m[0][i];total+=m[0][i]*(1+child);base+=w;loggain+=w*Math.log(m[0][i]/a[0][i]);z[0][i]=mode==="mm"?w/m[0][i]:mode==="root"?1:1+child;
 for(let j=0;j<V;j++){let ww=a[0][i]*a[i+1][j];loggain+=ww*Math.log(m[i+1][j]/a[i+1][j]);z[i+1][j]=mode==="mm"?ww/m[i+1][j]:mode==="root"?0:m[0][i];}}
 As[e]+=total;S[e]+=loggain;A0[e]+=base;
 if(grad)for(let j=0;j<=V;j++){let gp=z[j].map((v,i)=>p[j][i]<=q[j][i]?v:0),gq=z[j].map((v,i)=>p[j][i]>q[j][i]?v:0);G[e]=ma(G[e],ma(outer(gradprob(p[j],gp),s.fs[j].y),outer(gradprob(q[j],gq),s.fs[j].x)));}}
 let A=mean(As)/data.length,B=mean(A0)/data.length,S0=mean(S)/data.length,bound=B*Math.exp(S0/B);
 return {A,B,bound,G:G.map(g=>ms(g,1/data.length)),region:As.map(a=>a/data.length),regionBounds:A0.map((b,e)=>b/data.length*Math.exp(S[e]/b))};}
 for(let eta of [.02,.1,.4]){
 let F=[copy(W),copy(W)];for(let t=0;t<12;t++){let g=F.map((w,e)=>ms(ma(ce(w,clients[2*e]),ce(w,clients[2*e+1])),.5));F=update(F,g.map(x=>ms(x,-1)),eta);formalSteps+=4;}
 let ref=F.map(copy),anc=anchor(ref,cal),ancT=anchor(ref,test),refN=nll(ref,guard),refTest=nll(ref,testLabels),base=stats(ref,cal,anc,"window"),baseT=stats(ref,test,ancT,"window"),modes={ce:{eal:baseT.A,guard:mean(refN),nll:mean(refTest),maxTestDelta:0,cal:base.A,bound:base.A,steps:0,regionTest:refTest}};
 let g1=stats(ref,cal,anc,"window",true).G,g2=stats(ref,cal,anc,"mm",true).G;tangentError=Math.max(tangentError,...g1.map((g,e)=>norm(sub(g,g2[e]))));
 for(let kind of ["root","window","mm"]){let cur=ref.map(copy),best=null,cand=[];for(let it=1;it<=6;it++){
 let g=stats(cur,cal,anc,kind,true).G,nn=Math.sqrt(mean(g.map(x=>norm(x)**2)))||1;cur=update(cur,g.map(x=>ms(x,1/nn)),.08);auxSteps++;let v=stats(cur,cal,anc,kind),loss=nll(cur,guard),ok=loss.every((x,i)=>x<=refN[i]+.005);checks++;worst=Math.max(worst,v.bound-v.A,...v.regionBounds.map((x,i)=>x-v.region[i]));if(v.bound>v.A+1e-10||v.regionBounds.some((x,i)=>x>v.region[i]+1e-10))violations++;
 if(ok){cand.push({cur:cur.map(copy),v,it});}
 }
 let select=(key,cert)=>{let b=cand.filter(c=>!cert||c.v.bound>=base.A-1e-12).sort((a,b)=>b.v[key]-a.v[key])[0];if(!b||b.v.A<base.A)return modes.ce;let tn=nll(b.cur,testLabels);return {eal:stats(b.cur,test,ancT,kind).A,guard:mean(nll(b.cur,guard)),nll:mean(tn),maxTestDelta:Math.max(...tn.map((x,i)=>x-refTest[i])),cal:b.v.A,bound:b.v.bound,steps:b.it,regionTest:tn};};
 modes[kind]=select("A",false);if(kind==="mm")modes.mmCert=select("bound",true);
 }
 rows.push({wi,seed,eta,modes});
 }}}
 let summary=[.02,.1,.4].map(eta=>{let r=rows.filter(o=>o.eta===eta),modes={};for(let k of Object.keys(r[0].modes)){let a=r.map(o=>o.modes[k]);modes[k]={eal:mean(a.map(x=>x.eal)),nll:mean(a.map(x=>x.nll)),guard:mean(a.map(x=>x.guard)),cal:mean(a.map(x=>x.cal)),bound:mean(a.map(x=>x.bound)),accepted:a.filter(x=>x.steps>0).length,testViolations:a.filter(x=>x.maxTestDelta>.005+1e-12).length};}return {eta,modes};});
 return {evals,formalSteps,auxSteps,checks,violations,worst,tangentError,summary,rows};}

const fs=require('node:fs');console.log(JSON.stringify(scmProbe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
