// node joint.js weights.json checkpoint.json input.txt
function probe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}


 const zeros=()=>Array.from({length:V},()=>zero(d)),ma=(A,B)=>A.map((r,i)=>add(r,B[i])),ms=(A,t)=>A.map(r=>scale(r,t)),outer=(a,b)=>a.map(x=>scale(b,x)),sub=(A,B)=>ma(A,ms(B,-1)),tv=(a,b)=>.5*sum(a.map((x,i)=>Math.abs(x-b[i]))),over=(a,b)=>1-tv(a,b),mix=(a,b)=>scale(add(a,b),.5),gradprob=(p,g)=>{let t=dot(p,g);return p.map((v,i)=>v*(g[i]-t))},copy=A=>A.map(r=>r.slice());
 let words=[...new Set(names.toLowerCase().split(/\s+/).filter(w=>/^[a-z]{6,14}$/.test(w)))],rows=[],evals=0,formalSteps=0,draftSteps=0,maxOracle=0,maxFD=0,gridChecks=0,gridGap=0;
 for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);for(let seed of [11,29,73]){
 let pools=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,44));
 function sample(w){let h=[26,...w.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)],f=feature(h);evals++;return {...f,h,label:w.charCodeAt(3)-97}}
 let clients=pools.map(p=>p.slice(0,24).map(sample)),testLabels=pools.map(p=>p.slice(32,44).map(sample));
 let getTree=w=>{let s=sample(w);s.next=Array.from({length:V},(_,i)=>feature(s.h.concat(i)));evals+=V;return s};
 let cal=pools.flatMap(p=>p.slice(24,26).map(getTree)),test=pools.flatMap(p=>p.slice(26,30).map(getTree));
 function fit(eta){let r=[copy(W),copy(W)];for(let t=0;t<20;t++)for(let e=0;e<2;e++){let g=zeros();for(let s of clients[2*e].concat(clients[2*e+1])){let v=sm(mat(r[e],s.y));v[s.label]-=1;g=ma(g,outer(v,s.y))}r[e]=sub(r[e],ms(g,eta/48));formalSteps++}return r}
 function prep(data,P){return data.map(s=>({...s,p:P.map(w=>[sm(mat(w,s.y)),...s.next.map(t=>sm(mat(w,t.y)))])}))}
 function oracle(s){let [p1,p2]=s.p,q0=mix(p1[0],p2[0]),o1=0,o2=0,both=0;for(let i=0;i<V;i++){let a=.5*Math.min(p1[0][i],q0[i]),b=.5*Math.min(p2[0][i],q0[i]),D=tv(p1[i+1],p2[i+1]);o1+=a+b;o2+=a+b-Math.min(a,b)*D;both+=(a+b)*(1-.5*D)}let exact=2-tv(p1[0],p2[0])-.5*sum(p1[0].map((v,i)=>Math.min(v,p2[0][i])*tv(p1[i+1],p2[i+1])));maxOracle=Math.max(maxOracle,Math.abs(o1+o2-exact));return {oracle:exact,meanConditional:o1+both,root:o1}}
 function score(s,U,e=null,needGrad=false,objective="eal"){let xs=[s.x,...s.next.map(t=>t.x)],qs=xs.map(x=>sm(mat(U,x))),grad=zeros(),eal=0,a1=0,gg=xs.map(()=>zero(V));let E=e===null?[0,1]:[e],we=1/E.length;
 for(let k of E){let p=s.p[k];for(let i=0;i<V;i++){let m=Math.min(p[0][i],qs[0][i]),a=over(p[i+1],qs[i+1]);a1+=we*m;eal+=we*m*(1+a);if(needGrad){if(objective==="eal"){gg[0][i]+=we*(qs[0][i]<p[0][i]?1+a:0);for(let j=0;j<V;j++)gg[i+1][j]+=we*m*(qs[i+1][j]<p[i+1][j]?1:0)}else if(objective==="kl"){gg[0][i]+=we*p[0][i]/qs[0][i];for(let j=0;j<V;j++)gg[i+1][j]+=we*p[0][i]*p[i+1][j]/qs[i+1][j]}}}}
 if(needGrad&&objective==="survival"){for(let i=0;i<V;i++){let a=.5*Math.min(s.p[0][0][i],qs[0][i]),b=.5*Math.min(s.p[1][0][i],qs[0][i]),target=a>b?s.p[0][i+1]:b>a?s.p[1][i+1]:mix(s.p[0][i+1],s.p[1][i+1]);gg[0][i]=.5*(s.p[0][0][i]+s.p[1][0][i])/qs[0][i];for(let j=0;j<V;j++)gg[i+1][j]=(a+b)*target[j]/qs[i+1][j]}}
 if(needGrad)for(let i=0;i<xs.length;i++)grad=ma(grad,outer(gradprob(qs[i],gg[i]),xs[i]));return {a1,eal,grad}}
 function train(data,init,obj,e=null){let U=copy(init);for(let it=0;it<10;it++){let g=zeros();for(let s of data)g=ma(g,score(s,U,e,true,obj).grad);U=ma(U,ms(g,.2/data.length));draftSteps++}return U}
 function evalHeads(data,heads){let a=0,b=0;for(let s of data)for(let e=0;e<2;e++){let v=score(s,heads[e],e);a+=v.a1;b+=v.eal}return {a1:a/(2*data.length),eal:b/(2*data.length)}}
 function nll(P){let n=0;for(let e=0;e<2;e++)for(let s of testLabels[2*e].concat(testLabels[2*e+1]))n-=Math.log(sm(mat(P[e],s.y))[s.label]);return n/48}

 for(let eta of [.02,.1,.4]){let P=fit(eta),D=prep(cal,P),init=ms(ma(P[0],P[1]),.5),q=train(D,init,"kl"),modes={};
 let cases=[["ce",0,"root"],["root.1",.1,"root"],["root1",1,"root"],["window.1",.1,"window"],["window1",1,"window"]];
 let maxDecomp=0;
 for(let [name,beta,kind] of cases){let G=ms(ma(P[0],P[1]),.5),R=P.map(w=>sub(w,G)),F=P.map(copy);
 for(let it=0;it<5;it++){let gr=[];
 for(let e=0;e<2;e++){let cg=[];for(let c=2*e;c<2*e+2;c++){let g=zeros();for(let s of clients[c]){let p=sm(mat(F[e],s.y));p[s.label]-=1;g=ma(g,outer(p,s.y))}cg.push(ms(g,1/clients[c].length));formalSteps++}
 let g=ms(ma(cg[0],cg[1]),.5),jgrad=zeros();
 if(beta)for(let s of cal){let xs=[s.y,...s.next.map(t=>t.y)],qs=[s.x,...s.next.map(t=>t.x)].map(x=>sm(mat(q,x))),p=xs.map(x=>sm(mat(F[e],x))),z=xs.map(()=>zero(V));
 for(let i=0;i<V;i++){let m=Math.min(p[0][i],qs[0][i]);z[0][i]+=(p[0][i]<qs[0][i]?(kind==="window"?1+over(p[i+1],qs[i+1]):1):0);if(kind==="window")for(let j=0;j<V;j++)z[i+1][j]+=m*(p[i+1][j]<qs[i+1][j]?1:0)}
 for(let i=0;i<(kind==="window"?xs.length:1);i++)jgrad=ma(jgrad,outer(gradprob(p[i],z[i]),xs[i]));}
 gr.push(sub(g,ms(jgrad,beta/cal.length)));}
 let gg=ms(ma(gr[0],gr[1]),.5);G=sub(G,ms(gg,.1));R=R.map((r,e)=>sub(r,ms(sub(gr[e],gg),.05)));F=R.map(r=>ma(G,r));maxDecomp=Math.max(maxDecomp,...ma(R[0],R[1]).flat().map(Math.abs));
 }
 let ev=evalHeads(prep(test,F),[q,q]),ns=[0,1].map(e=>-mean(testLabels[2*e].concat(testLabels[2*e+1]).map(s=>Math.log(sm(mat(F[e],s.y))[s.label]))));modes[name]={...ev,nll:mean(ns),regionNll:ns,maxDecomp};
 }
 rows.push({wi,seed,eta,nll:nll(P),modes});
 }}}
 let summary=[.02,.1,.4].map(eta=>{let rr=rows.filter(x=>x.eta===eta),modes={};for(let k of Object.keys(rr[0].modes))modes[k]={a1:mean(rr.map(x=>x.modes[k].a1)),eal:mean(rr.map(x=>x.modes[k].eal)),nll:mean(rr.map(x=>x.modes[k].nll)),maxDecomp:Math.max(...rr.map(x=>x.modes[k].maxDecomp))};return {eta,nll:mean(rr.map(x=>x.nll)),modes}});
 return {evals,formalSteps,draftSteps,maxOracle,maxFD,gridChecks,gridGap,summary,rows}
}


const fs=require('node:fs');console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));