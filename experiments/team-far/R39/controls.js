// node controls.js weights.json checkpoint.json input.txt
function probe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}

 const tr=A=>A[0].map((_,i)=>A.map(r=>r[i])),mm=(A,B)=>{let T=tr(B);return A.map(r=>T.map(c=>dot(r,c)))},ma=(A,B)=>A.map((r,i)=>add(r,B[i])),ms=(A,t)=>A.map(r=>scale(r,t)),sub=(A,B)=>ma(A,ms(B,-1)),eye=()=>Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>+(i===j))),zeros=(n=d,m=d)=>Array.from({length:n},()=>zero(m)),outer=(x,y)=>x.map(v=>scale(y,v)),fro=A=>sum(A.map(r=>dot(r,r))),inv=A=>{let a=A.map((r,i)=>r.concat(eye()[i]));for(let i=0;i<d;i++){let j=i;for(let k=i+1;k<d;k++)if(Math.abs(a[k][i])>Math.abs(a[j][i]))j=k;[a[i],a[j]]=[a[j],a[i]];a[i]=scale(a[i],1/a[i][i]);for(let k=0;k<d;k++)if(k!==i)a[k]=add(a[k],scale(a[i],-a[k][i]))}return a.map(r=>r.slice(d))},maxdiff=(A,B)=>Math.max(...sub(A,B).flat().map(Math.abs)),overlap=(p,q)=>sum(p.map((v,i)=>Math.min(v,q[i])));
 const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{6,14}$/.test(x)))],rows=[];let evals=0,steps=0,maxNormal=0,maxReduced=0,maxFD=0,maxAggregate=0;
 for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);for(let seed of [11,29,73]){
 let clients=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,60));
 let sample=w=>{let pre=[26,...w.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)],f=feature(pre);evals++;return {pre,...f,label:w.charCodeAt(3)-97}};
 let train=clients.map(c=>c.slice(0,40).map(sample)),test=clients.map(c=>c.slice(40).map(sample));
 for(let c of test)for(let s of c){s.next=Array.from({length:V},(_,i)=>feature(s.pre.concat(i)));evals+=V}
 let mom=xs=>{let G=zeros(),H=zeros(),Y=zeros();for(let s of xs){G=ma(G,outer(s.x,s.x));H=ma(H,outer(s.x,s.y));Y=ma(Y,outer(s.y,s.y))}return {G:ms(G,1/xs.length),H:ms(H,1/xs.length),Y:ms(Y,1/xs.length)}};
 let m=train.map(mom),regions=[train[0].concat(train[1]),train[2].concat(train[3])],rm=regions.map(mom),glob=mom(train.flat());maxAggregate=Math.max(maxAggregate,maxdiff(glob.G,ms(ma(rm[0].G,rm[1].G),.5)),maxdiff(glob.H,ms(ma(rm[0].H,rm[1].H),.5)));
 const lambda=.1,gamma=.01,I=eye(),local=rm.map(t=>mm(inv(ma(t.G,ms(I,lambda))),t.H)),global=mm(inv(ma(glob.G,ms(I,lambda))),glob.H);
 let A=rm.map(t=>inv(ma(t.G,ms(I,lambda)))),P=ma(ms(I,gamma),ms(sub(I,ms(ma(A[0],A[1]),lambda*.5)),lambda)),rhs=ms(ma(mm(A[0],rm[0].H),mm(A[1],rm[1].H)),lambda*.5),M0=mm(inv(P),rhs),M=rm.map((t,i)=>mm(A[i],ma(t.H,ms(M0,lambda))));
 for(let i=0;i<2;i++)maxNormal=Math.max(maxNormal,maxdiff(mm(ma(rm[i].G,ms(I,lambda)),M[i]),ma(rm[i].H,ms(M0,lambda))));maxNormal=Math.max(maxNormal,maxdiff(ms(M0,lambda+gamma),ms(ma(M[0],M[1]),lambda*.5)));
 let S=ms(mm(tr(M0),M0),gamma);for(let i=0;i<2;i++){let R=zeros();for(let s of regions[i]){let err=add(s.y,scale(mat(tr(M[i]),s.x),-1));R=ma(R,outer(err,err))}S=ma(S,ms(ma(ms(R,1/regions[i].length),ms(mm(tr(sub(M[i],M0)),sub(M[i],M0)),lambda)),.5))}
 const value=(W,M0,M)=>{let t=gamma*fro(mm(M0,tr(W)));for(let i=0;i<2;i++){t+=.5*lambda*fro(mm(sub(M[i],M0),tr(W)));t+=.5*mean(regions[i].map(s=>{let z=mat(W,add(s.y,scale(mat(tr(M[i]),s.x),-1)));return dot(z,z)}))}return t};
 let reduced=fro([]);reduced=sum(W.map(w=>dot(w,mat(S,w))));maxReduced=Math.max(maxReduced,Math.abs(reduced-value(W,M0,M)));
 let plus=W.map(r=>r.slice()),minus=W.map(r=>r.slice());plus[0][0]+=1e-5;minus[0][0]-=1e-5;maxFD=Math.max(maxFD,Math.abs((value(plus,M0,M)-value(minus,M0,M))/2e-5-2*mat(S,W[0])[0]));
 function ceGrad(W,xs){let g=zeros(V,d);for(let s of xs){let p=sm(mat(W,s.y));p[s.label]-=1;g=ma(g,outer(p,s.y))}return ms(g,1/xs.length)}
 function trainW(beta){let w=W.map(r=>r.slice());for(let it=0;it<60;it++){let gs=train.map(x=>ceGrad(w,x)),g=ms(ma(ma(gs[0],gs[1]),ma(gs[2],gs[3])),.25);if(beta)g=ma(g,ms(w.map(v=>mat(S,v)),2*beta/V));w=sub(w,ms(g,.15));steps+=4}return w}
 function evaluate(w,bridges){let nll=0,acc=0,a1=0,a2=0,mse=0,n=0;for(let c=0;c<4;c++){let B=bridges[Math.floor(c/2)],bt=tr(B);for(let s of test[c]){let p=sm(mat(w,s.y)),q=sm(mat(w,mat(bt,s.x)));nll-=Math.log(p[s.label]);acc+=Number(p.indexOf(Math.max(...p))===s.label);let l=p.map((v,i)=>Math.min(v,q[i]));a1+=sum(l);a2+=sum(l.map((v,i)=>{let f=s.next[i];return v*overlap(sm(mat(w,f.y)),sm(mat(w,mat(bt,f.x))))}));mse+=dot(add(mat(w,s.y),scale(mat(w,mat(bt,s.x)),-1)),add(mat(w,s.y),scale(mat(w,mat(bt,s.x)),-1)))/V;n++}}return {nll:nll/n,acc:acc/n,a1:a1/n,a2:a2/n,eal:(a1+a2)/n,mse:mse/n}}

 let w0=trainW(0),modes={tied:evaluate(w0,[I,I]),hier:evaluate(w0,M)};
 // compare same initial gradient norm regularization; same frozen M for all runs
 let scaleL2=Math.sqrt(fro(W.map(v=>mat(S,v)))/fro(W));
 function l2Train(beta){let w=W.map(r=>r.slice());for(let it=0;it<60;it++){let gs=train.map(x=>ceGrad(w,x)),g=ms(ma(ma(gs[0],gs[1]),ma(gs[2],gs[3])),.25);g=ma(g,ms(w,2*beta*scaleL2/V));w=sub(w,ms(g,.15));steps+=4}return w}
 for(let beta of [.1,1,10]){modes["joint"+beta]=evaluate(trainW(beta),M);modes["l2"+beta]=evaluate(l2Train(beta),M)}
 // Independent head KL fitting, teacher probabilities detached, same features and target w0.
 function distill(xs,init,l){let u=init.map(r=>r.slice());for(let it=0;it<100;it++){let g=zeros(V,d);for(let s of xs){let pp=sm(mat(w0,s.y)),qq=sm(mat(u,s.x));g=ma(g,outer(add(qq,scale(pp,-1)),s.x))}g=ma(ms(g,1/xs.length),ms(sub(u,w0),l));u=sub(u,ms(g,.1))}return u}
 function evaluateHeads(w,heads){let nll=0,acc=0,a1=0,a2=0,mse=0,n=0;for(let c=0;c<4;c++){let U=heads[Math.floor(c/2)];for(let s of test[c]){let p=sm(mat(w,s.y)),q=sm(mat(U,s.x));nll-=Math.log(p[s.label]);acc+=Number(p.indexOf(Math.max(...p))===s.label);let l=p.map((v,i)=>Math.min(v,q[i]));a1+=sum(l);a2+=sum(l.map((v,i)=>{let f=s.next[i];return v*overlap(sm(mat(w,f.y)),sm(mat(U,f.x)))}));let delta=add(mat(w,s.y),scale(mat(U,s.x),-1));mse+=dot(delta,delta)/V;n++}}return {nll:nll/n,acc:acc/n,a1:a1/n,a2:a2/n,eal:(a1+a2)/n,mse:mse/n}}
 const cal=train.map(c=>c.slice(0,30)),val=train.map(c=>c.slice(30)),rCal=[cal[0].concat(cal[1]),cal[2].concat(cal[3])];
 function vscore(heads){let t=0,n=0;for(let c=0;c<4;c++)for(let s of val[c]){t+=overlap(sm(mat(w0,s.y)),sm(mat(heads[Math.floor(c/2)],s.x)));n++}return t/n}
 let chosen={};for(let kind of ["globalKL","localKL"]){let best=-1,config;for(let l of [0,.01,.1,1]){let u=kind==="globalKL"?distill(cal.flat(),w0,l):null,heads=u?[u,u]:rCal.map(xs=>distill(xs,w0,l));let s=vscore(heads);if(s>best){best=s;config=l}}let u=kind==="globalKL"?distill(train.flat(),w0,config):null;modes[kind]=evaluateHeads(w0,u?[u,u]:regions.map(xs=>distill(xs,w0,config)));chosen[kind]=config}
 rows.push({wi,seed,chosen,scaleL2,modes});
 }}
 let summary={};for(let k of Object.keys(rows[0].modes)){summary[k]={};for(let v of Object.keys(rows[0].modes[k]))summary[k][v]=mean(rows.map(r=>r.modes[k][v]))}
 return {evals,steps,maxNormal,maxReduced,maxFD,maxAggregate,summary,rows}
}


const fs=require('node:fs');console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));