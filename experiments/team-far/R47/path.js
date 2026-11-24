function pathProbe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}


 const zeros=()=>Array.from({length:V},()=>zero(d)),ma=(A,B)=>A.map((r,i)=>add(r,B[i])),ms=(A,t)=>A.map(r=>scale(r,t)),outer=(a,b)=>a.map(x=>scale(b,x)),sub=(A,B)=>ma(A,ms(B,-1)),tv=(a,b)=>.5*sum(a.map((x,i)=>Math.abs(x-b[i]))),over=(a,b)=>1-tv(a,b),mix=(a,b)=>scale(add(a,b),.5),gradprob=(p,g)=>{let t=dot(p,g);return p.map((v,i)=>v*(g[i]-t))},copy=A=>A.map(r=>r.slice());


 const norm=A=>Math.sqrt(sum(A.flat().map(v=>v*v)));
 let words=[...new Set(names.toLowerCase().split(/\s+/).filter(w=>/^[a-z]{6,14}$/.test(w)))],rows=[],evals=0,formalSteps=0,atoms=0,samples=0,maxMeanErr=0,maxVarViolation=0,maxRangeViolation=0;
 let random=rng(47001);const pick=p=>{let u=random();for(let i=0;i<p.length;i++){u-=p[i];if(u<=0)return i;}return p.length-1;};
 for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);for(let seed of [11,29,73]){
 let pools=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,48));
 function sample(w){let h=[26,...w.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)],f=feature(h);evals++;return {...f,h,label:w.charCodeAt(3)-97}}
 let clients=pools.map(p=>p.slice(0,24).map(sample)),guard=pools.map(p=>p.slice(24,30).map(sample)),testLabels=pools.map(p=>p.slice(34,44).map(sample));
 let getTree=w=>{let s=sample(w);s.fs=[s,...Array.from({length:V},(_,i)=>feature(s.h.concat(i)))];evals+=V;return s};
 let cal=pools.flatMap(p=>p.slice(30,31).map(getTree)),test=pools.flatMap(p=>p.slice(31,33).map(getTree));
 function ce(F,cs){let g=zeros();for(let s of cs){let p=sm(mat(F,s.y));p[s.label]-=1;g=ma(g,outer(p,s.y));}return ms(g,1/cs.length);}
 function update(F,g,lr){let gg=ms(ma(g[0],g[1]),.5);return F.map((w,e)=>ma(w,ms(ma(ms(gg,.5),ms(g[e],.5)),lr)));}
 function nll(F,data){return data.map((cs,c)=>-mean(cs.map(s=>Math.log(sm(mat(F[c>>1],s.y))[s.label]))));}
 function probs(F,s){let p=s.fs.map(f=>sm(mat(F,f.y))),q=s.fs.map(f=>sm(mat(F,f.x))),m=p.map((r,i)=>r.map((v,j)=>Math.min(v,q[i][j])));return {p,q,m};}
 function grad(F){return F.map(w=>{let g=zeros();for(let s of cal){let {p,q,m}=probs(w,s),z=m.map(()=>zero(V));for(let i=0;i<V;i++){z[0][i]=1+sum(m[i+1]);for(let j=0;j<V;j++)z[i+1][j]=m[0][i];}for(let j=0;j<=V;j++){let gp=z[j].map((v,i)=>p[j][i]<=q[j][i]?v:0),gq=z[j].map((v,i)=>p[j][i]>q[j][i]?v:0);g=ma(g,ma(outer(gradprob(p[j],gp),s.fs[j].y),outer(gradprob(q[j],gq),s.fs[j].x)));}}return ms(g,1/cal.length)});}
 function moments(F,N,s){let a=probs(F,s),b=probs(N,s),mu=0,dm=0,hard2=0,soft2=0,last2=0,direct2=0,cases=[],L=0;
 for(let f of s.fs)for(let x of [f.x,f.y]){let dz=mat(sub(N,F),x);L=Math.max(L,Math.max(...dz)-Math.min(...dz));}
 for(let i=0;i<V;i++){let q=a.q[0][i],m=a.m[0][i],al=m/q,r1=Math.log(b.m[0][i]/m),second=0,newchild=sum(b.m[i+1]),oldchild=sum(a.m[i+1]);
 for(let j=0;j<V;j++){let cj=a.q[i+1][j],n=a.m[i+1][j],be=n/cj,r2=r1+Math.log(b.m[i+1][j]/n),both=r1+r2;
 let soft=al*(r1+be*r2);
 mu+=m*cj*(r1+be*r2);hard2+=q*cj*al*((1-be)*r1*r1+be*both*both);soft2+=q*cj*soft*soft;
 second+=n*r2;atoms++;cases.push({i,j,q:q*cj,al,be,r1,r2,soft});}
 let last=al*(r1+second),direct=(b.m[0][i]*(1+newchild)-m*(1+oldchild))/q;last2+=q*last*last;direct2+=q*direct*direct;dm+=q*direct;
 for(let c of cases.slice(-V)){c.last=last;c.direct=direct;maxRangeViolation=Math.max(maxRangeViolation,Math.abs(c.soft)-3*L,Math.abs(last)-3*L,Math.abs(c.r1+c.r2)-3*L);}
 }
 let softMean=sum(cases.map(c=>c.q*c.soft)),lastMean=sum(cases.map(c=>c.q*c.last));maxMeanErr=Math.max(maxMeanErr,Math.abs(mu-softMean),Math.abs(mu-lastMean));maxVarViolation=Math.max(maxVarViolation,soft2-hard2,last2-soft2);
 let mc=[0,0,0,0];const qq=cases.map(c=>c.q);for(let k=0;k<128;k++){let c=cases[pick(qq)],u=random(),v=random();let hard=u<c.al?c.r1+(v<c.be?c.r2:0):0;[hard,c.soft,c.last,c.direct].forEach((z,j)=>mc[j]+=z/128);samples++;}
 let A0=sum(a.m[0].map((x,i)=>x*(1+sum(a.m[i+1]))));
 return {A0,mu,dm,L,varHard:hard2-mu*mu,varSoft:soft2-mu*mu,varLast:last2-mu*mu,varDirect:direct2-dm*dm,mc};}
 for(let eta of [.02,.1,.4]){let F=[copy(W),copy(W)];for(let t=0;t<12;t++){let g=F.map((w,e)=>ms(ma(ce(w,clients[2*e]),ce(w,clients[2*e+1])),.5));F=update(F,g.map(x=>ms(x,-1)),eta);formalSteps+=4;}
 let g=grad(F),nn=Math.sqrt(mean(g.map(x=>norm(x)**2)))||1,N=update(F,g.map(x=>ms(x,1/nn)),.04);
 let stats=F.flatMap((w,e)=>test.map(s=>moments(w,N[e],s))),a0=mean(stats.map(x=>x.A0)),mu=mean(stats.map(x=>x.mu)),dm=mean(stats.map(x=>x.dm)),L=Math.max(...stats.map(x=>x.L)),refN=nll(F,guard),refT=nll(F,testLabels),n=nll(N,guard),nt=nll(N,testLabels);
 // Stratified fixed-context mean: identical number of samples per context.
 let vv=Object.fromEntries(["varHard","varSoft","varLast","varDirect"].map(k=>[k,mean(stats.map(x=>x[k]))]));
 let mc=Array.from({length:4},(_,j)=>mean(stats.map(x=>x.mc[j])));
 let delta=.05/18,required=mu>0?Math.ceil((6*L)**2*Math.log(1/delta)/(2*mu*mu)):null;
 rows.push({wi,seed,eta,a0,mu,dm,L,certificate:a0*Math.exp(mu/a0)-a0,...vv,mc,required,guardPass:n.every((x,i)=>x<=refN[i]+.005),testPass:nt.every((x,i)=>x<=refT[i]+.005),maxTestDelta:Math.max(...nt.map((x,i)=>x-refT[i]))});
 }}}
 let summary=[.02,.1,.4].map(eta=>{let r=rows.filter(o=>o.eta===eta);let keys=["a0","mu","dm","L","certificate","varHard","varSoft","varLast","varDirect"];let a=Object.fromEntries(keys.map(k=>[k,mean(r.map(o=>o[k]))]));return {eta,...a,softRatio:a.varSoft/a.varHard,lastRatio:a.varLast/a.varHard,guard:r.filter(o=>o.guardPass).length,test:r.filter(o=>o.testPass).length,positive:r.filter(o=>o.mu>0).length,minN:r.filter(o=>o.required!==null).map(o=>o.required).sort((a,b)=>a-b)[0]??null,medianN:r.filter(o=>o.required!==null).map(o=>o.required).sort((a,b)=>a-b)[Math.floor(r.filter(o=>o.required!==null).length/2)]??null,rmse:[0,1,2,3].map(j=>Math.sqrt(mean(r.map(o=>(o.mc[j]-(j===3?o.dm:o.mu))**2))))}});
 return {evals,formalSteps,atoms,samples,maxMeanErr,maxVarViolation,maxRangeViolation,summary,rows};}

const fs=require('node:fs');console.log(JSON.stringify(pathProbe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
