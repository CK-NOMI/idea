// R36: node probe.js weights.json checkpoint.json input.txt
function probe(raws,names){
 const V=27,d=16,H=4,k=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},norm=x=>scale(x,1/sum(x)),tv=(a,b)=>sum(a.map((x,i)=>Math.abs(x-b[i])))/2,mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {p:sm(mat(W,seq.at(-1))),q:sm(mat(W,s))}}}
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function cat(cdf,u){let l=0,r=cdf.length-1;while(l<r){let m=(l+r)>>1;if(u<cdf[m])r=m;else l=m+1}return l}
 const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{5,14}$/.test(x)))],map=new Map();for(let x of words){let h=x.slice(0,3),i=(x.charCodeAt(3)-97)*V+x.charCodeAt(4)-97;if(!map.has(h))map.set(h,new Set());map.get(h).add(i)}
 const prefixes=[...map.keys()].filter(h=>map.get(h).size>=4),rows=[];let maxLaw=0,maxBound=0,maxMean=0,draws=0,evals=0;
 for(let wi=0;wi<raws.length;wi++){let run=model(raws[wi]);for(let seed of [11,29,73]){let rr=rng(seed+1000*wi),hs=prefixes.slice().sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,8);
 for(let h of hs){let pre=[26,...h.split("").map(x=>x.charCodeAt(0)-97)],root=run(pre),second=Array.from({length:V},(_,i)=>run(pre.concat(i)));evals+=1+V;let P=[],score=[],R=[],valid=[...map.get(h)].sort((a,b)=>a-b);
 for(let i=0;i<V;i++)for(let j=0;j<V;j++){P.push(root.p[i]*second[i].p[j]);score.push(root.p.map((v,z)=>Number(z===i)+Number(z===j)-v-second[i].p[z]));R.push(Number(map.get(h).has(i*V+j)))}
 let cdf=P.map((_,i)=>sum(P.slice(0,i+1)));for(let mode of ["one","half","all"]){let A=new Set(valid.slice(0,mode==="one"?1:mode==="half"?Math.floor(valid.length/2):valid.length)),a=sum([...A].map(i=>P[i])),mu=zero(V);for(let idx of A)mu=add(mu,scale(score[idx],P[idx]/a));
 let H0=zero(V),Q0=zero(V);for(let idx of A){let i=Math.floor(idx/V),j=idx%V;H0[i]+=second[i].p[j];Q0[i]+=second[i].q[j]}
 let pB=norm(root.p.map((v,i)=>v*(1-H0[i]))),qW=norm(root.q.map((v,i)=>v*(1-H0[i]))),qB=norm(root.q.map((v,i)=>v*(1-Q0[i])));let overlap=[1-tv(pB,root.q),1-tv(pB,qW),1-tv(pB,qB)];
 let local=(p,q,w)=>{let z=sum(p.map((v,i)=>v*w[i])),zq=sum(q.map((v,i)=>v*w[i]));if(!z||!zq)return;let pp=norm(p.map((v,i)=>v*w[i])),qq=norm(q.map((v,i)=>v*w[i]));let bound=(sum(p.map((v,i)=>w[i]*Math.abs(v-q[i])))+Math.abs(z-zq))/(2*Math.max(z,zq));maxBound=Math.max(maxBound,tv(pp,qq)-bound);let delta=tv(pp,qq);for(let j=0;j<V;j++){let m=Math.min(pp[j],qq[j]);let marginal=m+(delta>1e-15?sum(pp.map((v,i)=>Math.max(qq[i]-v,0)))*Math.max(pp[j]-qq[j],0)/delta:0);maxLaw=Math.max(maxLaw,Math.abs(marginal-pp[j]))}};
 local(root.p,root.q,H0.map(x=>1-x));for(let i=0;i<V;i++){let w=Array.from({length:V},(_,j)=>Number(!A.has(i*V+j))),p=second[i].p,q=second[i].q,pp=norm(p.map((v,j)=>v*w[j])),qq=norm(q.map((v,j)=>v*w[j]));local(p,q,w);overlap[0]+=pB[i]*(1-tv(pp,q));overlap[1]+=pB[i]*(1-tv(pp,qq));overlap[2]+=pB[i]*(1-tv(pp,qq))}
 overlap=scale(overlap,.5);let N=8,M=512,s1=zero(V),s2=zero(V),ss1=0,ss2=0,admit=0,known=0;
 for(let rep=0;rep<M;rep++){let ys=Array.from({length:N},()=>cat(cdf,rr())),K=sum(ys.map(i=>R[i]));known+=ys.filter(i=>A.has(i)).length;if(K===0||K===N)continue;admit++;let x=zero(V),y=zero(V),r=K/N,sd=Math.sqrt(r*(1-r));for(let i of ys){let coef=(R[i]-r)/(N*2*sd);x=add(x,scale(score[i],coef));y=add(y,scale(A.has(i)?mu:score[i],coef))}s1=add(s1,x);s2=add(s2,y);ss1+=dot(x,x);ss2+=dot(y,y)}
 draws+=N*M;
 let var1=ss1/M-dot(s1,s1)/(M*M),var2=ss2/M-dot(s2,s2)/(M*M);
 let Rmass=sum(P.filter((_,i)=>R[i])),Z=1-Rmass**N-(1-Rmass)**N;let expectedCondVar=0;
 // exact variance removed by conditioning on known-A identities; sum over multinomial counts m,b
 let covA=sum([...A].map(i=>P[i]/a*dot(score[i].map((v,j)=>v-mu[j]),score[i].map((v,j)=>v-mu[j]))));
 let choose=(n,k)=>{let z=1;for(let i=1;i<=k;i++)z*= (n-i+1)/i;return z};
 for(let m=1;m<=N;m++)for(let b=0;b<=N-m;b++){let K=m+b;if(K===N)continue;let r=K/N,coef=(1-r)/(N*2*Math.sqrt(r*(1-r)));let prob=choose(N,m)*choose(N-m,b)*a**m*(Rmass-a)**b*(1-Rmass)**(N-m-b);expectedCondVar+=prob*m*coef**2*covA}
 let costAR=1+sum(root.p.map(p=>1-(1-p)**N)),Afirst=new Set([...A].map(i=>Math.floor(i/V))),costNew=1+sum(root.p.map((p,i)=>Afirst.has(i)?1:1-(1-p)**N));
 let exactKnown=0;for(let i of A)exactKnown+=P[i];maxMean=Math.max(maxMean,Math.abs(exactKnown-a));
 rows.push({wi,seed,h,mode,size:A.size,a,Rmass,Z,overlap,var1,var2,expectedCondVar,admit,known,teacherNodes:1+Afirst.size,costAR,costNew});
 }}}
 }let summary=["one","half","all"].map(mode=>{let r=rows.filter(x=>x.mode===mode);return {mode,n:r.length,mass:mean(r.map(x=>x.a)),rewardMass:mean(r.map(x=>x.Rmass)),meanSize:mean(r.map(x=>x.size)),overlap:[0,1,2].map(i=>mean(r.map(x=>x.overlap[i]))),worseThanRaw:r.filter(x=>x.overlap[1]<x.overlap[0]-1e-12).length,worseThanQCond:r.filter(x=>x.overlap[1]<x.overlap[2]-1e-12).length,var1:mean(r.map(x=>x.var1)),var2:mean(r.map(x=>x.var2)),exactRemoved:mean(r.map(x=>x.expectedCondVar)),costAR:mean(r.map(x=>x.costAR)),costNew:mean(r.map(x=>x.costNew)),teacherNodes:mean(r.map(x=>x.teacherNodes)),known:sum(r.map(x=>x.known)),admit:sum(r.map(x=>x.admit))}});return {evals,draws,maxLaw,maxBound,maxMean,summary,rows}
}
const fs=require('node:fs'); console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
