// node probe.js weights.json checkpoint.json input.txt
function probe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}


 const zeros=(n,m)=>Array.from({length:n},()=>zero(m)),ma=(a,b)=>a.map((x,i)=>add(x,b[i])),ms=(a,s)=>a.map(x=>scale(x,s)),outer=(x,y)=>x.map(v=>scale(y,v)),sub=(a,b)=>ma(a,ms(b,-1)),overlap=(p,q)=>sum(p.map((v,i)=>Math.min(v,q[i])));
 const bfbuf=new ArrayBuffer(4),f32=new Float32Array(bfbuf),u32=new Uint32Array(bfbuf);const bf=x=>{f32[0]=x;let bits=u32[0];return ((bits+0x7fff+((bits>>>16)&1))>>>16)&65535};
 const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{6,14}$/.test(x)))],rows=[];let evals=0,steps=0,versions=0,violations=0,maxGradError=0;
 const block=16,K=7;
 function scales(w){return Array.from({length:Math.ceil(w.length/block)},(_,i)=>Math.max(1e-12,...w.slice(i*block,(i+1)*block).map(Math.abs))/K)}
 function quant(w,s){return w.map((v,i)=>Math.max(-K,Math.min(K,Math.round(v/s[Math.floor(i/block)]))))}
 function decode(q,s){return Array.from({length:V},(_,j)=>Array.from({length:d},(_,i)=>q[j*d+i]*s[Math.floor((j*d+i)/block)]))}
 function margin(w,q,s,i){let l=i*block,r=Math.min(l+block,w.length),m=Infinity;for(let j=l;j<r;j++){let k=q[j],v=w[j],a=k===-K?Infinity:v-(k-.5)*s[i],b=k===K?Infinity:(k+.5)*s[i]-v;m=Math.min(m,a,b)}return m}
 function fit(w,dat,eta){let g=zeros(V,d);for(let s of dat){let p=sm(mat(w,s.y));p[s.label]-=1;g=ma(g,outer(p,s.y))}steps++;return sub(w,ms(g,eta/dat.length))}
 for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);for(let seed of [11,29,73]){
 let clients=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,64));
 const sample=w=>{let p=[26,...w.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)],f=feature(p);evals++;return {p,...f,label:w.charCodeAt(3)-97}};
 let train=clients.map(c=>c.slice(0,48).map(sample)),test=clients.flatMap(c=>c.slice(48).map(sample));for(let s of test){s.next=Array.from({length:V},(_,i)=>feature(s.p.concat(i)));evals+=V}
 for(let eta of [.0005,.005,.05]){let w=W.map(x=>x.slice()),fixed=scales(w.flat()),q0=quant(w.flat(),fixed),stale=decode(q0,fixed),rec=[];
 for(let round=0;round<12;round++){
 let old=w.flat(),oldQ=quant(old,fixed),os=scales(old),oqd=quant(old,os);
 let region=[0,1].map(e=>{let a=w.map(x=>x.slice());for(let l=0;l<2;l++)a=ms(ma(fit(a,train[2*e],eta),fit(a,train[2*e+1],eta)),.5);return a});
 let nw=ms(ma(region[0],region[1]),.5),next=nw.flat(),newQ=quant(next,fixed),ns=scales(next),nqd=quant(next,ns);
 let N=fixed.length,cert=0,actual=0,globalCert=0,union=0,changedCode=0,changedBf=0,dynamic=0,codeDyn=0,miss=0;
 let R=region.map(r=>r.flat().map((v,i)=>v-old[i])),dd=next.map((v,i)=>v-old[i]);
 for(let i=0;i<N;i++){let a=i*block,b=Math.min(a+block,next.length),m=margin(old,oldQ,fixed,i),bound=.5*R.reduce((v,r)=>v+Math.max(...r.slice(a,b).map(Math.abs)),0),gb=Math.max(...dd.slice(a,b).map(Math.abs)),ok=bound<m,go=gb<m,ch=oldQ.slice(a,b).some((v,j)=>v!==newQ[a+j]);cert+=Number(ok);globalCert+=Number(go);actual+=Number(!ch);union+=Number(R.some(r=>r.slice(a,b).some((v,j)=>quant([old[a+j]+v],[fixed[i]])[0]!==oldQ[a+j])));if(ok&&ch)violations++;if(go&&ch)violations++;dynamic+=Number(os[i]===ns[i]&&!oqd.slice(a,b).some((v,j)=>v!==nqd[a+j]));miss+=Number(ch&&!R.some(r=>r.slice(a,b).some((v,j)=>quant([old[a+j]+v],[fixed[i]])[0]!==oldQ[a+j])));}
 for(let i=0;i<next.length;i++){changedCode+=Number(oldQ[i]!==newQ[i]);changedBf+=Number(bf(old[i])!==bf(next[i]));codeDyn+=Number(oqd[i]!==nqd[i])}
 // compare actual exact quantized q (fixed) with previous or original q, dynamic q, and unquantized shallow
 const heads={fresh:decode(newQ,fixed),previous:decode(oldQ,fixed),stale,dynamic:decode(nqd,ns),float:nw};let scores={};
 for(let [name,U] of Object.entries(heads)){let a1=0,eal=0,nll=0;for(let s of test){let p=sm(mat(nw,s.y)),q=sm(mat(U,s.x)),m=p.map((v,i)=>Math.min(v,q[i])),a=sum(m);a1+=a;eal+=a+sum(m.map((v,i)=>v*overlap(sm(mat(nw,s.next[i].y)),sm(mat(U,s.next[i].x)))));nll-=Math.log(p[s.label])}scores[name]={a1:a1/test.length,eal:eal/test.length,nll:nll/test.length}}
 rec.push({round,cert:cert/N,globalCert:globalCert/N,actual:actual/N,dynamic:dynamic/N,union:union/N,unionMiss:miss,changedCode:changedCode/next.length,changedBf:changedBf/next.length,codeDyn:codeDyn/next.length,scores});w=nw;versions++;
 }
 rows.push({wi,seed,eta,rec});
 }}}
 let summary=[.0005,.005,.05].map(eta=>{let rr=rows.filter(x=>x.eta===eta).flatMap(x=>x.rec),out={eta,n:rr.length};for(let k of ["cert","globalCert","actual","dynamic","union","unionMiss","changedCode","changedBf","codeDyn"])out[k]=mean(rr.map(x=>x[k]));out.scores={};for(let k of Object.keys(rr[0].scores)){out.scores[k]={};for(let f of ["a1","eal","nll"])out.scores[k][f]=mean(rr.map(x=>x.scores[k][f]))}return out});
 return {evals,steps,versions,violations,summary,rows}
}

const fs=require('node:fs');console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));