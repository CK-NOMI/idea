// node probe.js weights.json checkpoint.json input.txt
function probe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {p:sm(mat(W,seq.at(-1))),q:sm(mat(W,s))}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
 let map=new Map();for(let x of new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{5,14}$/.test(x)))){let h=x.slice(0,3),i=(x.charCodeAt(3)-97)*V+x.charCodeAt(4)-97;if(!map.has(h))map.set(h,new Set());map.get(h).add(i)}
 let prefixes=[...map.keys()].filter(h=>map.get(h).size>=8),rows=[],evals=0,draws=0,maxLaw=0,maxCostIdentity=0;
 for(let wi=0;wi<raws.length;wi++){let run=model(raws[wi]);for(let seed of [11,29,73])for(let h of prefixes.slice().sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,8)){
 let pre=[26,...h.split("").map(x=>x.charCodeAt(0)-97)],root=run(pre),second=Array.from({length:V},(_,i)=>run(pre.concat(i)));evals+=1+V;
 let paths=[];for(let i=0;i<V;i++)for(let j=0;j<V;j++){let id=i*V+j,p=root.p[i]*second[i].p[j],s=root.p.map((v,z)=>Number(z===i)+Number(z===j)-v-second[i].p[z]);paths.push({i,j,id,p,s,r:Number(map.get(h).has(id))})}

 let pi=sum(paths.map(x=>x.p*x.r));
 const norm=a=>{let z=sum(a);return a.map(x=>x/z)},overlap=(a,b)=>sum(a.map((x,i)=>Math.min(x,b[i]))),cat=(a,u)=>{let t=0;for(let i=0;i<a.length;i++){t+=a[i];if(u<t)return i}return a.length-1};
 for(let dup of [0,.3,.7]){
 let rr=rng(hash(h,seed+wi*101+Math.round(dup*100))),memb=paths.map(x=>{let v=[0,0,0,0];if(x.r){let owner=Math.floor(rr()*4);v[owner]=1;for(let e=0;e<4;e++)if(e!==owner&&rr()<dup)v[e]=1}return v});
 let owners=Array.from({length:V},(_,i)=>{let a=Array.from({length:4},(_,e)=>sum(paths.slice(i*V,(i+1)*V).map(x=>x.p*memb[x.id][e])));return a.indexOf(Math.max(...a))});
 let r=paths.map(x=>memb[x.id][owners[x.i]]),certMass=sum(paths.map(x=>x.p*r[x.id])),Z=1-certMass;
 let z=Array.from({length:V},(_,i)=>sum(second[i].p.map((p,j)=>p*(1-r[i*V+j])))),pB=norm(root.p.map((p,i)=>p*z[i])),qB=norm(root.q.map((q,i)=>q*z[i]));
 let pTail=second.map((x,i)=>z[i]>0?norm(x.p.map((p,j)=>p*(1-r[i*V+j]))):x.p),qTail=second.map((x,i)=>{let a=x.q.map((q,j)=>q*(1-r[i*V+j]));return sum(a)>0?norm(a):x.q});
 // root + tail finite conditional law
 maxLaw=Math.max(maxLaw,...paths.map(x=>Math.abs(pB[x.i]*pTail[x.i][x.j]-x.p*(1-r[x.id])/Z)));
 function sd(p,q,rand){let x=cat(q,rand());if(rand()<Math.min(1,p[x]/q[x]))return [x,true];let d=p.map((v,i)=>Math.max(0,v-q[i]));return [cat(norm(d),rand()),false]}
 // formal launches proxy for 2-token SD: one full block if first accepted, else one extra formal call
 // row queries count only nonempty certificate rows; root precomputed for all methods
 let has=Array.from({length:V},(_,i)=>r.slice(i*V,(i+1)*V).some(Boolean));
 let A=overlap(root.p,root.q),AB=overlap(pB,qB);
 let fullCalls=2-A,fullQuery=sum(root.p.map((p,i)=>p*Number(has[i])));
 let conditionedCalls=Z*(2-AB),condQuery=Z*sum(qB.map((q,i)=>q*Number(has[i])))+Z*sum(pB.map((p,i)=>Math.max(0,p-qB[i])*Number(has[i])));
 // post-tag SD already obtains correct law, integrated gradient class without preconditioning
 // pre-tag whole ordinary SD rejection: expected calls per original slot = ordinary SD exactly, absent shared cross-attempt cache
 let rejectCalls=Z*fullCalls/Z,rejectQuery=Z*fullQuery/Z;
 // root-conditioned then tail; tail proposal rejection work exactly cancels except fully certified cells
 let earlyTail=Z*sum(pB.map((p,i)=>z[i]>0?p/z[i]:0));let earlyQuery=Z*sum(pB.map((p,i)=>z[i]>0?p/z[i]*Number(has[i]):0));
 maxCostIdentity=Math.max(maxCostIdentity,Math.abs(earlyTail-sum(root.p.filter((_,i)=>z[i]>0))));
 let mc={full:{calls:0,queries:0,slots:0,cert:0},cond:{calls:0,queries:0,slots:0,cert:0},reject:{calls:0,queries:0,slots:0,cert:0},early:{calls:0,queries:0,slots:0,cert:0}},freq={full:Array(V*V).fill(0),cond:Array(V*V).fill(0),reject:Array(V*V).fill(0),early:Array(V*V).fill(0)};
 let rand=rng(hash(h,seed+wi*101+Math.round(dup*100)+999));const R=2048;
 for(let rep=0;rep<R;rep++){
  // produce full target pair with actual two-stage maximal coupling, first-rejection tail uses p directly
  function block(p0,q0,pp,qq,rand){let [i,ok]=sd(p0,q0,rand);let j=ok?sd(pp[i],qq[i],rand)[0]:cat(pp[i],rand());return {i,j,calls:ok?1:2,ok}}
  let y=block(root.p,root.q,second.map(x=>x.p),second.map(x=>x.q),rand),id=y.i*V+y.j;mc.full.calls+=y.calls;mc.full.queries+=Number(has[y.i]);mc.full.slots++;if(r[id])mc.full.cert++;else freq.full[id]++;
  if(rand()<certMass){for(let k of ["cond","reject","early"])mc[k].cert++;continue}
  let c=block(pB,qB,pTail,qTail,rand);mc.cond.calls+=c.calls; // query draft row as well as corrected row, exact expectation below tracked separately through formula
  freq.cond[c.i*V+c.j]++;mc.cond.slots++;
  let n=0;do{y=block(root.p,root.q,second.map(x=>x.p),second.map(x=>x.q),rand);id=y.i*V+y.j;mc.reject.calls+=y.calls;mc.reject.queries+=Number(has[y.i]);n++;if(n>10000)throw Error("restart limit")}while(r[id]);freq.reject[id]++;mc.reject.slots++;
  let i=cat(pB,rand()),j;do{j=cat(second[i].p,rand());mc.early.calls++;mc.early.queries+=Number(has[i])}while(r[i*V+j]);mc.early.slots++;freq.early[i*V+j]++;
 }
 for(let key in mc){for(let k in mc[key])mc[key][k]/=R;mc[key].tv=0.5*sum(freq[key].map((x,id)=>Math.abs(x/R-paths[id].p*(1-r[id]))))}
 // exact expected calls includes conditional first token root launch in early method; with root cached, root AR overhead zero, and row fetch only on miss.
 // use common un-cached per-trajectory cost proxy and compare with ideal caching separately
 let cachedMiss=sum(root.p.map((p,i)=>p*Number(!has[i])));let cachedCond=Z*sum(pB.map((p,i)=>p*Number(!has[i]))); 
 rows.push({wi,seed,h,dup,pi,certMass,fullCalls,fullQuery,conditionedCalls,condQuery,rejectCalls,rejectQuery,earlyTail,earlyQuery,cachedMiss,cachedCond,mc,fullAccept:A,condAccept:AB});draws+=R;
 }
 }}
 const summary=[0,.3,.7].map(dup=>{let r=rows.filter(x=>x.dup===dup),out={dup,n:r.length};for(let k of ["pi","certMass","fullCalls","fullQuery","conditionedCalls","condQuery","rejectCalls","rejectQuery","earlyTail","earlyQuery","cachedMiss","cachedCond","fullAccept","condAccept"])out[k]=mean(r.map(x=>x[k]));out.mc={};for(let m of ["full","cond","reject","early"]){out.mc[m]={};for(let k of ["calls","queries","slots","cert","tv"])out.mc[m][k]=mean(r.map(x=>x.mc[m][k]))}out.frontier=[0,.25,1,4,16].map(c=>({c,post:mean(r.map(x=>x.fullCalls+c*x.fullQuery)),conditioned:mean(r.map(x=>x.conditionedCalls+c*x.condQuery)),restart:mean(r.map(x=>x.rejectCalls+c*x.rejectQuery)),adaptive:mean(r.map(x=>Math.min(x.conditionedCalls+c*x.condQuery,x.fullCalls+c*x.fullQuery)))}));return out});return {evals,draws,maxLaw,maxCostIdentity,summary,rows}
}

const fs=require('node:fs');console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));