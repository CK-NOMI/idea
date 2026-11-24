// node probe.js weights.json checkpoint.json input.txt
function probe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {y:seq.at(-1),x:s}}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}


 const zeros=(n,m)=>Array.from({length:n},()=>zero(m)),ma=(a,b)=>a.map((x,i)=>add(x,b[i])),ms=(a,s)=>a.map(x=>scale(x,s)),outer=(x,y)=>x.map(v=>scale(y,v)),T=A=>A[0].map((_,i)=>A.map(r=>r[i])),eye=()=>Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>+(i===j))),maxdiff=(a,b)=>Math.max(...ma(a,ms(b,-1)).flat().map(Math.abs)),cat=(p,u)=>{let s=0;for(let i=0;i<p.length;i++){s+=p[i];if(u<s)return i}return p.length-1};
 const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],rows=[];let evals=0,updates=0,maxGrad=0,maxLoss=0,maxFD=0,draws=0;
 for(let wi=0;wi<raws.length;wi++){let {W,feature}=model(raws[wi]);for(let seed of [11,29,73]){
 let pools=Array.from({length:4},(_,c)=>words.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,80));
 let train=pools.map(c=>c.slice(0,40).map(w=>{evals++;return {...feature([26,...w.slice(0,3).split("").map(x=>x.charCodeAt(0)-97)]),label:w.charCodeAt(3)-97}}));
 function fit(xs,eta){let U=W.map(r=>r.slice());for(let it=0;it<30;it++){let g=zeros(V,d);for(let s of xs){let p=sm(mat(U,s.y));p[s.label]-=1;g=ma(g,outer(p,s.y))}U=ma(U,ms(g,-eta/xs.length));updates++}return U}
 let rg=[train[0].concat(train[1]),train[2].concat(train[3])];
 for(let drift of [.02,.1,.4]){let Hs=rg.map(xs=>fit(xs,drift)),common=ms(ma(Hs[0],Hs[1]),.5);
 for(let sharing of ["common","private"]){let init=eye();for(let method of ["AR","SDregional","SDshared"]){let A=init.map(r=>r.slice()),rheads=Hs.map(x=>x.map(r=>r.slice())),counts={global:0,local:0,pGlobal:0,pLocal:0,qGlobal:0,qLocal:0,proposals:0,accepted:0,blocks:0,commits:0,backwardLocal:0,backwardGlobal:0,backwardSeparate:0,loss:0},rounds=[];
 for(let round=0;round<4;round++){let global=new Map(),local=[new Set(),new Set()],pCache=[new Map(),new Map()],qCache=[new Map(),new Map()],qGlobal=new Map(),grad=zeros(d,d),merged=new Map(),regional=[new Map(),new Map()],headG=[zeros(V,d),zeros(V,d)],loss=0;
 const key=h=>h.join(",");
 function base(h,e){let k=key(h);if(!local[e].has(k)){local[e].add(k);counts.local++}if(!global.has(k)){global.set(k,feature(h));counts.global++;evals++}return global.get(k)}
 function p(h,e){let k=key(h),s=base(h,e);if(!pCache[e].has(k)){pCache[e].set(k,sm(mat(rheads[e],mat(A,s.y))));counts.pLocal++;counts.pGlobal++}return pCache[e].get(k)}
 function q(h,e){let k=key(h),s=base(h,e);if(!qCache[e].has(k)){counts.qLocal++;if(method==="SDshared"){if(!qGlobal.has(k)){qGlobal.set(k,sm(mat(common,s.x)));counts.qGlobal++}qCache[e].set(k,qGlobal.get(k))}else{qCache[e].set(k,sm(mat(rheads[e],s.x)));counts.qGlobal++}}return qCache[e].get(k)}
 let rand=rng(seed+wi*1000+Math.round(drift*100)*300+round*17+(sharing==="common"?111:333)+(method==="AR"?0:method==="SDregional"?1:2));
 let paths=[];
 let promptWord=pools[0][40+round],toPrompt=w=>[26,...w.slice(0,3).split("").map(x=>x.charCodeAt(0)-97)];
 for(let e=0;e<2;e++)for(let c=0;c<2;c++){let w=sharing==="common"?promptWord:pools[2*e+c][40+round],prompt=toPrompt(w);let group=[];
 for(let n=0;n<8;n++){let hist=prompt.slice(),out=[];while(out.length<4){
 if(method==="AR"){let y=cat(p(hist,e),rand());out.push(y);hist.push(y);counts.blocks++;continue}
 let cap=Math.min(2,4-out.length),draft=[],dh=hist.slice();for(let t=0;t<cap;t++){let qq=q(dh,e),x=cat(qq,rand());draft.push({x,qq,h:dh.slice()});dh.push(x);counts.proposals++}
 for(let t of draft)p(t.h,e);counts.blocks++;
 for(let t of draft){let pp=p(t.h,e),x=t.x,y=x,ok=rand()<Math.min(1,pp[x]/t.qq[x]);if(ok)counts.accepted++;else{let r=pp.map((v,i)=>Math.max(0,v-t.qq[i])),z=sum(r);y=cat(scale(r,1/z),rand())}out.push(y);hist.push(y);if(!ok)break}
 }
 let reward=out.reduce((s,t,i)=>s+Number(t===w.charCodeAt(3+i)-97),0)/4;group.push({prompt,out,reward});counts.commits+=out.length;draws++;
 }
 let avg=mean(group.map(x=>x.reward));for(let g of group)paths.push({...g,e,adv:(g.reward-avg)/(2*2*8*4)});
 }
 for(let path of paths){let hist=path.prompt.slice(),e=path.e;for(let y of path.out){let s=base(hist,e),v=mat(A,s.y),pp=p(hist,e),z=pp.slice();z[y]-=1;z=scale(z,path.adv);let adj=mat(T(rheads[e]),z),g=outer(adj,s.y),k=key(hist);grad=ma(grad,g);headG[e]=ma(headG[e],outer(z,v));loss-=path.adv*Math.log(pp[y]);counts.backwardSeparate++;
 if(!regional[e].has(k))regional[e].set(k,{adj:zero(d),y:s.y});regional[e].get(k).adj=add(regional[e].get(k).adj,adj);
 if(!merged.has(k))merged.set(k,{adj:zero(d),y:s.y});merged.get(k).adj=add(merged.get(k).adj,adj);
 hist.push(y)
 }}
 let localGrad=zeros(d,d),globalGrad=zeros(d,d);
 for(let e=0;e<2;e++)for(let x of regional[e].values())localGrad=ma(localGrad,outer(x.adj,x.y));
 for(let x of merged.values())globalGrad=ma(globalGrad,outer(x.adj,x.y));
 counts.backwardLocal+=regional[0].size+regional[1].size;counts.backwardGlobal+=merged.size;
 maxGrad=Math.max(maxGrad,maxdiff(grad,localGrad),maxdiff(grad,globalGrad));
 if(round===0){function fixedLoss(A2){let l=0;for(let path of paths){let hist=path.prompt.slice();for(let y of path.out){let s=global.get(key(hist)),pp=sm(mat(rheads[path.e],mat(A2,s.y)));l-=path.adv*Math.log(pp[y]);hist.push(y)}}return l}
 let B=A.map(x=>x.slice()),C=A.map(x=>x.slice());B[0][0]+=1e-5;C[0][0]-=1e-5;maxFD=Math.max(maxFD,Math.abs((fixedLoss(B)-fixedLoss(C))/2e-5-grad[0][0]));maxLoss=Math.max(maxLoss,Math.abs(fixedLoss(A)-loss));}
 A=ma(A,ms(globalGrad,-.05));rheads=rheads.map((x,e)=>ma(x,ms(headG[e],-.1)));counts.loss+=loss;
 rounds.push({global:global.size,local:local[0].size+local[1].size,backwardGlobal:merged.size,backwardLocal:regional[0].size+regional[1].size});
 }
 rows.push({wi,seed,drift,sharing,method,counts,rounds});
 }}}
 }}
 let summary=[];for(let drift of [.02,.1,.4])for(let sharing of ["common","private"])for(let method of ["AR","SDregional","SDshared"]){let rr=rows.filter(x=>x.drift===drift&&x.sharing===sharing&&x.method===method),c={};for(let k of Object.keys(rr[0].counts))c[k]=mean(rr.map(x=>x.counts[k]));summary.push({drift,sharing,method,...c,accept:c.proposals?c.accepted/c.proposals:0})}
 return {evals,updates,draws,maxGrad,maxLoss,maxFD,summary,rows}
}

const fs=require('node:fs');console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));