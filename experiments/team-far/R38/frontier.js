// node frontier.js weights.json checkpoint.json input.txt
function frontierProbe(raws,names){
 const V=27,d=16,H=4,k=4,N=8,L=2,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
 function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),layers=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of layers){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {p:sm(mat(W,seq.at(-1))),q:sm(mat(W,s))}}}
 const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
 function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}

 const Lx=4,words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],map=new Map();
 for(let w of words){let h=w.slice(0,3),s=w.slice(3,7);if(!map.has(h))map.set(h,new Set());map.get(h).add(s)}
 let prefixes=[...map.keys()].filter(h=>map.get(h).size>=8),rows=[],maxMass=0,maxCost=0,maxLaw=0,evals=0,draws=0;
 const cat=(a,u)=>{let s=0;for(let i=0;i<a.length;i++){s+=a[i];if(u<s)return i}return a.length-1};
 for(let wi=0;wi<raws.length;wi++){let run=model(raws[wi]);for(let seed of [11,29,73])for(let h of prefixes.slice().sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,8)){
 const prompt=[26,...h.split("").map(x=>x.charCodeAt(0)-97)],valid=[...map.get(h)].sort((a,b)=>hash(a,seed)-hash(b,seed));
 let tree={children:new Map(),mass:1,prefix:[],r:0};let chosen=valid.slice(0,Math.min(16,valid.length));
 for(let w of chosen){let n=tree;for(let c of w){let i=c.charCodeAt(0)-97;if(!n.children.has(i))n.children.set(i,{children:new Map(),prefix:n.prefix.concat(i),r:0});n=n.children.get(i)}n.r=1}
 let nodes=[],atoms=[],leaves=[],a=0;
 function visit(n,m){n.mass=m;if(n.prefix.length===Lx){leaves.push(n);a+=m*n.r;if(n.r<1)atoms.push({n,weight:m*(1-n.r),terminal:true});return}
 n.p=run(prompt.concat(n.prefix)).p;evals++;nodes.push(n);let d=sum(n.p.filter((_,i)=>!n.children.has(i)));n.exit=d;atoms.push({n,weight:m*d,terminal:false});
 for(let [i,c] of n.children)visit(c,m*n.p[i]);}
 visit(tree,1);
 let Z=1-a,front=sum(atoms.map(x=>x.weight)),tail=sum(atoms.map(x=>x.weight*(x.terminal?0:Lx-x.n.prefix.length-1))),exitCount=sum(atoms.filter(x=>!x.terminal).map(x=>x.weight));
 maxMass=Math.max(maxMass,Math.abs(front-Z));
 // Conditional probability of each first-exit atom matches exact product; post-tag law same for all outside-certificate work
 let byDepth=Array.from({length:Lx},(_,j)=>sum(atoms.filter(x=>!x.terminal&&x.n.prefix.length===j).map(x=>x.weight)));
 let expectedCachedAR=sum(nodes.map(n=>n.mass)),expectedUncached=Lx-expectedCachedAR,tailDirect=tail;
 maxCost=Math.max(maxCost,Math.abs(expectedUncached-tailDirect));
 // packed frontier scalar masses and node IDs vs cached 27-way p rows; additional sparse topology and model calls excluded
 let dense=nodes.length*V*4,packed=nodes.length*8;let rowCacheMiss=exitCount; // if only scalar masses retained, one p row re-eval per exiting slot
 let rand=rng(hash(h,seed+wi*101)),freq=Array(Lx+1).fill(0),condTail=0,R=4096;
 const norm=atoms.map(x=>x.weight/Z);
 for(let k=0;k<R;k++){let j=cat(norm,rand()),x=atoms[j],dep=x.n.prefix.length;freq[dep]++;condTail+=x.terminal?0:Lx-dep-1;draws++}
 maxLaw=Math.max(maxLaw,...freq.map((v,i)=>Math.abs(v/R-(i===Lx?sum(atoms.filter(x=>x.terminal).map(x=>x.weight)):byDepth[i])/Z)));
 rows.push({wi,seed,h,certCount:chosen.length,nodes:nodes.length,a,Z,byDepth,expectedCachedAR,expectedUncached,tailDirect,exitCount,denseBytes:dense,packedBytes:packed,rowCacheMiss,mcTail:condTail/R,analyticCondTail:tail/Z});
 }}
 let out={n:rows.length};for(let k of ["certCount","nodes","a","Z","expectedCachedAR","expectedUncached","tailDirect","exitCount","denseBytes","packedBytes","rowCacheMiss","mcTail","analyticCondTail"])out[k]=mean(rows.map(x=>x[k]));out.byDepth=Array.from({length:Lx},(_,j)=>mean(rows.map(x=>x.byDepth[j])));
 return {evals,draws,maxMass,maxCost,maxLaw,summary:out,rows}
}

const fs=require('node:fs');console.log(JSON.stringify(frontierProbe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));