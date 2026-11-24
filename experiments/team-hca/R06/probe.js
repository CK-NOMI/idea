// Download pinned public-domain Gutenberg texts as wallpaper.txt and alice.txt; node probe.js wallpaper.txt alice.txt
const fs=require('node:fs');
function probe(docs){
const alphabet=" abcdefghijklmnopqrstuvwxyz",V=alphabet.length,D=V*V;
const clean=(doc)=>{let s=doc.text.split(/\*\*\* START OF THIS PROJECT GUTENBERG EBOOK[^\n]*\n/)[1].split(/\*\*\* END OF THIS PROJECT GUTENBERG EBOOK/)[0];return s.toLowerCase().replace(/[^a-z]+/g," ").trim()};
const a=clean(docs[1]),w=clean(docs[0]),cut=a.lastIndexOf(" ",Math.floor(a.length/2));const regions=[w,a.slice(0,cut),a.slice(cut+1)].map(s=>Array.from(s,c=>alphabet.indexOf(c)));
let train=[],test=[],counts=[],all=new Float64Array(V**3),globalBi=new Float64Array(D);
for(const r of regions){let n=Math.floor(r.length*.8);train.push(r.slice(0,n));test.push(r.slice(n));}
for(const rr of train){let bi=new Float64Array(D),tri=new Float64Array(V**3);for(let i=2;i<rr.length;i++){let j=rr[i-1]*V+rr[i],k=rr[i-2]*D+j;bi[j]++;tri[k]++}for(let j=0;j<D;j++)globalBi[j]+=bi[j];for(let k=0;k<all.length;k++)all[k]+=tri[k];counts.push(bi)}
const norm=(a)=>{let t=a.reduce((s,v)=>s+v,0);return a.map(v=>v/t)};
let qb=Array.from({length:V},(_,b)=>norm(Array.from(globalBi.slice(b*V,(b+1)*V),x=>x+.5)));
let p=Array.from({length:D},(_,s)=>norm(Array.from(all.slice(s*V,(s+1)*V),(x,j)=>x+5*qb[s%V][j])));
let qs=counts.map(bi=>Array.from({length:V},(_,b)=>norm(Array.from(bi.slice(b*V,(b+1)*V),(x,j)=>x+20*qb[b][j]))));
let meta=test.map((rr,e)=>{let np=0,nq=0,ng=0,delta=0;for(let i=2;i<rr.length;i++){let s=rr[i-2]*V+rr[i-1],b=rr[i-1],y=rr[i];np-=Math.log(p[s][y]);nq-=Math.log(qs[e][b][y]);ng-=Math.log(qb[b][y]);delta+=p[s].reduce((a,x,j)=>a+Math.max(0,x-qs[e][b][j]),0)}let n=rr.length-2;return {region:e,train:train[e].length,test:rr.length,pNLL:np/n,qNLL:nq/n,globalQNLL:ng/n,teacherForcedOverlap:1-delta/n}});
const seeds=[11,29,73],lengths=[4,16,64],N=400,rows=[];let st=0;const U=()=>{st=(Math.imul(1664525,st)+1013904223)>>>0;return (st+.5)/4294967296};
const draw=a=>{let r=U();for(let j=0;j<V;j++){r-=a[j];if(r<0)return j}return V-1};
function pair(P,Q){let x=draw(Q);if(U()*Q[x]<P[x])return [x,x];let r=P.map((v,j)=>Math.max(0,v-Q[j])),z=r.reduce((s,v)=>s+v,0);return [x,draw(r.map(v=>v/z))]}
function addscore(g,b,x,k,Q){let base=b*V;for(let j=0;j<V;j++)g[base+j]-=k*Q[j];g[base+x]+=k}
function oracle(init,q,L){let occ=new Float64Array(D),g=new Float64Array(D);for(const s of init)occ[s]+=1/init.length;for(let t=0;t<L;t++){let next=new Float64Array(D);for(let s=0;s<D;s++){let b=s%V,m=occ[s];if(!m)continue;for(let j=0;j<V;j++){let z=m*p[s][j];next[b*V+j]+=z;g[b*V+j]+=m*(q[b][j]-p[s][j])}}occ=next}return g}
for(let e=0;e<3;e++){let q=qs[e],rr=test[e],init=Array.from({length:64},(_,i)=>{let k=Math.floor(i*(rr.length-2)/64);return rr[k]*V+rr[k+1]});for(const L of lengths){let truth=oracle(init,q,L);for(const seed of seeds){st=seed+1009*e+L;let names=["path","local","hard"],mom=Object.fromEntries(names.map(k=>[k,{s:new Float64Array(D),ss:0,nonzero:0}]));let prefix=0,pathEqual=0,localEqual=0,pathContext=0;for(let n=0;n<N;n++){let initial=init[Math.floor(U()*init.length)],sp=initial,bx=initial%V,sl=initial,alive=true;let gp=new Float64Array(D),gl=new Float64Array(D),gh=new Float64Array(D);for(let t=0;t<L;t++){let by=sp%V;let [x,y]=pair(p[sp],q[bx]);if(bx===by)pathContext++;if(bx===by&&x===y)pathEqual++;else{addscore(gp,bx,x,1,q[bx]);addscore(gp,by,y,-1,q[by])}alive=alive&&x===y;if(alive)prefix++;bx=x;sp=by*V+y;
let bl=sl%V,[xl,yl]=pair(p[sl],q[bl]);if(xl===yl)localEqual++;gl[bl*V+xl]++;gl[bl*V+yl]--;addscore(gh,bl,yl,-1,q[bl]);sl=bl*V+yl;
}for(const [k,g] of [["path",gp],["local",gl],["hard",gh]]){let m=mom[k];for(let j=0;j<D;j++){m.s[j]+=g[j];m.ss+=g[j]**2;if(Math.abs(g[j])>1e-12)m.nonzero++}}}
let out={region:e,L,seed,N,prefixMean:prefix/N,pathMatchedFraction:pathEqual/(N*L),localMatchedFraction:localEqual/(N*L),pathContextFraction:pathContext/(N*L)};for(const k of names){let m=mom[k],mn=Array.from(m.s,x=>x/N),sq=mn.reduce((a,x)=>a+x*x,0),err=mn.reduce((a,x,j)=>a+(x-truth[j])**2,0);out[k]={traceVariance:(m.ss-N*sq)/(N-1),meanSquaredError:err,meanNonzero:m.nonzero/N}}rows.push(out)}}}
return {source:docs.map(({text,...rest})=>rest),environment:"JS V8 CPU; real public text trained count-based character LM; not pretrained Transformer, no GPU/network timing",alphabet,meta,seeds,lengths,N,rows};
}
const source=[{"repo":"GITenberg/The-Yellow-Wallpaper_1952","path":"1952.txt","ref":"ce7c5bc1a5cf0fe23bcb265545a900365dbae29c","blob":"3a2b5f55c6f1d13a7ae623282d38fde5a6ff352b"},{"repo":"GITenberg/Alice-s-Adventures-in-Wonderland_11","path":"11.txt","ref":"7f18748e114b994d5d9454d5793581bc0b9cf9d7","blob":"9d73b1d7447b7ddc158b1b157f5f700d21bd2fb7"}];
console.log(JSON.stringify(probe(source.map((s,i)=>({...s,text:fs.readFileSync(process.argv[i+2],'utf8')}))),null,2));
