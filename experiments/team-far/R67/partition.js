function run(raws,names){
const V=27,d=16,H=4,k=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let shallow;let seq=h.map((t,i)=>rms(add(E[t],P[i])));for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!shallow)shallow=seq.at(-1).slice()}return {deep:seq.at(-1),shallow}}}}


const avg=a=>sum(a)/a.length,hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h},kl=(p,q)=>sum(p.map((v,i)=>v?v*Math.log(v/q[i]):0));
let forwards=0,evals=0,rows=[],audit=0,words=[...new Set(names.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))];
function clsFit(data,K){let J=Array.from({length:V},()=>zero(V));for(let a of data)for(let x=0;x<V;x++)for(let y=0;y<V;y++)J[x][y]+=a.root[x]*a.p[x][y];let gs=J.map((v,i)=>({ids:[i],v}));let H=v=>{let n=sum(v);return -sum(v.map(x=>x?x*Math.log(x/n):0))};while(gs.length>K){let best=Infinity,ii=0,jj=1;for(let i=0;i<gs.length;i++)for(let j=i+1;j<gs.length;j++){let a=gs[i].v,b=gs[j].v,c=add(a,b),z=H(c)-H(a)-H(b);if(z<best){best=z;ii=i;jj=j}}let a=gs[ii],b=gs[jj];gs[ii]={ids:a.ids.concat(b.ids),v:add(a.v,b.v)};gs.splice(jj,1)}let g=zero(V);gs.forEach((a,i)=>a.ids.forEach(x=>g[x]=i));return g}

function measures(a,g,K){let mass=zero(K),cent=Array.from({length:K},()=>zero(K)),P=a.p.map(p=>{let v=zero(K);p.forEach((x,i)=>v[g[i]]+=x);return v}),Q=a.q.map(p=>{let v=zero(K);p.forEach((x,i)=>v[g[i]]+=x);return v}),rep=Array(K).fill(-1);
for(let x=0;x<V;x++){let c=g[x];mass[c]+=a.root[x];for(let j=0;j<K;j++)cent[c][j]+=a.root[x]*P[x][j];if(rep[c]<0||a.root[x]>a.root[rep[c]])rep[c]=x}
cent=cent.map((v,i)=>scale(v,1/mass[i]));let loss=0,repLoss=0,overlap=0;
for(let x=0;x<V;x++){let c=g[x];loss+=a.root[x]*kl(P[x],cent[c]);repLoss+=a.root[x]*kl(P[x],P[rep[c]]);overlap+=a.root[x]*sum(P[rep[c]].map((v,j)=>Math.min(v,Q[rep[c]][j])))}
return {loss,repLoss,overlap}}
function evalMap(data,g,K){evals++;let a=data.map(x=>measures(x,g,K));return {loss:avg(a.map(x=>x.loss)),repLoss:avg(a.map(x=>x.repLoss)),overlap:avg(a.map(x=>x.overlap))}}
function score(data,g,K,beta){let a=evalMap(data,g,K);return a.repLoss+beta*(1-a.overlap)}
function rng(seed){let s=seed;return ()=>((s=Math.imul(s,1664525)+1013904223>>>0)/4294967296)}
function optimize(base,K,data,beta,seed,parent){let g=base.slice(),r=rng(seed),v=score(data,g,K,beta);
for(let t=0;t<32;t++){let i=Math.floor(r()*V),j=Math.floor(r()*V);if(g[i]===g[j]||(parent&&parent[i]!==parent[j]))continue;let h=g.slice();[h[i],h[j]]=[h[j],h[i]];let q=score(data,h,K,beta);if(q<v){g=h;v=q}}return g}
function nested(parent,K0,K,data,beta,seed){let g=parent.slice();for(let k=K0;k<K;k++){let best=Infinity,bg;for(let i=0;i<V;i++){if(g.filter(x=>x===g[i]).length<2)continue;let a=g.slice();a[i]=k;let v=score(data,a,k+1,beta);if(v<best){best=v;bg=a}}if(!bg)throw Error("no split");g=bg}return optimize(g,K,data,beta,seed,parent)}
function ctx(m,n){let pre=[26,...n.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)],root=sm(mat(m.W,m.feature(pre).deep)),p=[],q=[];forwards++;for(let i=0;i<V;i++){let h=m.feature(pre.concat(i));p.push(sm(mat(m.W,h.deep)));q.push(sm(mat(m.W,h.shallow)));forwards++}return {root,p,q}}
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]);for(let seed of [11,29,73]){
let clients=Array.from({length:4},(_,r)=>words.filter(x=>{let v=x.charCodeAt(0)-97;return (v>=13?2:0)+v%2===r}).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,24));
let tr=clients.map(a=>a.slice(0,16).map(n=>ctx(m,n))),te=clients.map(a=>a.slice(16).map(n=>ctx(m,n))),regions=[tr[0].concat(tr[1]),tr[2].concat(tr[3])],tests=[te[0].concat(te[1]),te[2].concat(te[3])],all=tr.flat();
for(let beta of [0,.25]){let parent=optimize(clsFit(all,4),4,all,beta,seed,null);let global=optimize(clsFit(all,8),8,all,beta,seed+1,null),globalNested=nested(parent,4,8,all,beta,seed+2);
for(let e=0;e<2;e++){let reg=regions[e],test=tests[e],nest=nested(parent,4,8,reg,beta,seed+3),free=optimize(clsFit(reg,8),8,reg,beta,seed+3,null);
for(let [kind,g,K] of [["parent",parent,4],["global",global,8],["globalNested",globalNested,8],["regionalNested",nest,8],["regionalFree",free,8],["delegatedFlat",nest,8]]){let v=evalMap(test,g,K);let parentOf=Array(K).fill(-1);if(kind.includes("Nested")||kind==="delegatedFlat"){for(let i=0;i<V;i++){if(parentOf[g[i]]>=0&&parentOf[g[i]]!==parent[i])throw Error("non nested");parentOf[g[i]]=parent[i]}}
rows.push({wi,seed,beta,e,kind,K,g,parent,...v})}
}
}
}}
let summary=[];for(let beta of [0,.25])for(let kind of ["parent","global","globalNested","regionalNested","regionalFree","delegatedFlat"]){let a=rows.filter(x=>x.beta===beta&&x.kind===kind);summary.push({beta,kind,n:a.length,loss:avg(a.map(x=>x.loss)),repLoss:avg(a.map(x=>x.repLoss)),overlap:avg(a.map(x=>x.overlap)),score:avg(a.map(x=>x.repLoss+beta*(1-x.overlap)))})}
return {rows,summary,forwards,evals};
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("fs");console.log(JSON.stringify(run([JSON.parse(fs.readFileSync(process.argv[2],"utf8")),JSON.parse(fs.readFileSync(process.argv[3],"utf8")).weights],fs.readFileSync(process.argv[4],"utf8")),null,2))}
