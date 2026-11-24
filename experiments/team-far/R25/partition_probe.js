// node partition_probe.js weights.json checkpoint.json input.txt
// Forward adapted from R25/probe.js. Teacher joint and class-conditioned block head.
const fs=require('node:fs');
function probe(raws,names){
const V=27,d=16,H=4,k=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i])));for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))})}return seq.at(-1)}}}
const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],rows=[],fitRows=[],audit={identity:0,monotone:0,normalization:0,conditional:0,grad:0};
function cent(p,g,R){let a=p.map(sum),M=Array.from({length:R},()=>zero(V)),w=zero(R);for(let i=0;i<V;i++){M[g[i]]=add(M[g[i]],p[i]);w[g[i]]+=a[i]}return {a,M:M.map((m,j)=>w[j]?scale(m,1/w[j]):Array(V).fill(1/V)),w}}
function kl(p,q){return sum(p.map((v,j)=>v?v*Math.log(v/q[j]):0))}
function evaluate(x,g,R){let {a,M,w}=cent(x.p,g,R),b=zero(V);for(let i=0;i<V;i++)b=add(b,x.p[i]);let I=sum(x.p.map((r,i)=>a[i]*kl(scale(r,1/a[i]),b))),J=sum(x.p.map((r,i)=>a[i]*kl(scale(r,1/a[i]),M[g[i]]))),G=sum(M.map((p,j)=>w[j]*kl(p,b))),tv=sum(x.p.map((r,i)=>sum(r.map((v,j)=>Math.abs(v-a[i]*M[g[i]][j])))))/2;
audit.identity=Math.max(audit.identity,Math.abs(I-J-G));audit.normalization=Math.max(audit.normalization,Math.abs(sum(w)-1));return {I,J,G,tv}}
function fit(p,R,seed){let best=null;for(let trial=0;trial<4;trial++){let rr=rng(seed+17*trial),g=Array.from({length:V},(_,i)=>i%R);for(let i=V-1;i>0;i--){let j=Math.floor(rr()*(i+1));[g[i],g[j]]=[g[j],g[i]]}let old=Infinity;for(let t=0;t<20;t++){let {a,M}=cent(p,g,R),v=sum(p.map((r,i)=>a[i]*kl(scale(r,1/a[i]),M[g[i]])));audit.monotone=Math.max(audit.monotone,Number.isFinite(old)?v-old:0);old=v;let newg=p.map((r,i)=>{let z=M.map(m=>kl(scale(r,1/a[i]),m));return z.indexOf(Math.min(...z))});if(newg.join(",")===g.join(","))break;g=newg}let e=evaluate({p},g,R).J;if(!best||e<best.e)best={g,e}}return best.g}
function pooled(xs){let p=Array.from({length:V},()=>zero(V));for(let x of xs)for(let i=0;i<V;i++)p[i]=add(p[i],scale(x.p[i],1/xs.length));return p}
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]),W=m.W;
function ctx(name){let j=3,pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],h=m.feature(pre),l=mat(W,h),a=sm(l),p=a.map((v,i)=>scale(sm(mat(W,m.feature(pre.concat(i)))),v));return {p,log:l,f:rms(h).concat(1),y:name.charCodeAt(j)-97,z:name.charCodeAt(j+1)-97,region:name.charCodeAt(0)%4}}
for(let seed of [11,29,73]){let names4=Array.from({length:4},(_,r)=>words.filter(x=>x.charCodeAt(0)%4===r).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,24)),train=names4.map(g=>g.slice(0,16).map(ctx)),test=names4.flatMap(g=>g.slice(16).map(ctx)),all=train.flat();
for(let R of [1,2,4,8,16]){let g=fit(pooled(all),R,seed),regional=train.map(x=>fit(pooled(x),R,seed)),fixed=Array.from({length:V},(_,i)=>i%R);for(let method of ["fixed","global","regional","oracle"]){let vals=test.map(x=>evaluate(x,method==="fixed"?fixed:method==="global"?g:method==="regional"?regional[x.region]:fit(x.p,R,seed),R));rows.push({wi,seed,R,method,I:sum(vals.map(x=>x.I))/vals.length,J:sum(vals.map(x=>x.J))/vals.length,tv:sum(vals.map(x=>x.tv))/vals.length})}
// Actual learned formal conditional block model: p1 frozen original, p2 given coarse token class and original h, contextual linear adjustment + class bias.
let D=d+1,dim=V*D+R*V,t=zero(dim),rr=rng(seed+wi*100),updates=0;
function loss(x,t,grad){let gvec=zero(dim),loss=0,{M,w}=cent(x.p,g,R);
for(let c=0;c<R;c++){if(!w[c])continue;let pred=sm(x.log.map((v,j)=>v+dot(t.slice(j*D,(j+1)*D),x.f)+t[V*D+c*V+j]));loss+=w[c]*kl(M[c],pred);if(grad)for(let j=0;j<V;j++){let z=w[c]*(pred[j]-M[c][j]);gvec[V*D+c*V+j]+=z;for(let i=0;i<D;i++)gvec[j*D+i]+=z*x.f[i]}}return {loss,gvec}}
function evaluateLearn(){let teacherKL=0,nll=0,oracleKL=0;for(let x of test){let cc=cent(x.p,g,R),a=cc.a,c=g[x.y],pred=sm(x.log.map((v,j)=>v+dot(t.slice(j*D,(j+1)*D),x.f)+t[V*D+c*V+j]));teacherKL+=loss(x,t,false).loss+evaluate(x,g,R).J;oracleKL+=evaluate(x,g,R).J;nll-=Math.log(a[x.y]*pred[x.z])}return {teacherKL:teacherKL/test.length,nll:nll/test.length,oracleKL:oracleKL/test.length}}
let initial=evaluateLearn();
function local(t,client){let v=t.slice();for(let step=0;step<2;step++){let gd=zero(dim);for(let j=0;j<4;j++){let x=train[client][Math.floor(rr()*train[client].length)];gd=add(gd,loss(x,v,true).gvec)}v=add(v,scale(gd,-.12/4));updates++}return v}
const avg=a=>scale(a.reduce(add),1/a.length);
for(let round=0;round<24;round++){let edges=[];for(let e=0;e<2;e++){let v=t.slice();for(let j=0;j<2;j++)v=avg([local(v,e*2),local(v,e*2+1)]);edges.push(v)}t=avg(edges)}
let final=evaluateLearn(),x=test[0],grad=loss(x,t,true).gvec;for(let i=0;i<8;i++){let j=Math.floor(i*dim/8),tt=t.slice();tt[j]+=1e-5;let l1=loss(x,tt,false).loss;tt[j]-=2e-5;let l2=loss(x,tt,false).loss;audit.grad=Math.max(audit.grad,Math.abs(grad[j]-(l1-l2)/2e-5))}
fitRows.push({wi,seed,R,params:dim,initial,final,updates,uses:updates*4})
}
}}
let summary=[];for(let R of [1,2,4,8,16])for(let method of ["fixed","global","regional","oracle"]){let a=rows.filter(x=>x.R===R&&x.method===method);summary.push({R,method,...Object.fromEntries(["I","J","tv"].map(k=>[k,sum(a.map(x=>x[k]))/a.length]))})}
let learned=[];for(let R of [1,2,4,8,16]){let a=fitRows.filter(x=>x.R===R);learned.push({R,params:a[0].params,initialKL:sum(a.map(x=>x.initial.teacherKL))/a.length,finalKL:sum(a.map(x=>x.final.teacherKL))/a.length,nll:sum(a.map(x=>x.final.nll))/a.length,oracleKL:sum(a.map(x=>x.final.oracleKL))/a.length,updates:sum(a.map(x=>x.updates)),uses:sum(a.map(x=>x.uses))})}
return {rows,fitRows,summary,learned,audit,trainContexts:384,testContexts:192,teacherPrefixForwards:576*28}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
