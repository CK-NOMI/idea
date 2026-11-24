// node certificate.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
const V=27,d=16,H=4,k=4,N=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v*b[i]*0+v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {h:seq.at(-1),s}}}}
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(a,u){let s=0;for(let i=0;i<a.length;i++){s+=a[i];if(u<s)return i}return a.length-1}
const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
function conv(a,b){let c=zero(a.length+b.length-1);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)c[i+j]+=a[i]*b[j];return c}
const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],rows=[],audit={normalization:0,localLaw:0,naiveMarginalTV:0,maxNaiveJointVoteTV:0,cohorts:0,draws:0,actualFirstDrafts:0,actualSecondDrafts:0};
function verify(x,p,q,rr){let alpha=q[x]===0?1:Math.min(1,p[x]/q[x]);if(rr()<=alpha)return {y:x,ok:true};let r=p.map((v,i)=>Math.max(0,v-q[i]));return {y:cat(scale(r,1/sum(r)),rr()),ok:false}}
function kernel(x,p,q){let alpha=q[x]===0?1:Math.min(1,p[x]/q[x]),r=p.map((v,i)=>Math.max(0,v-q[i])),s=sum(r);return r.map((v,i)=>Number(i===x)*alpha+(s?(1-alpha)*v/s:0))}
function posterior(q,bit,f0,f1){let z=sum(q.map((v,i)=>v*(bit[i]?f1:f0)));return q.map((v,i)=>v*(bit[i]?f1:f0)/z)}
const event=(s,mode,v)=>mode==="count"?s===v:Number(s>=2)===v;
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]),W=m.W,mom=p=>p.reduce((a,v,i)=>add(a,scale(W[i],v)),zero(d));
for(let seed of [11,29,73]){let groups=Array.from({length:N},(_,r)=>words.filter(x=>x.charCodeAt(0)%N===r).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,32));
let make=(name)=>{let j=3,pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],{h,s}=m.feature(pre);return {pre,h,s,y:name.charCodeAt(j)-97}};
let train=groups.map(g=>g.slice(0,16).map(make)),g=scale(train.flat().map(x=>sub(mom(sm(mat(W,x.h))),W[x.y])).reduce(add),1/(16*N)),shift=mat(W,scale(g,-1.5));
let test=groups.map(g=>g.slice(16).map(make)),rr=rng(seed+1000*wi);
for(let c=0;c<16;c++){let xs=test.map(a=>a[c]).map(x=>{let p=sm(mat(W,x.h)),p1=sm(add(mat(W,x.h),shift)),q=sm(mat(W,x.s)),bit=p.map((v,i)=>Number(p1[i]>v)),a=dot(q,bit),next=Array.from({length:V},(_,i)=>{let {h,s}=m.feature(x.pre.concat(i)),l=mat(W,h);return {p:[sm(l),sm(add(l,shift))],q:sm(mat(W,s))}});return {p:[p,p1],q,bit,a,next,y:x.y,h:x.h,s:x.s}}),suff=Array(N+1);suff[N]=[1];for(let i=N-1;i>=0;i--)suff[i]=conv([1-xs[i].a,xs[i].a],suff[i+1]);
let votePatterns=Array.from({length:1<<N},(_,mask)=>{let bits=xs.map((_,i)=>Number(!!(mask&(1<<i)))),w=bits.reduce((v,b,i)=>v*(b?xs[i].a:1-xs[i].a),1);return {bits,w,s:sum(bits)}});
audit.cohorts++;
rows.push({wi,seed,c,xs});
}}} return rows;}









function certificateStudy(cohorts){
const sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),V=27,D=17,sg=x=>1/(1+Math.exp(-Math.max(-40,Math.min(40,x)))),mean=a=>sum(a)/a.length,zero=n=>Array(n).fill(0),norm=a=>{let z=sum(a);return a.map(x=>x/z)};
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(p,r){let x=r()*sum(p),s=0;for(let i=0;i<p.length;i++){s+=p[i];if(x<s)return i}return p.length-1}
function makeTree(rows,kind,seed){let freq=Array(V).fill(.1);for(let z of rows)freq[z.y]++;let rr=rng(seed),uid=0;
function leaf(i){return {id:uid++,ids:[i],i,wt:freq[i]}}function join(l,r){return {id:uid++,ids:l.ids.concat(r.ids),l,r,wt:l.wt+r.wt}}
let a=Array.from({length:V},(_,i)=>leaf(i));if(kind==="huffman"){while(a.length>1){a.sort((x,y)=>x.wt-y.wt||x.id-y.id);a.push(join(a.shift(),a.shift()))}return a[0]}
if(kind==="random"){for(let i=a.length-1;i>0;i--){let j=Math.floor(rr()*(i+1));[a[i],a[j]]=[a[j],a[i]]}}
else {let scores=Array.from({length:V},(_,i)=>mean(rows.map(z=>z.p[0][i]-z.q[i])));a.sort((x,y)=>scores[x.i]-scores[y.i])}
function rec(a){if(a.length===1)return a[0];let m=a.length>>1;return join(rec(a.slice(0,m)),rec(a.slice(m)))}return rec(a)}
function nodes(t){return t.l?[t,...nodes(t.l),...nodes(t.r)]:[]}
function fit(t,rows,mode){let w=new Map(nodes(t).map(n=>[n.id,zero(D)]));for(let epoch=0;epoch<100;epoch++){let grad=new Map(nodes(t).map(n=>[n.id,zero(D)]));for(let z of rows){let x=[...(mode==="p"?z.h:z.s),1],prob=mode==="p"?z.p[0]:z.q;for(let n of nodes(t)){let mass=sum(n.ids.map(i=>prob[i])),right=sum(n.r.ids.map(i=>prob[i])),g=grad.get(n.id),a=mass*sg(dot(w.get(n.id),x))-right;for(let j=0;j<D;j++)g[j]+=a*x[j]}}
for(let [k,v] of w){let g=grad.get(k);for(let j=0;j<D;j++)v[j]-=.35*(g[j]/rows.length+.0005*v[j])}}return w}
function distribution(t,w,x){let p=zero(V);function go(n,m){if(!n.l){p[n.i]=m;return}let a=sg(dot(w.get(n.id),x));go(n.l,m*(1-a));go(n.r,m*a)}go(t,1);return p}
function mass(n,p){return sum(n.ids.map(i=>p[i]))}
function pathOverlap(n,p,q){if(!n.l)return 1;let a=mass(n,p),b=mass(n,q);return [n.l,n.r].reduce((s,c)=>s+Math.min(mass(c,p)/a,mass(c,q)/b)*pathOverlap(c,p,q),0)}
function oracle(t,p,q,rr,method,cap=Infinity){
let pc=new Map(),qc=new Map(),front=[{n:t,m:1,exact:false}],iters=0,miss=false;function branch(n,which){let c=which==="p"?pc:qc,prob=which==="p"?p:q;if(!c.has(n.id))c.set(n.id,mass(n.r,prob)/mass(n,prob));return c.get(n.id)}
function draw(n,which){while(n.l)n=rr()<branch(n,which)?n.r:n.l;return n.i}
function value(n,i,which){let a=1;while(n.l){let b=branch(n,which),r=n.r.ids.includes(i);a*=r?b:1-b;n=r?n.r:n.l}return a}
function reveal(i){let j=front.findIndex(f=>f.n.ids.includes(i));let f=front.splice(j,1)[0],n=f.n,m=f.m;while(n.l){let b=branch(n,"p"),r=n.r.ids.includes(i);front.push({n:r?n.l:n.r,m:m*(r?1-b:b),exact:false});m*=r?b:1-b;n=r?n.r:n.l}front.push({n,m:Math.max(0,m-value(t,i,"q")),exact:true})}
let x=draw(t,"q"),px=value(t,x,"p"),qx=value(t,x,"q"),y=x,ok=rr()<Math.min(1,px/qx);if(!ok){miss=true;if(method==="adaptive")reveal(x);while(true){iters++;if(iters>cap){for(let n of nodes(t)){branch(n,"p");branch(n,"q")}y=cat(p.map((v,i)=>Math.max(0,v-q[i])),rr);break}
if(method==="adaptive"){let f=front[cat(front.map(f=>f.m),rr)];if(f.exact){y=f.n.i;break}let j=draw(f.n,"p"),a=value(t,j,"p"),b=value(t,j,"q");reveal(j);if(rr()<Math.max(0,1-b/a)){y=j;break}}
else {let j=draw(t,"p"),a=value(t,j,"p"),b=value(t,j,"q");if(rr()<Math.max(0,1-b/a)){y=j;break}}}}
return {x,y,ok,miss,iters,p:pc.size,q:qc.size}}

return (function certificateRun(cohorts){
let cfg=new Map();for(let c of cohorts){let k=c.wi+","+c.seed;if(!cfg.has(k))cfg.set(k,[]);cfg.get(k).push(c)}let rows=[],audit={draws:0,violations:0,maxViolation:0,maxMassError:0};
function table(t,wp,wq){let out=new Map();function go(n){if(!n.l)return [zero(D),zero(D)];let L=go(n.l),R=go(n.r),p=wp.get(n.id),q=wq.get(n.id),z=[p.map((v,i)=>Math.abs(v)+Math.max(L[0][i],R[0][i])),p.map((v,i)=>Math.abs(v-q[i])+Math.max(L[1][i],R[1][i]))];out.set(n.id,z);return z}go(t);return out}
function run(t,p,q,rr,mode,bounds,dh,s){
let pc=new Map(),qc=new Map(),front=[],iters=0,evalBound=0;
function branch(n,which){let c=which==="p"?pc:qc,pr=which==="p"?p:q;if(!c.has(n.id))c.set(n.id,mass(n.r,pr)/mass(n,pr));return c.get(n.id)}
function draw(n,which){while(n.l)n=rr()<branch(n,which)?n.r:n.l;return n.i}
function value(n,i,which){let a=1;while(n.l){let b=branch(n,which),r=n.r.ids.includes(i);a*=r?b:1-b;n=r?n.r:n.l}return a}
function piece(n,pm,qm,limit=1){if(!n.l)return {n,pm,qm,g:Math.max(0,1-qm/pm),exact:true};
let g=1;if(mode==="cert"){let b=bounds.get(n.id),B=dot(b[0],dh)+dot(b[1],s);g=Math.max(0,1-(qm/pm)*Math.exp(-B));evalBound++}
if(mode==="oracle"){g=Math.max(...n.ids.map(i=>Math.max(0,1-q[i]/p[i])));evalBound++}
g=Math.min(limit,g);let trueG=Math.max(...n.ids.map(i=>Math.max(0,1-q[i]/p[i])));if(g+1e-12<trueG){audit.violations++;audit.maxViolation=Math.max(audit.maxViolation,trueG-g)}return {n,pm,qm,g,exact:false}}
function reveal(i){let j=front.findIndex(f=>f.n.ids.includes(i)),f=front.splice(j,1)[0],{n,pm,qm,g}=f;while(n.l){let bp=branch(n,"p"),bq=branch(n,"q"),r=n.r.ids.includes(i);front.push(piece(r?n.l:n.r,pm*(r?1-bp:bp),qm*(r?1-bq:bq),g));pm*=r?bp:1-bp;qm*=r?bq:1-bq;n=r?n.r:n.l;g=piece(n,pm,qm,g).g}front.push(piece(n,pm,qm,g))}
let x=draw(t,"q"),px=value(t,x,"p"),qx=value(t,x,"q"),y=x,ok=rr()<Math.min(1,px/qx),rootG=0;
if(!ok){front=[piece(t,1,1)];rootG=front[0].g;reveal(x);while(true){iters++;let f=front[cat(front.map(f=>f.pm*f.g),rr)];if(f.exact){y=f.n.i;break}let j=draw(f.n,"p"),a=value(t,j,"p"),b=value(t,j,"q"),g=Math.max(0,1-b/a),accept=rr()<g/f.g;reveal(j);if(accept){y=j;break}}}
return {y,p:pc.size,q:qc.size,iters,evalBound,ok:Number(ok),rootG}}
for(let [key,cs] of cfg){let train=cs.slice(0,8).flatMap(c=>c.xs),test=cs.slice(8).flatMap(c=>c.xs);for(let kind of ["huffman","random","signed"]){let t=makeTree(train,kind,Number(key.split(",")[1])),wp=fit(t,train,"p"),wq=fit(t,train,"q"),B=table(t,wp,wq),row={key,kind,methods:{}};for(let mode of ["adaptive","cert","oracle"])row.methods[mode]={p:0,q:0,iters:0,max:0,evalBound:0,accept:0,rootG:0,rej:0,empTV:0};
for(let z of test){let h=[...z.h,1],s=[...z.s,1],p=distribution(t,wp,h),q=distribution(t,wq,s),dh=h.map((v,i)=>Math.abs(v-s[i])),sa=s.map(Math.abs);
for(let mode of ["adaptive","cert","oracle"]){let rr=rng(130+Number(key.split(",")[1])+z.y*13),st=row.methods[mode],cnt=zero(V),R=256;for(let rep=0;rep<R;rep++){let a=run(t,p,q,rr,mode,B,dh,sa),w=1/(test.length*R);st.p+=a.p*w;st.q+=a.q*w;st.iters+=a.iters*w;st.evalBound+=a.evalBound*w;st.accept+=a.ok*w;st.rootG+=a.rootG*w;st.rej+=(1-a.ok)*w;st.max=Math.max(st.max,a.iters);cnt[a.y]++;audit.draws++}st.empTV+=sum(p.map((v,i)=>Math.abs(v-cnt[i]/R)))/(2*test.length)}}rows.push(row)}}
let summary=["huffman","random","signed"].map(kind=>{let r=rows.filter(x=>x.kind===kind);return {kind,methods:Object.fromEntries(["adaptive","cert","oracle"].map(m=>[m,Object.fromEntries(Object.keys(r[0].methods[m]).map(k=>[k,k==="max"?Math.max(...r.map(x=>x.methods[m][k])):mean(r.map(x=>x.methods[m][k]))]))]))}});return {summary,rows,audit}})(cohorts)}
console.log(JSON.stringify(certificateStudy(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
