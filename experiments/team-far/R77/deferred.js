function run(raws,names){
const d=16,V=26,T=4,B=2,N=4,sum=a=>a.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),avg=a=>sum(a)/a.length,zero=n=>new Float64Array(n),maxdiff=(a,b)=>Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
const soft=a=>{let m=Math.max(...a),b=Array.from(a,v=>Math.exp(v-m)),s=sum(b);return b.map(v=>v/s)};
function rng(seed){let a=seed>>>0;return ()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^a>>>15,1|a);t^=t+Math.imul(t^t>>>7,61|t);return (((t^t>>>14)>>>0)+.5)/4294967296}}
function draw(p,r){let u=r()*sum(p);for(let j=0;j<p.length;j++){u-=p[j];if(u<0)return j}return p.findLastIndex(v=>v>0)}
let off=0;function take(n,m){let z={o:off,n,m};off+=n*m;return z}let E=take(27,d),P=take(16,d),W=take(27,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(64,d),f2:take(d,64)}));if(off!==7264)throw Error("layout");
let forwardCalls=0,backCalls=0,maxGradErr=0,maxFDErr=0,maxReplay=0,updates=0,rows=[];
function local(raw,h,anc){
let ops=[],bytes=0;function node(v,b){let o={v:Array.from(v),g:zero(v.length),b};ops.push(o);bytes+=8*v.length;return o}
function plus(a,b){return node(a.v.map((v,j)=>v+b.v[j]),g=>{for(let j=0;j<g.length;j++){a.g[j]+=g[j];b.g[j]+=g[j]}})}
function row(A,i){return node(raw.slice(A.o+i*A.m,A.o+(i+1)*A.m),(g,G)=>{for(let j=0;j<g.length;j++)G[A.o+i*A.m+j]+=g[j]})}
function mat(A,x,n=A.n){return node(Array.from({length:n},(_,i)=>dot(raw.slice(A.o+i*A.m,A.o+(i+1)*A.m),x.v)),(g,G)=>{for(let i=0;i<n;i++)for(let j=0;j<A.m;j++){let k=A.o+i*A.m+j;G[k]+=g[i]*x.v[j];x.g[j]+=g[i]*raw[k]}})}
function rms(a){let f=1/Math.sqrt(dot(a.v,a.v)/d+1e-5);return node(a.v.map(v=>v*f),g=>{let c=dot(Array.from(g),a.v)*f*f*f/d;for(let j=0;j<d;j++)a.g[j]+=f*g[j]-c*a.v[j]})}
function relu(a){return node(a.v.map(v=>Math.max(0,v)),g=>{for(let j=0;j<g.length;j++)a.g[j]+=g[j]*(a.v[j]>0)})}
function attention(q,ks,vs){let out=[],as=[];for(let head=0;head<4;head++){let o=head*4,a=soft(ks.map(k=>dot(q.v.slice(o,o+4),k.v.slice(o,o+4))/2));as.push(a);for(let j=o;j<o+4;j++)out.push(sum(a.map((w,i)=>w*vs[i].v[j])))}bytes+=8*sum(as.map(a=>a.length));return node(out,g=>{for(let head=0;head<4;head++){let o=head*4,a=as[head],v=vs.map(x=>dot(Array.from(g.slice(o,o+4)),x.v.slice(o,o+4))),mu=dot(a,v);for(let i=0;i<ks.length;i++){let c=a[i]*(v[i]-mu)/2;for(let j=o;j<o+4;j++){q.g[j]+=c*ks[i].v[j];ks[i].g[j]+=c*q.v[j];vs[i].g[j]+=a[i]*g[j]}}}})}
let x=rms(plus(row(E,h.at(-1)),row(P,h.length-1))),kv=[],refs=[],sh;
for(let l=0;l<2;l++){let b=L[l],z=rms(x),q=mat(b.q,z),k=mat(b.k,z),v=mat(b.v,z),ks=[],vs=[];for(let a of anc){let pk=node(a.kv[l].k),pv=node(a.kv[l].v);refs.push({a,l,pk,pv});ks.push(pk);vs.push(pv)}ks.push(k);vs.push(v);let y=plus(x,mat(b.o,attention(q,ks,vs)));x=plus(y,mat(b.f2,relu(mat(b.f1,rms(y)))));kv.push({k,v});if(l===0)sh=x}
let logits=mat(W,x,V),draft=mat(W,sh,V),p=soft(logits.v),q=soft(draft.v);forwardCalls++;
return {p,q,kv:kv.map(a=>({k:a.k.v,v:a.v.v})),bytes,back(g,adj,G){for(let o of ops)o.g.fill(0);for(let j=0;j<V;j++)logits.g[j]+=g[j];for(let l=0;l<2;l++)for(let j=0;j<d;j++){kv[l].k.g[j]+=adj[l].k[j];kv[l].v.g[j]+=adj[l].v[j]}for(let i=ops.length-1;i>=0;i--)if(ops[i].b)ops[i].b(ops[i].g,G);backCalls++;return refs.map(r=>({a:r.a,l:r.l,k:r.pk.g,v:r.pv.g}))}}}
function engine(raw){let cache=new Map(),calls=0;
function get(h){let key=h.join(",");if(cache.has(key))return cache.get(key);let anc=[];for(let j=1;j<h.length;j++)anc.push(get(h.slice(0,j)));let tape=local(raw,h,anc),a={key,h:h.slice(),anc,kv:tape.kv,p:tape.p,q:tape.q,tape,bytes:tape.bytes};cache.set(key,a);calls++;return a}
return {raw,cache,get,calls:()=>calls}}

let early=Array.from({length:7264},(_,i)=>i<W.o||i>=L[0].q.o&&i<L[1].q.o);
function ce(w,examples){let e=engine(w),G=zero(7264),g=new Map();for(let {h,y} of examples){let s=e.get(h);if(!g.has(s.key))g.set(s.key,zero(V));let a=g.get(s.key);for(let j=0;j<V;j++)a[j]+=(s.p[j]-+(j===y))/examples.length}
let nodes=[...e.cache.values()],adj=new Map(nodes.map(s=>[s.key,Array.from({length:2},()=>({k:zero(d),v:zero(d)}))]));for(let s of nodes.sort((a,b)=>b.h.length-a.h.length))for(let o of s.tape.back(g.get(s.key)||zero(V),adj.get(s.key),G)){let z=adj.get(o.a.key)[o.l];for(let j=0;j<d;j++){z.k[j]+=o.k[j];z.v[j]+=o.v[j]}}return G}
function cache(){return {old:null,sh:new Map(),refresh:0,versions:0,hits:0,miss:0,deep:0,rows:0,rounds:0,accepted:0,attempts:0,tokens:0,drift:0}}
let maxPQ=0,training=0,trajectories=0,edgeChanges=0,cloudChanges=0;
function serve(w,store,anchors,method,rr){let fresh=engine(w),memo=new Map();
let changed=!store.old||w.some((v,i)=>early[i]&&v!==store.old[i]);if(changed){store.sh.clear();store.refresh++}store.old=w.slice();store.versions++;
let mul=(A,x,n=A.n)=>Array.from({length:n},(_,i)=>dot(w.slice(A.o+i*A.m,A.o+(i+1)*A.m),x)),row=(A,i)=>w.slice(A.o+i*A.m,A.o+(i+1)*A.m),add=(a,b)=>a.map((v,j)=>v+b[j]),norm=x=>{let a=1/Math.sqrt(dot(x,x)/d+1e-5);return x.map(v=>v*a)};
function layer(x,l,anc){let b=L[l],z=norm(x),q=mul(b.q,z),k=mul(b.k,z),v=mul(b.v,z),ks=anc.map(a=>a.k).concat([k]),vs=anc.map(a=>a.v).concat([v]),a=[];for(let head=0;head<4;head++){let o=head*4,p=soft(ks.map(k=>dot(q.slice(o,o+4),k.slice(o,o+4))/2));for(let j=o;j<o+4;j++)a.push(sum(p.map((v,i)=>v*vs[i][j])))}let y=add(x,mul(b.o,a));return {x:add(y,mul(b.f2,mul(b.f1,norm(y)).map(v=>Math.max(0,v)))),k,v}}
function get(h){let key=h.join(",");if(memo.has(key))return memo.get(key);let anc=[];for(let i=1;i<h.length;i++)anc.push(get(h.slice(0,i)));let sh=store.sh.get(key);if(sh){store.hits++}else{sh=layer(norm(add(row(E,h.at(-1)),row(P,h.length-1))),0,anc.map(a=>a.sh));store.sh.set(key,sh);store.miss++}let dp=layer(sh.x,1,anc.map(a=>a.dp)),p=soft(mul(W,dp.x,V)),q=soft(mul(W,sh.x,V)),ref=fresh.get(h);maxPQ=Math.max(maxPQ,maxdiff(p,ref.p),maxdiff(q,ref.q));store.deep++;store.rows++;let a={sh,dp,p,q};memo.set(key,a);return a}
for(let h0 of anchors){let h=h0.slice(),t=0;while(t<T){store.rounds++;if(method==="AR"){h.push(draw(get(h).p,rr));t++;continue}
let k=Math.min(B,T-t-1),ss=[],xs=[],cur=h.slice(),a=[1];for(let j=0;j<=k;j++){let s=get(cur);ss.push(s);if(j<k){let x=draw(s.q,rr);xs.push(x);cur.push(x);a.push(Math.min(1,a[j]*s.p[x]/s.q[x]));store.attempts++}}
let j=k,y;if(rr()<a[k])y=draw(ss[k].p,rr);else{for(j=k-1;j>=0;j--){let r=ss[j].p.map((v,i)=>Math.max(0,a[j]*v-ss[j].q[i])),m=sum(r);if(rr()<m/(m+1-a[j])){y=draw(r,rr);break}}if(j<0)throw Error("BV")}
h.push(...xs.slice(0,j),y);t+=j+1;store.accepted+=j}store.tokens+=T;trajectories++}
}
let words=[...new Set(names.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
let results=[];for(let wi=0;wi<raws.length;wi++)for(let seed of [11,29,73])for(let scenario of ["repeat","rotate"])for(let mode of ["aligned","deferred","reverse","reverseDeferred","fullHalf"]){
let data=words.slice().sort((a,b)=>hash(a,seed)-hash(b,seed)),bank=Array.from({length:4},(_,c)=>data.filter(s=>Math.min(3,Math.floor((s.charCodeAt(0)-97)/7))===c).slice(0,96)),tests=data.slice(1200,1248),raw=raws[wi].slice(),stores=Object.fromEntries(["AR","BV"].map(m=>[m,[cache(),cache(),cache()]])),rrs=Object.fromEntries(["AR","BV"].map((m,i)=>[m,rng(seed+wi*5551+i*7009)])),active=Array.from({length:4},()=>[0,0]),changes=[0,0,0],losses=[];
if(bank.some(a=>a.length<32))throw Error("data");
function nll(w){let e=engine(w);return avg(tests.map(s=>-Math.log(e.get([26,...s.slice(0,2).split("").map(c=>c.charCodeAt(0)-97)]).p[s.charCodeAt(2)-97])))}
function visit(w,scope,step){let ix=(scenario==="repeat"?0:step*2)%40,anchors=Array.from({length:4},(_,j)=>[26,...tests[(ix+j)%tests.length].slice(0,2).split("").map(c=>c.charCodeAt(0)-97)]);for(let m of ["AR","BV"])serve(w,stores[m][scope],anchors,m,rrs[m])}
let initial=nll(raw);for(let s=0;s<3;s++)visit(raw,s,0);
for(let cloud=0;cloud<4;cloud++){let ew=[raw.slice(),raw.slice()],deferred=zero(7264);for(let cyc=0;cyc<2;cyc++)for(let e=0;e<2;e++){let w=ew[e],cs=[];for(let j=0;j<2;j++){let c=e*2+j,block=(cloud+(mode.startsWith("reverse")?1:0))%2,delay=mode.toLowerCase().includes("deferred")&&block===0;active[c][block]++;if(mode==="fullHalf")active[c][1-block]++;let ex=Array.from({length:4},(_,i)=>{let s=bank[c][(cloud*8+cyc*4+i)%bank[c].length],n=2+(i%3);return {h:[26,...s.slice(0,n).split("").map(c=>c.charCodeAt(0)-97)],y:s.charCodeAt(n)-97}}),g=ce(w,ex);if(delay)for(let k=0;k<w.length;k++)if(early[k])deferred[k]+=.025*g[k]/4;cs.push(w.map((v,k)=>mode==="fullHalf"?v-.0125*g[k]:!delay&&(early[k]?0:1)===block?v-.025*g[k]:v));training++}let nw=w.map((v,k)=>(cs[0][k]+cs[1][k])/2);changes[e]+=+(nw.some((v,k)=>early[k]&&v!==w[k]));ew[e]=nw;visit(nw,e,cloud*2+cyc+1)}
let nw=raw.map((v,k)=>(ew[0][k]+ew[1][k])/2-deferred[k]);changes[2]+=+(nw.some((v,k)=>early[k]&&v!==raw[k]));raw=nw;visit(raw,2,cloud+1);losses.push(nll(raw))}
for(let method of ["AR","BV"]){let z={wi,seed,scenario,mode,method,active,changes,initial,final:losses.at(-1),losses,paramDelta:Math.sqrt(sum(raw.map((v,k)=>(v-raws[wi][k])**2))),scopes:stores[method].map(({old,sh,...s})=>({...s,cacheEntries:sh.size}))};for(let k of ["refresh","hits","miss","deep","rows","rounds","accepted","attempts","tokens"])z[k]=sum(z.scopes.map(s=>s[k]));z.hit=z.hits/z.rows;results.push(z)}}
let summary=[];for(let scenario of ["repeat","rotate"])for(let mode of ["aligned","deferred","reverse","reverseDeferred","fullHalf"])for(let method of ["AR","BV"]){let a=results.filter(r=>r.scenario===scenario&&r.mode===mode&&r.method===method),z={scenario,mode,method,n:a.length};for(let k of ["initial","final","paramDelta","refresh","hits","miss","deep","rows","rounds","accepted","attempts","tokens","hit"])z[k]=avg(a.map(r=>r[k]));z.changes=[0,1,2].map(i=>avg(a.map(r=>r.changes[i])));summary.push(z)}return {results,summary,training,trajectories,tokens:trajectories*T,forwardCalls,backCalls,maxPQ}}

if(require.main===module){const fs=require('fs');console.log(JSON.stringify(run([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2))}
