function run(raws,names,stage){
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
return {p,q,kv:kv.map(a=>({k:a.k.v,v:a.v.v})),bytes,back(g,adj,G,gd){for(let o of ops)o.g.fill(0);for(let j=0;j<V;j++){logits.g[j]+=g[j];draft.g[j]+=gd?gd[j]:0;}for(let l=0;l<2;l++)for(let j=0;j<d;j++){kv[l].k.g[j]+=adj[l].k[j];kv[l].v.g[j]+=adj[l].v[j]}for(let i=ops.length-1;i>=0;i--)if(ops[i].b)ops[i].b(ops[i].g,G);backCalls++;return refs.map(r=>({a:r.a,l:r.l,k:r.pk.g,v:r.pv.g}))}}}
function engine(raw){let cache=new Map(),calls=0;
function get(h){let key=h.join(",");if(cache.has(key))return cache.get(key);let anc=[];for(let j=1;j<h.length;j++)anc.push(get(h.slice(0,j)));let tape=local(raw,h,anc),a={key,h:h.slice(),anc,kv:tape.kv,p:tape.p,q:tape.q,tape,bytes:tape.bytes};cache.set(key,a);calls++;return a}
return {raw,cache,get,calls:()=>calls}}

let early=Array.from({length:7264},(_,i)=>i<W.o||i>=L[0].q.o&&i<L[1].q.o);
function ce(w,examples){let e=engine(w),G=zero(7264),g=new Map();for(let {h,y} of examples){let s=e.get(h);for(let i=1;i<=h.length;i++)closure.add(h.slice(0,i).join(","));if(!g.has(s.key))g.set(s.key,zero(V));let a=g.get(s.key);for(let j=0;j<V;j++)a[j]+=(s.p[j]-+(j===y))/examples.length}
let nodes=[...e.cache.values()].filter(s=>closure.has(s.key)),adj=new Map(nodes.map(s=>[s.key,Array.from({length:2},()=>({k:zero(d),v:zero(d)}))]));for(let s of nodes.sort((a,b)=>b.h.length-a.h.length))for(let o of s.tape.back(g.get(s.key)||zero(V),adj.get(s.key),G)){let z=adj.get(o.a.key)[o.l];for(let j=0;j<d;j++){z.k[j]+=o.k[j];z.v[j]+=o.v[j]}}return G}

const words=[...new Set(names.split(/\s+/).filter(x=>/^[a-z]{8,14}$/.test(x)))];
const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
const isShared=Array.from({length:7264},(_,j)=>j<L[1].q.o);
let results=[],gradUpdates=0,tokenTotal=0,fd=0,gradErr=0,reuseGradErr=0,reuseForward=0,replayForward=0,reuseBack=0,replayBack=0;
function policyGrad(raw,examples,existing){let e=existing||engine(raw),G=zero(7264),g=new Map(),closure=new Set();for(let {h,y,w} of examples){let s=e.get(h);if(!g.has(s.key))g.set(s.key,zero(V));let a=g.get(s.key);for(let j=0;j<V;j++)a[j]+=w*(s.p[j]-+(j===y))/examples.length;}let nodes=[...e.cache.values()],adj=new Map(nodes.map(s=>[s.key,Array.from({length:2},()=>({k:zero(d),v:zero(d)}))]));for(let s of nodes.sort((a,b)=>b.h.length-a.h.length))for(let o of s.tape.back(g.get(s.key)||zero(V),adj.get(s.key),G)){let z=adj.get(o.a.key)[o.l];for(let j=0;j<d;j++){z.k[j]+=o.k[j];z.v[j]+=o.v[j];}}return {G,keys:[...closure]}}
function prefixes(h,set){for(let i=1;i<=h.length;i++)set.add(h.slice(0,i).join(","))}
function rollout(e,prompt,r){let h=prompt.slice(),out=[],D=new Set(),C=new Set(),qKeys=new Set(),rounds=0,accepted=0,attempt=0;
while(out.length<6){let K=Math.min(2,6-out.length),candidates=[],t=h.slice();for(let j=0;j<K;j++){let q=e.get(t).q;qKeys.add(t.join(","));let x=draw(q,r);candidates.push({h:t.slice(),q,x});t.push(x);}
for(let a of candidates){e.get(a.h);prefixes(a.h,D)}if(out.length+K<6){e.get(t);prefixes(t,D);}rounds++;let all=true;
for(let a of candidates){let p=e.get(h).p;attempt++;prefixes(h,C);let y=a.x;if(r()<=Math.min(1,p[y]/a.q[y]))accepted++;else{let res=p.map((v,j)=>Math.max(0,v-a.q[j]));y=draw(res,r);all=false;}out.push({h:h.slice(),y});h.push(y);if(!all||out.length===6)break;}
if(all&&out.length<6){let p=e.get(h).p;prefixes(h,C);let y=draw(p,r);out.push({h:h.slice(),y});h.push(y);}
}return {out,D,C,qKeys,rounds,accepted,attempt}}
const union=sets=>new Set(sets.flatMap(s=>[...s]));
for(let wi=0;wi<raws.length;wi++)for(let seed of [17,29,43])for(let mode of ["common","regional","private"]){
let base=raws[wi].slice(),ww=Array.from({length:4},()=>base.slice()),train=Array.from({length:4},(_,c)=>words.filter(x=>hash(x,seed)%4===c).slice(0,64)),held=Array.from({length:4},(_,c)=>words.filter(x=>hash(x,seed)%4===c).slice(128,144));
const chars=x=>[26,...[...x].map(c=>c.charCodeAt(0)-97)];
for(let epoch=0;epoch<3;epoch++){let C=[],D=[],Q=[],gKeys=[],grads=[],rounds=0,attempt=0,accepted=0,allTokens=0,reward=0,heldNll=0,actualOutputDiff=0,gs=[];
for(let c=0;c<4;c++){let e=engine(ww[c]),cs=new Set(),ds=new Set(),qs=new Set(),examples=[];for(let k=0;k<6;k++){
let name=mode==="common"?train[0][k]:mode==="regional"?train[2*Math.floor(c/2)][k]:train[c][k],h=chars(name.slice(0,2));
let a=rollout(e,h,rng(seed+wi*10007+epoch*1031+c*101+k*11));for(let x of a.C)cs.add(x);for(let x of a.D)ds.add(x);for(let x of a.qKeys)qs.add(x);let target=name.charCodeAt(2)-97;
let R=a.out.reduce((s,v,j)=>s+Number(v.y===name.charCodeAt(j+2)-97),0)/6;reward+=R;for(let item of a.out)examples.push({...item,w:R});allTokens+=a.out.length;rounds+=a.rounds;attempt+=a.attempt;accepted+=a.accepted;
}
let f0=forwardCalls,b0=backCalls,pg=policyGrad(ww[c],examples,e);reuseForward+=forwardCalls-f0;reuseBack+=backCalls-b0;f0=forwardCalls;b0=backCalls;let direct=policyGrad(ww[c],examples);replayForward+=forwardCalls-f0;replayBack+=backCalls-b0;reuseGradErr=Math.max(reuseGradErr,maxdiff(pg.G,direct.G));gs.push(pg.G);gKeys.push(new Set(pg.keys));C.push(cs);D.push(ds);Q.push(qs);gradUpdates++;
for(let name of held[c]){let h=chars(name.slice(0,2)),y=name.charCodeAt(2)-97;heldNll-=Math.log(e.get(h).p[y])/64;}
}
for(let i=0;i<4;i++)for(let k of C[i])if(!D[i].has(k))throw Error("committed not verified");
const count=(sets,group)=>group==="global"?union(sets).size:group==="region"?union(sets.slice(0,2)).size+union(sets.slice(2,4)).size:sum(sets.map(x=>x.size));
const stats={};
for(let m of ["AR","SD"]){let X=m==="AR"?C:D,Y=gKeys;
let U=X.map((x,c)=>union([x,Y[c]]));stats[m]={clientSh:count(U,"client"),sharedSh:count(U,"global"),clientDeep:count(U,"client"),regionDeep:count(U,"region"),trainSh:count(Y,"global"),trainDeep:count(Y,"region"),qCalls:m==="SD"?count(Q,"global"):0,rounds:m==="SD"?rounds:allTokens};}
const sharedDiff=ww[0].reduce((m,x,j)=>isShared[j]?Math.max(m,...ww.map(w=>Math.abs(w[j]-x))):m,0);let regionDiff=Math.max(maxdiff(ww[0],ww[1]),maxdiff(ww[2],ww[3]));
if(sharedDiff!==0||regionDiff!==0)throw Error("version invariant");
// Compare separate-client formal gradients to direct joint regional computation on same collected examples? Each client G already exact. Linear tree reduction checked against direct flat average.
let flat=Array.from({length:7264},(_,j)=>avg(gs.map(g=>g[j]))),edge=[0,1].map(e=>Array.from({length:7264},(_,j)=>(gs[e*2][j]+gs[e*2+1][j])/2)),tree=edge[0].map((v,j)=>(v+edge[1][j])/2);gradErr=Math.max(gradErr,maxdiff(flat,tree));
for(let c=0;c<4;c++)ww[c]=ww[c].map((v,j)=>v-.05*(isShared[j]?tree[j]:edge[Math.floor(c/2)][j]));
tokenTotal+=allTokens;results.push({wi,seed,mode,epoch,heldNll,reward:reward/24,allTokens,rounds,attempt,accepted,sharedDiff,regionDiff,stats});
}
}
let summary=[];for(let mode of ["common","regional","private"]){let a=results.filter(x=>x.mode===mode);let get=(m,k)=>avg(a.map(r=>r.stats[m][k]));let o={mode,n:a.length,accept:sum(a.map(r=>r.accepted))/sum(a.map(r=>r.attempt)),tokens:avg(a.map(r=>r.allTokens)),nll:avg(a.map(r=>r.heldNll)),reward:avg(a.map(r=>r.reward))};for(let m of ["AR","SD"])o[m]=Object.fromEntries(["clientSh","sharedSh","clientDeep","regionDeep","trainSh","trainDeep","qCalls","rounds"].map(k=>[k,get(m,k)]));summary.push(o)}
return {spec:{mode:"fixed length stochastic SD, two same-size external character transformer checkpoints, 4 logical clients / 2 edges, all 3 rounds, shared embedding + first transformer layer + output head, regional second layer. Actual full model p/q computation and analytical reward-weighted gradient, no actual split execution, no physical network GPU or timing. AR is counterfactual replay of identical formal samples, not independent AR sampling."},results,summary,counts:{gradUpdates,tokenTotal,forwardCalls,backCalls,gradErr,reuseGradErr,reuseForward,replayForward,reuseBack,replayBack}};
}

if(require.main===module){const fs=require('fs');console.log(JSON.stringify(run([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'),1),null,2));}
