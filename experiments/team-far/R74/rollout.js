function run(raws,names,rep=6){
const V=26,d=16,H=4,B=3,T=6,sum=a=>a.reduce((s,v)=>s+v,0),avg=a=>sum(a)/a.length,dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),rms=x=>{let a=1/Math.sqrt(dot(x,x)/d+1e-5);return x.map(v=>v*a)},sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m)),s=sum(v);return v.map(t=>t/s)};
function rng(seed){let a=seed>>>0;return ()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^a>>>15,1|a);t^=t+Math.imul(t^t>>>7,61|t);return (((t^t>>>14)>>>0)+.5)/4294967296}}
function draw(a,r){let u=r()*sum(a);for(let j=0;j<a.length;j++){u-=a[j];if(u<0)return j}return a.findLastIndex(v=>v>0)}
let forwards=0,featureAudit=0,gradAudit=0,updates=0,raceMax=0,signViolations=0;const rows=[];
function model(raw){
let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));const E=take(27,d),P=take(16,d),W=take(27,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(64,d),f2:take(d,64)}));if(off!==7264)throw Error("shape");
let cache=new Map();
function get(h){let key=h.join(",");if(cache.has(key))return cache.get(key);let prev=h.length>1?get(h.slice(0,-1)):null,x=rms(add(E[h.at(-1)],P[h.length-1])),layers=[],sh;
for(let j=0;j<2;j++){let b=L[j],z=rms(x),q=mat(b.q,z),k=mat(b.k,z),v=mat(b.v,z),ks=(prev?prev.layers[j].ks:[]).concat([k]),vs=(prev?prev.layers[j].vs:[]).concat([v]),a=[];
for(let t=0;t<H;t++){let o=t*4,w=sm(ks.map(k=>dot(q.slice(o,o+4),k.slice(o,o+4))/2));for(let l=o;l<o+4;l++)a.push(sum(w.map((p,i)=>p*vs[i][l])))}
let y=add(x,mat(b.o,a));x=add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))));layers.push({ks,vs});if(j===0)sh=x}
let p=sm(mat(W.slice(0,V),x)),q=sm(mat(W.slice(0,V),sh)),a={key,p,q,deep:x,layers};cache.set(key,a);forwards++;return a}
function full(h){let seq=h.map((t,i)=>rms(add(E[t],P[i]))),sh;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*4,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+4),x.slice(t,t+4))/2));for(let l=t;l<t+4;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!sh)sh=seq.at(-1)}return {p:sm(mat(W.slice(0,V),seq.at(-1))),q:sm(mat(W.slice(0,V),sh))}}
return {get,full,W,clear:()=>cache.clear()}}
let words=[...new Set(names.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
const methods=["AR","tokenSD","blockSD","frontier","rootToken","rootBlock","edgeBlock"];
function episode(m,anchors,N,method,r){
let lanes=anchors.map((a,i)=>({h:a.slice(),out:[],e:i/(N/4)|0,comm:[],blocks:0})),z={rounds:0,laneBlocks:0,targetRows:0,draftRows:0,edgeRows:0,globalRows:0,peakRows:0,cloudRepair:0,edgeRepair:0,laneRepair:0,changed:0,accepted:0,verifyHops:0,raceTrials:0,blocks:0,raceCalls:0,rootRaceMax:0,tokens:N*T,committedUnique:0,firstVowel:0,pairVowel:0,meanVowel:0},done=l=>l.out.length===T;
function plan(i){let l=lanes[i],k=method==="AR"?0:Math.min(B,T-l.out.length-1),h=l.h.slice(),ss=[],xs=[];for(let j=0;j<=k;j++){let a=m.get(h);ss.push(a);if(j<k){let x=draw(a.q,r);xs.push(x);h=h.concat(x)}}l.blocks++;z.laneBlocks++;z.targetRows+=k+1;z.draftRows+=k;return {i,k,ss,xs,alive:true}}
function emit(a,j,x,accepted){let l=lanes[a.i],s=a.ss[j];if(s.key!==l.h.join(","))throw Error("wrong prefix");l.comm.push({s,x});l.out.push(x);l.h.push(x);if(accepted)z.accepted++}
function residual(a,w=1){return a.p.map((v,j)=>Math.max(0,w*v-a.q[j]))}
function couple(items,kind){
let y=items.map(a=>a.x);
function lr(n){return n.ids.reduce((s,k)=>s+Math.log(items[k].p[y[k]]/items[k].q[y[k]]),0)}
function force(n){if(lr(n)>1e-9)signViolations++;if(!n.ch){let j=n.ids[0],a=items[j];y[j]=draw(a.p.map((v,k)=>Math.max(0,v-a.q[k])),r);return}
for(let t=0;t<n.ch.length;t++){let ls=n.ch.map(lr),pref=1,ws=ls.map(v=>{let a=Math.exp(Math.min(0,v)),w=pref*(1-a);pref*=a;return w});if(sum(ws)<=0)throw Error("empty nonidentity");let f=draw(ws,r);for(let j=f;j<ls.length;j++)if(j===f||r()>=Math.exp(Math.min(0,ls[j])))force(n.ch[j]);z.verifyHops++;let now=n.ch.map(lr),L=sum(now),den=-Math.expm1(sum(now.map(v=>Math.min(0,-v)))),stop=L>0?-Math.expm1(-L)/den:0;if(r()<Math.min(1,stop))return}throw Error("cycle")}
let leaves=items.map((_,i)=>({ids:[i]}));
let ch=kind==="nested"?[0,1,2,3].map(e=>leaves.filter(l=>items[l.ids[0]].e===e)).filter(a=>a.length).map(a=>a.length===1?a[0]:({ids:a.flatMap(x=>x.ids),ch:a})):leaves;
if(kind==="independent"){for(let l of leaves)if(r()>=Math.exp(Math.min(0,lr(l))))force(l)}else{let root=ch.length===1?ch[0]:{ids:leaves.flatMap(x=>x.ids),ch};if(r()>=Math.exp(Math.min(0,lr(root))))force(root)}return y}
function block(group,implicit){
let k=group[0].k;if(group.some(a=>a.k!==k))throw Error("unaligned block");z.blocks++;
let w=[1];for(let j=0;j<k;j++){let lr=sum(group.map(a=>Math.log(a.ss[j].p[a.xs[j]]/a.ss[j].q[a.xs[j]])));w.push(Math.exp(Math.min(0,Math.log(w[j])+lr)))}
if(r()<w[k])return {tau:k,ys:group.map(a=>draw(a.ss[k].p,r))};
for(let j=k-1;j>=0;j--){
if(!implicit||group.length===1){let a=group[0],weights=residual(a.ss[j],w[j]),S=sum(weights),den=S+1-w[j];if(den<=0)throw Error("zero race");if(r()<S/den)return {tau:j,ys:[draw(weights,r)]};continue}
z.raceCalls++;let trials=0;for(;;){trials++;z.raceTrials++;if(trials>1000000)throw Error("race guard");if(r()>=w[j])break;let ys=group.map(a=>draw(a.ss[j].p,r)),lr=sum(group.map((a,i)=>Math.log(a.ss[j].q[ys[i]]/a.ss[j].p[ys[i]])))-Math.log(w[j]);if(lr<0&&r()<-Math.expm1(lr)){raceMax=Math.max(raceMax,trials);z.rootRaceMax=Math.max(z.rootRaceMax,trials);return {tau:j,ys}}}
raceMax=Math.max(raceMax,trials);z.rootRaceMax=Math.max(z.rootRaceMax,trials)}
throw Error("no output")}
while(lanes.some(l=>!done(l))){z.rounds++;let plans=lanes.map((l,i)=>done(l)?null:plan(i)).filter(Boolean),edgeSets=[0,1,2,3].map(()=>new Set()),global=new Set(),repair=new Set();for(let a of plans)for(let s of a.ss){edgeSets[lanes[a.i].e].add(s.key);global.add(s.key)}z.edgeRows+=sum(edgeSets.map(s=>s.size));z.globalRows+=global.size;z.peakRows=Math.max(z.peakRows,global.size);
function repairOne(a,j,x){z.laneRepair++;z.changed+=+(a.xs[j]!==x);repair.add(lanes[a.i].e)}
if(method==="blockSD"||method==="rootBlock"||method==="edgeBlock"){
let groups=method==="blockSD"?plans.map(a=>[a]):method==="rootBlock"?[plans]:[0,1,2,3].map(e=>plans.filter(a=>lanes[a.i].e===e)).filter(g=>g.length);
for(let g of groups){let b=block(g,method!=="blockSD");for(let i=0;i<g.length;i++){let a=g[i];for(let j=0;j<b.tau;j++)emit(a,j,a.xs[j],true);emit(a,b.tau,b.ys[i],false);if(b.tau<a.k)repairOne(a,b.tau,b.ys[i])}}
}else{
for(let j=0;j<=B;j++){let g=plans.filter(a=>a.alive&&j<a.k);if(g.length){let items=g.map(a=>({p:a.ss[j].p,q:a.ss[j].q,x:a.xs[j],e:lanes[a.i].e})),ys=couple(items,method==="frontier"?"nested":method==="rootToken"?"star":"independent"),any=ys.some((x,i)=>x!==g[i].xs[j]);for(let i=0;i<g.length;i++){let a=g[i],ok=ys[i]===a.xs[j];emit(a,j,ys[i],ok);if(!ok||method==="rootToken"&&any){repairOne(a,j,ys[i]);a.alive=false}}}
for(let a of plans)if(a.alive&&j===a.k){emit(a,j,draw(a.ss[j].p,r),false);a.alive=false}}
}
z.cloudRepair+=+(repair.size>0);z.edgeRepair+=repair.size}
const vowel=x=>[0,4,8,14,20].includes(x),weights=lanes.map(l=>sum(l.out.map(vowel))),adv=weights.map((v,i)=>{let a=weights.filter((_,j)=>lanes[j].e===lanes[i].e);return v-avg(a)}),grouped=new Map(),g=Array(V*d).fill(0),gg=Array(V*d).fill(0);
function accum(s,coef,out){for(let v=0;v<V;v++)for(let j=0;j<d;j++)out[v*d+j]+=coef[v]*s.deep[j]}
for(let i=0;i<N;i++)for(let {s,x} of lanes[i].comm){let c=s.p.map((v,k)=>adv[i]*(+(k===x)-v));accum(s,c,g);if(!grouped.has(s.key))grouped.set(s.key,{s,w:0,c:Array(V).fill(0)});let a=grouped.get(s.key);a.w+=adv[i];a.c[x]+=adv[i]}
for(let {s,w,c} of grouped.values())accum(s,c.map((v,j)=>v-w*s.p[j]),gg);
gradAudit=Math.max(gradAudit,...g.map((v,i)=>Math.abs(v-gg[i])));updates++;
z.committedUnique=grouped.size;z.firstVowel=avg(lanes.map(l=>+vowel(l.out[0])));z.pairVowel=+(vowel(lanes[0].out[0])&&vowel(lanes[N/2].out[0]));z.meanVowel=avg(weights)/T;return z}
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]);for(let seed of [11,29,73]){let data=words.slice().sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(512,520);
for(let s of data){let h=[26,...s.slice(0,4).split("").map(x=>x.charCodeAt(0)-97)],a=m.get(h),b=m.full(h);featureAudit=Math.max(featureAudit,...a.p.map((v,i)=>Math.abs(v-b.p[i])),...a.q.map((v,i)=>Math.abs(v-b.q[i])))}
for(let N of [8,16])for(let scenario of ["sameRegion","mixed","paired"]){m.clear();let vals=Object.fromEntries(methods.map(a=>[a,[]]));for(let t=0;t<rep;t++){let anchors=Array.from({length:N},(_,i)=>{let e=i/(N/4)|0,j=i%(N/4),name=data[(scenario==="sameRegion"?e:scenario==="mixed"?e+j:e*2+j%2)+t%2&7];return [26,...name.slice(0,2).split("").map(x=>x.charCodeAt(0)-97)]});for(let method of methods){let r=rng(1234567+wi*77777+seed*137+N*877+scenario.length*555+t*3119+methods.indexOf(method)*29989);vals[method].push(episode(m,anchors,N,method,r))}}
for(let method of methods){let a=vals[method],z={wi,seed,N,scenario,method,rep};for(let k of Object.keys(a[0]))z[k]=k==="rootRaceMax"?Math.max(...a.map(v=>v[k])):avg(a.map(v=>v[k]));rows.push(z)}}}}
let summary=[];for(let N of [8,16])for(let method of methods){let a=rows.filter(x=>x.N===N&&x.method===method),z={N,method,n:a.length};for(let k of Object.keys(a[0]).filter(k=>!["wi","seed","N","scenario","method","rep"].includes(k)))z[k]=k==="rootRaceMax"?Math.max(...a.map(v=>v[k])):avg(a.map(v=>v[k]));summary.push(z)}
return {rep,configs:rows.length/methods.length,episodes:rows.length*rep,lanes:sum(rows.map(x=>x.N*rep)),tokens:sum(rows.map(x=>x.tokens*rep)),forwards,featureAudit,gradAudit,updates,raceMax,signViolations,summary,rows}}
if(require.main===module){const fs=require("fs");console.log(JSON.stringify(run([JSON.parse(fs.readFileSync(process.argv[2],"utf8")),JSON.parse(fs.readFileSync(process.argv[3],"utf8")).weights],fs.readFileSync(process.argv[4],"utf8"),6),null,2))}
