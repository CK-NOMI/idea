function run(raws,names,partition){
const V=27,d=16,H=4,k=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))};
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let shallow;let seq=h.map((t,i)=>rms(add(E[t],P[i])));for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!shallow)shallow=seq.at(-1).slice()}return {deep:seq.at(-1),shallow}}}}



const avg=a=>sum(a)/a.length,hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h},words=[...new Set(names.split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))];
let rows=[],forwards=0,rollouts=0,maxNorm=0,maxFactor=0;
function rng(seed){let s=seed;return ()=>((s=Math.imul(s,1664525)+1013904223>>>0)/4294967296)}
function draw(p,u){for(let i=0;i<p.length;i++){u-=p[i];if(u<0)return i}return p.length-1}
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]);for(let seed of [11,29,73]){
let clients=Array.from({length:4},(_,r)=>words.filter(x=>{let v=x.charCodeAt(0)-97;return (v>=13?2:0)+v%2===r}).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(16,18));
for(let e=0;e<2;e++)for(let kind of ["parent","global","globalNested","regionalNested","regionalFree"]){let rec=partition.rows.find(x=>x.wi===wi&&x.seed===seed&&x.e===e&&x.beta===.25&&x.kind===kind),g=rec.g,K=rec.K;
for(let B of [2,4]){let r=rng(seed+wi*17+e*23+B*31),stat={};for(let mode of ["AR","SD","fixed4"])stat[mode]={round:0,rows:0,draft:0,emit:0,n:0};
for(let n of clients[2*e].concat(clients[2*e+1])){let pre=[26,...n.slice(0,3).split("").map(c=>c.charCodeAt(0)-97)],a=m.feature(pre);forwards++;let root=sm(mat(m.W,a.deep)),rep=Array(K).fill(-1),ids=Array.from({length:K},()=>[]);for(let i=0;i<V;i++){ids[g[i]].push(i);if(rep[g[i]]<0||root[i]>root[rep[g[i]]])rep[g[i]]=i}
let cache=new Map();
function classDist(h){let key=h.join(",");if(cache.has(key))return cache.get(key);let a=m.feature(pre.concat(h.map(c=>rep[c])));forwards++;let p=zero(K),q=zero(K);sm(mat(m.W,a.deep)).forEach((x,i)=>p[g[i]]+=x);sm(mat(m.W,a.shallow)).forEach((x,i)=>q[g[i]]+=x);maxNorm=Math.max(maxNorm,Math.abs(sum(p)-1),Math.abs(sum(q)-1));let z={p,q};cache.set(key,z);return z}
function emit(h,c,u){let a=m.feature(pre.concat(h));forwards++;let p=sm(ids[c].map(i=>dot(m.W[i],a.shallow)));maxNorm=Math.max(maxNorm,Math.abs(sum(p)-1));return ids[c][draw(p,u)]}
for(let t=0;t<16;t++){let tape=Array.from({length:B},()=>Array.from({length:5},r));for(let mode of ["AR","SD","fixed4"]){let h=[],fine=[],st=stat[mode];rollouts++;while(h.length<B){let k=mode==="AR"?0:Math.min(mode==="SD"?B-h.length-1:4,B-h.length),xs=[],z=h.slice();for(let j=0;j<k;j++){let d=classDist(z),x=draw(d.q,tape[h.length+j][0]);xs.push(x);z.push(x);st.draft++}st.round++;st.rows+=k+(h.length+k<B?1:0);let all=true;for(let x of xs){let d=classDist(h),u=tape[h.length],c=x;if(u[1]>=Math.min(1,d.p[x]/d.q[x])){let res=d.p.map((v,i)=>Math.max(0,v-d.q[i])),tot=sum(res);c=draw(res.map(v=>v/tot),u[2]);all=false}fine.push(emit(fine,c,u[3]));st.emit++;h.push(c);if(!all)break}
if(all&&h.length<B){let d=classDist(h),u=tape[h.length],c=draw(d.p,u[4]);fine.push(emit(fine,c,u[3]));st.emit++;h.push(c)}}st.n+=h.length}
}
}
rows.push({wi,seed,e,kind,B,stat})}
}
}}
let summary=[];for(let B of [2,4])for(let kind of ["parent","global","globalNested","regionalNested","regionalFree"]){let a=rows.filter(x=>x.B===B&&x.kind===kind);summary.push({B,kind,n:a.length,...Object.fromEntries(["AR","SD","fixed4"].map(mode=>[mode,Object.fromEntries(["round","rows","draft","emit"].map(k=>[k,avg(a.map(x=>x.stat[mode][k]/x.stat[mode].n))]))]))})}
return {rows,summary,forwards,rollouts,maxNorm};
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("fs");console.log(JSON.stringify(run([JSON.parse(fs.readFileSync(process.argv[2],"utf8")),JSON.parse(fs.readFileSync(process.argv[3],"utf8")).weights],fs.readFileSync(process.argv[4],"utf8"),JSON.parse(fs.readFileSync(process.argv[5],"utf8"))),null,2))}
