// node probe.js weights.json checkpoint.json input.txt
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
for(let c=0;c<16;c++){let xs=test.map(a=>a[c]).map(x=>{let p=sm(mat(W,x.h)),p1=sm(add(mat(W,x.h),shift)),q=sm(mat(W,x.s)),bit=p.map((v,i)=>Number(p1[i]>v)),a=dot(q,bit),next=Array.from({length:V},(_,i)=>{let {h,s}=m.feature(x.pre.concat(i)),l=mat(W,h);return {p:[sm(l),sm(add(l,shift))],q:sm(mat(W,s))}});return {p:[p,p1],q,bit,a,next,y:x.y}}),suff=Array(N+1);suff[N]=[1];for(let i=N-1;i>=0;i--)suff[i]=conv([1-xs[i].a,xs[i].a],suff[i+1]);
let votePatterns=Array.from({length:1<<N},(_,mask)=>{let bits=xs.map((_,i)=>Number(!!(mask&(1<<i)))),w=bits.reduce((v,b,i)=>v*(b?xs[i].a:1-xs[i].a),1);return {bits,w,s:sum(bits)}});
audit.cohorts++;
for(let mode of ["threshold","count"]){let keys=mode==="count"?[0,1,2,3,4]:[0,1],out={wi,seed,c,mode,expected:{naiveTV:0,jointVoteTV:0,correctOverlap:0,freshOverlap:0,revealedOverlap:0},methods:{}};
let methods=["naive","posterior","revealed","fresh"];for(let meth of methods)out.methods[meth]={n:0,firstAccepted:0,secondAccepted:0,secondReached:0,retained:0,firstReused:0,secondReused:0,actualOutput:0};
for(let v of keys){let prob=sum(votePatterns.filter(z=>event(z.s,mode,v)).map(z=>z.w));if(prob<1e-16)continue;let ver=mode==="count"?Number(v>=2):v;
let joint=zero(1<<N),target=zero(1<<N),margTV=0;
for(let mask=0;mask<1<<N;mask++){target[mask]=xs.reduce((w,x,i)=>{let a=dot(x.p[ver],x.bit);return w*((mask&(1<<i))?a:1-a)},1)}
for(let z of votePatterns.filter(z=>event(z.s,mode,v))){let cond=z.w/prob,probs=[];for(let i=0;i<N;i++){let x=xs[i],q=posterior(x.q,x.bit,1-z.bits[i],z.bits[i]),p=x.p[ver],yp=zero(V);for(let j=0;j<V;j++)if(q[j])yp=add(yp,scale(kernel(j,p,x.q),q[j]));probs.push(dot(yp,x.bit));let pref=sum(z.bits.slice(0,i)),f=b=>sum(suff[i+1].map((w,l)=>event(pref+b+l,mode,v)?w:0)),qc=posterior(x.q,x.bit,f(0),f(1)),check=zero(V);for(let j=0;j<V;j++)if(qc[j])check=add(check,scale(kernel(j,p,qc),qc[j]));audit.localLaw=Math.max(audit.localLaw,...check.map((v,j)=>Math.abs(v-p[j])));audit.normalization=Math.max(audit.normalization,Math.abs(sum(qc)-1));out.expected.correctOverlap+=z.w*sum(p.map((v,j)=>Math.min(v,qc[j])))/N;out.expected.freshOverlap+=z.w*sum(p.map((v,j)=>Math.min(v,x.q[j])))/N;out.expected.revealedOverlap+=z.w*sum(p.map((v,j)=>Math.min(v,q[j])))/N}
for(let mask=0;mask<1<<N;mask++)joint[mask]+=cond*probs.reduce((w,p,i)=>w*((mask&(1<<i))?p:1-p),1)}
let tv=sum(joint.map((w,i)=>Math.abs(w-target[i])))/2;out.expected.jointVoteTV+=prob*tv;audit.maxNaiveJointVoteTV=Math.max(audit.maxNaiveJointVoteTV,tv);
for(let i=0;i<N;i++){let x=xs[i],other=xs.filter((_,j)=>j!==i).reduce((a,x)=>conv(a,[1-x.a,x.a]),[1]),f=b=>sum(other.map((w,l)=>event(l+b,mode,v)?w:0)),qc=posterior(x.q,x.bit,f(0),f(1)),p=x.p[ver],yp=zero(V);for(let j=0;j<V;j++)if(qc[j])yp=add(yp,scale(kernel(j,p,x.q),qc[j]));margTV+=sum(yp.map((w,j)=>Math.abs(w-p[j])))/2/N}
out.expected.naiveTV+=prob*margTV;
}
for(let rep=0;rep<256;rep++){let X=xs.map(x=>cat(x.q,rr())),X2=xs.map((x,i)=>cat(x.next[X[i]].q,rr())),bits=X.map((j,i)=>xs[i].bit[j]),count=sum(bits),ver=Number(count>=2),v=mode==="count"?count:ver;audit.draws++;audit.actualFirstDrafts+=N;audit.actualSecondDrafts+=N;
for(let meth of methods){let pref=0,stats=out.methods[meth];for(let i=0;i<N;i++){let x=xs[i],p=x.p[ver],f=b=>sum(suff[i+1].map((w,l)=>event(pref+b+l,mode,v)?w:0)),q=meth==="posterior"?posterior(x.q,x.bit,f(0),f(1)):meth==="revealed"?posterior(x.q,x.bit,1-bits[i],bits[i]):x.q,xx=meth==="fresh"?cat(x.q,rr()):X[i],xx2=meth==="fresh"?cat(x.next[xx].q,rr()):X2[i],first=verify(xx,p,q,rr); // replaced below
stats.n++;stats.actualOutput++;if(first.ok){stats.firstAccepted++;stats.secondReached++;let second=verify(xx2,x.next[xx].p[ver],x.next[xx].q,rr);stats.actualOutput++;if(second.ok){stats.secondAccepted++;if(meth!=="fresh")stats.secondReused++}if(meth!=="fresh")stats.firstReused++}pref+=bits[i]}}}
rows.push(out)}
}}}
let summary=[];for(let mode of ["threshold","count"]){let a=rows.filter(x=>x.mode===mode);summary.push({mode,expected:Object.fromEntries(Object.keys(a[0].expected).map(k=>[k,mean(a.map(x=>x.expected[k]))])),methods:Object.fromEntries(Object.keys(a[0].methods).map(m=>{let v=Object.fromEntries(Object.keys(a[0].methods[m]).map(k=>[k,sum(a.map(x=>x.methods[m][k]))]));return [m,{...v,firstRate:v.firstAccepted/v.n,acceptedDraftsPerBlock:(v.firstAccepted+v.secondAccepted)/v.n}]}))})}
return {rows,summary,audit}
}
console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
