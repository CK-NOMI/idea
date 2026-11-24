// Node.js: node probe.js wallpaper.txt alice.txt (fixed source texts in record)
const fs=require("node:fs");
function run(texts){
const V=27,alphabet=" abcdefghijklmnopqrstuvwxyz",taus=[.5,.8,.9,.95],alphas=[.001,.01,.05,.2],sum=a=>a.reduce((s,v)=>s+v,0),tv=(p,q)=>sum(p.map((v,i)=>Math.abs(v-q[i])))/2;
function clean(s){const a=s.indexOf("*** START OF"),b=s.indexOf("*** END OF",a+1);if(a<0||b<0)throw Error("missing marker");return s.slice(s.indexOf("\n",a),b).toLowerCase().replace(/[^a-z]+/g," ").trim().split("").map(c=>alphabet.indexOf(c))}
const [w,a]=texts.map(clean);const regions=[w,a.slice(0,Math.floor(a.length/2)),a.slice(Math.floor(a.length/2))];
const train=regions.map(x=>x.slice(0,Math.floor(.8*x.length))),test=regions.map(x=>x.slice(Math.floor(.8*x.length)));
const arr=(n,m)=>Array.from({length:n},()=>Array(m).fill(0));function count(x){const big=arr(V,V),tri=arr(V*V,V),freq=Array(V*V).fill(0);for(let j=2;j<x.length;j++){const h=x[j-2]*V+x[j-1],y=x[j];big[x[j-1]][y]++;tri[h][y]++;freq[h]++}return {big,tri,freq}}
const trainC=train.map(count),testC=test.map(count);const pooled=arr(V*V,V);const pooledB=arr(V,V);
for(const c of trainC){for(let i=0;i<V*V;i++)for(let j=0;j<V;j++)pooled[i][j]+=c.tri[i][j];for(let i=0;i<V;i++)for(let j=0;j<V;j++)pooledB[i][j]+=c.big[i][j]}
const probs=(counts,prior,k)=>{let n=sum(counts)+k;return counts.map((x,i)=>(x+k*prior[i])/n)};
const uniform=Array(V).fill(1/V);
const gB=pooledB.map(x=>probs(x,uniform,1)),g=pooled.map((x,i)=>probs(x,gB[i%V],5));
function nucleus(p,t){const order=Array.from({length:V},(_,i)=>i).sort((i,j)=>p[j]-p[i]||i-j);let m=0,k=0;while(k<V&&m<t){m+=p[order[k++]]}let S=order.slice(0,k);let r=Array(V).fill(0);for(const i of S)r[i]=p[i]/m;const prev=m-p[order[k-1]],gap=k<V?p[order[k-1]]-p[order[k]]:Infinity;return {S,r,m,prev,gap,key:S.slice().sort((a,b)=>a-b).join(",")}}
function validCert(n,d,t){return n.gap>2*d&&n.m-d>=t&&n.prev+d<t}
const rows=[];let maxViolation=0,certFailures=0;
for(let e=0;e<3;e++)for(const alpha of alphas)for(const t of taus){
let s={region:e,alpha,t,positions:0,contexts:0,rawTV:0,nucTV:0,maskChanges:0,stableCertOracle:0,stableCertBound:0,oldDraftRawAccept:0,newDraftTopAccept:0,wrongOutputTV:0,maskBits:0,certBits:0,oracleCertBits:0,meanMask:0,deltaMaskBits:0};
for(let h=0;h<V*V;h++){const f=testC[e].freq[h];if(!f)continue;const p=probs(trainC[e].tri[h],gB[h%V],5),q=probs(trainC[e].big[h%V],gB[h%V],20),r=p.map((x,i)=>(1-alpha)*x+alpha*g[h][i]);const P=nucleus(p,t),R=nucleus(r,t),Q=nucleus(q,t),d=tv(p,r),ds=tv(P.r,R.r);const co=validCert(P,d,t),cb=validCert(P,alpha,t),same=P.key===R.key;
if((co||cb)&&!same)certFailures++;maxViolation=Math.max(maxViolation,ds-Math.min(1,(1-t+d)/t),same?ds-d/Math.max(P.m,R.m):0);
const bits=Math.min(V,5*Math.min(P.S.length,V-P.S.length)+1)+5; // count/header illustrative incl complement
const diff=P.S.filter(i=>!R.S.includes(i)).length+R.S.filter(i=>!P.S.includes(i)).length; // oracle both masks
const aRaw=r.map((x,i)=>Math.min(1,x/q[i]));const accepted=Q.r.map((x,i)=>x*aRaw[i]);const reject=1-sum(accepted),res=r.map((x,i)=>Math.max(0,x-q[i])),mass=sum(res);const wrong=accepted.map((x,i)=>x+(mass>0?reject*res[i]/mass:0));if(Math.abs(sum(wrong)-1)>1e-9)throw Error("bad mass");
s.positions+=f;s.contexts++;
const metrics={rawTV:d,nucTV:ds,maskChanges:Number(!same),stableCertOracle:Number(co),stableCertBound:Number(cb),oldDraftRawAccept:1-tv(r,q),newDraftTopAccept:1-tv(R.r,Q.r),wrongOutputTV:tv(wrong,R.r),maskBits:bits,certBits:1+(cb?0:bits),oracleCertBits:1+(co?0:bits),meanMask:P.S.length,deltaMaskBits:5+5*diff};
for(const k in metrics)s[k]+=f*metrics[k];
}
for(const k in s)if(!["region","alpha","t","positions","contexts"].includes(k))s[k]/=s.positions;rows.push(s)
}
return {environment:"JS V8 deterministic corpus-count experiment, no pretrained model or physical HFL/GPU",sources:["Wallpaper","Alice split halves"],alphabet,regionLengths:regions.map(x=>x.length),trainLengths:train.map(x=>x.length),testLengths:test.map(x=>x.length),maxViolation,certFailures,rows};
}
console.log(JSON.stringify(run(process.argv.slice(2).map(x=>fs.readFileSync(x,"utf8"))),null,2));
