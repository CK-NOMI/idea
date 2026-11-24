// node soft_gate_probe.js
function probe(){
const sum=a=>a.reduce((s,x)=>s+x,0),zero=n=>Array(n).fill(0),mean=a=>sum(a)/a.length;
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function conv(a,b){let c=zero(a.length+b.length-1);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)c[i+j]+=a[i]*b[j];return c}
const entropy=p=>p<=0||p>=1?0:-p*Math.log(p)-(1-p)*Math.log(1-p);
let rows=[],audit={normalization:0,versionFrequency:0,fullRevealDominance:0,independentLimit:0};
for(let seed of [11,29,73])for(let N of [4,8,16,64])for(let het of [.1,.5,.9]){
let rr=rng(seed+N*1000),a=Array.from({length:N},()=>.5+het*(rr()-.5)),p0=a.map(x=>Math.min(.99,Math.max(.01,x-.12))),p1=a.map(x=>Math.min(.99,Math.max(.01,x+.12))),pre=[[1]],suf=Array(N+1);suf[N]=[1];for(let i=0;i<N;i++)pre.push(conv(pre.at(-1),[1-a[i],a[i]]));for(let i=N-1;i>=0;i--)suf[i]=conv([1-a[i],a[i]],suf[i+1]);let T=Math.ceil(N/2),pi=sum(pre[N].slice(T));
for(let lam of [0,.25,.5,.75,1]){
const prob=(s,v)=>{let h=lam*Number(s>=T)+(1-lam)*pi;return v?h:1-h};
let acc=0,fresh=0,reveal=0,version=0,agree=0,info=entropy(pi),marginal=0;
for(let s=0;s<=N;s++){let pr=pre[N][s],h=prob(s,1);version+=pr*h;agree+=pr*(s>=T?h:1-h);info-=pr*entropy(h)}
for(let i=0;i<N;i++)for(let s=0;s<=i;s++)for(let v of [0,1]){
let f=b=>sum(suf[i+1].map((w,j)=>w*prob(s+b+j,v))),f0=f(0),f1=f(1),z=(1-a[i])*f0+a[i]*f1,pr=pre[i][s]*z;if(!z)continue;let q=a[i]*f1/z,p=v?p1[i]:p0[i];acc+=pr*(1-Math.abs(p-q))/N;fresh+=pr*(1-Math.abs(p-a[i]))/N;reveal+=pre[i][s]*((1-a[i])*f0*(1-p)+a[i]*f1*p)/N;if(i===0)marginal+=pr;
}
audit.normalization=Math.max(audit.normalization,Math.abs(marginal-1));audit.versionFrequency=Math.max(audit.versionFrequency,Math.abs(version-pi));audit.fullRevealDominance=Math.max(audit.fullRevealDominance,reveal-acc);if(lam===0)audit.independentLimit=Math.max(audit.independentLimit,Math.abs(acc-fresh));rows.push({seed,N,het,lambda:lam,version,agreement:agree,information:info,acceptance:acc,fresh,revealed:reveal})
}}
let summary=[];for(let N of [4,8,16,64])for(let lam of [0,.25,.5,.75,1]){let a=rows.filter(x=>x.N===N&&x.lambda===lam);summary.push({N,lambda:lam,...Object.fromEntries(["agreement","information","acceptance","fresh","revealed"].map(k=>[k,mean(a.map(x=>x[k]))]))})}return {rows,summary,audit}
}
console.log(JSON.stringify(probe(),null,2));
