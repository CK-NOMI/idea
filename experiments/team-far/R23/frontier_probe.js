// node frontier_probe.js
function audit(){
function rng(s){return ()=>((s=(Math.imul(s,1664525)+1013904223)>>>0)+.5)/4294967296}
const sum=a=>a.reduce((s,x)=>s+x,0);
function frontier(p,F){let K=p.length,C=[0];for(let x of p)C.push(C.at(-1)+x);let dp=Array(K+1).fill(Infinity),out=[F[0]],prev=Array(K+1).fill(0);for(let j=1;j<=K;j++)dp[j]=0;out=Array(K+1).fill(Infinity);out[K]=0;
for(let m=1;m<=K;m++){let best=Infinity;for(let i=m;i<=K;i++)best=Math.min(best,dp[i]+F[i]*(C[K]-C[i-1]));out[K-m]=best;if(m===K)break;let nd=Array(K+1).fill(Infinity);for(let j=m+1;j<=K;j++)for(let i=m;i<j;i++)nd[j]=Math.min(nd[j],dp[i]+F[i]*(C[j-1]-C[i-1]));dp=nd}return out}
function brute(p,F){let K=p.length,o=Array(K+1).fill(Infinity);for(let mask=0;mask<(1<<K);mask++){let b=0,last=0,c=0;for(let i=1;i<=K;i++){if(mask&(1<<(i-1)))b++;else last=i;c+=p[i-1]*F[last]}o[b]=Math.min(o[b],c)}return o}
function merge(a,b,B){return Array.from({length:B+1},(_,k)=>{let m=Infinity;for(let i=0;i<=k;i++)if(i<a.length&&k-i<b.length)m=Math.min(m,a[i]+b[k-i]);return m})}
let rows=[],max=0,assoc=0,nonmonotonic=0,nonDiminishing=0;
for(let seed of [11,29,73]){let rr=rng(seed);for(let K of [2,4,6,8,12,32,64,128]){let n=K<=12?40:4,err=0,nd=0;for(let rep=0;rep<n;rep++){let a=Array.from({length:K},()=>rr()**2),p=a.map(x=>x/sum(a)),F=[0];for(let i=1;i<=K;i++)F.push(F.at(-1)+1+rr()*10+(i===1?25:0));let f=frontier(p,F);if(K<=12){let b=brute(p,F);err=Math.max(err,...f.map((x,i)=>Math.abs(x-b[i])))}for(let i=1;i<=K;i++){if(f[i]>f[i-1]+1e-10)nonmonotonic++;if(i>=2&&f[i-1]-f[i]>f[i-2]-f[i-1]+1e-10)nd++}if(K<=12){let c=merge(merge(f,f,K),f,K),d=merge(f,merge(f,f,K),K);assoc=Math.max(assoc,...c.map((v,i)=>Math.abs(v-d[i])))} }max=Math.max(max,err);nonDiminishing+=nd;rows.push({seed,K,n,bruteError:K<=12?err:null,increasingMarginalCount:nd})}}
return {rows,maxBruteError:max,maxAssociativityError:assoc,nonmonotonic,nonDiminishing,example:{F_A:[0,100,101],F_B:[0,1,3],L:2,budget:2,greedySaved:3,optimalSaved:101}}
}
console.log(JSON.stringify(audit(),null,2));
