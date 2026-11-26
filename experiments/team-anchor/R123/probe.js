function run(){
const K=4,paths=Array.from({length:16},(_,i)=>i.toString(2).padStart(K,"0")),nodes=Array.from({length:15},(_,i)=>(i+1).toString(2).slice(1)),seeds=[11,29,73],Ns=[8,32,128],drifts=[.2,.8,1.5];const sig=x=>1/(1+Math.exp(-x));
function trial(seed){let state=seed>>>0;return()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296}}
function normal(r){return Math.sqrt(-2*Math.log(r()))*Math.cos(2*Math.PI*r())}
function gen(p,r){let h="";while(h.length<K)h+=r()<p[h]?"1":"0";return h}
function probs(p){return paths.map(y=>{let v=1;for(let i=0;i<K;i++)v*=y[i]==="1"?p[y.slice(0,i)]:1-p[y.slice(0,i)];return v})}
function counts(bank){let c={};for(const h of nodes)c[h]=[0,0];for(const y of bank)for(let i=0;i<K;i++)c[y.slice(0,i)][+y[i]]++;return c}
function mix(q,bank,alpha){const c=counts(bank),out={};for(const h of nodes){let n=c[h][0]+c[h][1];out[h]=n?(1-alpha)*q[h]+alpha*c[h][1]/n:q[h]}return out}
function eal(p,q){let a=0;function rec(h,w){if(h.length===K)return;for(const b of [0,1]){let m=Math.min(b?p[h]:1-p[h],b?q[h]:1-q[h]);a+=w*m;rec(h+b,w*m)}}rec("",1);return a}
function kernel(p,q){const P=probs(p),Q=probs(q),out=Array(16).fill(0);for(let j=0;j<16;j++){let y=paths[j],w=Q[j],h="";for(let i=0;i<K;i++){const b=+y[i],pb=b?p[h]:1-p[h],qb=b?q[h]:1-q[h],ar=qb?Math.min(1,pb/qb):0;let prefix=h+(1-b);for(let z=0;z<16;z++)if(paths[z].startsWith(prefix)){let suf=1;for(let k=i+1;k<K;k++)suf*=paths[z][k]==="1"?p[paths[z].slice(0,k)]:1-p[paths[z].slice(0,k)];out[z]+=w*(1-ar)*suf}w*=ar;h+=b}out[j]+=w}return Math.max(...out.map((v,i)=>Math.abs(v-P[i])))}
let results=[],maxErr=0;
for(const drift of drifts)for(const N of Ns)for(const seed of seeds){const r=trial(seed+1000*N+Math.round(drift*100));const sums={native:0,local:0,regional:0,global:0,regionFlat:0,regionalOnly:0};let n=0;
for(let rep=0;rep<120;rep++){const base={},reg=[{},{}];for(const h of nodes){base[h]=1.4*normal(r);for(let j=0;j<2;j++)reg[j][h]=drift*normal(r)}const P=[],Q=[],old=[],banks=[];for(let i=0;i<8;i++){let p={},q={},o={};for(const h of nodes){const z=base[h]+reg[i>>2][h]+.35*normal(r);p[h]=sig(z);q[h]=sig(z+.65*normal(r));o[h]=sig(z+.4*normal(r))}P.push(p);Q.push(q);old.push(o);banks.push(Array.from({length:N},()=>gen(o,r)))}
for(let i=0;i<8;i++){const regional=banks.flatMap((b,j)=>j!==i&&j>>2===i>>2?b:[]),global=banks.flatMap((b,j)=>j!==i?b:[]);const al=.25;const candidates={native:Q[i],local:mix(Q[i],banks[i],al),regional:mix(Q[i],regional,al),global:mix(Q[i],global,al),regionFlat:mix(Q[i],regional,al),regionalOnly:mix(Q[i],regional,1)};for(const [k,q] of Object.entries(candidates))sums[k]+=eal(P[i],q);if(rep===0&&i===0)maxErr=Math.max(maxErr,kernel(P[i],candidates.regional));n++}}
results.push({drift,N,seed,n,...Object.fromEntries(Object.entries(sums).map(([k,v])=>[k,v/n]))})}
return {K,maxErr,results}}

console.log(JSON.stringify(run(),null,2));
