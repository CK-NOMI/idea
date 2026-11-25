function run(input){
const V=26,B=4,L=3,seeds=[17,29,43],rounds=120,lr=12,reg=.0008;
const words=[...new Set(input.split(/\s+/).filter(s=>/^[a-z]{5,14}$/.test(s)))], arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length,hash=(s,k)=>{let h=k;for(let c of s)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h},rng=s=>()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return (s+.5)/4294967296},arg=a=>a.reduce((b,x,i)=>x>a[b]?i:b,0),soft=a=>{let m=Math.max(...a),p=a.map(x=>Math.exp(x-m)),z=sum(p);return p.map(x=>x/z)},dim=(L+1)*V*V;
const idx=(l,x,y)=>(l*V+x)*V+y, chars=s=>[...s.slice(0,B+1)].map(x=>x.charCodeAt(0)-97);
let checks=0,viol=0,gradSteps=0,rows=[],certChecks=0;
function logits(w,anchor,prev){let z=arr(V,y=>w[idx(0,anchor,y)]);for(let l=1;l<=prev.length;l++)for(let y=0;y<V;y++)z[y]+=w[idx(l,prev[prev.length-l],y)];return z}
function bounds(w){let lo=arr(L+1,()=>arr(V,()=>0)),hi=arr(L+1,()=>arr(V,()=>0)),pair=arr(L+1,()=>arr(V,()=>arr(V,()=>0)));for(let l=1;l<=L;l++)for(let y=0;y<V;y++){let a=arr(V,x=>w[idx(l,x,y)]);lo[l][y]=Math.min(...a);hi[l][y]=Math.max(...a);for(let z=0;z<V;z++)pair[l][y][z]=Math.min(...a.map((v,x)=>v-w[idx(l,x,z)]));}return {lo,hi,pair}}
function decode(w,bnd,anchor,noise,mode){
let truth=[];for(let i=0;i<B;i++)truth.push(arg(logits(w,anchor,truth).map((v,y)=>v+noise[i][y])));
let memo=arr(B,()=>new Map()),cacheHits=0,fullAdds=0,certAdds=0,certTerms=0;
function fullRow(i){let key=out.slice(0,i).join(",");if(memo[i].has(key)){cacheHits++;return memo[i].get(key)}let y=arg(logits(w,anchor,out.slice(0,i)).map((v,y)=>v+noise[i][y]));memo[i].set(key,y);full++;fullAdds+=V*(i+1);return y}
let q=noise.map(g=>arg(g.map((x,y)=>x+w[idx(0,anchor,y)]))),known=arr(B,()=>false),out=q.slice(),depth=0,full=0,cert=0,comp=0,fail=arr(L+1,()=>0),attempt=arr(L+1,()=>0),early=0;
while(!known.every(Boolean)){
 depth++;let next=known.slice(),nv=out.slice();for(let i=0;i<B;i++)if(!known[i]){
 let unknown=[];for(let l=1;l<=i;l++)if(!known[i-l])unknown.push(l);
 if(!unknown.length){nv[i]=fullRow(i);next[i]=true;continue}
 if(mode==="topo")continue;
 if(mode==="jacobi"){let p=fullRow(i);nv[i]=p;continue}
 certAdds+=V;let a=arr(V,y=>w[idx(0,anchor,y)]+noise[i][y]);for(let l=1;l<=i;l++)if(known[i-l]){certAdds+=V;for(let y=0;y<V;y++)a[y]+=w[idx(l,out[i-l],y)];}
 let guess=arg(a);if(mode==="hybrid"){guess=fullRow(i);nv[i]=guess;}for(let l of unknown)attempt[l]++;
 let ok=true;for(let y=0;y<V;y++)if(y!==guess){comp++;certTerms+=1+unknown.length;let m=a[guess]-a[y];for(let l of unknown)m+=(mode==="pair"||mode==="hybrid")?bnd.pair[l][guess][y]:bnd.lo[l][guess]-bnd.hi[l][y];if(m<=1e-12){ok=false;break}}
 certChecks++;if(ok){nv[i]=guess;next[i]=true;cert++;early++;if(guess!==truth[i])viol++;}else for(let l of unknown)fail[l]++;
 }
 // Jacobi's known flag is only structural-prefix certification, but if entire iterate unchanged, triangular fixed point is unique.
 if((mode==="jacobi"||mode==="hybrid")&&nv.every((v,i)=>v===out[i])){next.fill(true)}
 out=nv;known=next;if(depth>B)throw Error("nontermination");
}
checks++;if(out.some((v,i)=>v!==truth[i]))viol++;
return {depth,full,cert,comp,cacheHits,fullAdds,certAdds,certTerms,early,fail,attempt,qHit:sum(q.map((v,i)=>+(v===truth[i])))/B};
}
function nll(w,data){let n=0;for(let s of data){for(let i=0;i<B;i++)n-=Math.log(soft(logits(w,s[0],s.slice(1,i+1)))[s[i+1]])}return n/(B*data.length)}
function grad(w,data){let g=arr(dim,()=>0);for(let s of data)for(let i=0;i<B;i++){let p=soft(logits(w,s[0],s.slice(1,i+1)));for(let y=0;y<V;y++){let z=(p[y]-+(y===s[i+1]))/(data.length*B);g[idx(0,s[0],y)]+=z;for(let l=1;l<=i;l++)g[idx(l,s[i+1-l],y)]+=z;}}return g}
for(let seed of seeds)for(let split of ["hash","prefix","last"]){
let groups=arr(4,c=>words.filter(s=>(split==="hash"?hash(s,seed)%4:split==="prefix"?(s.charCodeAt(0)-97)%4:(s.charCodeAt(s.length-1)-97)%4)===c).sort((a,b)=>hash(a,seed+13)-hash(b,seed+13)).map(chars));
let train=groups.map(x=>x.slice(0,128)),cal=groups.map(x=>x.slice(128,144)),test=groups.map(x=>x.slice(144,208));if(test.some(x=>x.length<64))throw Error("split too small");
for(let arm of ["CE","uniform","verify"]){
let ws=arr(2,()=>arr(dim,()=>0)),weights=arr(2,()=>[0,1,1,1]);
for(let t=0;t<rounds;t++){
 let local=arr(4,c=>grad(ws[c>>1],train[c].slice((t%4)*32,(t%4+1)*32)));gradSteps+=4;
 for(let e=0;e<2;e++)for(let j=0;j<dim;j++){let g=(local[2*e][j]+local[2*e+1][j])/2,gl=mean(local.map(a=>a[j])),lag=Math.floor(j/(V*V));let z=lag===0?gl:.7*g+.3*gl;ws[e][j]-=lr*(z+(arm==="CE"||lag===0?0:reg*weights[e][lag]*ws[e][j]));}
 if(arm==="verify"&&t%8===7){for(let e=0;e<2;e++){let b=bounds(ws[e]),f=arr(L+1,()=>0);for(let c of [e*2,e*2+1])for(let k=0;k<cal[c].length;k++){let r=rng(seed+t*1009+c*71+k*13),noise=arr(B,()=>arr(V,()=>-Math.log(-Math.log(r()))));let d=decode(ws[e],b,cal[c][k][0],noise,"pair");for(let l=1;l<=L;l++)f[l]+=d.fail[l];}let m=mean(f.slice(1).map(x=>x+1));weights[e]=f.map((v,l)=>l===0?0:(v+1)/m);}}
}
let acc={};for(let mode of ["topo","jacobi","box","pair","hybrid"])acc[mode]={depth:0,full:0,cert:0,comp:0,cacheHits:0,fullAdds:0,certAdds:0,certTerms:0,qHit:0};
let n=0;for(let e=0;e<2;e++){let b=bounds(ws[e]);for(let c of [e*2,e*2+1])for(let k=0;k<test[c].length;k++){let r=rng(seed+100003+c*71+k*13),noise=arr(B,()=>arr(V,()=>-Math.log(-Math.log(r()))));for(let mode in acc){let d=decode(ws[e],b,test[c][k][0],noise,mode);for(let key in acc[mode])acc[mode][key]+=d[key];}n++;}}
for(let mode in acc)for(let k in acc[mode])acc[mode][k]/=n;
rows.push({seed,split,arm,n,nll:mean(test.map((d,c)=>nll(ws[c>>1],d))),weights,acc});
}
}
let summary=[];for(let arm of ["CE","uniform","verify"]){let a=rows.filter(r=>r.arm===arm);let v={arm,nll:mean(a.map(r=>r.nll)),modes:{}};for(let mode of ["topo","jacobi","box","pair","hybrid"])v.modes[mode]=Object.fromEntries(["depth","full","cert","comp","cacheHits","fullAdds","certAdds","certTerms","qHit"].map(k=>[k,mean(a.map(r=>r.acc[mode][k]))]));summary.push(v)}
return {spec:{V,B,L,seeds,rounds,lr,reg,notes:"real public names converted to 5-char blocks; trainable log-linear additive lag model, shared anchor and partly global regional lag parameters, four logical clients two regions; new formal model, not a Transformer or original model preserving. Gumbel vectors fixed per position. Actual SGD and simulation of exact decoding, no GPU, physical network or wall time. Verify feedback counts failed certificates involving unresolved lags, heuristic not gradient of expected runtime."},checks,certChecks,viol,gradSteps,rows,summary};
}
console.log(JSON.stringify(run(require('fs').readFileSync(process.argv[2],'utf8')),null,2));
