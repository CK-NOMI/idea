// R07: real-corpus count LMs and exact conditional query-coverage audit.
// node probe.js wallpaper.txt alice.txt
function run(wall,alice){
 const clean=t=>{const a=t.indexOf("*** START OF THIS PROJECT GUTENBERG"),b=t.indexOf("*** END OF THIS PROJECT GUTENBERG");if(a<0||b<0)throw Error("Gutenberg markers missing");return t.slice(t.indexOf("\n",a),b).toLowerCase().replace(/[^a-z]+/g," ").trim()};
 const V=27,cv=c=>c===" "?26:c.charCodeAt(0)-97,ids=t=>Array.from(t,cv),w=clean(wall),a=clean(alice),mid=Math.floor(a.length/2);
 const region=[ids(w),ids(a.slice(0,mid)),ids(a.slice(mid))],train=region.map(x=>x.slice(0,Math.floor(x.length*.8))),test=region.map(x=>x.slice(Math.floor(x.length*.8)));
 const zero=n=>Array(n).fill(0),counts=(seqs,n)=>{const c=zero(V**(n+1));for(const s of seqs)for(let i=n;i<s.length;i++){let j=0;for(let k=i-n;k<=i;k++)j=j*V+s[k];c[j]++}return c};
 const glob1=counts(train,1),glob2=counts(train,2),loc=train.map(s=>counts([s],1));
 const norm=x=>{let s=x.reduce((a,b)=>a+b,0);return x.map(v=>v/s)},rank=x=>Array.from({length:V},(_,i)=>i).sort((i,j)=>x[j]-x[i]||i-j);
 const sum=(a,ix)=>ix.reduce((s,i)=>s+a[i],0);
 const ptab=[],qtab=region.map(()=>[]),gqtab=[];
 for(let h=0;h<V*V;h++){
  const g=norm(glob1.slice((h%V)*V,(h%V+1)*V).map(x=>x+.5));
  ptab[h]=norm(glob2.slice(h*V,(h+1)*V).map((x,i)=>x+5*g[i]));
  gqtab[h]=g;
  for(let e=0;e<region.length;e++)qtab[e][h]=norm(loc[e].slice((h%V)*V,(h%V+1)*V).map((x,i)=>x+20*g[i]));
 }
 const stats=[],seeds=[11,29,73],ks=[1,2,4,8],Ns=4096;
 let state=0;const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296};
 const draw=p=>{let x=u();for(let j=0;j<p.length;j++){x-=p[j];if(x<0)return j}return p.length-1};
 let maxMarginalError=0,maxBoundViolation=0;
 for(let e=0;e<region.length;e++){
  // Exact average over observed held-out context frequencies; no model-generated test leakage.
  const hs=zero(V*V);for(let t=2;t<test[e].length;t++)hs[test[e][t-2]*V+test[e][t-1]]++;
  const total=hs.reduce((s,x)=>s+x,0);
  for(const qkind of ["regional","global"])for(const k of ks){
   let E={topP:0,topQ:0,independentDraft:0,sdDraftPlusQ:0,sdConditionalOracle:0,sdAcceptance:0,actualTextTopQ:0};
   const cached=[];
   for(let h=0;h<hs.length;h++)if(hs[h]){
    const p=ptab[h],q=(qkind==="regional"?qtab[e]:gqtab)[h],rq=rank(q),rp=rank(p);
    const alpha=p.reduce((s,v,i)=>s+Math.min(v,q[i]),0),delta=1-alpha;
    const r=p.map((v,i)=>Math.max(0,v-q[i])),zr=r.reduce((s,x)=>s+x,0),res=zr>1e-14?r.map(x=>x/zr):p;
    const freq=hs[h]/total,mat=[],sets=[],oracles=[];let condBest=0,sd=0,ind=0;
    for(let x=0;x<V;x++){
     const acc=Math.min(1,p[x]/q[x]),c=res.map((v,i)=>(1-acc)*v+(i===x?acc:0)),S=[x,...rq.filter(i=>i!==x)].slice(0,k),T=rank(c).slice(0,k);
     mat[x]=c;sets[x]=S;oracles[x]=T;
     sd+=q[x]*sum(c,S);condBest+=q[x]*sum(c,T);ind+=q[x]*sum(p,S);
    }
    for(let y=0;y<V;y++)maxMarginalError=Math.max(maxMarginalError,Math.abs(p[y]-q.reduce((s,v,x)=>s+v*mat[x][y],0)));
    maxBoundViolation=Math.max(maxBoundViolation,sd-condBest);
    E.topP+=freq*sum(p,rp.slice(0,k));E.topQ+=freq*sum(p,rq.slice(0,k));E.independentDraft+=freq*ind;E.sdDraftPlusQ+=freq*sd;E.sdConditionalOracle+=freq*condBest;E.sdAcceptance+=freq*alpha;
    cached[h]={p,q,mat,sets,oracles};
   }
   for(let t=2;t<test[e].length;t++){let h=test[e][t-2]*V+test[e][t-1];if(rank((qkind==="regional"?qtab[e]:gqtab)[h]).slice(0,k).includes(test[e][t]))E.actualTextTopQ+=1/total}
   const mc=[];
   for(const seed of seeds){state=seed;let hit=0,oracle=0,ind=0;for(let i=0;i<Ns;i++){let t=2+Math.floor(u()*total),h=test[e][t-2]*V+test[e][t-1],z=cached[h],x=draw(z.q),y=draw(z.mat[x]);hit+=Number(z.sets[x].includes(y));oracle+=Number(z.oracles[x].includes(y));ind+=Number(z.sets[x].includes(draw(z.p)))}mc.push({seed,n:Ns,sd:hit/Ns,conditionalOracle:oracle/Ns,ind:ind/Ns})}
   stats.push({region:e,qkind,k,heldoutContexts:total,...E,mc});
  }
 }
 // Exact endpoint collision for two regional rounds; same individual two-step endpoints.
 const endpoint=[{a:[0,0],b:[0,1]},{a:[0,.5],b:[0,2/3]}].map(({a,b})=>{let avg=(a[0]+a[1])/2,bb=(b[0]+b[1])/2;return{a,b,flatEndpoints:b.map((x,i)=>(1+a[i])*x),hierTwo:bb*(1+avg)}});
 return {env:"JS V8 actual corpus count fitting and probability enumeration/MC; not Transformer/GPU/real HFL",trainLengths:train.map(x=>x.length),testLengths:test.map(x=>x.length),V,seeds,Ns,stats,maxMarginalError,maxBoundViolation,endpoint};
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("node:fs");console.log(JSON.stringify(run(fs.readFileSync(process.argv[2],"utf8"),fs.readFileSync(process.argv[3],"utf8")),null,2))}
