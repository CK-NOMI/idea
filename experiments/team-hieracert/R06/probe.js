// R06: exact occupancy audit for independent rollouts; node probe.js
function run(){
 const V=4,L=6,B=24,seeds=[11,29,73],sigmas=[.15,.6,1.5],ns=[4,16,64],trials=40,mc=8;
 let st=0;const U=()=>{st=(Math.imul(1664525,st)+1013904223)>>>0;return (st+.5)/4294967296};
 const N=()=>Math.sqrt(-2*Math.log(U()))*Math.cos(2*Math.PI*U());
 const soft=a=>{const m=Math.max(...a),z=a.map(x=>Math.exp(x-m)),s=z.reduce((a,b)=>a+b,0);return z.map(x=>x/s)};
 const sum=a=>a.reduce((x,y)=>x+y,0);
 const levels=[['']];for(let l=1;l<L;l++)levels.push(levels[l-1].flatMap(s=>Array.from({length:V},(_,v)=>s+v)));
 const states=levels.flat(), draw=(p,u)=>{for(let i=0;i<p.length;i++){u-=p[i];if(u<0)return i}return p.length-1};
 const occ=(p,n)=>-Math.expm1(n*Math.log1p(-p));
 const rank=(m)=>states.filter(s=>s!=='').sort((a,b)=>m[b]-m[a]||a.length-b.length||a.localeCompare(b)).slice(0,B-1);
 let rows=[];
 for(const sigma of sigmas)for(const seed of seeds){
  st=seed;const acc=ns.map(n=>({n,ar:0,qHit:0,oracleHit:0,qExtra:0,oracleExtra:0,qAccepted:0,oracleAccepted:0,mcAR:0,mcSD:0,mcHit:0,oracleRankingDisagreement:0,maxSetError:0,pairedGroups:0}));
  for(let t=0;t<trials;t++){
   let P={'':1},Q={'':1},cond={};
   for(const h of states){let base=Array.from({length:V},()=>N());let p=soft(base),q=soft(base.map(x=>x+sigma*N()));cond[h]=p;for(let v=0;v<V;v++){P[h+v]=P[h]*p[v];Q[h+v]=Q[h]*q[v]}}
   let tree=new Set(['',...rank(Q)]),oracle=new Set(['',...rank(P)]);
   for(const s of tree)if(s&&!tree.has(s.slice(0,-1)))throw Error('not prefix closed');
   for(const a of acc){let n=a.n,op=Object.fromEntries(states.map(s=>[s,occ(P[s],n)])),opt=new Set(['',...rank(op)]);
    if([...opt].some(s=>!oracle.has(s)))a.oracleRankingDisagreement++;
    const ar=sum(states.map(s=>op[s])),qh=sum([...tree].map(s=>op[s])),oh=sum([...oracle].map(s=>op[s]));
    a.ar+=ar/trials;a.qHit+=qh/trials;a.oracleHit+=oh/trials;a.qExtra+=(B-qh)/trials;a.oracleExtra+=(B-oh)/trials;
    // Candidate length before first tree miss, counting the root distribution as one available prediction.
    a.qAccepted+=n*sum([...tree].map(s=>P[s]))/trials;a.oracleAccepted+=n*sum([...oracle].map(s=>P[s]))/trials;
    for(let k=0;k<mc;k++){let seen=new Set();for(let i=0;i<n;i++){let h='';for(let j=0;j<L;j++){seen.add(h);h+=draw(cond[h],U())}}
     let joint=new Set([...seen,...tree]);let hits=[...tree].filter(s=>seen.has(s)).length;
     a.mcAR+=seen.size/(trials*mc);a.mcSD+=joint.size/(trials*mc);a.mcHit+=hits/(trials*mc);
     a.maxSetError=Math.max(a.maxSetError,Math.abs(joint.size-seen.size-(B-hits)));a.pairedGroups++;
    }
   }
  }
  for(const a of acc)rows.push({sigma,seed,...a});
 }
 return {environment:'V8 CPU; synthetic complete conditional trees; exact expectations + paired Monte Carlo counts; no GPU or wall time',V,L,B,seeds,sigmas,ns,trials,mc,rows};
}
console.log(JSON.stringify(run(),null,2));
