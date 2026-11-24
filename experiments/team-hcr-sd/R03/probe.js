// Node.js, no dependencies. Synthetic parity expert-set probe; no GPU or timing.
function probe(){
 const masks=Array.from({length:64},(_,i)=>i).filter(x=>x.toString(2).replace(/0/g,"").length===3);
 const bit=(x,i)=>(x>>i)&1;
 const routes=Array.from({length:8},(_,x)=>(1<<bit(x,0))|(1<<(2+bit(x,1)))|(1<<(4+bit(x,2))));
 const pop=x=>x.toString(2).replace(/0/g,"").length;
 const parity=x=>(bit(x,0)+bit(x,1)+bit(x,2))%2;
 const choose=(counts,mode)=>{
   let best=masks[0], val=-1;
   for(const c of masks){let score=0;for(let x=0;x<8;x++){let n=pop(c&routes[x]);score+=counts[x]*(mode==="exact"?Number(n===3):mode==="pair"?n*(n-1)/2:n);}
   if(score>val){best=c;val=score;}}
   return best;
 };
 let state=0;const rand=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296};
 const dist=(rho,sgn)=>Array.from({length:8},(_,x)=>(1+rho*sgn*(parity(x)===0?1:-1))/8);
 const draw=p=>{let u=rand();for(let x=0;x<8;x++){u-=p[x];if(u<0)return x}return 7};
 const cost=(c,p)=>p.reduce((a,v,x)=>a+v*Number((c&routes[x])!==routes[x]),0);
 const rows=[];
 for(const rho of [0,.4,1])for(const n of [16,64,256])for(const seed of [11,29,73]){
 state=seed;let sum=[0,0,0,0,0,0],maxdiff=0;
 for(let trial=0;trial<200;trial++){
 let local=[],ps=[];let pooled=Array(8).fill(0);
 for(let e=0;e<4;e++){
 let p=dist(rho,e%2===0?1:-1);ps.push(p);let clients=[];
 for(let k=0;k<2;k++){let c=Array(8).fill(0);for(let t=0;t<n;t++)c[draw(p)]++;clients.push(c);}
 let c=clients[0].map((v,x)=>v+clients[1][x]);local.push(c);c.forEach((v,x)=>pooled[x]+=v);
 }
 let shared=choose(pooled,"exact");
 for(let e=0;e<4;e++){
 let p=ps[e],c=local[e];let exact=choose(c,"exact"),flat=choose(c,"exact");
 let v=[cost(choose(c,"marginal"),p),cost(choose(c,"pair"),p),cost(exact,p),cost(shared,p),Math.min(...masks.map(m=>cost(m,p))),cost(flat,p)];
 maxdiff=Math.max(maxdiff,Math.abs(v[2]-v[5]));v.forEach((a,i)=>sum[i]+=a);
 }}
 rows.push({rho,n,seed,mean:sum.map(x=>x/800),maxdiff});
 }
 return {methods:["marginal","pair","edge_exact","global_shared","oracle","flat_same_info"],rows};
}
console.log(JSON.stringify(probe(),null,2));
