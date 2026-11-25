/* R86 synthetic integer certificate audit; node audit.js */
function run(){
const dims=[16,256,4096], bits=[0,2,4,6,8], grids=[64,128,256], modes=["dense","sparse","region_shift"], seeds=[17,29,43], N=32;
let rows=[],checks=0,errors=0,refineErrors=0,nonmonotone=0;
const rng=(s)=>()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296};
const Q=(x,g)=>Math.floor((x+g/2)/g);
for(const mode of modes)for(const seed of seeds){
const r=rng(seed+1009*modes.indexOf(mode)); const acc={};
for(const d of dims)for(const g of grids)for(const b of bits)acc[[d,g,b]]={coords:0,cert:0,whole:0,old:0,oldCert:0,wire:0,midErr:0};
for(let t=0;t<N;t++){
let a=[],z=[];for(let j=0;j<4096;j++){
 let u=0,v=0;const s=(r()<.5?-1:1)*48;
 for(let k=0;k<8;k++){let x=mode==="sparse"?(r()<.9?0:Math.floor(r()*255)-127):mode==="region_shift"?Math.max(-127,Math.min(127,Math.floor((r()-.5)*64)+(k<4?s:-s))):Math.floor(r()*255)-127;if(k<4)u+=x;else v+=x;}
 a.push(u);z.push(v);
}
for(const g of grids){
let prev=null;
for(const b of bits){
 const w=2**(10-b);let cert=0,old=0,oldCert=0,midErr=0;
 for(let j=0;j<4096;j++){
  const lo=Math.floor((z[j]+512)/w)*w-512,hi=lo+w-1;
  const l=Q(a[j]+lo,g),h=Q(a[j]+hi,g),f=Q(a[j]+z[j],g),ok=l===h;
  if(prev&&prev[j]&&!ok)nonmonotone++;
  if(ok&&l!==f)errors++;
  const reconstructed=ok?l:Q(a[j]+lo+(z[j]-lo),g);
  if(reconstructed!==f)refineErrors++;
  checks++;cert+=Number(ok);old+=Number(f===0);oldCert+=Number(ok&&f===0);midErr+=Number(Q(a[j]+Math.floor((lo+hi)/2),g)!==f);
  if(b===0){if(!prev){}}
  if(j+1===16||j+1===256||j+1===4096){
   const d=j+1, o=acc[[d,g,b]];
   o.coords+=d;o.cert+=cert;o.whole+=Number(cert===d);o.old+=old;o.oldCert+=oldCert;o.midErr+=midErr;
   o.wire+=b*d+(10-b)*(d-cert)+d; 
  }
 }
 prev=a.map((x,j)=>{const lo=Math.floor((z[j]+512)/w)*w-512;return Q(x+lo,g)===Q(x+lo+w-1,g)});
}
}
}
for(const d of dims)for(const g of grids)for(const b of bits){const o=acc[[d,g,b]];rows.push({mode,seed,d,g,b,cell:o.cert/o.coords,whole:o.whole/N,old:o.old/o.coords,oldCert:o.oldCert/o.coords,midErr:o.midErr/o.coords,bitsPer:o.wire/o.coords});}
}
const summary=[];for(const d of dims)for(const g of grids)for(const b of bits){const x=rows.filter(v=>v.d===d&&v.g===g&&v.b===b);const mean=k=>x.reduce((s,v)=>s+v[k],0)/x.length;summary.push({d,g,b,cell:mean("cell"),whole:mean("whole"),old:mean("old"),midErr:mean("midErr"),bitsPer:mean("bitsPer")});}
return {spec:{dims,bits,grids,modes,seeds,N,notes:"integer synthetic deltas; 8 clients, 2 logical edges; first edge sum exact, second signed 10-bit sum; downlink mask cost 1 bit/coord; no GPU, SGD, actual network, timing or LLM. All missing fine bits for uncertain cells, old target code zero."},checks,errors,refineErrors,nonmonotone,summary,rows};
}
console.log(JSON.stringify(run(),null,2));
