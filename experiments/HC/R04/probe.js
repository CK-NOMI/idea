// node experiments/HC/R04/probe.js
function probe(){
 const seeds=[11,29,73],sizes=[1024,8192,32768],ks=[2,32,256],T=12;let state=0,vis=0;
 const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+.5)/4294967296};
 const normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
 const priority=i=>{let x=(i+1)>>>0;x=Math.imul(x^(x>>>16),0x45d9f3b);x=Math.imul(x^(x>>>16),0x45d9f3b);return (x^(x>>>16))>>>0};
 const A=t=>t?t.sa:0,B=t=>t?t.sb:0;
 const upd=t=>{t.sa=t.a+A(t.l)+A(t.r);t.sb=t.b+B(t.l)+B(t.r);return t};
 const lt=(a,b)=>a.ratio<b.ratio||(a.ratio===b.ratio&&a.i<b.i);
 function merge(a,b){if(!a||!b)return a||b;vis++;if(a.pr<b.pr){a.r=merge(a.r,b);return upd(a)}b.l=merge(a,b.l);return upd(b)}
 function insert(t,n){if(!t)return upd(n);vis++;if(lt(n,t)){t.l=insert(t.l,n);if(t.l.pr<t.pr){let s=t.l;t.l=s.r;s.r=upd(t);return upd(s)}}else{t.r=insert(t.r,n);if(t.r.pr<t.pr){let s=t.r;t.r=s.l;s.l=upd(t);return upd(s)}}return upd(t)}
 function erase(t,n){if(!t)throw Error("missing");vis++;if(t.i===n.i)return merge(t.l,t.r);if(lt(n,t))t.l=erase(t.l,n);else t.r=erase(t.r,n);return upd(t)}
 const node=(i,a,b)=>({i,a,b,ratio:a/b,pr:priority(i),l:null,r:null,sa:a,sb:b});
 const mass=(a,b,za,zb)=>a/za-b/zb;
 function positive(t,za,zb){if(!t)return 0;vis++;if(t.ratio<=za/zb)return positive(t.r,za,zb);return mass(A(t.r)+t.a,B(t.r)+t.b,za,zb)+positive(t.l,za,zb)}
 function sample(t,za,zb,target){while(t){vis++;if(t.ratio<=za/zb){t=t.r;continue}let right=mass(A(t.r),B(t.r),za,zb),own=mass(t.a,t.b,za,zb);if(target<right){t=t.r;continue}target-=right;if(target<own)return t.i;target-=own;t=t.l}return -1}
 const rows=[];
 for(const V of sizes)for(const k of ks)for(const seed of seeds){state=seed;let a=[],b=[],root=null,nodes=[];
 for(let i=0;i<V;i++){let z=normal();a.push(Math.exp(z));b.push(Math.exp(z+.2*normal()));let n=node(i,a[i],b[i]);nodes.push(n);root=insert(root,n)}
 let buildVisits=vis,updates=0,queries=0,maxMassError=0,mismatch=0,minMass=1,maxMass=0;vis=0;
 for(let t=0;t<T;t++){
 const ids=new Set();while(ids.size<k)ids.add(Math.floor(u()*V));let start=vis;
 for(const i of ids){root=erase(root,nodes[i]);if(u()<.5)a[i]*=Math.exp(.3*normal());else b[i]*=Math.exp(.3*normal());let n=node(i,a[i],b[i]);nodes[i]=n;root=insert(root,n)}
 updates+=vis-start;
 const za=a.reduce((s,x)=>s+x,0),zb=b.reduce((s,x)=>s+x,0);
 const d=a.map((x,i)=>Math.max(0,x/za-b[i]/zb)),ref=d.reduce((s,x)=>s+x,0);start=vis;const m=positive(root,root.sa,root.sb);queries+=vis-start;maxMassError=Math.max(maxMassError,Math.abs(m-ref));minMass=Math.min(minMass,ref);maxMass=Math.max(maxMass,ref);
 const order=Array.from({length:V},(_,i)=>i).sort((i,j)=>a[j]/b[j]-a[i]/b[i]||j-i);
 for(let j=0;j<8;j++){const r=u();let w=r*ref,expected=-1;for(const i of order){w-=d[i];if(w<0){expected=i;break}}start=vis;const got=sample(root,root.sa,root.sb,r*m);queries+=vis-start;if(got!==expected)mismatch++}
 }
 rows.push({V,k,seed,T,meanUpdateNodeVisits:updates/T,meanQueryNodeVisits:queries/(9*T),maxMassError,mismatch,minMass,maxMass,buildNodeVisits:buildVisits});vis=0;
 }
 return {environment:"JS V8 CPU; synthetic logits/weight updates; FP64 arithmetic, no GPU or wall time",seeds,sizes,ks,T,rows}
}
console.log(JSON.stringify(probe(),null,2));
