function run(){
const Vs=[16,64,128],Es=[2,4,8],seeds=[17,29,43],lams=[.25,1,2],N=3;
const arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),sum=a=>a.reduce((u,v)=>u+v,0),uniq=a=>[...new Set(a)].sort((x,y)=>x-y);
const rng=s=>()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296};
const soft=z=>{let m=Math.max(...z),a=z.map(x=>Math.exp(x-m)),s=sum(a);return a.map(x=>x/s)};
let rows=[],nChild=0,maxViol=0;
function child(p,c,a,s,l){nChild++;let V=c.length,E=a.length,segs=[];for(let y=0;y<V;y++){let k=uniq([0,1,...p.map(t=>t[y])]);for(let j=0;j<k.length-1;j++){let v=-l*s*c[y];for(let e=0;e<E;e++)if(p[e][y]>k[j])v+=a[e];segs.push({y,j,w:k[j+1]-k[j],v});}}
segs.sort((a,b)=>b.v-a.v||a.y-b.y||a.j-b.j);let left=1,z=arr(V,()=>0),val=0;for(let t of segs){if(left<=1e-14)break;let d=Math.min(left,t.w);z[t.y]+=d;val+=d*t.v;left-=d;}return {val,z};}
function root(p,c,s,x,l){if(s<=0)return {s,v:0,A:0,C:0};let E=p.length,a=p.map(t=>Math.min(t[0][x],s)/E),cp=p.map(t=>t[x+1]),cc=child(cp,c[x+1],a,s,l),A=sum(a.map((w,e)=>w*(1+sum(cc.z.map((z,y)=>Math.min(z,cp[e][y])))))),C=s*(c[0][x]+sum(cc.z.map((z,y)=>z*c[x+1][y])));return {s,v:A-l*C,A,C}}
function hull(pts){let h=[];for(let t of pts){while(h.length>=2){let a=h[h.length-2],b=h[h.length-1];if((b.v-a.v)/(b.s-a.s)<=(t.v-b.v)/(t.s-b.s)+1e-13)h.pop();else break;}h.push(t);}return h;}
function alloc(hulls){let slopes=[],V=hulls.length;for(let x=0;x<V;x++)for(let j=0;j<hulls[x].length-1;j++){let a=hulls[x][j],b=hulls[x][j+1];slopes.push({x,j,w:b.s-a.s,m:(b.v-a.v)/(b.s-a.s)});}slopes.sort((a,b)=>b.m-a.m||a.x-b.x||a.j-b.j);let q=arr(V,()=>0),left=1,upper=0;for(let t of slopes){if(left<=1e-14)break;let d=Math.min(left,t.w);q[t.x]+=d;upper+=d*t.m;left-=d;}return {q,upper};}
function solve(p,c,l,eps=.001){let V=p[0][0].length,E=p.length,ks=arr(V,x=>uniq([0,1,...p.map(t=>t[0][x])])),exact=arr(V,()=>false),visited=new Set(),h=ks.map((k,x)=>hull(k.map(s=>({s,v:2*sum(p.map(t=>Math.min(t[0][x],s)/E))-l*s*(c[0][x]+Math.min(...c[x+1]))})))),initial=alloc(h).upper,best=-Infinity,bestA=0,bestC=0,upper=initial,rounds=0,early=null,cache=new Map();
const val=(x,s)=>{if(s<=0)return {s,v:0,A:0,C:0};let key=x+":"+s.toPrecision(16);if(!cache.has(key)){visited.add(x);cache.set(key,root(p,c,s,x,l));}return cache.get(key);};
let begin=nChild;while(rounds<=V+1){rounds++;let u=alloc(h);upper=u.upper;let active=u.q.map((s,x)=>s>1e-13?x:-1).filter(x=>x>=0),ev=active.map(x=>val(x,u.q[x])),lo=sum(ev.map(t=>t.v));if(lo>best){best=lo;bestA=sum(ev.map(t=>t.A));bestC=sum(ev.map(t=>t.C));}maxViol=Math.max(maxViol,Math.max(0,best-upper));
if(upper-best<=eps||active.every(x=>exact[x])){early={read:visited.size,refined:exact.filter(Boolean).length,calls:nChild-begin,rounds,lower:best,upper,gap:upper-best,A:bestA,C:bestC};break;}
for(let x of active)if(!exact[x]){h[x]=hull(ks[x].map(s=>val(x,s)));exact[x]=true;}
}
let allBegin=nChild,fullH=ks.map((k,x)=>hull(k.map(s=>root(p,c,s,x,l)))),full=alloc(fullH),fullV=full.q.map((s,x)=>root(p,c,s,x,l)),fullL=sum(fullV.map(t=>t.v)),fullA=sum(fullV.map(t=>t.A)),fullC=sum(fullV.map(t=>t.C));maxViol=Math.max(maxViol,Math.max(0,fullL-early.upper,early.lower-full.upper));
return {...early,initial,fullUpper:full.upper,fullLower:fullL,fullA,fullC,fullCalls:nChild-allBegin,readRatio:early.read/V,callRatio:early.calls/(nChild-allBegin)}
}
for(let V of Vs)for(let E of Es)for(let seed of seeds){let rand=rng(seed+101*V+1009*E),rnd=()=>2*rand()-1;for(let t=0;t<N;t++){let base=arr(V+1,()=>arr(V,()=>rnd()*2)),drift=[.2,.8,1.6][t],p=arr(E,()=>base.map(z=>soft(z.map(v=>v+drift*rnd()))));let r=arr(E,()=>arr(V,()=>Math.floor(rand()*8))),res=arr(E,()=>Math.floor(rand()*8)),c=[arr(V,x=>sum(r.map((a,e)=>+(a[x]!==res[e])))/E),...arr(V,x=>arr(V,y=>sum(r.map((a,e)=>+(a[y]!==res[e]&&a[y]!==a[x])))/E))];for(let l of lams)rows.push({V,E,seed,t,drift,lambda:l,...solve(p,c,l)});}}
let summary=[];for(let V of Vs)for(let E of Es)for(let l of lams){let x=rows.filter(x=>x.V===V&&x.E===E&&x.lambda===l),m=k=>sum(x.map(r=>r[k]))/x.length;summary.push({V,E,lambda:l,n:x.length,readRatio:m("readRatio"),callRatio:m("callRatio"),rounds:m("rounds"),gap:m("gap"),maxGap:Math.max(...x.map(r=>r.gap)),lower:m("lower"),fullLower:m("fullLower"),fullUpper:m("fullUpper"),epsPass:x.filter(r=>r.gap<=.001+1e-10).length});}
return {spec:{Vs,Es,seeds,lams,N,K:2,eps:.001,notes:"simulated deferred probability read: all data generated in memory; read count is logical row-access, not actual privacy or communication. All regions' mean child cost and minimum known upfront; cost of acquiring omitted. No model forward, training, SD or wall-clock."},rows,summary,maxViol,nChild};
}
console.log(JSON.stringify(run(),null,2));
