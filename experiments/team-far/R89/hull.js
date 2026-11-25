function run(){
const seeds=[17,29,43],Vs=[4,16,64],Es=[2,4,8],lams=[.25,1,2],N=6;
let rows=[],exactChecks=0,maxViolation=0,maxChildErr=0,childChecks=0,evals=0,segments=0,feasErr=0;
const rng=s=>()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296};
const soft=z=>{let m=Math.max(...z),a=z.map(x=>Math.exp(x-m)),s=a.reduce((u,v)=>u+v,0);return a.map(x=>x/s)};
const arr=(n,f)=>Array.from({length:n},(_,i)=>f(i)),sum=a=>a.reduce((u,v)=>u+v,0),uniq=a=>[...new Set(a)].sort((x,y)=>x-y);
function child(p,c,a,s,lam){const V=p[0].length,E=p.length,segs=[];for(let y=0;y<V;y++){let knots=uniq([0,1,...p.map(t=>t[y])]),v=-lam*s*c[y]+sum(a);for(let j=0;j<knots.length-1;j++){let lo=knots[j],hi=knots[j+1];v=-lam*s*c[y];for(let e=0;e<E;e++)if(p[e][y]>lo)v+=a[e];segs.push({y,j,w:hi-lo,v});}}
segs.sort((x,y)=>y.v-x.v||x.y-y.y||x.j-y.j);segments+=segs.length;let left=1,z=arr(V,()=>0),val=0;for(let t of segs){let d=Math.min(left,t.w);if(d<=0)break;z[t.y]+=d;val+=d*t.v;left-=d;}feasErr=Math.max(feasErr,Math.abs(sum(z)-1));return {val,z}}
function knots(p,x){return uniq([0,1,...p.map(t=>t[x])])}
function root(p,c,s,x,lam){evals++;let a=p.map(t=>Math.min(t[0][x],s)/p.length);let cc=child(p.map(t=>t[x+1]),c[x+1],a,s,lam);return {s,v:sum(a)-lam*s*c[0][x]+cc.val,z:cc.z}}
function exactSmall(p,c,lam){let V=p[0][0].length,ks=arr(V,x=>knots(p.map(t=>t[0]),x)),best=-Infinity;
for(let free=0;free<V;free++){let q=arr(V,()=>0);function rec(i,t){if(i===V){let f=1-t;if(f<-1e-10||f>1+1e-10)return;q[free]=Math.max(0,f);let v=sum(q.map((s,x)=>root(p,c,s,x,lam).v));best=Math.max(best,v);return;}if(i===free){rec(i+1,t);return;}for(let s of ks[i])if(t+s<=1+1e-10){q[i]=s;rec(i+1,t+s);}}rec(0,0);}return best}
function solve(p,c,lam){let V=p[0][0].length,ps=p.map(t=>t[0]),points=arr(V,x=>knots(ps,x).map(s=>root(p,c,s,x,lam))),hulls=points.map(pts=>{let h=[];for(let t of pts){while(h.length>=2){let a=h[h.length-2],b=h[h.length-1];if((b.v-a.v)/(b.s-a.s)<=(t.v-b.v)/(t.s-b.s)+1e-13)h.pop();else break;}h.push(t);}return h;});
let slopes=[];for(let x=0;x<V;x++)for(let j=0;j<hulls[x].length-1;j++){let a=hulls[x][j],b=hulls[x][j+1];slopes.push({x,j,w:b.s-a.s,m:(b.v-a.v)/(b.s-a.s)})}slopes.sort((a,b)=>b.m-a.m||a.x-b.x||a.j-b.j);
let q=arr(V,()=>0),left=1,upper=sum(hulls.map(h=>h[0].v)),frac=-1;
for(let v of slopes){if(left<=1e-14)break;let t=Math.min(left,v.w);q[v.x]+=t;upper+=t*v.m;if(t<v.w-1e-13)frac=v.x;left-=t;}
let vals=q.map((s,x)=>root(p,c,s,x,lam)),lower=sum(vals.map(t=>t.v)),gap=upper-lower;
let a=0,cost=0;for(let x=0;x<V;x++){for(let e=0;e<p.length;e++){let m=Math.min(p[e][0][x],q[x])/p.length;a+=m*(1+sum(vals[x].z.map((z,y)=>Math.min(z,p[e][x+1][y]))))}cost+=q[x]*(c[0][x]+sum(vals[x].z.map((z,y)=>z*c[x+1][y])))}
maxViolation=Math.max(maxViolation,Math.abs(lower-(a-lam*cost)),Math.max(0,-gap));feasErr=Math.max(feasErr,Math.abs(sum(q)-1));
let myop=child(ps,c[0],arr(p.length,()=>1/p.length),1,lam).z, mv=myop.map((s,x)=>root(p,c,s,x,lam));let myA=0,myC=0;
for(let x=0;x<V;x++){for(let e=0;e<p.length;e++)myA+=Math.min(p[e][0][x],myop[x])/p.length*(1+sum(mv[x].z.map((z,y)=>Math.min(z,p[e][x+1][y]))));myC+=myop[x]*(c[0][x]+sum(mv[x].z.map((z,y)=>z*c[x+1][y])))}
return {upper,lower,gap,frac,A:a,C:cost,myA,myC,myU:myA-lam*myC}}
for(let V of Vs)for(let E of Es)for(let seed of seeds){let rand=rng(seed+101*V+1009*E),rnd=()=>2*rand()-1;for(let t=0;t<N;t++){let base=arr(V+1,()=>arr(V,()=>rnd()*2)),drift=[.2,.8,1.6][t%3],p=arr(E,()=>base.map(z=>soft(z.map(v=>v+drift*rnd()))));let r=arr(E,()=>arr(V,()=>Math.floor(rand()*8))),res=arr(E,()=>Math.floor(rand()*8)),c=[arr(V,x=>sum(r.map((a,e)=>+(a[x]!==res[e])))/E),...arr(V,x=>arr(V,y=>sum(r.map((a,e)=>+(a[y]!==res[e]&&a[y]!==a[x])))/E))];
for(let lam of lams){let s=solve(p,c,lam),exact=null;if(V===4&&E<=4){exact=exactSmall(p,c,lam);exactChecks++;maxViolation=Math.max(maxViolation,Math.max(0,s.lower-exact,exact-s.upper));}rows.push({V,E,seed,t,drift,lambda:lam,...s,exact});}
}}
let summary=[];for(let V of Vs)for(let E of Es)for(let l of lams){let a=rows.filter(x=>x.V===V&&x.E===E&&x.lambda===l),mean=k=>sum(a.map(x=>x[k]))/a.length;summary.push({V,E,lambda:l,n:a.length,gap:mean("gap"),maxGap:Math.max(...a.map(x=>x.gap)),zero:a.filter(x=>x.gap<1e-9).length,lower:mean("lower"),myU:mean("myU"),A:mean("A"),C:mean("C"),myA:mean("myA"),myC:mean("myC"),myWin:a.filter(x=>x.lower+1e-10>=x.myU).length,exact:a.filter(x=>x.exact!==null).length});}
return {spec:{seeds,Vs,Es,lams,N,K:2,notes:"synthetic categorical p, 8 hypothetical expert routes, full-information offline oracle. Hull tolerance FP64; no actual SD/LLM/network/SGD/GPU/time. Child sorted decreasing segment slopes, root upper concave envelope then one fractional coordinate repair."},rows,summary,exactChecks,maxViolation,feasErr,evals,segments};
}
console.log(JSON.stringify(run(),null,2));
