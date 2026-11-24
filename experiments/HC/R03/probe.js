// node experiments/HC/R03/probe.js
// Synthetic CPU categorical tests; no GPU/wall-clock measurement.
function probe(){
 const V=64,N=512,C=40,seeds=[11,29,73],sigmas=[.1,.5,1.5];let state;
 const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+.5)/4294967296};
 const normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
 const sum=a=>a.reduce((s,x)=>s+x,0),norm=a=>{let z=sum(a);return a.map(x=>x/z)},soft=a=>norm(a.map(x=>Math.exp(x-Math.max(...a)))),sq=a=>sum(a.map(x=>x*x));
 const draw=(p,r=u())=>{for(let i=0;i<p.length;i++){r-=p[i];if(r<0)return i}return p.length-1};
 let rows=[];
 for(const sig of sigmas)for(const seed of seeds){state=seed;let out=Array(10).fill(0);
 for(let e=0;e<4;e++)for(let c=0;c<C;c++){
 const z=Array.from({length:V},normal),p=soft(z),q=soft(z.map((x,i)=>x+sig*normal()+(e-1.5)*.1*Math.sin(i)));
 const delta=q.map((x,i)=>x-p[i]),tv=sum(delta.map(Math.abs))/2,r=tv>0?norm(delta.map(x=>Math.max(0,-x))):p;
 const theoretical=[1-sq(p),2-2*sum(p.map((x,i)=>x*q[i]))-sq(delta),2*tv-sq(delta)];
 let vals=Array.from({length:4},()=>Array(V).fill(0)),ss=Array(4).fill(0),reject=0;
 for(let j=0;j<N;j++){const y=draw(p),x=draw(q),x2=draw(q),y2=u()<Math.min(1,p[x2]/q[x2])?x2:draw(r),U=u(),x3=draw(q,U),y3=draw(p,U);
 let g=[q.map((x,i)=>x-Number(i===y)),Array(V).fill(0),Array(V).fill(0),Array(V).fill(0)];
 g[1][x]++;g[1][y]--;g[2][x2]++;g[2][y2]--;g[3][x3]++;g[3][y3]--;reject+=Number(x2!==y2);
 for(let k=0;k<4;k++){ss[k]+=sq(g[k]);g[k].forEach((v,i)=>vals[k][i]+=v/N)}
 }
 let measured=ss.map((s,k)=>(s/N-sq(vals[k]))*N/(N-1));
 let mse=sq(vals[2].map((x,i)=>x-delta[i]));
 let v=[tv,...theoretical,...measured,reject/N,mse];v.forEach((a,i)=>out[i]+=a/(4*C))
 }
 rows.push({sig,seed,values:out})
 }
 return {V,N,C,seeds,sigmas,columns:["TV","theoryCE","theoryIndependent","theorySD","empCE","empIndependent","empSD","empCommonU","reject","SDmeanSquaredError"],rows};
}
function train(){
 const V=32,B=16,T=500,seeds=[11,29,73],sigmas=[.1,.5,1.5],methods=["CE","independent","SD","commonU","exact"];let state;
 const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+.5)/4294967296};
 const normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
 const sum=a=>a.reduce((s,x)=>s+x,0),norm=a=>{let z=sum(a);return a.map(x=>x/z)},soft=a=>{let m=Math.max(...a);return norm(a.map(x=>Math.exp(x-m)))};
 const draw=(p,r=u())=>{for(let i=0;i<p.length;i++){r-=p[i];if(r<0)return i}return p.length-1};
 let rows=[];
 for(const sig of sigmas)for(const seed of seeds){state=seed;let z0=Array.from({length:V},normal),ps=Array.from({length:4},()=>soft(z0.map(x=>x+sig*normal()))),pbar=z0.map((_,i)=>sum(ps.map(p=>p[i]))/4);
 const loss=q=>sum(pbar.map((p,i)=>p*Math.log(p/q[i]))),accept=q=>sum(ps.map(p=>sum(p.map((v,i)=>Math.min(v,q[i])))))/4;
 for(const method of methods){state=seed+1000;let z=z0.slice(),nonzero=0,trace=[];
 for(let t=0;t<T;t++){let q=soft(z),g=Array(V).fill(0);
 for(const p of ps){let r0=q.map((v,i)=>Math.max(0,p[i]-v)),r=sum(r0)>0?norm(r0):p;
 if(method==="exact"){for(let i=0;i<V;i++)g[i]+=(q[i]-p[i])/4;continue}
 for(let j=0;j<B;j++){let x,y;
 if(method==="CE"){y=draw(p);for(let i=0;i<V;i++)g[i]+=q[i]/(4*B);g[y]-=1/(4*B);nonzero++;continue}
 if(method==="SD"){x=draw(q);y=u()<Math.min(1,p[x]/q[x])?x:draw(r)}
 else if(method==="independent"){x=draw(q);y=draw(p)}
 else{let a=u();x=draw(q,a);y=draw(p,a)}
 if(x!==y){g[x]+=1/(4*B);g[y]-=1/(4*B);nonzero++}
 }}
 z=z.map((v,i)=>v-.8*g[i]);if([0,49,99,249,499].includes(t))trace.push(loss(soft(z)));
 }
 rows.push({sig,seed,method,initial:loss(soft(z0)),final:loss(soft(z)),accept:accept(soft(z)),active:method==="exact"?null:nonzero/(T*4*B),trace})
 }}
 return {V,B,T,rows};
}
console.log(JSON.stringify({variance:probe(),training:train()},null,2));
