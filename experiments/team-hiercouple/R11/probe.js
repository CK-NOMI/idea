// R11: forward/cache code reused from team-hcr-sd/R11 via team-hiercouple/R10; new reward threshold audit.
function run(weights,docs){
 const D=16,V=27,BOS=26,zero=(n=D)=>Array(n).fill(0),mat=(n=D,m=D)=>Array.from({length:n},()=>zero(m)),sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),add=(a,b)=>a.map((x,i)=>x+b[i]),lin=(m,x)=>m.map(r=>dot(r,x)),trans=(m,x)=>m[0].map((_,j)=>sum(m.map((r,i)=>r[j]*x[i]))),rms=x=>x.map(v=>v/Math.sqrt(dot(x,x)/D+1e-5)),soft=x=>{let m=Math.max(...x),a=x.map(v=>Math.exp(v-m)),s=sum(a);return a.map(v=>v/s)},clone=m=>m.map(r=>r.slice()),diff=(a,b)=>Math.max(...a.flatMap((r,i)=>r.map((v,j)=>Math.abs(v-b[i][j])))),avg=(a,b)=>a.map((r,i)=>r.map((v,j)=>(v+b[i][j])/2)),id=c=>c.charCodeAt(0)-97;
 const hash=s=>{let h=2166136261;for(const c of s)h=Math.imul(h^c.charCodeAt(0),16777619);return h>>>0};
 const pool=Array.from({length:4},(_,i)=>[...new Set(docs)].filter(n=>/^[a-z]{6,14}$/.test(n)&&id(n[0])%4===i).sort((a,b)=>hash(a)-hash(b)).slice(0,400));
 function build(w){let at=0;const take=(r,c)=>Array.from({length:r},()=>{let a=w.slice(at,at+c);at+=c;return a});let e=take(V,D),pos=take(16,D),head=take(V,D),layers=Array.from({length:2},()=>({q:take(D,D),k:take(D,D),v:take(D,D),o:take(D,D),a:take(4*D,D),b:take(D,4*D)}));if(at!==w.length)throw Error("weights length");return {e,pos,head,layers}}
 function attention(layer,x,anc){let z=rms(x),q=lin(layer.q,z),k=lin(layer.k,z),v=lin(layer.v,z),ks=anc.map(n=>n.k).concat([k]),vs=anc.map(n=>n.v).concat([v]),a=[];for(let h=0;h<4;h++){let f=4*h,t=f+4,p=soft(ks.map(k=>dot(q.slice(f,t),k.slice(f,t))/2));for(let j=f;j<t;j++)a.push(sum(p.map((p,i)=>p*vs[i][j])))}return {x,k,v,a}}
 function tail(layer,n,O){let t=add(n.x,lin(O,n.a)),z=rms(t),pre=lin(layer.a,z),f=pre.map(v=>Math.max(0,v)),y=add(t,lin(layer.b,f));return {t,z,pre,y}}
 function cache(m){let c1=new Map,c2=new Map,need1=new Set,need2=new Set,counts={one:0,two:0};const key=x=>x.join(",");function get1(x){let k=key(x);if(c1.has(k))return c1.get(k);let anc=x.slice(0,-1).map((_,i)=>get1(x.slice(0,i+1))),h=rms(add(m.e[x.at(-1)],m.pos[x.length-1])),n=attention(m.layers[0],h,anc);n.y=tail(m.layers[0],n,m.layers[0].o).y;c1.set(k,n);counts.one++;return n}function get2(x){let k=key(x);if(c2.has(k))return c2.get(k);let anc=x.slice(0,-1).map((_,i)=>get2(x.slice(0,i+1))),n=attention(m.layers[1],get1(x).y,anc);c2.set(k,n);counts.two++;return n}return {counts,c1,c2,reset(){need1.clear();need2.clear()},demand(){return [need1.size,need2.size]},sh(x){for(let j=1;j<=x.length;j++)need1.add(key(x.slice(0,j)));return get1(x)},deep(x){for(let j=1;j<=x.length;j++){let k=key(x.slice(0,j));need1.add(k);need2.add(k)}return get2(x)}}}
 function out(m,n,O){return soft(lin(m.head,tail(m.layers[1],n,O).y))}
 function grad(m,n,O,tok,A){let f=tail(m.layers[1],n,O),p=soft(lin(m.head,f.y)),gz=trans(m.head,p.map((v,j)=>A*(v-Number(j===tok)))),u=trans(m.layers[1].b,gz).map((v,j)=>f.pre[j]>0?v:0),b=trans(m.layers[1].a,u),r=Math.sqrt(dot(f.t,f.t)/D+1e-5),a=dot(b,f.t)/(D*r*r*r),g=gz.map((v,j)=>v+b[j]/r-f.t[j]*a);return {g:g.map(v=>n.a.map(x=>v*x)),loss:-A*Math.log(p[tok]),p}}


 const D2=V*D,zero2=()=>Array(D2).fill(0),ax=(a,b,s=1)=>{for(let i=0;i<a.length;i++)a[i]+=s*b[i];return a},norm2=a=>sum(a.map(x=>x*x)),dist=(a,b)=>Math.sqrt(norm2(a.map((x,i)=>x-b[i]))),mean=a=>sum(a)/a.length;
 let state=0;const U=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+.5)/4294967296},draw=p=>{let r=U();for(let i=0;i<p.length;i++){r-=p[i];if(r<0)return i}return p.length-1};
 const results=[];let maxError=0,maxCertError=0,allAttempts=0,allAccept=0;
 for(let [name,raw] of weights){const w=Array.isArray(raw)?raw:Object.keys(raw).sort((a,b)=>+a-+b).map(k=>raw[k]),m=build(w),c=cache(m),O=m.layers[1].o;
 const feature=x=>({h:tail(m.layers[1],c.deep(x),O).y,s:c.sh(x).y});
 const old=x=>soft(lin(m.head,feature(x).h)),q=x=>soft(lin(m.head,feature(x).s));
 const examples=pool.flatMap(ns=>ns.slice(0,40).flatMap(n=>{let z=[BOS,...Array.from(n,id)];return Array.from({length:5},(_,j)=>({h:feature(z.slice(0,j+2)).h,y:z[j+2]}))}));
 function rollout(prompt,sd){let ctx=prompt.slice(),ys=[],blocks=0,attempt=0,accept=0,positions=[];const commit=y=>{positions.push({h:feature(ctx).h,p:old(ctx),y});ctx.push(y);ys.push(y)};
 while(ys.length<4){blocks++;if(!sd){commit(draw(old(ctx)));continue}let k=Math.min(2,4-ys.length),x=ctx.slice(),ds=[],qs=[];for(let j=0;j<k;j++){let a=q(x),y=draw(a);ds.push(y);qs.push(a);x.push(y)}
 const ps=Array.from({length:k},(_,j)=>old(x.slice(0,ctx.length+j))),bonus=ys.length+k<4?old(x):null;let all=true;
 for(let j=0;j<k;j++){attempt++;const p=ps[j],a=qs[j],y=ds[j];if(U()<Math.min(1,p[y]/a[y])){accept++;commit(y)}else{let res=p.map((v,i)=>Math.max(0,v-a[i])),s=sum(res);commit(draw(res.map(v=>v/s)));all=false;break}}
 if(all&&bonus)commit(draw(bonus))}
 return {ys,positions,blocks,attempt,accept}}
 for(let seed of [11,29,73]){state=seed;let W=clone(m.head);for(let i=0;i<500;i++){let e=examples[Math.floor(U()*examples.length)],p=soft(lin(W,e.h));for(let a=0;a<V;a++)for(let b=0;b<D;b++)W[a][b]-=.04*(p[a]-Number(a===e.y))*e.h[b]}
 for(let method of ["AR","SD"]){state=seed+100;let groups=[];for(let n=0;n<24;n++){const nm=pool[n%4][80+Math.floor(n/4)],prompt=[BOS,id(nm[0]),id(nm[1])],target=Array.from(nm.slice(2,6),id),seqs=Array.from({length:8},(_,i)=>({...rollout(prompt,method==="SD"),client:Math.floor(i/2),idx:i}));allAttempts+=sum(seqs.map(x=>x.attempt));allAccept+=sum(seqs.map(x=>x.accept));groups.push({seqs,target})}
 for(let version of ["same","updated"])for(let reward of ["weighted","dense"]){let metrics={groups:0,trajectories:0,clipPositions:0,positions:0,earlySign:0,earlyOneBranch:0,earlyOneBranchStrictBeforeLast:0,ambiguousBranch:0,buckets:0,activeBuckets:0,oneWayVectors:0,localNormError:0,localNormDen:0,gradError:0,certError:0,meanDistinctRewards:0,nonzeroGroups:0,meanBlocks:0,accept:0,attempt:0};
 for(const {seqs,target} of groups){let data=seqs.map(x=>{let rr=reward==="weighted"?sum(x.ys.map((y,j)=>Number(y===target[j])*2**j))/15:mean(x.ys.map((y,j)=>1-Math.abs(y-target[j])/26));let bp=zero2(),bn=zero2(),clipped=0;
 for(const a of x.positions){let p=soft(lin(version==="same"?m.head:W,a.h)),rho=p[a.y]/a.p[a.y],v=zero2();for(let j=0;j<V;j++)for(let k=0;k<D;k++)v[j*D+k]=rho*(Number(j===a.y)-p[j])*a.h[k]/4; // maximize clipped surrogate
 if(rho<=1.2)ax(bp,v);if(rho>=.8)ax(bn,v);clipped+=Number(rho>1.2||rho<.8)}
 return {...x,r:rr,bp,bn,clipped,delta:bp.map((v,i)=>v-bn[i])}})
 let mu=mean(data.map(x=>x.r)),sd=Math.sqrt(mean(data.map(x=>(x.r-mu)**2)))+1e-6,exact=zero2(),local=zero2(),reconstructed=zero2();metrics.nonzeroGroups+=Number(data.some(x=>x.r!==data[0].r));metrics.meanDistinctRewards+=new Set(data.map(x=>x.r)).size;
 for(let x of data)ax(exact,x.r>=mu?x.bp:x.bn,(x.r-mu)/(8*sd));
 for(let region=0;region<2;region++){let ds=data.filter(x=>Math.floor(x.client/2)===region),B0=zero2(),B1=zero2(),bins=new Map(),lm=mean(ds.map(x=>x.r)),ls=Math.sqrt(mean(ds.map(x=>(x.r-lm)**2)))+1e-6;
 for(let x of ds){ax(local,x.r>=lm?x.bp:x.bn,(x.r-lm)/(8*ls));ax(B0,x.bn);ax(B1,x.bn,x.r);if(!bins.has(x.r))bins.set(x.r,zero2());ax(bins.get(x.r),x.delta)}
 ax(reconstructed,B1,1/(8*sd));ax(reconstructed,B0,-mu/(8*sd));for(let [r,v] of bins)if(r>mu)ax(reconstructed,v,(r-mu)/(8*sd));
 let active=[...bins.values()].filter(v=>norm2(v)>1e-26).length;metrics.buckets+=bins.size;metrics.activeBuckets+=active;metrics.oneWayVectors+=2+active}
 let order=data.slice().sort((a,b)=>(a.blocks+(a.client%2)*.3)-(b.blocks+(b.client%2)*.3)||a.idx-b.idx);
 // all ties complete together: block count + synthetic 0.3 delay, NOT wall time
 for(let x of order){let t=x.blocks+(x.client%2)*.3,done=order.filter(y=>y.blocks+(y.client%2)*.3<=t),S=sum(done.map(y=>y.r)),lo=S/8,hi=(S+8-done.length)/8,sign=x.r<=lo+1e-14||x.r>=hi-1e-14,noDiff=norm2(x.delta)<1e-26,one=sign||noDiff;
 let chosen=x.r>=hi-1e-14?x.bp:x.bn,actual=x.r>=mu?x.bp:x.bn;
 if(sign&&Math.abs(x.r-mu)>1e-12)metrics.certError=Math.max(metrics.certError,dist(chosen,actual));
 metrics.earlySign+=Number(sign);metrics.earlyOneBranch+=Number(one);metrics.earlyOneBranchStrictBeforeLast+=Number(one&&done.length<8);metrics.ambiguousBranch+=Number(!one)}
 metrics.gradError=Math.max(metrics.gradError,dist(exact,reconstructed));metrics.localNormError+=norm2(exact.map((v,i)=>v-local[i]));metrics.localNormDen+=norm2(exact);metrics.groups++;metrics.trajectories+=8;metrics.positions+=32;metrics.clipPositions+=sum(data.map(x=>x.clipped));metrics.meanBlocks+=mean(data.map(x=>x.blocks));metrics.accept+=sum(data.map(x=>x.accept));metrics.attempt+=sum(data.map(x=>x.attempt))}
 maxError=Math.max(maxError,metrics.gradError);maxCertError=Math.max(maxCertError,metrics.certError);results.push({name,seed,method,version,reward,...metrics})
 }
 }}
 }
 // Finite information bound: fixed r_i=i+1; exact hinge queries at mu=i+.5 distinguish 2^n patterns.
 const distinguish=[];for(let n of [4,8,12]){let signatures=new Set();for(let mask=0;mask<2**n;mask++){let s=[];for(let j=0;j<n;j++){let mu=j+.5;s.push(sum(Array.from({length:n},(_,i)=>Math.max(0,i+1-mu)*((mask>>i)&1))))}signatures.add(s.join(","))}distinguish.push({n,patterns:2**n,distinct:signatures.size})}
 return {env:"V8 FP64, actual two external checkpoints, real AR/SD trajectories, actual 500-step CE head update, exact analytic head gradient; timing order proxy only",results,maxError,maxCertError,allAttempts,allAccept,distinguish};
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("fs");let a=JSON.parse(fs.readFileSync(process.argv[2],"utf8")),b=JSON.parse(fs.readFileSync(process.argv[3],"utf8")),d=fs.readFileSync(process.argv[4],"utf8").split(/\r?\n/).map(x=>x.trim()).filter(Boolean);console.log(JSON.stringify(run([["weights",a],["checkpoint",b.weights]],d),null,2))}
