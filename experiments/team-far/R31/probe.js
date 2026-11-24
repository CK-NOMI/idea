// node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
const V=27,d=16,H=4,k=4,N=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((v,i)=>v*b[i]*0+v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,s)=>a.map(v=>v*s),zero=n=>Array(n).fill(0),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),v=x.map(t=>Math.exp(t-m));return scale(v,1/sum(v))},mean=a=>sum(a)/a.length;
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(V,d),P=take(16,d),W=take(V,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");return {W,feature:h=>{let seq=h.map((t,i)=>rms(add(E[t],P[i]))),s;for(let b of L){let z=seq.map(rms),Q=z.map(x=>mat(b.q,x)),K=z.map(x=>mat(b.k,x)),v=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let j=0;j<H;j++){let t=j*k,w=sm(K.slice(0,i+1).map(x=>dot(Q[i].slice(t,t+k),x.slice(t,t+k))/2));for(let l=t;l<t+k;l++)a.push(sum(w.map((p,s)=>p*v[s][l])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(!s)s=seq.at(-1)}return {h:seq.at(-1),s}}}}
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(a,u){let s=0;for(let i=0;i<a.length;i++){s+=a[i];if(u<s)return i}return a.length-1}
const hash=(x,seed)=>{let h=seed>>>0;for(let c of x)h=Math.imul(h^c.charCodeAt(0),16777619)>>>0;return h};
function conv(a,b){let c=zero(a.length+b.length-1);for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)c[i+j]+=a[i]*b[j];return c}
const words=[...new Set(names.toLowerCase().split(/\s+/).filter(x=>/^[a-z]{7,14}$/.test(x)))],rows=[],audit={normalization:0,localLaw:0,naiveMarginalTV:0,maxNaiveJointVoteTV:0,cohorts:0,draws:0,actualFirstDrafts:0,actualSecondDrafts:0};
function verify(x,p,q,rr){let alpha=q[x]===0?1:Math.min(1,p[x]/q[x]);if(rr()<=alpha)return {y:x,ok:true};let r=p.map((v,i)=>Math.max(0,v-q[i]));return {y:cat(scale(r,1/sum(r)),rr()),ok:false}}
function kernel(x,p,q){let alpha=q[x]===0?1:Math.min(1,p[x]/q[x]),r=p.map((v,i)=>Math.max(0,v-q[i])),s=sum(r);return r.map((v,i)=>Number(i===x)*alpha+(s?(1-alpha)*v/s:0))}
function posterior(q,bit,f0,f1){let z=sum(q.map((v,i)=>v*(bit[i]?f1:f0)));return q.map((v,i)=>v*(bit[i]?f1:f0)/z)}
const event=(s,mode,v)=>mode==="count"?s===v:Number(s>=2)===v;
for(let wi=0;wi<raws.length;wi++){let m=model(raws[wi]),W=m.W,mom=p=>p.reduce((a,v,i)=>add(a,scale(W[i],v)),zero(d));
for(let seed of [11,29,73]){let groups=Array.from({length:N},(_,r)=>words.filter(x=>x.charCodeAt(0)%N===r).sort((a,b)=>hash(a,seed)-hash(b,seed)).slice(0,32));
let make=(name)=>{let j=3,pre=[26,...name.slice(0,j).split("").map(c=>c.charCodeAt(0)-97)],{h,s}=m.feature(pre);return {pre,h,s,y:name.charCodeAt(j)-97}};
let train=groups.map(g=>g.slice(0,16).map(make)),g=scale(train.flat().map(x=>sub(mom(sm(mat(W,x.h))),W[x.y])).reduce(add),1/(16*N)),shift=mat(W,scale(g,-1.5));
let test=groups.map(g=>g.slice(16).map(make)),rr=rng(seed+1000*wi);
for(let c=0;c<16;c++){let xs=test.map(a=>a[c]).map(x=>{let p=sm(mat(W,x.h)),p1=sm(add(mat(W,x.h),shift)),q=sm(mat(W,x.s)),bit=p.map((v,i)=>Number(p1[i]>v)),a=dot(q,bit),next=Array.from({length:V},(_,i)=>{let {h,s}=m.feature(x.pre.concat(i)),l=mat(W,h);return {p:[sm(l),sm(add(l,shift))],q:sm(mat(W,s))}});return {p:[p,p1],q,bit,a,next,y:x.y}}),suff=Array(N+1);suff[N]=[1];for(let i=N-1;i>=0;i--)suff[i]=conv([1-xs[i].a,xs[i].a],suff[i+1]);
let votePatterns=Array.from({length:1<<N},(_,mask)=>{let bits=xs.map((_,i)=>Number(!!(mask&(1<<i)))),w=bits.reduce((v,b,i)=>v*(b?xs[i].a:1-xs[i].a),1);return {bits,w,s:sum(bits)}});
audit.cohorts++;
rows.push({wi,seed,c,xs});
}}} return rows;}






function study(cohorts){
const sum=x=>x.reduce((s,v)=>s+v,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),zero=n=>Array(n).fill(0),norm=x=>{let z=sum(x);return z?x.map(v=>v/z):x},ov=(p,q)=>sum(p.map((v,i)=>Math.min(v,q[i]))),V=27;
function rng(seed){let x=seed>>>0;return ()=>((x=(Math.imul(x,1664525)+1013904223)>>>0)+.5)/4294967296}
function cat(p,r){let s=0,u=r();for(let i=0;i<p.length;i++){s+=p[i];if(u<s)return i}return p.length-1}
function direct(p,q,x,r){let a=r()<=Math.min(1,p[x]/q[x]);return {x,y:a?x:cat(norm(p.map((v,i)=>Math.max(0,v-q[i]))),r),a:Number(a)}}
function reverse(p,q,y,r){let a=r()<=Math.min(1,q[y]/p[y]);return {x:a?y:cat(norm(q.map((v,i)=>Math.max(0,v-p[i]))),r),y,a:Number(a)}}
function block(z,v,r,rev){let p=z.p[v],q=z.q;if(!rev){let x=cat(q,r),x2=cat(z.next[x].q,r),s=direct(p,q,x,r);if(s.a){let t=direct(z.next[x].p[v],z.next[x].q,x2,r);return [x,x2,s.y,t.y,s.a,t.a]}return [x,x2,s.y,cat(z.next[s.y].p[v],r),0,-1]}
let y=cat(p,r),s=reverse(p,q,y,r),y2=cat(z.next[y].p[v],r);
if(s.a){let t=reverse(z.next[y].p[v],z.next[y].q,y2,r);return [s.x,t.x,y,y2,1,t.a]}return [s.x,cat(z.next[s.x].q,r),y,y2,0,-1]}
function moments(p,q,y,lambda){let e=zero(V),m2=0,rb2=0,base2=0,accepted=0;for(let j=0;j<V;j++){let reward=Number(j===y),b=p.map((v,i)=>reward*(v-Number(i===j))),d=q.map((v,i)=>lambda*(v-Number(i===j))),reject=1-Math.min(1,q[j]/p[j]),g=b.map((v,i)=>v+reject*d[i]);for(let i=0;i<V;i++)e[i]+=p[j]*g[i];m2+=p[j]*((1-reject)*dot(b,b)+reject*dot(b.map((v,i)=>v+d[i]),b.map((v,i)=>v+d[i])));rb2+=p[j]*dot(g,g);base2+=p[j]*reject*(1-reject)*dot(d,d);accepted+=p[j]*(1-reject)}let m=dot(e,e);return {direct:m2-m,rb:rb2-m,conditional:base2,err:Math.abs(m2-rb2-base2),mean:e,accepted}}
function shifted(p,theta){let mx=Math.max(...theta);return norm(p.map((v,i)=>v*Math.exp(theta[i]-mx)))}
let audit={maxJoint:0,maxVarianceIdentity:0,maxReverseConditional:0,positions:0,blockDraws:0},rows=[],train=[];
for(let {wi,seed,c,xs} of cohorts){let row={wi,seed,c,direct:[0,0,0,0,0],reverse:[0,0,0,0,0],varDirect:0,varRB:0,cond:0};let rr=rng(1000*wi+seed*100+c*17);
for(let z of xs){let p=z.p[0],q=z.q,d=sum(p.map((v,i)=>Math.max(0,v-q[i])));for(let x=0;x<V;x++)for(let y=0;y<V;y++){let fw=x===y?Math.min(p[x],q[x]):0,rv=x===y?p[y]*Math.min(1,q[y]/p[y]):0;let a=d?Math.max(0,q[x]-p[x])*Math.max(0,p[y]-q[y])/d:0,b=d?p[y]*(1-Math.min(1,q[y]/p[y]))*Math.max(0,q[x]-p[x])/d:0;audit.maxJoint=Math.max(audit.maxJoint,Math.abs(fw-rv),Math.abs(a-b))}
let m=moments(p,q,z.y,.5);row.varDirect+=m.direct/4;row.varRB+=m.rb/4;row.cond+=m.conditional/4;audit.maxVarianceIdentity=Math.max(audit.maxVarianceIdentity,m.err);audit.positions++;
for(let i=0;i<512;i++)for(let [key,rev] of [["direct",false],["reverse",true]]){let b=block(z,0,rr,rev),out=row[key];out[0]+=b[4];out[1]+=Number(b[5]===1);out[2]+=Number(b[2]===z.y);out[3]+=Number(b[2]===b[3]);out[4]+=Number(b[1]===b[0]);audit.blockDraws++}}
rows.push(row)}
let config=new Map();for(let c of cohorts){let key=c.wi+","+c.seed;if(!config.has(key))config.set(key,[]);config.get(key).push(c)}
for(let [key,cs] of config){let tr=cs.slice(0,8),te=cs.slice(8),[wi,seed]=key.split(",").map(Number);for(let run=0;run<3;run++){let theta={direct:zero(V),reverse:zero(V),rb:zero(V),formal:zero(V)},r1=rng(10000+seed+wi*1000+run*99),r2=rng(20000+seed+wi*1000+run*99),r3=rng(30000+seed+wi*1000+run*99),r4=rng(40000+seed+wi*1000+run*99);let rr=rng(50000+seed+wi*1000+run*99);
function metric(th){let e=0,a=0;for(let c of te)for(let z of c.xs){let p=shifted(z.p[0],th),q=shifted(z.q,th);e+=p[z.y]/(4*te.length);a+=ov(p,q)/(4*te.length)}return {reward:e,overlap:a}}
let initial=metric(theta.direct),counts={direct:0,reverse:0,rb:0,formal:0};
for(let it=0;it<160;it++){let c=tr[Math.floor(rr()*tr.length)];let U=c.xs.map(()=>r2());for(let method of Object.keys(theta)){let th=theta[method],g=zero(V);for(let i=0;i<4;i++){let z=c.xs[i],p=shifted(z.p[0],th),q=shifted(z.q,th),s;if(method==="direct"){s=direct(p,q,cat(q,r1),r1)}else {let y=cat(p,()=>U[i]);s=method==="reverse"?reverse(p,q,y,r3):{y,a:0}}
let reward=Number(s.y===z.y),mask=method==="formal"?0:method==="rb"?1-Math.min(1,q[s.y]/p[s.y]):1-s.a;
for(let j=0;j<V;j++)g[j]+=(reward*(p[j]-Number(j===s.y))+.5*mask*(q[j]-Number(j===s.y)))/4;counts[method]+=mask}
for(let j=0;j<V;j++)th[j]-=.3*g[j]}}
train.push({key,run,initial,final:Object.fromEntries(Object.entries(theta).map(([m,t])=>[m,metric(t)])),counts})}}
let sumRows=key=>sum(rows.map(r=>r[key]))/rows.length,agg=key=>{let v=Array.from({length:5},(_,i)=>sum(rows.map(r=>r[key][i]))/(rows.length*4*512));return v};
let summary={direct:agg("direct"),reverse:agg("reverse"),varianceDirect:sumRows("varDirect"),varianceRB:sumRows("varRB"),conditionalVariance:sumRows("cond"),training:Object.fromEntries(["direct","reverse","rb","formal"].map(m=>[m,{reward:sum(train.map(r=>r.final[m].reward))/train.length,overlap:sum(train.map(r=>r.final[m].overlap))/train.length}])),initial:{reward:sum(train.map(r=>r.initial.reward))/train.length,overlap:sum(train.map(r=>r.initial.overlap))/train.length}};
return {summary,audit,rows,train}}
console.log(JSON.stringify(study(probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8'))),null,2));
