// Feature implementation adapted from team-hieracert/R11/probe.js at bcbfe8bc069664d25e76ba48b8fb9c4df76db75f (blob ce0ea6f4d4638ff6f3691e643539473da81054d1).
// node probe.js weights.json checkpoint.json input.txt
const fs=require("node:fs");
function run(raw,names){
const V=27,d=16,BOS=26,L=4,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((x,i)=>x+b[i]),rms=x=>{let s=Math.sqrt(dot(x,x)/d+1e-5);return x.map(t=>t/s)},sm=x=>{let m=Math.max(...x),a=x.map(v=>Math.exp(v-m)),s=sum(a);return a.map(v=>v/s)},arg=x=>x.indexOf(Math.max(...x)),tv=(a,b)=>sum(a.map((v,i)=>Math.abs(v-b[i])))/2;
let st=0;const U=()=>{st=(Math.imul(st,1664525)+1013904223)>>>0;return(st+.5)/4294967296};const draw=p=>{let u=U();for(let i=0;i<p.length;i++){u-=p[i];if(u<0)return i}return p.length-1};const randn=()=>Math.sqrt(-2*Math.log(U()))*Math.cos(2*Math.PI*U());
const vals=Array.from({length:7264},(_,i)=>raw[String(i)]);if(vals.some(x=>!Number.isFinite(x)))throw Error("checkpoint numeric schema");
let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=vals.slice(off,off+m);off+=m;return r});
const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");
const copy=w=>w.map(r=>r.slice()),zero=()=>Array.from({length:V},()=>Array(d).fill(0)),avg=ws=>ws[0].map((r,i)=>r.map((_,j)=>sum(ws.map(w=>w[i][j]))/ws.length));
function features(h,depth=2){let seq=h.map((t,i)=>rms(add(E[t],P[i]))),first=null;
 for(let l=0;l<depth;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(l===0)first=seq.map(x=>x.slice());}
 return {shallow:first,deep:seq};
}

const docs=names.split(/\r?\n/).map(s=>s.trim()).filter(s=>/^[a-z]{6,14}$/.test(s));
const buckets=Array.from({length:4},(_,c)=>docs.filter(x=>Math.min(3,Math.floor((x.charCodeAt(0)-97)/7))===c));
const examples=s=>{let h=[BOS,s.charCodeAt(0)-97,s.charCodeAt(1)-97];return s.slice(2,6).split("").map(c=>{let f=features(h),y=c.charCodeAt(0)-97;h.push(y);return {y,hd:f.deep.at(-1),hs:f.shallow.at(-1)}})};
const evals=buckets.map(a=>a.slice(-16).flatMap(examples)),trains=buckets.map(a=>a.slice(0,-16));
const probs=(w,h)=>sm(mat(w,h));
function learn(w,qw,batch,eta){let gp=zero(),gq=zero();for(const f of batch){let p=probs(w,f.hd),q=probs(qw,f.hs);for(let v=0;v<V;v++)for(let j=0;j<d;j++){gp[v][j]+=p[f.y]*(Number(v===f.y)-p[v])*f.hd[j];gq[v][j]+=(p[v]-q[v])*f.hs[j]}}return [w.map((r,v)=>r.map((z,j)=>z+eta*gp[v][j]/batch.length)),qw.map((r,v)=>r.map((z,j)=>z+eta*gq[v][j]/batch.length))]}
function joint(p,q,a){let delta=tv(p,q);return q.map((qx,x)=>p.map((py,y)=>(1-a)*((x===y?Math.min(qx,py):0)+(delta>1e-14?Math.max(0,qx-p[x])*Math.max(0,py-q[y])/delta:0))+a*qx*py))}
function clip(r,A){return Math.min(A*r,A*Math.max(.8,Math.min(1.2,r)))}
function gate(r,A){return A>=0?r>1.2:r<.8}
function audit(w,qw,f,a){const p0=probs(W,f.hd),q0=probs(W,f.hs),p1=probs(w,f.hd),q1=probs(qw,f.hs),J0=joint(p0,q0,a),J1=joint(p1,q1,a);let marginal=0,jointLoss=0,missing=0,gateDis=0,condVar=0,meanRatioErr=0,maxErr=0,oldA=0,newA=0;
for(let y=0;y<V;y++){let A=Number(y===f.y)-p0[f.y],r=p1[y]/p0[y],m=0,m2=0;marginal+=p0[y]*clip(r,A);for(let x=0;x<V;x++){const u=J0[x][y],v=J1[x][y];if(x===y){oldA+=u;newA+=v}if(u===0){missing+=v;continue}let rr=v/u; m+=u*rr/p0[y];m2+=u*rr*rr/p0[y];jointLoss+=u*clip(rr,A);gateDis+=u*Number(gate(rr,A)!==gate(r,A))}
meanRatioErr+=p0[y]*Math.abs(m-r);maxErr=Math.max(maxErr,Math.abs(m-r));condVar+=p0[y]*Math.max(0,m2-m*m)}
return {marginal,jointLoss,gap:marginal-jointLoss,missing,gateDis,condVar,meanRatioErr,maxErr,oldA,newA,tvP:tv(p0,p1),tvQ:tv(q0,q1)}}
let rows=[];for(let seed of [11,29,73])for(let eta of [.02,.2,1]){st=seed;let w=copy(W),qw=copy(W);for(let r=0;r<2;r++){let ew=[],eq=[];for(let e=0;e<2;e++){let ww=copy(w),qq=copy(qw);for(let l=0;l<2;l++){let clients=[];for(let c=2*e;c<2*e+2;c++){let batch=Array.from({length:8},()=>trains[c][Math.floor(U()*trains[c].length)]).flatMap(examples);clients.push(learn(ww,qq,batch,eta))}ww=avg(clients.map(x=>x[0]));qq=avg(clients.map(x=>x[1]))}ew.push(ww);eq.push(qq)}w=avg(ew);qw=avg(eq)}
for(let mode of ["draft-only","both"])for(let a of [0,.05,.25,1]){let ar=evals.flatMap(b=>b.map(f=>audit(mode==="draft-only"?W:w,qw,f,a)));let out={seed,eta,mode,a,n:ar.length};for(let k of Object.keys(ar[0]))out[k]=k==="maxErr"?Math.max(...ar.map(x=>x[k])):sum(ar.map(x=>x[k]))/ar.length;out.gapNegative=ar.filter(x=>x.gap<-1e-10).length;rows.push(out)}
}
return {params:off,docs:docs.length,buckets:buckets.map(x=>x.length),evalContexts:evals.flat().length,rows};
}
if(require.main===module){const a=JSON.parse(fs.readFileSync(process.argv[2],"utf8")),b=JSON.parse(fs.readFileSync(process.argv[3],"utf8")),t=fs.readFileSync(process.argv[4],"utf8");console.log(JSON.stringify({weights:run(a,t),checkpoint:run(b.weights,t)},null,2))}
