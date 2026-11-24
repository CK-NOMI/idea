// node probe.js weights.json checkpoint.json input.txt
const fs=require('node:fs');
function probe(raws,names){
const d=16,H=4,k=4,BOS=26, sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),mul=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),scale=(a,s)=>a.map(x=>x*s),norm=x=>Math.sqrt(dot(x,x)),rms=x=>scale(x,1/Math.sqrt(dot(x,x)/d+1e-5)),sm=x=>{let m=Math.max(...x),a=x.map(v=>Math.exp(v-m));return scale(a,1/sum(a))};
function basis(rows){let bs=[];for(let row of rows){let v=row.slice();for(let j=0;j<2;j++)for(let b of bs)v=sub(v,scale(b,dot(b,v)));let n=norm(v);if(n>1e-10*Math.max(1,norm(row)))bs.push(scale(v,1/n))}return bs}
function proj(q,B){return B.reduce((a,b)=>add(a,scale(b,dot(q,b))),q.map(()=>0))}
function model(raw){let off=0;const take=(n,m)=>Array.from({length:n},()=>Array.from({length:m},()=>raw[off++]));let E=take(27,d),P=take(16,d),W=take(27,d),L=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error('shape');return {E,P,W,L}}
const docs=[...new Set(names.split(/\r?\n/).map(s=>s.trim()).filter(s=>/^[a-z]{12,14}$/.test(s)))];const buckets=Array.from({length:4},(_,c)=>docs.filter(s=>Math.min(3,Math.floor((s.charCodeAt(0)-97)/7))===c));let rows=[],maxLogit=0,maxOut=0,maxGrad=0;
for(let wi=0;wi<raws.length;wi++){let {E,P,L}=model(raws[wi]);for(let c=0;c<4;c++)for(let name of buckets[c].slice(0,32)){let t=[BOS,...name.split('').map(x=>x.charCodeAt(0)-97)],seq=t.map((x,i)=>rms(add(E[x],P[i])));for(let l=0;l<2;l++){let b=L[l],z=seq.map(rms),Q=z.map(x=>mul(b.q,x)),K=z.map(x=>mul(b.k,x)),V=z.map(x=>mul(b.v,x));let output=seq.map((x,i)=>{let a=[];for(let h=0;h<H;h++){let sl=v=>v.slice(h*k,(h+1)*k),q=sl(Q[i]),ks=K.slice(0,i+1).map(sl),vs=V.slice(0,i+1).map(sl),w=sm(ks.map(v=>dot(q,v)/2));for(let j=0;j<k;j++)a.push(sum(w.map((v,t)=>v*vs[t][j])));
if([2,3,4,5,8,12].includes(i)){let m=i,pk=ks.slice(0,m),pv=vs.slice(0,m),B=basis(pk),C=basis(pv),qp=proj(q,B),perp=sub(q,qp),full=ks.map(v=>dot(q,v)/2),reduced=ks.map((v,t)=>dot(t<m?qp:q,v)/2),p=sm(reduced),o=vs[0].map((_,j)=>sum(w.map((v,t)=>v*vs[t][j]))),op=vs[0].map((_,j)=>sum(p.map((v,t)=>v*vs[t][j]))),g=Array.from({length:k},(_,j)=>Math.sin((j+1)*(h+1)+i+c)),coeff=w.slice(0,m).map((v,t)=>v*dot(g,sub(vs[t],o))/2);let ref=coeff.map(v=>scale(q,v)),naive=coeff.map(v=>scale(qp,v)),fix=naive.map((a,t)=>add(a,scale(perp,coeff[t]))),den=sum(ref.flat().map(v=>v*v)),err=sum(ref.flat().map((v,j)=>(v-naive.flat()[j])**2));maxLogit=Math.max(maxLogit,...full.map((v,j)=>Math.abs(v-reduced[j])));maxOut=Math.max(maxOut,...o.map((v,j)=>Math.abs(v-op[j])));maxGrad=Math.max(maxGrad,...ref.flat().map((v,j)=>Math.abs(v-fix.flat()[j])));rows.push({wi,c,l,h,m,rK:B.length,rV:C.length,den,err,queryNorm:norm(q),perpNorm:norm(perp)})}}
let y=add(x,mul(b.o,a));return add(y,mul(b.f2,mul(b.f1,rms(y)).map(x=>Math.max(0,x))))});seq=output}}}
let groups=[2,3,4,5,8,12].map(m=>{let a=rows.filter(x=>x.m===m),n=a.length;return {m,n,avgRankK:sum(a.map(x=>x.rK))/n,avgRankV:sum(a.map(x=>x.rV))/n,fullK:a.filter(x=>x.rK===k).length/n,naiveRelativeGradientError:Math.sqrt(sum(a.map(x=>x.err))/sum(a.map(x=>x.den))),avgQuerySave:1-sum(a.map(x=>x.rK))/(n*k)}});return {rows,groups,maxLogit,maxOut,maxGrad,docs:docs.length,buckets:buckets.map(x=>x.length)}
}
function proxy(){
const d=128,Hq=28,Hkv=4,nt=128;let rows=[];
for(let r of [16,32,64,128])for(let ratio of [1,1.5,2,3,4]){
let nv=nt*ratio,B=Hq*(2*d*(nv+nt)+nv+2*nt),save=Hq*(d-r)*(nv-nt),basis=r<d?Hkv*d*r:0;
rows.push({r,ratio,nt,nv,R10Elements:B,querySaving:save/(Hq*d*nv),wholeInterfaceSavingBeforeBasis:save/B,basisElements:basis,wholeInterfaceSavingAfterBasis:(save-basis)/B,replicationElements:4*Hkv*d*r,netToReplication:(B-save+basis)/(4*Hkv*d*r)})}
return rows}

const source={"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}]};
console.log(JSON.stringify({source,...probe([JSON.parse(fs.readFileSync(process.argv[2],'utf8')),JSON.parse(fs.readFileSync(process.argv[3],'utf8')).weights],fs.readFileSync(process.argv[4],'utf8')),proxy:proxy()},null,2));
