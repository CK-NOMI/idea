// node probe.js wallpaper.txt alice.txt (pinned raw source versions in results.json)
const fs=require('node:fs');
function probe(docs){
 const V=27,D=V*V,alphabet=" abcdefghijklmnopqrstuvwxyz",sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i])),norm=a=>{let s=sum(a);return a.map(x=>x/s)},zero=()=>Array(V).fill(0),sq=a=>dot(a,a);
 const clean=s=>s.split(/\*\*\* START OF THIS PROJECT GUTENBERG EBOOK[^\n]*\n/)[1].split(/\*\*\* END OF THIS PROJECT GUTENBERG EBOOK/)[0].toLowerCase().replace(/[^a-z]+/g," ").trim();
 let parts=[];for(const doc of docs){let s=clean(doc.text),k=s.lastIndexOf(" ",Math.floor(s.length/2));parts.push(s.slice(0,k),s.slice(k+1))}
 let global=new Float64Array(V**3),big=new Float64Array(D),counts=[],sizes=[];
 for(const part of parts){let chars=Array.from(part,c=>alphabet.indexOf(c)),train=chars.slice(0,Math.floor(chars.length*.8)),tri=new Float64Array(V**3);for(let t=2;t<train.length;t++){let s=train[t-2]*V+train[t-1],v=train[t];tri[s*V+v]++;global[s*V+v]++;big[(s%V)*V+v]++}counts.push(tri);sizes.push({train:train.length,heldout:chars.length-train.length})}
 const q=Array.from({length:V},(_,b)=>norm(Array.from(big.slice(b*V,(b+1)*V),x=>x+.5)));
 let context=Array.from({length:D},(_,s)=>({s,n:sum(Array.from(global.slice(s*V,(s+1)*V)))})).sort((a,b)=>b.n-a.n).slice(0,64).map(x=>x.s);
 const ps=s=>counts.map(c=>norm(Array.from(c.slice(s*V,(s+1)*V),(v,j)=>v+5*q[s%V][j])));
 function cov(p,r,Q){let d1=sum(p.map((x,j)=>Math.max(0,x-Q[j]))),d2=sum(r.map((x,j)=>Math.max(0,x-Q[j]))),rr1=d1?p.map((v,j)=>Math.max(0,v-Q[j])/d1):zero(),rr2=d2?r.map((v,j)=>Math.max(0,v-Q[j])/d2):zero();let C=0;for(let x=0;x<V;x++){let a=Math.min(1,p[x]/Q[x]),b=Math.min(1,r[x]/Q[x]);let u=rr1.map((v,j)=>(1-a)*v+(j===x?a:0)-p[j]),v=rr2.map((v,j)=>(1-b)*v+(j===x?b:0)-r[j]);C+=Q[x]*dot(u,v)}return {C,d1,d2,v1:1-sq(p),v2:1-sq(r)}}
 const analytic=[];for(const s of context){let p=ps(s);for(let e=0;e<2;e++){let c=cov(p[2*e],p[2*e+1],q[s%V]),base=(c.v1+c.v2)/4;analytic.push({s,e,...c,meanIndependent:base,meanShared:base+c.C/2,disagreementIndependent:base,disagreementShared:base-c.C/2})}}
 let st=0;const U=()=>{st=(Math.imul(1664525,st)+1013904223)>>>0;return (st+.5)/4294967296};const draw=(p,u=U())=>{for(let j=0;j<V;j++){u-=p[j];if(u<0)return j}return V-1};
 function run(p,Q,curv,seed,method){
  st=seed;const R=30,E=2,K=2,eta=.1;let w=zero(),y=[zero(),zero()],zPower=0,yPower=0,gapSum=0,acc=0,nSD=0,qDraws=0,pDraws=0,resDraws=0;const opt=zero();let scale=sum(curv);
  for(let j=0;j<V;j++)opt[j]=sum(p.map((v,i)=>curv[i]*v[j]))/scale;
  let res=p.map(v=>{let d=v.map((x,j)=>Math.max(0,x-Q[j]));return sum(d)>0?norm(d):zero()});
  const gap=v=>scale/8*sum(v.map((x,j)=>(x-opt[j])**2));
  for(let t=0;t<R;t++){
   let end=[],z=Array.from({length:4},zero);
   for(let e=0;e<2;e++){
    let wE=w.slice();for(let r=0;r<E;r++){
     let ws=[wE.slice(),wE.slice()];for(let k=0;k<K;k++){
      let labels=[];if(method==="sharedSD"||method==="indSD"){
       let x0=draw(Q);qDraws++;for(let i=0;i<2;i++){let n=2*e+i,x=i===0||method==="sharedSD"?x0:draw(Q);if(i===1&&method==="indSD")qDraws++;let accepted=U()*Q[x]<p[n][x];nSD++;acc+=Number(accepted);if(!accepted)resDraws++;labels.push(accepted?x:draw(res[n]))}
      }else if(method==="exact"){labels=[-1,-1]}
      else {let u=U();for(let i=0;i<2;i++){let n=2*e+i,uu=i===0||method==="commonAR"?u:method==="antiAR"?1-u:U();labels.push(draw(p[n],uu));pDraws++}}
      for(let i=0;i<2;i++){let n=2*e+i;for(let j=0;j<V;j++)ws[i][j]-=eta*(curv[n]*(ws[i][j]-(labels[i]<0?p[n][j]:Number(labels[i]===j)))+z[n][j]+y[e][j])}
     }
     let mean=ws[0].map((v,j)=>(v+ws[1][j])/2);for(let i=0;i<2;i++){let n=2*e+i;for(let j=0;j<V;j++)z[n][j]+=(ws[i][j]-mean[j])/(eta*K);zPower+=sq(z[n])/4/E/R}wE=mean;
    }end.push(wE)
   }
   w=end[0].map((v,j)=>(v+end[1][j])/2);for(let e=0;e<2;e++){for(let j=0;j<V;j++)y[e][j]+=(end[e][j]-w[j])/(eta*K*E);yPower+=sq(y[e])/2/R}gapSum+=gap(w)/R
  }
  return {gap:gap(w),gapMean:gapSum,zPower,yPower,accept:nSD?acc/nSD:null,qDraws,pDraws,resDraws};
 }
 const methods=["indSD","sharedSD","indAR","commonAR","antiAR","exact"],seeds=[11,29,73],curves=[[1,1,1,1],[.5,1.5,.5,1.5],[.2,1.8,1.8,.2]],rows=[];
 for(let c=0;c<curves.length;c++)for(const seed of seeds)for(const method of methods){let a=[];for(const s of context)a.push(run(ps(s),q[s%V],curves[c],seed+4099*s,method));let mean=k=>sum(a.map(x=>x[k]??0))/a.length;rows.push({c,seed,method,gap:mean("gap"),gapMean:mean("gapMean"),zPower:mean("zPower"),yPower:mean("yPower"),accept:method.endsWith("SD")?mean("accept"):null,qDraws:mean("qDraws"),pDraws:mean("pDraws"),resDraws:mean("resDraws")})}
 let group=[];for(let c=0;c<curves.length;c++)for(const method of methods){let a=rows.filter(r=>r.c===c&&r.method===method);let o={c,method};for(const k of ["gap","gapMean","zPower","yPower","accept","qDraws","pDraws","resDraws"])o[k]=a[0][k]===null?null:sum(a.map(r=>r[k]))/a.length;group.push(o)}
 let total=k=>sum(analytic.map(r=>r[k]));return {source:docs.map(({text,...r})=>r),sizes,context,curves,seeds,analytic,rows,group,summary:{positive:analytic.filter(r=>r.C>0).length,n:analytic.length,meanVarRatio:total("meanShared")/total("meanIndependent"),disagreementRatio:total("disagreementShared")/total("disagreementIndependent"),overlap:1-(total("d1")+total("d2"))/(2*analytic.length)},environment:"JS V8 CPU; real corpus fitted count teacher distributions; simulated quadratic MTGC-style optimizer with fixed teacher targets; no transformer/GPU/network/true SD block decoding"};
}
const source=[{"repo":"GITenberg/The-Yellow-Wallpaper_1952","path":"1952.txt","ref":"ce7c5bc1a5cf0fe23bcb265545a900365dbae29c","blob":"3a2b5f55c6f1d13a7ae623282d38fde5a6ff352b"},{"repo":"GITenberg/Alice-s-Adventures-in-Wonderland_11","path":"11.txt","ref":"7f18748e114b994d5d9454d5793581bc0b9cf9d7","blob":"9d73b1d7447b7ddc158b1b157f5f700d21bd2fb7"}];
console.log(JSON.stringify(probe(source.map((s,i)=>({...s,text:fs.readFileSync(process.argv[i+2],'utf8')}))),null,2));
