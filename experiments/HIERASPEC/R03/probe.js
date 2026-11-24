// HIERASPEC/R03: exact finite path enumeration and residual sampler; Node.js, no dependencies.
const seeds=[11,29,73], ks=[4,8,12], sigmas=[.1,.6,1.2], Y=3,C=8, p=[.45,.35,.2];
let state=0;
function u(){state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296}
function normal(){return Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u())}
const sigmoid=x=>1/(1+Math.exp(-x));
const soft=a=>{let m=Math.max(...a),b=a.map(x=>Math.exp(x-m)),s=b.reduce((x,y)=>x+y,0);return b.map(x=>x/s)};
const sum=a=>a.reduce((x,y)=>x+y,0);
const mean=a=>sum(a)/a.length;
function prod(a){let v=[1];for(let t of a)v=v.flatMap(x=>[x*(1-t),x*t]);return v}
function kl(a,b){return a.reduce((s,x,i)=>s+(x>0?x*Math.log(x/b[i]):0)+(x<1?(1-x)*Math.log((1-x)/(1-b[i])):0),0)}
function draw(w){let x=u()*sum(w);for(let i=0;i<w.length;i++){x-=w[i];if(x<0)return i}return w.length-1}
function evaluate(){
let rows=[];
for(let k of ks)for(let sigma of sigmas)for(let seed of seeds){
 state=seed;let b=Array.from({length:Y},()=>Array.from({length:k},()=>normal()*.8));
 let shift=Array.from({length:2},()=>b.map(v=>v.map(()=>sigma*normal())));
 let L=Array.from({length:C},(_,c)=>b.map((v,y)=>v.map((a,j)=>a+shift[c>>2][y][j]+.25*normal())));
 let Q=Array.from({length:C},()=>soft(p.map(v=>Math.log(v)+.35*normal())));
 let R=(clients,y)=>L[0][y].map((_,j)=>sigmoid(mean(clients.map(c=>L[c][y][j]))));
 let global=p.map((_,y)=>R([...Array(C).keys()],y));
 let regional=[0,1].map(e=>p.map((_,y)=>R([...Array(4).keys()].map(i=>e*4+i),y)));
 let out={}, names=["canonical","uniform","global","edge","flat_same_groups","posterior_oracle"];
 let maxerr=0,maxViolation=0,projected=0,independent=0,invalidTV=0;
 for(let c=0;c<C;c++){
  let s=L[c].map(v=>v.map(sigmoid)), q=Q[c];
  let proj=sum(p.map((v,y)=>Math.min(v,q[y])));projected+=proj/C;independent+=sum(p.map((v,y)=>v*q[y]))/C;
  let ss=s.map(prod);
  // Wrong single-canonical-path masses, normalized and used as if true marginal q.
  let wrong=q.map((v,y)=>v*ss[y][0]),z=sum(wrong);wrong=wrong.map(v=>v/z);
  let acc=q.map((v,y)=>v*Math.min(1,p[y]/wrong[y])), rej=1-sum(acc);
  let rr=p.map((v,y)=>Math.max(0,v-wrong[y])),rz=sum(rr);
  let actual=acc.map((v,y)=>v+(rz>1e-15?rej*rr[y]/rz:0));
  invalidTV+=sum(actual.map((v,y)=>Math.abs(v-p[y])))/(2*C);
  const maps={canonical:p.map(()=>Array(k).fill(0)),uniform:p.map(()=>Array(k).fill(.5)),global,edge:regional[c>>2],flat_same_groups:regional[c>>2],posterior_oracle:s};
  for(let name of names){
   let r=maps[name],rs=r.map(prod),a=0,gapKL=0,bound=0;
   for(let y=0;y<Y;y++){
    let m=0,rMass=0,tv=0;
    for(let i=0;i<ss[y].length;i++){let Qz=q[y]*ss[y][i],Pz=p[y]*rs[y][i];m+=Math.min(Qz,Pz);rMass+=Math.max(0,Pz-Qz);tv+=Math.abs(ss[y][i]-rs[y][i])/2;}
    maxerr=Math.max(maxerr,Math.abs(m+rMass-p[y]));a+=m;
    gapKL+=p[y]*kl(r[y],s[y]);bound+=Math.min(p[y],q[y])*tv;
   }
   maxViolation=Math.max(maxViolation,proj-a-bound,0);
   let o=out[name]??={accept:0,elboGap:0,lossBound:0};o.accept+=a/C;o.elboGap+=gapKL/C;o.lossBound+=bound/C;
  }
 }
 rows.push({k,sigma,seed,projected,independent,invalidTV,maxerr,maxViolation,methods:out});
}
// Explicit MC sanity check: 2 target tokens x 2 paths, compare exact lifted residual rejection.
state=20261124;const P=[.4,.1,.2,.3],Q=[.1,.3,.15,.45],N=200000;
let count=[0,0],accepted=0,attempts=0;
for(let i=0;i<N;i++){let z=draw(Q);if(u()<Math.min(1,P[z]/Q[z]))accepted++;else{do{z=draw(P);attempts++}while(u()>=Math.max(0,1-Q[z]/P[z]))}count[z>>1]++}
return {environment:"JS V8 CPU; exact enumeration, synthetic latent segmentation product posterior, no GPU or time",p,seeds,ks,sigmas,rows,mc:{N,count,accepted,attempts,theoryAcceptance:sum(P.map((v,i)=>Math.min(v,Q[i])))}};
}
console.log(JSON.stringify(evaluate(),null,2));
