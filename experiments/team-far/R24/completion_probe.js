// node completion_probe.js results.json
const fs=require('node:fs');
function optimize(data){
const G=8,eps=1e-6,sum=a=>a.reduce((s,x)=>s+x,0),avg=a=>sum(a)/a.length;
function choose(n,k){let v=1;for(let j=1;j<=k;j++)v*=(n-j+1)/j;return v}
const bin=(n,a,p)=>choose(n,a)*p**a*(1-p)**(n-a);
function coeff(m){if(m===0||m===G)return [0,0];let s=Math.sqrt(m*(G-m))/G+eps;return [(G-m)/(G*G*s),-m/(G*G*s)]}
function state(row,b,pa,mode,a,r){
let p=row.p,n=G-b,hp=0,hm=0,hd=0,prob=bin(b,a,p),mixed=a>0&&a<b;
if(mode!=="gate"||mixed){for(let c=0;c<=n;c++){let z=bin(n,c,pa),[P,N]=coeff(a+c);hp+=z*P;hm+=z*N;if(mode==="oracle")hd+=z*(P*c*(1-pa)-N*(n-c)*pa)}}
let proxyOn=(mode!=="gate"||mixed),D=0,fullExtra=0;
for(let c=0;c<=n;c++){let z=bin(n,c,p),m=a+c,[P,N]=coeff(m);
let mean=(P-hp)*a*(1-p)-(N-hm)*(b-a)*p+P*c*(1-p)-N*(n-c)*p-hd;
let v=(P-hp)**2*a*row.vp+(N-hm)**2*(b-a)*row.vm+P*P*c*row.vp+N*N*(n-c)*row.vm;
D+=z*(v+mean*mean*row.Delta);
if(m>0&&m<G)fullExtra+=z*(proxyOn?n:G)}
return {prob,D,c0:b+r*(proxyOn?b:0),c1:n+r*fullExtra}
}
function agg(rows,b,phat,mode,r){
let baseV=0,baseC=0,S=Array.from({length:b+1},()=>({prob:0,D:0,c0:0,c1:0}));
for(let row of rows){let base=row.methods.find(m=>m.b===8&&m.method==="all");baseV+=base.var;baseC+=base.gen+r*base.back;for(let a=0;a<=b;a++){let s=state(row,b,mode==="oracle"?row.p:phat,mode,a,r),v=S[a];v.prob+=s.prob;for(let k of ["D","c0","c1"])v[k]+=s.prob*s[k]}}
return {baseV,baseC,S,n:rows.length}
}
function evaluate(s,pi,b,r){let V=s.baseV,C=0,gen=0;for(let a=0;a<=b;a++){let v=s.S[a],t=pi[a];V+=v.D*(1/t-1);C+=v.c0+t*v.c1;gen+=v.prob*(b+t*(G-b))}return {variance:V/s.baseV,cost:C/s.baseC,varCost:V*C/(s.baseV*s.baseC),gen:gen/s.n,back:(C-gen)/(r*s.n)}}
let out=[],groups={};
for(let row of data.rows){let key=[row.wi,row.seed,row.region].join(",");(groups[key]??=[]).push(row)}
for(let [key,rows] of Object.entries(groups)){
let train=rows.slice(0,12),test=rows.slice(12),phat=avg(train.map(x=>x.p));
for(let r of [.25,1,2,5,10]){
let options=[];
for(let b of [2,3,4,6])for(let mode of ["proxy","gate","oracle"]){
let tr=agg(train,b,phat,mode,r),te=agg(test,b,phat,mode,r),best=null;
for(let e=-14;e<=14;e+=.25){let lam=10**e;let pi=tr.S.map(s=>Math.max(.02,Math.min(1,Math.sqrt(s.D/(lam*s.c1||1e-99)))));let fit=evaluate(tr,pi,b,r);if(!best||fit.varCost<best.train.varCost)best={b,mode,pi,train:fit,test:evaluate(te,pi,b,r)}}
options.push(best)}
let feasible=options.filter(x=>x.mode!=="oracle").sort((a,b)=>a.train.varCost-b.train.varCost)[0],oracle=options.filter(x=>x.mode==="oracle").sort((a,b)=>a.train.varCost-b.train.varCost)[0];
out.push({key,r,phat,feasible,oracle})
}}
let summary=[];
for(let r of [.25,1,2,5,10]){let a=out.filter(x=>x.r===r);for(let m of ["feasible","oracle"])summary.push({r,mode:m,win:a.filter(x=>x[m].test.varCost<1-1e-10).length,n:a.length,cost:avg(a.map(x=>x[m].test.cost)),variance:avg(a.map(x=>x[m].test.variance)),varCost:avg(a.map(x=>x[m].test.varCost)),best:Math.min(...a.map(x=>x[m].test.varCost)),worst:Math.max(...a.map(x=>x[m].test.varCost))})}
let f=data.summary.filter(x=>x.method==="all"),avgB=f.find(x=>x.b===8).back,realProfiles=[{name:"RL-Think",G:87,T:137.4,L:18.1,train:30.5},{name:"RL-Zero",G:56.6,T:107.5,L:18.1,train:30.5}].map(x=>({...x,r:(x.L+x.train)/x.G*8/avgB,methods:f.map(s=>{let newT=x.T-x.G-x.L-x.train+x.G*s.gen/8+(x.L+x.train)*s.back/avgB;return {b:s.b,timeProxy:newT/x.T,varianceRatio:s.varRatio,varCost:newT/x.T*s.varRatio}})}));
return {out,summary,realProfiles}
}
console.log(JSON.stringify(optimize(JSON.parse(fs.readFileSync(process.argv[2],'utf8'))),null,2));
