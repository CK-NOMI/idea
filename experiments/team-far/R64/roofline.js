function run(){
const rhoList=[64,128,312000/1555],bs=[1,8,32,64,128],alphas=[.55,.75,.9],drafts=[.04,.12,.25],ratios=[.5,2,8,32],avg=a=>a.reduce((s,x)=>s+x,0)/a.length;
let rows=[],checks=0;
function solve(A,R){
let best={t:Infinity};const gen=A.filter(x=>x.g>0).map(x=>({...x,x:x.m/x.g,y:x.t/x.g})),train=A.filter(x=>x.g===0&&x.m>0).reduce((a,b)=>a.t/a.m<=b.t/b.m?a:b);
function upd(t,a,b,w){if(t<best.t)best={t,a:a?{k:a.k,m:a.m}:null,b:b?{k:b.k,m:b.m}:null,w}}
let slope=train.t/train.m;
for(let a of gen)if(a.x<=R)upd(a.y+(R-a.x)*slope,a,train,1);
for(let i=0;i<gen.length;i++)for(let j=i+1;j<gen.length;j++){
let a=gen[i],b=gen[j];if(a.x>b.x){let q=a;a=b;b=q}if(a.x<=R&&R<=b.x&&b.x>a.x){let w=(b.x-R)/(b.x-a.x);upd(w*a.y+(1-w)*b.y,a,b,w)}checks++
}
return best
}
for(let rho of rhoList)for(let b of bs)for(let alpha of alphas)for(let draft of drafts){
let train=Array.from({length:32},(_,j)=>{let m=16*(j+1);return {k:-1,g:0,m,t:Math.max(1,m/rho)}});
let gen=Array.from({length:9},(_,k)=>{let g=b*(1-alpha**(k+1))/(1-alpha);return {k,m:0,g,t:Math.max(1,b*(k+1)/rho)+draft*k}});
let joint=gen.flatMap(a=>Array.from({length:33},(_,j)=>{let m=16*j;return {...a,m,t:Math.max(1,(b*(a.k+1)+m)/rho)+draft*a.k+(m>0?.05:0)}}));
let solo=gen.reduce((a,b)=>a.g/a.t>=b.g/b.t?a:b);
for(let R of ratios){
let sep=solve([...gen,...train],R),ar=solve([...joint.filter(a=>a.k===0),...train],R),fixedSolo=solve([...joint.filter(a=>a.k===solo.k),...train],R),opt=solve([...joint,...train],R);
let fixed=gen.map(x=>solve([...joint.filter(a=>a.k===x.k),...train],R)).reduce((a,b)=>a.t<=b.t?a:b);
rows.push({rho,b,alpha,draft,R,soloK:solo.k,sep:sep.t,ar:ar.t,fixedSolo:fixedSolo.t,fixed:fixed.t,opt:opt.t,vsSep:opt.t/sep.t,vsSolo:opt.t/fixedSolo.t,vsFixed:opt.t/fixed.t,vsAR:opt.t/ar.t,chosen:opt.a,other:opt.b,w:opt.w});
}
}
let summary=[];for(let b of bs){let a=rows.filter(r=>r.b===b);summary.push({b,n:a.length,vsSep:avg(a.map(x=>x.vsSep)),vsSolo:avg(a.map(x=>x.vsSolo)),vsFixed:avg(a.map(x=>x.vsFixed)),vsAR:avg(a.map(x=>x.vsAR)),gain15Solo:a.filter(x=>x.vsSolo<=.85).length,gain15Fixed:a.filter(x=>x.vsFixed<=.85).length,arOptimal:a.filter(x=>x.vsAR>=1-1e-10).length})}
return {assumptions:{rhoList,bs,alphas,drafts,ratios,overhead:.05,trainChunk:16,maxTrain:512},checks,summary,rows};
}
console.log(JSON.stringify(run(),null,2));