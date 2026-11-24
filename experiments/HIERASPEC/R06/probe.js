// HIERASPEC/R06: exact integer transportation enumeration; no dependencies
function run(){
 const N=24,V=3,seeds=[11,29,73],sigmas=[.15,.7,1.5],trials=100,lambdas=[0,.25,.5,1,2,4,8];
 let st=0;const u=()=>{st=(Math.imul(st,1664525)+1013904223)>>>0;return (st+.5)/4294967296};
 const normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
 const sum=a=>a.reduce((s,x)=>s+x,0), h=x=>x>0?-x*Math.log2(x):0;
 function probs(z){let m=Math.max(...z),a=z.map(x=>Math.exp(x-m)),s=sum(a);return a.map(x=>x/s)}
 function counts(p){let a=p.map(x=>x*(N-V)),b=a.map(x=>Math.floor(x)+1),idx=a.map((x,i)=>i).sort((i,j)=>(a[j]-Math.floor(a[j]))-(a[i]-Math.floor(a[i])));for(let k=0;k<N-sum(b);){break}let rem=N-sum(b);for(let j=0;j<rem;j++)b[idx[j]]++;return b}
 function evalM(m,q,p){let A=0,H=0;for(let i=0;i<V;i++)for(let j=0;j<V;j++){let x=m[i*V+j];if(i===j)A+=x; if(x>0)H-=x/N*Math.log2(x/q[i]);}return {alpha:A/N,H}}
 function enumerate(q,p,fn){let M=Array(9).fill(0),c=p.slice();function go(i,j,left){if(i===2){if(sum(c)!==q[2])throw Error("bad");for(let k=0;k<3;k++)M[6+k]=c[k];fn(M);return}if(j===2){if(left>c[2])return;M[i*3+2]=left;c[2]-=left;go(i+1,0,q[i+1]);c[2]+=left;return}for(let x=0;x<=Math.min(left,c[j]);x++){M[i*3+j]=x;c[j]-=x;go(i,j+1,left-x);c[j]+=x;}}go(0,0,q[0]);}
 function standard(q,p){let m=Array(9).fill(0),d=q.map((v,i)=>Math.max(0,v-p[i])),r=p.map((v,i)=>Math.max(0,v-q[i])),z=sum(d);for(let i=0;i<3;i++){m[i*3+i]=Math.min(q[i],p[i]);if(z)for(let j=0;j<3;j++)m[i*3+j]+=d[i]*r[j]/z}return m}
 const rows=[];
 for(const seed of seeds)for(const sigma of sigmas){st=seed;let acc={stdH:0,maxEntropyOptimalH:0,minH:0,minHAlpha:0,alpha:0,arQCrossEntropy:0,arOracleEntropy:0,fano:0,conflict:0,strictMaxFaceGain:0,matrices:0,lambda:lambdas.map(l=>({lambda:l,alpha:0,H:0,cost:0}))};
 for(let t=0;t<trials;t++){let z=Array.from({length:3},normal),p=counts(probs(z)),q=counts(probs(z.map(x=>x+sigma*normal()))),std=evalM(standard(q,p),q,p),maxA=std.alpha;
 let bestH=Infinity,bestA=0,face=Infinity,best=lambdas.map(()=>({cost:Infinity,alpha:0,H:0}));
 enumerate(q,p,m=>{acc.matrices++;let {alpha,H}=evalM(m,q,p);if(H<bestH-1e-12||Math.abs(H-bestH)<1e-12&&alpha>bestA){bestH=H;bestA=alpha}if(Math.abs(alpha-maxA)<1e-12)face=Math.min(face,H);
 lambdas.forEach((l,i)=>{let cost=H+l*(1-alpha);if(cost<best[i].cost-1e-12||Math.abs(cost-best[i].cost)<1e-12&&alpha>best[i].alpha)best[i]={cost,alpha,H}})});
 if(face>std.H+1e-10)throw Error("face exceeds standard");
 acc.stdH+=std.H;acc.maxEntropyOptimalH+=face;acc.minH+=bestH;acc.minHAlpha+=bestA;acc.alpha+=maxA;acc.arQCrossEntropy-=sum(p.map((v,i)=>v/N*Math.log2(q[i]/N)));acc.arOracleEntropy+=sum(p.map(v=>h(v/N)));let d=1-maxA;acc.fano+=h(d)+h(1-d)+d*Math.log2(V-1);acc.conflict+=Number(bestA<maxA-1e-12);acc.strictMaxFaceGain+=Number(face<std.H-1e-12);best.forEach((x,i)=>{for(let k of ["alpha","H","cost"])acc.lambda[i][k]+=x[k]});
 }
 for(let k of Object.keys(acc))if(k!=="lambda"&&k!=="matrices")acc[k]/=trials;for(let x of acc.lambda)for(let k of ["alpha","H","cost"])x[k]/=trials;
 rows.push({seed,sigma,...acc});
 }
 return {env:"V8 deterministic finite integer-grid enumeration; synthetic distributions, no GPU/real model/network",N,V,seeds,sigmas,trials,lambdas,rows};
}
console.log(JSON.stringify(run(),null,2));
