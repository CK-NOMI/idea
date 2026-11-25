function run(raw){
 const sum=a=>a.reduce((s,x)=>s+x,0),pos=x=>Math.max(0,x),tv=(a,b)=>sum(a.map((x,i)=>Math.abs(x-b[i])))/2;
 const rows=raw.rows.filter(x=>x.K===27&&x.N===32),K=64,grid=Array.from({length:K+1},(_,j)=>j/K);
 function curve(a,b){let s=a.p.map((p,i)=>p?pos(p-a.q[i])/p:0),t=b.q.map((q,i)=>q?pos(q-b.p[i])/q:0);return {a,b,s,t,da:tv(a.p,a.q),db:tv(b.p,b.q)}}
 function cumulative(c,u){return [sum(c.s.map((s,i)=>Math.min(s,u)*c.a.p[i])),sum(c.t.map((t,i)=>Math.min(t,u)*c.b.q[i]))]}
 function benefit(c,l,r){let a=cumulative(c,l),b=cumulative(c,r);return (b[0]-a[0])*(b[1]-a[1])/(r-l)}
 function cost(c,edges){return c.da+c.db-sum(edges.slice(1).map((r,i)=>benefit(c,edges[i],r)))}
 function exact(c){return c.da+c.db-sum(c.s.flatMap((s,i)=>c.t.map((t,j)=>c.a.p[i]*c.b.q[j]*Math.min(s,t))))}
 function proj(r){let R=sum(r.p.map((p,i)=>p>r.q[i]?p:0)),D=sum(r.p.map((p,i)=>p<r.q[i]?p:0)),d=tv(r.p,r.q);return {...r,q:r.p.map((p,i)=>p>r.q[i]?p*(1-d/R):p<r.q[i]?p*(1+d/D):p)}}
 function kl(r){return sum(r.p.map((p,i)=>p*Math.log(p/r.q[i])))}
 function optimize(cs,m){let A=Array.from({length:K+1},()=>Array(K+1).fill(0));for(let c of cs){let v=grid.map(u=>cumulative(c,u));for(let i=0;i<K;i++)for(let j=i+1;j<=K;j++)A[i][j]+=(v[j][0]-v[i][0])*(v[j][1]-v[i][1])/(grid[j]-grid[i]);}
 let dp=Array.from({length:m+1},()=>Array(K+1).fill(-Infinity)),prev=Array.from({length:m+1},()=>Array(K+1).fill(-1));dp[0][0]=0;
 for(let k=1;k<=m;k++)for(let j=k;j<=K;j++)for(let i=k-1;i<j;i++){let v=dp[k-1][i]+A[i][j];if(v>dp[k][j]){dp[k][j]=v;prev[k][j]=i;}}
 let idx=[K],j=K;for(let k=m;k>0;k--){j=prev[k][j];idx.unshift(j)}return idx.map(i=>grid[i]);}
 let cases=[];for(let wi of [0,1])for(let seed of [11,29,73]){let bank=rows.filter(x=>x.wi===wi&&x.seed===seed);for(let i=0;i<bank.length;i++)for(let j=i+1;j<bank.length;j++){cases.push({wi,seed,i,j,c:curve(bank[i],bank[j])});}}
 let audit={maxExactPartition:0,maxUniformBoundViolation:0,maxMonotonicViolation:0,maxTVChange:0,maxKLIncrease:0,maxProjectedCostIncrease:0,maxKernelError:0,maxLeafRepairError:0,maxJointOutputError:0};
 function verify(c,edges){let a=c.a,b=c.b,rec=b.p.map((v,z)=>pos(v-b.q[z])),dy=a.p.length,dx=b.q.length,m=edges.length-1;
 let out=Array.from({length:dy},()=>Array(dx).fill(0)),totalB=Array.from({length:dx},()=>Array(dx).fill(0)),joint=0,ra=0;
 for(let k=0;k<m;k++){let l=edges[k],r=edges[k+1],w=r-l,as=c.s.map((s,y)=>a.p[y]*Math.max(0,Math.min(s,r)-l)),bs=c.t.map((t,x)=>b.q[x]*Math.max(0,Math.min(t,r)-l));let col=Array(dx).fill(0);
 for(let x=0;x<dx;x++){let row=0;for(let z=0;z<dx;z++){let v=x===z?w*b.q[x]-bs[x]:c.db?bs[x]*rec[z]/c.db:0;row+=v;col[z]+=v;totalB[x][z]+=v;audit.maxKernelError=Math.max(audit.maxKernelError,-v)}audit.maxKernelError=Math.max(audit.maxKernelError,Math.abs(row-w*b.q[x]));}
 for(let y=0;y<dy;y++){let ay=as[y];ra+=ay;audit.maxKernelError=Math.max(audit.maxKernelError,-ay,ay-w*a.p[y]);for(let z=0;z<dx;z++)out[y][z]+=a.p[y]*col[z];}joint+=sum(as)*sum(bs)/w;
 }
 for(let y=0;y<dy;y++)for(let z=0;z<dx;z++)audit.maxJointOutputError=Math.max(audit.maxJointOutputError,Math.abs(out[y][z]-a.p[y]*b.p[z]));
 for(let x=0;x<dx;x++)for(let z=0;z<dx;z++){let expected=x===z?Math.min(b.p[x],b.q[x]):c.db?pos(b.q[x]-b.p[x])*rec[z]/c.db:0;audit.maxKernelError=Math.max(audit.maxKernelError,Math.abs(totalB[x][z]-expected))}
 audit.maxLeafRepairError=Math.max(audit.maxLeafRepairError,Math.abs(ra-c.da),Math.abs(c.da+c.db-joint-cost(c,edges)));
 }
 for(let r of rows){let p=proj(r);audit.maxTVChange=Math.max(audit.maxTVChange,Math.abs(tv(r.p,r.q)-tv(p.p,p.q)));audit.maxKLIncrease=Math.max(audit.maxKLIncrease,kl(p)-kl(r));}
 let results=[],calibrations=[];for(let wi of [0,1])for(let seed of [11,29,73]){let test=cases.filter(x=>x.wi===wi&&x.seed===seed),train=cases.filter(x=>x.wi!==wi||x.seed!==seed).map(x=>x.c);let configs={};for(let m of [2,4,8])configs[m]=optimize(train,m);calibrations.push({wi,seed,configs});
 for(let item of test){let c=item.c,e=exact(c),p=curve(proj(c.a),proj(c.b)),ex=exact(p),uniform={},calibrated={},oracle={};let last=cost(c,[0,1]);for(let m of [2,4,8,16]){let u=cost(c,Array.from({length:m+1},(_,i)=>i/m));uniform[m]=u;audit.maxUniformBoundViolation=Math.max(audit.maxUniformBoundViolation,u-e-1/(4*m));audit.maxMonotonicViolation=Math.max(audit.maxMonotonicViolation,u-last);last=u;if(m<=8){calibrated[m]=cost(c,configs[m]);oracle[m]=cost(c,optimize([c],m));verify(c,configs[m]);}}
 let edges=[...new Set([0,1,...c.s])].sort((a,b)=>a-b);audit.maxExactPartition=Math.max(audit.maxExactPartition,Math.abs(cost(c,edges)-e));audit.maxProjectedCostIncrease=Math.max(audit.maxProjectedCostIncrease,ex-e);
 let full=0;for(let i=0;i<c.a.p.length;i++)for(let j=0;j<c.b.p.length;j++)full+=Math.abs(c.a.p[i]*c.b.p[j]-c.a.q[i]*c.b.q[j])/2;
 results.push({wi,seed,i:item.i,j:item.j,ind:cost(c,[0,1]),exact:e,full,uniform,calibrated,oracle,projectedExact:ex,projectedFixed:cost(p,[0,.5,1]),fixed:cost(c,[0,.5,1]),exactFlags:edges.length-1});}}
 let avg=f=>sum(results.map(f))/results.length;
 return {cases:results.length,audit,means:{ind:avg(r=>r.ind),exact:avg(r=>r.exact),full:avg(r=>r.full),uniform:Object.fromEntries([2,4,8,16].map(m=>[m,avg(r=>r.uniform[m])])),calibrated:Object.fromEntries([2,4,8].map(m=>[m,avg(r=>r.calibrated[m])])),oracle:Object.fromEntries([2,4,8].map(m=>[m,avg(r=>r.oracle[m])])),projectedExact:avg(r=>r.projectedExact),projectedFixed:avg(r=>r.projectedFixed),meanFlags:avg(r=>r.exactFlags),maxFlags:Math.max(...results.map(r=>r.exactFlags)),meanKL:sum(rows.map(kl))/rows.length,meanProjectedKL:sum(rows.map(r=>kl(proj(r))))/rows.length},calibrations,results};
}
const fs=require('fs'); console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(process.argv[2],'utf8'))),null,2));
