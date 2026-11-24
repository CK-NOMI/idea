function run(corpus){
const V=28,BOS=26,EOS=27,S=V*V*V, seeds=[11,29,73],sum=a=>a.reduce((s,x)=>s+x,0),avg=a=>sum(a)/a.length,idx=(a,b,c)=>(a*V+b)*V+c, names=corpus.split(/\r?\n/).filter(x=>/^[a-z]{3,18}$/.test(x)), enc=s=>[BOS,BOS,...[...s].map(c=>c.charCodeAt(0)-97),EOS];
let rows=[],rollouts=0,updates=0,checks=0,pathChecks=0,pathMismatch=0,maxMarg=0,maxState=0,countsMade=0;
for(let seed of seeds){
let state=seed;const rng=()=>((state=Math.imul(state,1664525)+1013904223>>>0)/4294967296);
let clients=Array.from({length:4},()=>[]);for(let n of names){let c=n.charCodeAt(0)-97;clients[(c>=13?2:0)+(c%2)].push(n)}
function shuffle(a){a=a.slice();for(let i=a.length-1;i>0;i--){let j=Math.floor(rng()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
clients=clients.map(shuffle);
function counts(a){let c=new Float64Array(S);for(let n of a){let h=enc(n);for(let i=2;i<h.length;i++){c[idx(h[i-2],h[i-1],h[i])]++;countsMade++}}return c}
let tr=clients.map(a=>counts(a.slice(0,1000))),ca=clients.map(a=>counts(a.slice(1000,1250))),te=clients.map(a=>counts(a.slice(1250,1500)));
function merge(a,b){return a.map((v,i)=>v+b[i])}
let train=[merge(tr[0],tr[1]),merge(tr[2],tr[3])],cal=[merge(ca[0],ca[1]),merge(ca[2],ca[3])],test=[merge(te[0],te[1]),merge(te[2],te[3])];
function sparse(a){let r=[];for(let i=0;i<a.length;i++)if(a[i])r.push([Math.floor(i/V/V),Math.floor(i/V)%V,i%V,a[i]]);return r}
let st=train.map(sparse),sc=cal.map(sparse),se=test.map(sparse);
function fit(map,dat,K){let D=K+2, a=new Float64Array(D*D*V),b=new Float64Array(D*V);for(let [x,y,z,n] of dat){a[(map[x]*D+map[y])*V+z]+=n;b[map[y]*V+z]+=n}
let q=Array.from({length:D},(_,i)=>{let r=Array.from(b.slice(i*V,(i+1)*V));r[BOS]=0;r=r.map((x,j)=>j===BOS?0:x+.5);let s=sum(r);return r.map(x=>x/s)});
let p=Array.from({length:D*D},(_,i)=>{let r=Array.from(a.slice(i*V,(i+1)*V)).map((x,j)=>x+4*q[i%D][j]);let s=sum(r);return r.map(x=>x/s)});
return {D,p,q,map,K}}
function overlaps(p,q,map,D){let tok=0,a=Array(D).fill(0),b=Array(D).fill(0);for(let i=0;i<V;i++){tok+=Math.min(p[i],q[i]);a[map[i]]+=p[i];b[map[i]]+=q[i]}return [tok,sum(a.map((v,i)=>Math.min(v,b[i])))]}
function evalModel(m,dat){let n=0,nll=0,tok=0,cls=0;let ov=m.p.map((p,i)=>overlaps(p,m.q[i%m.D],m.map,m.D));for(let [x,y,z,w] of dat){let i=m.map[x]*m.D+m.map[y];n+=w;nll-=w*Math.log(m.p[i][z]);tok+=w*ov[i][0];cls+=w*ov[i][1]}return {n,nll:nll/n,tok:tok/n,cls:cls/n}}
function clustering(K,dat){let freq=Array.from({length:26},()=>Array(V).fill(.01));for(let [x,y,z,n] of dat)if(y<26)freq[y][z]+=n;let groups=freq.map((v,i)=>({ids:[i],v}));let ent=v=>{let n=sum(v);return -sum(v.map(x=>x*Math.log(x/n)))};while(groups.length>K){let best=Infinity,bi=0,bj=1;for(let i=0;i<groups.length;i++)for(let j=i+1;j<groups.length;j++){let a=groups[i].v,b=groups[j].v,c=a.map((x,k)=>x+b[k]),d=ent(c)-ent(a)-ent(b);if(d<best){best=d;bi=i;bj=j}}let a=groups[bi],b=groups[bj];groups[bi]={ids:[...a.ids,...b.ids],v:a.v.map((x,j)=>x+b.v[j])};groups.splice(bj,1)}let m=Array(V);groups.forEach((g,i)=>g.ids.forEach(x=>m[x]=i));m[BOS]=K;m[EOS]=K+1;return m}
function randomMap(K){let m=Array(V);shuffle(Array.from({length:26},(_,i)=>i)).forEach((x,j)=>m[x]=j%K);m[BOS]=K;m[EOS]=K+1;return m}
function optimize(K,regions,beta,base){function loss(map){updates++;return avg(regions.map(e=>{let a=evalModel(fit(map,st[e],K),sc[e]);return a.nll+beta*(1-a.cls)}))}let m=base.slice(),v=loss(m);for(let j=0;j<40;j++){let a=m.slice(),i=Math.floor(rng()*26),k=Math.floor(rng()*26);if(a[i]===a[k])continue;[a[i],a[k]]=[a[k],a[i]];let q=loss(a);if(q<v){m=a;v=q}}return m}
function sample(w,u){let s=sum(w),v=u*s;for(let i=0;i<w.length;i++){v-=w[i];if(v<0)return i}return w.findLastIndex(x=>x>0)}
function pair(p,q,x,u1,u2,u3,u4,map,D,mode){let acc=Math.min(1,p[x]/q[x]);if(u1<acc)return x;let a=q.map((v,i)=>Math.max(0,v-p[i])),b=p.map((v,i)=>Math.max(0,v-q[i]));if(mode!=="class")return sample(b,u2);let A=Array(D).fill(0),B=Array(D).fill(0);for(let i=0;i<V;i++){A[map[i]]+=a[i];B[map[i]]+=b[i]}let g=map[x],m=Math.min(A[g],B[g]);let h=u2<m/A[g]?g:sample(B.map((v,i)=>Math.max(0,v-A[i])),u3);return sample(b.map((v,i)=>map[i]===h?v:0),u4)}
function audit(m){for(let s=0;s<m.p.length;s++){let p=m.p[s],q=m.q[s%m.D],map=m.map,D=m.D;let A=Array(D).fill(0),B=Array(D).fill(0),a=q.map((v,i)=>Math.max(0,v-p[i])),b=p.map((v,i)=>Math.max(0,v-q[i]));for(let i=0;i<V;i++){A[map[i]]+=a[i];B[map[i]]+=b[i]}let C=A.map((v,i)=>Math.min(v,B[i])),D0=sum(A)-sum(C),col=Array(V).fill(0),classMass=0;for(let x=0;x<V;x++){col[x]+=Math.min(p[x],q[x]);classMass+=Math.min(p[x],q[x]);if(a[x]===0)continue;let g=map[x];for(let y=0;y<V;y++){if(b[y]===0)continue;let h=map[y],v=0;if(g===h&&B[h]>0)v+=a[x]/A[g]*C[g]*b[y]/B[h];if(D0>1e-14)v+=a[x]/A[g]*(A[g]-C[g])*(B[h]-C[h])/D0*b[y]/B[h];col[y]+=v;if(g===h)classMass+=v}}
maxMarg=Math.max(maxMarg,...col.map((v,i)=>Math.abs(v-p[i])));maxState=Math.max(maxState,Math.abs(classMass-overlaps(p,q,map,D)[1]));checks++}}
function simulate(m,K,mode,tapes,bonusOn=true){let count=0,round=0,rows=0,drafts=0,salvage=0,nonid=0,repairs=0,rejoin=0,hadRepair=false,h=[m.K,m.K],out=[];
while(count<tapes.length){let path=[],st=h.slice(),cap=Math.min(K,tapes.length-count);for(let j=0;j<cap;j++){let t=tapes[count+j],q=m.q[st[1]],x=sample(q,t[0]);path.push({st:st.slice(),x});st=[st[1],m.map[x]];if(x===EOS)break}
round++;rows+=path.length;drafts+=path.length;let bonus=bonusOn&&path.at(-1).x!==EOS&&count+path.length<tapes.length;if(bonus)rows++;
let ok=true,hadMismatch=false;hadRepair=false;
for(let a of path){let t=tapes[count],cached=h[0]===a.st[0]&&h[1]===a.st[1],x=a.x;
if(mode==="repair"&&!cached){x=sample(m.q[h[1]],t[0]);rows++;round++;drafts++;repairs++;hadRepair=true}
else if(!cached)throw Error("unsafe reuse");
else if(hadRepair){rejoin++;hadRepair=false}
let p=m.p[h[0]*m.D+h[1]],q=m.q[h[1]],y=pair(p,q,x,...t.slice(1,5),m.map,m.D,mode==="repair"?"class":mode);if(hadMismatch&&cached)salvage++;out.push(y);count++;if(y!==x)nonid++;let same=m.map[y]===m.map[x];h=[h[1],m.map[y]];if(y===EOS)return {count,round,rows,drafts,salvage,nonid,out,repairs,rejoin};if(y!==x)hadMismatch=true;if(mode!=="repair"&&(mode==="standard"?y!==x:!same)){ok=false;break}}
if(ok&&bonus&&count<tapes.length){if(mode==="repair"&&(h[0]!==st[0]||h[1]!==st[1])){round++;rows++;repairs++}
let y=sample(m.p[h[0]*m.D+h[1]],tapes[count][5]);out.push(y);count++;if(y===EOS)break;h=[h[1],m.map[y]]}
}
return {count,round,rows,drafts,salvage,nonid,out,repairs,rejoin}}
function reference(m,tapes){let h=[m.K,m.K],out=[];for(let t of tapes){let p=m.p[h[0]*m.D+h[1]],q=m.q[h[1]],x=sample(q,t[0]),y=pair(p,q,x,...t.slice(1,5),m.map,m.D,"class");out.push(y);if(y===EOS)break;h=[h[1],m.map[y]]}return out}

for(let K of [4,8,16,26]){
let g=K===26?Array.from({length:V},(_,i)=>i):clustering(K,[...st[0],...st[1]]);
let configs=K===26?[{kind:"identity",maps:[g,g]}]:[{kind:"random",maps:[randomMap(K),randomMap(K)]},{kind:"globalCE",maps:[g,g]},{kind:"regionalCE",maps:[0,1].map(e=>optimize(K,[e],0,clustering(K,st[e])))},{kind:"regionalJoint",maps:[0,1].map(e=>optimize(K,[e],1,clustering(K,st[e])))}];
for(let c of configs)for(let e=0;e<2;e++){let m=fit(c.maps[e],st[e],K),v=evalModel(m,se[e]);audit(m);for(let block of [4,8]){let arms={};let tapes=Array.from({length:192},()=>Array.from({length:24},()=>Array.from({length:6},rng)));for(let mode of ["standard","reuse","class","repair"]){let n=0,r=0,rr=0,d=0,s=0,mis=0,repair=0,rejoin=0;for(let t of tapes){let a=simulate(m,block,mode,t);rollouts++;n+=a.count;r+=a.round;rr+=a.rows;d+=a.drafts;s+=a.salvage;mis+=a.nonid;repair+=a.repairs;rejoin+=a.rejoin}arms[mode]={tokens:n,rounds:r,rows:rr,drafts:d,salvage:s,mismatch:mis,repair,rejoin,roundPerToken:r/n,rowsPerToken:rr/n}}
for(let t of tapes.slice(0,24)){let a=simulate(m,block,"repair",t,false).out,b=reference(m,t);pathChecks++;if(a.join(",")!==b.join(","))pathMismatch++}rows.push({seed,K,kind:c.kind,e,block,map:c.maps[e],...v,arms})}}
}
}
let summary=[];for(let K of [4,8,16,26])for(let kind of ["random","globalCE","regionalCE","regionalJoint","identity"]){let a=rows.filter(x=>x.K===K&&x.kind===kind&&x.block===4);if(!a.length)continue;summary.push({K,kind,n:a.length,nll:avg(a.map(x=>x.nll)),tok:avg(a.map(x=>x.tok)),cls:avg(a.map(x=>x.cls)),standard:avg(a.map(x=>x.arms.standard.roundPerToken)),reuse:avg(a.map(x=>x.arms.reuse.roundPerToken)),coupled:avg(a.map(x=>x.arms.class.roundPerToken)),rowsStd:avg(a.map(x=>x.arms.standard.rowsPerToken)),rowsClass:avg(a.map(x=>x.arms.class.rowsPerToken)),salvage:sum(a.map(x=>x.arms.class.salvage)),repair:avg(a.map(x=>x.arms.repair.roundPerToken)),repairRows:avg(a.map(x=>x.arms.repair.rowsPerToken)),rejoins:sum(a.map(x=>x.arms.repair.rejoin)),repairs:sum(a.map(x=>x.arms.repair.repair)),gain:1-avg(a.map(x=>x.arms.class.roundPerToken/x.arms.standard.roundPerToken))})}
return {seeds,names:names.length,countsMade,rollouts,updates,checks,pathChecks,pathMismatch,maxMarg,maxState,summary,rows};
}
if(typeof module!=="undefined"&&require.main===module){const fs=require("fs");console.log(JSON.stringify(run(fs.readFileSync(process.argv[2],"utf8")),null,2))}
