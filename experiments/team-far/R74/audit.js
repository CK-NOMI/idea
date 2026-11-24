function audit(){
let seed=15531;const u=()=>((seed=Math.imul(seed,1664525)+1013904223>>>0)+.5)/4294967296;
const sum=a=>a.reduce((s,v)=>s+v,0),K=4,T=3,k=2;let maxErr=0,maxRaceIdentity=0,maxReachErr=0,meanRaces=0,states=0,maxConditional=0;
for(let t=0;t<120;t++){
const p=[],q=[];
for(let i=0;i<2;i++){p.push(new Map());q.push(new Map());for(let len=0;len<T;len++)for(let h=0;h<2**len;h++){let key=len+":"+h;p[i].set(key,.03+.94*u());q[i].set(key,.03+.94*u())}}
function dist(h,kind){let bank=kind==="p"?p:q, a=0,b=0;for(let x of h){a=a*2+(x>>1);b=b*2+(x&1)}let x=bank[0].get(h.length+":"+a),y=bank[1].get(h.length+":"+b);return [(1-x)*(1-y),(1-x)*y,x*(1-y),x*y]}
function walk(h,kind,w,cb){if(h.length===T){cb(h,w);return}let a=dist(h,kind);for(let x=0;x<K;x++)walk(h.concat(x),kind,w*a[x],cb)}
let target=new Map(),actual=new Map(),races=0;
walk([],"p",1,(h,w)=>target.set(h.join(","),w));
function complete(h,w){walk(h,"p",w,(y,v)=>{let key=y.join(",");actual.set(key,(actual.get(key)||0)+v)})}
for(let a=0;a<K;a++)for(let b=0;b<K;b++){
let xs=[a,b],w=[1],pr=1;
for(let j=0;j<k;j++){let p=dist(xs.slice(0,j),"p"),q=dist(xs.slice(0,j),"q");w.push(Math.min(1,w[j]*p[xs[j]]/q[xs[j]]));pr*=q[xs[j]]}
let mass=pr;complete(xs,mass*w[k]);mass*=1-w[k];
for(let j=k-1;j>=0;j--){let h=xs.slice(0,j),p=dist(h,"p"),q=dist(h,"q"),res=p.map((v,i)=>Math.max(0,w[j]*v-q[i])),S=sum(res),den=S+1-w[j];if(den===0){if(mass>1e-14)throw Error("zero reach");continue}
races+=mass/den;maxConditional=Math.max(maxConditional,1/den);
for(let y=0;y<K;y++)if(res[y]>0)complete(h.concat(y),mass*res[y]/den);
mass*= (1-w[j])/den;
}
maxReachErr=Math.max(maxReachErr,Math.abs(mass))}
for(let [key,v] of target)maxErr=Math.max(maxErr,Math.abs(v-(actual.get(key)||0)));
maxRaceIdentity=Math.max(maxRaceIdentity,Math.abs(races-k));meanRaces+=races;states+=target.size}
return {cases:120,states,maxJointError:maxErr,maxRaceIdentityError:maxRaceIdentity,maxRemainingMass:maxReachErr,meanExpectedRaces:meanRaces/120,maxConditionalRaceMean:maxConditional}}
if(require.main===module)console.log(JSON.stringify(audit(),null,2));
