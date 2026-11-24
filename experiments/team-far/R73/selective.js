function run(input,M=4096){
let rows=input.rows.filter(x=>x.K===27&&x.N===32),sum=a=>a.reduce((s,x)=>s+x,0),avg=a=>sum(a)/a.length;
function rng(seed){let x=seed>>>0;return ()=>((x=Math.imul(x,1664525)+1013904223>>>0)+.5)/4294967296}
function cdf(a){let s=0;return a.map(v=>s+=v)}
function draw(c,u){let z=u*c[c.length-1],l=0,h=c.length-1;while(l<h){let m=(l+h)>>1;if(z<c[m])h=m;else l=m+1}return l}
function ratios(ls){let L=sum(ls),logD=sum(ls.map(v=>Math.min(0,v))),O=-Math.expm1(logD),I=-Math.expm1(logD-L),t=L>=0?O:Math.exp(L)*I;return {L,O,I,t:Math.max(0,t),rank:ls.filter(x=>x>0).length}}
function policy(ls,mode){let {L,O,I,t,rank}=ratios(ls);if(!t)return 0;if(mode==="regional")return 0;if(mode==="nested")return 1;if(mode.startsWith("rank"))return +(rank===Number(mode.slice(4)));if(mode==="cheap"){let a=sum(ls.map(v=>-Math.expm1(Math.min(0,v))))/O,b=sum(ls.map(v=>-Math.expm1(Math.min(0,-v))))/I;return +(a*b<=1.35)}if(mode==="half")return .5;throw Error(mode)}
let out=[],violations=0,maxRankHops=0;
for(let wi of [0,1])for(let seed of [11,29,73])for(let N of [32,64])for(let scenario of ["homogeneous","mixed","cluster"]){
let data=rows.filter(x=>x.wi===wi&&x.seed===seed),n=N/4,leaves=Array.from({length:N},(_,i)=>{let e=i/n|0,j=i%n,{p,q}=data[scenario==="homogeneous"?e:scenario==="mixed"?(e+j)%8:e*2+j%2];return {i,log:p.map((v,k)=>Math.log(v/q[k])),res:cdf(p.map((v,k)=>Math.max(0,v-q[k]))),qc:cdf(q),delta:sum(p.map((v,k)=>Math.max(0,v-q[k])))}});
let groups=Array.from({length:4},(_,e)=>leaves.slice(e*n,(e+1)*n)),get=(g,y)=>sum(g.map(l=>l.log[y[l.i]]));
function changed(ls,rr,act){let pref=1,ws=ls.map(l=>{let a=Math.exp(Math.min(0,l)),w=pref*(1-a);pref*=a;return w});if(sum(ws)<=0)throw Error("empty conditional");let j=draw(cdf(ws),rr());for(let k=j;k<ls.length;k++)if(k===j||rr()>=Math.exp(Math.min(0,ls[k])))act(k)}
function force(g,y,rr){if(get(g,y)>=0)violations++;for(let t=0;t<=g.length;t++){let ls=g.map(l=>l.log[y[l.i]]);changed(ls,rr,j=>{let l=g[j];y[l.i]=draw(l.res,rr())});let z=ratios(g.map(l=>l.log[y[l.i]]));let stop=z.L>0?-Math.expm1(-z.L)/z.I:0;if(rr()<Math.min(1,stop))return}throw Error("cycle")}
let modes=["regional","nested","half","rank1","rank2","rank3","cheap"],rs=rng(123+seed+wi*5671+N*29+scenario.length*181),stats=Object.fromEntries(modes.map(m=>[m,{g:0,e:0,e2:0,m:0,m2:0,mx:0,tail:0,hops:0,diagGain:0,first:0,pair:0}]));
for(let t=0;t<M;t++){let x=leaves.map(l=>draw(l.qc,rs())),initial=groups.map(g=>get(g,x)),z0=ratios(initial);
for(let k=0;k<modes.length;k++){let mode=modes[k],s=stats[mode],rr=rng((t+1)*2654435761+seed*101+wi*17+N*73+k*10007+scenario.length*123),y=x.slice(),hops=0;
let alpha=policy(initial,mode),start=z0.O-alpha*z0.t;s.diagGain+=alpha*z0.t;
if(rr()<Math.max(0,start)){while(true){let ls=groups.map(g=>get(g,y));changed(ls,rr,j=>force(groups[j],y,rr));hops++;let z=ratios(groups.map(g=>get(g,y))),a=policy(groups.map(g=>get(g,y)),mode),tOverI=z.L>=0?Math.exp(-z.L)*z.t/z.I:z.t/Math.exp(z.L)/z.I;let stop=1-a*tOverI;if(rr()<Math.max(0,Math.min(1,stop)))break;if(hops>4)throw Error("root cycle")} }
if(mode.startsWith("rank"))maxRankHops=Math.max(maxRankHops,hops);
let cs=groups.map(g=>sum(g.map(l=>+(y[l.i]!==x[l.i])))),e=sum(cs.map(v=>+(v>0))),m=sum(cs);s.g+=+(m>0);s.e+=e;s.e2+=e*e;s.m+=m;s.m2+=m*m;s.mx+=Math.max(...cs);s.tail+=+(m>=8);s.hops+=hops;s.first+=+(y[0]===0);s.pair+=+(y[0]===0&&y[n]===0)
}}
for(let mode of modes){let s=stats[mode];for(let k of Object.keys(s))s[k]/=M;out.push({wi,seed,N,scenario,mode,...s})}}
let summary=[];for(let N of [32,64])for(let mode of ["regional","nested","half","rank1","rank2","rank3","cheap"]){let a=out.filter(x=>x.N===N&&x.mode===mode),b={N,mode};for(let k of ["g","e","e2","m","m2","mx","tail","hops","diagGain"])b[k]=avg(a.map(x=>x[k]));let base=out.filter(x=>x.N===N&&x.mode==="regional"),end=out.filter(x=>x.N===N&&x.mode==="nested"),mix=b.diagGain/avg(end.map(x=>x.diagGain));b.mix=mix;for(let k of ["e2","m2","mx","tail"])b["mix_"+k]=avg(base.map(x=>x[k]))*(1-mix)+avg(end.map(x=>x[k]))*mix;summary.push(b)}
return {M,draws:out.length*M,violations,maxRankHops,summary,rows:out}}
if(require.main===module){const fs=require("fs");console.log(JSON.stringify(run(JSON.parse(fs.readFileSync(process.argv[2],"utf8"))),null,2))}
