# R10 actual source and raw results

Run `node probe.js weights.json checkpoint.json input.txt` (extract JS below). V8 CPU actual execution; no timing, GPU, training, or physical HFL. Source adapted from FEIA R11 as noted. All per-cohort conditional moments and block lengths retained.

~~~javascript
// Forward/rollout adapted from team-hieracert/R11/probe.js at 2507648eb168131a862a887b1c834d5511a3c223; new gradient-stratification audit.
function probe(raws,names){
const V=27,d=16,BOS=26,N=8,sum=a=>a.reduce((s,x)=>s+x,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),mat=(a,x)=>a.map(r=>dot(r,x)),add=(a,b)=>a.map((x,i)=>x+b[i]),rms=x=>{let s=Math.sqrt(dot(x,x)/d+1e-5);return x.map(t=>t/s)},sm=x=>{let m=Math.max(...x),a=x.map(v=>Math.exp(v-m)),s=sum(a);return a.map(v=>v/s)},arg=x=>x.indexOf(Math.max(...x)),sq=x=>dot(x,x),zero=()=>Array(V*d).fill(0);
let st=0;const U=()=>{st=(Math.imul(st,1664525)+1013904223)>>>0;return(st+.5)/4294967296},draw=p=>{let u=U();for(let i=0;i<p.length;i++){u-=p[i];if(u<0)return i}return p.length-1};
const all=[...new Set(names.split(/\r?\n/).map(x=>x.trim()).filter(s=>/^[a-z]{10,14}$/.test(s)))];const clients=Array.from({length:4},(_,c)=>all.filter(x=>(x.charCodeAt(0)-97)%4===c));const rows=[];
for(let k=0;k<raws.length;k++){const raw=raws[k],vals=Array.from({length:7264},(_,i)=>raw[String(i)]);if(vals.some(x=>!Number.isFinite(x)))throw Error("schema");let off=0;const take=(n,m)=>Array.from({length:n},()=>{let r=vals.slice(off,off+m);off+=m;return r});const E=take(V,d),P=take(16,d),W=take(V,d),blocks=Array.from({length:2},()=>({q:take(d,d),k:take(d,d),v:take(d,d),o:take(d,d),f1:take(4*d,d),f2:take(d,4*d)}));if(off!==7264)throw Error("shape");
function features(h,depth=2){let seq=h.map((t,i)=>rms(add(E[t],P[i]))),first=null;for(let l=0;l<depth;l++){let b=blocks[l],z=seq.map(rms),qs=z.map(x=>mat(b.q,x)),ks=z.map(x=>mat(b.k,x)),vs=z.map(x=>mat(b.v,x));seq=seq.map((x,i)=>{let a=[];for(let head=0;head<4;head++){let j=head*4,p=sm(ks.slice(0,i+1).map(k=>dot(qs[i].slice(j,j+4),k.slice(j,j+4))/2));for(let k=j;k<j+4;k++)a.push(sum(p.map((w,t)=>w*vs[t][k])))}let y=add(x,mat(b.o,a));return add(y,mat(b.f2,mat(b.f1,rms(y)).map(v=>Math.max(0,v))))});if(l===0)first=seq.map(x=>x.slice());}return {shallow:first,deep:seq}}
function rollout(h,cap,greedy=false){let full=h.slice(),out=[],groups=[],accepted=0,attempted=0,proposed=0;while(out.length<N){let group=[],tmp=full.slice(),xs=[],qs=[],n=Math.min(cap,N-out.length);
for(let j=0;j<n;j++){let f=features(tmp,1),q=sm(mat(W,f.shallow.at(-1))),x=greedy?arg(q):draw(q);xs.push(x);qs.push(q);tmp.push(x);proposed++}
const f=features(tmp),base=full.length-1;let ok=true;
function commit(x,index){let z=f.deep[index],p=sm(mat(W,z)),g=p.flatMap((v,i)=>z.map(a=>(v-Number(i===x))*a));group.push(g);out.push(x);full.push(x)}
for(let j=0;j<n;j++){let p=sm(mat(W,f.deep[base+j])),q=qs[j],x=xs[j];attempted++;if(greedy?x===arg(p):U()<Math.min(1,p[x]/q[x])){accepted++;commit(x,base+j)}else{let r=p.map((v,i)=>Math.max(0,v-q[i])),z=sum(r);if(!greedy&&z<=0)throw Error("empty residual");commit(greedy?arg(p):draw(r.map(v=>v/z)),base+j);ok=false;break}}
if(ok&&out.length<N){let p=sm(mat(W,f.deep[base+n]));commit(greedy?arg(p):draw(p),base+n)}groups.push(group)}
return {out,groups,accepted,attempted,proposed}}
function stats(groups){let a=groups.flat(),n=a.length,m=groups.length,mean=zero(),start=zero(),weighted=zero();for(let g of a)for(let j=0;j<g.length;j++)mean[j]+=g[j]/n;
for(let b of groups)for(let j=0;j<mean.length;j++){start[j]+=b[0][j]/m;weighted[j]+=b.length*b[0][j]/n}
const strat=gs=>sum(gs.map(b=>{let u=zero();for(let g of b)for(let j=0;j<u.length;j++)u[j]+=g[j]/b.length;return b.length*sum(b.map(g=>sq(g.map((v,j)=>v-u[j]))))/n**2}));
let equal=Array.from({length:m},(_,i)=>a.slice(Math.floor(i*n/m),Math.floor((i+1)*n/m)));let pop=sum(a.map(g=>sq(g.map((v,j)=>v-mean[j]))))/n;
return {n,m,gnorm:sq(mean),startErr:sq(start.map((v,j)=>v-mean[j])),weightedErr:sq(weighted.map((v,j)=>v-mean[j])),strat:strat(groups),equal:strat(equal),srs:(1/m-1/n)*n/(n-1)*pop}}
for(let seed of [11,29,73])for(let cap of [1,2,4])for(let c=0;c<4;c++){st=seed+1009*c;let cases=[];for(let t=0;t<4;t++){let name=clients[c][(seed*17+t*13)%clients[c].length],h=[BOS,...name.slice(0,2).split("").map(x=>x.charCodeAt(0)-97)],gold=name.slice(2,10).split("").map(x=>x.charCodeAt(0)-97);let r=Array.from({length:3},()=>rollout(h,cap));let rewards=r.map(x=>sum(x.out.map((v,i)=>Number(v===gold[i])))/N),avg=sum(rewards)/3;
let groups=r.flatMap((x,i)=>x.groups.map(b=>b.map(g=>g.map(v=>(rewards[i]-avg)*v))));let s=stats(groups);cases.push({...s,rewards,accepted:sum(r.map(x=>x.accepted)),attempted:sum(r.map(x=>x.attempted)),proposed:sum(r.map(x=>x.proposed)),lengths:groups.map(g=>g.length)})}
rows.push({checkpoint:k,seed,cap,client:c,cases})}}
return {clientSizes:clients.map(x=>x.length),rows}}
const fs=require('node:fs'); console.log(JSON.stringify(probe([JSON.parse(fs.readFileSync(process.argv[2])),JSON.parse(fs.readFileSync(process.argv[3])).weights],fs.readFileSync(process.argv[4],'utf8')),null,2));
~~~

~~~json
{
  "source": [
    {
      "path": "weights.json",
      "sha": "df323b7e714ceeb10ca070f9945a82c51b2c233c",
      "repo": "sanchayan7432/microGPT",
      "ref": "aa28d7c0aa66704ce56b263dfd3270579f3d5f6d"
    },
    {
      "path": "checkpoint.json",
      "sha": "7f0b4968c56290ebad706ab740de942c369f42a5",
      "repo": "sanchayan7432/microGPT",
      "ref": "aa28d7c0aa66704ce56b263dfd3270579f3d5f6d"
    },
    {
      "path": "input.txt",
      "sha": "736d88e18b70b292750a30e5cca6a92b0b333bd2",
      "repo": "sanchayan7432/microGPT",
      "ref": "aa28d7c0aa66704ce56b263dfd3270579f3d5f6d"
    }
  ],
  "summary": [
    {
      "k": 0,
      "cap": 1,
      "cases": 48,
      "n": 1152,
      "m": 630,
      "gnorm": 0.1790353484404994,
      "startErr": 0.11904173497358528,
      "weightedErr": 0.13127428276262187,
      "strat": 0.12292866553234642,
      "equal": 0.12902205804054423,
      "srs": 0.122063430389829,
      "accepted": 563,
      "attempted": 630,
      "proposed": 630,
      "relStart": 0.815417881599057,
      "relWeighted": 0.8562891362413486,
      "stratRatio": 1.007088405919399,
      "equalRatio": 1.0570082917421848
    },
    {
      "k": 0,
      "cap": 2,
      "cases": 48,
      "n": 1152,
      "m": 466,
      "gnorm": 0.15489108046040168,
      "startErr": 0.20916156723272164,
      "weightedErr": 0.2519951542413147,
      "strat": 0.23445612015115874,
      "equal": 0.21473743509324597,
      "srs": 0.2066279093815652,
      "accepted": 789,
      "attempted": 873,
      "proposed": 908,
      "relStart": 1.1620578238480568,
      "relWeighted": 1.2755071286158712,
      "stratRatio": 1.134677889607861,
      "equalRatio": 1.0392470007365049
    },
    {
      "k": 0,
      "cap": 4,
      "cases": 48,
      "n": 1152,
      "m": 348,
      "gnorm": 0.11297052124996504,
      "startErr": 0.2574925191358628,
      "weightedErr": 0.34987264113945593,
      "strat": 0.3187131359051867,
      "equal": 0.25598154985904026,
      "srs": 0.2538701197624504,
      "accepted": 931,
      "attempted": 1025,
      "proposed": 1170,
      "relStart": 1.5097315876434476,
      "relWeighted": 1.7598369232439932,
      "stratRatio": 1.2554180704819093,
      "equalRatio": 1.0083169697109906
    },
    {
      "k": 1,
      "cap": 1,
      "cases": 48,
      "n": 1152,
      "m": 617,
      "gnorm": 0.13173831359601682,
      "startErr": 0.11365783199477081,
      "weightedErr": 0.12373191032119867,
      "strat": 0.11768724053728247,
      "equal": 0.11522118966435606,
      "srs": 0.11054350512769909,
      "accepted": 570,
      "attempted": 617,
      "proposed": 617,
      "relStart": 0.9288458263933722,
      "relWeighted": 0.9691361848806642,
      "stratRatio": 1.0646237461109178,
      "equalRatio": 1.0423153267236582
    },
    {
      "k": 1,
      "cap": 2,
      "cases": 48,
      "n": 1152,
      "m": 470,
      "gnorm": 0.1920164942800807,
      "startErr": 0.2636986687897195,
      "weightedErr": 0.28509108861768456,
      "strat": 0.28199259170327906,
      "equal": 0.264128783354304,
      "srs": 0.25202332769229885,
      "accepted": 797,
      "attempted": 868,
      "proposed": 908,
      "relStart": 1.1718842043799567,
      "relWeighted": 1.2184916375706953,
      "stratRatio": 1.1189146428840524,
      "equalRatio": 1.0480330760364571
    },
    {
      "k": 1,
      "cap": 4,
      "cases": 48,
      "n": 1152,
      "m": 326,
      "gnorm": 0.15368768693294413,
      "startErr": 0.3890332407512873,
      "weightedErr": 0.4786529311364374,
      "strat": 0.4470215674165474,
      "equal": 0.3646297920525382,
      "srs": 0.36202818526317965,
      "accepted": 952,
      "attempted": 1024,
      "proposed": 1121,
      "relStart": 1.5910133300624056,
      "relWeighted": 1.76478101950387,
      "stratRatio": 1.2347700693292183,
      "equalRatio": 1.0071861995702553
    }
  ],
  "clientSizes": [
    235,
    79,
    179,
    54
  ],
  "rows": [
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 1,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.001226894212769273,
          "startErr": 0.0025624989409178626,
          "weightedErr": 0.002924859458610508,
          "strat": 0.0018711189532351709,
          "equal": 0.0015721683516910297,
          "srs": 0.0015762414175905194,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 11,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.008349805170448707,
          "startErr": 0.006998639354162745,
          "weightedErr": 0.006998639354162745,
          "strat": 0.006168050086749713,
          "equal": 0.006168050086749713,
          "srs": 0.005952890141406749,
          "rewards": [
            0.125,
            0,
            0.25
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0013371217927371313,
          "startErr": 0.0018810171977120447,
          "weightedErr": 0.0018810171977120447,
          "strat": 0.0016158104768447376,
          "equal": 0.0016158104768447376,
          "srs": 0.0015717738999731171,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0019348450953345234,
          "startErr": 0.0014054767278699047,
          "weightedErr": 0.0014054767278699047,
          "strat": 0.0014598397283263158,
          "equal": 0.0014598397283263158,
          "srs": 0.0016426570689220855,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 1,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.003428357362328815,
          "startErr": 0.0018384983919821358,
          "weightedErr": 0.0019519416261927617,
          "strat": 0.0019376994831818955,
          "equal": 0.002097440044506493,
          "srs": 0.0018189831925768951,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.002727594337827089,
          "startErr": 0.0020537783640324665,
          "weightedErr": 0.002647527125543586,
          "strat": 0.002108637411091275,
          "equal": 0.00214656737165138,
          "srs": 0.0019800220790420243,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 1,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0014007170452830658,
          "startErr": 0.0010784965627943198,
          "weightedErr": 0.001329112875878881,
          "strat": 0.001127893622201296,
          "equal": 0.0014687084619779793,
          "srs": 0.0012585837932202152,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.004453821634337205,
          "startErr": 0.0059181269694660275,
          "weightedErr": 0.007945614317552713,
          "strat": 0.007303981508687509,
          "equal": 0.006461546910114475,
          "srs": 0.006530371768560659,
          "rewards": [
            0.25,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 1,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0064363857388394065,
          "startErr": 0.0037806475232514585,
          "weightedErr": 0.004221386878357015,
          "strat": 0.005733872036493755,
          "equal": 0.004920016787028079,
          "srs": 0.004944751451870943,
          "rewards": [
            0,
            0.25,
            0.125
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.06663591903573841,
          "startErr": 0.01605090689891852,
          "weightedErr": 0.01845288702796144,
          "strat": 0.023855512709709682,
          "equal": 0.027842280861945626,
          "srs": 0.029712926485881273,
          "rewards": [
            0.5,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.001660633942876096,
          "startErr": 0.001866697542568727,
          "weightedErr": 0.0021290987346030362,
          "strat": 0.001837677520607525,
          "equal": 0.0016998118574717547,
          "srs": 0.0016439609454097827,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0025916793648344816,
          "startErr": 0.0019914561230462953,
          "weightedErr": 0.0019914561230462953,
          "strat": 0.00237594321060733,
          "equal": 0.00237594321060733,
          "srs": 0.002578985800158436,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 2,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.006898460709297588,
          "startErr": 0.015209590467742406,
          "weightedErr": 0.017822947631702616,
          "strat": 0.014000644124000471,
          "equal": 0.014501216625419092,
          "srs": 0.01261291607698996,
          "rewards": [
            0,
            0.125,
            0.25
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0011797360513748246,
          "startErr": 0.0038721077568936104,
          "weightedErr": 0.0037464059554392624,
          "strat": 0.002957363769085898,
          "equal": 0.0031265752959644634,
          "srs": 0.0027808168546839965,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            2,
            3,
            2,
            3,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0018681504840817335,
          "startErr": 0.0030505665308811587,
          "weightedErr": 0.004269407243031912,
          "strat": 0.0029414343740895446,
          "equal": 0.0030150806168831733,
          "srs": 0.002459120096236333,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            3,
            1,
            1,
            3,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.001878296443437868,
          "startErr": 0.0023789093043618836,
          "weightedErr": 0.0023874383544136087,
          "strat": 0.0028309240518085456,
          "equal": 0.0026383096204743457,
          "srs": 0.0025626687998457076,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            1,
            1,
            3,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 2,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0015207981693377611,
          "startErr": 0.0030797344470556457,
          "weightedErr": 0.0029881174945594756,
          "strat": 0.002725528605077252,
          "equal": 0.0027191441811281534,
          "srs": 0.0024217157573713437,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            1,
            1,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            2,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.001200548096936668,
          "startErr": 0.0020779889205676818,
          "weightedErr": 0.002547224357272922,
          "strat": 0.0029234666362888875,
          "equal": 0.002878002366756258,
          "srs": 0.002716058362711267,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 2,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0026840007153628408,
          "startErr": 0.0033651020815235787,
          "weightedErr": 0.004041816079665905,
          "strat": 0.003694806686987143,
          "equal": 0.003744993058244838,
          "srs": 0.0033291032666110986,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0014954210702725385,
          "startErr": 0.0038989966978535124,
          "weightedErr": 0.0039128455093766735,
          "strat": 0.0034270316163856947,
          "equal": 0.0030856050273743423,
          "srs": 0.002696586725108792,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 15,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            1,
            3,
            1,
            3,
            3,
            2,
            2,
            3,
            3
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.0019030639688697873,
          "startErr": 0.002717752673994809,
          "weightedErr": 0.003167251151655394,
          "strat": 0.0028293368001486656,
          "equal": 0.0025898062024779607,
          "srs": 0.0022877001525901837,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 15,
          "attempted": 19,
          "proposed": 21,
          "lengths": [
            3,
            2,
            3,
            2,
            1,
            3,
            2,
            3,
            3,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0030391662568710473,
          "startErr": 0.0032289660327084043,
          "weightedErr": 0.0034108994714183,
          "strat": 0.004654550823002973,
          "equal": 0.0028473343461242985,
          "srs": 0.003742085493047382,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            1,
            3,
            3,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 2,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0013205845390995854,
          "startErr": 0.0023672142971442245,
          "weightedErr": 0.0034464769329309703,
          "strat": 0.003312820328239265,
          "equal": 0.0021412090615745994,
          "srs": 0.0020413518865983206,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 15,
          "attempted": 19,
          "proposed": 22,
          "lengths": [
            1,
            1,
            3,
            2,
            1,
            3,
            3,
            2,
            1,
            3,
            3,
            1
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0016517362761689935,
          "startErr": 0.0030256569837241593,
          "weightedErr": 0.004247893723170851,
          "strat": 0.0031741656031760352,
          "equal": 0.003230248371504506,
          "srs": 0.0026011659236067105,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            1,
            1,
            3,
            2,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0020905573494009256,
          "startErr": 0.002526551492795351,
          "weightedErr": 0.003721602504812221,
          "strat": 0.003672850867654539,
          "equal": 0.003102268680478867,
          "srs": 0.0028299202149481184,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 4,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0035871255154159404,
          "startErr": 0.007661802481918371,
          "weightedErr": 0.009171539511500622,
          "strat": 0.0063540184446305055,
          "equal": 0.004711655980177429,
          "srs": 0.004860305569636601,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            3,
            2,
            5,
            1
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0027222412881247712,
          "startErr": 0.0036947962147258926,
          "weightedErr": 0.004404145257101731,
          "strat": 0.004347336890375061,
          "equal": 0.0032262796371675725,
          "srs": 0.0027642958535776647,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 17,
          "attempted": 22,
          "proposed": 28,
          "lengths": [
            5,
            2,
            1,
            5,
            2,
            1,
            3,
            1,
            4
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.001813418494863524,
          "startErr": 0.00336177384320573,
          "weightedErr": 0.003363664036698635,
          "strat": 0.0057333579544363664,
          "equal": 0.0041460984993593394,
          "srs": 0.00400001386484947,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            3,
            2,
            5,
            1,
            5,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0023723990380887443,
          "startErr": 0.005499451273123009,
          "weightedErr": 0.007549601529129455,
          "strat": 0.006255616331360605,
          "equal": 0.004311009840382186,
          "srs": 0.004615788763855807,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 26,
          "lengths": [
            5,
            3,
            3,
            1,
            4,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 4,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 6,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0013568629785372915,
          "startErr": 0.005033740095877614,
          "weightedErr": 0.005830597193642648,
          "strat": 0.004623652720205657,
          "equal": 0.003877481989712493,
          "srs": 0.00398359929500382,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0017511270641274355,
          "startErr": 0.00690703612953314,
          "weightedErr": 0.008441081497399016,
          "strat": 0.005937782485465362,
          "equal": 0.0050038045371751634,
          "srs": 0.004853117957917325,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 24,
          "lengths": [
            4,
            4,
            5,
            1,
            2,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0032350963643044636,
          "startErr": 0.003022494179735122,
          "weightedErr": 0.006302142469637611,
          "strat": 0.004009021380699663,
          "equal": 0.0025321765829340783,
          "srs": 0.0025254169967344766,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 17,
          "attempted": 22,
          "proposed": 34,
          "lengths": [
            1,
            5,
            2,
            2,
            5,
            1,
            3,
            1,
            1,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 4,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.00751565630375055,
          "startErr": 0.01083722544287456,
          "weightedErr": 0.01542988063720857,
          "strat": 0.017153758843138082,
          "equal": 0.012457178723543974,
          "srs": 0.013132351923325177,
          "rewards": [
            0,
            0.125,
            0.25
          ],
          "accepted": 19,
          "attempted": 22,
          "proposed": 25,
          "lengths": [
            3,
            5,
            5,
            1,
            2,
            4,
            4
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.001823090458841466,
          "startErr": 0.0024482110096816275,
          "weightedErr": 0.0025301992522618858,
          "strat": 0.0039306894391061276,
          "equal": 0.0036814817874044584,
          "srs": 0.00362325881718179,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 22,
          "proposed": 26,
          "lengths": [
            2,
            5,
            1,
            3,
            3,
            2,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0015247416791866386,
          "startErr": 0.005645641132741271,
          "weightedErr": 0.009492653397745264,
          "strat": 0.0067080322490336115,
          "equal": 0.00597242739631277,
          "srs": 0.005719732906975221,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            2,
            5,
            1,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0007952916780286911,
          "startErr": 0.003037962182503756,
          "weightedErr": 0.0057068482720816485,
          "strat": 0.004417472933166539,
          "equal": 0.0030795288359912834,
          "srs": 0.003039046266075539,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 18,
          "attempted": 22,
          "proposed": 32,
          "lengths": [
            5,
            3,
            1,
            5,
            2,
            1,
            1,
            3,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 11,
      "cap": 4,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.006842168270530559,
          "startErr": 0.011876444570374248,
          "weightedErr": 0.020255944556412976,
          "strat": 0.01697033343234537,
          "equal": 0.013237031068461691,
          "srs": 0.013115752692776053,
          "rewards": [
            0.25,
            0,
            0.125
          ],
          "accepted": 19,
          "attempted": 22,
          "proposed": 25,
          "lengths": [
            5,
            3,
            5,
            3,
            1,
            4,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0011285516059707204,
          "startErr": 0.007767222202415346,
          "weightedErr": 0.009079122443868012,
          "strat": 0.004590810542589228,
          "equal": 0.0045524322894782985,
          "srs": 0.004167566359616382,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            2,
            5,
            1,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.015478352657140801,
          "startErr": 0.034847918664015765,
          "weightedErr": 0.04506849579917781,
          "strat": 0.03734333913312052,
          "equal": 0.029344769172941338,
          "srs": 0.028763869404031805,
          "rewards": [
            0,
            0.375,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            2,
            5,
            1,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0013948060927752477,
          "startErr": 0.01048638467917947,
          "weightedErr": 0.011295012578527304,
          "strat": 0.006715159887233875,
          "equal": 0.005871932283838194,
          "srs": 0.005720621534813618,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            3,
            5,
            5,
            3,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 1,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.004970995460413486,
          "startErr": 0.005622054690028957,
          "weightedErr": 0.006421751141381006,
          "strat": 0.006099634679276437,
          "equal": 0.005800472643442695,
          "srs": 0.004931302635478156,
          "rewards": [
            0.125,
            0.25,
            0
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0017339509073250373,
          "startErr": 0.0008324129571781803,
          "weightedErr": 0.0010052645104439589,
          "strat": 0.0016425066285842274,
          "equal": 0.0015462519847796842,
          "srs": 0.001346935772621341,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 11,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.00694168749287751,
          "startErr": 0.002929888473130486,
          "weightedErr": 0.004983009904850019,
          "strat": 0.005890731132096591,
          "equal": 0.00580315870689822,
          "srs": 0.00450296957309941,
          "rewards": [
            0,
            0.125,
            0.25
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0015783423324020928,
          "startErr": 0.0020395510923447747,
          "weightedErr": 0.0021130641619721397,
          "strat": 0.0013988565239262058,
          "equal": 0.0015210407102300007,
          "srs": 0.001472444671904897,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 1,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.006809534821509042,
          "startErr": 0.004442080978771461,
          "weightedErr": 0.00435625572928884,
          "strat": 0.004750139886701294,
          "equal": 0.0054243590124250654,
          "srs": 0.004150101148248771,
          "rewards": [
            0,
            0,
            0.25
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0076245214319821475,
          "startErr": 0.008205987079065018,
          "weightedErr": 0.00885229872045969,
          "strat": 0.005370670251745838,
          "equal": 0.006514152775201292,
          "srs": 0.006049169182338259,
          "rewards": [
            0.25,
            0,
            0.25
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.00255792563549522,
          "startErr": 0.005461294413583903,
          "weightedErr": 0.005461294413583903,
          "strat": 0.0017276910865858747,
          "equal": 0.0017276910865858747,
          "srs": 0.0015945940908398102,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 1,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0019672891075313325,
          "startErr": 0.0016511405882654968,
          "weightedErr": 0.0016511405882654968,
          "strat": 0.0016657143368988296,
          "equal": 0.0016657143368988296,
          "srs": 0.0017637597599860133,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0028252401310944492,
          "startErr": 0.0011646060798286566,
          "weightedErr": 0.0012626044508865255,
          "strat": 0.0017772643292449674,
          "equal": 0.0017595602893248497,
          "srs": 0.0015867067561162176,
          "rewards": [
            0.125,
            0.125,
            0.25
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 1,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.001888498994522228,
          "startErr": 0.001571899719342244,
          "weightedErr": 0.0018769421063029835,
          "strat": 0.0020383783605477934,
          "equal": 0.001994656740849151,
          "srs": 0.0016543856158447048,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 15,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 11,
          "attempted": 15,
          "proposed": 15,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.002308154579138603,
          "startErr": 0.001536749685546642,
          "weightedErr": 0.001810534571493071,
          "strat": 0.001568830210196594,
          "equal": 0.0015184610957794094,
          "srs": 0.001442756907894269,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 11,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 2,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0020596521109620698,
          "startErr": 0.002778877784405569,
          "weightedErr": 0.003214303689124563,
          "strat": 0.003174079496177693,
          "equal": 0.0029480349874270706,
          "srs": 0.00283070706557074,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            2,
            3
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.0017058562335850239,
          "startErr": 0.002645606583531579,
          "weightedErr": 0.0038291817289200464,
          "strat": 0.003182314026356684,
          "equal": 0.0027056125730547534,
          "srs": 0.002572136355871639,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 15,
          "attempted": 19,
          "proposed": 21,
          "lengths": [
            3,
            3,
            2,
            2,
            1,
            3,
            2,
            3,
            3,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.0013298714935000054,
          "startErr": 0.002586871497508425,
          "weightedErr": 0.002946312521644011,
          "strat": 0.002846597854245295,
          "equal": 0.0020350023104360455,
          "srs": 0.0024138902122361186,
          "rewards": [
            0.25,
            0.25,
            0.375
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            3,
            3,
            1,
            1,
            3,
            3,
            2,
            3,
            3,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0025043167154306075,
          "startErr": 0.0028034578270939485,
          "weightedErr": 0.002708965165846112,
          "strat": 0.0030983605312208517,
          "equal": 0.003050547859257232,
          "srs": 0.002795087597256503,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            2,
            3,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 2,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0021201073410621867,
          "startErr": 0.0021914198380992354,
          "weightedErr": 0.0024648874814263205,
          "strat": 0.0031271128169159277,
          "equal": 0.002875780746714958,
          "srs": 0.002859549129429052,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            3,
            3,
            1,
            1,
            1,
            3,
            3,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            2,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            2,
            3,
            3,
            3,
            3,
            2,
            1,
            3,
            3,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 2,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            2,
            3,
            3,
            3,
            2,
            3,
            2,
            3
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.001902660344881687,
          "startErr": 0.00289756733452259,
          "weightedErr": 0.003588167084876838,
          "strat": 0.002936128975187399,
          "equal": 0.0024514672816363806,
          "srs": 0.0024812091552310588,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 19,
          "proposed": 22,
          "lengths": [
            1,
            1,
            1,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.02024207236512729,
          "startErr": 0.019779494407268658,
          "weightedErr": 0.026346234989377992,
          "strat": 0.023289402836334572,
          "equal": 0.020917723775096823,
          "srs": 0.02105255717304479,
          "rewards": [
            0.125,
            0,
            0.375
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 2,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0014666260250099147,
          "startErr": 0.0038022568689141186,
          "weightedErr": 0.004575899278768018,
          "strat": 0.0037142439375360977,
          "equal": 0.003423560071214581,
          "srs": 0.003271839120449981,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.008549452136628425,
          "startErr": 0.00812457853989052,
          "weightedErr": 0.01010853644582095,
          "strat": 0.008633008261378678,
          "equal": 0.006568225198885494,
          "srs": 0.00805535568676292,
          "rewards": [
            0,
            0.25,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            1,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0011847580059447759,
          "startErr": 0.0021856911569563403,
          "weightedErr": 0.0022731060169732357,
          "strat": 0.0020553507629693948,
          "equal": 0.0024524869108491407,
          "srs": 0.002147812105105903,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 14,
          "attempted": 19,
          "proposed": 20,
          "lengths": [
            3,
            2,
            3,
            3,
            2,
            3,
            2,
            3,
            1,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 4,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0020619852702854395,
          "startErr": 0.005321978358676842,
          "weightedErr": 0.006513344998305795,
          "strat": 0.005064755276293985,
          "equal": 0.004568475404575798,
          "srs": 0.004451283604565659,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            2,
            5,
            1,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.002237843228854164,
          "startErr": 0.005654725835968203,
          "weightedErr": 0.007582606947882731,
          "strat": 0.006682512741104315,
          "equal": 0.005147904823427061,
          "srs": 0.004895699606418283,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            1,
            2
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.0020062966417016344,
          "startErr": 0.0024212706908598013,
          "weightedErr": 0.005612697298203461,
          "strat": 0.005564897367816523,
          "equal": 0.0035058601389113295,
          "srs": 0.003702973158656315,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 18,
          "attempted": 21,
          "proposed": 29,
          "lengths": [
            1,
            5,
            2,
            5,
            3,
            2,
            1,
            5
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.002369391980248043,
          "startErr": 0.006694393511049351,
          "weightedErr": 0.005182279463707023,
          "strat": 0.003980097236020766,
          "equal": 0.005251524923043271,
          "srs": 0.004820536890487246,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            3,
            2,
            5,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 4,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.0021706951977052994,
          "startErr": 0.003293335664805309,
          "weightedErr": 0.004461807793637979,
          "strat": 0.005163123933741683,
          "equal": 0.003671193020661453,
          "srs": 0.0036055871229824757,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 19,
          "attempted": 23,
          "proposed": 26,
          "lengths": [
            5,
            3,
            3,
            3,
            1,
            1,
            4,
            4
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.001827774194526295,
          "startErr": 0.0029711511114870574,
          "weightedErr": 0.00513981725080412,
          "strat": 0.005242287710312428,
          "equal": 0.003296200023535354,
          "srs": 0.003243258467560642,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 18,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            2,
            1,
            2,
            5,
            1,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            2,
            5,
            1,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.001043457643257139,
          "startErr": 0.003365847861812699,
          "weightedErr": 0.00510229820148181,
          "strat": 0.005949118675528236,
          "equal": 0.0044685420038480085,
          "srs": 0.004315147099578968,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            1,
            2,
            3,
            5,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 4,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 6,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0018716810408250801,
          "startErr": 0.0029578503237943,
          "weightedErr": 0.0031754629881459583,
          "strat": 0.005084657143860053,
          "equal": 0.0045137406073486,
          "srs": 0.004164660425211424,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 25,
          "lengths": [
            5,
            3,
            1,
            5,
            2,
            4,
            4
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.0010163714794679838,
          "startErr": 0.0037719309265623266,
          "weightedErr": 0.006964647289765487,
          "strat": 0.006103387996726251,
          "equal": 0.003796561830190217,
          "srs": 0.0038320429687464202,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 25,
          "lengths": [
            5,
            3,
            2,
            5,
            1,
            5,
            1,
            2
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 19,
          "attempted": 22,
          "proposed": 25,
          "lengths": [
            2,
            5,
            1,
            3,
            4,
            1,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 29,
      "cap": 4,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.005667434933224736,
          "startErr": 0.01770803486672282,
          "weightedErr": 0.024810809643626523,
          "strat": 0.018626655404263486,
          "equal": 0.015642039634607238,
          "srs": 0.017232855006001697,
          "rewards": [
            0.25,
            0,
            0.125
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0019588832724785343,
          "startErr": 0.007136825359352223,
          "weightedErr": 0.008062915004127029,
          "strat": 0.0062747873555793025,
          "equal": 0.005948362309801437,
          "srs": 0.0054368105730070925,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            3,
            2,
            5,
            1
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.008430857585486987,
          "startErr": 0.01153837583911829,
          "weightedErr": 0.013265117351963755,
          "strat": 0.019446161622871765,
          "equal": 0.017170810930383692,
          "srs": 0.016566941080792753,
          "rewards": [
            0.125,
            0,
            0.25
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            3,
            3,
            5
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 1,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.001177608444876906,
          "startErr": 0.0021581386276448224,
          "weightedErr": 0.0021581386276448224,
          "strat": 0.0015164620842263799,
          "equal": 0.0015164620842263799,
          "srs": 0.0017142738774381507,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0030373538000772657,
          "startErr": 0.0019904632331298766,
          "weightedErr": 0.0019904632331298766,
          "strat": 0.0018348830569975758,
          "equal": 0.0018348830569975758,
          "srs": 0.0019133892963535475,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0014156279592672835,
          "startErr": 0.001920980945463259,
          "weightedErr": 0.002207709029133535,
          "strat": 0.0011926496504829941,
          "equal": 0.0014585142813492405,
          "srs": 0.0011586728569811541,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.001553908666229189,
          "startErr": 0.0019147669312366896,
          "weightedErr": 0.0020544556158650193,
          "strat": 0.0016751283745231305,
          "equal": 0.0016702402041867321,
          "srs": 0.0015992455204009303,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 1,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.007338618782328524,
          "startErr": 0.004418059739231984,
          "weightedErr": 0.004773527843251552,
          "strat": 0.0031817671754513235,
          "equal": 0.004895320261000569,
          "srs": 0.003966574515574994,
          "rewards": [
            0.25,
            0.125,
            0
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.005109316068460677,
          "startErr": 0.009830063605433665,
          "weightedErr": 0.009830063605433665,
          "strat": 0.006975213176387567,
          "equal": 0.006975213176387567,
          "srs": 0.0064434552034830055,
          "rewards": [
            0,
            0,
            0.25
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0009250925769555062,
          "startErr": 0.0012223032478870053,
          "weightedErr": 0.0013184091054328337,
          "strat": 0.0018398845884255296,
          "equal": 0.0019117191289964466,
          "srs": 0.0015795490461806697,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 1,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0023396423969007526,
          "startErr": 0.0018125341089049564,
          "weightedErr": 0.002061007189844187,
          "strat": 0.0016542182804379674,
          "equal": 0.0016390444157648862,
          "srs": 0.0014718011221972402,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0021478581028173643,
          "startErr": 0.001109604172054479,
          "weightedErr": 0.0016713463442112725,
          "strat": 0.0021233176920858795,
          "equal": 0.002026470269580926,
          "srs": 0.0016280758212470561,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.001904690115602157,
          "startErr": 0.0014580559414692092,
          "weightedErr": 0.0011262584185309874,
          "strat": 0.0013411391763961317,
          "equal": 0.0020046771776284582,
          "srs": 0.0016380781505115087,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 10,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.001146777541414924,
          "startErr": 0.0012976992665278046,
          "weightedErr": 0.0013744774384854604,
          "strat": 0.0016098627026980821,
          "equal": 0.0014277961090233026,
          "srs": 0.0012165712651381736,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 1,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 14,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 15,
          "gnorm": 0.0019055114431112137,
          "startErr": 0.0010521037027570478,
          "weightedErr": 0.0010621934672039297,
          "strat": 0.0012153679362985376,
          "equal": 0.0011156968796776758,
          "srs": 0.0008574719993454442,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 15,
          "proposed": 15,
          "lengths": [
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.005643426914812265,
          "startErr": 0.005973055098036168,
          "weightedErr": 0.005973055098036168,
          "strat": 0.005442317464394475,
          "equal": 0.005442317464394475,
          "srs": 0.005168997555992586,
          "rewards": [
            0.25,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 2,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.024556527681562274,
          "startErr": 0.01884897073981123,
          "weightedErr": 0.0323671190603274,
          "strat": 0.028534417558094435,
          "equal": 0.022890869816824123,
          "srs": 0.021858764751156327,
          "rewards": [
            0.125,
            0.125,
            0.5
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            1,
            3,
            1
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0013971078977621616,
          "startErr": 0.0022552099117692606,
          "weightedErr": 0.002136058270160146,
          "strat": 0.002849375477398943,
          "equal": 0.0025199634095163185,
          "srs": 0.0023716477926988017,
          "rewards": [
            0.125,
            0.125,
            0.25
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            1,
            1,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.007488298912383541,
          "startErr": 0.015033471747282125,
          "weightedErr": 0.015136107940797016,
          "strat": 0.013600248795210369,
          "equal": 0.01351169007788838,
          "srs": 0.013164621088613439,
          "rewards": [
            0,
            0.25,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            2,
            3,
            3,
            3,
            2,
            2,
            3,
            3
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0014705654440558757,
          "startErr": 0.0016930779781644065,
          "weightedErr": 0.002464230741441252,
          "strat": 0.0021573131663844185,
          "equal": 0.0019085498148545782,
          "srs": 0.001997700899485127,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            1,
            3,
            3,
            1,
            3,
            2,
            3,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 2,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.016913796211698112,
          "startErr": 0.02516538225984078,
          "weightedErr": 0.029517704197030117,
          "strat": 0.027236973375165858,
          "equal": 0.026148312200861605,
          "srs": 0.024321145830693603,
          "rewards": [
            0,
            0.375,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            2,
            3,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            1,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 15,
          "attempted": 19,
          "proposed": 20,
          "lengths": [
            2,
            3,
            2,
            1,
            3,
            1,
            3,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0019495018155664034,
          "startErr": 0.00404197959421384,
          "weightedErr": 0.0043351325238948045,
          "strat": 0.004123272099799659,
          "equal": 0.0039664220625713256,
          "srs": 0.0039399615912301305,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            2,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 2,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.01829402870192532,
          "startErr": 0.025502353120536697,
          "weightedErr": 0.027744848955596476,
          "strat": 0.029610151764948564,
          "equal": 0.028822911276354612,
          "srs": 0.028232199839390387,
          "rewards": [
            0,
            0,
            0.375
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            2,
            3,
            3
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.002308118072579243,
          "startErr": 0.003997913617509958,
          "weightedErr": 0.0043539618459710015,
          "strat": 0.002981259590535041,
          "equal": 0.0026918855418170493,
          "srs": 0.0026387253643165762,
          "rewards": [
            0.25,
            0.125,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            1,
            3,
            3,
            1,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.002899829409618528,
          "startErr": 0.0030439463453997436,
          "weightedErr": 0.0025736953792405985,
          "strat": 0.0029899950516561356,
          "equal": 0.0022114569081989398,
          "srs": 0.002518410213591207,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 16,
          "attempted": 19,
          "proposed": 20,
          "lengths": [
            2,
            3,
            3,
            2,
            3,
            1,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 2,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 20,
          "proposed": 20,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.005817413410606046,
          "startErr": 0.00898430239275618,
          "weightedErr": 0.009600374514627717,
          "strat": 0.011171589487697774,
          "equal": 0.01101713881538166,
          "srs": 0.010023378799081667,
          "rewards": [
            0.25,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            2,
            3,
            3,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 14,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            1,
            3,
            3,
            1,
            1,
            3,
            3,
            1,
            3,
            2,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 4,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.007398735607484138,
          "startErr": 0.012657700444128976,
          "weightedErr": 0.02705792328521984,
          "strat": 0.030097230584025098,
          "equal": 0.022865437379070055,
          "srs": 0.02274282257277143,
          "rewards": [
            0,
            0,
            0.25
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            2,
            5,
            1,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0014895907226308216,
          "startErr": 0.0046413725373333934,
          "weightedErr": 0.0033442420735356,
          "strat": 0.0051830459443184245,
          "equal": 0.004917341578478638,
          "srs": 0.005091325042835476,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.004522933634390667,
          "startErr": 0.020282256228324844,
          "weightedErr": 0.024839204908352546,
          "strat": 0.02035867169358707,
          "equal": 0.019698404247433948,
          "srs": 0.019149167430183047,
          "rewards": [
            0.25,
            0,
            0.125
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            3,
            5,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.001191655265708818,
          "startErr": 0.0030450426179717644,
          "weightedErr": 0.005375253941758175,
          "strat": 0.005359091637511982,
          "equal": 0.0037545300719111227,
          "srs": 0.0036435532668189973,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 25,
          "lengths": [
            5,
            2,
            1,
            5,
            3,
            1,
            5,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 4,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0018233734046984838,
          "startErr": 0.005691940769856023,
          "weightedErr": 0.007410379379677786,
          "strat": 0.004811906273619024,
          "equal": 0.0038643566119668004,
          "srs": 0.0038870386510028204,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 20,
          "attempted": 23,
          "proposed": 26,
          "lengths": [
            3,
            2,
            3,
            4,
            4,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            2,
            1,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0021947942165856577,
          "startErr": 0.003849845945264493,
          "weightedErr": 0.005558320744015741,
          "strat": 0.004784789197104152,
          "equal": 0.0036440166982873953,
          "srs": 0.003425729256237465,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            1,
            2,
            5,
            3,
            3,
            5
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            1,
            5,
            2,
            5,
            3,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 4,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0016835980869617376,
          "startErr": 0.002519621155170037,
          "weightedErr": 0.002741836176211603,
          "strat": 0.005373204654937037,
          "equal": 0.004054308967806509,
          "srs": 0.004198098557463002,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            2,
            1,
            4,
            4
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.25,
            0.25,
            0.25
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 26,
          "lengths": [
            1,
            5,
            2,
            5,
            3,
            2,
            5,
            1
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            2,
            5,
            1,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            2,
            1,
            5,
            3,
            3,
            5
          ]
        }
      ]
    },
    {
      "checkpoint": 0,
      "seed": 73,
      "cap": 4,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.001601822429081954,
          "startErr": 0.0024369530155775505,
          "weightedErr": 0.0023882320357912438,
          "strat": 0.0027474958121040246,
          "equal": 0.0027005449116471437,
          "srs": 0.002626798719675487,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 18,
          "attempted": 22,
          "proposed": 32,
          "lengths": [
            5,
            3,
            2,
            1,
            1,
            2,
            2,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.002256848583254831,
          "startErr": 0.0039262464461173605,
          "weightedErr": 0.0059368510462703,
          "strat": 0.006095371351080715,
          "equal": 0.004592061337932785,
          "srs": 0.004763476540886458,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            2,
            1,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.002803567341419769,
          "startErr": 0.003477715524004156,
          "weightedErr": 0.005425664884578194,
          "strat": 0.0056295036258738,
          "equal": 0.0029040437792921623,
          "srs": 0.003189575514196472,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 17,
          "attempted": 22,
          "proposed": 33,
          "lengths": [
            2,
            5,
            1,
            1,
            1,
            3,
            1,
            2,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            2,
            5,
            1,
            5,
            3,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 1,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0007614306291752759,
          "startErr": 0.0018807220203880684,
          "weightedErr": 0.002040026365186804,
          "strat": 0.0013330933855065373,
          "equal": 0.001200416045409236,
          "srs": 0.0011540078328822095,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0029499161362229864,
          "startErr": 0.0022302881760193787,
          "weightedErr": 0.0022302881760193787,
          "strat": 0.002143986271313324,
          "equal": 0.002143986271313324,
          "srs": 0.002196030223479958,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0015201689191871154,
          "startErr": 0.0016227379665181584,
          "weightedErr": 0.00163946980066973,
          "strat": 0.0017423998506866053,
          "equal": 0.0020372176972449155,
          "srs": 0.0015104540797175412,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0025556522981332822,
          "startErr": 0.001750074147339972,
          "weightedErr": 0.0018967933482153178,
          "strat": 0.0019272292776605763,
          "equal": 0.0018004396510891939,
          "srs": 0.0017977326124721275,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 1,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0020822631067434166,
          "startErr": 0.0021707948412114588,
          "weightedErr": 0.0021707948412114588,
          "strat": 0.0018935384614162715,
          "equal": 0.0018935384614162715,
          "srs": 0.00189335966885955,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0018246916669121989,
          "startErr": 0.0014911633720461367,
          "weightedErr": 0.0017681666106749204,
          "strat": 0.0013731044840670178,
          "equal": 0.0015449925699446724,
          "srs": 0.001289287138327022,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1,
            1,
            2,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0014621243967744538,
          "startErr": 0.002028851615319346,
          "weightedErr": 0.0024384253033264023,
          "strat": 0.0017883113847094714,
          "equal": 0.001443103649825137,
          "srs": 0.0014584131392183584,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0013860025554773357,
          "startErr": 0.001432273879165517,
          "weightedErr": 0.0016066691421947198,
          "strat": 0.00147912368285168,
          "equal": 0.0014480135797665435,
          "srs": 0.001292969804940916,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 1,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0026593896899972276,
          "startErr": 0.0020059040740383347,
          "weightedErr": 0.002323682555107661,
          "strat": 0.002875943542127704,
          "equal": 0.0028395797726893024,
          "srs": 0.0025852232385090374,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0015219696381808528,
          "startErr": 0.0007902466277686085,
          "weightedErr": 0.0012755451751864288,
          "strat": 0.0016893783222060065,
          "equal": 0.0017630768462028107,
          "srs": 0.0015371429307886062,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            1,
            2,
            1,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.006593134039765929,
          "startErr": 0.005632573307581262,
          "weightedErr": 0.005632573307581262,
          "strat": 0.004803000851592954,
          "equal": 0.004803000851592954,
          "srs": 0.004682293229912491,
          "rewards": [
            0.125,
            0,
            0.25
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 1,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0022785994005103654,
          "startErr": 0.0025522554110472043,
          "weightedErr": 0.0031194765284972225,
          "strat": 0.0023859585993475088,
          "equal": 0.0020572463363809405,
          "srs": 0.0018084095824454638,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.011124016225352756,
          "startErr": 0.010104431348091,
          "weightedErr": 0.010104431348091,
          "strat": 0.009208411750689309,
          "equal": 0.009208411750689309,
          "srs": 0.009396103456733208,
          "rewards": [
            0.25,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.004010061861642326,
          "startErr": 0.005602933285476197,
          "weightedErr": 0.005841498935073558,
          "strat": 0.004663918614134775,
          "equal": 0.005081747833352789,
          "srs": 0.004422597786769713,
          "rewards": [
            0.25,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0009438167276783564,
          "startErr": 0.0017107193929609594,
          "weightedErr": 0.001984181804070658,
          "strat": 0.00148954268041025,
          "equal": 0.001330982088952358,
          "srs": 0.001205704518990989,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 2,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0022597826879039943,
          "startErr": 0.002647394228363726,
          "weightedErr": 0.0031819553771379266,
          "strat": 0.004566150197485632,
          "equal": 0.004791036692227683,
          "srs": 0.004127899079822214,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.003166389090613629,
          "startErr": 0.0029352141945833696,
          "weightedErr": 0.0034331355450625462,
          "strat": 0.003685181883571384,
          "equal": 0.003996924970487645,
          "srs": 0.0036257072071402517,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.023784154443305887,
          "startErr": 0.020548579027147306,
          "weightedErr": 0.02090052584801268,
          "strat": 0.017829043526051275,
          "equal": 0.018552075395135106,
          "srs": 0.01992104814356414,
          "rewards": [
            0.25,
            0.25,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            1,
            3,
            1,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 2,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            3,
            3,
            2,
            1,
            3,
            1,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0021718184382761027,
          "startErr": 0.0030323248774927624,
          "weightedErr": 0.0033084569083266166,
          "strat": 0.0033536197502449525,
          "equal": 0.0033863215095605885,
          "srs": 0.003112786223175177,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            1,
            3,
            3,
            1,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0016226150541893915,
          "startErr": 0.0017688468221596036,
          "weightedErr": 0.002023732268141689,
          "strat": 0.002854086372078415,
          "equal": 0.0027094249450019215,
          "srs": 0.002625134685414355,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 2,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.011540105212212224,
          "startErr": 0.013492881657835364,
          "weightedErr": 0.018655119510174838,
          "strat": 0.015930078357375713,
          "equal": 0.013446647673735133,
          "srs": 0.012256299900961515,
          "rewards": [
            0,
            0.25,
            0
          ],
          "accepted": 15,
          "attempted": 19,
          "proposed": 21,
          "lengths": [
            3,
            2,
            1,
            2,
            1,
            3,
            3,
            1,
            3,
            2,
            3
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            2,
            3,
            3
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            2,
            3,
            3,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.0025318421920445663,
          "startErr": 0.00248976888574117,
          "weightedErr": 0.003133070003497206,
          "strat": 0.00253840670335789,
          "equal": 0.0023702602838151037,
          "srs": 0.002533356154179827,
          "rewards": [
            0.25,
            0.125,
            0.125
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            1,
            3,
            3,
            1,
            1,
            3,
            3,
            1,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 2,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.001359036580500808,
          "startErr": 0.0032153511215449477,
          "weightedErr": 0.003992364807390219,
          "strat": 0.003034726502713941,
          "equal": 0.0026798608750318903,
          "srs": 0.0023783293614362756,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 15,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            3,
            1,
            3,
            1,
            3,
            3,
            1,
            1,
            2,
            3,
            3
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.001561802083931942,
          "startErr": 0.002644545776406569,
          "weightedErr": 0.0034039177158948406,
          "strat": 0.0034783767720381873,
          "equal": 0.0031998729838436804,
          "srs": 0.0026280015034177862,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            1,
            3,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            1,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 4,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0035018131374015015,
          "startErr": 0.006824866764072195,
          "weightedErr": 0.00802435332503637,
          "strat": 0.006078570816452566,
          "equal": 0.004601277124838566,
          "srs": 0.0052135723873696135,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0021602043931987407,
          "startErr": 0.009790720638242746,
          "weightedErr": 0.009293188909491148,
          "strat": 0.0072227333782865725,
          "equal": 0.007018192253866066,
          "srs": 0.007023956207047509,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            3,
            4,
            4
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.00432600717513553,
          "startErr": 0.009560399692932361,
          "weightedErr": 0.011184444032981557,
          "strat": 0.008811739158078329,
          "equal": 0.007799093808851669,
          "srs": 0.008125311625394364,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            3,
            5,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0023614831615662297,
          "startErr": 0.0059859924183024785,
          "weightedErr": 0.006521806104631364,
          "strat": 0.006396363750773603,
          "equal": 0.006626325480292954,
          "srs": 0.006354352763007812,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            3,
            3,
            5
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 4,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.00833871598687784,
          "startErr": 0.015257193746253615,
          "weightedErr": 0.020317655165974325,
          "strat": 0.018841536813968725,
          "equal": 0.015585763586260175,
          "srs": 0.014007037052556397,
          "rewards": [
            0.125,
            0,
            0.25
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            3,
            5,
            3,
            1,
            5,
            2
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.0016991542961513886,
          "startErr": 0.0021237498032229602,
          "weightedErr": 0.0024722596799541187,
          "strat": 0.004193142319139411,
          "equal": 0.003984475169075859,
          "srs": 0.0037413288322887123,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 22,
          "proposed": 25,
          "lengths": [
            5,
            3,
            5,
            2,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.001537332681197081,
          "startErr": 0.004681515898971436,
          "weightedErr": 0.007944699465289142,
          "strat": 0.006226187526598156,
          "equal": 0.005069383428582681,
          "srs": 0.004811445493327312,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.017869036436166746,
          "startErr": 0.04135894540599786,
          "weightedErr": 0.054019464864899926,
          "strat": 0.06457573226675395,
          "equal": 0.045612328629109536,
          "srs": 0.048594207811397006,
          "rewards": [
            0.375,
            0,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 25,
          "lengths": [
            5,
            3,
            1,
            5,
            2,
            3,
            5
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 4,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.008007469669881459,
          "startErr": 0.03250580101471675,
          "weightedErr": 0.041229461183268014,
          "strat": 0.03080414131231624,
          "equal": 0.02999572388443666,
          "srs": 0.02878394583105556,
          "rewards": [
            0,
            0.25,
            0
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 23,
          "lengths": [
            3,
            5,
            5,
            3,
            4,
            4
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            3,
            5,
            2,
            5,
            1,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0018372441328962815,
          "startErr": 0.007218930424874891,
          "weightedErr": 0.00815068489698076,
          "strat": 0.007189820946890742,
          "equal": 0.006610933092015957,
          "srs": 0.0065412378206148175,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 21,
          "attempted": 22,
          "proposed": 22,
          "lengths": [
            4,
            4,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0029064156666605075,
          "startErr": 0.005084385192768327,
          "weightedErr": 0.00657094433618606,
          "strat": 0.006044627492376831,
          "equal": 0.00569059579947147,
          "srs": 0.005757273203317434,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            3,
            5,
            3,
            5
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 11,
      "cap": 4,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.002904056664265666,
          "startErr": 0.004888623897564289,
          "weightedErr": 0.0069651071241866366,
          "strat": 0.005757868922398999,
          "equal": 0.0048490229467411244,
          "srs": 0.0049270092277888715,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0018817174095291196,
          "startErr": 0.005159847558546008,
          "weightedErr": 0.006015140289364138,
          "strat": 0.006462595702262962,
          "equal": 0.004799653029459149,
          "srs": 0.005086445389050436,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 24,
          "lengths": [
            5,
            3,
            3,
            3,
            2,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.002089249543528112,
          "startErr": 0.0066699945029655115,
          "weightedErr": 0.006598017586110828,
          "strat": 0.007028530090285223,
          "equal": 0.006376309548695314,
          "srs": 0.006312656428536189,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0016946802350036659,
          "startErr": 0.005947332373208575,
          "weightedErr": 0.0069617815370162425,
          "strat": 0.005468205687254154,
          "equal": 0.005055274370024152,
          "srs": 0.004770829479783945,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            3,
            5,
            3,
            1,
            5,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 1,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.015046331641066274,
          "startErr": 0.012253482979289135,
          "weightedErr": 0.012253482979289135,
          "strat": 0.013646974290169662,
          "equal": 0.013646974290169662,
          "srs": 0.012859730673906394,
          "rewards": [
            0,
            0.125,
            0.375
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.25,
            0.25,
            0.25
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 1,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0011419058963979007,
          "startErr": 0.0014966852408549575,
          "weightedErr": 0.0019451106004859501,
          "strat": 0.0015328263642229044,
          "equal": 0.0014772436398012621,
          "srs": 0.0015362433335334004,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 11,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0009959413902118261,
          "startErr": 0.0009064631671843635,
          "weightedErr": 0.0009597139624372365,
          "strat": 0.001349776221796838,
          "equal": 0.0013076408189879587,
          "srs": 0.0012407497564634754,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 11,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0020498455814983124,
          "startErr": 0.0018046779987337142,
          "weightedErr": 0.0018046779987337142,
          "strat": 0.002323218508330724,
          "equal": 0.002323218508330724,
          "srs": 0.0023719300997829197,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 1,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0027936248863546474,
          "startErr": 0.0021679793938471155,
          "weightedErr": 0.0025364622556179597,
          "strat": 0.002142763047882565,
          "equal": 0.002157660058965182,
          "srs": 0.00183533890202351,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0018306059758961403,
          "startErr": 0.002467907418977434,
          "weightedErr": 0.002467907418977434,
          "strat": 0.0021387232958994173,
          "equal": 0.0021387232958994173,
          "srs": 0.0020971353571452673,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 11,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0014820502191163765,
          "startErr": 0.0017366792930195503,
          "weightedErr": 0.0017366792930195503,
          "strat": 0.002080896914231104,
          "equal": 0.002080896914231104,
          "srs": 0.0021196277304772835,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 1,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0020427489183570133,
          "startErr": 0.0024211940923811234,
          "weightedErr": 0.0024211940923811234,
          "strat": 0.002549634846063514,
          "equal": 0.002549634846063514,
          "srs": 0.0024266723844552107,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.007588289889190732,
          "startErr": 0.004994687602086774,
          "weightedErr": 0.004994687602086774,
          "strat": 0.005730458108471776,
          "equal": 0.005730458108471776,
          "srs": 0.005532473035456914,
          "rewards": [
            0,
            0.125,
            0.25
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0015081682368496797,
          "startErr": 0.0007377425081410771,
          "weightedErr": 0.0008497884991387215,
          "strat": 0.0011795681965182294,
          "equal": 0.0011363909364730173,
          "srs": 0.0009524506069468348,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 2,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.02145947368661755,
          "startErr": 0.027513546680975564,
          "weightedErr": 0.027643463881655134,
          "strat": 0.031824470739280346,
          "equal": 0.027682976152408036,
          "srs": 0.02826840575739688,
          "rewards": [
            0,
            0.375,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            1,
            3,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.009187009403011493,
          "startErr": 0.015291359535981135,
          "weightedErr": 0.014412471120720128,
          "strat": 0.015115678011065368,
          "equal": 0.015219388027610978,
          "srs": 0.013099547371649107,
          "rewards": [
            0.25,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0037947880283724436,
          "startErr": 0.0042857147542050295,
          "weightedErr": 0.0043565247346980486,
          "strat": 0.0038108615704248844,
          "equal": 0.0032028364428767467,
          "srs": 0.0035236012085929852,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            1,
            1,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 2,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.0016912588204797128,
          "startErr": 0.0024562041976888653,
          "weightedErr": 0.0030762793969330684,
          "strat": 0.0027436318736869807,
          "equal": 0.002200108514922059,
          "srs": 0.002338206528044378,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 15,
          "attempted": 18,
          "proposed": 21,
          "lengths": [
            3,
            1,
            1,
            3,
            3,
            3,
            1,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0019823692150141184,
          "startErr": 0.0035893263283601215,
          "weightedErr": 0.002304567375152785,
          "strat": 0.0034380491953426746,
          "equal": 0.004049365396691937,
          "srs": 0.003366007644651475,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.005768897919545567,
          "startErr": 0.0142988914441337,
          "weightedErr": 0.013396624340257438,
          "strat": 0.011528161269653309,
          "equal": 0.011426108162949667,
          "srs": 0.011052450219340848,
          "rewards": [
            0.25,
            0.125,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            2,
            3,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 2,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0019442750926839188,
          "startErr": 0.003526594073295032,
          "weightedErr": 0.004064364300039458,
          "strat": 0.00369475072969447,
          "equal": 0.0033302256267002195,
          "srs": 0.0032340296789130133,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 15,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            3,
            3,
            1,
            1,
            3,
            1,
            3,
            1,
            2,
            3,
            3
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.002321115019781452,
          "startErr": 0.00280378650813192,
          "weightedErr": 0.0031134345357806776,
          "strat": 0.0036891370721771947,
          "equal": 0.003860217866316489,
          "srs": 0.0031387155532434222,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            2,
            3,
            3,
            3,
            1,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 2,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0016432814346138407,
          "startErr": 0.0027537108264462126,
          "weightedErr": 0.003304646932658067,
          "strat": 0.0032742525009804876,
          "equal": 0.0032267976813746416,
          "srs": 0.003045273394949367,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.008765207293298485,
          "startErr": 0.006910431119771752,
          "weightedErr": 0.009281874956219842,
          "strat": 0.010963497565112928,
          "equal": 0.0067277754521828225,
          "srs": 0.006331902580205857,
          "rewards": [
            0,
            0.125,
            0.25
          ],
          "accepted": 15,
          "attempted": 18,
          "proposed": 21,
          "lengths": [
            1,
            3,
            3,
            1,
            3,
            3,
            1,
            1,
            1,
            3,
            3,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.002571707629617368,
          "startErr": 0.0022314960748601093,
          "weightedErr": 0.0017606964549917324,
          "strat": 0.0023924651328830083,
          "equal": 0.0022189572467257133,
          "srs": 0.0022560817700444692,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 14,
          "attempted": 20,
          "proposed": 23,
          "lengths": [
            3,
            3,
            2,
            1,
            2,
            1,
            3,
            1,
            2,
            3,
            1,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.002061888113224445,
          "startErr": 0.004200371466271627,
          "weightedErr": 0.004401171580574978,
          "strat": 0.003925577372106229,
          "equal": 0.0038776488114899347,
          "srs": 0.0035823324280386164,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 4,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0026064850527202595,
          "startErr": 0.006001682117091626,
          "weightedErr": 0.00493661920160255,
          "strat": 0.00427286333998176,
          "equal": 0.004994543773749683,
          "srs": 0.004836317074568488,
          "rewards": [
            0.125,
            0.25,
            0.125
          ],
          "accepted": 20,
          "attempted": 23,
          "proposed": 26,
          "lengths": [
            5,
            3,
            3,
            2,
            3,
            4,
            4
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.003605854995671647,
          "startErr": 0.006540389420533277,
          "weightedErr": 0.006245877875433079,
          "strat": 0.006297627233690854,
          "equal": 0.006299369160710524,
          "srs": 0.006631688702409115,
          "rewards": [
            0.25,
            0.125,
            0.125
          ],
          "accepted": 21,
          "attempted": 22,
          "proposed": 22,
          "lengths": [
            4,
            4,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.008021495324225535,
          "startErr": 0.015023146221219972,
          "weightedErr": 0.02202655209663569,
          "strat": 0.026813074652974108,
          "equal": 0.02104487637642037,
          "srs": 0.021009731035425156,
          "rewards": [
            0,
            0.25,
            0.25
          ],
          "accepted": 19,
          "attempted": 23,
          "proposed": 27,
          "lengths": [
            5,
            3,
            4,
            4,
            3,
            1,
            4
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.001783403686069054,
          "startErr": 0.004134537731492216,
          "weightedErr": 0.005501505106592444,
          "strat": 0.005647998584390963,
          "equal": 0.004655567412064474,
          "srs": 0.004506495135287318,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            1,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 4,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 6,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            3,
            3,
            5
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 18,
          "attempted": 21,
          "proposed": 25,
          "lengths": [
            1,
            5,
            2,
            5,
            3,
            3,
            5
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.0018939472006184118,
          "startErr": 0.003385403454342394,
          "weightedErr": 0.006311555536166953,
          "strat": 0.006376671600890742,
          "equal": 0.0043615415656018935,
          "srs": 0.004105068379427658,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 26,
          "lengths": [
            5,
            3,
            2,
            5,
            1,
            1,
            5,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 4,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.002616360734944582,
          "startErr": 0.004355799901427529,
          "weightedErr": 0.00449051393907525,
          "strat": 0.006826635428193837,
          "equal": 0.005856899740794488,
          "srs": 0.005474669635893841,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            1,
            2,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0032586884315493198,
          "startErr": 0.007525931868651101,
          "weightedErr": 0.00938557019237155,
          "strat": 0.008696174683186298,
          "equal": 0.0053818003268327206,
          "srs": 0.006123500787277283,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            3,
            5,
            3,
            1,
            5,
            2
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.011458775199551428,
          "startErr": 0.014541536441371817,
          "weightedErr": 0.018916816166378556,
          "strat": 0.019495306392647536,
          "equal": 0.015157942729024595,
          "srs": 0.014924373140047931,
          "rewards": [
            0.25,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            3,
            1,
            5,
            2,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 26,
          "lengths": [
            2,
            5,
            1,
            5,
            3,
            1,
            5,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 29,
      "cap": 4,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 6,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0041115819786845245,
          "startErr": 0.011288522515896373,
          "weightedErr": 0.017878066302831674,
          "strat": 0.019254396689737656,
          "equal": 0.014211974162391141,
          "srs": 0.013748473005068824,
          "rewards": [
            0.25,
            0.125,
            0
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            3,
            2,
            5,
            1
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0018984607371055882,
          "startErr": 0.004605057175365236,
          "weightedErr": 0.007652654048341191,
          "strat": 0.0061068215643430325,
          "equal": 0.004506683057120586,
          "srs": 0.004705280092515717,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 24,
          "lengths": [
            5,
            3,
            5,
            3,
            2,
            4,
            2
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.003279905088186745,
          "startErr": 0.005028777237188251,
          "weightedErr": 0.007675248891838142,
          "strat": 0.006033741969610384,
          "equal": 0.005607790116364842,
          "srs": 0.005730830054019952,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 1,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0021457413659764354,
          "startErr": 0.0019727637303630333,
          "weightedErr": 0.0021306534193907367,
          "strat": 0.0016423022481647643,
          "equal": 0.0015966085291653758,
          "srs": 0.001441040833766155,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0021189328066940826,
          "startErr": 0.001523645353733424,
          "weightedErr": 0.001523645353733424,
          "strat": 0.0019313518863869077,
          "equal": 0.0019313518863869077,
          "srs": 0.002006505148165991,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.016802862946675632,
          "startErr": 0.017660322885053745,
          "weightedErr": 0.017660322885053745,
          "strat": 0.017064587901896276,
          "equal": 0.017064587901896276,
          "srs": 0.016427592536447974,
          "rewards": [
            0,
            0.375,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 1,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.002126367999347246,
          "startErr": 0.0012362247848390067,
          "weightedErr": 0.0013816861309545888,
          "strat": 0.0013575835496386437,
          "equal": 0.0013313359316428472,
          "srs": 0.001265860041014074,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            1,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0016511553302312064,
          "startErr": 0.0024998779636968057,
          "weightedErr": 0.0026822543665922997,
          "strat": 0.002247187359096567,
          "equal": 0.0019628647691268953,
          "srs": 0.0018371121950379883,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0020808289913730412,
          "startErr": 0.0020627252327905425,
          "weightedErr": 0.0020627252327905425,
          "strat": 0.0018209015739140807,
          "equal": 0.0018209015739140807,
          "srs": 0.001761973510021424,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 1,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 15,
          "gnorm": 0.014186126624357058,
          "startErr": 0.0053384965955779635,
          "weightedErr": 0.010529511990943864,
          "strat": 0.007997898434181722,
          "equal": 0.006266161910708714,
          "srs": 0.006265903304906074,
          "rewards": [
            0,
            0.25,
            0
          ],
          "accepted": 11,
          "attempted": 15,
          "proposed": 15,
          "lengths": [
            1,
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            1
          ]
        },
        {
          "n": 24,
          "m": 14,
          "gnorm": 0.0030344635076725397,
          "startErr": 0.0012821562468400538,
          "weightedErr": 0.0014771234673505,
          "strat": 0.0013920257701615923,
          "equal": 0.0014571830401261627,
          "srs": 0.0013186551532321422,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 14,
          "proposed": 14,
          "lengths": [
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0023909275904244187,
          "startErr": 0.001814384029872642,
          "weightedErr": 0.001814384029872642,
          "strat": 0.0016037448979005864,
          "equal": 0.0016037448979005864,
          "srs": 0.001781286194013566,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0.0016028841380027544,
          "startErr": 0.0010721391406196747,
          "weightedErr": 0.00122624862934511,
          "strat": 0.001579995790034992,
          "equal": 0.0014639742266252364,
          "srs": 0.001494761645431986,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 1,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.0011305532165832113,
          "startErr": 0.0011783587180439687,
          "weightedErr": 0.0011783587180439687,
          "strat": 0.0015347220399984209,
          "equal": 0.0015347220399984209,
          "srs": 0.0015098508347119728,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 13,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 12,
          "attempted": 13,
          "proposed": 13,
          "lengths": [
            1,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        },
        {
          "n": 24,
          "m": 12,
          "gnorm": 0.002314719151986425,
          "startErr": 0.002023268153853153,
          "weightedErr": 0.002023268153853153,
          "strat": 0.002043158133601163,
          "equal": 0.002043158133601163,
          "srs": 0.0022308826067113465,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 12,
          "attempted": 12,
          "proposed": 12,
          "lengths": [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 2,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0017917731909326134,
          "startErr": 0.002464782587650024,
          "weightedErr": 0.0031610455732746052,
          "strat": 0.0035817039391488155,
          "equal": 0.0028941253979392527,
          "srs": 0.002810223931769054,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            1,
            1,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.004761384807427014,
          "startErr": 0.009564404218983925,
          "weightedErr": 0.009408729763896125,
          "strat": 0.01025611955714233,
          "equal": 0.00936131306680182,
          "srs": 0.007851495954576723,
          "rewards": [
            0.125,
            0.25,
            0
          ],
          "accepted": 15,
          "attempted": 18,
          "proposed": 20,
          "lengths": [
            3,
            3,
            1,
            1,
            3,
            3,
            2,
            1,
            3,
            3,
            1
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.007227845965432932,
          "startErr": 0.00997478861710059,
          "weightedErr": 0.011392199088669262,
          "strat": 0.010894328494022387,
          "equal": 0.011989039173393223,
          "srs": 0.011189662465475364,
          "rewards": [
            0,
            0.25,
            0.25
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.001787776572613051,
          "startErr": 0.003174343729168664,
          "weightedErr": 0.0033781807984357305,
          "strat": 0.0035519115359950574,
          "equal": 0.003462814955187605,
          "srs": 0.0033262565129750406,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 2,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0024214969230768003,
          "startErr": 0.002355046562552107,
          "weightedErr": 0.0030804514574936798,
          "strat": 0.003311307337946162,
          "equal": 0.002816341758509903,
          "srs": 0.002700685480557445,
          "rewards": [
            0,
            0.125,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            1,
            3,
            1,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            1,
            3,
            1
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.0019455799177724952,
          "startErr": 0.0031570494138905376,
          "weightedErr": 0.003242023799297438,
          "strat": 0.003466680964286825,
          "equal": 0.0036163822769552695,
          "srs": 0.003405552203282556,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            2,
            3,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.002247547309623384,
          "startErr": 0.005005524365390267,
          "weightedErr": 0.00512818464627502,
          "strat": 0.003952435349201174,
          "equal": 0.004284173518183032,
          "srs": 0.0038017627660711178,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 18,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 2,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0013744001099492518,
          "startErr": 0.0017684316051215345,
          "weightedErr": 0.0019266953431385546,
          "strat": 0.0037237552011433824,
          "equal": 0.002833349525073437,
          "srs": 0.002872868542702395,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            1,
            3,
            3,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.019419810302164796,
          "startErr": 0.02849998223342367,
          "weightedErr": 0.03073836273146413,
          "strat": 0.027393208522463894,
          "equal": 0.031167892622079817,
          "srs": 0.02851882856571979,
          "rewards": [
            0,
            0.125,
            0.375
          ],
          "accepted": 17,
          "attempted": 18,
          "proposed": 18,
          "lengths": [
            3,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            1,
            3,
            3,
            1,
            3,
            3,
            2,
            3,
            2,
            3
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.00992706924358138,
          "startErr": 0.016925488088128174,
          "weightedErr": 0.023450483420653555,
          "strat": 0.021311900887882377,
          "equal": 0.017026805444194614,
          "srs": 0.017936699967986226,
          "rewards": [
            0.375,
            0.125,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            3,
            2,
            3,
            3,
            3,
            1,
            1
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 2,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.0023278590565171957,
          "startErr": 0.0032058139812658737,
          "weightedErr": 0.0024705886918886203,
          "strat": 0.003338024601260937,
          "equal": 0.0031005374539169244,
          "srs": 0.0030236679799279134,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            2,
            3,
            3,
            3,
            2,
            1,
            3,
            3,
            1
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.006521149897222303,
          "startErr": 0.007997084906898263,
          "weightedErr": 0.011663122041263466,
          "strat": 0.008818036203140723,
          "equal": 0.007799632306015082,
          "srs": 0.006650226861078783,
          "rewards": [
            0.125,
            0.25,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            1,
            3,
            3,
            1,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 11,
          "gnorm": 0.015079539673218184,
          "startErr": 0.02512758918569843,
          "weightedErr": 0.02063267284261106,
          "strat": 0.021678008778049877,
          "equal": 0.018936726341646865,
          "srs": 0.01898112284000784,
          "rewards": [
            0.375,
            0,
            0.125
          ],
          "accepted": 16,
          "attempted": 19,
          "proposed": 21,
          "lengths": [
            3,
            3,
            1,
            1,
            2,
            1,
            3,
            2,
            3,
            3,
            2
          ]
        },
        {
          "n": 24,
          "m": 10,
          "gnorm": 0.001994443871310365,
          "startErr": 0.0018419996930516011,
          "weightedErr": 0.0022699508260033813,
          "strat": 0.0030449672342699676,
          "equal": 0.0026848188033191515,
          "srs": 0.002509157225986674,
          "rewards": [
            0.125,
            0,
            0
          ],
          "accepted": 16,
          "attempted": 18,
          "proposed": 19,
          "lengths": [
            3,
            3,
            2,
            1,
            3,
            3,
            1,
            3,
            3,
            2
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 4,
      "client": 0,
      "cases": [
        {
          "n": 24,
          "m": 9,
          "gnorm": 0.007481301304441549,
          "startErr": 0.017636061823092207,
          "weightedErr": 0.023374518425064415,
          "strat": 0.020203444338243875,
          "equal": 0.013210425813913886,
          "srs": 0.012009480462106243,
          "rewards": [
            0,
            0.125,
            0.25
          ],
          "accepted": 18,
          "attempted": 21,
          "proposed": 26,
          "lengths": [
            5,
            1,
            1,
            1,
            5,
            1,
            2,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.007141486145176924,
          "startErr": 0.014152389643142253,
          "weightedErr": 0.018456807557224234,
          "strat": 0.022419866179450843,
          "equal": 0.016299130388146586,
          "srs": 0.016745075373291667,
          "rewards": [
            0,
            0.125,
            0.25
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            2,
            1
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.002235561627177758,
          "startErr": 0.006112505436120748,
          "weightedErr": 0.007430069351772663,
          "strat": 0.007331855446260046,
          "equal": 0.006611533151472381,
          "srs": 0.006703446376007247,
          "rewards": [
            0.125,
            0.25,
            0.125
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 21,
          "attempted": 22,
          "proposed": 22,
          "lengths": [
            5,
            3,
            5,
            3,
            4,
            4
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 4,
      "client": 1,
      "cases": [
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.00240131140396997,
          "startErr": 0.006814308011800848,
          "weightedErr": 0.006906596116637436,
          "strat": 0.006377544938539351,
          "equal": 0.006619875895090337,
          "srs": 0.00613099921038396,
          "rewards": [
            0.125,
            0.125,
            0
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 23,
          "lengths": [
            3,
            4,
            1,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0031990024698610813,
          "startErr": 0.004785234811234006,
          "weightedErr": 0.004590110558524072,
          "strat": 0.00483239359150224,
          "equal": 0.0034773594316559447,
          "srs": 0.0035190824930158436,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 21,
          "proposed": 24,
          "lengths": [
            5,
            3,
            5,
            3,
            1,
            5,
            2
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0040074174586947845,
          "startErr": 0.030140834822649135,
          "weightedErr": 0.03440337413195847,
          "strat": 0.018459956354476927,
          "equal": 0.01634205813675414,
          "srs": 0.015983971885574023,
          "rewards": [
            0.125,
            0,
            0.25
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.008983173934215183,
          "startErr": 0.03964084514312631,
          "weightedErr": 0.03608485714072361,
          "strat": 0.026271205540447252,
          "equal": 0.025071771112034846,
          "srs": 0.024490688451720942,
          "rewards": [
            0,
            0,
            0.25
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 4,
      "client": 2,
      "cases": [
        {
          "n": 24,
          "m": 8,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 26,
          "lengths": [
            1,
            5,
            2,
            5,
            3,
            2,
            5,
            1
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.004731455318824567,
          "startErr": 0.010264282964872966,
          "weightedErr": 0.017586757579714337,
          "strat": 0.017008101072549256,
          "equal": 0.011234030134239488,
          "srs": 0.01082911696411855,
          "rewards": [
            0.25,
            0,
            0.125
          ],
          "accepted": 19,
          "attempted": 21,
          "proposed": 27,
          "lengths": [
            1,
            5,
            2,
            5,
            3,
            1,
            5,
            2
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0.125,
            0.125,
            0.125
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 23,
          "lengths": [
            5,
            3,
            5,
            3,
            4,
            3,
            1
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0.0012382125231249783,
          "startErr": 0.004516639219556522,
          "weightedErr": 0.005578691138361617,
          "strat": 0.005659401669336799,
          "equal": 0.004711923473856053,
          "srs": 0.004808924270167759,
          "rewards": [
            0,
            0.125,
            0.125
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        }
      ]
    },
    {
      "checkpoint": 1,
      "seed": 73,
      "cap": 4,
      "client": 3,
      "cases": [
        {
          "n": 24,
          "m": 8,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 19,
          "attempted": 22,
          "proposed": 27,
          "lengths": [
            5,
            1,
            2,
            3,
            2,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 8,
          "gnorm": 0.0028547489545422706,
          "startErr": 0.004507530124651291,
          "weightedErr": 0.005076070548986746,
          "strat": 0.005621392200308622,
          "equal": 0.004328119845843577,
          "srs": 0.00420312863748633,
          "rewards": [
            0.125,
            0,
            0.125
          ],
          "accepted": 18,
          "attempted": 21,
          "proposed": 26,
          "lengths": [
            3,
            5,
            1,
            5,
            1,
            1,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 6,
          "gnorm": 0,
          "startErr": 0,
          "weightedErr": 0,
          "strat": 0,
          "equal": 0,
          "srs": 0,
          "rewards": [
            0,
            0,
            0
          ],
          "accepted": 21,
          "attempted": 21,
          "proposed": 21,
          "lengths": [
            5,
            3,
            5,
            3,
            5,
            3
          ]
        },
        {
          "n": 24,
          "m": 7,
          "gnorm": 0.0017644767741280811,
          "startErr": 0.004973525333821214,
          "weightedErr": 0.0058750907288320645,
          "strat": 0.005913297761948552,
          "equal": 0.0049702240967342045,
          "srs": 0.004757234544829793,
          "rewards": [
            0,
            0,
            0.125
          ],
          "accepted": 20,
          "attempted": 22,
          "proposed": 26,
          "lengths": [
            2,
            2,
            4,
            5,
            3,
            5,
            3
          ]
        }
      ]
    }
  ]
}
~~~
