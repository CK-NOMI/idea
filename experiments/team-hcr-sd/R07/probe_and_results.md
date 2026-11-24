<!-- R07 actual experiment source and full results; extract JS fence to probe.js -->

~~~javascript
// Download pinned Gutenberg files from sources listed in README. Node.js: node probe.js wallpaper.txt alice.txt
function probe(texts){
 const V=27,D=V*V,N=16,a=.3,rho=1,lr=4,gamma=.35,rt=Math.SQRT2;
 const clean=t=>{let start=t.indexOf("*** START"),end=t.indexOf("*** END",start+1);t=t.slice(t.indexOf("\n",start)+1,end>0?end:undefined).toLowerCase().replace(/[^a-z]+/g," ").trim();return [...t].map(c=>c===" "?26:c.charCodeAt(0)-97)};
 const data=texts.map(clean), pub=data.map(x=>x.slice(0,Math.floor(.1*x.length))), train=data.map(x=>x.slice(Math.floor(.1*x.length),Math.floor(.8*x.length))), test=data.map(x=>x.slice(Math.floor(.8*x.length)));
 const count=x=>{let n=new Float64Array(V**3);for(let i=2;i<x.length;i++)n[(x[i-2]*V+x[i-1])*V+x[i]]++;return n};
 const add=(arr,x)=>{for(let i=0;i<arr.length;i++)arr[i]+=x[i];return arr};
 const pubC=pub.map(count).reduce(add,new Float64Array(V**3)), testC=test.map(count).reduce(add,new Float64Array(V**3));
 let init=new Float64Array(D),cond=new Float64Array(D);
 for(let j=0;j<V;j++){let f=Array(V).fill(.5);for(let k=0;k<V;k++){let c=0;for(let y=0;y<V;y++){let x=pubC[(k*V+j)*V+y];f[y]+=x;c+=x}cond[k*V+j]=c+.5}for(let y=0;y<V;y++)init[j*V+y]=Math.log(f[y]/f.reduce((s,v)=>s+v,0));}
 const probs=(w,k,j,target)=>{let z=Array.from({length:V},(_,y)=>w[j*V+y]+(target?gamma*w[k*V+y]:0)),m=Math.max(...z);let p=z.map(x=>Math.exp(x-m)),s=p.reduce((a,b)=>a+b,0);return p.map(x=>x/s)};
 function grads(c){let gp=new Float64Array(D),gq=new Float64Array(D),tot=c.reduce((a,b)=>a+b,0);for(let k=0;k<V;k++)for(let j=0;j<V;j++){let o=(k*V+j)*V, n=0;for(let y=0;y<V;y++)n+=c[o+y];if(!n)continue;let p=probs(init,k,j,true),q=probs(init,k,j,false);for(let y=0;y<V;y++){let dp=(n*p[y]-c[o+y])/tot,dq=(n*q[y]-c[o+y])/tot;gp[j*V+y]+=dp;gp[k*V+y]+=gamma*dp;gq[j*V+y]+=dq}}return [gp,gq]}
 const clients=train.flatMap(x=>Array.from({length:8},(_,i)=>count(x.slice(Math.floor(i*x.length/8),Math.floor((i+1)*x.length/8))))).map(grads);
 const norm=x=>Math.sqrt(x.reduce((s,v)=>s+v*v,0));
 const common=clients.map(([p,q])=>p.map((v,i)=>(v+q[i])/rt)),diff=clients.map(([p,q])=>p.map((v,i)=>(v-q[i])/rt));
 const clip=(x,c)=>x.map(v=>v*Math.min(1,c/Math.max(1e-30,norm(x))));
 function evaluate(pw,qw){let acc=0,nll=0,qnll=0,n=0;for(let k=0;k<V;k++)for(let j=0;j<V;j++){let o=(k*V+j)*V,m=0;for(let y=0;y<V;y++)m+=testC[o+y];if(!m)continue;let p=probs(pw,k,j,true),q=probs(qw,k,j,false);for(let y=0;y<V;y++){acc+=m*Math.min(p[y],q[y]);nll-=testC[o+y]*Math.log(p[y]);qnll-=testC[o+y]*Math.log(q[y])}n+=m}return {accept:acc/n,targetNLL:nll/n,draftNLL:qnll/n}}
 function distill(pw){let w=new Float64Array(D);for(let j=0;j<V;j++){let x=new Float64Array(V),n=0;for(let k=0;k<V;k++){let c=cond[k*V+j],p=probs(pw,k,j,true);n+=c;for(let y=0;y<V;y++)x[y]+=c*p[y]}for(let y=0;y<V;y++)w[j*V+y]=Math.log(x[y]/n)}return w}
 let state;const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+.5)/4294967296},normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
 const rows=[];const avg=(items)=>Object.fromEntries(Object.keys(items[0]).map(k=>[k,items.reduce((s,v)=>s+v[k],0)/items.length]));
 const uc=common.map(x=>clip(x,a));let uSum=uc.reduce(add,new Float64Array(D));
 for(let b of [.015,.05,.15,.3,.6]){
 let vc=diff.map(x=>clip(x,b)),vSum=vc.reduce(add,new Float64Array(D)),pg=uSum.map((x,i)=>(x+vSum[i])/rt),qg=uSum.map((x,i)=>(x-vSum[i])/rt),pw=init.map((x,i)=>x-lr*pg[i]/N),qw=init.map((x,i)=>x-lr*qg[i]/N);
 let s2=(a*a+b*b)/(2*rho),corr=Math.max(0,(a*a-b*b)/(a*a+b*b)),sT=(a+b)/(2*Math.sqrt(rho));
 let cleanMetric=evaluate(pw,qw),targetOnlyVar=sT*sT;
 for(let seed of [11,29,73]){state=seed;let metrics={independent:[],paired:[],targetPostprocess:[],targetOnlyPostprocess:[]};
 for(let rep=0;rep<24;rep++){let z1=Float64Array.from({length:D},normal),z2=Float64Array.from({length:D},normal);
 let tp=pw.map((x,i)=>x-lr*Math.sqrt(s2)*z1[i]/N),qi=qw.map((x,i)=>x-lr*Math.sqrt(s2)*z2[i]/N),qc=qw.map((x,i)=>x-lr*Math.sqrt(s2)*(corr*z1[i]+Math.sqrt(1-corr*corr)*z2[i])/N),ts=pw.map((x,i)=>x-lr*sT*z1[i]/N);
 metrics.independent.push(evaluate(tp,qi));metrics.paired.push(evaluate(tp,qc));metrics.targetPostprocess.push(evaluate(tp,distill(tp)));metrics.targetOnlyPostprocess.push(evaluate(ts,distill(ts)))}
 rows.push({b,seed,a,rho,correlation:corr,marginalVariance:s2,targetOnlyVar,actualSensitivityBound:a*a/(s2*(1+corr))+b*b/(s2*(1-corr)),diffClipped:diff.filter(x=>norm(x)>b).length,commonClipped:common.filter(x=>norm(x)>a).length,cleanMetric,metrics:Object.fromEntries(Object.entries(metrics).map(([k,v])=>[k,avg(v)]))})
 }}
 return {env:"V8 CPU, real public corpus; one clipped update on shared-shaped linear heads, not Transformer or SD execution; deterministic LCG only experiment, not production DP RNG",sizes:{clean:data.map(x=>x.length),public:pub.map(x=>x.length),train:train.map(x=>x.length),test:test.map(x=>x.length)},N,D,gamma,lr,initial:evaluate(init,init),norms:{common:common.map(norm),difference:diff.map(norm)},rows,summary:[.015,.05,.15,.3,.6].map(b=>({b,clipped:rows.find(x=>x.b===b).diffClipped,clean:rows.find(x=>x.b===b).cleanMetric,metrics:Object.fromEntries(Object.keys(rows[0].metrics).map(k=>[k,avg(rows.filter(x=>x.b===b).map(r=>r.metrics[k]))]))}))}
}
const fs=require('fs'); console.log(JSON.stringify(probe(process.argv.slice(2).map(p=>fs.readFileSync(p,'utf8'))),null,2));
~~~

~~~json
{
  "env": "V8 CPU, real public corpus; one clipped update on shared-shaped linear heads, not Transformer or SD execution; deterministic LCG only experiment, not production DP RNG",
  "sizes": {
    "clean": [
      30342,
      135138
    ],
    "public": [
      3034,
      13513
    ],
    "train": [
      21239,
      94597
    ],
    "test": [
      6069,
      27028
    ]
  },
  "N": 16,
  "D": 729,
  "gamma": 0.35,
  "lr": 4,
  "initial": {
    "accept": 0.7807552008280471,
    "targetNLL": 2.321859510956047,
    "draftNLL": 2.2726359574582866
  },
  "norms": {
    "common": [
      0.04262900502688573,
      0.040777549681303496,
      0.042717005715050625,
      0.046849592279985586,
      0.04494729108741929,
      0.04816266600998241,
      0.046705607400857004,
      0.04617957519301941,
      0.024891655187391545,
      0.02688848582480243,
      0.029442664897087273,
      0.02840302805833181,
      0.03083802117054358,
      0.03738513663882115,
      0.030100045604091945,
      0.04121182495829984
    ],
    "difference": [
      0.02432676550295363,
      0.024318968887562034,
      0.024759303890632578,
      0.024595914192732722,
      0.023239416094337228,
      0.02522933530689707,
      0.02272773593887039,
      0.02225792202082562,
      0.022480303842217244,
      0.022741589019607558,
      0.022490649306157153,
      0.02321160337740671,
      0.0221192967731123,
      0.024120738805225698,
      0.022974993034836085,
      0.022912249787487165
    ]
  },
  "rows": [
    {
      "b": 0.015,
      "seed": 11,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.9950124688279302,
      "marginalVariance": 0.0451125,
      "targetOnlyVar": 0.024806250000000002,
      "actualSensitivityBound": 1.9999999999999953,
      "diffClipped": 16,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.782972383540273,
        "targetNLL": 2.318851766855307,
        "draftNLL": 2.272196402298571
      },
      "metrics": {
        "independent": {
          "accept": 0.781058972358117,
          "targetNLL": 2.3202894270771535,
          "draftNLL": 2.2736140339948667
        },
        "paired": {
          "accept": 0.7828729316600843,
          "targetNLL": 2.3202894270771535,
          "draftNLL": 2.2736110331724233
        },
        "targetPostprocess": {
          "accept": 0.8273528346151675,
          "targetNLL": 2.3202894270771535,
          "draftNLL": 2.3094043060217255
        },
        "targetOnlyPostprocess": {
          "accept": 0.8273761948394925,
          "targetNLL": 2.3196762131519013,
          "draftNLL": 2.308820380021347
        }
      }
    },
    {
      "b": 0.015,
      "seed": 29,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.9950124688279302,
      "marginalVariance": 0.0451125,
      "targetOnlyVar": 0.024806250000000002,
      "actualSensitivityBound": 1.9999999999999953,
      "diffClipped": 16,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.782972383540273,
        "targetNLL": 2.318851766855307,
        "draftNLL": 2.272196402298571
      },
      "metrics": {
        "independent": {
          "accept": 0.7801623610136428,
          "targetNLL": 2.3201878724019305,
          "draftNLL": 2.273394853410901
        },
        "paired": {
          "accept": 0.78227451193242,
          "targetNLL": 2.3201878724019305,
          "draftNLL": 2.2732980061675656
        },
        "targetPostprocess": {
          "accept": 0.8268957362503851,
          "targetNLL": 2.3201878724019305,
          "draftNLL": 2.30910992797272
        },
        "targetOnlyPostprocess": {
          "accept": 0.8270446714218864,
          "targetNLL": 2.3195981027276873,
          "draftNLL": 2.3086009986444735
        }
      }
    },
    {
      "b": 0.015,
      "seed": 73,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.9950124688279302,
      "marginalVariance": 0.0451125,
      "targetOnlyVar": 0.024806250000000002,
      "actualSensitivityBound": 1.9999999999999953,
      "diffClipped": 16,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.782972383540273,
        "targetNLL": 2.318851766855307,
        "draftNLL": 2.272196402298571
      },
      "metrics": {
        "independent": {
          "accept": 0.7815355774839329,
          "targetNLL": 2.3203729466216685,
          "draftNLL": 2.2733478858248906
        },
        "paired": {
          "accept": 0.7830408249805294,
          "targetNLL": 2.3203729466216685,
          "draftNLL": 2.273751217486176
        },
        "targetPostprocess": {
          "accept": 0.8274784371607168,
          "targetNLL": 2.3203729466216685,
          "draftNLL": 2.309251455527743
        },
        "targetOnlyPostprocess": {
          "accept": 0.8274668296536368,
          "targetNLL": 2.3197393611707224,
          "draftNLL": 2.308708840123949
        }
      }
    },
    {
      "b": 0.05,
      "seed": 11,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.9459459459459459,
      "marginalVariance": 0.04625,
      "targetOnlyVar": 0.030624999999999996,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7821624112683162,
          "targetNLL": 2.319719664090089,
          "draftNLL": 2.2735635762028035
        },
        "paired": {
          "accept": 0.7839032049160707,
          "targetNLL": 2.319719664090089,
          "draftNLL": 2.2736364606331496
        },
        "targetPostprocess": {
          "accept": 0.8273776236474032,
          "targetNLL": 2.319719664090089,
          "draftNLL": 2.3089214677331173
        },
        "targetOnlyPostprocess": {
          "accept": 0.8273967581818682,
          "targetNLL": 2.3192502912722133,
          "draftNLL": 2.308476067781698
        }
      }
    },
    {
      "b": 0.05,
      "seed": 29,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.9459459459459459,
      "marginalVariance": 0.04625,
      "targetOnlyVar": 0.030624999999999996,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7812628076245107,
          "targetNLL": 2.319616692599211,
          "draftNLL": 2.273330034000809
        },
        "paired": {
          "accept": 0.7832371350278442,
          "targetNLL": 2.319616692599211,
          "draftNLL": 2.2732525388843063
        },
        "targetPostprocess": {
          "accept": 0.8269140114065103,
          "targetNLL": 2.319616692599211,
          "draftNLL": 2.3086231671196464
        },
        "targetOnlyPostprocess": {
          "accept": 0.827025437419053,
          "targetNLL": 2.3191642303198154,
          "draftNLL": 2.3082324555275116
        }
      }
    },
    {
      "b": 0.05,
      "seed": 73,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.9459459459459459,
      "marginalVariance": 0.04625,
      "targetOnlyVar": 0.030624999999999996,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7826422163962027,
          "targetNLL": 2.319815665471486,
          "draftNLL": 2.2732897198480138
        },
        "paired": {
          "accept": 0.7840840625795704,
          "targetNLL": 2.319815665471486,
          "draftNLL": 2.2736688000637857
        },
        "targetPostprocess": {
          "accept": 0.8275048957643865,
          "targetNLL": 2.319815665471486,
          "draftNLL": 2.30877662753223
        },
        "targetOnlyPostprocess": {
          "accept": 0.8274987581619454,
          "targetNLL": 2.319329372816302,
          "draftNLL": 2.308359646005299
        }
      }
    },
    {
      "b": 0.15,
      "seed": 11,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.6000000000000001,
      "marginalVariance": 0.056249999999999994,
      "targetOnlyVar": 0.05062499999999999,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7817971620499654,
          "targetNLL": 2.3200171798553657,
          "draftNLL": 2.2738461052954992
        },
        "paired": {
          "accept": 0.7830635667416218,
          "targetNLL": 2.3200171798553657,
          "draftNLL": 2.274003973195888
        },
        "targetPostprocess": {
          "accept": 0.8273639791057565,
          "targetNLL": 2.3200171798553657,
          "draftNLL": 2.3092010855983913
        },
        "targetOnlyPostprocess": {
          "accept": 0.8273716085931399,
          "targetNLL": 2.319850046710077,
          "draftNLL": 2.3090442140111236
        }
      }
    },
    {
      "b": 0.15,
      "seed": 29,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.6000000000000001,
      "marginalVariance": 0.056249999999999994,
      "targetOnlyVar": 0.05062499999999999,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7807978854466309,
          "targetNLL": 2.3199052616427918,
          "draftNLL": 2.273586578593677
        },
        "paired": {
          "accept": 0.7821600077274112,
          "targetNLL": 2.3199052616427918,
          "draftNLL": 2.2735325339682224
        },
        "targetPostprocess": {
          "accept": 0.8268490714520995,
          "targetNLL": 2.3199052616427918,
          "draftNLL": 2.308872701153216
        },
        "targetOnlyPostprocess": {
          "accept": 0.8268849995319801,
          "targetNLL": 2.3197430196577233,
          "draftNLL": 2.3087323796984314
        }
      }
    },
    {
      "b": 0.15,
      "seed": 73,
      "a": 0.3,
      "rho": 1,
      "correlation": 0.6000000000000001,
      "marginalVariance": 0.056249999999999994,
      "targetOnlyVar": 0.05062499999999999,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7823233933548117,
          "targetNLL": 2.3201223918138925,
          "draftNLL": 2.2735485659406716
        },
        "paired": {
          "accept": 0.7833393636756613,
          "targetNLL": 2.3201223918138925,
          "draftNLL": 2.27380474115185
        },
        "targetPostprocess": {
          "accept": 0.8275061011662972,
          "targetNLL": 2.3201223918138925,
          "draftNLL": 2.3090403303396845
        },
        "targetOnlyPostprocess": {
          "accept": 0.8275056785488152,
          "targetNLL": 2.3199501994861293,
          "draftNLL": 2.3088922374481293
        }
      }
    },
    {
      "b": 0.3,
      "seed": 11,
      "a": 0.3,
      "rho": 1,
      "correlation": 0,
      "marginalVariance": 0.09,
      "targetOnlyVar": 0.09,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7805934340698105,
          "targetNLL": 2.3210115717471105,
          "draftNLL": 2.2747843356919346
        },
        "paired": {
          "accept": 0.7805934340698105,
          "targetNLL": 2.3210115717471105,
          "draftNLL": 2.2747843356919346
        },
        "targetPostprocess": {
          "accept": 0.8273119377711472,
          "targetNLL": 2.3210115717471105,
          "draftNLL": 2.310126515241983
        },
        "targetOnlyPostprocess": {
          "accept": 0.8273119377711472,
          "targetNLL": 2.3210115717471105,
          "draftNLL": 2.310126515241983
        }
      }
    },
    {
      "b": 0.3,
      "seed": 29,
      "a": 0.3,
      "rho": 1,
      "correlation": 0,
      "marginalVariance": 0.09,
      "targetOnlyVar": 0.09,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7793141119288459,
          "targetNLL": 2.3208756245185618,
          "draftNLL": 2.2744491335623978
        },
        "paired": {
          "accept": 0.7793141119288459,
          "targetNLL": 2.3208756245185618,
          "draftNLL": 2.2744491335623978
        },
        "targetPostprocess": {
          "accept": 0.8266539715502854,
          "targetNLL": 2.3208756245185618,
          "draftNLL": 2.3097129248430406
        },
        "targetOnlyPostprocess": {
          "accept": 0.8266539715502854,
          "targetNLL": 2.3208756245185618,
          "draftNLL": 2.3097129248430406
        }
      }
    },
    {
      "b": 0.3,
      "seed": 73,
      "a": 0.3,
      "rho": 1,
      "correlation": 0,
      "marginalVariance": 0.09,
      "targetOnlyVar": 0.09,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7812430993600223,
          "targetNLL": 2.321142568780948,
          "draftNLL": 2.2744240252061436
        },
        "paired": {
          "accept": 0.7812430993600223,
          "targetNLL": 2.321142568780948,
          "draftNLL": 2.2744240252061436
        },
        "targetPostprocess": {
          "accept": 0.8274918288502291,
          "targetNLL": 2.321142568780948,
          "draftNLL": 2.309919764509543
        },
        "targetOnlyPostprocess": {
          "accept": 0.8274918288502291,
          "targetNLL": 2.321142568780948,
          "draftNLL": 2.309919764509543
        }
      }
    },
    {
      "b": 0.6,
      "seed": 11,
      "a": 0.3,
      "rho": 1,
      "correlation": 0,
      "marginalVariance": 0.22499999999999998,
      "targetOnlyVar": 0.20249999999999996,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7759303147345021,
          "targetNLL": 2.324924393475364,
          "draftNLL": 2.2784363955751337
        },
        "paired": {
          "accept": 0.7759303147345021,
          "targetNLL": 2.324924393475364,
          "draftNLL": 2.2784363955751337
        },
        "targetPostprocess": {
          "accept": 0.8270817086050858,
          "targetNLL": 2.324924393475364,
          "draftNLL": 2.3137076276770645
        },
        "targetOnlyPostprocess": {
          "accept": 0.8271216398915446,
          "targetNLL": 2.3242763284409986,
          "draftNLL": 2.313118274559737
        }
      }
    },
    {
      "b": 0.6,
      "seed": 29,
      "a": 0.3,
      "rho": 1,
      "correlation": 0,
      "marginalVariance": 0.22499999999999998,
      "targetOnlyVar": 0.20249999999999996,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7739107306180314,
          "targetNLL": 2.324730713450132,
          "draftNLL": 2.277878118429936
        },
        "paired": {
          "accept": 0.7739107306180314,
          "targetNLL": 2.324730713450132,
          "draftNLL": 2.277878118429936
        },
        "targetPostprocess": {
          "accept": 0.8260150162871828,
          "targetNLL": 2.324730713450132,
          "draftNLL": 2.313057539894872
        },
        "targetOnlyPostprocess": {
          "accept": 0.8261108909860044,
          "targetNLL": 2.324090051507253,
          "draftNLL": 2.312501368690984
        }
      }
    },
    {
      "b": 0.6,
      "seed": 73,
      "a": 0.3,
      "rho": 1,
      "correlation": 0,
      "marginalVariance": 0.22499999999999998,
      "targetOnlyVar": 0.20249999999999996,
      "actualSensitivityBound": 2,
      "diffClipped": 0,
      "commonClipped": 0,
      "cleanMetric": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7768128282232952,
          "targetNLL": 2.325125776600198,
          "draftNLL": 2.277936566363915
        },
        "paired": {
          "accept": 0.7768128282232952,
          "targetNLL": 2.325125776600198,
          "draftNLL": 2.277936566363915
        },
        "targetPostprocess": {
          "accept": 0.8273553599387021,
          "targetNLL": 2.325125776600198,
          "draftNLL": 2.3133689085888456
        },
        "targetOnlyPostprocess": {
          "accept": 0.8273816806869195,
          "targetNLL": 2.324467858946582,
          "draftNLL": 2.312798246433479
        }
      }
    }
  ],
  "summary": [
    {
      "b": 0.015,
      "clipped": 16,
      "clean": {
        "accept": 0.782972383540273,
        "targetNLL": 2.318851766855307,
        "draftNLL": 2.272196402298571
      },
      "metrics": {
        "independent": {
          "accept": 0.7809189702852309,
          "targetNLL": 2.3202834153669176,
          "draftNLL": 2.273452257743553
        },
        "paired": {
          "accept": 0.7827294228576779,
          "targetNLL": 2.3202834153669176,
          "draftNLL": 2.2735534189420545
        },
        "targetPostprocess": {
          "accept": 0.8272423360087564,
          "targetNLL": 2.3202834153669176,
          "draftNLL": 2.3092552298407294
        },
        "targetOnlyPostprocess": {
          "accept": 0.8272958986383386,
          "targetNLL": 2.3196712256834373,
          "draftNLL": 2.308710072929923
        }
      }
    },
    {
      "b": 0.05,
      "clipped": 0,
      "clean": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7820224784296766,
          "targetNLL": 2.319717340720262,
          "draftNLL": 2.273394443350542
        },
        "paired": {
          "accept": 0.7837414675078285,
          "targetNLL": 2.319717340720262,
          "draftNLL": 2.2735192665270807
        },
        "targetPostprocess": {
          "accept": 0.8272655102727667,
          "targetNLL": 2.319717340720262,
          "draftNLL": 2.3087737541283313
        },
        "targetOnlyPostprocess": {
          "accept": 0.8273069845876222,
          "targetNLL": 2.319247964802777,
          "draftNLL": 2.3083560564381695
        }
      }
    },
    {
      "b": 0.15,
      "clipped": 0,
      "clean": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7816394802838027,
          "targetNLL": 2.32001494443735,
          "draftNLL": 2.273660416609949
        },
        "paired": {
          "accept": 0.782854312714898,
          "targetNLL": 2.32001494443735,
          "draftNLL": 2.27378041610532
        },
        "targetPostprocess": {
          "accept": 0.8272397172413845,
          "targetNLL": 2.32001494443735,
          "draftNLL": 2.30903803903043
        },
        "targetOnlyPostprocess": {
          "accept": 0.8272540955579784,
          "targetNLL": 2.319847755284643,
          "draftNLL": 2.308889610385895
        }
      }
    },
    {
      "b": 0.3,
      "clipped": 0,
      "clean": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.7803835484528929,
          "targetNLL": 2.3210099216822067,
          "draftNLL": 2.274552498153492
        },
        "paired": {
          "accept": 0.7803835484528929,
          "targetNLL": 2.3210099216822067,
          "draftNLL": 2.274552498153492
        },
        "targetPostprocess": {
          "accept": 0.827152579390554,
          "targetNLL": 2.3210099216822067,
          "draftNLL": 2.3099197348648555
        },
        "targetOnlyPostprocess": {
          "accept": 0.827152579390554,
          "targetNLL": 2.3210099216822067,
          "draftNLL": 2.3099197348648555
        }
      }
    },
    {
      "b": 0.6,
      "clipped": 0,
      "clean": {
        "accept": 0.7841294140357565,
        "targetNLL": 2.3182522048578775,
        "draftNLL": 2.2721115821718816
      },
      "metrics": {
        "independent": {
          "accept": 0.775551291191943,
          "targetNLL": 2.3249269611752315,
          "draftNLL": 2.278083693456328
        },
        "paired": {
          "accept": 0.775551291191943,
          "targetNLL": 2.3249269611752315,
          "draftNLL": 2.278083693456328
        },
        "targetPostprocess": {
          "accept": 0.8268173616103236,
          "targetNLL": 2.3249269611752315,
          "draftNLL": 2.3133780253869274
        },
        "targetOnlyPostprocess": {
          "accept": 0.8268714038548229,
          "targetNLL": 2.324278079631611,
          "draftNLL": 2.3128059632280666
        }
      }
    }
  ]
}
~~~
