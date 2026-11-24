# R08 FIA-SD audit experiment
Input: HC/R05 chat.json, code.json, summarization.json at f60a62ce1b34b8822540ec7f3efc35111c5962f8.
Extract JS fence as probe.js. Run node probe.js /path/to/experiments/HC/R05

~~~javascript
// Run from R08: node probe.js /path/to/experiments/HC/R05
function probe(data){
 const N=data.length,sum=a=>a.reduce((s,v)=>s+v,0),ar=sum(data.map(x=>x.ar)),sd=sum(data.map(x=>x.sd)),bad=data.map(x=>!(x.equalText&&x.arTokens===x.sdTokens)),kbad=bad.filter(Boolean).length,alpha=.05;
 const logMiss=(N,k,m)=>{if(m>N-k)return -Infinity;let l=0;for(let j=0;j<m;j++)l+=Math.log1p(-k/(N-j));return l};
 const rows=[];let rng;const U=()=>{rng=(Math.imul(1664525,rng)+1013904223)>>>0;return(rng+.5)/4294967296};
 for(const weighting of ["equal","length"]){
 let raw=data.map(x=>weighting==="equal"?1:x.arTokens),total=sum(raw),w=raw.map(x=>x/total),order=w.map((v,i)=>i).sort((i,j)=>w[j]-w[i]);
 for(const tau of [.05,.1,.2]){
 let k=0,v=0;while(v<tau-1e-14){v+=w[order[k++]]}
 let m=0;while(logMiss(N,k,m)>Math.log(alpha))m++;
 let d=w.map(x=>-Math.expm1(Math.log(alpha)*x/tau)),missObsW=Math.exp(Math.log(alpha)*sum(w.filter((x,i)=>bad[i]))/tau),missObsU=Math.exp(logMiss(N,kbad,m));
 const baseCostW=sum(d.map((x,i)=>x*data[i].ar)),baseCostU=m/N*ar;
 const failReplayW=sum(data.map((x,i)=>x.ar*(1-missObsW*(bad[i]?1:1-d[i]))));
 const missObsUgood=Math.exp(logMiss(N,kbad+1,m)),failReplayU=sum(data.map((x,i)=>x.ar*(1-(bad[i]?missObsU:missObsUgood))));
 const adversary=new Set(order.slice(0,k)),missW=Math.exp(Math.log(alpha)*v/tau),missU=Math.exp(logMiss(N,k,m));
 let mc=[];
 for(const seed of [11,29,73]){
 rng=seed;let nu=0,nw=0,n=20000;
 for(let rep=0;rep<n;rep++){
 let hit=false;for(let i of adversary)if(U()<d[i])hit=true;if(!hit)nw++;
 let ids=Array.from({length:N},(_,i)=>i);hit=false;for(let j=0;j<m;j++){let l=j+Math.floor(U()*(N-j));[ids[j],ids[l]]=[ids[l],ids[j]];if(adversary.has(ids[j]))hit=true}if(!hit)nu++;
 }
 mc.push({seed,n,weightedMiss:nw/n,uniformMiss:nu/n})
 }
 rows.push({weighting,tau,minCorruptedCount:k,worstTopMass:v,observedMismatchCount:kbad,observedMismatchWeight:sum(w.filter((x,i)=>bad[i])),uniform:{m,guaranteedMiss:missU,observedMiss:missObsU,expectedAuditCostMs:baseCostU,sdPlusAuditOverAr:(sd+baseCostU)/ar,sdPlusCohortFallbackOverAr:(sd+failReplayU)/ar},weighted:{expectedAudits:sum(d),guaranteedMiss:alpha,topSetMiss:missW,observedMiss:missObsW,expectedAuditCostMs:baseCostW,sdPlusAuditOverAr:(sd+baseCostW)/ar,sdPlusCohortFallbackOverAr:(sd+failReplayW)/ar},mc})
 }}
 return {environment:"V8 CPU replay/MC on public log summaries; no actual training, proof, hardware, adversary, or GPU",N,ar,sd,kbad,groups:["chat","code","summarization"].map(task=>({task,n:data.filter(x=>x.task===task).length,mismatches:data.filter(x=>x.task===task&&!(x.equalText&&x.arTokens===x.sdTokens)).length,arTokens:sum(data.filter(x=>x.task===task).map(x=>x.arTokens))})),rows}
}
const fs=require('fs'),path=require('path');const data=['chat','code','summarization'].flatMap(n=>JSON.parse(fs.readFileSync(path.join(process.argv[2],n+'.json'),'utf8')).pairs);console.log(JSON.stringify(probe(data),null,2));
~~~

~~~json
{
  "environment": "V8 CPU replay/MC on public log summaries; no actual training, proof, hardware, adversary, or GPU",
  "N": 72,
  "ar": 951128.11,
  "sd": 406852.20000000007,
  "kbad": 36,
  "groups": [
    {
      "task": "chat",
      "n": 24,
      "mismatches": 17,
      "arTokens": 6144
    },
    {
      "task": "code",
      "n": 24,
      "mismatches": 9,
      "arTokens": 6144
    },
    {
      "task": "summarization",
      "n": 24,
      "mismatches": 10,
      "arTokens": 2775
    }
  ],
  "rows": [
    {
      "weighting": "equal",
      "tau": 0.05,
      "minCorruptedCount": 4,
      "worstTopMass": 0.05555555555555555,
      "observedMismatchCount": 36,
      "observedMismatchWeight": 0.5000000000000001,
      "uniform": {
        "m": 38,
        "guaranteedMiss": 0.0450781986605624,
        "observedMiss": 0,
        "expectedAuditCostMs": 501984.28027777775,
        "sdPlusAuditOverAr": 0.9555352961629721,
        "sdPlusCohortFallbackOverAr": 1.4277575183851943
      },
      "weighted": {
        "expectedAudits": 40.67181206580444,
        "guaranteedMiss": 0.05,
        "topSetMiss": 0.03584355822184433,
        "observedMiss": 9.765624999999988e-14,
        "expectedAuditCostMs": 537279.2186169965,
        "sdPlusAuditOverAr": 0.9926437970769223,
        "sdPlusCohortFallbackOverAr": 1.4277575183851228
      },
      "mc": [
        {
          "seed": 11,
          "n": 20000,
          "weightedMiss": 0.03575,
          "uniformMiss": 0.0432
        },
        {
          "seed": 29,
          "n": 20000,
          "weightedMiss": 0.0374,
          "uniformMiss": 0.0459
        },
        {
          "seed": 73,
          "n": 20000,
          "weightedMiss": 0.0365,
          "uniformMiss": 0.04555
        }
      ]
    },
    {
      "weighting": "equal",
      "tau": 0.1,
      "minCorruptedCount": 8,
      "worstTopMass": 0.11111111111111113,
      "observedMismatchCount": 36,
      "observedMismatchWeight": 0.5000000000000001,
      "uniform": {
        "m": 22,
        "guaranteedMiss": 0.04485570363718975,
        "observedMiss": 2.11936514999842e-9,
        "expectedAuditCostMs": 290622.47805555555,
        "sdPlusAuditOverAr": 0.7333130739407497,
        "sdPlusCohortFallbackOverAr": 1.4277575168851375
      },
      "weighted": {
        "expectedAudits": 24.50653169895801,
        "guaranteedMiss": 0.05,
        "topSetMiss": 0.03584355822184431,
        "observedMiss": 3.124999999999998e-7,
        "expectedAuditCostMs": 323734.04413173633,
        "sdPlusAuditOverAr": 0.7681260142040555,
        "sdPlusCohortFallbackOverAr": 1.427757256745654
      },
      "mc": [
        {
          "seed": 11,
          "n": 20000,
          "weightedMiss": 0.03395,
          "uniformMiss": 0.04655
        },
        {
          "seed": 29,
          "n": 20000,
          "weightedMiss": 0.0356,
          "uniformMiss": 0.04415
        },
        {
          "seed": 73,
          "n": 20000,
          "weightedMiss": 0.0365,
          "uniformMiss": 0.0437
        }
      ]
    },
    {
      "weighting": "equal",
      "tau": 0.2,
      "minCorruptedCount": 15,
      "worstTopMass": 0.2083333333333334,
      "observedMismatchCount": 36,
      "observedMismatchWeight": 0.5000000000000001,
      "uniform": {
        "m": 12,
        "guaranteedMiss": 0.04603739074938355,
        "observedMiss": 0.00008147201310864084,
        "expectedAuditCostMs": 158521.35166666665,
        "sdPlusAuditOverAr": 0.594424185051861,
        "sdPlusCohortFallbackOverAr": 1.4276890321537585
      },
      "weighted": {
        "expectedAudits": 13.52325489842116,
        "guaranteedMiss": 0.05,
        "topSetMiss": 0.04413269219225253,
        "observedMiss": 0.0005590169943749473,
        "expectedAuditCostMs": 178643.7204525496,
        "sdPlusAuditOverAr": 0.6155805030854884,
        "sdPlusCohortFallbackOverAr": 1.4272487072787807
      },
      "mc": [
        {
          "seed": 11,
          "n": 20000,
          "weightedMiss": 0.0413,
          "uniformMiss": 0.0457
        },
        {
          "seed": 29,
          "n": 20000,
          "weightedMiss": 0.0454,
          "uniformMiss": 0.04855
        },
        {
          "seed": 73,
          "n": 20000,
          "weightedMiss": 0.04475,
          "uniformMiss": 0.0432
        }
      ]
    },
    {
      "weighting": "length",
      "tau": 0.05,
      "minCorruptedCount": 3,
      "worstTopMass": 0.05098585939055965,
      "observedMismatchCount": 36,
      "observedMismatchWeight": 0.5222731195644957,
      "uniform": {
        "m": 45,
        "guaranteedMiss": 0.04904426559356137,
        "observedMiss": 0,
        "expectedAuditCostMs": 594455.06875,
        "sdPlusAuditOverAr": 1.0527575183851943,
        "sdPlusCohortFallbackOverAr": 1.4277575183851943
      },
      "weighted": {
        "expectedAudits": 39.487772517869615,
        "guaranteedMiss": 0.05,
        "topSetMiss": 0.04713216089512877,
        "observedMiss": 2.5712251673730216e-14,
        "expectedAuditCostMs": 559876.0049543785,
        "sdPlusAuditOverAr": 1.0164016758524554,
        "sdPlusCohortFallbackOverAr": 1.4277575183851756
      },
      "mc": [
        {
          "seed": 11,
          "n": 20000,
          "weightedMiss": 0.04795,
          "uniformMiss": 0.04765
        },
        {
          "seed": 29,
          "n": 20000,
          "weightedMiss": 0.0459,
          "uniformMiss": 0.04855
        },
        {
          "seed": 73,
          "n": 20000,
          "weightedMiss": 0.0461,
          "uniformMiss": 0.0471
        }
      ]
    },
    {
      "weighting": "length",
      "tau": 0.1,
      "minCorruptedCount": 6,
      "worstTopMass": 0.10197171878111931,
      "observedMismatchCount": 36,
      "observedMismatchWeight": 0.5222731195644957,
      "uniform": {
        "m": 28,
        "guaranteedMiss": 0.045181140154922195,
        "observedMiss": 4.005143603142251e-13,
        "expectedAuditCostMs": 369883.1538888889,
        "sdPlusAuditOverAr": 0.8166464072740831,
        "sdPlusCohortFallbackOverAr": 1.427757518384943
      },
      "weighted": {
        "expectedAudits": 24.07485813698313,
        "guaranteedMiss": 0.05,
        "topSetMiss": 0.04713216089512877,
        "observedMiss": 1.603504027863049e-7,
        "expectedAuditCostMs": 345315.46801497095,
        "sdPlusAuditOverAr": 0.7908163580771166,
        "sdPlusCohortFallbackOverAr": 1.4277573852877965
      },
      "mc": [
        {
          "seed": 11,
          "n": 20000,
          "weightedMiss": 0.04775,
          "uniformMiss": 0.0461
        },
        {
          "seed": 29,
          "n": 20000,
          "weightedMiss": 0.0499,
          "uniformMiss": 0.04675
        },
        {
          "seed": 73,
          "n": 20000,
          "weightedMiss": 0.04635,
          "uniformMiss": 0.0452
        }
      ]
    },
    {
      "weighting": "length",
      "tau": 0.2,
      "minCorruptedCount": 12,
      "worstTopMass": 0.20394343756223862,
      "observedMismatchCount": 36,
      "observedMismatchWeight": 0.5222731195644957,
      "uniform": {
        "m": 15,
        "guaranteedMiss": 0.046037390749383574,
        "observedMiss": 0.00000481880054155141,
        "expectedAuditCostMs": 198151.68958333333,
        "sdPlusAuditOverAr": 0.6360908517185276,
        "sdPlusCohortFallbackOverAr": 1.427753659667284
      },
      "weighted": {
        "expectedAudits": 13.392747306393805,
        "guaranteedMiss": 0.05,
        "topSetMiss": 0.04713216089512877,
        "observedMiss": 0.0004004377639363012,
        "expectedAuditCostMs": 193259.78607456072,
        "sdPlusAuditOverAr": 0.6309475871494964,
        "sdPlusCohortFallbackOverAr": 1.4273951213427467
      },
      "mc": [
        {
          "seed": 11,
          "n": 20000,
          "weightedMiss": 0.0455,
          "uniformMiss": 0.0448
        },
        {
          "seed": 29,
          "n": 20000,
          "weightedMiss": 0.0478,
          "uniformMiss": 0.0471
        },
        {
          "seed": 73,
          "n": 20000,
          "weightedMiss": 0.04865,
          "uniformMiss": 0.04725
        }
      ]
    }
  ]
}
~~~
