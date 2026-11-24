# R10 public checkpoint actual run
Extract JS as probe.js; download weights.json, checkpoint.json, input.txt from pinned sanchayan7432/microGPT; node probe.js weights.json checkpoint.json input.txt.

~~~javascript
function probe(weights,docs){
 const D=16,V=27,BOS=26,mat=(r,c)=>Array.from({length:r},()=>Array(c).fill(0)),transpose=a=>a[0].map((_,i)=>a.map(x=>x[i])),mm=(a,b)=>a.map(r=>b[0].map((_,j)=>r.reduce((s,x,k)=>s+x*b[k][j],0))),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),add=(a,b)=>a.map((x,i)=>x+b[i]),rms=x=>{let d=Math.sqrt(dot(x,x)/D+1e-5);return x.map(v=>v/d)},soft=x=>{let m=Math.max(...x),a=x.map(v=>Math.exp(v-m)),s=a.reduce((s,v)=>s+v,0);return a.map(v=>v/s)},lin=(w,x)=>w.map(r=>dot(r,x)),sum=a=>a.reduce((s,v)=>s+v,0),clone=a=>a.map(r=>r.slice()),maxdiff=(a,b)=>Math.max(...a.flatMap((r,i)=>r.map((x,j)=>Math.abs(x-b[i][j]))));
 const solve=(a,b)=>{let n=a.length,k=b[0].length,m=a.map((r,i)=>r.concat(b[i]));for(let i=0;i<n;i++){let p=i;for(let j=i+1;j<n;j++)if(Math.abs(m[j][i])>Math.abs(m[p][i]))p=j;[m[i],m[p]]=[m[p],m[i]];let z=m[i][i];if(Math.abs(z)<1e-14)throw Error("singular");for(let j=i;j<n+k;j++)m[i][j]/=z;for(let j=0;j<n;j++)if(j!==i){let c=m[j][i];for(let l=i;l<n+k;l++)m[j][l]-=c*m[i][l]}}return m.map(r=>r.slice(n))};
 function build(w){let at=0;const take=(r,c)=>Array.from({length:r},()=>{let a=w.slice(at,at+c);at+=c;return a});let e=take(V,D),pos=take(16,D),head=transpose(take(V,D)),layers=Array.from({length:2},()=>({q:take(D,D),k:take(D,D),v:take(D,D),o:take(D,D),a:take(4*D,D),b:take(D,4*D)}));if(at!==w.length)throw Error("parameter count");return {e,pos,head,layers}}
 function forward(model,ids,depth=2){let keys=[[],[]],values=[[],[]],out=[];for(let i=0;i<ids.length;i++){let x=rms(add(model.e[ids[i]],model.pos[i])),sh;for(let l=0;l<depth;l++){let w=model.layers[l],z=rms(x),q=lin(w.q,z),k=lin(w.k,z),v=lin(w.v,z);keys[l].push(k);values[l].push(v);let att=[];for(let h=0;h<4;h++){let from=h*4,to=from+4,p=soft(keys[l].map(kt=>dot(q.slice(from,to),kt.slice(from,to))/2));for(let a=from;a<to;a++)att.push(sum(p.map((p,j)=>p*values[l][j][a])))}x=add(x,lin(w.o,att));x=add(x,lin(w.b,lin(w.a,rms(x)).map(v=>Math.max(0,v))));if(l===0)sh=x.slice()}out.push({sh,deep:x})}return out}
 const logits=(h,w)=>w[0].map((_,j)=>h.reduce((s,v,i)=>s+v*w[i][j],0)),id=c=>c.charCodeAt(0)-97;
 const hash=s=>{let h=2166136261;for(let c of s)h=Math.imul(h^c.charCodeAt(0),16777619);return h>>>0};
 const unique=[...new Set(docs)].filter(x=>/^[a-z]+$/.test(x)&&x.length<=15),groups=Array.from({length:4},(_,i)=>unique.filter(x=>id(x[0])%4===i)),split=groups.map(g=>({train:g.filter(x=>hash(x)%5!==0).sort((a,b)=>hash(a)-hash(b)).slice(0,128),test:g.filter(x=>hash(x)%5===0).sort((a,b)=>hash(a)-hash(b)).slice(0,64)}));
 function features(model,names){return names.flatMap(n=>{let y=[...n].map(id).concat(BOS),x=[BOS,...y.slice(0,-1)],f=forward(model,x);return f.map((v,i)=>({...v,y:y[i]}))})}
 function stats(a){let G=mat(D,D),H=mat(D,D);for(let {sh,deep} of a)for(let i=0;i<D;i++)for(let j=0;j<D;j++){G[i][j]+=sh[i]*sh[j];H[i][j]+=sh[i]*deep[j]}return {G,H,n:a.length}}
 function aggregate(s){return {G:s[0].G.map((r,i)=>r.map((_,j)=>sum(s.map(x=>x.G[i][j])))),H:s[0].H.map((r,i)=>r.map((_,j)=>sum(s.map(x=>x.H[i][j])))),n:sum(s.map(x=>x.n))}}
 function map(s){let a=s.G.map((r,i)=>r.map((v,j)=>v/s.n+(i===j?.01:0))),b=s.H.map(r=>r.map(v=>v/s.n));return solve(a,b)}
 function ceStep(w,a,lr,teacher){let g=mat(D,V);for(let f of a){let x=teacher?f.sh:f.deep,p=soft(logits(x,w)),t=teacher?soft(logits(f.deep,teacher)):Array.from({length:V},(_,j)=>Number(f.y===j));for(let i=0;i<D;i++)for(let j=0;j<V;j++)g[i][j]+=x[i]*(p[j]-t[j])}return w.map((r,i)=>r.map((v,j)=>v-lr*g[i][j]/a.length))}
 function metric(a,p,q){let n=a.length,ov=0,nll=0,kl=0;for(let f of a){let P=soft(logits(f.deep,p)),Q=soft(logits(f.sh,q));ov+=sum(P.map((v,j)=>Math.min(v,Q[j])));nll-=Math.log(P[f.y]);kl+=sum(P.map((v,j)=>v*Math.log(v/Math.max(Q[j],1e-300))))}return {overlap:ov/n,nll:nll/n,kl:kl/n}}
 let state;const u=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return(state+.5)/4294967296};const draw=p=>{let r=u();for(let j=0;j<p.length;j++){r-=p[j];if(r<0)return j}return p.length-1};const arg=p=>p.indexOf(Math.max(...p));
 function generate(model,p,q,prompt,greedy=false,sd=true){let x=prompt.slice(),out=[],proposed=0,accepted=0,attempted=0,blocks=0,shallowPositions=0,fullPositions=0;let limit=8;while(out.length<limit&&out.at(-1)!==BOS){if(!sd){let P=soft(logits(forward(model,x).at(-1).deep,p));fullPositions+=x.length;let y=greedy?arg(P):draw(P);x.push(y);out.push(y);blocks++;continue}let k=Math.min(3,limit-out.length),draft=[],qs=[],ctx=x.slice();for(let j=0;j<k;j++){let Q=soft(logits(forward(model,ctx,1).at(-1).sh,q));shallowPositions+=ctx.length;let y=greedy?arg(Q):draw(Q);draft.push(y);qs.push(Q);ctx.push(y);if(y===BOS)break}proposed+=draft.length;let ff=forward(model,ctx.slice(0,-1));fullPositions+=ctx.length-1;blocks++;let all=true;for(let j=0;j<draft.length;j++){let P=soft(logits(ff[x.length-1].deep,p)),Q=qs[j],y=draft[j];attempted++;let ok=greedy?y===arg(P):u()<Math.min(1,P[y]/Q[y]);if(ok){accepted++;x.push(y);out.push(y)}else{let r=P.map((v,i)=>Math.max(0,v-Q[i])),z=sum(r);let y2=greedy?arg(P):draw(r.map(v=>v/z));x.push(y2);out.push(y2);all=false;break}if(out.length===limit||y===BOS){all=false;break}}if(all&&out.length<limit&&out.at(-1)!==BOS){let P=soft(logits(forward(model,x).at(-1).deep,p));fullPositions+=x.length;blocks++;let y=greedy?arg(P):draw(P);x.push(y);out.push(y)}}return {out,proposed,accepted,attempted,blocks,shallowPositions,fullPositions}}
 let results=[];
 for(let [name,w] of weights){let model=build(w),train=split.map(g=>features(model,g.train)),test=split.map(g=>features(model,g.test)),all=train.flat(),evals=test.flat(),s=train.map(stats),gs=aggregate(s),M=map(gs),Mdirect=map(stats(all)),maps=[map(aggregate(s.slice(0,2))),map(aggregate(s.slice(2)))],initial=clone(model.head),p=clone(initial);
 let rows=[];for(let epoch=0;epoch<4;epoch++){let regions=[];for(let e=0;e<2;e++){let local=clone(p);for(let step=0;step<2;step++){let a=ceStep(local,train[e*2],.025),b=ceStep(local,train[e*2+1],.025);local=a.map((r,i)=>r.map((v,j)=>(v+b[i][j])/2))}regions.push(local)}p=regions[0].map((r,i)=>r.map((v,j)=>(v+regions[1][i][j])/2));let q=mm(M,p),qstale=mm(M,initial),direct=mm(Mdirect,p);
 // Fresh logit regression (recompute features from exact same raw names and form teacher logits explicitly).
 let fresh=split.flatMap(g=>features(model,g.train)),rhs=mat(D,V);for(let f of fresh){let z=logits(f.deep,p);for(let i=0;i<D;i++)for(let j=0;j<V;j++)rhs[i][j]+=f.sh[i]*z[j]/fresh.length}let qfresh=solve(gs.G.map((r,i)=>r.map((v,j)=>v/gs.n+(i===j?.01:0))),rhs);
 let qkl=clone(q);for(let j=0;j<50;j++)qkl=ceStep(qkl,all,.015,p);
 let qm=maps.map(x=>mm(x,p)),qavg=qm[0].map((r,i)=>r.map((v,j)=>(v+qm[1][i][j])/2));
 rows.push({epoch:epoch+1,headNLL:metric(evals,p,p).nll,tied:metric(evals,p,p),stale:metric(evals,p,qstale),closed:metric(evals,p,q),fresh:metric(evals,p,qfresh),kl:metric(evals,p,qkl),meanLocal:metric(evals,p,qavg),statsError:maxdiff(M,Mdirect),freshError:maxdiff(q,qfresh),directError:maxdiff(q,direct)});
 if(epoch===3){let gen=[];for(let seed of [11,29,73])for(let [method,qh] of [["tied",p],["stale",qstale],["closed",q],["kl",qkl]]){state=seed;let g={seed,method,requests:0,generated:0,proposed:0,accepted:0,attempted:0,blocks:0,shallowPositions:0,fullPositions:0,greedyMismatch:0,arPositions:0};for(let group of split)for(let n of group.test.slice(0,8)){let prompt=[BOS,id(n[0])],a=generate(model,p,qh,prompt,false,true),b=generate(model,p,qh,prompt,true,true),c=generate(model,p,qh,prompt,true,false);g.requests++;g.generated+=a.out.length;for(let k of ["proposed","accepted","attempted","blocks","shallowPositions","fullPositions"])g[k]+=a[k];g.arPositions+=c.fullPositions;g.greedyMismatch+=Number(b.out.join()!==c.out.join())}gen.push(g)}results.push({name,trainPositions:train.map(a=>a.length),testPositions:test.map(a=>a.length),initial:metric(evals,initial,initial),rows,gen})}}
 }
 return {env:"V8 CPU, loaded external JSON pretrained weights, manually transcribed inference, frozen backbone head-only CE local updates; no actual GPU, no optimized KV/batched verifier, no network; original training corpus split can overlap pretraining",sourceNames:docs.length,uniqueNames:unique.length,split:split.map(x=>({train:x.train.length,test:x.test.length})),results}
}
const fs=require('fs');const w=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),c=JSON.parse(fs.readFileSync(process.argv[3],'utf8')),d=fs.readFileSync(process.argv[4],'utf8').split(/\r?\n/).map(x=>x.trim()).filter(Boolean);console.log(JSON.stringify(probe([['weights.json',w],['checkpoint.json',c.weights]],d),null,2));
~~~

~~~json
{
  "env": "V8 CPU, loaded external JSON pretrained weights, manually transcribed inference, frozen backbone head-only CE local updates; no actual GPU, no optimized KV/batched verifier, no network; original training corpus split can overlap pretraining",
  "sourceNames": 32033,
  "uniqueNames": 29494,
  "split": [
    {
      "train": 128,
      "test": 64
    },
    {
      "train": 128,
      "test": 64
    },
    {
      "train": 128,
      "test": 64
    },
    {
      "train": 128,
      "test": 64
    }
  ],
  "results": [
    {
      "name": "weights.json",
      "trainPositions": [
        925,
        929,
        961,
        881
      ],
      "testPositions": [
        476,
        466,
        470,
        448
      ],
      "initial": {
        "overlap": 0.9009632917444139,
        "nll": 2.2631943812828843,
        "kl": 0.04037332107520345
      },
      "rows": [
        {
          "epoch": 1,
          "headNLL": 2.261985779678354,
          "tied": {
            "overlap": 0.901004550976859,
            "nll": 2.261985779678354,
            "kl": 0.040347621492046695
          },
          "stale": {
            "overlap": 0.922244061635857,
            "nll": 2.261985779678354,
            "kl": 0.029908460729818166
          },
          "closed": {
            "overlap": 0.9223665749147093,
            "nll": 2.261985779678354,
            "kl": 0.029896623848777327
          },
          "fresh": {
            "overlap": 0.9223665749147093,
            "nll": 2.261985779678354,
            "kl": 0.02989662384877725
          },
          "kl": {
            "overlap": 0.9229610207053904,
            "nll": 2.261985779678354,
            "kl": 0.02963987205262947
          },
          "meanLocal": {
            "overlap": 0.9220753144124748,
            "nll": 2.261985779678354,
            "kl": 0.030017502733133066
          },
          "statsError": 9.41226263595496e-14,
          "freshError": 3.1796787425264483e-13,
          "directError": 1.4166445794216997e-13
        },
        {
          "epoch": 2,
          "headNLL": 2.2608620942285333,
          "tied": {
            "overlap": 0.9010470932824077,
            "nll": 2.2608620942285333,
            "kl": 0.040321373584586186
          },
          "stale": {
            "overlap": 0.9220179465640432,
            "nll": 2.2608620942285333,
            "kl": 0.02992803856990294
          },
          "closed": {
            "overlap": 0.9224061084615592,
            "nll": 2.2608620942285333,
            "kl": 0.029865137077057416
          },
          "fresh": {
            "overlap": 0.9224061084615598,
            "nll": 2.2608620942285333,
            "kl": 0.029865137077057506
          },
          "kl": {
            "overlap": 0.9230008340072048,
            "nll": 2.2608620942285333,
            "kl": 0.029608971377254312
          },
          "meanLocal": {
            "overlap": 0.9221171130972189,
            "nll": 2.2608620942285333,
            "kl": 0.029985485996383742
          },
          "statsError": 9.41226263595496e-14,
          "freshError": 2.51909604287448e-13,
          "directError": 1.4233059175694507e-13
        },
        {
          "epoch": 3,
          "headNLL": 2.2598158923566265,
          "tied": {
            "overlap": 0.9010903147655847,
            "nll": 2.2598158923566265,
            "kl": 0.04029462724909183
          },
          "stale": {
            "overlap": 0.9217223282746436,
            "nll": 2.2598158923566265,
            "kl": 0.02998187683964678
          },
          "closed": {
            "overlap": 0.9224464298456339,
            "nll": 2.2598158923566265,
            "kl": 0.029833873831063517
          },
          "fresh": {
            "overlap": 0.9224464298456327,
            "nll": 2.2598158923566265,
            "kl": 0.029833873831063777
          },
          "kl": {
            "overlap": 0.923041158931144,
            "nll": 2.2598158923566265,
            "kl": 0.02957825078538705
          },
          "meanLocal": {
            "overlap": 0.9221594580439697,
            "nll": 2.2598158923566265,
            "kl": 0.0299537114525028
          },
          "statsError": 9.41226263595496e-14,
          "freshError": 3.304023721284466e-13,
          "directError": 1.4210854715202004e-13
        },
        {
          "epoch": 4,
          "headNLL": 2.2588404698635323,
          "tied": {
            "overlap": 0.9011341462924952,
            "nll": 2.2588404698635323,
            "kl": 0.04026743244134936
          },
          "stale": {
            "overlap": 0.9213469572355792,
            "nll": 2.2588404698635323,
            "kl": 0.030065404564521503
          },
          "closed": {
            "overlap": 0.9224871784623659,
            "nll": 2.2588404698635323,
            "kl": 0.029802849105653416
          },
          "fresh": {
            "overlap": 0.922487178462366,
            "nll": 2.2588404698635323,
            "kl": 0.029802849105653718
          },
          "kl": {
            "overlap": 0.92308178580213,
            "nll": 2.2588404698635323,
            "kl": 0.02954772992244412
          },
          "meanLocal": {
            "overlap": 0.9222017058264361,
            "nll": 2.2588404698635323,
            "kl": 0.029922193141694805
          },
          "statsError": 9.41226263595496e-14,
          "freshError": 3.644862189844389e-13,
          "directError": 1.425526363618701e-13
        }
      ],
      "gen": [
        {
          "seed": 11,
          "method": "tied",
          "requests": 32,
          "generated": 193,
          "proposed": 185,
          "accepted": 145,
          "attempted": 167,
          "blocks": 100,
          "shallowPositions": 858,
          "fullPositions": 571,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 11,
          "method": "stale",
          "requests": 32,
          "generated": 189,
          "proposed": 177,
          "accepted": 145,
          "attempted": 161,
          "blocks": 98,
          "shallowPositions": 806,
          "fullPositions": 548,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 11,
          "method": "closed",
          "requests": 32,
          "generated": 192,
          "proposed": 183,
          "accepted": 147,
          "attempted": 164,
          "blocks": 99,
          "shallowPositions": 824,
          "fullPositions": 552,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 11,
          "method": "kl",
          "requests": 32,
          "generated": 189,
          "proposed": 178,
          "accepted": 144,
          "attempted": 159,
          "blocks": 100,
          "shallowPositions": 809,
          "fullPositions": 564,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 29,
          "method": "tied",
          "requests": 32,
          "generated": 204,
          "proposed": 185,
          "accepted": 151,
          "attempted": 171,
          "blocks": 105,
          "shallowPositions": 845,
          "fullPositions": 611,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 29,
          "method": "stale",
          "requests": 32,
          "generated": 193,
          "proposed": 178,
          "accepted": 147,
          "attempted": 164,
          "blocks": 99,
          "shallowPositions": 807,
          "fullPositions": 555,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 29,
          "method": "closed",
          "requests": 32,
          "generated": 196,
          "proposed": 187,
          "accepted": 146,
          "attempted": 165,
          "blocks": 104,
          "shallowPositions": 852,
          "fullPositions": 592,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 29,
          "method": "kl",
          "requests": 32,
          "generated": 201,
          "proposed": 184,
          "accepted": 149,
          "attempted": 166,
          "blocks": 108,
          "shallowPositions": 854,
          "fullPositions": 623,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 73,
          "method": "tied",
          "requests": 32,
          "generated": 190,
          "proposed": 175,
          "accepted": 146,
          "attempted": 161,
          "blocks": 98,
          "shallowPositions": 789,
          "fullPositions": 543,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 73,
          "method": "stale",
          "requests": 32,
          "generated": 186,
          "proposed": 167,
          "accepted": 150,
          "attempted": 160,
          "blocks": 92,
          "shallowPositions": 750,
          "fullPositions": 502,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 73,
          "method": "closed",
          "requests": 32,
          "generated": 189,
          "proposed": 177,
          "accepted": 149,
          "attempted": 162,
          "blocks": 96,
          "shallowPositions": 803,
          "fullPositions": 530,
          "greedyMismatch": 0,
          "arPositions": 662
        },
        {
          "seed": 73,
          "method": "kl",
          "requests": 32,
          "generated": 189,
          "proposed": 177,
          "accepted": 149,
          "attempted": 162,
          "blocks": 96,
          "shallowPositions": 803,
          "fullPositions": 530,
          "greedyMismatch": 0,
          "arPositions": 662
        }
      ]
    },
    {
      "name": "checkpoint.json",
      "trainPositions": [
        925,
        929,
        961,
        881
      ],
      "testPositions": [
        476,
        466,
        470,
        448
      ],
      "initial": {
        "overlap": 0.9085704172547467,
        "nll": 2.2814172732088362,
        "kl": 0.03174044316459355
      },
      "rows": [
        {
          "epoch": 1,
          "headNLL": 2.2807284787607194,
          "tied": {
            "overlap": 0.9085048627266806,
            "nll": 2.2807284787607194,
            "kl": 0.031770439022349724
          },
          "stale": {
            "overlap": 0.9340208600470689,
            "nll": 2.2807284787607194,
            "kl": 0.022027211708854046
          },
          "closed": {
            "overlap": 0.9340107757480912,
            "nll": 2.2807284787607194,
            "kl": 0.022025278211606412
          },
          "fresh": {
            "overlap": 0.9340107757480918,
            "nll": 2.2807284787607194,
            "kl": 0.022025278211606464
          },
          "kl": {
            "overlap": 0.9341352562410431,
            "nll": 2.2807284787607194,
            "kl": 0.02182359974101379
          },
          "meanLocal": {
            "overlap": 0.9339761455182969,
            "nll": 2.2807284787607194,
            "kl": 0.02198780352541529
          },
          "statsError": 6.639133687258436e-14,
          "freshError": 2.1715962361668062e-13,
          "directError": 1.51684220739412e-13
        },
        {
          "epoch": 2,
          "headNLL": 2.2800607597705933,
          "tied": {
            "overlap": 0.9084408286225121,
            "nll": 2.2800607597705933,
            "kl": 0.03179973508769425
          },
          "stale": {
            "overlap": 0.9338680988863332,
            "nll": 2.2800607597705933,
            "kl": 0.022055257436485363
          },
          "closed": {
            "overlap": 0.9339845537978095,
            "nll": 2.2800607597705933,
            "kl": 0.022040766644341757
          },
          "fresh": {
            "overlap": 0.9339845537978102,
            "nll": 2.2800607597705933,
            "kl": 0.022040766644341705
          },
          "kl": {
            "overlap": 0.9341080215781465,
            "nll": 2.2800607597705933,
            "kl": 0.02183909882213229
          },
          "meanLocal": {
            "overlap": 0.9339501545576764,
            "nll": 2.2800607597705933,
            "kl": 0.022003670217684946
          },
          "statsError": 6.639133687258436e-14,
          "freshError": 1.3592946213059065e-13,
          "directError": 1.5165646516379638e-13
        },
        {
          "epoch": 3,
          "headNLL": 2.279413156475338,
          "tied": {
            "overlap": 0.9083783774197662,
            "nll": 2.279413156475338,
            "kl": 0.03182834560644183
          },
          "stale": {
            "overlap": 0.9336534999687729,
            "nll": 2.279413156475338,
            "kl": 0.02209275522028454
          },
          "closed": {
            "overlap": 0.9339592741735204,
            "nll": 2.279413156475338,
            "kl": 0.02205580058383995
          },
          "fresh": {
            "overlap": 0.933959274173519,
            "nll": 2.279413156475338,
            "kl": 0.022055800583840112
          },
          "kl": {
            "overlap": 0.9340818927031265,
            "nll": 2.279413156475338,
            "kl": 0.021854143345252538
          },
          "meanLocal": {
            "overlap": 0.9339250713827458,
            "nll": 2.279413156475338,
            "kl": 0.0220190754181204
          },
          "statsError": 6.639133687258436e-14,
          "freshError": 1.7263968032921184e-13,
          "directError": 1.5165646516379638e-13
        },
        {
          "epoch": 4,
          "headNLL": 2.278784763567744,
          "tied": {
            "overlap": 0.9083174729418839,
            "nll": 2.278784763567744,
            "kl": 0.031856285026927965
          },
          "stale": {
            "overlap": 0.9333834981835729,
            "nll": 2.278784763567744,
            "kl": 0.02213904302479918
          },
          "closed": {
            "overlap": 0.9339349533511316,
            "nll": 2.278784763567744,
            "kl": 0.02207038893113596
          },
          "fresh": {
            "overlap": 0.9339349533511314,
            "nll": 2.278784763567744,
            "kl": 0.0220703889311361
          },
          "kl": {
            "overlap": 0.9340567790440613,
            "nll": 2.278784763567744,
            "kl": 0.021868742080055428
          },
          "meanLocal": {
            "overlap": 0.9339006710739698,
            "nll": 2.278784763567744,
            "kl": 0.022034028116441517
          },
          "statsError": 6.639133687258436e-14,
          "freshError": 1.9628743075372768e-13,
          "directError": 1.51684220739412e-13
        }
      ],
      "gen": [
        {
          "seed": 11,
          "method": "tied",
          "requests": 32,
          "generated": 185,
          "proposed": 170,
          "accepted": 144,
          "attempted": 157,
          "blocks": 94,
          "shallowPositions": 775,
          "fullPositions": 529,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 11,
          "method": "stale",
          "requests": 32,
          "generated": 184,
          "proposed": 167,
          "accepted": 144,
          "attempted": 154,
          "blocks": 96,
          "shallowPositions": 735,
          "fullPositions": 526,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 11,
          "method": "closed",
          "requests": 32,
          "generated": 199,
          "proposed": 185,
          "accepted": 152,
          "attempted": 167,
          "blocks": 104,
          "shallowPositions": 866,
          "fullPositions": 588,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 11,
          "method": "kl",
          "requests": 32,
          "generated": 189,
          "proposed": 164,
          "accepted": 146,
          "attempted": 155,
          "blocks": 95,
          "shallowPositions": 738,
          "fullPositions": 535,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 29,
          "method": "tied",
          "requests": 32,
          "generated": 197,
          "proposed": 186,
          "accepted": 149,
          "attempted": 168,
          "blocks": 99,
          "shallowPositions": 838,
          "fullPositions": 563,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 29,
          "method": "stale",
          "requests": 32,
          "generated": 192,
          "proposed": 169,
          "accepted": 148,
          "attempted": 161,
          "blocks": 99,
          "shallowPositions": 762,
          "fullPositions": 553,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 29,
          "method": "closed",
          "requests": 32,
          "generated": 191,
          "proposed": 168,
          "accepted": 147,
          "attempted": 160,
          "blocks": 99,
          "shallowPositions": 754,
          "fullPositions": 552,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 29,
          "method": "kl",
          "requests": 32,
          "generated": 192,
          "proposed": 169,
          "accepted": 148,
          "attempted": 161,
          "blocks": 99,
          "shallowPositions": 762,
          "fullPositions": 553,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 73,
          "method": "tied",
          "requests": 32,
          "generated": 189,
          "proposed": 167,
          "accepted": 147,
          "attempted": 157,
          "blocks": 97,
          "shallowPositions": 757,
          "fullPositions": 541,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 73,
          "method": "stale",
          "requests": 32,
          "generated": 193,
          "proposed": 170,
          "accepted": 147,
          "attempted": 159,
          "blocks": 99,
          "shallowPositions": 772,
          "fullPositions": 563,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 73,
          "method": "closed",
          "requests": 32,
          "generated": 191,
          "proposed": 174,
          "accepted": 147,
          "attempted": 162,
          "blocks": 96,
          "shallowPositions": 787,
          "fullPositions": 541,
          "greedyMismatch": 0,
          "arPositions": 691
        },
        {
          "seed": 73,
          "method": "kl",
          "requests": 32,
          "generated": 194,
          "proposed": 173,
          "accepted": 149,
          "attempted": 162,
          "blocks": 98,
          "shallowPositions": 788,
          "fullPositions": 554,
          "greedyMismatch": 0,
          "arPositions": 691
        }
      ]
    }
  ]
}
~~~
