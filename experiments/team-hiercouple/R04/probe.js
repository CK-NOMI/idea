// R04: arithmetic re-analysis of published stage timings, not measured HFL.
// Run: node experiments/team-hiercouple/R04/probe.js
const rows=[
 {paper:"Iso et al. 2604.26779 Table 1",name:"Qwen3-8B RL-Think",arT:185.3,arG:133.6,sdT:137.4,sdG:87},
 {paper:"Iso et al. 2604.26779 Table 1",name:"Qwen3-8B-Base RL-Zero",arT:151.2,arG:100,sdT:107.5,sdG:56.6},
 {paper:"EfficientRollout 2606.18967 Table 2",name:"Qwen2.5-7B",arT:132.6,arG:82.4,sdT:115.7,sdG:66.3},
 {paper:"EfficientRollout 2606.18967 Table 2",name:"Qwen2.5-14B",arT:221.1,arG:126.6,sdT:197.2,sdG:105.3},
 {paper:"EfficientRollout 2606.18967 Table 2",name:"Llama3.1-8B",arT:186.9,arG:126.4,sdT:172.2,sdG:112.9}
];
const goal=.15; const round=x=>+x.toFixed(6);
const table=rows.map(r=>({...r,observedStepSaving:round(1-r.sdT/r.arT),observedGenSaving:round(1-r.sdG/r.arG),
 arGenShare:round(r.arG/r.arT),sdGenShare:round(r.sdG/r.sdT),
 requiredExtraGenSaving:round(goal*r.sdT/r.sdG),
 maxAddedSerialWanForObserved15pct:round((r.arT-r.sdT)/goal-r.arT),
 gainIfExtra20pctGen:round(.2*r.sdG/r.sdT)}));
const scenarios=[];
for(const r of rows)for(const W of [0,20,60])for(const h of [0,5])for(const overlap of [0,.5]){
 // W is additional SERIAL common HFL overhead; overlap is a fixed amount
 // of the baseline generation phase that could be fully hidden by other tasks.
 // Baseline T remains sdT+W; this is a headroom upper-bound sensitivity,
 // not an exact pipeline reconstruction. Exposed generation Gexp=(1-overlap)*G.
 const Gexp=(1-overlap)*r.sdG,base=r.sdT+W;
 scenarios.push({name:r.name,W,h,overlap,Gexp:round(Gexp),base:round(base),
 minFractionOfExposedGenRemoved:round((goal*base+h)/Gexp),
 maxSaving:round((Gexp-h)/base),
 savingIfRemove20pctExposed:round((.2*Gexp-h)/base)});
}
// two parallel paths with real stage totals as numerical anchors, hypothetical WAN.
// All durations independent of scheduling in this proxy.
const paths=[];
for(const W of [0,20,40,80]){
 const a={C:137.4-87,G:87},b={C:107.5-56.6+W,G:56.6};
 const T=Math.max(a.C+a.G,b.C+b.G),deadline=.85*T;
 const r=Math.max(0,(a.C+a.G-deadline)/a.G,(b.C+b.G-deadline)/b.G);
 paths.push({W,T:round(T),deadline:round(deadline),minCommonGenReduction:round(r),
 infeasibleEvenZeroGen:r>1,saveAOnly20pct:round(1-Math.max(a.C+.8*a.G,b.C+b.G)/T),
 saveBOnly20pct:round(1-Math.max(a.C+a.G,b.C+.8*b.G)/T),
 saveBoth20pct:round(1-Math.max(a.C+.8*a.G,b.C+.8*b.G)/T)});
}
console.log(JSON.stringify({evidence:"Published timings transcribed; arithmetic re-analysis and hypothetical added costs only; no raw traces or HFL execution",goal,table,scenarios,paths},null,2));