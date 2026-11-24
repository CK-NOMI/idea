# team-himerge/R07｜HMC-SD：多级最大耦合与共同验证保留概率
2026-11-24；起始 main bfaad2e2838e87840c367fd5070cbad9f31aee47、README blob 4c8bf0d22364a2b9bbe7b918a3d5715178c1c46c，259716 JS chars。核对R06已读快照完整前缀，阅读新增FRGB/CFA/SCA/FTCD/FEC全文及HCA/R06、HIERASPEC/R06实际代码；未独立复跑他队。归档前main 30247fc69d0bf0e61c3544aeaa658a3a7e5b2352 README原样不变。与PRE-07跨版本共同质量、A/R04跨轨迹耦合、FEC熵耦合、FTCD序列重绑定不同：固定前缀三个深度的联合分布，同时保留相邻最大耦合。共同概率质量及多元耦合并非新数学。
**CRITICAL / Reject and Pivot**。T=限定理论，E=本轮实际V8真实公开文本统计LM和单步合成MC，P=成本代理，O=全概率oracle，H=未验证。无真实预训练Transformer/GPU/物理HFL/实际多token自投机生成/网络/总时间/CI。本轮未重新探测CUDA或下载模型。
## 1. 场景、机制、理论
假设同一固定前缀h、同一合法正式模型快照下三种内生退出分布p0,p1,p2，分别映射到client/edge/cloud；p2为明确正式目标，非独立完整draft LLM。实际跨设备分层和模型层计算分置不等于HFL，需另证真实学习及数据驻留必要性。区域持有授权特征与概率，允许必要通信，不声称DP。希望精确生成且提高跨层共同保留计算，再用共同保留率指导正式学习。单步联合设计如下。
令 X_i~p_i，A_j=Σ_v min(p_j(v),p_{j+1}(v))。标准相邻最大耦合具有对角质量m01(v)=min(p0,p1),m12(v)=min(p1,p2)。固定中间状态X1=v（p1(v)>0），两侧匹配条件概率a_v=m01(v)/p1(v), b_v=m12(v)/p1(v)。其余边际分别由残差r0=(p0−p1)_+/δ01、r2=(p2−p1)_+/δ12恢复（δ=0时永远不需该残差）。普通两段独立随机的Markov组合，三级全同概率 C_chain=Σ_v p1(v)a_v b_v。
新组合：条件X1=v，共享U~Unif(0,1)，X0=v iff U<a_v，否则独立从r0抽；X2=v iff U<b_v，否则独立从r2抽。两侧残差互相独立且独立于U。每个相邻对仍是标准最大耦合（反向条件核也是同一联合矩阵）；两侧匹配事件嵌套。C_joint=Σ p1 min(a_v,b_v)=Σ min(p0,p1,p2)，达到任意联合分布全同事件的上界。即使实际方向从X0开始，也可先按标准K01采X1，再根据是否X0=X1抽U~Unif(0,a_v)或Unif(a_v,1)，再按U<b_v判X2，否则抽r2；得到同一联合分布。边界a=0/1时只在正概率分支采。需要知道a_v（由真实p0(v),p1(v)决定）和第一阶段匹配事件，不能从旧logprob或当前候选ID随意推断。
精确损失 C_joint−C_chain=Σ_{v:p1(v)>max(p0(v),p2(v))} min(p0(v),p2(v)) [1−max(p0(v),p2(v))/p1(v)]。证明分三种中间值排序，只有p1严格大于两端且两端正质量时损失正。两端匹配（包括两次都不匹配中间却两端相等）令 d=Σ r0(v)r2(v)，则Markov端点概率 Σ p1[ab+(1−a)(1−b)d]，共享U端点概率 Σ p1[min(a,b)+(1−max(a,b))d]。共同机制端点不劣于Markov；增量=(1+d)(C_joint−C_chain)。端点仍≤直接最大耦合Σ min(p0,p2)。此为固定单步概率，不是实际多级block算法的接受长度；同一token未必意味跨设备缓存、隐藏状态或版本可复用。条件反馈影响提议时需FCRR/LBS真实条件律审查。
T：三对同时最大一般不可实现，已有multimaximal coupling经典反例：p0=(.5,.5,0),p1=(0,.5,.5),p2=(.5,0,.5)。三对相等事件分别必须在不同token，彼此互斥，每个最大.5，合计1.5矛盾。正支持版本在各分布混入ε均匀概率、ε<.25时仍不可同时最大：两两最大和=1.5+1.5ε；对任意三变量相等事件数≤1+2·I(全同)，而全同概率≤ε，故1.5+1.5ε>1+2ε。不能将三个局部TV相似度直接解释为一致的全局联合机制。
T/H：固定两端，若p1(v)≥min(p0(v),p2(v))对所有v，则三级全同最大等于直接端点最大。令p1是两端凸组合，进一步p1逐词处于两端之间，则Markov本来就无三级损失，无需共享随机机制。这只是概率空间可行性，不保证内生中层容量或正式NLL/区域任务质量。候选学习目标 L_task + λ Σ_h E_w Σ_v[min(p0(v),p2(v))−p1(v)]_+ ，固定/stop-grad两端时惩罚三级共同质量差距；若两端也更新，则改变正式目标并可能牺牲质量，PRE-04/JVT需审计。验证中已知拒绝不等于完整梯度；普通全概率蒸馏或直接插值、普通AR均可计算此目标。HFL线性参数平均不保证概率空间凸组合；同信息平坦可执行相同训练及委托。
随机正确性：每个固定合法前缀的p2边际由联合构造保证，若所有已揭示历史下重新按真实分布构造且随机独立、只提交正式Y2，则链式归纳保持序列分布；这不是已实现的多token批验证协议。不能只因早期接受条件就提交未来路径，bonus、EOS、KV、温度/top-p、有限精度需另证。Greedy应以正式p2 argmax和统一tie-break，不能从随机三方相等定理推出greedy实现精确。
## 2. 实际公开语料实验 E/P
实际用GitHub connector读取两份公开Gutenberg正文（同HCA/R06来源，引用并承认统计LM训练类似；本轮新对象为三方固定前缀耦合）：[Wallpaper](https://github.com/GITenberg/The-Yellow-Wallpaper_1952/blob/ce7c5bc1a5cf0fe23bcb265545a900365dbae29c/1952.txt) blob 3a2b5f55c6f1d13a7ae623282d38fde5a6ff352b、[Alice](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/blob/7f18748e114b994d5d9454d5793581bc0b9cf9d7/11.txt) blob 9d73b1d7447b7ddc158b1b157f5f700d21bd2fb7。真实文本清洗至27字符，Alice按中间空格拆两区、Wallpaper一区，各80%训练20%留出。实际计数训练长度24273/54054/54055，留出6069/13514/13514，合计33091个有效双字符前缀位置（33097−6）。p0=区域bigram+smooth×全局bigram；p1=全局bigram加.5平滑；p2=全局trigram+5×p1。均为统计LM而非共享深度神经模型或真正HFL；三个区域、smooth1/20/100共9设置，重复使用留出位置共99273个位置配置，非独立样本。精确枚举概率加权均值：
smooth e n adj01 adj12 direct_endpoint chain_all joint_all chain_endpoint joint_endpoint positive_gap_fraction
1 0 6067 .9097489791802482 .6418571794255927 .6345244545491676 .5909046357727283 .6056656823832232 .5966143131653016 .6131747401668645 .9993406955661798
1 1 13512 .9622102567070618 .6404651418333244 .64061849827362 .6204018827273468 .6273666313791522 .622846867586215 .6307413140452843 .9999999999999999
1 2 13512 .9464226159005139 .6442548298493582 .6422209746540996 .6152257431405469 .6239740200078004 .6193707744480195 .6295228889428514 .9997779751332145
20 0 6067 .9118937918526849 .6418571794255927 .6349701845028444 .5920416019852554 .6066729163411851 .5975564010383871 .613968732379803 .9993406955661798
20 1 13512 .9628406025040426 .6404651418333244 .6407741438075484 .6207352141184381 .6277120409366259 .6230959265870469 .6310155253935817 .9999999999999999
20 2 13512 .9471857348257073 .6442548298493582 .6424021339722976 .6156132144457165 .6243444825826351 .619677135414816 .6298084626690208 .9997779751332145
100 0 6067 .9186211961897983 .6418571794255927 .6359336044060693 .5955059884227926 .6095376457322168 .6004870961696029 .6162180775276802 .9993406955661798
100 1 13512 .964534451321076 .6404651418333244 .6408477606032306 .6215754132964677 .6283922055161069 .6237601278645281 .6314873351323471 .9999999999999999
100 2 13512 .94941090637109 .6442548298493582 .6425606177751655 .6167124232549077 .6252043387928606 .6205502815186837 .6303983064630891 .9997779751332145
共同保留绝对提升.6817–1.4761个百分点；联合端点仍低于直接最大耦合。非真实接受长度、训练质量、GPU时延或学习速度，未优化模型参数，统计平滑不等于新训练机制。
另实际V8合成MC：p0=(.2,.6,.2),p1=(.6,.2,.2),p2=(.3,.3,.4)，理论相邻匹配.6/.7、独立三级.5、共享三级.6、独立端点.5666667、共享端点.7、直接端点.7。3seed11/29/73×2方式×100000=600000实际抽样。原始计数：seed mode h0 h1 h2 matches01,12,02,all
11 independent [20056,59952,19992] [59933,20075,19992] [29753,30186,40061] [60123,69820,56823,50051]
11 shared [19906,60096,19998] [60055,19947,19998] [30095,29944,39961] [59851,70040,69848,59851]
29 independent [19802,60123,20075] [59963,19962,20075] [30007,29950,40043] [59839,70044,56796,50103]
29 shared [19945,60085,19970] [59881,20149,19970] [29959,30148,39893] [60064,70078,70063,60064]
73 independent [20133,59819,20048] [60101,19851,20048] [29886,29975,40139] [60032,69785,56632,49913]
73 shared [20131,59640,20229] [59681,20090,20229] [30027,29960,40013] [60450,70346,70320,60450]
本轮只实现以中间状态为条件的MC，不是生产正向级联；未实际多token验证。
## 3. 可复算代码（V8，实际执行函数；Node读取固定文本）
复制为probe.js，下载上述固定文本命名wallpaper.txt alice.txt，node probe.js wallpaper.txt alice.txt。文档原始文本未重复上传。输出完整9行及MC。
~~~javascript
function probe(docs){
const V=27,D=V*V,alphabet=" abcdefghijklmnopqrstuvwxyz",sum=a=>a.reduce((s,x)=>s+x,0),norm=a=>{let z=sum(a);return a.map(x=>x/z)};
const clean=d=>d.text.split(/\*\*\* START OF THIS PROJECT GUTENBERG EBOOK[^\n]*\n/)[1].split(/\*\*\* END OF THIS PROJECT GUTENBERG EBOOK/)[0].toLowerCase().replace(/[^a-z]+/g," ").trim();
let a=clean(docs[1]),w=clean(docs[0]),cut=a.lastIndexOf(" ",Math.floor(a.length/2)),regions=[w,a.slice(0,cut),a.slice(cut+1)].map(s=>Array.from(s,c=>alphabet.indexOf(c))),train=[],test=[],biAll=new Float64Array(D),triAll=new Float64Array(V**3),local=[];
for(let x of regions){let n=Math.floor(x.length*.8);train.push(x.slice(0,n));test.push(x.slice(n));}
for(let x of train){let bi=new Float64Array(D);for(let i=2;i<x.length;i++){let j=x[i-1]*V+x[i];bi[j]++;biAll[j]++;triAll[x[i-2]*D+j]++}local.push(bi)}
let mid=Array.from({length:V},(_,i)=>norm(Array.from(biAll.slice(i*V,(i+1)*V),x=>x+.5))),target=Array.from({length:D},(_,i)=>norm(Array.from(triAll.slice(i*V,(i+1)*V),(x,j)=>x+5*mid[i%V][j])));
let rows=[],cases=[];
for(let smooth of [1,20,100])for(let e=0;e<3;e++){let rr=test[e],shallow=Array.from({length:V},(_,i)=>norm(Array.from(local[e].slice(i*V,(i+1)*V),(x,j)=>x+smooth*mid[i][j]))),freq=new Float64Array(D);for(let i=2;i<rr.length;i++)freq[rr[i-2]*V+rr[i-1]]++;
let m={adj01:0,adj12:0,direct:0,chainAll:0,sharedAll:0,chainEnd:0,sharedEnd:0,strict:0},n=sum(freq);
for(let i=0;i<D;i++){if(!freq[i])continue;let p=shallow[i%V],q=mid[i%V],r=target[i],a=p.map((x,j)=>Math.min(x,q[j])),b=r.map((x,j)=>Math.min(x,q[j])),d0=1-sum(a),d2=1-sum(b),l=p.map((x,j)=>Math.max(0,x-q[j])),z=r.map((x,j)=>Math.max(0,x-q[j])),dot=d0>1e-12&&d2>1e-12?sum(l.map((x,j)=>x*z[j]))/(d0*d2):0;
let t={adj01:sum(a),adj12:sum(b),direct:sum(p.map((x,j)=>Math.min(x,r[j]))),chainAll:0,sharedAll:0,chainEnd:0,sharedEnd:0};for(let j=0;j<V;j++){let A=a[j]/q[j],B=b[j]/q[j],c=A*B,s=Math.min(A,B);t.chainAll+=q[j]*c;t.sharedAll+=q[j]*s;t.chainEnd+=q[j]*(c+(1-A)*(1-B)*dot);t.sharedEnd+=q[j]*(s+(1-Math.max(A,B))*dot)}t.strict=Number(t.sharedAll-t.chainAll>1e-12);for(let k in m)m[k]+=freq[i]*t[k]/n;
if(smooth===20&&i%71===0)cases.push({e,i,p,q,r,t});}
rows.push({e,smooth,n,...m});}
return {train:train.map(x=>x.length),test:test.map(x=>x.length),rows,cases:cases.length};
}
function sampleProbe(){
const p=[.2,.6,.2],q=[.6,.2,.2],r=[.3,.3,.4],seeds=[11,29,73],N=100000;
const sum=a=>a.reduce((s,x)=>s+x,0),a=p.map((x,i)=>Math.min(x,q[i])/q[i]),b=r.map((x,i)=>Math.min(x,q[i])/q[i]),z=p.map((x,i)=>Math.max(0,x-q[i])),w=r.map((x,i)=>Math.max(0,x-q[i])),Z=sum(z),W=sum(w),rows=[];
for(let seed of seeds){let st=seed;const u=()=>{st=(Math.imul(1664525,st)+1013904223)>>>0;return(st+.5)/4294967296};const draw=(v)=>{let t=u()*sum(v);for(let i=0;i<v.length;i++){t-=v[i];if(t<0)return i}return v.length-1};
for(let mode of ["independent","shared"]){let h=Array.from({length:3},()=>[0,0,0]),m=[0,0,0,0];for(let i=0;i<N;i++){let y=draw(q),U=u(),V=mode==="shared"?U:u(),x=U<a[y]?y:draw(z),k=V<b[y]?y:draw(w);h[0][x]++;h[1][y]++;h[2][k]++;m[0]+=+(x===y);m[1]+=+(y===k);m[2]+=+(x===k);m[3]+=+(x===y&&y===k)}rows.push({seed,mode,h,m})}}
return {N,p,q,r,a,b,rows}}
const fs=require("node:fs");
console.log(JSON.stringify({corpus:probe(process.argv.slice(2,4).map(p=>({text:fs.readFileSync(p,"utf8")}))),mc:sampleProbe()},null,2));
~~~
## 4. 原始文献核查与成本
2026-11-24检索 hierarchical self speculative decoding, cascaded early-exit, multimaximal categorical coupling, federated early exits；核对原始摘要/出版页及PDF检索摘要，未全篇逐式核查。
- [HiSpec](https://arxiv.org/abs/2510.01336) 2025，early exit中间验证，架构直接重合。
- [Fast Inference via Hierarchical Speculative Decoding](https://arxiv.org/abs/2510.19705) 2025/2026，逐级block验证及成本分析。
- [Cascade Speculative Drafting](https://papers.nips.cc/paper/2024/file/9cb5b083ba4f5ca6bd05dd307a2fb354-Paper-Conference.pdf) NeurIPS2024，多级草稿。
- [PipeSpec](https://arxiv.org/abs/2505.01572) 2025，多级设备执行、打破阶段依赖。
- [LayerSkip](https://aclanthology.org/2024.acl-long.681/) ACL2024，共享早退训练及内生自投机。
- [ReeFL](https://proceedings.mlr.press/v235/lee24h.html) ICML2024，异构联邦早退共享分类器。
- [3-Model Speculative Decoding](https://arxiv.org/abs/2510.12966) 2025，三级验证近邻，宽松接受须另审计精确性。
- [Dzhafarov & Kujala Contextuality-by-Default 2.0](https://www.psych.purdue.edu/~ehtibar/publications/DzhKujQI16d.pdf) 2016/2017，非二元multimaximal coupling不存在经典反例，明确不宣称本轮首次发现。
- [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html) ICML2023基础。无正式文字查重；遗漏最新耦合优化、多边际OT、未公开工作的风险。实际论文机制与本轮单步联合构造是否等价需进一步审计完整算法。
P：每级同一前缀需真实分布值/残差及正向条件a，示意每位置额外传FP32 a和匹配bit至少约5B（含padding及协议前的示意，非下界），若目标级没有中层全概率则仍需全量p1或可核查的查询/残差采样协议；V151936的FP32向量607744B/位置，普通SD已有分布处理不可重复计费。跨设备BF16 hidden d1536为3072B/位置，版本/KV/通信延迟/训练额外损失另计。普通直接浅层→目标最大耦合无需中间阶段，HiSpec/HSD/LayerSkip、普通局部学习+SD、直接蒸馏与同信息平坦是强基线。全同改善绝对<.015不是壁钟速度；若只按每多一个共同token节省一个等成本位置且其它成本不变，直接节省工作上限<1.5%全部位置；真实跨层每位置权重、串行依赖或批效率不同则不能套。实际未测。
## 5. 裁决
CRITICAL：直接端点最大耦合更强；真实语料共同保留收益小；概率耦合与多级SD均有近邻；正式学习惩罚可用完整概率或普通AR计算，同信息平坦可复制，HFL双向必要性未证。主观创新6.2/10（问题7、机制6、理论7、价值5.5、距离4.5，权重25/25/20/15/15，合计6.15），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL（限定单步数学/MC，完整随机/greedy/有限精度未验证）；G4 FAIL真实文本统计模型非两尺度预训练；G5/G6未验证，无80%依据；G7局部可复算/整体未过。好结果同质量/预算最强公平基线全流程真实时间−15%或有效工作+20%，两预训练尺度三非IID/时延三独立种子、CI、全部G。未执行1–2×40GB。仅有新真实约束时再测同一早退Transformer三层在≥10000真实前缀的p0/p1/p2、各层计算/通信/回退、真实质量及总耗时，1–2×40GB、两个预训练尺度、8逻辑客户端2区域，比较优化HiSpec/HSD、直接LayerSkip、普通HFL+SD、同信息平坦；单卡模拟非物理并行。**Reject and Pivot**，保留闭式损失、条件共享耦合、经典反例的正支持推广、真实文本及MC实际结果。下一轮优先真正预训练模型及物理跨层约束，避免再以统计LM或单步概率替代端到端证据；若不能获得，做真实代码/trace审计而非无依据的新架构。
