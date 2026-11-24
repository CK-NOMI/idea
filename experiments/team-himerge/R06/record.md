# team-himerge/R06 | FCF：投机耦合的联邦Fisher曲率估计及半正定障碍
2026-11-24；起始 main 773b453d0483e5059618e0ca245c730d6e6dd515，README blob 994dbc520ccbaf7d88febee50f25105d650301f6，229656 JS chars。核对R05已读快照为完整前缀，阅读新增FOAS全文及HCA/R05一阶梯度代码/结果摘要、HC/R05三类请求日志摘要、HIERASPEC/R05源指标并计算字段和计数。归档前读取main 96a0c0c9b7bcec763841916411232fba011eb1c9新增FCKG全文，与本轮二阶矩不同，但共享最大耦合和额外同前缀采样限制。PRE-04被拒后缀Fisher/KL、A/R05–06多保真控制变量、HC/R03一阶蒸馏、HCA/R05双向KL近邻，非全新通用估计方法。未复跑他队实验。
**CRITICAL / Reject and Pivot**。T=限定数学，E=本轮实际V8 CPU有限概率矩枚举和公开日志字段审计，P=计算通信代理，O=全知控制系数，H=未验证。没有本队真实预训练模型、HFL训练、SD生成、GPU、网络、真实wall time或CI。本轮未新尝试加载权重，不宣称已测环境CUDA或模型访问。

## 1. 问题及机制
各客户端持授权非IID上下文及训练数据，区域执行同一正式模型固定快照p_theta、轻量内生预测头q（无独立完整草稿LLM），边缘/云聚合目标梯度与曲率。允许区域访问自身真实p/q及计算图，不声称DP。研究目标：同模型质量、预算下缩短正式训练+推理的总时间。区别于仅草稿更新：尝试用标准SD最大耦合提供正式模型自然梯度的曲率估计；正式模型训练更新会改变后续p与内生q关系。对于固定外生前缀h、完整概率向量p,q，固定正式logit Jacobian J=d z_p/d theta，logit Fisher F(p)=diag(p)−pp^T，参数Fisher J^T F(p) J（温度1）。q曲率作为控制量必须用同一个J，不可用草稿自身不同Jacobian偷换。对不同h按预定权重加总参数Fisher/梯度，不能先平均p后算F（差一个上下文间协方差），也不能平均局部逆矩阵等同全局逆。训练数据损失梯度另算，Fisher非任意任务Hessian。
每个h独立执行两次标准单步SD耦合(X_i~q,Y_i~p)，i=1,2；固定h、p,q、独立随机流，不可用相邻自回归token或bonus替代第二次同前缀独立样本。令 B(a,b)=.5(e_a−e_b)(e_a−e_b)^T，则 E B(Y1,Y2)=F(p)。已知F(q)时估计 H=F(q)+B(Y1,Y2)−B(X1,X2)。两次均接受时修正为0。区域可累计稀疏有符号秩1修正（每项最多2 token行），另计F(q)、J、VJP、样本ID、梯度和归约。云拟用阻尼预条件更新，但H未必PSD，不能直接作为安全自然梯度。标准SD用真实p,q、min(1,p/q)及正残差，合法停止/bonus、因果前缀和独立随机性保证理想实数当前目标序列；greedy用目标argmax。完整SD、温度/top-p/有限精度/实际KV未实现。更新后目标不是旧模型。正式采样和额外曲率抽样不得反馈改变当前提议条件。

## 2. 理论资产 T
T1: B的期望由独立性展开=.5(E e_a e_a^T+E e_b e_b^T−2pp^T)=F(p)。故H无偏。设δ=TV(p,q)，最大耦合每对匹配概率1−δ，两对独立，则两对均匹配概率(1−δ)^2；修正为0（反向未必成立）。B的Frobenius范数平方≤1，两PSD矩阵内积非负，故||B_Y−B_X||_F²≤2；方差迹 E||H−F(p)||_F²≤2(2δ−δ²)−||F(p)−F(q)||_F²。最优一般控制系数a*=Cov_F(B_Y,B_X)/Var_F(B_X)，H_a= B_Y+a(F(q)−B_X)无偏，固定a或独立估计；a*依赖未知矩及真实p，实验为oracle，a≠1时双接受修正不一定为0。直接计算完整F(p)无方差。
T2: 无偏不等于PSD。二元p=(.8,.2),q=(.5,.5)的最大耦合联合质量为(X0,Y0)=.5,(X1,Y0)=.3,(X1,Y1)=.2。两独立对以2*.5*.3=.3概率发生X1≠X2、Y1=Y2，H=−.25[[1,−1],[-1,1]]，特征值−.5和0，严格非PSD。一般事件Y1=Y2且X1≠X2足以推出H非PSD：沿d=e_X1−e_X2，d^T F(q)d=Var_q(d_X)≤1，d^T B_X d=2，因此d^T H d≤−1。事件概率=Σ_y[p_y²−Σ_x Γ_xy²]（两独立对），只是不定概率下界。PSD投影会增加所有被裁剪负特征值对应的迹，若负事件概率正则E trace严格大于原F(p)；阻尼即使保证可逆也不保证E[(H+λI)^−1]=(F(p)+λI)^−1。例正标量H=1或3各半，λ=1，E逆=.375而均值逆=1/3。若使用PSD裁剪、足够阻尼或批平均，须独立验证偏差及收敛；不能仅凭无偏矩宣称自然梯度无偏。
T3: 已有验证器通常计算完整p，精确HVP F(p)v=p⊙v−p(p^Tv)只需O(V)算术、无需V²矩阵或第二次目标采样。参数Fisher向量积需Jv及J^T，不能凭token行支持跳过主干。对冻结特征线性头W，J将logit扰动映射回特征外积；完整HVP同样可按结构执行。普通AR可执行同样最大耦合、直接采样、协方差控制变量、FedFisher；同信息平坦可聚合同样曲率，未发现HFL双向不可替代。若区域只上传曲率样本而不传p，需对比直接本地HVP、对角Fisher、K-FAC和同预算sketch，不能以完整V²矩阵作为唯一基线。
## 3. 本轮实际E/P/O实验
实际V8 CPU执行，LCG/Box-Muller只用于复算；V12，80合成上下文/配置，seed11/29/73，σ=.1/.5/1.5，共720上下文配置（240个p树在3个σ重复使用），p=softmax(N(0,I)),q=softmax(z+σN)。对两独立耦合对精确枚举所有联合概率，非MC频率。比较直接独立目标两样本Fisher、已知p的单样本score外积、最大耦合控制变量、独立p/q控制变量、同U逆CDF控制变量，另报告每种oracle最优系数。方差定义E||H−EH||_F²，非优化收敛或实际时间。已知完整p的确定性矩阵为零方差强基线。
|σ|δ均值|直接两样本|score|最大耦合a1|最大耦合oracle|独立a1|同U a1|不定概率下界|两对匹配|
|---|---|---|---|---|---|---|---|---|---|
|.1|.035754|.718596|.634007|.094226|.090966|1.435368|.292497|.009524|.929913|
|.5|.172901|.718596|.634007|.406807|.350515|1.411338|.851727|.037392|.687401|
|1.5|.440659|.718596|.634007|.790915|.603397|1.285587|1.116035|.060361|.329856|
相对直接两样本方差降86.9%/43.4%/负10.1%，相对score降85.1%/35.8%/负24.7%。oracle系数并非实际可用，也不保零修正。无正式CI、GPU、LLM、HFL、优化器或时间测量；σ是合成扰动非真实非IID测量。逐种子实际选定结果（σ seed δ direct score max maxOpt ind common maxNegLower maxBothAccept）：
~~~text
.1 11 .03506389117910124 .7093919064836804 .6318406995409936 .09251223069418447 .08920606132932829 1.418039932403843 .280692431625078 .009881257202246493 .9312383668318358
.1 29 .037305323357990715 .7218116883541668 .6330906022140815 .09789413739990604 .0943536613155824 1.442329929405869 .293220027935968 .010400180129985745 .9269525043640644
.1 73 .0348931610587978 .7245844107337567 .6370886023063363 .09227062681482186 .08933903700504658 1.4457330641574568 .30357707962629105 .0082912041000398 .9315495842960463
.5 11 .16930032844791956 .7093919064836804 .6318406995409936 .3999021096110454 .3425501035293138 1.3996038638418025 .8337243364895218 .039574517366001316 .6934206870251133
.5 29 .18331973908757682 .7218116883541668 .6330906022140815 .42502893383479273 .36405027182859084 1.4165764697076149 .8491537672764528 .04053112624191676 .6709268394350809
.5 73 .1660837451616733 .7245844107337567 .6370886023063363 .3954900539133588 .3449447565348735 1.4178336393057298 .8723037041055967 .03207098642594918 .6978564731868099
1.5 11 .4361269230492384 .7093919064836804 .6318406995409936 .7886591404141063 .5923531370749017 1.2876872724366029 1.1250525832968907 .06732798905894677 .33718167958276635
1.5 29 .4629389854601492 .7218116883541668 .6330906022140815 .8057411688231386 .61492368463129 1.2788136939206396 1.1145080287142497 .06209097799394105 .30596488790755233
1.5 73 .4229122003870069 .7245844107337567 .6370886023063363 .7783435196077558 .6029140440640578 1.2902589340565584 1.1085444540915486 .051663287789593594 .3464204734021032
~~~
实际执行脚本复制为probe.js，node probe.js；完整结果含其他oracle/匹配/不定事件指标：
~~~javascript
function probe(){
 const V=12,N=80,seeds=[11,29,73],sigmas=[.1,.5,1.5];let state=0;
 const u=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+.5)/4294967296};
 const normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
 const sum=a=>a.reduce((s,x)=>s+x,0),soft=z=>{let m=Math.max(...z),a=z.map(x=>Math.exp(x-m)),s=sum(a);return a.map(x=>x/s)};
 const sq=a=>sum(a.map(x=>x*x)),normF=p=>sq(p)-2*sum(p.map(x=>x*x*x))+sq(p)**2;
 const diffF=(p,q)=>{let s=0;for(let i=0;i<V;i++)for(let j=0;j<V;j++)s+=((i===j?p[i]-q[i]:0)-p[i]*p[j]+q[i]*q[j])**2;return s};
 function joint(p,q,kind){let J=[],delta=sum(p.map((v,i)=>Math.max(0,v-q[i]))),a=0,b=0;
 const cp=[0],cq=[0];for(let i=0;i<V;i++){cp.push(cp[i]+p[i]);cq.push(cq[i]+q[i])}
 for(let x=0;x<V;x++)for(let y=0;y<V;y++){let w=kind==="max"?(x===y?Math.min(p[y],q[x]):0)+(delta?Math.max(0,q[x]-p[x])*Math.max(0,p[y]-q[y])/delta:0):kind==="ind"?q[x]*p[y]:Math.max(0,Math.min(cp[y+1],cq[x+1])-Math.max(cp[y],cq[x]));if(w>0)J.push({x,y,w})}return J}
 function stats(p,q){
  let fp=normF(p),fq=normF(q),df=diffF(p,q),vp=1-sq(p)-fp,vq=1-sq(q)-fq,delta=sum(p.map((v,i)=>Math.max(0,v-q[i])));let row={delta,direct:vp,score:sum(p.map((v,i)=>v*(1-2*v+sq(p))**2))-fp};
  for(let kind of ["max","ind","common"]){let J=joint(p,q,kind),ip=0,neg=0,zero=0;
   for(let a of J)for(let b of J){let w=a.w*b.w;let z=Number(a.y===a.x)-Number(a.y===b.x)-Number(b.y===a.x)+Number(b.y===b.x);ip+=w*z*z/4;if(a.y===b.y&&a.x!==b.x)neg+=w;if(a.x===a.y&&b.x===b.y)zero+=w}
   let cov=ip-(fp+fq-df)/2;row[kind]=vp+vq-2*cov;row[kind+"Opt"]=vp-cov*cov/vq;row[kind+"NegLower"]=neg;row[kind+"BothAccept"]=zero;
  }return row;
 }
 let rows=[];for(let sig of sigmas)for(let seed of seeds){state=seed;let totals={};for(let t=0;t<N;t++){let z=Array.from({length:V},normal),p=soft(z),q=soft(z.map(x=>x+sig*normal()));let a=stats(p,q);for(let k in a)totals[k]=(totals[k]??0)+a[k]/N}rows.push({sig,seed,...totals})}
 return {V,N,seeds,sigmas,rows};
}
console.log(JSON.stringify(probe(),null,2));
~~~
## 4. 公开日志字段审计
库中[HIERASPEC/R05/source_metrics.json](../../HIERASPEC/R05/source_metrics.json)转录源 shreyansh26/Speculative-Decoding commit 76335be2f8f1a7bf25739310ae80435b83782b23，实际读取500请求（5方法×100）；temperature全部0，E3/E6/MTP1/MTP2/Medusa matches_baseline分别44/42/100/100/45，proposed总8846/9395/4432/7188/19679，accepted总3196/2651/3272/4912/7092。只有请求级计数、时长、匹配、prompt ID等，不含同h的两次独立随机(X,Y)、p/q logits、Jacobian、训练成本；不能以接受计数重建Fisher或把greedy结果当随机耦合。HC/R05的chat/code/summarization三文件亦是请求级记录，非本队硬件实测。未回源独立核验硬件或日志语义，不能据此判定论文精确性或真实方差。
## 5. 文献、成本及裁决
2026-11-24检索 federated Fisher information aggregation natural gradient; multilevel Monte Carlo covariance estimation; speculative decoding online learning maximal coupling。核对原始会议页面及摘要，非穷尽全文、公式、专利、未公开工作；无文字查重。
- [FedFisher AISTATS2024](https://proceedings.mlr.press/v238/jhunjhunwala24a.html) Fisher聚合及曲率近邻。
- [FedFish](https://arxiv.org/abs/2311.10291) 2023函数空间Fisher聚合。
- [FedFiOT](https://arxiv.org/abs/2611.06783) 2026-11 Fisher加权OT模型合并，近期近邻（摘要）。
- [Covariance Expressions for Multifidelity Sampling](https://doi.org/10.1137/23M1607994) 2024多保真协方差控制变量。
- [Covariance estimation using h-statistics](https://arxiv.org/abs/2311.01336) 2023预印/2024发表，多层MC协方差无偏估计，直接数学近邻；[Mycek & De Lozzo](https://doi.org/10.1137/18M1216389) 2019多层MC协方差。
- [Limitations of the Empirical Fisher](https://arxiv.org/abs/1905.12558) NeurIPS2019，Fisher与优化近似的限制。
- [When Drafts Evolve ICML2026](https://proceedings.mlr.press/v306/qian26k.html) 验证反馈在线学习；[DistillSpec](https://arxiv.org/abs/2310.08461)、[Leviathan](https://proceedings.mlr.press/v202/leviathan23a.html)、[LayerSkip](https://aclanthology.org/2024.acl-long.681/)基础。多保真协方差/自然梯度领域已有直接近邻，不能宣称首个此类估计。
成本P：单上下文两个对最多4个32位ID=16B，仅当接收端已有同h、J、F(q)及版本等才能重构logit修正；不包括基底、梯度、通信和安全。直接2个目标ID仅8B；密集F(p)v计算O(V)，实际主干JVP/VJP更昂贵。两对均匹配时可省修正，若修正成本占强基线总时间f，且其他不变，理想节省≤f E[(1−δ)^2]−新增成本；三组均值要达到15%至少f=.1613/.2182/.4547，零新增开销且不可当实际时间。额外同前缀独立抽样可能只需已存概率上的抽样，但随机数、残差、缓存、构造曲率、PSD修正、求逆、跨区域共享J均计费。普通AR/标准SD+精确HVP、FedFisher/K-FAC、普通对角曲率/低秩sketch、同信息平坦与同U/独立控制变量是强对照。
**CRITICAL:** 无偏低方差不等于可用PSD曲率；高异质性方差优势消失；完整p存在时精确HVP更强；普通AR和同信息平坦可复制。主观创新6.0/10（问题7、机制5.5、理论6.5、价值6、距离4，25/25/20/15/15%=6.025），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（理想曲率与标准SD，完整实现未验）；G4 FAIL（CPU合成）；G5/G6未验证，无80%依据；G7局部可复算/整体未过。好结果同质量/预算最强基线全流程真实时间−15%或有效工作+20%，两预训练尺度、三配置三种子CI、全部G。H未执行：1–2×40GB，Qwen2.5-1.5B/3B，LoRA+内生头，8逻辑客户端2边缘，公开数学/代码/通用任务，三非IID/时延三种子；先采≥10000真实同前缀(p,q,J)或HVP trace、独立双样本、实际曲率/逆偏差、PSD修复和计算时间，比较精确HVP、FedFisher、K-FAC、普通HFL+SD、AR耦合与公平平坦，计训练推理通信、质量、GPU/总时、CI。单卡模拟不等于物理分布式。没有证据支持当前方向投入GPU。Reject and Pivot；保留二阶差分公式、半正定反例、实际数值及日志缺失字段审计。下一轮优先获取真实随机SD逐位置trace与实际HFL数据驻留/学习瓶颈，不能用greedy请求接受计数代替；只有明确强基线后仍有可消除全流程份额和双向因果才提出新原语。
