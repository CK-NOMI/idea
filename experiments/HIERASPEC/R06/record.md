# HIERASPEC/R06｜FEC：精确自投机耦合与联邦训练传输熵的冲突
日期2026-11-24；起始main 02d70efca8ef4337672993164064b22791f5e813，README blob 2e3c012e6a571153ddd80868d097109e6fc0a657，2349行。完整获取README并核对R05归档为原样前缀，阅读新增QCRA/FCF/FRGB/CFA全部README、FCF详细记录相关部分、CFA脚本及HCA/R05梯度记录。查阅R05外部源指标并做受限传输代理计算。提交前阅读并发HCR/R06 SCA（历史轨迹精确性与源码审计，不同对象）。未复跑他队。与HC/R03/HCA/R05稀疏梯度、HCA/R04通信残差、R02隐私、R04摘要及A/R11概率耦合近邻；本轮不同是固定双边缘概率的接受率与条件源编码熵冲突，非梯度/混合/隐私机制。
**CRITICAL / Reject and Pivot**。T=限定理论，E=实际CPU有限概率整数矩阵枚举及公开日志数值重分析，P=代理，O=已知p,q、共同码表oracle，E-H=外部真实模型日志非本队复现，H=未证。没有本轮真实LLM、HFL、GPU、实际网络、完整编码器/解码器、wall time或CI。

## 1. 系统和机制（H/T）
Cross-silo HFL：客户端私有non-IID文本，各区域通过正式HFL学习区域教师/个性化模型p_e；共享主干/早退或轻量内生头q_e（无独立完整草稿LLM）。固定正式模型版本、同词表、授权公共prompt/教师生成文本可传至云作样本蒸馏，不传原始私有数据，不声称DP或合成文本无泄漏。正式目标是每区域固定p_e的序列，不是把混合模型冒充原p。若区域不允许发布任何轨迹则场景无效；若云已有完整教师p和同前缀可免费生成，则传输动机减弱。提出在教师端和接收端共享草稿候选X（前缀、头、hidden/KV和随机数必须能够真实重现，不能免费假设；若不同则传X并计费），教师输出Y~p。云需要重建教师实际Y以按原样本目标训练。将概率耦合同时作为候选验证核和通信预测对象。
给定q,p，联合质量Gamma_xy>=0、行和q_x、列和p_y。采X~q，验证端采Y|X=x=Gamma_xy/q_x；若Y=X则接受，否者提交Y、截断候选后缀。实数单步边际Σ_x Gamma_xy=p_y。多步须每到达前缀真实q、条件耦合和独立随机性、无未来候选窥视、正确bonus/EOS/KV；可逐位置归纳同p。greedy独立用正式argmax，概率耦合非其证明；温度/截断须先定义真实p,q，生产top-p/有限精度/完整实现未审计。
目标A(Gamma)=Σ_x Gamma_xx，C(Gamma)=H(Y|X)=−Σ Gamma_xy log2(Gamma_xy/q_x)。给定lambda>=0，最小化C+lambda(1−A)，lambda仅为假设的每单位失配相对通信代价（bit/token单位）；也可先固定最大A再最小化C。实际块长度、验证成本、网络、码表和额外训练需另计。对于已共享条件码表、接收方仅用X及消息而无其他相关随机性、固定同分布独立样本的前缀/无损可唯一解码编码，平均码长≥H(Y|X)，条件Shannon码每符号小于H+1；长块算术编码可趋近该率（计模型/精度/终止）。非所有共享随机通信协议的下界：若双方掌握完整p和公共随机数可直接得到Y；若有额外相关侧信息S，应考虑H(Y|X,S)或信道模拟；若码表只在教师端，不能宣称H是实际传输量。HFL训练可基于完整恢复的Y执行同原样本损失；反馈q更新可使用普通蒸馏，但没有证明正式HFL训练更新不可替代。相同信息平坦可复制。

## 2. 数学发现（T）
最大接受A*=Σ min(p,q)=1−TV(p,q)。标准最大耦合Gamma_xy=min(p_x,q_x)1{x=y}+[q_x−p_x]_+[p_y−q_y]_+/delta（delta>0）；它不是唯一最大耦合，一般也不是最低H。若q=(.8,.2),p=(.2,.8)，最大耦合矩阵[[.2,.6],[0,.2]]，A=.4，H=.8 h2(.25)=.6490225 bit。确定性置换Y=1−X，A=0,H=0，边际完全正确；证明两个目标可严格冲突。二元所有耦合形成线段，H+lambda(1−A)凹，最小值在端点；lambda<.6490225/.4≈1.62256 bit时置换更优（限定代理），不意味着实际SD速度更好。
保持A不变仍可降低H：q=(.4,.4,.1,.1),p=(.1,.1,.4,.4)，delta=.6,A=.4。标准残差把每个盈余行分给两个缺额列，每个盈余行条件(.25,.375,.375)，H=.8[h2(.25)+.75]=1.2490225 bit。改为每个盈余行的.3质量分别全送不同缺额列，保持相同对角线和边际，H=.8 h2(.25)=.6490225 bit，节省.6bit/位置，不损单步A。该稀疏剩余transport属于已有最小熵耦合/贪心质量搬运，不声称首创；最多V−1个非零残差边可由普通transport greedy实现（盈余和缺额集合互斥，实际非空节点≤V）。V=3时至少一侧盈余/缺额只有一个点，所以最大耦合唯一；实验中最优最大接受面与标准H相同是结构事实。一般V≥4才可能有自由残差耦合。
Fano型上界：D=1{Y!=X},H(Y|X)=H(D|X)+H(Y|D,X)≤h2(delta)+delta log2(V−1)，不是下界或实际码长。V=151936的固定宽Token ID需18bit；可直接发相等标志+不同token ID或对标志/修正ID熵编码。强普通AR可使用q算术编码，理想模型率H(p)+KL(p||q)（bit），若双方知道p则H(p)，也可用共享Gumbel/指数竞赛等减少协议消息。最低熵耦合本身NP-hard（既有文献）；对整数边缘运输多面体的顶点均为整数（全幺模约束矩阵）；H+线性失配惩罚为凹函数，最小值存在于顶点，故枚举所有可行整数矩阵包含连续域全局最优（浮点容差）；最大接受面亦为原多面体的面。该性质只针对固定有限整数概率，不能推广真实任意精度大词表可解性。
信息缺陷：优化Gamma通常需要p,q联合全量信息；若教师p私有且不传码表，云不能直接用H(Y|X)最优条件码。若传V×V稠密码表或逐前缀传稀疏映射，其开销可能远超原Token ID；稀疏极点至多2V−1非零项，仍是O(V)每上下文，不保证可复用。即使完整p可共享且随机性也可共享，普通AR可使用相同编码/生成，无HFL必要性。普通SD用标准残差+相同熵编码，独立学习及平坦均为强基线。

## 3. 实际数值实验 E/O/P
[probe.js](probe.js)，[results.json](results.json)，node experiments/HIERASPEC/R06/probe.js。JS V8，LCG/Box-Muller seeds11/29/73，V=3，整数分母N=24，p=softmax(高斯)，q=softmax(同z+sigma*独立高斯)，均用最大余数法量化为正整数总24；sigma=.15/.7/1.5，100上下文/seed/sigma，共900合成配置（p跨sigma重用）。枚举所有合法3×3整数运输矩阵，实际共204354个（91496+69645+43213）；每组比较标准最大耦合、最大接受面最低熵、无约束最低熵、7个lambda=0/.25/.5/1/2/4/8、普通AR q交叉熵及p entropy oracle。输出精确概率质量指标，不是采样频率或实际编码器性能。已重新读回代码重跑，JSON逐值相同。三seed均值：
|σ|最大A|标准H(bit)|最小H|最小H时最高A|冲突比例|AR q交叉熵|AR p熵|
|---:|---:|---:|---:|---:|---:|---:|---:|
|.15|.968611|.130913|.124814|.948611|.033333|1.329551|1.321240|
|.7|.849444|.456542|.361398|.677778|.500000|1.451104|1.321240|
|1.5|.717778|.694425|.526114|.503056|.756667|1.773330|1.321240|
冲突=所有最低熵解中的最高A仍低于最大A（容差1e−12）。三词最大接受面无额外自由；四词解析反例另见上。σ1.5时lambda .0/.25/.5/1/2/4/8的A .503056/.542639/.608194/.661111/.701944/.717500/.717778，H .526114/.532576/.556263/.594725/.652714/.693150/.694425。数值上最低熵降24.24%，但A绝对降21.47pp；非真实模型质量/通信或wall time。各配置Fano上界均值.208099/.701815/1.029088 bit。无实际entropy coder、网络、训练、CI。
E-H/P附加审计：本队R05已归档固定外部Qwen2.5-7B greedy日志 [source_metrics.json](../R05/source_metrics.json) blob f0c7d555b4c42ad40f869e4dcc3251c7a5ecbdc4；其中MTP1/2各100条输出匹配但逐token目标执行（不能当高效SD）；实际读取及重新求和 target_forwards F=12204/12188，accepted A=3272/4912，steps S=4466/3638。若接收方免费重现每块候选、只发未接受提交Token的18bit ID与每块接受长度ceil(log2(K+1)) bit，代理B=18(F−A)+S ceil(log2(K+1))，分别165242/138244 bit，对照18F=219672/219384 bit，约24.78%/36.99%下降。F含末尾截断前额外处理，不等于实际发送token数，K=1/2；不是实际编码、通信、独立随机耦合或准确块协议数据；未含seed同步、特征/模型版本、请求边界、真实回退、报头、字节对齐、重算、训练。源全greedy且无逐位置p/q/候选轨迹，不能用来估计本轮熵或证明随机正确性。复算片段：const a=data.methods.MTP1.rows; const sum=k=>a.reduce((s,x)=>s+x[k],0); const F=sum("target_forwards"),A=sum("accepted_draft_tokens"),S=sum("speculation_steps"); const B=18*(F-A)+S*Math.ceil(Math.log2(a[0].draft_len+1)); MTP2同式。
P成本：若纯通信节省比例s、该通信在强基线总时间的暴露比例f，忽略重算/码表/学习且串行可消除，则总体节省≤fs；s=.3699时需f≥.4055才可能满足15%（若其它环节无收益）。1024个18bit token仅2304B，100Mbps理想传输约0.184ms；相比模型/hidden/更新常更小，真实WAN RTT另计，压缩比特不自动减少RPC。若发送d1536 BF16 hidden需3072B/位置，远超过省下Token bits；若接收端必须额外跑共享主干重建hidden，计算也可能抵消。实际无网络测量。优化lambda为cost proxy不代表G5。

## 4. 相关工作（2026-11-24）
实际检索 speculative decoding channel simulation/exponential races, distributed inference communication, minimum entropy couplings, lossless language compression, federated distillation/side information。核对原始摘要或可访问HTML（DISCO 2610.35957正文及ERSS出版页），未穷尽全文/专利。
- [Speculative Sampling via Exponential Races](https://aclanthology.org/2025.findings-acl.936/) ACL Findings2025：SD与信道模拟、信息论速度界及ERSS，直接理论近邻。
- [Minimum-Entropy Couplings and Their Applications](https://arxiv.org/abs/1901.07530) 2019 IEEE TIT：最低联合熵、NP-hard及近似；固定边缘时最小条件熵等价，非新原语。
- [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) ICLR2024：预测模型和无损熵编码。
- [Coupling without Communication and Drafter-Invariant SD](https://arxiv.org/abs/2408.07978) 2024/2025：公共随机性、Gumbel耦合、不传完整分布；直接强基线。
- [Distributed LLM Inference via Communication-Efficient SD](https://arxiv.org/abs/2610.35957) 2026-10（其DISCO不同于动态前瞻同名论文）：服务端草稿副本、精确采样、标量/ID通信。
- [SpecFed](https://arxiv.org/abs/2604.25777) 2026-04：[TK-SLT](https://arxiv.org/abs/2509.04576) 2025：稀疏概率通信，必须区分实际截断后候选与近似残差。
- [Communication-Efficient Federated Distillation](https://arxiv.org/abs/2012.00632) 2020：公共数据软标签量化/压缩；[FedKD](https://www.nature.com/articles/s41467-022-29763-x) 2022：蒸馏与梯度压缩；[Wyner-Ziv Estimators](https://proceedings.mlr.press/v130/mayekar21a.html) AISTATS2021：联邦侧信息通信近邻。
盲区：最小熵耦合最新近似、限制通信的耦合前沿、因果序列编码、未索引工作、所有附录；无正式文字查重，不能称新信息论基础。

## 5. 验收与后续
CRITICAL：目标函数是已知最小熵耦合加失配惩罚，普通AR+熵编码、SD标准残差+编码、公共随机信道模拟、直接传修正ID、同信息平坦及联邦蒸馏可替代；最优码表/重算开销、真实训练通信占比及授权需求未知。HFL学习q改变耦合，但SD反馈仅可普通蒸馏，未改变不可替代正式学习规则。主观创新6.475≈6.5/10（问题7.5×25%、机制6.5×25%、理论7×20%、价值6×15%、距离4.5×15%），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（单步与条件序列T，真实完整greedy/随机/top-p/浮点未验证）；G4 FAIL（有限合成+公开greedy代理）；G5/G6未验证，无80%证据；G7局部代码/结果重跑可复算，完整未过。好结果同目标质量资源最强基线真实全流程−15%或有效工作+20%，两预训练尺度三非IID/时延三训练seed及CI、全部G。
H未执行：只有真实授权跨层轨迹传输为显著暴露瓶颈且双方可低成本复现内生候选时才投入1–2×40GB，Qwen2.5-1.5B/3B，8逻辑客户端2区域，公开数学/代码/通用语料，固定模型/随机性合同，采至少10000真实(p,q,X,Y,h版本,实际字节,RTT)位置；实际条件编码与码表更新测量，对照普通AR q算术编码、普通SD+修正ID/编码、DISD/ERSS、SpecFed/DISCO适用设置、普通HFL及同信息平坦；记录训练/推理/通信/回退/额外头学习时间、准确性、质量、CI，不以合成bit率当通信时间。若相同授权普通AR/SD匹配、接收端额外前向过高、暴露比例不足，则停止。当前不为该候选投入GPU。**Reject and Pivot**，保留接受率与条件熵冲突、同最大接受面稀疏残差反例、900组实际oracle、真实日志通信代理及码表信息边界。下一轮优先验证真实HFL数据驻留约束下的训练传输与执行任务（不是通用概率编码），在构造机制前取得真实各阶段profile。
