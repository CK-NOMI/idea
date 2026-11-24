# HC/R04｜FIV-SD：模型版本增量残差树与稀疏更新复用
2026-11-24；起始 main 965729e19917f83eba61d5ae907c9ea807b47622，README blob 4b41882464b8c4301465eef636f903a64ccc9335，1974行。取得完整README、验证HC/R03已读快照为原样前缀、阅读全部新增FVC/FVT-SD/CPG，读取HCA/R04脚本及部分逐种子数据、CPG/FVT-SD脚本，未独立复跑他队。提交前读并发HCA/R04 HPRS摘要及详细记录相关部分：其对象是未知归一化常数的单次分片race上包络，本轮是同一隐藏状态跨稀疏版本更新的持久顺序统计树；共同涉及残差归一化和p包络强对照，不宣称独立新数学领域。A/R01冻结层缓存、A/R03跨版本状态、HC/R03稀疏CE梯度是近邻，普通冻结缓存仍是强基线。
**CRITICAL / Reject and Pivot（保留增量残差计算资产，不立项）**。T=限定理论；E=实际JS V8 CPU合成数值实验；P=节点访问/存储/通信代理；H=待验证。环境实际检查PyTorch2.10.0+cpu、CUDA false、无transformers、无HF hub缓存、/mnt/data空；curl Hugging Face tiny-gpt2 config返回could not resolve host。未获真实预训练模型/执行trace，无GPU、真实HFL、网络、wall time或CI，不声称复现真实系统。

## 1. 对象、场景和机制（T/H）
固定同一已确认历史h，冻结正式主干及草稿特征产生器，采用未绑定输入embedding的线性输出头W_p,W_q；两者隐藏向量各自固定（可不同），模型版本仅改变已知少量输出行。HFL客户端持授权nonIID数据，区域聚合有效行更新，云发布严格版本；边缘持本地固定上下文缓存与实际p/q，允许稀疏行同步，不声称DP。HC/R03的冻结特征草稿CE差分每拒绝只改两个梯度行，可用无动量无衰减SGD使草稿更新支持稀疏；正式p可固定或额外有合法稀疏行更新，但本机制不能将原密集正式NLL梯度改为稀疏并宣称等价。普通真实联邦更新往往密集，场景待证。无独立完整草稿LLM；单一正式目标，不混合概率。
设a_v=exp(z_p,v)>0,b_v=exp(z_q,v)>0（固定尺度，温度1无截断），A=Σa,B=Σb。p_v=a_v/A,q_v=b_v/B。标准精确SD拒绝后需按r_v=(p_v−q_v)_+补采。问题：仅k个a或b变化时，能否不扫描V维重建残差？全局归一化改变，残差支持可能大规模改变：初始a=b=1，减小一个a后其余V−1词残差变正，增大则这些词变负。仅更新局部残差条目错误。
FIV：维护按(a_v/b_v,v)排序的平衡树，每节点存两种子树质量A_sub,B_sub及叶ID/权重。正残差支持恰ratio>A/B。每个完整正子树残差质量R=A_sub/A−B_sub/B。取阈值右侧suffix总和，沿搜索路径累加完整右子树和当前节点，O(log V)（平衡树）；抽U∼Unif[0,R_total)，按降序沿树以右子树质量和节点质量划分区间，阈值下节点仅进入右树，直至选中词。每个区间长度r_v故严格按r_v/R_total采样。更新k行仅删除/重插变化键并维护子树和，O(k log V)，平衡树空间O(V)，初建O(V log V)（排序后可线性建树）；实际代码为哈希优先级treap，未证明对抗输入最坏O(log V)，生产用AVL/红黑树。计算新的行logits需O(kd)，未知新隐藏向量需全部重算O(Vd)。树只用于已缓存固定(p,q,h)权重，q抽样另需合法动态采样或树权重查询；greedy另维护正式argmax，标准接受min(1,p_X/q_X)，拒绝独立残差抽样；逐位置真实条件p/q、版本锁定及正确KV/EOS/bonus时序列分布等于同一正式p AR。不得将跨token不同h复用成同一分布；top-p后大量支持可能变，需另处理；浮点比值、指数溢出及极小δ消差未审计，近零残差应回退稳定的参考计算，非逐位保证。
聚合闭合只在真实有效参数更新上：ΔW=Σw_iΔW_i，非零行属于局部支持并集；同一行更新先聚合再用新logits计算，不能直接平均exp或树键。m个独立均匀k行支持的期望并集V[1−(1−k/V)^m]（不代表语言实际分布），逐层并集可能迅速稠密。优化器动量可在零当前梯度行更新，AdamW衰减可改变全部行；共享输入embedding或训练主干使h变化。若使用普通dense NLL，输出行一般均更新。强行截断学习会改变目标质量；不得称无损。普通平坦同信息复制完全相同的行聚合/树。

## 2. 实际CPU实验（E/P）
[probe.js](probe.js)、[results.json](results.json)。node experiments/HC/R04/probe.js；实际V8执行，LCG+Box-Muller，3种子11/29/73，V1024/8192/32768，k=2/32/256，12版本/配置（27组、324次更新）。初始a=exp(N(0,1)),b=exp(同一logit+.2N)，每版随机k不同ID、随机一侧乘exp(.3N)；是合成权重更新而非真实模型梯度。每版重算全词表参考归一化/残差，8个共享U与按比值降序参考逆CDF对比；共2592配对残差样本0不一致，最大质量绝对误差4.676814491233472e-15。参考排序仅用于同U配对，普通dense残差抽样无需排序，不以排序时间为基线。树构建及更新/查询节点访问数计数，查询数含1质量+8采样；未测时间。三种子均值近似：
|V|k|每版本更新访问|平均每次查询访问|
|---:|---:|---:|---:|
|1024|2|47.9|12.4|
|1024|32|832.9|12.1|
|1024|256|6608.7|12.2|
|8192|2|63.2|16.2|
|8192|32|1056.5|16.4|
|8192|256|8508.8|15.9|
|32768|2|76.4|18.8|
|32768|32|1223.5|18.7|
|32768|256|9774.3|18.3|
访问不是FLOP或时间，与向量化dense scan每版V条残差计算不可直接等同；初建、缓存容量、新h、q采样与greedy未测。配对数值一致仅支持测试分布下实数机制近似正确，不证明所有浮点条件。没有实际SD整序列、模型训练或GPU。V8脚本及原始JSON为局部可复算证据，非完整R。

## 3. 成本及最强替代（T/P）
核心强基线：两个独立动态权重树分别采p,q，标准接受；拒绝后反复抽Z~p，以(1−q_Z/p_Z)_+概率接受。单次成功δ=TV，条件平均1/δ次提议；拒绝概率δ，故每实际候选位置无条件期望恰1次p提议（δ>0），δ=0不触发。p/q树O(k log V)更新、每次O(log V)采样；相同渐近平均复杂度，无需比值重排。若N为实际额外提议次数，P(N>n)=δ(1−δ)^n（n≥0），树的潜在独特收益仅为控制拒绝补采长尾及直接求δ，非平均复杂度优势；未运行该基线的实际计时。HCA/R04已讨论p包络及预采，必须引用。也可先进行有限次数独立p提议，失败后精确残差扫描，按正确条件核不会改变分布。普通AR动态p采样直接省去所有草稿/残差，且树可在同信息平坦复用。FlashSampling/FlashSpec等融合算子、缓存logits后GPU dense scan为实际强对照。
每头稀疏FP16行更新载荷k(2d+4)B（ID32位），全量2Vd；实际双头、客户端→边缘→云、广播、版本、聚合、特征、元数据及安全成本另计。每缓存上下文仅两权重FP64+两子树和FP64+ID32位就至少36V字节（未计ratio/优先级/指针/分配器），V128000至少4.608MB/上下文，1000个约4.608GB；如果仅缓存logits，2FP32×V约1.024MB/上下文。缓存初建O(Vd+VlogV)且跨版本需刷新k行，若h重复少，亏损；固定h的重复评分、前缀缓存及多次采样可用普通AR批处理。设强基线时间T中仅可复用固定h的残差/输出计算占f、命中率c、理想被省比例s，则其他阶段不变时净节省≤f c s−新增开销/T；f c<.15不可能单靠此达G5。没有真实f/c/s证据。
关键矛盾：HC/R03仅稀疏草稿蒸馏，不改变正式HFL训练；跨版本重用和HFL层级无必然关系，普通平坦和在线SD均可复制；不能因语法上共享更新而给G2通过。普通动态采样在同条件下平均复杂度相同，树只在有真实反复固定上下文、严格稀疏更新及尾延迟需求时可能有价值。

## 4. 文献（2026-11-24）
本轮实际检索dynamic categorical sampling, incremental softmax, sparse federated updates, evolving target SD, speculative vocabulary；核对原始摘要/页面，不声称全文逐式审查、正式文本查重。
- [DySD EMNLP Findings2026](https://aclanthology.org/2026.findings-emnlp.45/) 已研究目标更新下在线草稿对齐；[Online SD ICML2024](https://proceedings.mlr.press/v235/liu24y.html) 在线KD。
- [SpecVocab ACL Findings2026](https://aclanthology.org/2026.findings-acl.2000/) 输出投影开销及按上下文缩小草稿词表，非跨版本残差树；[When Drafts Evolve](https://arxiv.org/abs/2603.12617) 在线反馈与动态regret（本轮检索镜像，先前原始核查见HC/R03）。
- [Exact Efficient Sampler for Dynamic Discrete Distributions](https://arxiv.org/abs/2506.14062) 动态有限精度权重精确采样；[Maintaining Discrete Probability Distributions in Practice ALENEX2024](https://doi.org/10.1137/1.9781611977929.14)、[Efficient Dynamic Weighted Set Sampling VLDB](https://doi.org/10.14778/3617838.3617840) 已有动态权重结构，树非首次。
- [SecEmb ICML2025](https://proceedings.mlr.press/v267/mai25a.html) 稀疏embedding联邦更新；[Sparse and Aligned Adaptive Optimization](https://arxiv.org/abs/2405.17932)、[FedSparQ](https://arxiv.org/abs/2511.05591) 稀疏联邦优化；非本轮保证的精确SGD。
- [Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html)、HCA/R04 HPRS中列举的[FlashSampling](https://arxiv.org/abs/2603.15854)、[Speculative Sampling via Exponential Races](https://aclanthology.org/2025.findings-acl.936/) 为强近邻（后两篇本轮据仓库记录，未独立核对全文）。盲区：动态残差专门数据结构、精确浮点、最新融合kernel/未索引工作。独立写作不等于原创性检测。

## 5. 验收及下一步
主观评分问题7、机制6.5、理论7、价值5.5、距离4.5，权重25/25/20/15/15=6.30/10，非同行评审。G1 FAIL不足8；G2 CRITICAL FAIL（平坦、普通AR/在线SD、动态采样可替代，正式HFL学习未改变）；G3 PARTIAL/FAIL（限定实数和CPU检查，完整greedy/随机序列/浮点未验）；G4 FAIL仅合成；G5/G6未验证（无真实时间、无80%依据）；G7局部代码数据/整体未过。好结果同目标/任务质量预算下最强公平基线全流程真实时间降≥15%或有效工作+20%，两预训练尺度、三非IID/延迟、三种子、CI及全部G。
未执行H：若新证据改变前提，1–2×40GB Qwen2.5-1.5B/3B（核对是否绑定输出头，若绑定需冻结或明确改目标），8逻辑客户端2区域，先收集至少1万真实(p,q,h,version)位置；量同h跨版本实际复用率、参数行支持、累计union、缓存内存、残差尾延迟、dense/动态p包络及树真实GPU耗时。对比同目标AR、普通HFL+最优SD、OnlineSD/DySD、普通动态p/q、有限预采+dense回退、GPU融合kernel及同信息平坦；计实际训练/通信、额外缓存构建、内存带宽、重算、任务质量、TPOT/P95及CI。单卡逻辑模拟不称实际多设备。若f c<.15或同信息平坦/普通动态采样匹配，停止。当前不建议为已否决方案投入GPU。
**Reject and Pivot**：保留归一化改变引发全局残差支持变化的反例、动态比值树、p包络平均/尾部界、行支持与缓存成本、实际CPU数据。禁止把通用动态采样/冻结缓存/稀疏蒸馏重新包装为双向机制。下一轮优先真实预训练trace及真实HFL约束，不继续用toy替代实际问题；无真实资源则只做严格证据审计。
