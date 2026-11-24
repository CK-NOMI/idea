# team-himerge/R05 | FCMC 联邦预测间隔证书约束
2026-11-24，起始 main 1fe2fa6cf306a19953c6afc8f2118f0e26da2fcb，README blob 46416c231592d124498dbafa166e1d26bbe149fb。核对R04已读快照完整前缀；阅读新增 FVT-SD、CPG、HPRS及HC/R04、HieraCert/R05代码；归档前读a3a0ee76449e1cb593fbd3b18fb63915fd434b46新增FSOT和FIV-SD，分别为事务修正和动态残差树，与本轮不同。A/R03零空间保护、PRE-05、PRE-07、HieraCert/R01区间证书为近邻。本轮放松完整激活不变为有限输出决策不等式。没有独立重跑其他团队全部实验。
**CRITICAL / Reject and Pivot**。T=限定数学，E=实际CPU数值优化，P=代理，H=假说。CUDA不可用，torch2.10.0+cpu，transformers未安装，无HF缓存，访问Hugging Face DNS失败，未运行真实预训练模型、HFL多设备或SD生成。
## 理论与机制
冻结特征提取器h，独立可训练线性输出头W（不绑定embedding），区域持授权旧前缀特征与已验证greedy y。内生预测头或浅层提出候选，不部署独立完整草稿LLM。对锚点i、竞争词v，旧唯一胜者间隔m_iv=(W_y-W_v)h_i>0。更新Δ要求(Δ_y−Δ_v)h_i≥−(1−κ)m_iv，0<κ≤1，故新间隔≥κm_iv，greedy严格不变。约束AΔ≥b构成凸多面体，0可行；满足共同约束的更新之非负加权平均仍可行。局部可行不保证全局：二元w=(1,1)，各区域保护e1/e2；更新(1,−4)、(−4,1)各自可行，均值(−1.5,−1.5)使两预测翻转。候选全局QP最小化||Δ−u||²/2，其中u为普通聚合更新。低通信基线ray：η=min(1,min_{j:a_j u<0} b_j/(a_j u))，区域计算局部最小后树形min，精确最大安全步长。无负方向则1。零空间AΔ=0在A满列秩时禁止非零更新，间隔约束可保留学习能力；保护旧错误不能等同质量提高。若二元要求全部h的符号相同，仅允许正比例缩放（非零w）。同信息平坦可复制QP及ray。私有特征需授权或分布式约束查询；不声称DP。非线性主干变化、LoRA因子平均不能直接用此定理。
执行：仅同前缀、特征、位置、版本、配置均匹配的已认证greedy决策可复用，其余正常正式验证；若缺下一状态/KV仍需前向。普通AR版本缓存同样适用。随机语义：同正温度下独立全支持Gumbel噪声G，旧/新argmax对几乎所有共同G相同，当且仅当z'−z=c·1。若某两词差改变，取G_b−G_a在两个胜负阈值之间且其它G足够低，则非空开集上胜者不同，概率正，矛盾。故全噪声不变量等于softmax不变。二元logit差1→2，greedy同但p从.731059→.880797，公共U胜者不一致概率.149738。有限训练噪声不等于全噪声保证，参与训练的种子不能未经独立性审查用于生成。标准SD用真实p'、q及min(1,p'/q)、正残差补采可保持新目标，因果前缀和独立随机数下归纳序列正确；本轮未实现完整随机SD、EOS/KV/top-p/浮点认证，不能直接提交旧随机token。
## 实际运行
[probe.py](probe.py)，OPENBLAS_NUM_THREADS=1 python experiments/team-himerge/R05/probe.py。Python3.13.5 NumPy2.3.5 SciPy1.17.0，CPU FP64 SLSQP；4区域32维二元冻结特征，8/32/128锚点每区，教师漂移σ=.25/1/2，seed11/29/73，27组；每区256合成新特征、软标签、单步本地梯度步长32后平均，κ=.05。比较plain/local/ray/globalQP/null。全部A秩32；QP零锚点greedy翻转，最大约束违反3.14e−16；null近零更新。随机差异是二元共同U解析概率|p'−p|，非MC。下表三种子均值，最后列单次SLSQP CPU时间中位数；无真实模型、任务、网络或CI。
|σ|锚点/区|plain flip|local flip|QP保留单步CE收益|ray保留|QP随机差异|ms|
|---|---|---|---|---|---|---|---|
|.25|8|.010417|0|.993441|.677425|.004334|.136|
|.25|32|.028646|.023438|.848954|.225060|.003782|.205|
|.25|128|.020833|.016927|.703512|.072390|.003134|1.729|
|1|8|.093750|.062500|.937336|.439484|.015167|.142|
|1|32|.080729|.052083|.693947|.063750|.011802|.311|
|1|128|.065755|.043620|.409990|.019509|.007637|2.659|
|2|8|.177083|.135417|.864601|.284620|.022255|.194|
|2|32|.117188|.070312|.604518|.039246|.017365|.390|
|2|128|.106771|.058594|.288810|.011716|.009041|2.834|
实际逐种子选定指标见results.txt；代码重跑完整JSON。QP保护全部锚点但高漂移密集约束牺牲约71.1%普通更新单步收益，非长期收敛或生成命中率。
## 文献和成本
2026-11-24核对原始论文页面/摘要，非全篇公式审查：[BCWI EACL2024](https://aclanthology.org/2024.eacl-long.174/)权重插值；[Positive-Congruent Training CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html)兼容学习；[Backward-Compatible Prediction Updates NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/012d9fe15b2493f21902cd55603382ec-Abstract.html)有限重评预算；[GEM NeurIPS2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)约束梯度投影；[DySD EMNLP2026](https://aclanthology.org/2026.findings-emnlp.45/)目标演进时在线适配；[When Drafts Evolve ICML2026](https://proceedings.mlr.press/v306/qian26k.html)在线学习；[Acceptance-Aware Draft Training](https://arxiv.org/abs/2609.24150)2026接受长度；[Gumbel-Max SCM ICML2019](https://proceedings.mlr.press/v97/oberst19a.html)共同噪声耦合；[Speculative Streaming EMNLP2025](https://aclanthology.org/2025.emnlp-main.986/)内生多流候选。无正式文字查重，盲区为最新约束FL/缓存及未索引工作。
M个锚点V−1约束/锚点，输出头维度Vd；结构化传h、y及margin或分离查询，仍需竞争词检查。d1536,V128000,M1000时FP16特征3.072MB，全量FP32 margins约512MB，非通信下界；ray每边缘仅回传1标量但仍有更新下发、全部约束检查；QP需多轮或集中。若输出头可节省占比f、真实缓存命中c、额外开销h，则固定其他阶段下节省≤fc−h；fc<.15不能达到G5。未测实际f,c,h。强基线普通AR版本缓存、普通HFL+SD、后验输出检查、BCWI/GEM、DySD、同信息平坦。缺少HFL与SD双向不可替代性。
G1 FAIL，主观创新6.025/10（问题7、机制5.5、理论6.5、价值6、距离4，权重25/25/20/15/15）；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（限定greedy定理，完整SD未验）；G4 FAIL合成；G5/G6未验证，无80%依据；G7部分代码与结果、整体未过。好结果同质量预算最强基线真实总wall time−15%或有效工作+20%，两预训练尺度三配置三种子CI及全部G。未执行：1–2×40GB，Qwen2.5-1.5B/3B独立头、8逻辑客户端2边缘，公开数学/代码/通用任务，三非IID/时延三种子，先测真实前缀重复、头耗时、保护约束开销、质量、通信、TPOT/P95、总时间与CI。单卡模拟非物理并行。**Reject and Pivot**，保留约束闭合反例、greedy与全噪声限制和实际数据；普通兼容学习、缓存、投影不得换名重投。下一轮优先获取真实模型/可复核trace和实际数据驻留约束，测强基线后尚有≥15%可优化关键路径；无资源则证据审计，不靠toy提高评分。
