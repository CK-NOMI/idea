# team-hca/R04｜HPRS：未知归一化常数下的参数化残差采样上包络

日期：2026-11-24；起始 main commit 60010389c564e5da3f61a681df38f923ae39e7e5，README blob 2e51d61014e8adc29c8dc30ba3d04092b0901aef，1781行。读取固定版本，验证R03已读历史原样保留，检查递归目录、新增HCR/FBE、HiMerge/JVT及HierCouple/R03、HIERASPEC/R03代码结果摘要。上传前重新读取NF-RSD、FLPS、FBS、FCR-SD，另读并发HiMerge/R04 FVC、HCR/R04 FVT-SD及HierCouple/R04 CPG，最新检查 main 965729e19917f83eba61d5ae907c9ea807b47622、blob 4b41882464b8c4301465eef636f903a64ccc9335。发现并发更新即停止旧README写入，改先保存独立报告再追加索引。未独立重跑他队。

**CRITICAL / Reject and Pivot（否决HFL×SD正式立项，保留通用精确SD算子线索）**。T=限定数学；E=实际CPU数值/MC；P=成本代理；E-H=外部论文报告实测；H=待验证假设。无本队GPU/预训练LLM/HFL实际训练/网络/wall time。实际环境检查Python3.13.5、torch2.10.0+cpu、CUDA false、0 GPU、无默认HF hub缓存；未取得真实模型trace。实验用JS V8 CPU。公开论文实测仅支持一般输出层瓶颈，不证明HFL特异瓶颈。

## 1. 场景与计算问题
client-edge-cloud HFL以私有non-IID数据训练一个共享目标模型，边缘/云聚合参数。固定推理快照，额外假设输出权重W按词表V_g不重叠分片，由区域/区域内设备持有并授权执行hidden h。p=softmax(W h)，不是多个概率模型混合；区域持各自目标logits及实际内生草稿q的对应分片，协调端有候选X和q(X)。无独立完整draft LLM；可用LayerSkip/MTP/EAGLE式内生头。实际q生成、分片归一化、分发、缓存、版本和hidden交换都计费；不声称DP。普通HFL可完整复制模型，分片是额外部署假设，必须证明实际权重驻留/资源限制。
训练CE L=-ℓ_y+log Z，Z=Σ_v expℓ_v；局部Z_g可聚合，hidden梯度Σ_v expℓ_v W_v/Z−W_y可合并d维加权矩。已有CCE及词表并行，W梯度仍需每行p、保存或重算。SD随机接受和正残差都依赖全局Z。本轮问：在未知全局Z时，各分片能否仅扫描一次本地目标输出，预备可合并精确残差摘要，避免拒绝后的第二次读取或依赖往返？优化同质量预算下实际总训练+推理时间，含全部额外开销，不改正式目标。

## 2. 机制及理论 T
设a_v=expℓ_v，Z=Σa_v，p_v=a_v/Z，q为真实归一化草稿概率。标准SD：X~q，以min(1,a_X/(Z q_X))接受，否则残差权重r_v=(a_v−Zq_v)_+。预取独立E_v~Exp(1)，令直线L_v(t)=a_v/E_v−t q_v/E_v。若δ=TV(p,q)>0，argmax L_v(Z)>0且等于argmax r_v/E_v。E_v/r_v独立指数变量，rate=r_v，最先到达的概率为r_v/Σr。噪声独立于候选、接受判定、模型。接受质量min(p,q)+残差质量(p−q)_+=p。δ=0无拒绝。逐位置用真实条件q、独立随机数、标准截断/bonus及合法KV可归纳正式序列精确，尚未实现审计多步、EOS、top-p、浮点。Greedy正式argmax可分片归约。q为点质量时候选残差为0、其余为a_v，普通单pass race即可，FlashSpec Part I已研究。
定义分片摘要(Z_g,H_g,a_X若拥有候选)，H_g为t≥Z_g时的上包络支持线。因为全局Z≥Z_g，裁剪区间外的线不会改变查询。合并A、B：Z_AB=Z_A+Z_B，H_AB=Hull(H_A∪H_B,t≥Z_AB)。父域包含于各子域，max合并保持精确，递归根求t=Z即可；根也可直接遍历收到的支持线。tile同理。数学上允许一次读取logits生成局部摘要，但普通排序O(n log n)，当前实验保存完整数组作为oracle，未实现真正fused kernel。最终hull线数不等于峰值临时空间或FLOPs。
若q也以未归一化b_v给出，t=Σa/Σb，L=a/E−tb/E；需合法全局比值下界（或t≥0），不能直接用Σa_local，未实验。实际需要共同稳定尺度，防指数溢出及接近p=q时消差，生产浮点精确性未证。

## 3. 反例与回退 T
最坏有n条支持线：j=0,...,n−1，m_j=−(n−j)，b_j=2n²−j(j+1)/2。相邻交点1,2,...,n−1，每条线在t∈(0,n)某区间为唯一正最大。令a_j=εb_j，q_j=−εm_j，ε充分小使Z_g<.5且Σq_j<1；E_j在ε邻域有正概率，严格比较保持。其他分片质量可令全局Z位于各区间，并可令其他词残差非正（局部Σa/Σq>n）。因此用支持线子集表达所有合法阈值的最坏需求为n；非任意编码通信下界。经验短摘要不是普适保证。
硬容量回退必须保留同一随机race（例如counter-based RNG重建E），因为溢出事件依赖E；若在此条件下重新抽样，可能偏置。反例两个等权残差token，若E1<E2就改用新独立样本，否则用原race，最终token1概率.25、token2 .75而非.5/.5。非声称特定hull cap必有此偏差，而是必须证明条件核或重用原噪声。若固定回退事件独立于E，重新抽样可正确。溢出标记仅实际拒绝时才需回退，完整扫描或重算都计费。

## 4. 实际CPU实验 E/P
[probe.js](probe.js)，[results.json](results.json)，node experiments/team-hca/R04/probe.js。实际V8运行并保存逐种子结果；LCG/Box-Muller供复算，非生产RNG。不测wall time。
A：4分片；V=1024/8192/32768，σ=.1/.5/1.5，seed11/29/73，每种子配置48行，共1296行。p=softmax高斯，q=softmax(base+σ独立高斯)，a=Vp（根Z=V，局部用本地质量）。完整race与局部hull同E，0不一致。20B/线（2 FP64系数+32bit ID）+4×8B质量，未计候选及报头。均值：
|V|σ|平均线数/片|任一最终>8|任一最终>16|总摘要B|解析接受率|
|---:|---:|---:|---:|---:|---:|---:|
|1024|.1|6.10417|.18750|0|520.33|.96011|
|1024|.5|5.90972|.14583|0|504.78|.80228|
|1024|1.5|5.70833|.11111|0|488.67|.45248|
|8192|.1|7.24479|.57639|0|611.58|.96019|
|8192|.5|7.18229|.56250|0|606.58|.80281|
|8192|1.5|7.13021|.56250|0|602.42|.45366|
|32768|.1|7.90104|.82639|0|664.08|.96009|
|32768|.5|7.87153|.84722|0|661.72|.80244|
|32768|1.5|7.96354|.84722|0|669.08|.45267|
B：32768词，4片，128词tile，a=exp(base)且Z随行变化；同三种子三σ每配置24行，共216行。实际增量hull合并与完整race零不一致。σ=.1/.5/1.5最终线数均值7.90625/7.89583/7.99306；片内合并历史峰值均值9.84028/9.68056/9.72917，任一片峰值>8比例.98611/1/1，>16均0；观测最大峰值15（不含临时tile数组）。8条cap几乎总溢出；16条在有限合成样本无溢出，非保证。未实际执行hard cap回退。A+B=1512行完整/摘要一致。
C：p=(.35,.25,.25,.15),q=(.15,.35,.1,.4)，两片，共同尺度3.7。理论接受.65，残差(.2,0,.15,0)/.35；每seed100000次，合计300000次实际单步MC。完整/摘要argmax零不一致：
11：接受65087，输出[34803,24941,25106,15150]，残差[19785,0,15128,0]
29：接受65056，输出[34968,24993,25042,14997]，残差[19941,0,15003,0]
73：接受64831，输出[34818,25132,25054,14996]，残差[19998,0,15171,0]
汇总接受.6499133，输出(.34863,.25022,.2506733,.1504767)，经验TV=.00137；有限抽样波动不证明严格浮点精确性。无真实LLM、HFL训练、GPU、网络、实际多步或CI，非完整R级。

## 5. 强基线与成本 P/T
32768词FP32全量权重131072B，本法σ=.5摘要661.7B，弱基线载荷约降99.5%。最强简单精确双阶段：4片先传每片8B局部Z+候选所属片8B a_X，共40B；拒绝才广播Z每片8B，并回每片8B局部race score+4B ID，共80B附加。平均40+80δ B；δ≈.19756时55.8B，本法含候选约670B，约12倍载荷。二阶段本地保存/重算logits，支持并行归约、只扫描首拒绝位置，不需全量上传。两者均未计共同请求、hidden、候选q分发、回传token等费用。
更强预采M个独立p样本：每片用race返回局部最大及其a,q，根取全局winner，得到Z后以[1−q/p]_+接受每个p提议。拒绝后的单次成功δ，M次均失败才精确残差回退，无条件回退δ(1−δ)^M（本队R01已有p包络思想）。回退独立于后备残差随机性时可重新抽样。M=4、4片、每提议每片28B(score/ID/a/q)，合计448B+40B=488B；σ=.5无条件回退约8.2%。需多个race但可并行，不能忽略开销。
理想单位置串行时间差ΔT=ΔC+Δbytes/BW−δ RTT。σ=.5、有效100Mbps聚合带宽，额外约614B→.049ms；若ΔC=0需RTT>.25ms；若额外计算1ms需>5.3ms才有净收益。非实测，真实并发/拥塞需另建模。K=4草稿且独立恒α=.80244时至少一拒绝约.585；双阶段可并行各位置Z并只修正首拒绝，本法要为所有候选构造摘要，不可把单步收益乘K。Bonus共通。
普通AR+FlashSampling、确定性draft+FlashSpec、普通SD双阶段、预采候选、Block Verification/ERSS适用版本、CCE/词表并行、普通HFL+SD与同信息同物理拓扑平坦/TP为强基线。普通TP/平坦能完全复制包络。HC/R03 FCR-SD已提出固定教师耦合差分CE梯度，HPRS也可返回(X,Y)，但普通SD一样，不能拼接冒充新训练机制。HFL训练依旧普通CE，SD没有不可替代反向作用。若可消除部分全流程占比f<.15，即使完全消除也无法达到G5时间门槛。没有HFL真实瓶颈占比，不能据公开论文TPOT外推。

## 6. 文献核查（2026-11-24）
查询 speculative decoding vocabulary parallel residual softmax normalization streaming one pass fused kernel; parametric Gumbel/exponential race sampling affine weights upper envelope; fused linear cross entropy; vocab sharding; client-edge-cloud HFL。核对原始页面/摘要及部分HTML正文：
- [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009) 2024，CCE输出线性层+在线logsumexp，论文报告Gemma2 2B交叉熵显存24GB→1MB。
- [FlashSampling](https://arxiv.org/abs/2603.15854) 2026，LM-head与Gumbel-max融合、词表tile/TP精确归约；原始摘要报告部分vLLM模型TPOT最高降19%，版本数字可能不同。
- [FlashSpec Part I](https://jialefu.github.io/blog/2026/flashspec/) 2026-05-14作者技术博客，确定性草稿的单遍验证/恢复，直接近邻但非已确认同行评审论文。未找到确认的随机草稿Part II，不证明不存在。
- [Megatron-LM](https://arxiv.org/abs/1909.08053) 2019，词表/张量并行；[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) 2018，在线归一化。
- [Speculative Sampling via Exponential Races](https://aclanthology.org/2025.findings-acl.936/) ACL Findings 2025，指数竞赛SD已有，通道模拟/多候选；未完整逐式排除参数化包络重合。
- [Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html) ICML2023；[LayerSkip](https://aclanthology.org/2024.acl-long.681/) ACL2024；[HierFAVG](https://arxiv.org/abs/1905.06641) 2019/ICC2020。
盲区：参数化几何上包络、加权reservoir/动态抽样、最新未索引fused sampler、专利、真实CUDA kernel、完整近邻附录。无正式文字查重；不声称经典指数竞赛或几何工具首次。其他团队NF-RSD为中位数目标，FLPS为分词，FBS为halo梯度，FVC为共批容量，FVT-SD为验证激活，CPG为关键路径；均不同，强平坦/AR替代结论共同适用。

## 7. G1–G7及后续
主观加权创新（问题8×25%、机制7.5×25%、理论7×20%、价值7×15%、文献距离6×15%）=7.225/10≈7.2，非审稿意见。G1未过：不足8、近邻未完全排除；G2 CRITICAL未过：普通TP/平坦复制、正式HFL学习无反向新机制；G3局部T/整体未过：单步实数及条件序列理论、CPU MC，greedy/完整多步/有限精度实现未验；G4未过：合成非真实双尺度；G5未验证：无实际全流程时间；G6未验证：无80%证据；G7局部可复算/整体未过。好结果仍同目标质量预算最强基线实际全流程时间−15%或有效工作+20%，两预训练尺度三种子三异质/网络设置及CI、全部G。无成功概率或发表概率估计。
仅在真实词表分片需求/输出头占比证实后进行未执行预实验：1–2×40GB NVIDIA、Qwen2.5-1.5B/3B，8逻辑客户端2边缘，公开数学/代码/通用文本，先采≥10000真实(p,q)行，测hull线数/临时峰值/溢出/数值边界与输出层实际耗时；GPU tile fused实现对比同目标FlashSampling/FlashSpec（各自适用）、双阶段、预采M、普通HFL AR+CCE、普通HFL+SD及同信息TP/平坦。测TTFT/TPOT/P95、吞吐、HBM读写/显存、字节/RPC、总训练推理时间及CI。单卡模拟不冒称物理分布式。真实hull过大、排序抵消收益、双阶段/预采匹配、瓶颈占比不足即停止；通用SD算子收益也不等于HFL双向创新。
**Reject and Pivot（HFL×SD目标）**。保留上包络精确残差、域裁剪可合并证明、最坏线数反例、随机自适应回退警告及实际数据。下一轮先获得真实小模型或公开原始执行trace，核算共享训练—解码瓶颈占全流程时间及数据驻留约束，寻找学习对象与执行规则双向因果，不以普通TP、共批、缓存、反馈蒸馏换名。资源缺失如实未验证，不用toy数量替代真实证据。
