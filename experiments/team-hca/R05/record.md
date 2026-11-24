# team-hca/R05｜FCKG：最大耦合下的双向KL稀疏梯度与拒绝门控方差障碍

日期：2026-11-24。研究起始 main dcf9635cbc5b33efcc487889ef00c93f411efd26，README blob 9a5af995c55e9c7d47c01eac1a43fa6fedb8ba19，2050行。读取固定快照，核验上轮3779878846fdb6827f4031dbd3a2dd33d8937eb3原样前缀，检查新增FSOT、FIV-SD、LBS及R05相关代码/记录，特别读取HiMerge/FCMC与HierCouple/R05脚本/数据。写前读取并发FCRR-SD、NC-FSD、FCMC，最新已读 main c7cc52fc57212dacbe1902ab0c9d40dc50baf407，blob 79d2fd9195d509524f6ad696e7c31743341529a6，2177行。未独立重跑他队。与HC/R03固定教师CE差分梯度直接相邻，不能重复声称其创新；本轮研究双向KL的正式路径项、额外目标样本、共享头行支持与拒绝门控方差下界。与HiMerge/R03联合接受长度目标不同；本轮不再重复R04归一化残差算子。

**CRITICAL / Reject and Pivot**。T=限定数学推导；E=实际CPU概率矩枚举/MC；P=方差、通信及工作代理；O=完整概率或最优基线oracle；H=假说。外部论文结果非本队复现。环境实际检查Python3.13.5、torch2.10.0+cpu、CUDA false/0设备、transformers未安装、无默认HF hub缓存；未获得真实预训练模型或执行trace。实际数值程序JS V8 CPU。无真实LLM/HFL训练、GPU、网络、wall time、正式CI。不能据代理或外部结果宣称成功概率。

## 1. 数学对象、系统、机制

固定同一合法外生前缀h和参数快照θ；客户端私有非IID数据，边缘授权共享主干/内生MTP或早退路径，云线性聚合正常训练更新。正式p=softmax(z_p)，草稿q=softmax(z_q)，两者全支持、同词表、同温度（温度吸收入logits）；可共享线性LM头W，z_p=W h_p,z_q=W h_q。不是独立完整草稿LLM。实际候选X~q与正式Y~p来自标准最大耦合SD，X被接受的概率1−δ，δ=TV(p,q)。只有真实因果条件q、独立随机性、合法修正/截断/bonus/EOS/KV才可归纳完整序列p；greedy由正式argmax；真实top-p/有限精度/完整实现未审计。
研究对象D=KL(p||q)，区别于冻结p的KL/CE蒸馏。若p也参与D梯度，则F=NLL(p)+λD是新训练目标，不能宣称原NLL更新不变；需同时保证正式质量。每个h的固定快照梯度与抽样上下文分布的梯度是不同对象：若h依赖θ且需要全路径目标导数，则须加路径score等项，本轮只证明固定外生前缀条件梯度。普通HFL原NLL梯度继续独立算。
令f_v=log(p_v/q_v)，μ=Σp f。精确logit梯度g_p=p⊙(f−μ)，g_q=q−p。令Z~p为与(X,Y)独立的同一前缀额外正式样本（不是下一位置bonus或另一个客户端不同h的token）。构造 G_q=e_X−e_Y，G_p=.5(f_Y−f_Z)(e_Y−e_Z)。由Y,Z iid p及协方差恒等式，E G_p=E_p[f e]−E_p[f]p=g_p；E G_q=q−p。所有样本、系数detach，分别反传到实际z_p,z_q；不能对f或抽样路径再次autodiff。共享线性头梯度G_W=G_p h_p^T+G_q h_q^T，行支持⊆{X,Y,Z}，每个上下文至多3行；h_p/h_q若依赖共享主干仍需相应VJP且可能稠密。f_Y−f_Z=(z_pY−z_qY)−(z_pZ−z_qZ)，归一化常数抵消；但实际X,Y,Z生成和所需4个logit取值不免费。可用surrogate .5 stopgrad(f_Y−f_Z)(z_pY−z_pZ)+z_qX−z_qY，样本索引固定。同步一次线性聚合无偏，非多步FedAvg/AdamW无偏参数结论。非IID加权须按预定上下文权重，不能只除以拒绝样本数；旧版本与参与训练历史样本的条件分布须重证（FCRR-SD/LBS）。不保证隐私。
T：teacher两样本估计就是RLOO两样本协方差形式，非新数学；q项就是HC/R03差分。联合构造最多3行并消掉logZ是受限接口价值。普通AR额外采Y,Z并独立X或人为最大耦合同样可实现，平坦同信息同委托可完全复制；这不是已证明HFL特异原语。标准SD的目标采样不改变；但本轮双向训练目标可改变新模型，不能将新旧模型的质量或采样分布混同。

## 2. 理论、反例与致命限制 T

(1) 记V(G)=tr Cov(G)=E||G||²−||EG||²。最大耦合G_q有V_q=2δ−||q−p||²，独立X,Y差分V_ind=2(1−p·q)−||q−p||²，普通硬标签CE logit梯度q−e_Y的方差V_hard=1−||p||²。最大耦合只在差分样本类最优，不保证强于硬标签或全概率。
(2) G_p方差V_p=Var_p(f)−||g_p||²：当Y≠Z时||e_Y−e_Z||²=2，而Y=Z差值为0，故E||G_p||²=.5 E(f_Y−f_Z)²=Var_p(f)。普通一p样本score估计(f(Y)−b)(e_Y−p)的迹方差=Σ p_v(1−2p_v+||p||²)(f_v−b)²−||g_p||²；最优常数b为上述非负权重下f平均（oracle）。此baseline需要p向量或等价反传且不需要第二个p样本；两样本稀疏不意味着方差更小。对单位h_p,h_q且内积c，G_W的Frobenius方差迹=V_p+V_q+2c E[(G_p−g_p)·(G_q−g_q)]；须计交叉协方差，不能仅加两方差。在实验c=.5作代理。
(3) **拒绝门控普适限定下界**：任意无偏向量估计H满足E H=g且在标准SD接受事件上H=0。由Cauchy–Schwarz，||g||²=||E[H 1_reject]||²≤δ E||H||²，因此V(H)≥(1/δ−1)||g||²。可用于正式KL梯度g_p，不意味着所有允许接受时更新的估计器都受此下界，也不是通信复杂度定理。对q梯度g_q范数通常随δ收缩；对p梯度则无统一收缩，除非额外控制log-ratio。
(4) **高接受率仍可有非零正式梯度**：p=(ε,1−ε),q=(ε exp(−1/ε),1−ε exp(−1/ε))，ε→0，δ=ε(1−exp(−1/ε))→0，α→1。f_1=1/ε，f_2=log((1−ε)/(1−ε exp(−1/ε)))→0；g_p1=ε(1−ε)(f_1−f_2)→1，g_p2→−1。于是任何仅拒绝时更新的无偏正式logit估计V≥(1/δ−1)||g_p||²~2/ε→∞；本轮G_p两样本方差也~1/ε（并非解决稀有尾部）。这是正概率实数序列反例，极小q在数值实验会下溢，使用解析log ratio，不宣称浮点真实模型测量。高接受率不能保证双向KL训练容易，不能直接把HC/R03零跳过推广到正式路径。若p=q严格相同，D梯度为零但NLL仍未必零。
(5) **质量退化**：固定数据r=(.8,.2)、受限q=(.2,.8)，p优化NLL_r(p)+λ KL(p||q)，p=.8时NLL最小，但KL项在该点的p标量导数log(16)>0，任何λ>0均将最优p向低质量q移动（严格凸、内点唯一解）；λ→∞时p→q，接受率→1、额外NLL→KL(r||q)=.6 ln4≈.831777 nat。对原正式学习不一定有益；与HiMerge/JVT接受损失和HCA/R03坐标退化有关但目标不同。Soft-MTP明确采用stop-gradient teacher并保留共享头CE，直接提供更强的质量/计算基线。
(6) 若把Z换成其他位置的bonus或受同一验证事件影响的样本，协方差恒等式可能失效；例如Z=Y时G_p恒0而g_p可非零。HFL按区域平均不能修复此偏差。

## 3. 本轮实际执行 E/P/O

[代码](probe.js)，[逐种子原始结果](results.json)，运行 node experiments/team-hca/R05/probe.js（无依赖，LCG+Box-Muller供复算，非生产RNG）。实际V8计算，V32、seed11/29/73、σ=.1/.5/1.5、每配置160个高斯概率上下文，共1440。p=softmax(standard normal logits)，q=softmax(z+σ normal)。精确有限概率矩枚举最大耦合(X,Y)及独立Z，未训练模型。共同单位hidden内积c=.5是可实现的共享线性头代理，不是实际hidden。比较普通独立差分、硬CE及最优单样本teacher baseline（oracle），三种子均值：
|σ|δ|teacher两样本V|teacher最优score V|q耦合V|q独立差分V|q硬CE V|联合Frobenius V(c=.5)|硬CE+同teacher V|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|.1|.037824|.008551|.008046|.075090|1.855915|.928186|.081730|.936744|
|.5|.187007|.213769|.201154|.359622|1.844246|.928186|.534140|1.141993|
|1.5|.504160|1.923922|1.810383|.872137|1.737884|.928186|2.606993|2.852220|
联合独立差分V=1.864474/2.058052/3.661918。q耦合相较硬CE降91.9/61.3/6.0%，与HC/R03同类已有结论；联合相对硬CE+同teacher降91.3/53.2/8.6%，而teacher稀疏RLOO方差高于最优score约6.3%（三种σ均近似相同）。全概率精确梯度条件方差0，强oracle。注意联合baseline已用同额外Z，若与只做stopgrad Soft-MTP比较学习目标不同。不能将方差下降等同训练收敛速度。
平均每上下文联合非零候选行支持（按两个路径贡献的并集计，不扣恰好跨路径抵消）1.89425/2.04451/2.36938；独立差分2.78466/2.78682/2.80225。q零概率96.22/81.30/49.58%，两项同时零仅6.93/5.91/3.61%。因此不能跳过整个联合反向96%。大batch可能覆盖许多行，主干梯度稠密。
压力例ε=.1/.01/.001/.0001，实际解析矩计算g_p范数平方1.654315/1.960594/1.996006/1.999600；拒绝门控方差下界14.8896/194.0988/1994.0100/19994.0010；两样本V7.5363/97.0593/997.0060/9997.0006。q小于浮点范围时以解析f处理。
额外真实MC：p=(.4,.3,.2,.1),q=(.25,.35,.1,.3)，3种子各200000次，600000单步。理论α=.75，实际总接受450141/600000=.750235；实际非零teacher梯度420182次，其中接受时318355次（约75.77%非零teacher样本发生于接受）；四维teacher真值(.11979167,−.09740254,.10452455,−.12691367)，三种子MC均值/计数见JSON。各种子teacher坐标最大绝对误差约.00112，q最大约.00164（有限样本误差）；正式Y边际由标准SD，实际计数见JSON。不测真实多步、训练、时间、通信或CI；非完整R级。
注：压力例没有真实运行巨大logit LLM，仅计算解析矩；主实验枚举是实际CPU计算但非实际生成任务。

## 4. 成本与强基线 P

单上下文只发送3个32bit ID和一个FP32差分系数=16B（若接收端已有合法同版本hidden、W及索引语义；X,Y已经可见时增量可小至Z及系数8B）。若两个logit路径分置，需要双方分数、特征、VJP、版本及权限，不能宣称完整协议16B。完整FP32 p/q传输8V B是弱基线，普通硬标签仅Y ID且本地teacher forcing/Soft-MTP无需上传分布；区域共置计算最强。若直接传稀疏行梯度，FP32+d维+ID，期望r(4d+4)B，r最多3；d1536中σ r≈2.0445约12570B/上下文，不含hidden和主干/优化器；相比同梯度目标的全量W上传可少，但普通FL多本地步合并已只每轮上传一次参数/LoRA，不能拿逐token稠密梯度当唯一对照。独立批次支持并集可稠密，不能从单样本支持推断整轮通信。
额外Z采样：若目标logits仍在显存，可一次额外categorical采样并取两路所需分数；若融合头不物化logits，额外race/查询/重算/缓存可能成本不小；R04已给强双阶段/FlashSampling/晚绑定机制，本轮不假设免费。训练仍需正式NLL、主干VJP、优化器与同步。冻结教师只需q项，退回HC/R03；完整双向项即使SD接受也需处理。假设可跳过q路径反向占全流程f_q，额外开销比例h，则理想收益≤f_q(1−δ)−h，若f_q(1−δ)<.15，即使零额外开销也达不到G5。中σ至少f_q≈.1845、高σ≈.3025才有15%理想空间；这不是实测占比。
强基线：普通HFL AR+硬/软蒸馏、HFL+标准内生SD、HC/R03固定教师耦合、Soft-MTP精确共享头恢复（stopgrad、质量目标需区分）、全概率双向KL、普通AR+RLOO+独立X、同预算人为最大耦合、同信息平坦/共置计算及已调优的OnlineSPEC/DistillSpec/REAL适用训练目标。若同信息平坦可用相同耦合，机制和学习估计完全等价，G2不通过。联邦层级不是此无偏性必要条件。缺乏真正HFL特有驻留约束与正式学习价值，不能将一般SD采样梯度包装为融合。

## 5. 文献核查 2026-11-24

实际检索 MTP KL gradient recovery, federated multi-exit, speculative draft distillation, RLOO covariance/variance, online verification feedback；打开原始页面、Soft-MTP arXiv abs页面可读正文和附录（html直链失败改abs正文），其余原始摘要/部分正文。
- [Multi-Token Prediction Done Right](https://arxiv.org/abs/2610.38331)，2026-10-28，Soft-MTP：stop-gradient NTP teacher，恢复MTP hidden梯度，输出头保持CE；167M–1.2B作者报告训练/解码改善，非本队实测；正文§4.1、§5及§6.3/§7明确头部自蒸馏退化/并行兼容，是最直接近邻和质量反证线索。
- [Double Control Variates](https://proceedings.mlr.press/v151/titsias22a.html)，AISTATS2022，RLOO score估计与额外控制变量；本轮teacher两样本公式不是首次。
- [LayerSkip](https://aclanthology.org/2024.acl-long.681/)，ACL2024共享早退训练、自投机。
- [ReeFL](https://proceedings.mlr.press/v235/lee24h.html)，ICML2024，异构客户端联邦早退和共享分类器。
- [Online SD](https://proceedings.mlr.press/v235/liu24y.html)，ICML2024，在线蒸馏；[When Drafts Evolve / OnlineSPEC](https://proceedings.mlr.press/v306/qian26k.html)，ICML2026，验证反馈在线学习。
- [DistillSpec](https://arxiv.org/abs/2310.08461)，ICLR2024，on-policy蒸馏及divergence选择；[Direct Alignment](https://arxiv.org/abs/2403.00858)，2024，TVD++方差减少。
- [REAL Loss](https://arxiv.org/abs/2611.14399)，2026-11-10，真实EAL路径梯度及条件积分，与双向KL目标不同但接受指标对照。
- [Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html)，ICML2023，精确最大耦合；[HierFAVG](https://arxiv.org/abs/1905.06641)，分层聚合。
未穷尽耦合score/稀疏梯度、最新论文、全文/专利、未索引工作；无实际文字查重。规范引用不能保证内容原创。HC/R03已经证明固定教师q差分、方差及零梯度；HiMerge/R03与HCA/R03给联合接受损失/质量退化，不能换名宣称。

## 6. 审核、裁决及后续

主观加权创新（问题7×25%、机制5.5×25%、理论7×20%、价值6.5×15%、文献距离4.5×15%）=6.175/10≈6.2，非审稿意见。G1 FAIL：RLOO、HC/R03、Soft-MTP近邻；G2 CRITICAL FAIL：普通AR及平坦可复制，正式质量与HFL特有因果未证；G3 PARTIAL/FAIL：固定前缀梯度及标准SD理论，完整实际greedy、采样、top-p/精度未审计；G4 FAIL仅合成；G5未验证无真实总时间；G6未验证无80%证据；G7局部代码结果/整体FAIL。好结果仍同质量/资源对最强公平基线真实总wall time−15%或同成本有效工作+20%，两预训练尺度、三非IID/时延设置、三种子CI及全部G门槛。
未执行的最小验证（仅真实证据改变前提时）：1–2×40GB NVIDIA，Qwen2.5-1.5B/3B或经核实的共享MTP头模型，两尺度，8逻辑客户端2边缘，公开数学/代码/通用任务；先收集≥1万固定快照实际(p,q,h_p,h_q)及合法X,Y,Z，测尾部logratio、真实梯度协方差/行支持、附加目标抽样/重算、q反向占比及NLL质量。再比较普通HFL+SD、Soft-MTP、HC/R03固定教师耦合、全概率双向KL、AR+RLOO及同信息平坦，匹配训练目标、模型质量、通信与资源，记录真实训练+推理总时间、GPU峰值、字节、回退、P95、CI。单卡逻辑模拟非物理并行；未执行GPU，不为救此机制投入。若缺实际模型/trace，后续应优先建立真实小模型可复核测量，而不是继续增加toy；若q反向占比不足或平坦/Soft-MTP完全匹配即停止。全局研究下一轮可先审计已有真实MTP/联邦早退代码及可取得权重，在明确数据驻留、跨层约束和正式学习的不可替代性后再创造机制。
**Reject and Pivot**。保留双向KL三Token行支持/归一化抵消、拒绝门控方差下界和高接受率稀尾反例、同前缀独立样本条件、实际有限枚举及MC。禁止把RLOO+HC/R03+联邦聚合拼接当原创，不能以接受率、梯度方差或稀疏行数直接声称端到端收益。
