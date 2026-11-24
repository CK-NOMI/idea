# HIERASPEC/R03 — FLPS：异构分词隐变量上的联邦变分辅助耦合
2026-11-24；研究开始 main 9ec9c3b013e162dd720a17735024a4c088fb30af，README blob 11af5f16375b12850c7e97fe8bafc8bbc1e34e14（1618行）。已核对前轮完整历史及新增HC/R02、读取文件树、HC/R02和HieraCert/R03脚本及结果，未独立复跑他队。并发新增HieraCert/R03 FPQ、HCA/R03 FSRA、HCR/R03 FBE、HiMerge/R03 JVT（及详细记录）、HierCouple/R03 NF-RSD均已阅读：分别为正式状态等价、坐标对齐、MoE专家集合、接受长度梯度、非线性归一化；本轮对象不同。与HierCouple/R02的同词表短语头区别为异构分词的多对一映射。与R01同属残差采样但提升潜变量而非混合目标。与HieraCert/R03的等价状态DP是近邻。
**CRITICAL / Reject and Pivot**。T=限定数学；E=本轮实际CPU合成枚举及MC；P/O=代理/全知；H=假说。没有真实HFL训练、预训练LLM、GPU、真实网络、wall time或CI。

## 场景与问题
客户端非IID私有文本，边缘持区域分词字典及使用共享模型内部特征的轻量草稿头，云学习固定正式模型p、词表及共享辅助后验参数r。允许区域个性化，按字节/正式token对齐共享参数，不直接平均不同语义嵌入；固定服务版本，无独立完整草稿LLM。交换授权参数、路径、标量，非DP证明。异构词表价值必须先证；共同词表头是强基线。
固定正式前缀h，有限停止路径集合Z_e，映射f_e:Z_e→V，每条路径映射一个正式token，每个p正质量token须存在路径（可额外定义纯修正路径），q_e(z)为真实草稿路径分布，Q_e(y)=Σ_{f_e(z)=y}q_e(z)。初始限于已知分词边界、无歧义正式映射；任意BPE的lookahead、非单射、跨token路径切分、EOS/KV仍未解决。优化精确正式输出下的全流程真实成本，含路径评分、补采、通信及训练摊销。
反例：q(A1,A2,B1,B2)=(.3,.3,.1,.3)，p(A,B)=(.5,.5)，真实Q=(.6,.4)。只取规范路径A1/B1并归一得到错误qhat=(.75,.25)。实际仍采Q，却用qhat验证及(p−qhat)_+修正，接受质量(.4,.4)，拒绝.2全部补B，最终(.4,.6)，TV=.1。若改变实际候选只采规范路径并用真实新概率则可精确，但算法变了。
限定黑盒查询障碍：a/aa拼接a^n有F_{n+1}条路径。假定各正到达历史的末端b/c条件可任意独立改变，精确求a^n b的概率需查全部末端历史（否则未查历史可变而不影响已观察）。不是所有精确采样/真实BPE的下界；Timor/Geh已研究指数或计算困难，DPE、结构化BPE与状态DP可避开部分路径枚举。

## 提升目标机制与数学
r_e(z|y)是可采样、点概率可计算的规范化条件分布，支撑f_e^{-1}(y)，定义P_e(z)=p(f_e(z))r_e(z|f_e(z))。先z~q_e，接受min(1,P_e(z)/q_e(z))；接受直接提交f_e(z)。拒绝后先Y~p，再z~r_e(.|Y)即从P_e提议，以[1−q_e(z)/P_e(z)]_+接受，反复直到成功，提交f_e(z)、截断后缀。P_e=0不提议。一次修正成功质量为[P_e−q_e]_+，与初始接受质量min(P_e,q_e)相加为P_e，故推前为p。r不准确只损效率不改变精确性。δ=TV(P_e,q_e)>0时修正尝试Geom(δ)，条件均值1/δ，无条件均值1；δ=0不调用。长尾、每次路径评分/RPC必须计，有限截断后须回退至真实提升残差而非p。经典最大耦合/辅助变量拒绝采样应用，不声称新基础原理。
固定前缀、真实条件q、因果重置、独立随机数、首次拒绝截断、bonus由p生成时可沿链式归纳正式序列正确；草稿头须能在不调用完整目标前向时生成后续条件。真实块、任意BPE、KV、EOS、top-p、浮点未审计。greedy用正式argmax及同tie-break，非随机证明的替代。
定义s_e(z|y)=q_e(z)/Q_e(y)（Q_e(y)>0），投影最优α_map=Σ_y min(p,Q_e)，提升α_r=Σ_z min(P_e,q_e)。按组求和及min(p r,Q s)≥min(p,Q)min(r,s)，得到
0≤α_map−α_r≤Σ_y min(p,Q_e)TV(r_e(.|y),s_e(.|y))。
r=s达投影最优；Q=0项为0。反例上述q和p，r每组均匀，α_r=.85、α_map=.9。
ELBO_e(y)=E_r[log q_e(z)−log r_e(z|y)]，log Q_e(y)−ELBO_e(y)=KL(r||s)。正支撑、有限KL且训练分布为正式p时，Pinsker+Cauchy-Schwarz给α_map−α_r≤sqrt(E_{Y~p}KL(r||s)/2)。训练数据d≠p须控制偏移；p正且Q=0时不能套有限界。拟区域用本地教师token优化ELBO，共享参数HFL聚合、局部条件保留；r梯度可从r采样并评分q，不需全量Q但方差/费用未知。同一r作为训练变分后验与精确验证提升分布，建立数学联系，但普通变分训练/平坦同样适用，未证正式HFL双向学习贡献。固定p防接受率牺牲质量（HiMerge/JVT及HC/FSRA反例）。

## 本轮实际实验
[probe.js](probe.js)，[results.json](results.json)，命令 node experiments/HIERASPEC/R03/probe.js。实际JS V8、LCG+Box-Muller、seed11/29/73。3目标token、8客户端2区域、k=4/8/12二元歧义单元（每token有2^k路径），区域logit漂移σ=.1/.6/1.2，客户端噪声.25，共27配置216客户端-配置。p=(.45,.35,.2)，合成q=Q(y)s(z|y)，Q=softmax(log p+.35 Gaussian)，s为独立Bernoulli乘积，区域/全局r为相应客户posterior logits均值sigmoid，是合成反向KL精确最优拟合而非真实学习。该分解中Q本身容易计算，不能据此声称真实指数加速。平坦同组同信息严格匹配（构造等价，不是物理独立实现）。以下为实际精确枚举概率质量，非LLM采样频率。
|k|σ|投影oracle|独立匹配|全局r|区域r=平坦同组|均匀r|错误qhat输出TV|
|---:|---:|---:|---:|---:|---:|---:|---:|
|4|.1|.899804|.360221|.859755|.870709|.737915|.124734|
|4|.6|.899804|.360221|.797725|.872725|.669271|.130569|
|4|1.2|.899804|.360221|.703615|.878191|.542303|.151079|
|8|.1|.883859|.368520|.823245|.836411|.612170|.161383|
|8|.6|.883859|.368520|.716283|.838555|.521633|.172690|
|8|1.2|.883859|.368520|.565283|.844356|.374366|.191490|
|12|.1|.900664|.359759|.813175|.828927|.514336|.191145|
|12|.6|.900664|.359759|.696505|.831218|.438797|.193092|
|12|1.2|.900664|.359759|.526872|.837020|.307388|.207187|
k12 σ1.2 全局/区域ELBO gap .816648/.049622 nat，区域改善31.01pp但公平平坦完全匹配；规范单路径r接受.0201%（人为极端，不等于TLI或真实频率）；真实后验oracle达投影最优。最大正式边际误差9.83e−15；界最大违反2.78e−15（浮点容差，非严格数值证明）。
MC单步P=(.4,.1,.2,.3)、q=(.1,.3,.15,.45)，2词×2路径，N=200000 seed20261124，正式计数[99672,100328]，接受130056(.65028，理论.65)，修正提议199987（无条件.999935，条件约2.859），经验TV=.00164。未测时间/CI/完整序列。上传结果重跑核验逐值相同。
P成本：k12每词4096条路径（仅toy黑盒）；r用12个Bernoulli采样及评分。拒绝后每到达位置无条件约1次额外路径查询（δ>0），不等于1次神经网络forward；需计起草、r评分/训练、远程q重算、通信与KV。区域3词×12 FP32 logit共144B的示意，不含模型协议，真实词表独立r不可扩展；均值拒绝率.163代入条件尝试约6.1仅示意。没有真实端到端结果。

## 原始文献与去重
2026-11-24搜索 speculative decoding heterogeneous vocabularies, cross-tokenizer likelihood, tokenization marginalization, latent proposal variational SD, dynamic programming segmentation, federated tokenizer, auxiliary variable extended target sampling。核对原始摘要/部分HTML；Timor算法2–4及复杂度正文，PDF截图失败后改读HTML。非完整附录审计。
- [Timor et al., ICML2025](https://proceedings.mlr.press/v267/timor25a.html)，[HTML](https://arxiv.org/html/2502.05202)：SLEM、SLRS、TLI；多路径求和与指数复杂度、跨分词块验证均已有；本轮提升P并用r换取不显式边缘化，但SLEM/TLI为强基线。
- [TokenTiming](https://arxiv.org/abs/2510.15545) 2025-10：动态对齐与概率传递，精确条件待逐式核查。
- [Geh et al. EMNLP2024](https://aclanthology.org/2024.emnlp-main.230/)：字符串边缘概率困难；[Cao/Rimell 2021](https://aclanthology.org/2021.emnlp-main.161/)、[Chirkova 2023](https://aclanthology.org/2023.acl-short.1/)：分词边缘似然。
- [DPE ACL2020](https://aclanthology.org/2020.acl-main.275/)：隐变量分词、精确边缘似然及DP。
- [Cross-Tokenizer Likelihood Scoring ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/40d50fe20d5e6922dbb52d7da6de03f1-Abstract-Conference.html)：BPE递归、子集精确O(1)模型评估/token、一般精确及近似，不能将黑盒指数泛化。
- [VSD ICML2026](https://proceedings.mlr.press/v306/zou26i.html)：潜草稿路径、ELBO与接受率优化、EM；本轮r是分词提升条件而非接受后验，仍高度相关。
- [Training a Tokenizer for Free with Private Federated Learning](https://machinelearning.apple.com/research/training-a-tokenizer)：联邦分词适配已有；[BLT](https://arxiv.org/abs/2412.09871)：动态字节patch，统一字节/正式头为替代；[Auxiliary Gradient-Based Sampling](https://academic.oup.com/jrsssb/article/80/4/749/7048390) 2018：辅助目标；[HierFAVG](https://arxiv.org/abs/1905.06641) 2019/2020。
盲区：任意BPE多步、最新辅助变量耦合、未索引/未公开工作、专利、全部附录；无正式文字查重，规范引用不替代机制原创。

## 强基线、裁决及下一轮
CRITICAL：平坦同信息保留区域r完全匹配；普通本地变分训练亦可，正式HFL学习没有不可替代反向机制；同一正式词表内生头直接避免歧义，HFL未必需要异构分词。普通HFL AR、HFL+Medusa/LayerSkip、同目标词表直接头、SLRS/SLEM/TLI、DPE及BPE递归精确边缘化、平坦同组同委托、后验oracle均需比较。合成实验不支持真实加速；路径长尾、重算、跨边界状态、真实质量未知。
H未执行：仅新证据表明异构词表不能被同容量正式头替代时，1–2×40GB Qwen2.5-1.5B/3B、8逻辑客户端2区域，公开多语言/代码/数学，三种子三非IID/RTT、两尺度。先统计真实分词歧义/覆盖率/重编码失败、q评分和通信、质量；计训练+推理总wall time、TTFT/TPOT/P95、吞吐（字节与正式token）、GPU time/显存、字节/RPC、回退、CI，单卡逻辑模拟非物理并行。若共同头匹配即停。好结果同正式目标质量/预算下最强公平基线总真实时间降≥15%或有效工作+20%，全部G通过；无80%把握证据。
主观评分问题7.5×25%、机制6.5×25%、理论7×20%、价值6.5×15%、距离4.5×15%=6.55/10（非同行评审）。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（单步及条件序列T，任意分词/greedy/精度未验）；G4 FAIL仅合成；G5 FAIL无实际总时间；G6 FAIL未知80%；G7 PARTIAL/FAIL（代码数据已复算，完整系统未复现）。
**Reject and Pivot**。保留规范路径概率错误反例、提升分布精确性、ELBO差与接受率损失界、实际区域与平坦等价、残差长尾。禁止改名重启普通跨分词/变分/残差/接受损失。下一轮优先获取真实小预训练模型执行trace，定位学习与内生SD共同、不可替代的工作和物理约束，先做普通AR/SD和平坦区分实验及Amdahl门槛，再创造机制。
