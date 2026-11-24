# team-hca/R06｜FTCD：真实语料上的序列耦合梯度与前缀重绑定

2026-11-24。起始 main 1f9535bda0058c20ae0ef29c0570902d8bcf5eb3，README blob 9635ad0cab50fb4e2185bccccb7574066ec42301，2271行。读取固定版本完整文本并核验R05提交96a0c0c9b7bcec763841916411232fba011eb1c9原样前缀，阅读新增CGE、QCRA、FCF，检查文件树及R05/R06记录。写前读并发FRGB（二元奖励组屏障）、CFA（独立关键路径表观协同），最新已读main 6185039cde6c8cbe111b508b50949381310c4f13，blob 2e3c012e6a571153ddd80868d097109e6fc0a657。未独立复跑他队。HC/R03单步固定教师耦合梯度为直接先例，本轮不把单步估计换名当新创新。区别R05双向KL、HiMerge/R06 Fisher、JVT期望接受长度、FCRR历史缓存条件偏差、FOAS多独立目标树：本轮冻结正式p、固定外生起始上下文，研究序列级前向KL的整条轨迹耦合和每步正式前缀重绑定的差异。

**CRITICAL / Reject and Pivot**。T=限定理论；E=实际CPU真实公开语料训练的字符统计语言模型、概率DP及MC；P=方差/工作/通信代理；O=精确期望oracle；H=假说。没有预训练Transformer、实际神经网络优化、GPU、真实HFL多设备、实际网络或wall time、正式CI。实际环境Python3.13.5、torch2.10.0+cpu、CUDA false/0设备、无transformers、~/.cache/huggingface/hub和/opt/models不存在，huggingface.co及raw.githubusercontent.com DNS gaierror Errno -3。尝试GitHub char-rnn固定commit 6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e的tinyshakespeare/input.txt，返回空文本（blob 7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9），改用成功取得的两部公开Gutenberg原文。实际实验JS V8 CPU。不是预训练Transformer或真实HFL系统。

## 1. 场景及数学对象 T/H

客户端持non-IID语料/合法外生起始上下文h0，edge汇总区域统计，cloud聚合共享正式模型p。内生轻量预测路径q可由共享模型的低阶统计/早退特征产生，不另训独立完整草稿LLM。实验代理p为带全局bigram平滑的trigram，q_e为区域bigram加全局backoff，分层计数加和与同信息平坦完全等价。固定正式p和实际q参数快照、固定生成长度L、共同词表/全支持，训练目标为F_e=E_{h0} KL(P(.|h0)||Q_e(.|h0))，P/Q为完整自回归序列分布，优化q（不更新p的梯度），原正式模型学习独立，非双向必要性已证。前缀分布外生，若依赖训练参数需额外导数；无DP声明。

令s_phi(a|h)=∇_phi log q_phi(a|h)，S_phi(U)=Σ_t s_phi(U_t|h0,U_<t)。精确目标梯度g=−E_{Y~P}S_phi(Y)，因p冻结；E_{X~Q}S_phi(X)=0。

方案A（整条轨迹耦合）：每个位置给定两条已有历史，最大耦合实际q(.|h_X)、p(.|h_Y)，两边边际分别保持Q/P。H_path=S_phi(X)−S_phi(Y)无偏。若两条完整历史与token相同，则该位置score严格抵消；历史一旦分叉，即使当前token再次相同，通常仍是两个不同score/Jacobian，不能跳过。实验Markov模型只需最后一个字符相同即可抵消q score，这是比完整历史Transformer更乐观的重会合条件。分叉后仍推进原q轨迹是为了控制变量，不等于标准SD可使用的草稿后缀；实际SD必须丢弃错误后缀或在正式前缀重新起草。该方案是用于研究梯度的耦合采样器，不宣称实现真实批SD。

方案B（正式前缀重绑定）：对每个正式历史H_t=(h0,Y_<t)，条件最大耦合A_t~q(.|H_t)、Y_t~p(.|H_t)。H_local=Σ_t[s_phi(A_t|H_t)−s_phi(Y_t|H_t)]。条件E s_phi(A_t|H_t)=0，因此H_local同样无偏，且Y整条序列边际P。每步接受时该差分严格为0；不会因过去的拒绝永久失去可抵消性。固定特征softmax logits下等于e_A−e_Y，HC/R03单步公式的直接延伸，普通AR+额外耦合也可实现；不是新学习原语。若实际SD的bonus没有合法同前缀A，需另用普通硬标签/完整梯度或额外采样，且不可用已丢弃的非正式前缀候选替代。实际q须为真实条件提议，样本及索引detach，独立随机流及正式因果状态；固定快照同步加权梯度可聚合，非任意多步FedAvg/AdamW无偏。greedy用正式argmax；完整随机SD、top-p、EOS、KV及有限精度未实现。

方案C（强硬标签基线）：H_hard=−S_phi(Y)；还须比较全概率teacher forcing和条件积分、最优控制变量、Block Verification（目标前缀耦合更强）、OnlineSPEC/SpecOPD、普通HFL+SD及公平平坦。每步重绑定与全路径估计均为控制变量；不宣称重绑定有普适最小方差（跨时间协方差、参数Jacobians和起始上下文异质性影响结果）。L=1时A/B等价，尖峰p=(1,0),q=(.5,.5)时硬标签方差0而耦合差分有正方差。若沿所有可能匹配历史的条件接受率≤a<1，则连续匹配长度M满足Pr(M≥k)≤a^k，E M≤a(1−a^L)/(1−a)。即使同一步边际接受很高，长期共同前缀不保证线性增长；非所有模型存在统一a。局部重绑定的零梯度只限各位置差分，对共享主干仍需保留/重算依赖激活，不能简单省掉所有已接受前缀的反传。

## 2. 真实公开语料与实际执行 E/O/P

[代码](probe.js)，[原始逐配置结果](results.json)。实际从固定GitHub文本读取：
- [The Yellow Wallpaper](https://github.com/GITenberg/The-Yellow-Wallpaper_1952/blob/ce7c5bc1a5cf0fe23bcb265545a900365dbae29c/1952.txt)，blob 3a2b5f55c6f1d13a7ae623282d38fde5a6ff352b；
- [Alice's Adventures in Wonderland](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/blob/7f18748e114b994d5d9454d5793581bc0b9cf9d7/11.txt)，blob 9d73b1d7447b7ddc158b1b157f5f700d21bd2fb7。
复现：下载上述固定版本原始文本保存为wallpaper.txt和alice.txt，node experiments/team-hca/R06/probe.js wallpaper.txt alice.txt。程序在V8实际执行，用GitHub读取的原文作为输入；归档脚本从本地文件读取相同文本。未把原始文本重新上传。清洗：取Gutenberg START和END之间内容（保留其中作品标题/少量编辑说明），小写、连续非a-z字符替换空格，trim；Alice按中点附近空格拆两区，每区前80%训练、后20%测试，WallPaper为另一区。27字符（含空格）。每区训练字符数24273/54054/54055，共132382；测试6069/13514/13514，共33097。区域计数从第3字符起；全局bigram q_b由加总bigram count + .5平滑，正式p(ab,c)由加总trigram count+5 q_b(c|b)归一化；区域q_e(b,c)由区域bigram count+20 q_b(c|b)归一化。是实际训练的有限阶统计LM，不是Transformer。固定训练计数没有独立seed变化；seed只影响后续抽样。

Heldout真实文本（teacher forcing）：
|区|p NLL nat/char|区域q NLL|全局q NLL|p/q条件重叠均值|
|---|---:|---:|---:|---:|
|Wallpaper|1.82214|2.25150|2.27175|.634970|
|Alice前半|1.75300|2.24216|2.23679|.640774|
|Alice后半|1.75300|2.22643|2.22859|.642402|
区域q并不始终优于全局q；重叠是已观测前缀条件均值，不是完整SD端到端接受率。

梯度实验：每区从heldout均匀位置取64个固定起始二字符状态；L=4/16/64，seed11/29/73（内部加区域和长度偏移），每区长度种子400组，每组两条不同的最大耦合过程（path和local，hard共用local的Y），共27配置10800组，每种估计器10800个向量。用LCG，仅复算非生产RNG。固定p的729状态转移DP计算真实期望的q logit梯度（oracle），初始64状态均匀；实际采样与该期望比较。梯度为每个序列求和，不除L；三种子/三区均值：
|L|共同连续前缀期望长度(经验)|path score可抵消位置比例|local零差分比例|path方差迹|local方差迹|hard方差迹|
|---:|---:|---:|---:|---:|---:|---:|
|4|1.45333|.417569|.638889|4.150998|2.873726|3.283365|
|16|1.83028|.328368|.637934|18.561238|11.505202|13.147132|
|64|1.82361|.305777|.635443|76.427190|46.312245|52.366699|
样本方差迹用N−1分母，包含初始状态变化及跨时间协方差。path可抵消位置按q所需上一字符相同且采样字符相同统计，不要求整个历史相同。local/hard共享同一Y，path使用另一条边际同分布Y。非独立训练模型重复，未做CI。MC梯度样本均值相对DP真值的平方误差（跨配置均值）：
|L|path|local|hard|
|---:|---:|---:|---:|
|4|.010605|.007070|.008423|
|16|.047463|.027168|.030467|
|64|.204301|.104039|.132093|
平均整序列非零logit坐标数：
|L|path|local|hard|
|---:|---:|---:|---:|
|4|64.60|2.87|103.01|
|16|233.70|10.93|274.70|
|64|436.75|37.28|498.26|
在L64，path比hard方差高45.95%，local比hard低11.56%，比path低39.40%。共同前缀平均只约1.82字符，长期path优势快速消失，有限Markov重会合已对path有利。真实语料训练统计模型证据，但不能外推Transformer梯度方差、任务质量、收敛或实际时间；没有实际SGD更新/真实神经训练/网络，不能用真实语料代替G4。记录的路径score坐标数不是LM头实际GPU行数。

## 3. 计算、通信和强基线 P

H_path完整Q/P各L字符，分叉后两条历史一般都需推进q状态、评分和VJP；H_local只在正式历史附加q候选，max coupling需要实际p/q与正残差，通常标准SD中前缀上已有部分计算，但bonus、首拒绝及额外样本/激活须真实计。H_hard普通AR正式序列+q teacher forcing可并行训练；全概率teacher forcing在p已计算时是条件精确梯度（仍有上下文采样方差）；本有限Markov模型DP可以精确算目标，是更强基线。训练路径耦合并不是新SD加速器；不能把额外q路径算成无代价。按同版本接收端已知上下文/状态的示意，local拒绝位置可传2个32bit token ID和一个32bit位置，共12B/拒绝；L64 local平均拒绝.36456，约280B/序列（不含正常正式输出、版本、权重、协议；完整Y本来也可能已知，普通硬标签仅需Y ID）。实际在边缘共置计算则只上传正常梯度/LoRA，平坦可完全复制。计数模型每客户端dense trigram 27^3个32bit计数约78,732B，bigram可从trigram加和；本轮实际未测物理传输。候选路径每次拒绝后仍生成后缀的训练成本，比较普通SD丢弃后缀、块验证、前缀缓存、直接teacher forcing；在神经模型中稀疏logit不等于稀疏主干梯度。

理想可跳过q梯度工作占强基线全流程f，实际可跳过比例a=.6354时，零额外开销需f≥.15/a≈.2361才能通过G5；若path抵消比例.3058则需f≥.4906，实际还要计双路径、batch/VJP与通信。仅必要条件/代理，不是实际速度。若HFL正式学习目标或参数也改变，应另外证明质量与无偏性，不使用R05高接受率掩盖正式梯度。

## 4. 文献去重（2026-11-24）

核对原始页面/摘要，未逐式穷尽全文或未公开研究：
- [SpecOPD](https://arxiv.org/abs/2611.02846)，2026-11-02，branch-parallel on-policy distillation，真实prefix和draft-sampled history，是直接序列训练近邻。
- [When Drafts Evolve/OnlineSPEC](https://proceedings.mlr.press/v306/qian26k.html)，ICML2026，验证反馈在线适配。
- [SpecKD](https://arxiv.org/abs/2510.24021)，2025-10-28，接受token选择性蒸馏，目标与本轮固定P的无偏差分不同。
- [Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325)，ICLR2025，teacher/student interleaved sampling。
- [Block Verification](https://arxiv.org/abs/2403.10444)，ICLR2025，块级验证强对照。
- [DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html)，ICLR2024，草稿on-policy数据和任务divergence。
- [GKD](https://arxiv.org/abs/2306.13649)，ICLR2024，序列蒸馏与on-policy。
- [Online SD](https://proceedings.mlr.press/v235/liu24y.html)，ICML2024；[Federated Learning of N-Gram Language Models](https://aclanthology.org/K19-1012/)，2019，联邦训练RNN并近似成n-gram，并非本轮直接计数方案；[Geometric Variance Reduction in Markov Chains](https://jmlr.org/papers/v7/munos06a.html)，2006，顺序控制变量先例。
HC/R03单步耦合、R05拒绝门控、FCRR条件核、HCR/R04验证tape、FOAS联合独立组须引用。本轮控制变量和前缀重绑定都不是全新基础数学，未做正式文字查重。强平坦同信息可复制所有步骤。正式HFL学习无反向不可替代改变，G2致命失败。

## 5. 验收与后续

主观加权创新6.0/10（问题7×25%、机制5×25%、理论6×20%、价值7×15%、文献距离4.5×15%=5.925，约5.9；以5.9为正式评分）。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（固定外生前缀双边际及无偏梯度，完整真实greedy/采样/top-p/浮点未验证）；G4 FAIL（真实公开文本统计LM，不是两预训练Transformer尺度）；G5未验证；G6未验证，无80%证据；G7局部代码与原始数据、完整未过。好结果同目标质量预算最强公平基线实际总时间−15%或有效工作+20%，两预训练尺度、三非IID/时延、三独立训练seed、CI与全部G。

未执行GPU计划仅在新证据改变前提时：1–2×40GB NVIDIA，Qwen2.5-1.5B/3B的合适共享浅层/附属头，8逻辑客户端2区域、公开通用文本/数学/代码，固定p/q快照收集≥1万位置真实(p,q,h,版本、真实条件提议、拒绝位置、VJP、延迟)；同起始prefix、L4/16/64及更长，比较普通HFL AR+teacher forcing、标准HFL内生SD、HC/R03逐位置差分、整路径coupled、Block Verification、SpecOPD、OnlineSPEC、精确软标签及同信息平坦。记录真实梯度方差、每组成本、显存、字节、正式质量、总训练推理时间及CI。单卡逻辑模拟非物理并行。若实际q反向份额不足、平坦/已有学习机制匹配则停止。不为已否决的FTCD投入GPU。

**Reject and Pivot**。保留真实语料数据取得路径和固定版本、计数LM与DP、序列耦合无偏性及前缀衰减、实际长序列负结果。下一轮不再用增加统计模型/随机toy规模代替真实Transformer：优先接入可用GPU/预训练权重及真实权限或系统trace；若无法取得，转向正式G2可辨识性与实际资源约束审计，不能靠新命名绕过平坦等价或以局部KL梯度替代正式学习。