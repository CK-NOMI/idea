# HIERASPEC/R08｜FSG：精确自投机样本复用与正式监督Softmax梯度控制变量

日期2026-11-24；起始main 860e69d7403e23ee38f48557748fe9658ad56dfc，README blob 6b9039e7be81abb1265d8b334ad6477d0977f2c7，2610行。完整获取README，核验R07最终d7160316...为原样前缀，阅读新增LEA-SD/FCPR/FUDG/FDC全文及HC/R06详细记录、检查HC/R03 FCR-SD/HCA/R05 FCKG/R06 FTCD等历史记录。提交前阅读并发HCR/R08 FIA-SD（不可信抽查，不同对象）。HC/R06实际小Transformer深层校正已有，本轮不再重复；HC/R03最大耦合差分方差公式直接相邻，增量仅为外生监督标签的正式Softmax概率项、最优标量控制系数、强残差分层对照、同标签复用反例及实际公开语料矩估计。未独立复跑他队。
**CRITICAL / Reject and Pivot**。T=限定数学，E=本队实际V8公开语料字符计数拟合、概率矩与MC，P=成本代理，O=完整p/q可见oracle，H=假说。无本队预训练Transformer、实际HFL、完整SD rollout、GPU、物理网络、真实收敛/全流程wall time/CI；本轮未重测硬件或下载权重。

## 1. 场景及机制（H/T）
固定合法参数theta、外生监督样本(h,l)（前缀及真实标签），正式未截断温度1分布p=softmax(W z)，z为正式隐藏状态，q为同主干内生浅层/辅助头分布，无独立完整draft LLM。HFL客户端持授权监督数据，区域聚合固定快照一步梯度，再云加权；不声称DP。设同前缀另有实际标准SD联合样本X~q,Y~p，独立于外生标签选择给定(h,l,theta)（同一个请求是否恰好和监督前缀一致需实证）；样本来自最大耦合，接受概率A=Σmin(p,q)=1−δ，δ=TV(p,q)。拒绝时Y按正残差。固定真实监督CE L=−log p_l，其logit梯度g=p−e_l。普通抽样估计g0=e_Y−e_l无偏，条件方差迹v0=1−||p||²；对固定h,z，W梯度=(p−e_l)z^T，估计误差Frobenius方差乘||z||²；完整主干梯度需真实Jacobian/VJP且可能稠密，不等于logit迹方差的最优。正式学习目标不变，不训练额外KL作为替代正式NLL。
拟使用同一次SD返回的X,Y作为控制变量，g_beta=e_Y−e_l−beta(e_X−q)，beta可由公开固定校准或真实p/q计算，必须在本次X,Y揭示前确定且detach；样本及q权重作为停止梯度的系数。beta=1时g=q−e_l+e_Y−e_X，接受时随机修正为0，但稠密q−e_l不为0。云若已持同授权q和z可重建baseline并接收稀疏修正ID/系数；否则需要q,z或实际局部梯度，不能假设免费。不同区域q可局部学习及校准；SD样本影响正式梯度估计但没有证据优于普通生成/普通梯度压缩或平坦。一次固定快照无偏不保证多轮AdamW/梯度裁剪或优化器状态逐步等价，需独立收敛实证。

## 2. 数学及反例（T）
对固定外生样本，E[g_beta]=p−e_l。令 B=1−||q||²>0，C=Pr(X=Y)−p·q，v0=1−||p||²。联合迹方差
v_beta=v0+beta² B−2 beta C。
在beta固定非负时，最大耦合通过最大化Pr(X=Y)最小化此类迹方差；若beta为负不适用。最大耦合C=A−p·q≥0（逐坐标min≥pq）。最优固定标量beta*=C/B、v*=v0−C²/B。若q为点质量则B=0、中心化项恒0。beta=1时v=2δ−||p−q||²，与HC/R03草稿差分相同公式，不宣称首创。q=p时beta=1可得精确稠密梯度（q本来就给出p）；若p=(.9999,.0001),q=(.99,.01)，A=.9901而v0=.00019998、beta=1方差.01960398，约98倍恶化；高接受不保证胜过目标样本估计。
强基线1：固定已知p/q，分别采Z+~(p−q)_+/δ、Z−~(q−p)_+/δ（δ=0直接返回q−e_l）；两支支持不交，g_strat=q−e_l+δ(e_Z+−e_Z−)无偏，迹方差2δ²−||p−q||²。与beta=1标准SD差分相比严格少2δ(1−δ)（0<δ<1）；需要两个残差样本、真实δ及完整残差，非免费。每次固定两修正行 vs 标准耦合期望2δ行；若计通信/重算可能有不同权衡，不声称一定快。它相当于对拒绝Bernoulli随机性分层/Rao-Blackwell化并保留残差样本，普通AR也可直接构造，无实际SD必要性。强基线2：条件于X=x，按真实SD核m_x= a_x e_x+(1−a_x)R（a=min(1,p_x/q_x),R=(p−q)_+/δ）；g_RB=q−e_l+m_x−e_x，是beta1估计条件期望，方差不增，但一般稠密且需要真实残差分布。强基线3：完整p−e_l梯度精确、CCE等融合，以及传统重要性采样/梯度压缩，普通HFL+SD与同信息平坦均可用。
严格标签边界：若监督标签本身就是这次正式生成Y，给定标签l后Y不再是独立p样本，直接e_Y−e_l恒为0，而固定标签CE梯度通常非零。其条件偏差平方对l~p平均=1−||p||²（无条件平均梯度本来为0，不得把条件偏差与无条件偏差混淆）。奖励加权例p=q=Bern(.5)、R(Y)=Y，真实策略梯度E[R(Y)(Y−p)]=.25，使用同Y或因全接受X=Y作为负样本使梯度恒0；必须使用独立样本或正确条件估计，不能把正式rollout token当独立负样本。对于监督数据即便p/q来自正确SD，若只选接受事件、特定块或奖励事件，条件边缘改变，需重证。实际生成目标若使用temperature/top-p变换则不是温度1完整训练p，不能直接套本估计；可设共同未截断T=1并用真实q。标准SD真实条件q、min(1,p/q)、正残差、因果截断及合法bonus/EOS/KV可保持同目标随机序列；greedy独立参考argmax。完整模型随机/greedy、处理器、数值、训练版本合同未实现。

## 3. 实际运行（E/P/O）
[probe.js](probe.js)、[results.json](results.json)。V8实际执行，读回固定源及上传代码重新执行，数值及附加source metadata逐值一致。Node命令 node experiments/HIERASPEC/R08/probe.js wallpaper.txt alice.txt；脚本输出数值主体，归档JSON另附sources元数据。固定公开语料 [Wallpaper](https://github.com/GITenberg/The-Yellow-Wallpaper_1952/blob/ce7c5bc1a5cf0fe23bcb265545a900365dbae29c/1952.txt) blob 3a2b5f55c6f1d13a7ae623282d38fde5a6ff352b；[Alice](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/blob/7f18748e114b994d5d9454d5793581bc0b9cf9d7/11.txt) blob 9d73b1d7447b7ddc158b1b157f5f700d21bd2fb7。与其他团队同公开语料，独立读取及执行，不冒称新数据集。START/END之间统一小写、非字母段变空格，可能保留制作说明；V27，Wallpaper一区、Alice按中间字符分两区，各80%训练20%留出。训练24273/54055/54055，共132383字符；测试6069/13514/13514，共33097，有效双字符前缀33091。全局bigram .5伪计数，正式p=全局trigram+5×全局bigram；区域q=局部bigram+smooth×全局bigram，smooth1/20/100，三区×三平滑=9设置。是真实字符计数拟合非神经预训练，不实际执行HFL或完整SD。真实留出前缀频数加权精确条件矩；MC按每区域留出位置随机抽取前缀，固定p/q下生成单步最大耦合及独立正负残差对，seed11/29/73，每配置8192组，共221184联合抽样组；不是三独立训练种子，也不是独立真实轨迹。真实监督标签在方差式中抵消，不计算实际训练收敛。最大RB期望误差2.89e−15，最大2δ二阶矩误差5.33e−15，MC与解析各配置最大绝对差.019876。位置加权均值：
|smooth|A|plain|independent CV β=1|SD β=1|SD optimal β*|direct stratified residual|RB β=1|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|.640163|.700719|1.524569|.593730|.403184|.195276|.299845|
|20|.640382|.700719|1.524855|.593447|.402936|.195113|.299860|
|100|.640654|.700719|1.525334|.593126|.402583|.194951|.299935|
smooth20: 相对普通目标样本方差降15.31%/42.50%/72.16%/57.21%（耦合1/最优系数/直接残差/RB）；独立CV恶化。β*位置平均.556471，逐位置依赖正式p，oracle不是可免费校准。标准耦合修正期望.719235行，所有q均27非零行，完整g非稀疏。MC smooth20按区域位置加权seed11/29/73：
plain .701252/.701711/.702367；coupled .601769/.596926/.595241；optimal .408600/.407632/.405312；stratified .195844/.196348/.195760；RB .301560/.302230/.301417。没有完整CE反向/优化器训练、真实通信、GPU时间/CI或任务质量测量。

## 4. 通信计算及强基线（P/H）
仅当接收端已有同版本q和正式z、外生标签及合法上下文时，β1随机修正只需不相等时发送两个词ID（或者两个梯度行；ID可按实际已有信息减少），以V151936的32bit ID示意，条件修正平均2δ×4B，smooth20字符模型代理2.87694B/位置；普通g0同样只需1个正式样本ID 4B且标签另计，若Y已经提交则此ID甚至零增量。β*还需传真实系数或共同可重算，且接受时不再只在拒绝发送修正（但X=Y时合并成一行）。直接残差需固定2个ID和δ（若都未共享），完整方差更低，通信/计算Pareto需实测。如果q不可重建，传FP32 q=607744B/位置（未压缩示意），正式hidden d1536 BF16=3072B；稠密q及深层正式z不可免费。普通本地计算+逐轮聚合避免逐token跨层通信；低秩LoRA/共享embedding、参数压缩、误差反馈、普通AR+采样、CCE融合、SD普通蒸馏/直接残差以及同信息平坦为强基线。不能以逐位置全量稠密梯度上传作为唯一弱基线。若已完成全词表验证，直接计算完整梯度通常可利用现有概率；若想省正式前向，p及真实Y采样并非免费。本轮不证明任何forward减省，主干/VJP/AdamW/额外SD与校准均需计。若可消除部分占总时间f、相对节省s，必要条件fs−overhead/T≥.15；方差降幅或修正稀疏率不是s。数据驻留、同监督前缀实际命中、额外成本、收敛质量与端到端均未知。

## 5. 文献（2026-11-24）
实际检索 adaptive importance sampling softmax, sampled softmax bias, speculative decoding online learning, sparse gradient communication；核对原始页面/摘要及索引，不穷尽全文。
- [Bengio/Senecal AISTATS2003](https://proceedings.mlr.press/r4/bengio03a.html)、[IEEE TNN 2008](https://pubmed.ncbi.nlm.nih.gov/18390314/)：训练概率模型的重要性采样，非新思想。
- [Adaptive Sampled Softmax ICML2018](https://proceedings.mlr.press/v80/blanc18a.html)、[RF-softmax NeurIPS2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/e43739bba7cdb577e9e3e4e42447f5a5-Abstract.html)：自适应采样及真实softmax抽样的无偏条件。
- [Sampled Estimators For Softmax Must Be Biased NeurIPS2025](https://papers.neurips.cc/paper_files/paper/2025/hash/42b7c2f6d320d1fe1afa899a6319d6d7-Abstract-Conference.html)：仅访问采样子集的无偏障碍；本轮已有完整目标样本和真实q，不满足其受限信息模型，不声称反驳。
- [CCE ICLR2025](https://arxiv.org/abs/2411.09009)：融合交叉熵、减少显存物化，强完整梯度工程基线（过滤梯度需区别）。
- [QSGD NeurIPS2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html)：普通梯度量化通信强替代。
- [Online SD ICML2024](https://proceedings.mlr.press/v235/liu24y.html)、[When Drafts Evolve ICML2026](https://proceedings.mlr.press/v306/qian26k.html)、[EAGLE ICML2024](https://proceedings.mlr.press/v235/li24bt.html)、LayerSkip及标准Leviathan。前述草稿在线学习与耦合差分近邻，HC/R03/HCA/R05已揭示多个关键限制。未穷尽近期随机梯度估计/分层压缩/全部附录或未公开论文，未正式文字查重。

## 6. 裁决与下一步
**CRITICAL / Reject and Pivot**。虽明确正式监督目标且条件无偏，标准SD样本复用不必要，直接残差在相同完整信息下方差显著更低，普通目标样本/完整CE及压缩可替代；q baseline稠密，真正同监督前缀重用及物理驻留未证，HFL同信息平坦可复制，正式学习性能未知。主观创新6.175≈6.2/10（问题7×25%、机制5.5×25%、理论7×20%、价值6.5×15%、距离4.5×15%），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（限定无偏、真实标准SD条件，完整greedy/随机/top-p/浮点未验）；G4 FAIL（真实语料统计模型非两个预训练Transformer）；G5/G6未验证，无80%证据；G7局部代码/固定数据及结果重跑/完整未过。好结果仍同质量预算最强公平基线全流程真实时间−15%或有效工作+20%，两预训练尺度三非IID/时延三独立训练seed及CI、全部G。
H未执行：仅当真实训练-生成同一版本同一监督前缀大量共现且目标样本非标签自身、实际通信或反向为关键路径时，才在1–2×40GB Qwen2.5-1.5B/3B、8逻辑客户端2区域、公开数学/代码/通用任务、三种子三配置，记录≥10000位置(h hash, label, actual X/Y, p/q/采样设置, version, hidden availability, gradient/optimizer, bytes, time)；比较完整CCE、普通AR+样本/直接分层残差、普通梯度压缩、HC/R03蒸馏及HFL+生产自投机、同信息平坦；测同目标质量/预算真实全流程和CI。缺同前缀命中或共享q/z、强基线胜则停。当前不为该候选投入GPU。下一轮先获取真实训练/验证联合profile并验证学习机制实际改变必要计算；不再用耦合差分、普通多保真梯度或稀疏通信换名重启。保留正式外生标签无偏条件、最优标量系数、严格残差方差差、标签/奖励反例、实际真实语料矩和MC及强基线负结果。
