# HC/R03｜FCR-SD：最大耦合残差梯度与联邦蒸馏
2026-11-24；起始 main d27b72ada362b76c92c4800fe3731333a39d73e0、README blob b1d2d50b049522040a80d0433516ba43360d5a98，1726行。读取完整快照并确认旧固定版本原样前缀，核实HC/R02代码结果。对照FPQ、FSRA及HCR/NF-RSD脚本结果；上传前读新增FBE、HiMerge/JVT（含详细记录）、NF-RSD及HIERASPEC/FLPS摘要。JVT联合p/q接受长度梯度与本轮固定p、保持CE梯度不同；A/R05–06控制变量、A/R09近似深层反向、A/R12选择偏差是数学近邻；不重复概率混合、坐标对齐、状态合并、专家部署及异构分词。
**CRITICAL / Reject and Pivot（保留有条件的梯度优化，不批准顶刊立项）**。T=限定数学，E=实际执行CPU合成MC与同步SGD，P=梯度方差/稀疏率/成本代理，O=全概率oracle，H=未验证假说。无GPU、真实LLM、真实HFL网络或wall time/CI；非完整R。

## 1. 系统、目标、机制
单一正式固定快照 p_theta(v|h)，内生草稿 q_phi(v|h)（共享hidden或轻量附属头，不独立完整LLM）；客户端持授权非IID上下文，边缘起草、可信验证端按标准SD验证。边缘可知自己的X、正式Y与版本，不一定能获全教师概率。验证端需要实际q或可重建q（计通信）；不声称DP。教师停止梯度、固定外生上下文，目标为 L=-Σp log q，q全支持可微。原正式HFL NLL独立执行。优化同质量总训练+生成时间；不改变目标p。
X~q，标准min(1,p/q)接受，拒绝按归一化(p-q)_+补采Y，Y~p，P(X≠Y)=δ=TV(p,q)。g=∇log q(X)−∇log q(Y)，采样detach；E∇log q(X)=0故Eg=∇L。接受X=Y时g严格0，仅拒绝计算，分母为全部有效样本，不是拒绝数。softmax logits z下 g=e_X−e_Y，等价 z_X−z_Y，无logsumexp；线性头 z=Wh、固定h则∇W=(e_X−e_Y)h^T，仅2行。若h可训练，J^Tg可能稠密。不能跳过正式目标模型梯度。前向/验证已发生，反向激活保存/重算非免费。
边缘与云按全部样本数/固定权重聚合，线性期望无偏；同步一次梯度与平坦完全等价。旧q回放均值q_old−p而非q_new−p，须校正或合法IS；教师版本变化需处理。多步仅实际到达候选位置条件证明，不能用拒绝后缀当真实目标历史，bonus无X须另用CE/排除并定义目标；按接受情况选前缀可能改变训练分布（A/R12）。固定p、实际条件q、因果前缀、独立随机数、正确残差与KV的标准SD随机序列精确；greedy另按目标argmax，确定性X不满足本估计器的采样前提。top-p、温度、EOS、有限精度未实现。

## 2. 方差最优、反例和边界（T）
g_CE=q−e_Y，迹方差1−||p||²。任意合法耦合的差分g=e_X−e_Y，均值q−p，迹方差2P(X≠Y)−||p−q||²，故最大耦合使此类logit差分迹方差最小 V_SD=2δ−||p−q||²；独立差分 V_ind=2−2p·q−||p−q||²。非任意参数空间/Jacobian几何或所有估计器最优；完整教师 q−p 条件方差0，强oracle。
反例 p=(1,0),q=(.5,.5)，CE方差0、SD方差.5。p=q时g恒0不代表正式任务梯度0。优势需2δ−||p−q||²<1−||p||²。
只观察(X,A)且固定q不足以恢复CE：q=(.4,.3,.3)，p1=(.2,.5,.3)，p2=(.2,.3,.5)，接受条件向量均(.5,1,1)，观测分布相同而梯度不同。固定有限次数、有限值的离散反馈估计器期望连续，TV符号梯度在p跨q时不连续，不能普遍无偏（不排除自适应查询/其他信息）。平滑目标1/2Σ(q−p)_+²可用拒绝bit R∇q(X)无偏估计，因 E[R|X]=(q−p)_+/q；不是CE或TV，未试。
若全p可用，给定Y=y，E[e_X|Y=y]=(min(p_y,q_y)/p_y)e_y+((p_y−q_y)_+/p_y)*(q−p)_+/δ，δ>0,p_y>0；Rao–Blackwell降方差，直接q−p更强。普通AR额外构造最大耦合也可复用本梯度；同信息平坦FL+SD完全复制。

## 3. 实际运行（E/P/O）
[probe.js](probe.js)、[results.json](results.json)，命令 node experiments/HC/R03/probe.js；JS V8实际执行并确定性重跑上传，LCG+Box-Muller非生产RNG，无依赖。
A：V64、4区域×40上下文、每512样本、seed11/29/73、σ=.1/.5/1.5，共1440上下文、737280次/方法；p=softmax标准高斯logits，q加σ噪声与区域sin小偏移。比较硬CE、独立差分、标准SD差分、同U逆CDF差分。N−1修正样本协方差迹，三种子均值：
|σ|TV|CE|ind|SD|commonU|SD理论|拒绝率|
|---|---|---|---|---|---|---|---|
|.1|.049742|.960172|1.920813|.097874|.604641|.098938|.049202|
|.5|.194142|.961842|1.914224|.377497|1.275639|.379435|.193091|
|1.5|.517558|.961809|1.829263|.937422|1.621731|.936644|.517647|
SD比CE方差降低89.8%/60.8%/2.5%；均值估计平方误差.0001888/.0007667/.0017982。非实际耗时或概率。
B：V32、4固定区域p_e=softmax(z0+σ噪声)，共享无上下文区分q初始softmax(z0)，每轮每区16样本，500轮同步梯度平均，固定LR .8，3种子×3σ×5方法=45运行，exact用完整概率。p_e为同一固定教师在不同上下文的条件分布，非混合服务目标；平均前向KL与KL(pbar||q)差常数。实际三种子均值：
|σ|初始KL|CE终值|ind终值|SD终值|commonU终值|exact终值|SD非零比例|
|---|---|---|---|---|---|---|---|
|.1|.00140268|.00376237|.00551643|.00019424|.00115013|.000000625|.032479|
|.5|.03745199|.00386397|.00540084|.00096791|.00293779|.00001736|.159510|
|1.5|.40963371|.00393314|.00786527|.00348893|.00624626|.00062092|.447292|
终点平均接受CE/SD/exact低.951888/.967156/.968256、中.834187/.841076/.841213、高.552952/.554937/.554613。exact KL始终最低，KL与TV最优不等价。低σ固定LR导致CE末期噪声，需调LR/衰减/批量后公平比较；不可宣称胜过调优最强CE。仅32维logits合成SGD，非真实模型反向、网络或实际HFL训练，平坦同步完全等价。无正式CI；逐种子及轨迹见JSON。非零样本比例非实际反向时间。

## 4. 成本和强基线
线性头每拒绝2行，FP32及4B ID时每样本期望2δ(4d+4)B，不计特征/版本/同步；dense W梯度4Vd是弱基线，硬标签蒸馏仅Y ID（本来返回），公共h可重建时云直接算更便宜。B个同分布独立样本，活跃行数期望Σ_v[1−(1−|q_v−p_v|)^B]，因每对拒绝出现词v的概率|q_v−p_v|；异上下文用逐位置乘积。批量大时可能变稠密。前向F、普通反向B_c、额外筛选/重算C，成本代理(F+δ B_c+C)/(F+B_c)，真实GPU kernel、padding、缓存及通信未知。全流程可跳过草稿反向占比f，最多节省f(1−δ)；f<.15时单此不可能达到G5。
强基线：同目标AR+硬标签蒸馏，普通HFL+内生SD+软/硬标签在线KD，独立/同U差分，训练期间直接最大耦合，条件积分及精确q−p（不同信息时分别报告），梯度压缩、采样softmax、普通稀疏行更新、学习率批量调优、同信息平坦FL+SD。**CRITICAL：** SD改变附属头梯度计算，但并非HFL独有且未改变正式目标学习；同信息平坦复制，既有控制变量数学与在线反馈近邻。非独立顶刊机制。

## 5. 文献（2026-11-24）
检索 verification feedback draft learning, coupled score difference gradient, acceptance-aware training, federated distillation hard-label。核对原始摘要/页面及Online Spec HTML正文片段，未穷尽公式、未公开工作、耦合梯度/采样softmax等。无实际文字查重。
- [When Drafts Evolve](https://proceedings.mlr.press/v306/qian26k.html) ICML2026，[HTML](https://arxiv.org/html/2603.12617v1) 在线验证反馈、DPO式优化、optimistic更新、EAGLE集成。
- [Verification-Aware Training](https://arxiv.org/abs/2608.30135) 2026-08-31，验证头和拒绝自适应权重；[Acceptance-Aware Draft Training](https://arxiv.org/abs/2609.24150) 2026-09-21 EAL/WTV及RL，区别固定CE估计。
- [Variational SD](https://proceedings.mlr.press/v306/zou26i.html) ICML2026 路径接受ELBO/EM；[Online SD](https://proceedings.mlr.press/v235/liu24y.html) ICML2024在线KD；[DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html) ICLR2024 on-policy与divergence。
- [Direct Alignment of Draft Model](https://arxiv.org/abs/2403.00858) 2024-02-29 TVD++方差降低，须进一步逐式核对；[AdaSPEC](https://arxiv.org/abs/2510.19779) 2025-10-22选择性蒸馏。
- [TallyTrain](https://arxiv.org/abs/2607.00173) 2026-06-30 hard-label联邦蒸馏/稀疏合并；[Communication-Aware KD](https://arxiv.org/abs/2509.01750) 2025-09-01联邦LLM Top-K logits。
- [Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html) 最大耦合；[LayerSkip](https://aclanthology.org/2024.acl-long.681/) 自推测。A/R05–06、A/R09、HiMerge/R03（REAL/EDR近邻）必须引用，后两篇本轮未独立逐式核对。经典score-function控制变量、耦合、条件积分并非新数学。

## 6. G1–G7及后续
主观创新问题7.5×25%、机制6.5×25%、理论7×20%、价值7×15%、文献距离4.5×15%=6.625/10约6.6。G1 FAIL不足8/近邻；G2 CRITICAL FAIL平坦复制、正式学习未改变；G3局部T/整体未过（完整greedy/随机/精度未实现）；G4 FAIL仅合成；G5未验证无真实wall time；G6未验证无80%证据；G7局部可复算/整体未过。好结果同正式目标/质量/资源下最强公平基线全流程实际时间−15%或同成本有效工作+20%、两预训练尺度、三种子三非IID/时延配置及CI和全部G。
仅改变前提才做预实验H：1×40GB Qwen2.5-1.5B，扩3B/7B，冻结教师主干、轻量内生头、8逻辑客户端2边缘、公开数学/代码/通用任务；先记录(X,Y,q快照,h)及教师p/位置选择，测真实梯度方差、反向耗时、稀疏行覆盖、激活重算及通信占比，调优公平基线；双卡实际物理部署才称分布式测量。若f(1−δ)<15%、平坦/全概率/调优KD匹配，停止；未执行GPU，不估成功概率或发表率。
**Reject and Pivot**：保留无偏耦合梯度、限定方差最优、不可辨识/尖峰反例、稀疏支持式、版本/选择偏差及合成SGD证据；不以在线KD、控制变量或拒绝训练换名重启。下一轮先获取真实预训练模型执行trace，定位正式学习与验证共享的实际工作及区域信息/异步屏障约束、时间占比，预设普通AR/SD和同信息平坦；不继续用toy结果代替实际问题存在性。
