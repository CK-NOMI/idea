# HC/R07｜FSC-Median：联邦非线性聚合的浅层梯度候选、精确区间证书与无偏性障碍
2026-11-24；起始 main e913dfd74ead040fbf95eb34a9f62b78c47e378c，README blob 4d533746be7909cd7a4c96250636b5c92f24bc1c（2643行）。完整获取README并验证HC/R05归档为原样前缀，分段读全部16项新增记录，读HC/R06详细报告、主程序和CSV、HiMerge/R08详细记录。HC/R06已占用故本轮HC/R07。写前读并发FSP-DPO、FSDC、FSG、FCAS、FMR，与本轮非线性正式聚合不同；NF-RSD为非线性概率聚合，不同于本轮梯度中位数；FUDG/FDC只针对线性平均，FRGB非线性组统计，FCF非线性逆/PSD警示。未独立复跑他队。
**CRITICAL / Reject and Pivot**。T=限定数学；E=实际CPU公开源码小型Transformer训练及真autograd、数值审计；P=成本代理；O=知道真实误差的oracle；H=假说。实查Python3.13.5 torch2.10.0+cpu、CUDA false/0、无HF默认缓存，未重新尝试网络下载；无真实预训练LLM、GPU、物理HFL、真实SD rollout、网络/总wall time/CI。实验没有攻击、密码证明或真实鲁棒收敛测量。

## 1. 系统场景、正式对象与候选机制（T/H）
固定公共参数版本theta、奇数N=2m+1个等权客户端、预定数据batch；共享浅层内生头q及深层正式p，不部署独立完整草稿LLM。定义共同目标L=CE(p,label)+.3 CE(q,label)，g_i=∇L_i，c_i=∇[1.3 CE(q,label)]（缺失深层坐标填零）；所有对照采用同一定义，不冒充纯NLL。选定正式聚合规则M_j=median_i g_ij（一个固定快照的坐标中位数，不是普通FedAvg或一定存在的全局损失梯度）；不保证在非IID/恶意攻击下具有优于其他鲁棒规则的学习质量。区域只持合法本地数据/梯度，可向云报告授权区间、候选、精确梯度或按查询返回秩计数；未证明实际私有约束、DP或安全认证。机制拟以早退出起草路径的c提出梯度候选，再用合法可信上界b_ij≥|g_ij−c_ij|证书验证最终聚合。注意这借用了SD的候选—验证计算思想，不等同已经执行token级SD。希望省完整梯度计算；正式更新后p和q会改变，但没有真实SD反馈改变聚合规则的证据，必须以普通鲁棒学习+SD和同信息平坦为强对照。
若推理执行标准SD，当前固定正式p与真实提议q（同温度、实际top-p/processor）按min(1,p/q)接受、正残差补采、真实因果前缀、独立随机数、首拒绝截断、合法bonus/EOS/KV时，单步核p且条件归纳序列分布正确；greedy按正式argmax及统一tie-break。不是本轮实现的新采样器，未运行完整greedy/随机/浮点契约；报告1−TV为teacher-forced理论单步重叠而非实际接受率。

## 2. 三个限定数学结果（T）
(1) 层级median非结合：三区各(0,1,100)、(2,3,101)、(4,5,6)，局部中位数1,3,5，再取中位数3；全九项中位数4。即使无攻击且组大小相等也不同。若目标本来就是median-of-medians则不同规则合法，不应称精确全局median。按各坐标的精确秩计数（给阈值t，各区计算# g≤t并求和）或分布式精确选择可以恢复全局顺序统计，普通平坦和区域委托均可实现，不能靠区域中位数直接替代。
(2) 区间证书：每坐标对所有合法g_i∈[l_i,u_i]，median(l)≤median(g)≤median(u)，由坐标单调性。两个端点通过全部取l或u可达，在笛卡尔积不确定集内区间界锐利。若median(l)=median(u)，即精确证书；否则在未观察其它约束时至少有两个可能聚合结果。逐次选择一个客户端查完整g，把其区间缩为单点直至全D坐标闭合。若该客户端是某坐标严格唯一中位数，且其未查区间非退化，则在该区间内小幅改变这个值仍保持名次而改变median；故在仅有独立区间、完整客户端向量查询的模型，所有此类中位数所属客户端均须查询。这是证书模型下界，非任意梯度相关结构、精确误差等式、跨坐标线性关系、参数块查询或实际GPU FLOP的下界。真实梯度区间可能相依；实际程序使用真实|g−c|作为上界oracle（仅当作不等式，非已知精确绝对误差等式）。若误差界不可信、恶意客户端伪报，则证书不成立；无安全协议。
(3) 线性无偏校正不能穿过median。例g=(1,1,−1),c=0，均匀抽1项完整，HT估计H_i=3g_i若抽中否则0，各项无偏，median(H)=0恒定但median(g)=1。更一般，设N奇数、独立均匀符号x_i∈{−1,1}，f(x)=majority=median(x)。完整阶Walsh系数a=E[f(x)Πx_i]=(-1)^m binom(2m,m)/2^(2m)≠0（按正号数k求和、交替二项式恒等式）。任何预先独立于x选择查询子集S的估计器H，若S不全，条件期望h_S(x_S)与Πx_i正交。于是对所有x无偏需要π=Pr(S=all)>0；若固定预算最多N−1次，普适无偏不可能。对全揭示事件，投影系数必须为a/π。Jensen/正交投影给出输入与随机性平均MSE E_{x,S,R}(H−f)^2≥a²[(1−π)+π(1/π−1)^2]=a²(1/π−1)。证明取条件于S的均值、在全阶Walsh基投影；不完整S误差系数−a。若每客户端独立半概率查询，则π=2^-N。适用于独立于x的查询掩码、仅观测x_S及独立随机性、任意输入无偏；可适应查询但每路径最多N−1的决策树多项式次数≤N−1亦不能无偏，但上面方差界不直接适用适应掩码。若有相关结构/合法廉价信息、允许有偏聚合或以抽样参与规则为新目标，则不适用此普适结论。经典Boolean Fourier/有限查询论的应用，不声称首创。
最强简单无偏基线：直接估计完整非线性输出，H=M(c)+I/π_global [M(g)−M(c)]，整个组共同Bernoulli且π_global>0。无偏且条件方差=(1/π_global−1)||M(g)−M(c)||²；这避开独立半查询π=2^-N但仍需整组深层计算，属于普通控制变量。此基线仅理论分析未实际训练；各客户端独立修正与整组修正应严格区分。普通AR学习亦可实现全部数学。

## 3. 实际执行（E/P/O）
[实际代码](probe.py)；[18个逐cohort完整指标CSV](results.csv)，命令 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python experiments/HC/R07/probe.py。实际脚本保存results.json（包括31个输入文件SHA256、环境、完整曲线）；CSV将分数曲线以精确整数闭合坐标计数表示，分母D，其他原值保留。实际本地脚本Git blob c93b2a79a377f9685c4e2c35b97540bd49c6d41c，CSV blob 710f8a6f312c0754c492e0b36895afef0f97d10c，远端均已匹配；本地原始JSON SHA256 1ce0cd8484551d0751725e5cdddc3fe5a12dbfc1df6d22ad462f6bfe356804e4，JSON可由代码重新生成（不同环境可能数值不同），归档另存metadata含输入manifest及Fourier表。数据本机CPython3.13.5公开标准库[上游源码](https://github.com/python/cpython/tree/v3.13.5/Lib)，email/json/urllib各顶层.py，31文件，按文件名拼接字节，前80%训练后20%留出；训练298207/39078/129748=467033字节、留出74552/9770/32437=116759字节（可能存在相关模式，不宣称benchmark无泄漏）。
两尺度分别2层d32=34304参数、4层d48=101664参数，2头、preLN/GELU、序列24、batch4、256 byte词表，浅层为中点层并共享输出头，AdamW .003，3独立初始化seed11/29/73每尺度100步从头训练，三域轮换。每个模型3个固定快照cohort，每cohort九逻辑客户端，三区各3个同域留出batch，实际18个cohort/162对真实autograd梯度、15552个teacher-forced位置（可能重复）；没有真实多机构或联邦续训。FP32梯度转FP64做median/证书，非生产数值精确性证明。实际计算了所有完整梯度才能得到误差界；oracle证书回放不省真实运算。三个区间粒度：每客户全向量max残差、每参数张量max残差、每坐标真实绝对残差（均oracle），非退化区间向外nextafter；按与尚未闭合的中位数区间重叠坐标数最多选择客户端并揭示完整梯度，直到所有坐标闭合。对照完整全局中位数、区域median-of-medians、浅梯度median、均匀选3或6项完整的hybrid median及HT修正median（每cohort每预算12次随机无放回样本，共432次）；相对平方误差=||估计−精确M||²/(||M||²+1e−30)，不是任务loss/攻击成功率。三种子×3cohort的等权均值：
|尺度|理论p/q重叠|median-of-medians误差²|浅层median误差²|唯一中位且非零残差的坐标比例|所有winner客户端数|
|---|---:|---:|---:|---:|---:|
|2|.930056|.251874|.170143|.778043|9/9（全部9组）|
|4|.951079|.172106|.106595|.887349|9/9（全部9组）|
三证书54次均查9/9完整客户端，最终最大数值误差0。查0/3/6/8/9时最紧oracle坐标证书闭合比例：2层 .220467/.286481/.479714/.713053/1；4层 .111830/.185717/.352623/.612704/1。较粗client-max证书查8后仅 .223709/.113485；tensor-max查8后 .256112/.175837。所有九客户端均拥有严格唯一中位且残差非零的坐标，因此在上述独立区间、全客户端查询模型下，换查询顺序仍必须全查。不能外推为任意模型普遍不可能。
|尺度|完整查询数|hybrid相对平方误差|HTmedian相对平方误差|12次HTmedian均值相对平方误差|
|---|---:|---:|---:|---:|
|2|3|.109975|.150510|.069804|
|2|6|.051039|.059371|.019627|
|4|3|.073189|.113516|.055712|
|4|6|.034396|.040144|.015496|
末列含有限MC误差，不是实际偏差的精确估计，普适有偏由三标量反例独立证明。未做长期鲁棒学习收敛、真实SD rollout、真实网络、总时间、CI或GPU。
实际枚举全部2^N符号，N3/5/7/9，a=−.5/.375/−.3125/.2734375，π=1/8,1/32,1/128,1/512，积分MSE下界1.75/4.359375/12.40234375/38.20648193359375。这个是实际有限枚举加解析下界，不是Transformer误差或经验方差。

## 4. 成本与强基线（P/T）
完整9客户端FP32 D维上行9×4D；浅层候选全发、再9项完整梯度时，朴素载荷至少18×4D+界/查询消息（若利用零坐标/压缩则另计，不能当物理通信）。D=10M示例360MB→720MB，不含下行、版本、区域→云、排序、重复/安全协议。粗界每客户一个数，tensor界每张量一个，coordinate界本身约9D个值；oracle获取实际完整梯度且通信不免费。若区域仅报median改变目标；可用精确分布式选择/秩查询与局部原始数据，普通平坦同授权和委托可复制；实际算法可能比全向量上传好，不能宣称通信普适下界。省单客户端完整梯度的理想成本模型：C_full=f，廉价候选s>0，查询后的新增完整计算d≥0，若浅层完全复用d=f−s，则C_candidate=N s+k(f−s)+额外，节省比例(1−k/N)(1−s/f)−额外/(Nf)。本轮k=N即理想无节省，另计真实额外反向、证书计算、通信/同步及推理；没有测真实f,s,d。若只做普通全梯度中位数+生产SD已经强，不能把其推理收益算给本机制。整组随机校正可比较成本×方差但不保证同质量或鲁棒性，未在真实训练测。

## 5. 文献与审查（2026-11-24）
实际web检索并核对原始论文页面/摘要：[Yin et al. 2018](https://proceedings.mlr.press/v80/yin18a.html) median/trimmed mean鲁棒GD；[BOBA AISTATS2024](https://proceedings.mlr.press/v238/bao24a.html) label skewness；[Client subsampling/local updates ICML2024](https://proceedings.mlr.press/v235/allouah24a.html) 部分参与与鲁棒FL；[Communication Compression for Byzantine Robust Learning AISTATS2024](https://proceedings.mlr.press/v238/rammal24a.html) 通信压缩；[No-Attack Accuracy ICML2025](https://proceedings.mlr.press/v267/yang25aa.html) 鲁棒聚合误差；[Aggregating Normalized Gradients NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/29c71a71cf3b354036d9b413cd049cbf-Abstract-Conference.html) 计算/异质性；[Delayed Momentum Aggregation ICML2026](https://proceedings.mlr.press/v306/otsuka26a.html) 部分参与/通信；[Reconciling Compression and Robustness AISTATS2026](https://proceedings.mlr.press/v300/gupta26a.html) 压缩与鲁棒性冲突；[Exact Distributed Quantile Computation](https://arxiv.org/abs/2511.12025) 精确选择近邻；[LayerSkip](https://aclanthology.org/2024.acl-long.681/) 与 [ReeFL](https://proceedings.mlr.press/v235/lee24h.html) 早退；[MLMC gradients](https://proceedings.mlr.press/v130/shi21d.html) 随机去偏。未穷尽最新区间选择算法、Boolean Fourier、鲁棒优化及全文/专利/未索引工作，未正式文字查重；基本中位数、区间单调和Fourier非新基础数学。

## 6. 验收与下一轮
主观创新6.325≈6.3/10（问题7.5、机制5.5、理论7.5、价值6、距离4.5；权重25/25/20/15/15），非同行评审。G1 FAIL（已知鲁棒聚合/选择/多保真）；G2 CRITICAL FAIL（普通AR+鲁棒训练及同信息平坦可复制，无真实SD反向因果）；G3 PARTIAL/FAIL（证书与受限下界成立，完整精确SD未实现，真实区间不可免费获得）；G4 FAIL（两小模型从头训练非两公开预训练模型，无真实鲁棒训练或物理HFL）；G5未验证，oracle全查否决本次假设下省深层计算；G6未验证，无80%证据；G7局部复算/整体未过。好结果同正式聚合/任务质量/资源/鲁棒性最强公平基线真实全流程−15%或有效工作+20%，两预训练尺度三非IID/时延三独立训练种子、CI及全部G。
H未执行：仅发现合法不需深层计算的真实梯度约束且可按参数块独立生成VJP、实际中位winner覆盖低或跨坐标可认证关系后，再考虑1–2×40GB Qwen2.5-1.5B/3B、9逻辑客户端3区域、公开数学/代码/通用任务、LoRA固定版本监督或有真正rollout的RL；比较完整median+生产SD、精确分布式选择、普通HFL+SD、鲁棒压缩/部分参与、整组随机校正、同信息平坦和普通AR候选；至少10000真实位置、真实梯度块及获得证书的实际成本、前后向/通信/回退、任务质量及攻击条件（同威胁模型）、CI。现不推荐为本机制耗费GPU。下一轮先从实际在线FL/RL系统接口核实真实跨层驻留约束，取得预训练模型/真实联合学习验证trace；停止继续包装普通梯度代理及中位数查询。保留非结合反例、独立区间证书、winner覆盖下界、有限查询无偏方差障碍、真实小模型负结果与整组校正强基线。
