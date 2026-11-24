# HIERASPEC/R09｜FAM-SD：自投机学习随机校正与FedAdam二阶矩非负无偏障碍
2026-11-24；起始main bffbeb29570c700c55da8f24bff5ad5b58a47bc3、README blob abbb7adb2a114717480af43c881723cabb7eba10，2778行。完整取得README并核验R08 e5eb076...原样前缀，读全部新增FCAS/FMR/FQR-SD，检查HC/R06完整报告/代码和CSV相关部分、FMR详细记录及FCAS结果；重审FUDG/FDC、FSOT、FSG。写前读并发HC/R07 FSC-Median（非线性中位数与全阶查询障碍，与本轮平方及非负性不同，但两者都否证一阶无偏推出非线性聚合无偏）。未独立复跑他队。**CRITICAL / Reject and Pivot**。T=限定数学；E=实际CPU从头训练小Transformer、单步投机采样、真实autograd和条件枚举；P=成本代理；O=预先完整梯度的审计oracle；H=未验证。实查torch2.10.0+cpu、CUDA false/0、无transformers/HF cache或/opt/models；未获得预训练权重，本轮未重新联网下载。无预训练LLM、物理HFL、完整多token SD、实际在线RL续训、GPU/网络/全流程wall time或CI。

## 1. 场景与机制 T/H
固定合法模型快照、已承诺生成轨迹/奖励及固定cohort，p为完整模型、q为共享主干浅层内生头，无独立完整draft LLM。正式学习对象是同一批轨迹的奖励加权负log p梯度（固定奖励及轨迹detach），而非修改正式损失为浅层损失；实验用单token真实下一字符奖励，非GRPO完整组。区域等权聚合，云服务器拟按FedAdam处理一轮更新；只讨论固定一步梯度/等价负学习率模型增量，非多步本地AdamW/FedAvg。所有g_i、h_i已吸收客户权重，H=sum h_i，d_i=g_i−h_i，G=H+sum d_i。先得到已知廉价h_i及预定pi_i>0，独立I_i~Bern(pi_i)，实际查询完整g_i仅当I_i=1，构造Ghat=H+sum I_i d_i/pi_i。固定cohort条件无偏（FUDG已有）。本次采样只作用于反向，不以接受率跳过正式验证；实际q/p/奖励/版本均需合法。若pi依赖未观测d或未承诺的当前随机输出，重证；不声称DP。
真实Flower FedAdam源码[文档L467–492](https://flower.ai/docs/framework/_modules/flwr/serverapp/strategy/fedadam.html) 对聚合delta_t平方维护v_t；本轮使用梯度符号与首步bias-corrected Adam代理，并非运行Flower。普通Adam对原随机梯度二阶矩的估计并无此“错误”；本轮要求匹配固定完整cohort的参考G^2。E[Ghat^2]=G^2+sum (1−pi_i)/pi_i d_i^2（逐坐标）。若额外传可加统计S=sum I_i(1−pi_i)d_i^2/pi_i^2，则M=Ghat^2−S满足E M=G^2；两级线性加法可归约。条件方差已知时直接减真方差也是oracle。若采样相关，需要pair inclusion及协方差，不能套独立公式。M可能为负；取M+或把EMA/分母裁剪后不再保证无偏，且E[m/sqrt(v)]不等于参考更新。只说明有限固定cohort的一步语义，不否证自适应优化器容忍随机梯度、长期一致性或同质量收敛。

## 2. 非负无偏查询下界 T
任意固定一个坐标，假设H已知、所有d_i为任意独立可变实数，采样子集S独立于未知d且只可见d_S，额外估计随机性亦与未查询值无关。要求Z≥0 a.s.且对所有d满足E Z=(H+sum d)^2。对任意真不完整S及已见d_S，可选一个未查询d_j使总G=0（其它未知置0）。此世界E Z=0且非负，故在该正概率S和已见值下Z=0 a.s.。观察相同的任何其它d世界也必须返回0。故Z只可在全查事件非零；若P_all=0无普适无偏估计。若P_all>0且G≠0，Jensen给E Z^2≥G^4/P_all，Var(Z)/G^4≥1/P_all−1，Z=1{all}G^2/P_all达界。独立同pi时P_all=pi^N，pi=.5,N4/8/16的相对方差下界15/255/65535。普通整cohort一次Bern(pi)全查或全不查，等期望深层查询数Npi，P_all=pi，平方非负估计相对方差1/pi−1；它不保证同一阶方差、每轮计算预算或实际Adam更新等价，且仍可能让更新等待尖峰。若某坐标有已知界且部分视图不能扩展为G=0、可访问公共精确充分统计、先验关联、许可偏差、不同查询模型或目标为E[Ghat^2]，下界不适用。不能称所有FedAdam/SD不可能。与HC/R07 median的高阶信息下界邻近，但此处平方为二次函数、允许有符号的无偏估计，无须全查；障碍来自非负性。经典非负无偏估计和HT方差估计已有，非新基础理论。
最小反例 H=0,d=(1,−1),pi=.5。仅一人查时Ghat=±2,M=4−2=2；两人均查时Ghat=0,M=−4；皆不查为0。E M=0但E max(M,0)=1；真实G^2=0。固定一阶无偏不保证非线性更新。例如标量G=1，估计−1或3各半，E=1，但忽略epsilon的首步Adam方向平均0而参考1。两级区域若各自对区域G_e平方，漏掉跨区域交叉项2G_e G_f；须先聚合Ghat和S，再算平方，普通同信息平坦完全复制。SD反馈没有独立的正式优化器原语。实际标准SD给定真实p/q、正确正残差、独立随机、因果截断、合法bonus/EOS/KV及processor/top-p、数值语义才能逐位置归纳；greedy正式argmax及tie-break。未做完整生产正确性或随机联合序列认证。

## 3. 实际执行 E/P/O
[probe.py](probe.py)、[完整原始JSON无损gzip+base64](results.json.gz.b64)。运行 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python experiments/HIERASPEC/R09/probe.py > results.json；归档解码 base64 -d experiments/HIERASPEC/R09/results.json.gz.b64 | gzip -d > results_original.json。原始JSON SHA256 6d54cca754f88a1c741ec818d2a71924ec48bd1acf8585aa58eaa4154de57d13；实际脚本SHA256 90294d4399852fcf62031c8ea1a9a88fce6c2fd37416769884d9b83c5ba8c32d；远端blob分别（压缩文本）ee93ac51ca0617d32b3eaf30140c9882ba1623d1、0dbe2e3fc7a2135128eeb8003cd3d1d2c28b5570，与本地原件git hash-object逐字节匹配。未独立重跑模型；计时不应期望逐值相等。
本机CPython标准库email/json/urllib三域31文件，[上游](https://github.com/python/cpython/tree/v3.13.5/Lib)，本机发行版可能有补丁，原始字节连接的hash及长度见JSON；小写非字母合为空格V27，训练328703字符、留出82176字符（212572/26208/89923训练，53143/6552/22481留出）。T12、B4、2头、共享LayerNorm和输出投影、dropout0、L2 d24 11403参数与L3 d32 27835参数；seed11/29/73各60步AdamW .004，CE(p)+.3CE(q)从随机权重实际训练（非公开预训练模型）。每模型8固定cohort×4逻辑客户端（域i%3，同域两个客户端，共享同一CPU，无真实分层进程）×4留出前缀，共48 cohort、192对真实正式/浅层梯度、768单步实际SD样本。真实reward=1{Y=真实下一字符}−.05（固定baseline），实际p/q为FP32 softmax再FP64归一化用于接受/正残差torch.multinomial；训练log_softmax为FP32，未证明位级一致。先算全部正式及浅层梯度，再枚举16个审计mask，三pi .25/.5/.75，实际2304个条件mask配置，18行逐seed统计（每行8 cohort均值）。不是实际按mask省时、不是多token SD或在线RL更新/GRPO、未做AR多token对照。条件解析矩按已取得的真实autograd计算，初次Adam代理m/v从0开始、bias-corrected、eps .001，比较E[一步方向]与完整梯度参考方向G/(|G|+eps)的相对L2误差；不执行参数更新。clipped M仅此一步裁剪，非完整DP-AdamBC长期方法。结果三seed平均：
|L|pi|观测接受|额外平方质量/||G||²|负M坐标比例|裁剪后平方质量相对超额|naive一步方向相对误差|clipped一步方向相对误差|
|---:|---:|---:|---:|---:|---:|---:|---:|
|2|.25|.921875|.489380|.143849|.108806|.399052|.829145|
|2|.5|.921875|.163127|.185547|.036921|.244503|.453154|
|2|.75|.921875|.054376|.151471|.007373|.113950|.248793|
|3|.25|.911458|.761958|.134963|.171712|.455120|1.040594|
|3|.5|.911458|.253986|.198269|.061418|.279836|.585385|
|3|.75|.911458|.084662|.175602|.014407|.132861|.309147|
最大逐seed平均坐标无偏误差梯度3.81e−18、平方4.58e−18（FP64枚举；不是生产位级正确）。负坐标比例是按mask概率加权后全参数比例，再cohort和seed平均。即使高接受，校正后仍负；裁剪方向误差全部六均值比naive大；非一般收敛结论，未调eps/动量/长期裁剪，也未测任务质量。全梯度基线参考方向误差0（相同计算）。整组Bernoulli理论对照未执行实际训练。普通AR生成相同边际、独立小批/完整梯度、已有方差估计与平坦均可替代。不能称小模型成功满足G4。
真实串行CPU独立full/shallow前反向中位数的三seed均值 L2 3.25148/2.23321ms,L3 5.36545/2.55200ms；每次full与shallow各重新forward，含autograd及扁平化转换，不含生成、训练optimizer、网络、统计归约；未测融合校正。用非复用s+pi f代理，pi=.5时约3.859/5.235ms，相对完整约+18.7%/−2.4%，不是实际端到端时间，L3仅微小代理。理想完全复用的s+pi(f−s)也需实际测；所有当前cohort全算，所以没有实际省时证据。通信P：额外S向量若稠密FP32 D维为4D B/区域/轮，原Ghat同维，两向量8D B，相对单向量多100%；在本模型D11403/27835时额外45612/111340B，10M训练参数示意40MB，未计压缩、稀疏深层坐标、版本、上下行、协议及安全归约。可本地联合压缩或近似校准，但更改估计条件；不能与故意低效全量传输基线比较。实际梯度反馈与真实通信占比、长期收敛、同质量预算全流程未测。

## 4. 文献与裁决
2026-11-24实际检索并核对原始页面/摘要及部分HTML正文：
- [Adam](https://arxiv.org/abs/1412.6980) ICLR2015；[Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) ICLR2021；[Flower FedAdam源码](https://flower.ai/docs/framework/_modules/flwr/serverapp/strategy/fedadam.html)，基本优化器及真实接口。
- [DP-AdamBC](https://arxiv.org/abs/2312.14334) AAAI2024（正文§3–4及收敛附录相关段落，独立DP噪声减方差和非负截断已有，直接近邻；其长期一致性条件不等于本轮固定cohort首步）。
- [On nonnegative unbiased estimators](https://arxiv.org/abs/1309.6473) Jacob/Thiery，2015，非负无偏估计的一般限制，直接理论威胁。
- [HT variance estimation](https://www150.statcan.gc.ca/n1/pub/12-001-x/2013001/article/11831/section2-eng.htm) 固定有限总体及联合包含概率，普通平方校正并非新原语。
- [Dissecting Adam](https://proceedings.mlr.press/v80/balles18a.html) ICML2018，随机梯度符号、幅度与方差；[Adam+](https://arxiv.org/abs/2011.11985) 自适应方差缩减。
- [SPEC-RL](https://arxiv.org/abs/2509.23232)、[EfficientRollout](https://arxiv.org/abs/2606.18967)、[SpecRoll](https://arxiv.org/abs/2608.04962)、[System-integrated SD in RL](https://arxiv.org/abs/2604.26779) 真实在线生成与学习系统近邻。未穷尽全文/近期FedAdam方差校正或未公开研究，无正式文字查重；不声称首创。
**CRITICAL / Reject and Pivot**：主观创新6.325≈6.3/10（问题7.5×25%、机制5×25%、理论7.5×20%、价值7×15%、距离4.5×15%，算术为6.325）；G1/G2/G4 FAIL；G3/G7局部成立/整体未过；G5/G6未验证，无80%依据。理论障碍可用于所有抽样正式梯度研究的验收，但DP-AdamBC/HT/普通全批与整组抽样、同信息平坦可替代；非负修正不证明训练加速，SD未形成不可替代反向机制。好结果仍同质量预算最强公平基线真实全流程−15%或有效工作+20%、两公开预训练尺度三非IID/时延三独立训练seed、CI及全部G。未执行1–2×40GB预实验，不为此否决机制投入GPU。下一轮若获得实际Qwen2.5-1.5B/3B或公开兼容内生头模型，8逻辑客户端2边缘，真实在线RLVR/GRPO，先测固定cohort与优化器状态、实际层级权限和计算/通信，比较普通HFL+生产自SD、AR、固定完整梯度、常规采样与方差修正、同信息平坦。若缺真实部署约束及联合trace，停止继续改名概率估计器；保留二阶矩冲突、非负查询下界、裁剪反例、真实小模型autograd及负实验。
