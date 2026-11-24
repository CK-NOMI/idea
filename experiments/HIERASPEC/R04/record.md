# HIERASPEC/R04｜LBS：晚绑定草稿的精确验证摘要与有限随机缓存边界

日期2026-11-24；起始 main 1ccba02e5c828fe2e9061b3848a9d753a9d3e851，README blob 4b41882464b8c4301465eef636f903a64ccc9335，1974行。读取固定快照，核验上轮最终README为原样前缀，阅读新增FBS、FCR、FVC、FVT、CPG和相关A/R07–09、HiMerge/R01及FVC详细记录；检查递归树。上传前读并发HCA/R04 HPRS全文及HieraCert/R05 FSOT（含代码）。HPRS是未知归一化参数的残差上包络，且已独立记录预采p候选强基线；本轮有限缓存算法与其预采及HIERASPEC/R01正残差回退边界重合，不宣称新算法。增量是晚绑定q时确定性通用摘要的信息量障碍、缓存样本自适应依赖反例、真实公开输出层瓶颈审计、实际数值验证及成本门槛。FSOT是多步优化事务，与此不同。未独立复跑其他团队。
**CRITICAL / Reject and Pivot**（保留限定边界，不立项）。T=数学，E=本轮实际CPU合成随机模拟/归约检查，E-H=论文报告GPU实测未独立复现，P=算子及通信成本代理，H=未验证。无本队预训练权重、真实模型trace、HFL训练、GPU、实际网络/wall time/CI。不得把论文真实结果冒充本队测量。

## 1. 真实问题审计、场景及数学对象
原始Qwen2.5 1.5B/3B官方config：V=151936,d=1536/2048，BF16输出矩阵2Vd=466747392/622329856 B，8192训练位置的BF16 logits矩阵约2.489GB（仅形状推算，不含缓存/梯度/参数共享等，不是显存测量）。[1.5B config](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json)、[3B config](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json)。
[CCE](https://arxiv.org/abs/2411.09009) Table1报告Gemma2 2B/8192 token、V256000,d2304、A100-SXM4 80GB、PyTorch2.4.1 CUDA12.4：普通loss峰值24GB、loss+gradient 207ms/28GB；torch.compile 143ms/16GB；CCE 135ms/1164MB、loss 1MB；无梯度过滤CCE 362ms/1162MB。大显存收益不等于时间收益；梯度过滤不是实数完全精确反向。[FlashSampling](https://arxiv.org/abs/2603.15854) 原始正文§4.5 B200 vLLM报告Qwen3-1.7B/8B/32B、Llama3.3-70B TPOT最高下降10.2/8.7/2.9/2.7%；摘要写最高19%，采用明确正文设置数字，不混用版本/场景。其Table1融合采样占matmul+sample kernel时间2.3/2.3/6.0/6.2%（B=1/16/64/256）；非全模型占比。SIMPLE亦已有决策面及hot-vocab拒绝采样。上述证明一般输出层瓶颈，不证明HFL特别瓶颈或本候选优于强基线。
HFL训练单个正式模型（非混合），边缘持同版本的内生轻量头q，无独立LLM；假设输出投影由授权边缘/云执行或按词表切片，固定同一前缀/hidden h、目标p=softmax(W h/T)且T>0、无截断。HFL本身不要求词表分片；真实权重驻留/晚到q是否常见尚未证。正式训练CE=-ell_y+logsumexp(ell)，SD需p(x),q(x),及正残差。研究晚到的实际q或候选x（可能是已完成训练/验证输出投影后才知道的任务），只允许紧凑缓存、合法点积查询、必要时重算；目标同版本精确随机及greedy与全流程质量时间。若参数、前缀、温度或采样设置改变，不可复用；不承诺DP。

## 2. T：晚绑定草稿的确定性摘要障碍
假设先读完整p，得到确定性b-bit摘要S(p)，之后任意q才被指定，不允许点查询/重算，要求仅凭S(p),q,x精确返回标准最大耦合接受概率（数值而非随机决定）。选q=点质量delta_i，则候选x=i，接受概率p_i。若所有i可查询，则S必须唯一确定p。对偶数V和整数L>=1，构造V/2个词对，各对未归一权重(a,2L+1−a)，a∈{1,...,2L}独立变化，总质量固定(V/2)(2L+1)。共有(2L)^(V/2)个不同p，故b≥(V/2)log2(2L)。例如L=128,V=151936时至少607744 bit=75968B；是受限网格信息下界，不是必须保存FP32向量的证明。实数无精度限制时有限位摘要不能精确编码任意p。此定理仅适用确定性通用数值查询，不是所有随机精确耦合/采样/压缩协议下界；对于点质量q只需精确p采样再匹配即可，不用返回p_i数值。若缓存logZ且可访问W_i,h，则p_i=exp(ell_i-logZ)可点查询，绕开下界但要计计算/通信，不能宣称HFL信息约束自动成立。
T：训练损失前向可用分片(m_g,s_g,label logit)，m_g=max ell,s_g=sum exp(ell−m_g)，合并m=max m_g,s=sum s_g exp(m_g−m)。精确输入梯度∂h CE=(Σp_i W_i−W_y)/T（ell=W h/T），能另聚合d维加权矩；若W可训练，各行梯度(p_i−1{i=y})h/T，通常稠密，logZ与少数样本不能提供完整梯度。可缓存/重算或另作无偏估计，后者不是精确梯度。CCE/TP已有精确归一化归约，不能重复当新原语。训练和推理同权重并不保证同hidden；未证明真实命中率。

## 3. T：固定随机缓存和回退正确性（非新算法）
在首次目标词表扫描时，附带K个独立p样本Z_j（有放回），可用每j独立Gumbel-max，各片记录局部winner/score，根取最大；同时保存logZ、已知候选logit（未知可点查询）。保证随机缓存独立于q的选择、候选X~q和接受随机数；各样本可保留logit，q(Z_j)需真实评分。先标准a(X)=min(1,p(X)/q(X))，接受则提交；拒绝后按j顺序用新独立U_j接受Z_j，概率[1−q(Z_j)/p(Z_j)]_+。K次均失败则独立精确残差采样（第二遍归约，或同分布其它精确方法），不得直接抽p。若δ=TV(p,q)>0，每提议成功δ、成功质量[p−q]_+，故输出边际p。回退事件依赖此前缓存随机性，但独立后备精确残差样本且每次成功分布同残差，因此混合正确。δ=0不拒绝。多步需每到达前缀的真实条件q、合法因果独立随机数、首次拒绝截断、bonus/EOS/KV正确；greedy正式argmax统一tie-break；未实现完整解码、top-p、有限精度。
无条件回退率F_K=δ(1−δ)^K（K=0为δ）；已生成缓存固定K个，实际使用提议数期望1−(1−δ)^K（δ=0为0）。K≥1时maxδ F_K=K^K/(K+1)^(K+1)，在δ=1/(K+1)，与R01 M=1特例相同；K=8最坏约4.33%，但不等于延迟上界。独立随机性的反例：p=(1/3,1/3,1/3)，缓存Z~p，若Z=0则选q=delta_1，否则q=delta_0；候选必为q支撑，接受1/3，拒绝时Z必在残差支持且以1接受，输出(4/9,3/9,2/9)非p（TV=1/9）。因此不能把同一随机缓存反馈给区域草稿选择器后继续套独立性证明；如需自适应必须用新的独立缓存或重新证明条件核。对比HCA/R04 HPRS的随机race溢出回退警告：此处是缓存影响实际q选择，不同但共通随机条件风险。

## 4. E：本轮实际运行与原始结果
[probe.js](probe.js)、[results.json](results.json)，node experiments/HIERASPEC/R04/probe.js；JS V8、LCG/Box-Muller仅可复算，非生产RNG。实际重读已上传脚本并执行，JSON逐值一致。64词，3seed(11,29,73)×3σ(.1,.6,1.5)，每组p=softmax(高斯)，q=softmax(同logits+σ独立噪声)。K=0/1/2/4/8，每方法100000次单步模拟，共45组450万次，按理论q/P/residual CDF抽样；缓存按使用时懒抽，与独立预采等价，不是实现融合GPU kernel。另模拟真实词表尺寸151936的随机logits，4独立Gumbel流，2/4/8词表分片及3种子，比较归约与单扫描；本程序预存随机数和logits，未实现无物化kernel。9/9 winner及候选logit相同，LSE最大误差2.043e−14（FP64，不是生产精度证明）。
三种子平均，Gumbel匹配概率解析Σ_i 1/Σ_j max(p_j/p_i,q_j/q_i)（DISD近邻），标准最大耦合α=1−δ；F_K为实际抽样回退频率：
|σ|δ|α|shared Gumbel|F0|F1|F2|F4|F8|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|.1|.046256|.953744|.935010|.046110|.044440|.042390|.038053|.031973|
|.6|.263454|.736546|.662296|.263807|.194920|.143607|.076810|.022963|
|1.5|.604741|.395259|.329980|.604350|.239493|.094087|.015157|.000317|
F8理论三种子均值.031651/.022888/.000362；K8实际接受.953317/.736467/.395053；实际使用提议.318177/.912147/.998933。经验输出TV约.0084–.0100为各方法三种子均值，有限抽样误差，不是某方法更精确；无CI/时间/真实模型/梯度。p/q人造高斯无真实分布证据。
P成本：比较K0“仅实际拒绝时精确第二次扫描”，归一化其一次后备成本C2。K个缓存额外生成各c（含通信/随机数/维护），实际查询q及额外管理暂假设免费，则额外成本Kc+C2 F_K。K>0严格胜出必要充分条件 c/C2<δ[1−(1−δ)^K]/K（该简化模型）。第一样本边际收益δ²C2。K8三种子平均阈值分别0.1826%、3.0071%、7.5547% C2；若q点查询非免费则更低。高接受率σ.1时即使K8将F降低约1.46pp，缓存每样本只要超过C2的.183%即得不偿失。在每样本成本1%C2且K≤32的同模型代理最优K分别0/7/4；5%时0/2/3。计算为上传数据的实际JS重分析，非实测硬件。一般预算要考虑共享GEMM可使边际成本非线性、并行/重叠、多次q打分与首次拒绝位置，而非简单每Token乘K。
通信P：S个词表分片，FP32局部max和sum各4B、K个winner/score/logit每12B，单向摘要S(8+12K)B，加候选点logit、q评分、标识、下行/回退。S=2,K=8为208B（示意）；同精确分片两阶段只需8S+4B候选，拒绝时广播归一化/残差race并回局部winner，不需要传完整词表。高WAN RTT可有收益但数据驻留、真实拓扑、FP64数值、batch与通信均未测；不能以弱全量FP32 607744B为唯一对照。K个独立race需额外RNG/比较，按朴素实现KV次；普通AR+FlashSampling或DISD同U只需一个race，点质量draft+FlashSpec更强。若训练和推理不共用同前缀/hidden，无法复用计算。实际训练显存与反向成本须另计。

## 5. 文献（2026-11-24）
搜索 cut cross entropy; fused LM-head sampling; speculative sampling residual single pass; shared randomness coupling; vocabulary parallel exact softmax; model config. 核对原始页面/摘要及可读正文（FlashSampling HTML直接失败后abs正文恢复），非穷尽全文/附录。
- [CCE ICLR2025](https://arxiv.org/abs/2411.09009)：输出层训练融合及梯度过滤，实测见上。
- [FlashSampling 2026](https://arxiv.org/abs/2603.15854)：词表tile Gumbel-max及TP精确摘要，最直接前向归约先例。
- [DISD 2024/2025](https://arxiv.org/abs/2408.07978)：共享Gumbel、无通信耦合、固定随机种子下drafter-invariant；无需显式归一化/残差，是强对照。
- [SIMPLE 2025](https://arxiv.org/abs/2512.00719)：决策面分离、hot-vocab拒绝校正、生产采样。
- [ERSS ACL2025](https://aclanthology.org/2025.findings-acl.936/)：指数竞赛SD与信息论。
- [FlashSpec Part I 2026-05-14](https://jialefu.github.io/blog/2026/flashspec/)：确定性draft单遍验证/补采（作者博客非同行评审）。
- [Online normalizer 2018](https://arxiv.org/abs/1805.02867)：在线softmax归约；[Megatron-LM 2019](https://arxiv.org/abs/1909.08053)：张量/词表并行训练；[Leviathan ICML2023](https://proceedings.mlr.press/v202/leviathan23a.html)：标准精确SD；[FlashInfer batch-invariant log_probs](https://docs.flashinfer.ai/generated/flashinfer.sampling.log_probs.html)：训练重评分/推理logprob同位数值流的工程近邻。
盲区：随机摘要通信复杂度、加权reservoir/多样本Gumbel最新GPU实现、任意q流式残差算法、未公开/未索引工作及完整附录。无正式文字查重。与HCA/R04直接重合预采算法，后者几何包络不同；本轮不主张新SD原语或两个独立创新。

## 6. 裁决与下一步
CRITICAL：普通TP/同信息平坦可完全复制，普通HFL不一定分片或迟到q；CCE训练、普通AR/SD及DISD+FlashSampling、FlashSpec、双阶段残差、HPRS、预采样等强对照已存在；HFL未改变输出归约数学，SD未反向改变正式学习规则，只能尝试共享状态。随机缓存额外成本可能超过收益，训练梯度并未被省去。公开数据非HFL时间；无真实瓶颈份额和可复用上下文命中率。主观分项问题7×25%、机制5×25%、理论6.5×20%、价值6.5×15%、距离4×15%=5.875/10（约5.9），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（限定单步及数值归约，实际greedy/随机序列/top-p/有限精度未验）；G4 FAIL（无真实两尺度）；G5未验证；G6未验证（无80%依据）；G7局部脚本、原始结果及重跑可复算/完整未过。好结果同正式目标/质量/预算最强公平基线全流程真实wall time−15%或有效工作+20%，三种子三配置两尺度CI、全部G；不估发表概率。
H未执行：只有真实晚绑定q、驻留约束及输出层暴露关键路径份额证实才投入1–2×40GB GPU Qwen2.5-1.5B/3B，8逻辑客户端2区域，公开数学/代码/通用数据，三种子、三非IID/时延。固定目标及版本，采≥10000实际(p,q,hidden-hash,version)行、拒绝及缓存复用记录，torch/CUDA profiler测LM-head/normalizer/采样/反向/通信/重算真实份额，计全流程及CI。强基线普通HFL+CCE+AR/SD、DISD、FlashSampling、FlashSpec适用模式、HPRS、两阶段、相同信息平坦TP（同计算委托和物理拓扑）。不建议投入GPU挽救当前方向。下一轮若可获取真实模型或公开原始执行trace，预先登记具体瓶颈和强对照，且必须证明正式学习更新和自投机共享不可替代操作（非普通输出层kernel）。若无法取得真实证据，继续标记未验证，不再用toy扩大数量或外部GPU结果替代。
