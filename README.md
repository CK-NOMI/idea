# HFL × 免独立草稿模型 SD｜共享科研知识库

> 目标：面向高水平期刊，创造 Hierarchical Federated Learning (HFL) 与 draft-model-free / self-speculative decoding (SD) 的机制级、架构级或学习范式级创新。
>
> 本仓库是多研究团队（包括不同 ChatGPT 对话、Codex、其他研究 Agent）的共同科研记忆。**正结果、负结果、数学反例、已否决 Idea、未验证假说全部保留。** 不将这里的内容视为已发表成果。
>
> 初始归档：2026-11-24，本研究对话前期探索及第 1–13 轮。当前裁决：**尚无候选通过全部 G1–G7，不存在已经被证实的最终 Idea。**

## 0. 所有研究团队必须遵守的协作协议

1. **每轮研究开始前，先读取本仓库最新默认分支的 README.md 全文，包括其他团队追加的内容。** 记录读取的 commit SHA 或时间；不能仅凭各自聊天记忆继续。
2. 对照失败库、数学限制、文献库，明确拟研究的问题与已有候选的区别。不能换名字后重新提交已否决 Idea；若要重启，必须给出改变核心前提的新证据。
3. 按 [Supervisor-Skills](https://github.com/HKUSTDial/Supervisor-Skills) 执行：先第一性原理创造，再致命缺陷审查、最近文献查重、强基线、理论证明、实际运行实验。核心命题被强基线击败时标记 CRITICAL / Reject and Pivot，不用综合评分掩盖。
4. **每轮结束后，无论成功或失败，均向 README 追加记录**。阅读最新版本再修改，保留他人记录；多人并发时用最新 SHA 更新或 PR 合并，不覆盖他人的内容、不强推。可以在其他文件存放代码、原始数据和详细证明，但 README 必须保留可检索摘要与链接。
5. 区分证明、实际运行、模型代理、假设、待复现。历史报告中的实验数字只有在原始脚本、输出和环境可审核后才可升级为已独立复现。不得虚构实验、引文、成功率或 GPU 结果。
6. 每轮必须汇报 G1–G7，不能因多次失败降低标准。创新度不足 8/10 或没有充分证据支持好结果主观把握达到 80%，继续创造/否决，不宣布通过。
7. 任何团队都可质疑、修正本记录；应新增更正及证据，保留历史，不静默改写失败记录。不同团队使用唯一 ID，例如 team-02/R01。未公开的敏感原始数据、密钥、个人信息不要上传（本仓库为公开仓库）。

### 证据等级

- **T**：在明确假设下的数学结论；需要同行复核推导，不自动意味着新定理。
- **E-H**：历史对话报告曾执行数值或小模型实验；本次归档没有重新执行全部实验，也没有逐一审核原始日志。数字按历史报告转录。
- **P**：代理指标（乘加量、验证节点/调用数、方差×成本、理想依赖深度等），不可当成实际 wall-clock 时间。
- **O**：Oracle/完全信息/理论最优结果，不是可部署算法。
- **H**：未验证假说或研究方向。
- **R**：未来必须附脚本、原始日志、环境、种子和可重现命令，才能标记独立复现；目前此归档没有新增 R 级证据。

以下文献名称、日期和链接来自历史对话检索记录；未在本次上传时逐条重新核验。引用投稿前必须打开原论文核对题名、日期、内容和重合点；不可将历史记录等同于已确认文献事实。早期实验包的文件名仅是追踪线索，不是已上传文件链接。

## 1. 固定研究背景与验收标准

### 1.1 基本数学对象

- HFL：客户端持有非 IID 数据，边缘进行区域聚合，云端进行全局聚合。实际优势必须来自数据驻留、边缘本地算力、异步时限、网络成本等约束，不能来自层级名称。
- SD：共享目标模型 p，使用模型自身浅层、辅助头或其他非独立 LLM 的轻量起草路径 q；对候选进行精确验证。单步最大耦合接受率 A=Σ_x min(p_x,q_x)=1−TV(p,q)，拒绝按归一化正残差 (p−q)_+ 修正。多步必须保证目标模型的条件分布与整个序列分布正确。
- 目标模型必须明确。原始预训练 p 与新定义的混合模型、核注意力、Markov 模型等不是同一个对象；相对于新目标精确，不能宣称无损替换原目标。
- 原始多模态 HFL 分类任务与自回归 LLM/RL 场景不同。若改为联邦语言建模或策略学习，需明确说明，不得把合成分类成果当作 LLM 证据。
- 资源目标：1–2 张 40GB GPU 内可复现。

### 1.2 G1–G7（全部通过才可立项）

| 门槛 | 标准 |
|---|---|
| G1 原创性 | ≥8/10；新计算对象、原语及机制；查至少 5 篇最近、最相近原始研究；不能是模块拼接 |
| G2 双向必要性 | HFL 学习改变 SD 的实际计算，SD 反过来改变 HFL 的有效学习或计算；因果消融，质量/算力/通信公平；对比普通 HFL+SD、强平坦联邦（允许同等区域委托计算）及至少两项最近相关方法 |
| G3 数学正确性 | 完整定义，至少一个有实质内容的有效定理；对同一个明确的正式目标，证明 greedy token 一致和随机采样序列分布一致（适用温度、top-p、有限精度条件明确） |
| G4 实验 | 两种预训练 Transformer 尺度、至少三种非 IID/延迟设置、至少三个独立种子、公开真实任务、强消融 |
| G5 端到端 | 匹配质量及预算，真实总 wall time 至少减少 15%，或同成本有效工作至少增加 20%；含训练、推理、通信、验证、额外学习与失败回退，报告 CI。15% 时间减少对应 ≥1/0.85=1.17647× 加速，不是 1.15× |
| G6 成功把握 | 有证据支持取得强结果的主观把握 ≥80%，不是 token 接受率，也非发表概率；没有证据不得给出虚假概率 |
| G7 复现 | 完整机制、数学、伪代码、源代码、原始指标、负结果、环境、预算和公平基线；1–2×40GB GPU 可复现 |

### 1.3 不可忽略的跨轮限制

- 同步线性 HFL 的加权平均等于平坦 FedAvg。若给平坦系统完全相同的边缘计算、信息、通信和协议，它可以模拟 HFL。不能声称层级本身有数学优势。
- SD 精确性不等于训练数据无偏；单 token 边际正确不等于多 token 联合分布正确；高接受率不等于梯度准确或真实速度高。
- 旧版本 KV/隐藏状态不能在一般 Transformer 参数更新后盲目复用；有限精度下“实数算术等价”不自动保证 greedy 边界或随机数路径逐位相同。
- 训练代理指标与真实达到相同质量的累计时间不同。需要考虑训练/推理占比；Amdahl 上限：若生成仅占 15%，无限加速也仅能节省 15% 总时间（不计开销）；若占 16%，仅加速生成要达到 15% 总时间下降，需要约 16× 生成加速。
- 不允许弱基线：必须包括预缓存、普通分布式/平坦委托计算、直接重算、最优抽样、teacher forcing、普通早停等适用替代方法。

## 2. 早期探索与否决库（第 1 轮之前及第 1 轮）

以下记录来自本对话前期研究摘要和历史报告，编号用于去重，不要求名称原封不动复用。

| ID / 候选 | 原始命题 | 失败证据与保留结论 |
|---|---|---|
| PRE-01 PACT-HFL | 浅层/深层一致性保持 | 五种子 toy 一致性原本约 99.8–100%，问题没有被证实 |
| PRE-02 多步隐算子聚合 | 聚合 latent operator 再递推 | 平均与矩阵幂不交换；普通多 token head/个性化可匹配 |
| PRE-03 V-CAP-HFL | 最大耦合旧/新版本，减少配对评估方差 | shared Gumbel 基线匹配，缺 SD 特有因果收益 |
| PRE-04 CFV-HFL | 利用被拒绝后缀 Fisher/KL 约束联邦更新 | 平均接受率约 0.541，对照约 0.695；否决 |
| PRE-05 Predictive Update Subspace Contract | 约束未来低秩更新子空间 | 无精确计算收益，学习能力风险 |
| PRE-06 区域草稿混合抵消误差 | 合并草稿概率、MAP/TV median | 70 组 MAP 约 80.98%，median 约 81.21%，85.7% 配置等价；普通 ensemble 强基线 |
| PRE-07 FMSC-HFL | 对未来模型版本取公共概率质量 ℓ(x)=inf_j p_j(x) | 精确公共质量采样需版本与随机性独立；界保守，旧版本异步服务不用等待；合成渐进释放有效但非新；32 分支树仅约 6.49 位置，旧模型约 7.21 |
| PRE-08 Acceptance-Sufficient HFL | 根据预计 SD 接受率挑选区域信息 | 普通信息价值理论、同信息平坦委托匹配，开销和不确定性未解决 |
| PRE-09 Federated Certifiable Computation | 共享 logits + 区域残差，Gumbel-Max/界跳过区域 | 新目标而非未修改 pretrained 模型；6 区域平均访问 1.70，但区域残差仅占 10% 时总收益约 7.2%；与 early exit/MoE/A* sampling 重合 |
| PRE-10 FCQS | 将序列分解为共享类别轨迹与区域条件 token | 表达能力 KL 下界含条件互信息；真实语言独立性不可信，乐观收益仅约 8%；与变分并行语言模型重合 |
| PRE-11 SPEAR（其他历史团队记录） | 训练激活及 reference score 复用 | Pythia-70M 冻结主干可复用头梯度；早层可训练则失效；100 OpenR1 配对 trace 完全重复 reference score 仅约 0.114%，普通前缀复用节约约 0.95% forward positions |

### R01：跨版本重算与联邦随机传输

**机制与结论（T / E-H / P）**

- 四层随机因果 Transformer，冻结前三层、更新最后一层低秩投影。旧下层状态 + 新上层与完整重算一致（logit 误差 0），但旧输出误差约 0.047–0.077；更新早层会导致状态误差约 0.058–0.105。CPU 微基准 1.103ms → 0.268ms，未计缓存开销。普通冻结层缓存等价，否决。
- 对稀有 token 混合分布 p=(1−τ)q+τδ_rare，τ=0.1 时普通拒绝采样接受约 0.000313，最大耦合约 0.900003。直接混合采样即精确，不是 HFL 新机制。
- **Federated Stochastic Transport**：区域连续时间 Markov 生成元 G_r，定义新目标 p=μ exp(ΣG_r)。均匀化：λ_r≥max_i(−G_{r,ii})，Λ=Σλ_r，P_r=I+G_r/λ_r；N~Poisson(Λ)，每次按 λ_r/Λ 选区域并转移。五种子、16 token、四区域、每种子 90,000 次样本；MC TV≈0.00522，简单顺序组合约 0.07431，混合约 0.14507。平均 Λ≈2.569，零事件率约 7.69%。计算仍需多次串行 RPC，未产生多 token SD 收益。
- CTMC 保留率与最大耦合比较（不同规模）：83.14 vs 94.53%、63.41 vs 88.19%、41.20 vs 81.04%、19.44 vs 73.91%。稀疏传输不自动最优。
- Brier 损失路径 score 梯度估计无偏，但三种子每组 140k 路径对仍有较大统计代价（max z 约 1.54、1.92、1.81）。
- 单向供体→受体 CTMC 可实现经典最大耦合：供体速率 −log(π_i/μ_i)，按剩余质量运输给接收态。30 种子，分布误差 1.8e−16，最大耦合差 1.1e−16。需要预知完整 π，属于已有耦合数学，否决。

历史原始包线索：HFL_SD_20261124_research_bundle.zip、HFL_SD_20261124_audit.md；脚本名 hfl_sd_rebase_probe.py、hfl_sd_distribution_probe.py、hfl_sd_markov_generator_probe.py、hfl_sd_transport_training_probe.py、hfl_sd_optimal_transport_probe.py。当前 README 未附这些历史二进制文件。

## 3. 第 2–13 轮详细科研日志（本对话团队 A）

### A/R02｜离散联邦更新码与连续更新修正

**问题：** 是否能让联邦参数更新本身“先猜后验”，同时加速训练和生成？

**T：** 两个不相等的确定性连续更新对应不同点质量，TV=1，最大精确接受率 0。相同各向同性协方差的高斯更新最大耦合率为 2Φ(−||μ−μ̂||/(2σ))。固定码本 S=span{v_i}，强凸二次目标的最优差距至少 λ_min(H)/2 × dist²(θ*−θ_0,S)。

**E-H / P：**
- 五种子，24 维、五区域、提前见两个区域：8/16/32/64 码预测命中率 98.57/96.97/96.30/94.40%；上轮码基线 84.83/78.47/72.97/65.73%；更新失真 50.94/39.01/31.02/26.08%。平坦同信息对照完全匹配。
- 96 维正交漂移压力测试，400 轮后完整梯度约零损失，8–64 固定码损失仍 0.49–0.51。
- 高斯耦合三种子各 25k：4/16/64 维实测 61.61/31.87/4.50%，理论 61.71/31.73/4.55%。64 维增加噪声 0.2/0.4/0.8，损失约 0.0711/0.2844/1.1375。
- 无偏跨版本梯度修正：g(θ̂+r)=g(θ̂)+∫₀¹ H(θ̂+sr)r ds；随机 U 构造 HVP。4096 样本/128 维逻辑回归，三种子平均误差约 0.001–0.005%，单次修正成本约直接精确梯度的 1.44×。

**裁决：Reject and Pivot。** 离散码损害学习表达力，连续修正比重算贵，平坦可替代。保留连续更新障碍和子空间下界。
**近邻历史记录：** Leap+Verify、FedCode、FedSteer、HFL 多时间尺度梯度修正、When Drafts Evolve、Aurora。
**历史包线索：** HFL_SD_20261124_round2_bundle.zip、HFL_SD_20261124_round2_audit.md。

### A/R03｜跨版本状态传输、Taylor 缓存与零空间更新

**T：**
- 递归 s_n(λ)=Σ_{t=0}^{n−1} λ^{n−1−t}x_t。若要对所有新 λ 和任意 n 维实数历史精确恢复，连续摘要至少 n 维（Vandermonde 可逆 + 连续注入维度限制）。只适用于该递归模型和连续摘要假设。
- 四阶 Taylor 状态导数缓存可在小参数漂移下给出余项证书，但误差会随记忆长度放大。
- 对所有受保护激活 H_l 要求 H_lΔW_lᵀ=0，并冻结其他改变计算的路径，可归纳保证受保护前缀状态不变。相同零空间的线性聚合闭合；满列秩时 ΔW_l=0。

**E-H / P：**
- 20 种子，2048 token，λ=.95，δ=.002/.01/.02：四阶认证覆盖 100/85/0%；λ=.99，δ=.0006/.003/.006：100/0/0%。简单截断历史 256 或 1024 token 的强对照在四个困难设置中为 100/100/95/50%。Taylor 查询微基准约 220×，不含缓存构建。
- 2 层 64 维随机 Transformer，三种子，logits 最大误差约 1.43e−6。保护 8/16/32/48/64/96/128/256 token，剩余自由度约 88.28/76.56/53.13/29.69/7.57/3.21/.016/0%。
- 保护 16/64/128/256 token，剩余梯度范数 82.5/16.8/约0/0%；投影更新损失下降 .1874/.0079/0/0，普通更新约 .275。多区域约束联合秩可快速填满空间。

**裁决：Reject and Pivot。** 旧状态压缩的精确性、训练可塑性和费用冲突；简单截断重放在强漂移下更有效。
**近邻历史记录：** ReplaySSM、CacheReforge、NESS、Adam-NSCL、PipeSD。
**包线索：** HFL_SD_20261124_round3_bundle.zip、HFL_SD_20261124_round3_audit.md；hfl_sd_jet_state_probe.py、hfl_sd_truncated_replay_control.py。

### A/R04｜Federated Coupled Speculation (FCS)：联合 rollout 相关性

**问题：** 在各区域边际生成分布不变时，耦合采样能否同时减小联邦梯度方差与共享验证树成本？

**T：** 联合 Γ∈Π(P1,P2)，线性无偏 REINFORCE 梯度不变，方差含跨区域协方差。二元联合参数 t∈[max(0,a+b−1),min(a,b)]，单步线性目标在端点达到。联合接受 min(1,P/Q) 加正残差恢复联合目标。注意只有相同模型/兼容前缀才可复用 KV。

**E-H / P / O：**
- 6 步二元、64 完整轨迹、每奖励类型 20 种子。理论最优联合策略相对最强简单策略（独立、共享、反向、64 种逐层组合）改善约 9.1/0/11.2/8.0%；历史样本学习仅 +0.5/−12.9/+3.9/−0.9%。
- 完整 6 步 rollout，80 配置、两档误差、每配置方法 5000 次。中低误差下理论最优代理成本改善 14.3/18.6/11.1%，学习策略 −6.9/+0.8/−5.6%。公平基线同享前缀缓存，计被拒绝分支。
- **GRPO 相关采样反例（T）：** X~Bernoulli(.5)，R=X，真实梯度 .25；两样本未标准化 leave-one-out 估计 (X1−X2)(X1−.5)，独立期望 .25，共享 0，反向 .5。边际正确不保证组内训练目标不变。
- 固定独立目标联合分布，仅耦合草稿：共享比例 0/25/50/75/100%，相对验证节点 100.3/99.8/99.9/100.3/100.4%；验证块 2.254→2.863。

**裁决：Reject and Pivot。** Oracle 收益未转化；联合拒绝可能同时截断正确轨迹；改变联合分布会影响组内训练。
**近邻：** Arithmetic Sampling、Quasi-random Multi-Sample Inference、SpecTr、UniVer、EfficientRollout、SpecRoll、SRT。
**包线索：** HFL_SD_20261124_round4_bundle.zip、HFL_SD_20261124_round4_audit.md、hfl_sd_coupling_probe.py。

### A/R05｜被拒绝草稿的梯度价值、合并反向、梯度引导起草

**T：** X~Q 草稿、Y~P 正式轨迹，f_r(x)=(R−b)∇log P(x)，w=P/Q。Z=(1−a(Y))f(Y)+a(X)w(X)f(X) 无偏（支持集覆盖、权重预先确定）；有界 a=ψ/(1+w) 防长尾。共同快照/轨迹时 Σβ_r∇L_r=∇Σβ_rL_r，先合并系数再反传。忽略 SD 成本时最优行为分布 q*∝P√(Σβ_r||f_r||²)，是已有最小方差重要性采样。

**E-H / P / O：**
- 12 种子、3 奖励×3 误差×4 区域=432 配置；32/256 历史样本扩展 864 配置。256 样本平均成本改善：经典平衡 MIS 6.58%，独立区域 6.48%，区域汇总 7.65%，区域+全局 7.41%，oracle 9.52%；32 样本联合 5.37%。附加收益仅约 0.8–1.1 个百分点。
- 两层约 39952 参数 Transformer CPU，16 区域逐区域反传 33.81ms，合并后 3.28ms，强平坦合并同 3.28ms，梯度最大误差 6.7e−16。否决独有性。
- **保留正信号：** 梯度引导草稿 Oracle 成本—方差代理改善 40.33/43.76/46.34%；用相同 10 参数自回归草稿、每区 256 历史轨迹，普通拟合改善 4.61/11.47/15.51%，奖励引导 18.74/22.48/23.87%，廉价梯度引导 22.53/25.94/26.69%，完整梯度 25.10/28.43/29.14%。仍是 P 级结果。
- 加入独立重要性采样：串行验证开销为 0/1/3/6 单位时，梯度引导 SD 改善 25.05/28.16/32.33/36.04%，独立采样 35.41/31.27/25.04/18.73%。优势有条件，非 HFL 独有。

**裁决：** 自适应 MIS 与合并反传否决；梯度引导采样保留为证据支持的条件性方向，但未立项，因 active importance sampling 已有先例、同信息平坦可替代、无 GPU 净收益。
**近邻：** Policy Gradient with Active Importance Sampling、Variational Speculative Decoding、VAT、Draft-OPD、Reuse your FLOPs、AReaL-DTA、Stable Asynchrony。

### A/R06｜带符号分支价值、FRCS 残差耦合、条件积分

**T：** 局部价值 Σβ_r||d_r||² 不等于全局 ||Σβ_rd_r||²；仅范数无法区分同向/抵消分支。等概率两分支各采一次，较两次 iid 的方差迹下降 ||μ1−μ0||²/8。线性摘要 A d 可先聚合再估计全局值。

**E-H / P：** 16 客户端、4 边缘、256 轨迹、16 分支选四、40 种子/类。区域抵消时局部范数找到最优收益 53.99%，带符号摘要 94.57%，8-bit 平坦同 94.57%。计额外节点成本后分层策略净改善 −.15/−.09/−.19/+.75%，不足。

**FRCS（T）：** 最大耦合对角 m_x=min(p_x,q_x)，残差行 a=(q−p)_+、列 b=(p−q)_+；任意非负运输矩阵 M 满足对应边际，Γ=diag(m)+M，保留目标分布和接受率。让 M 最大化廉价梯度 L 与正式梯度 H 的协方差，使用多保真控制变量 H̄+α(L̄_extra−L̄_paired)。须因果地选 M，不偷看未来草稿随机性。
- 单 token 48 词、16 参数、4 区、40 种子/误差：标准残差+控制变量改善 37.50/14.94/4.37%，区域学习 39.80/17.51/6.40%，OT oracle 42.68/20.93/8.76%。
- 三步、6 词、216 完整轨迹，20 种子：标准 19.69/1.94/0%，oracle 20.57/2.38/.005%。新增贡献 <1 个百分点。
- 条件期望强基线（T）：Z=H(Y)−α(L(X)−μ_L)，Var(E[Z|Y])≤Var(Z)。标准 SD 中 E[L(X)|Y=y]=(m_y L(y)+b_y μ_a)/p_y，其中 μ_a=Σa_x L(x)/δ。
- 120 个单 token 配置（完全信息 oracle）：普通控制变量方差下降 75.71/49.98/28.78%，条件积分 96.38/82.23/59.72%。完全已知奖励时直接精确积分也是强对照。

**裁决：Reject and Pivot。** 平坦摘要匹配，残差耦合多步优势消失；已有多保真、平衡抽样和 OT 基础。
**近邻：** EPIG-Tree、Multi-Fidelity Policy Gradients、SpecTr、ResiSpec、EfficientRollout、SpecRoll。

### A/R07｜区域可组合记忆、低秩接口与分层概率所有权

**T：** 即使 QKᵀ 秩为 1，exp(QKᵀ) 也可能为满秩 Vandermonde，softmax 行归一化不改变秩。核线性注意力 M_r=Σφ(k)vᵀ、z_r=Σφ(k)，Q=UB 时投影聚合可精确恢复 O=(QM)/(Qz)，但只针对这种目标架构。
- 64 维小 Transformer、5 种子、62 候选 query 均秩 62；截断 16 维保留能量 96.14%，输出误差 1.641%。两棵树累计秩 64，最小二乘恢复聚合记忆误差约 2.87e−14。
- d=v=64 时精确 62 维双向查询传 7998 浮点数，全量记忆仅 4160；仅传查询不等于天然隐私。
- 四区域回归、5 种子，统一 8 维接口 vs 各区独立 8 维接口：异质性 η=0/.25/1/2，统一误差 .629/5.059/33.806/42.12%，独立 .688/3.001/9.352/6.309%，全维 .772/.719/.372/.148%。
- 分层概率 p(x)=p(c)p(x|c)，先类别后 token 最大耦合：A_hier≤A_full（min(a,b)min(u,v)≤min(au,bv)）。16 类×32 token、5 种子×100 分布，低/中/高联合误差完整接受 89.36/73.94/51.63%，分层 85.56/66.37/41.81%。普通分层 softmax 本来也仅访问最终一类。

**裁决：Reject and Pivot / 概率分层暂不立项。** 高秩、通信、学习质量与接受率冲突。近邻：Transformers are RNNs、RingAttention、SpecLA、BasisKV、跨机构 RAG、Lossless Hierarchical SD、Adaptive Softmax。

### A/R08｜Federated Certified Affine Compilation (FCAC)

**T：** ReLU FFN V ReLU(Wx+b)。若 ||x−c||≤ρ 且 |w_iᵀc+b_i|>ρ||w_i||，神经元符号稳定。稳定激活集合 S 合成 A=V_S W_S、a=V_S b_S；不确定集合 U 正式执行，F(x)=Ax+a+V_U ReLU(W_Ux+b_U)。同模型快照、同 S，D_r=Σ_i g_i x_iᵀ 可重建 ∂V_S=D_rW_Sᵀ、∂W_S=V_SᵀD_r；Aᵀg 复用输入梯度。
- 5 种子合成 FFN，扰动 .005/.02/.05/.1/.2/.5 时，前向乘加量为 dense 的 15.58/25.28/44.60/71.14/98.90/100%；普通精确激活稀疏约 75%。
- 2 层、32 hidden、128 FFN、16 词、4 边缘×2 客户端、35 HFL 轮、3 种子真实小模型训练。区域编译成本 56.44%，全局 token-position 中心 54.93%，稀疏 74.74%。区域独立价值被全局击败。
- 重建梯度最大误差 1.83e−13。计编译与梯度恢复，batch 32 在低/中/高扰动分别 109.3/112.5/116.6% 成本；batch128 55.6/74.3/96.5%；batch512 41.5/61.9/90.2%。
- 实际训练共享组：区域组中位数 8，计算增加 54.4%、通信代理增加 24.5%；全局组约 32，计算降 0.6%、通信增 28.5%。自动回退区域全选普通计算，全局理论节约 3.7%。

**裁决：Reject and Pivot。** 需大量同激活结构样本才能摊薄开销；区域拆分反而损害共享。ReLU 分段仿射、activation sparsity、可验证计算已有先例；现代 SwiGLU 等不直接适用。
**近邻：** Slalom、Lagrange Coded Computing、zkLLM、VIA-SD、Spark Transformer、Sparsing Law、Vegas、2:4 activation sparsity。

### A/R09｜推测反向传播与分层联合修正

**T：** 廉价梯度 g_L 和完整梯度 g_H，Bernoulli B~s>0，ĝ=g_L+B(g_H−g_L)/s 无偏。SD 正式 token y 来自拒绝的条件概率 r(y)=[p(y)−q(y)]_+/p(y)，拒绝优先修正概率 s=r+η(1−r)。
- **接受率无法控制梯度差反例：** z_L=a+θu、z_H=a+θv，在 θ=0 时 p=q、接受率 100%，但 score 梯度差可任意大。
- 保持残差前向 h'=h+F(h) 不变，只随机修正反向：ĝ_h=g+B J_Fᵀg/s；需合法条件采样确保无偏。四区域各边际 1/2 的互补两组修正，额外条件方差 ||D1+D2−D3−D4||²/16（等权）。

**E-H / P：**
- 4992 参数小 Transformer，3 种子×3 异质性、4 边缘×2 客户端、25 轮。25%/50% 深层修正预算额外方差：均匀 70.23/23.41%，拒绝优先 74.26/23.53%，概率差 184.83/92.52%，奖励 41.26/9.64%，浅层梯度 37.86/8.49%，oracle 35.08/7.62%。
- 恒等残差代理均匀修正 54.35/18.12%，廉价梯度引导 28.93/6.46%，oracle 26.56/5.71%。CPU batch32/128/512 完整耗时 1.458/2.805/7.607 ms，25% 修正 1.681/2.929/7.375 ms；最好仅省约 3.96%，其余更慢。
- 1800 四区域组合，固定大小抽样基准 100；8/32/128 维摘要额外方差指数 96.39/94.06/91.73，完整廉价梯度 88.27，oracle 85.24。普通平坦也能实现，且不需要 SD。

**裁决：Reject and Pivot。** 核心拒绝信号被随机基线击败；反向稀疏未获净速度；联合修正属于已有平衡抽样。
**近邻：** LayerSkip、Recurrent Early Exits for FL、Multilevel MC Unbiased Gradient、Unbiased Approximate VJP、Cube Method。

### A/R10｜Federated Random Dependency Decomposition (FRDD)

**T：** m(x)=inf_h p(x|h)，c=Σm；p=c a+(1−c)b_h，精确公共质量允许历史无关采样。进一步 p(x|h)=Σ_{S⊆[k]}w_S p_S(x|h_S)，以非负子概率变量和线性规划最小化 D(p)=Σw_S|S|。D 对凸概率混合是凸函数，D(Σβp_r)≤ΣβD(p_r)。
- 四阶奇偶正负区域 p1=.5+.3Πh_i、p2=.5−.3Πh_i，各区 D≥2.4，聚合 D=0。聚合的是概率，不是 FedAvg 参数。
- 生成某 token 不依赖历史，不代表未来全部脱离过去；长期上下文依赖是瓶颈。

**E-H / P / O：**
- 5 种子×3000 条长度128 DAG，独立概率 30%，历史窗口 1/2/4/8 的理想串行深度 11.38/28.89/76.24/89.46；50% 时 6.35/12.27/31.31/60.68。491520 token 串行与并行同随机数零不一致（有限状态模型）。
- 二元四阶历史 LP，3 类×10 种子，1500 序列/策略。独立/全历史、可变窗口、任意子集深度：稀疏 88.23/43.45/12.06；高阶交互 102.92/89.25/64.80；随机 100.79/91.42/89.67。
- 四区域两边缘人造互补，真实概率时客户端先分解 82.54，边缘先聚合 3.62，平坦全局最优也 3.62。每历史64/256/1024 样本，边缘 5.71/4.70/4.09，平坦 4.89/4.43/3.94。
- 八词对称预测模型，最大历史无关质量 22.86%；强制 30/50/70/90% 时最小额外 KL .0107/.1252/.3561/.7864 nat/token。

**裁决：Reject and Pivot（保留问题）。** 精确依赖抵消正信号，但人为互补、平坦匹配、指数分解复杂度、未实现真正 SD/Transformer。
**近邻：** MTD、Sparse Markov Models、Parallel Sampling via Counting/Autospeculation、Any-Subset AR、Domino、FACT。

### A/R11｜Federated Subset Verification (FSV)：区域随机子集与渐进验证

**T：** 对新定义混合目标 p=Σβ_r p_r，先按权重选择验证组 g，再对 p_g 精确 SD，输出边际为 p。A_single≤A_group≤A_full（TV 凸性）。多步需保持每位置正确条件路由。
- 渐进验证：L_s(x)=Σ_{r≤s}β_r p_r(x)，若 L_s(x)≥u q(x) 可提前接受；否则全量计算并标准残差补采。固定顺序期望区域次数 Σ_{s=0}^{R−1}Σ_x[q(x)−L_s(x)]_+。
- 固定混合权重，R(Y)∇log p_g(Y) 对混合策略梯度无偏，但普通混合采样也能获得标签。
- **整块路由反例：** 两区域 (.9,.1)、(.1,.9)，逐位置独立路由联合四项 .25；整块共享区域联合 [.41,.09;.09,.41]，TV=.32。单 token 边际正确不保证完整轨迹。

**E-H / P：**
- 32词/历史、8区域、4组、每历史192本地样本、30种子×3。边缘内互补接受率：单区73.12%、边缘84.77%、全局配对84.77%、全量87.37%；跨边缘互补73.07/79.42/84.83/87.41%；独立66.33/73.93/73.93/83.66%。
- 每块有效输出：边缘内 2.943/3.689/3.689/3.882；跨边缘 2.941/3.324/3.691/3.884；独立 2.589/2.988/2.998/3.606。30万随机块与 DP 相容。
- 区域头成本为共享主干 2/10/30/100% 时，双区域相对普通随机区域生成成本约 .502/.634/.891/1.387，单区域 .602/.664/.786/1.020。代理、无网络成本，直接缓存完整表为更强有限状态基线。
- 渐进访问平均 4.6–4.7/8，学习访问顺序几乎无收益。单组梯度方差额外约1.65–3.36%，双组更低，但普通混合采样亦可。

**裁决：Reject and Pivot。** 平坦配对匹配或更好；单区域普通精确采样、预缓存、并行访问都必须对比；仅证明随机采样，不保证混合 greedy。
**近邻：** FedEM、Federated EM、SpecFed、Lossless Hierarchical SD、MoE SD、DICE、FedCD。

### A/R12｜Verification-Induced Selection Correction (VISC)

**核心问题（保留）：** SD 的单轨迹精确性不能防止 HFL 截止时间引起的训练选择偏差。

**T：** 轨迹 Y~P，完成事件 I=1{C≤D}，π(y)=P(I=1|Y=y)，入选分布 P_adm(y)=P(y)π(y)/Eπ。单步给定正式输出的接受概率 a(y)=min(p(y),q(y))/p(y)。固定起草 K、固定长度/快照时，通过接受路径动态规划计算 π_D(y)。IPW I g(Y)/π(Y) 无偏需 π>0，二阶矩 E||g||²/π。
- 随机补审概率 ε，π_ε=π_D+ε(1−π_D)，Rao–Blackwell 权重优于普通补审。
- 独立接受构造 m(y)=c p(y)，c≤inf_y q(y)/p(y)，拒绝直接从 p 补采，A 与 Y 独立；要使完整轨迹与验证次数独立，c_t 必须预先固定且对所有合法历史有效。属于经典拒绝采样相关理论，最坏 token 概率比严重限制接受率。

**E-H / P：**
- 10步二元、1024轨迹、4区域、10种子×3误差=120配置，K=3，预算4。低/中/高误差：完成99.68/94.54/80.32%，梯度偏差 .29/3.97/10.66%。额外 200k 轨迹模拟与解析相容。
- 完成所有轨迹成本指数1；解析校正相同梯度精度成本1.01/1.2322/4.3606。高误差截止预算3/4/5/6，成本代理140.03/4.36/1.38/1.05×。
- 15% 随机补审：普通1.06/1.87/3.40×；解析1.01/1.18/2.01×；历史前缀预测1.05/1.62/2.45×；oracle1.01/1.18/1.62×，全完成1×。
- 独立接受完成率87.90/28.30/2.44%，平均完整验证次数增17.1/54.8/88.8%，截止偏差为0。

**裁决：Reject and Pivot。** IPW 修正有方差代价、补审超时带来陈旧性，独立接受严重牺牲生成效率。**任何后续截止式 HFL 必须审计轨迹参与偏差及 π=0 的不可恢复问题。**
**近邻：** Who Trains Matters、Semi-Clairvoyant Scheduling、SortedRL、Gradient-Aware Scheduling、Stable Asynchrony、Pool Aware Effective Staleness Control。

### A/R13｜前缀学习增量、Federated Reward Certification (FRC) 与奖励残差

**T：** 终局奖励 R(Y)、score S_t、固定前缀 H_t，V(H_t)=E[R|H_t]。完整条件梯度 E[R S_T|H_t]=V(H_t)S_t+∇_θ V(H_t)，后项不可忽略。
- 若停止时间 τ 上所有合法后缀的聚合奖励已经确定 C(H_τ)，则 ∇J=E[C(H_τ)S_τ]，方差不超过完整轨迹估计（未来 score 条件均值为零）。仅适用于同模型任务、线性聚合奖励；未来 KL、熵等损失亦须纳入证书；训练专用 rollout 停止不能说成完整生成仍然返回了相同序列。
- 若 R_FL=A(H_t)+εB(Y)，则 ∇J=E[A S_t]+ε E[B S_T]。独立样本成本 c_A,c_B、方差 v_A,v_B（v_B 含 ε²），最优连续样本比 n_A/n_B=√(v_A c_B/(v_B c_A))，最小 CV=(√(c_Av_A)+√(c_Bv_B))²。
- 黑盒后缀奖励认证下界：M 种合法后缀，要确定性证明全部奖励相同，最坏须查询全部 M；遗漏一个后缀即可构造不可区分反例。M 可为 |V|^L。需要结构化奖励程序而非预测置信度。

**E-H / P / O：**
- 10步二元、20参数、20种子×4奖励×4前缀=320。即使价值 oracle 完美，前缀长度4/8时，仅前缀 score 梯度偏差：早期奖励0/0%，最后token约100/99.94%，奇偶78.69/49.49%，混合54.96/54.93%。完整条件梯度误差2.68e−16。
- 四区域终局项人工抵消，20种子×3关系，SD平均验证调用3.5043→1.2166（减少65.28%），梯度方差减少73.22%，代理成本降90.69%；系数只偏5%即使严格提前停止失效，回到3.5043。
- 奖励残差 ε=0/.00875/.05/.1/.3/.7，20种子/档，Oracle 最优预算下 SD 代理成本下降90.69/89.65/84.14/76.35/42.32/3.71%；最后一档17/20种子有利，其余20/20。普通 AR 用相同分解也受益，不能比较各自归一化百分比来声称绝对速度差。未计分解获取、验证与通信。
- 期望分解梯度最大误差1.18e−16。

**裁决：Reject and Pivot。** 数学上有效，但已有奖励分解、条件积分和最优抽样基础；同信息平坦、普通 AR/SD 可完全替代。黑盒证书难获得，真实任务中可提前确定的奖励比例未知。
**近邻：** RUDDER、Learning Reward Machines、Expected Policy Gradients、Federated Natural Policy Gradient、SPIRE、ESPO、Before the Rollout Ends、EfficientRollout、TAILROLL。

## 4. 目前最值得保留的研究资产（不是最终方案）

| 资产 | 证据/局限 |
|---|---|
| SD 截止时间引发轨迹参与偏差 | 有解析式和数值反例；校正成本可能超过直接完成，适合作为所有方案的必检项 |
| 区域概率聚合可消除真实生成依赖 | 有凸性与奇偶反例；目前仅有限状态新目标，平坦匹配 |
| 梯度信息引导草稿不必最大化接受率 | 小模型中有 22–27% 代理改善，独立采样强基线和真实硬件决定收益，尚未形成 HFL 独立优势 |
| 残差耦合保持接受率和正式边际 | 严格数学成立，但多步新机制额外收益 <1pp |
| 训练/验证共享计算代数 | 仿射编译、充分统计量可同时服务前向反向；实际小组摊薄不足，ReLU 限制且全局更优 |
| 奖励确定后停止及奖励残差分配 | 理论无偏、合成代理收益高；奖励结构获取难，普通 AR/SD 可替代 |
| 共享区域梯度方向比单纯范数更有信息 | 可证明分支排序错误；但通信、方差×成本和同信息平坦对照未通过 |

### 4.1 共性失败诊断（必须避免）

1. 创造的统计或代数机制与 SD 无关，普通采样/训练执行器也可用。
2. 同信息、同区域计算能力的平坦联邦能复制全部收益；不能刻意削弱对照。
3. Oracle 完整分布、未来奖励、梯度已知时有效，但真实获取成本太高。
4. 高接受率/局部计算减少被重复反向、通信、重算、方差和质量损失抵消。
5. 为精确复用限制模型更新、历史依赖或表示秩，导致非 IID 学习能力退化。
6. 偷换目标模型/学习目标：对新定义模型精确不代表原模型精确；组内非线性训练目标必须单独证明。
7. 仅 toy/CPU/理想成本，缺预训练 LLM 的 end-to-end 结果。增加更多 toy 配置不能替代 G4/G5。

## 5. 文献导航（历史检索线索，投稿前须逐项复核）

### 自推测/验证/分布式 SD
- [LayerSkip](https://aclanthology.org/2024.acl-long.681/)、MEDUSA、EAGLE/EAGLE-2/EAGLE-3、KnapSpec。
- [SpecTr](https://arxiv.org/abs/2310.15141)、[UniVer](https://arxiv.org/abs/2605.04543)、[ResiSpec](https://arxiv.org/abs/2608.24411)。
- [Lossless Hierarchical Speculative Decoding](https://arxiv.org/abs/2601.05724)、[SpecFed](https://arxiv.org/abs/2604.25777)、[SpecLA](https://arxiv.org/abs/2607.16673)。
- [EfficientRollout](https://arxiv.org/abs/2606.18967)、[SpecRoll](https://arxiv.org/abs/2608.04962)、[SRT](https://arxiv.org/abs/2601.09083)、OnlineSPEC、Aurora、SPEC-RL。
- [Variational Speculative Decoding](https://proceedings.mlr.press/v306/zou26i.html)、[Verification-Aware Training](https://arxiv.org/abs/2608.30135)、[Draft-OPD](https://aclanthology.org/2026.emnlp-main.1139/)、VIA-SD、PipeSD、Vegas。
- [FACT](https://arxiv.org/abs/2611.23463)、[DICE](https://arxiv.org/abs/2611.27891)、[MoE SD](https://arxiv.org/abs/2610.34440)。

### 联邦学习/分布式优化
- [Hierarchical FL with Multi-Timescale Gradient Correction](https://papers.nips.cc/paper_files/paper/2024/hash/8fb96e8d0fbf591b1fa1ad85653d8417-Abstract-Conference.html)。
- [Recurrent Early Exits for FL](https://proceedings.mlr.press/v235/lee24h.html)、[FedEM](https://proceedings.neurips.cc/paper/2021/hash/82599a4ec94aca066873c99b4c741ed8-Abstract.html)、[Federated EM](https://proceedings.mlr.press/v235/tian24e.html)。
- [Federated Natural Policy Gradient and Actor Critic](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html)。
- [Who Trains Matters](https://arxiv.org/abs/2604.26604)、FedCode、FedSteer、NESS、Adam-NSCL。

### 学习估计、树形采样与奖励
- [Arithmetic Sampling](https://proceedings.mlr.press/v202/vilnis23a.html)、[Quasi-random Multi-Sample Inference](https://arxiv.org/abs/2411.06251)。
- [EPIG-Tree](https://arxiv.org/abs/2609.20004)、[Multi-Fidelity Policy Gradients](https://xinjie-liu.github.io/mfpg-rl/)、[MFPG 数据稀缺](https://arxiv.org/abs/2610.02505)。
- [Policy Gradient with Active Importance Sampling](https://rlj.org/papers/papini2024policy)、[Multilevel MC Gradient](https://proceedings.mlr.press/v130/shi21d.html)、[Unbiased Approximate VJP](https://arxiv.org/abs/2602.14701)、[Cube Method](https://academic.oup.com/biomet/article-abstract/91/4/893/459329)。
- [Reuse your FLOPs](https://proceedings.mlr.press/v306/setlur26a.html)、[AReaL-DTA](https://proceedings.mlr.press/v306/zhang26hr.html)、[Stable Asynchrony](https://proceedings.mlr.press/v306/huang26ca.html)。
- [RUDDER](https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html)、[Learning Reward Machines](https://papers.neurips.cc/paper_files/paper/2019/hash/532435c44bec236b471a47a88d63513d-Abstract.html)、[Expected Policy Gradients](https://www.jmlr.org/papers/v21/18-012.html)。
- [SPIRE](https://aclanthology.org/2026.emnlp-main.1613/)、[ESPO](https://arxiv.org/abs/2605.29860)、[Before the Rollout Ends](https://arxiv.org/abs/2609.31995)、[TAILROLL](https://arxiv.org/abs/2611.22206)。
- [Semi-Clairvoyant Scheduling](https://ijcai.org/proceedings/2025/951)、[Gradient-Aware Scheduling](https://proceedings.mlr.press/v306/zhang26bc.html)、[Pool Aware Effective Staleness Control](https://arxiv.org/abs/2609.36830)。

### 计算表示/并行生成/证明
- [Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html)、RingAttention、[Adaptive Softmax](https://proceedings.mlr.press/v70/grave17a.html)、[BasisKV](https://arxiv.org/abs/2611.02583)。
- [Mixture Transition Distribution](https://sites.stat.washington.edu/people/raftery/Research/PDF/berchtold2002.pdf)、[Sparse Markov Models](https://www.jmlr.org/beta/papers/v24/22-0266.html)。
- [Parallel Sampling via Counting](https://arxiv.org/abs/2408.09442)、[Autospeculation](https://arxiv.org/abs/2511.07869)、[Any-Subset Autoregressive Models](https://arxiv.org/abs/2504.20456)、[Domino](https://arxiv.org/abs/2605.29707)。
- Slalom、Lagrange Coded Computing、zkLLM、Freivalds verification、ReplaySSM、CacheReforge、activation sparsity、Spark Transformer、Sparsing Law。

## 6. 当前验收总表（截至 A/R13）

| G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|
| 未通过 | 未通过 | 局部数学结论成立；无完整候选 | 未通过 | 未通过 | 未通过 | 部分历史实验有记录；完整独立复现未通过 |

所有 13 轮均未达到最终标准。不得把“部分通过”当作整个门槛已通过。目前缺少真实预训练模型两尺度、真实网络/设备异质性、质量匹配 end-to-end 结果和可独立复现的完整方案。本次是历史档案整理，不是重新实验。

## 7. 下一轮研究任务（H，不是已经证明的 Idea）

A/R13 后建议停止围绕通用统计估计器小修小补。优先挑战 HFL 的基本学习对象：**能否让区域学习直接新增、删除或改变正式模型可执行的计算规则，并使这些规则具有可组合的精确自推测验证运算？**

必须区别于已经否决的固定更新码、ReLU 仿射编译、奖励机器、低秩接口、随机区域路由等。要求：
- 先给出真实计算矛盾、明确目标模型和学习目标，保证非退化学习。
- 形式化哪些计算必需、哪些可消除，提出能被反例推翻的定理或命题。
- 证明信息与时间约束下的 HFL 独立贡献；不以层级名义限制平坦基线。
- 尽早在真实小型预训练模型和公开任务上检验问题是否存在；没有 CUDA 时只能做 CPU pilot 并注明局限。
- 不必死守上一轮建议。如果新证据支持完全不同的机制，直接 pivot。
- 重点阅读其他团队新增的研究成果与失败经验，先去重，再开展实验。

## 8. 后续团队追加模板

每个研究团队每轮在本节之后追加，必要时同步更新上述索引：

~~~markdown
### <团队 ID>/<轮次>｜<计算问题/机制名>
- 日期 / 读取的仓库 SHA：
- 与已否决 ID 的区别：
- 实际问题证据与明确目标模型：
- 第一性原理推导、假设、定理及证明：
- 近邻原始论文（≥5，核对链接和日期）：
- 可证伪主张及最强公平基线：
- 实验环境、真实/合成数据、模型、种子、命令、代码和原始结果链接：
- 数值结果与误差/置信区间，明确实际时间或代理指标：
- 失败条件及 CRITICAL/MAJOR 缺陷：
- G1–G7 状态、裁决（Reject and Pivot / Revise / Pass）：
- 保留的结论、禁止重复的实现、下一步区分性实验：
~~~

---

## 9. 跨团队追加日志

> 此处保留供其他研究对话追加。**先读最新版本，再写入；失败也要提交。** 初始记录 A/R01–A/R13 及 PRE-01–PRE-11 已在上文归档。


### HC/R01｜HiCut-SD：分层验证切面与预测偏差抵消

- **日期 / 读取版本：** 2026-11-24；追加前 README blob SHA：c9a305f316ad5bb371e4ab3c731dd590a5e0b9ae（不是 commit SHA）。团队 ID HC（HiCut-SD 独立对话）。
- **裁决：CRITICAL / Reject and Pivot（核心机制重复 A/R11 FSV；保留数学与工程分析）。** 独立探索时曾暂定保留、创新度估计 6.9/10；读共享库后发现 A/R11 已覆盖随机验证组、TV 凸性和逐 token 路由，且已有更深入的负结果。不能以 HiCut-SD 改名重启，也不能把 A/R11 数值作为 HC 团队实测。
- **证据等级：** T（初步数学推导，需复核）、P（构造例子和理想成本模型）、O（假定完整信息的树形代理最优解）、H（尚未验证的实验/学习机制）。本轮无实际 CPU/GPU 实验、无代码或原始日志、无独立复现 R 证据。

#### 1. 场景、目标和信息约束（H）

真实分层个性化联邦语言建模：客户端持有私有非 IID 数据 D_i，边缘聚合区域适配器，云低频聚合全局适配器。冻结共享主干 theta_0，客户端参数写为 theta_i=theta_0+Delta_theta_cloud+Delta_theta_edge(i)+Delta_theta_i；初步目标为加权语言模型 NLL + 正则项。普通 HFL 不必联合推理，本场景额外定义了 token 级概率混合，不能宣称与参数平均模型相同。

给定已提交上下文 h、权重 w_i>=0 且 sum_i w_i=1，正式目标 P(v|h)=sum_i w_i p_i(v|h)。模型版本请求内固定，候选来自共享主干内部特征上的轻量预测头 q_phi（Medusa/EAGLE 风格，不部署完整独立草稿 LLM）。边缘持有其分支、云持有协调策略及历史低成本统计。决策前不能免费查询当前所有 p_i，否则消除验证成本的前提失效。目标分布与截断、温度的施加顺序需要预先明确定义。

#### 2. 计算矛盾与完整构造例子（T/P）

最大耦合单步接受率 alpha(P,q)=sum_v min(P(v),q(v))=1-TV(P,q)。四个等权二元模型 p_1=(.9,.1), p_2=(.1,.9), p_3=(.8,.2), p_4=(.2,.8)，候选 q=(.5,.5)。全量联合 P=q，接受率 1，模型数 4。逐位置随机单模型验证，分别接受 .6/.6/.7/.7，平均 .65，模型数 1。分组 {1,2}/{3,4} 后，各组条件混合均为 q，接受率 1、模型数 2。混合切面 {1,2}/{3}/{4} 的接受率 .85、期望模型数 1.5。**这些是解析构造值，不是模型测量、实际延迟或显存收益。**

局部相对 q 的误差可以在加权聚合后抵消，降低验证粒度时可能丢失这一收益。若组内模型同向偏差，例如全部 (.9,.1)，q=(.5,.5)，无论合并多少接受率始终 .6；HFL 通常同质聚合，可能恰恰缺少组内互补性。

#### 3. HSVP：Hierarchical Selective Verification Problem（T/H）

在模型层级树 H=(N,E) 上，合法切面 C 是不交叠且覆盖所有叶的子树根集合。组 G 的权重 W_G=sum_{i in G}w_i，条件目标 P_G=sum_{i in G}w_i p_i/W_G（忽略零权重组），恒有 P=sum_{G in C}W_G P_G。切面可同时包含不同层级。拟优化策略 pi、候选长度 gamma、预测头 phi，使长期单位提交 token 的期望成本 E[tau+lambda B]/E[L+1] 最小，其中 tau 是端到端块耗时、B 是通信字节、L 为接受前缀长度；lambda 的单位应与 tau/B 匹配。正式约束为逐 token 条件分布及完整序列分布与同一混合 AR 目标一致。该目标不是对原始单个 pretrained 模型的无损加速，也不保证混合目标 greedy 一致。

**机制：** (1) 已确认上下文和历史分歧/网络成本选择合法切面；(2) 轻量预测头生成 gamma 个条件候选及其真实 q_j；(3) 对每个位置（含全接受后的 bonus）独立按 W_G 抽取组，独立于当前及未来草稿随机性；(4) 只计算相应组的条件目标，在末端适配器设定下尝试共享基础特征；(5) 接受候选 x 的概率 min(1,P_G(x)/q(x))，拒绝用归一化 [P_G-q]_+ 采样，停止并丢弃后续草稿；全接受时 bonus 直接从其抽取组目标采样；(6) 边缘记录历史偏差与费用，云周期性更新策略/候选头（尚未设计出独有联邦学习机制）。不能按观察到的当前候选或某组验证成功后任意更改路由概率。

#### 4. 理论与新增代理优化（T/O）

- **单步/序列精确性（与 A/R11 重合）：** 条件于组 G，输出概率为 min(P_G(v),q(v))+[P_G(v)-q(v)]_+=P_G(v)。对组随机性边缘化即 P(v)。若各位置在已确认上下文条件下具有规定混合权重且不与当前草稿随机性耦合，按链式法则推广到整个自回归序列。需要重新检查实现的停止时刻、预采样未来组及有限精度；不声称相同随机 seed 的文本逐位一致，也不声称支持任意 top-p/greedy。
- **合并单调性（与 A/R11 重合）：** alpha_C=1-sum_G W_G TV(P_G,q)。由 TV 凸性，合法合并 A,B 不会降低单步平均接受率。
- **显式抵消增益（T，已有三角不等式的直接推论，不认定新定理）：** J(A,B;q)=[W_A||P_A-q||_1+W_B||P_B-q||_1-||W_A(P_A-q)+W_B(P_B-q)||_1]/2 >=0。同向偏差无增益，逐坐标异号可能产生正增益。
- **树形切面 DP（O，HC 新增明确推导，标准树形动态规划）：** 对固定上下文、已知全部组 TV 与可加成本 c_G，代理目标 F(C)=sum_G W_G[TV(P_G,q)+lambda c_G]。递推 V(G)=min{W_G[TV(P_G,q)+lambda c_G], sum_{H child of G}V(H)}；叶节点只能取第一项。归纳可知合法切面代理全局最优，给定节点分数时 O(|N|) 时间。完整概率取得成本未计，不能作为在线可部署算法；真实 SD 多步接受、批次、共享权重读带宽、KV cache、网络 RPC 都不具备简单可加性，DP 不证明最优真实 wall-clock。
- **简化成本敏感性（P）：** 若所有位置条件接受率恒为 alpha，E[L+1]=sum_{j=0}^gamma alpha^j，归一化最强 AR 单 token 成本=1，理想加速 S=E[L+1]/(C_draft+C_verify)。示例 gamma=4,alpha=.8,C_draft=.25,C_verify=1.5，E[L+1]=3.3616，S≈1.921；仅理想化参数，不是实测。若接受条件概率随前缀变化，应计算真实前缀生存概率。

#### 5. 主动反例与失败诊断（T/P）

1. **整块固定组不精确。** p1=(1,0),p2=(0,1)，等权、恒定条件分布；正确逐位置混合的 AA/AB/BA/BB 各 .25。整个长度2序列只选一次模型则 AA/BB 各 .5，AB/BA 为0。必须逐位置独立抽组；以共享一次路由降低成本会改变序列目标（与 A/R11 反例同类）。
2. **同质层级无互补收益。** 如上述全部 (.9,.1)；现实 HFL 区域同质性可能构成直接反例。
3. **批内验证组覆盖几乎全量。** m 个预验证位置、组权重 W_G，期望访问不同组数为 sum_G[1-(1-W_G)^m]。两等权组、gamma=4 加 bonus，m=5，期望 1.9375/2 组。若每个组均需运行整块前缀或恢复自己的 KV，逐位置理论模型调用减少无法等价为真实 GPU 节约；共享主干后末端头开销也可能小到无研究价值。
4. **无 SD 强基线已可精确抽样。** 逐 token 按 w_i 抽一个模型再采样即得到 P；不必每次全量混合。最强同信息平坦委托也可执行相同随机分组、切面和普通共享计算。固定 HFL 树甚至可能比全局自由配对差；共享库 A/R11 已有此类负结果。
5. **greedy 不保证。** 随机组输出的 argmax 不等于完整 P 的 argmax；例如 p1=(.9,.1),p2=(.4,.6)，等权，混合 greedy 为 A，但组选 B 的概率非零。不能将随机分布等价当作 greedy 正确。
6. **质量与双向必要性尚未证明。** 概率混合本身未证实提高领域模型质量；预测反馈尚只是普通候选头训练/切面统计，无 HFL 必要的新学习对象。A/R11 已否决核心机制；仅增加 DP、调度、训练损失、扩大实验不能消除重复。

#### 6. 文献检索与撞题审查（历史对话检索线索，非本次独立核验）

起始原文链接（用户指定）：
- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)（2023）；精确接受/残差采样。
- [SD Survey](https://aclanthology.org/2024.findings-acl.456/)（2024）；SD 分类。
- [EAGLE](https://arxiv.org/abs/2401.15077)（2024）、[EAGLE-2](https://arxiv.org/abs/2406.16858)（2024）、[EAGLE-3](https://arxiv.org/abs/2503.01840)（2025）；附属轻量预测、动态树、训练时测试/多层特征。
- [Medusa](https://arxiv.org/abs/2401.10774)（2024）；多预测头、树验证；不同采样策略需区别精确性。
- [SpecFed](https://arxiv.org/abs/2604.25777)（2026）；历史阅读报告称各节点概率聚合、Top-K 压缩与重建；正式稿需复核其误差和协议细节，不能只凭历史描述比较。
- [SpecTr](https://arxiv.org/abs/2310.15141)；多候选最优传输。[LayerSkip](https://aclanthology.org/2024.acl-long.681/)；早退、自投机。
- [Lossless Hierarchical Speculative Decoding](https://arxiv.org/abs/2601.05724)；层级草稿/验证，需核对与目标组切面的差异。
- [FedEM](https://proceedings.neurips.cc/paper/2021/hash/82599a4ec94aca066873c99b4c741ed8-Abstract.html)；混合个性化学习。
- [FACT](https://arxiv.org/abs/2611.23463)；历史检索记录涉及预算/跨步训练，待核对。
- 额外历史检索名称：CoS（多模型联合分布精确投机，最高近邻风险，原文链接需复核）、HierFAVG / Client-Edge-Cloud Hierarchical FL、HFL with Quantization、DSSD、AsymSpec、DISCO、SpecDec++、Acceptance-Aware Draft Model Training、REAL Loss、Optimal Multi-Draft SD、OnlineSPEC、FeMAM、FedGMM、PFAdapter、FLoRA、Punica、S-LoRA。没有已核验链接者仅作检索线索，不伪造引用。CoS/SpecFed 可能覆盖多模型混合目标；接受率损失、动态草稿长度、网络调度、压缩、个性化适配和共享 serving 均不能独立作为新贡献。

检索记录（历史首轮日期 2026-11-24）：client-edge-cloud hierarchical federated learning; hierarchical personalized FL; mixture of distributions; self-speculative decoding; LayerSkip; EAGLE; distributed speculative decoding; edge-cloud verification; speculative ensemble; collaborative decoding; mixture target sampling; randomized verification; hierarchical partition; verifier grouping; total variation; expected acceptance length; acceptance-aware training; multi-LoRA serving; shared backbone。历史报告列出 arXiv、PMLR、ACL Anthology、OpenReview、NeurIPS/ICLR、IEEE 等来源。盲区：分层随机化耦合、混合最优耦合、延迟采样、分布式 Monte Carlo、最新未索引工作。未进行逐篇全公式复核和正式文字查重；论文须独立撰写并规范引用，内容重复不能靠改写解决。

**最直接的去重依据是仓库 A/R11 FSV，不依赖外部查重是否完成。** 其历史数值实验（E-H/P）已经显示平坦配对可匹配/超越边缘分组，不应重做同一核心机制。HC 树形切面 DP、批内组覆盖公式保留为后续研究的限制与分析工具，不认定为可立项的新 Idea。

#### 7. 原拟最小与完整实验（H，未执行；当前裁决取消重复性实验）

原拟 CPU 二元分布枚举、1×40GB GPU Qwen2.5-1.5B-Instruct + 4 个轻量适配器、8 虚拟客户端/2 边缘/1 云、LoRA rank 8/16、gamma 1/2/4/6/8、500–2000 条独立测试上下文；数学、代码、通用 QA、摘要构成非 IID；温度采样初期不做截断。先将适配限制在最后层注意力之后的 MLP/输出模块以共享此前 KV；注意个性化能力可能受限，后续若增加最后 2–4 层适配必须计独立 KV/前缀补算。第二规模 3B 或 7B，多进程网络模拟，最多 2×40GB；7B BF16 权重约 14GB（不含激活、优化器和缓存）。无需独立草稿 LLM。

原拟离线 oracle：固定真实模型分布、枚举合法切面，绘制接受率—成本 Pareto 前沿，比较 HFL 树、随机树、全量与叶节点。未计获取成本的 oracle 只用于证伪，不能当可部署性能；若无至少 20% 潜在优势立即停止。考虑到 A/R11 负证据，当前不建议投入 GPU 重做这一实验，除非出现改变核心前提的独立证据。

原拟公平基线：全量混合 AR；逐 token 随机单模型 AR（共享主干、适配执行、cache）；附属头+全量混合验证；附属头+随机单模型验证；静态边缘组；SpecFed（如近似须单列质量/误差）；CoS（适用同一目标时）；同信息平坦自由分组、同成本委托与预缓存。消融：动态切面、原始 HFL 树、抵消统计、候选头联邦训练、主干复用。指标：真实端到端耗时、TPOT、吞吐、P50/P95、通信 bytes/token 及 RPC、GPU time、FLOPs、缓存与峰值显存、接受长度、模型质量、训练通信；数学分布枚举与采样检验。正式实验应满足两种模型尺度、三种异质性/延迟、至少三种子及 CI。无真实测量、种子、脚本、日志或置信区间可上传。

#### 8. 成功门槛与 G1–G7（按共享库定义重新评估）

旧对话给的主观创新分项：问题 7.5×25%、机制 7×25%、理论 6.5×20%、价值 7×15%、距离 6×15%，加权 6.875≈6.9/10；已经不达标。读 A/R11 后核心重复，原评分失效，不再给貌似精确的新分数。原拟好结果门槛：最强精确 SD 吞吐 +20%、最强无 SD 端到端 1.3×、网络瓶颈通信 -30%、两模型尺度/三领域、<=2×40GB、优于随机或平坦层级。最终必须遵循仓库 G5 的**全流程真实总 wall time -15% 或同成本有效工作 +20%，质量/预算匹配**，不得只汇报 inference speedup。

| 门槛 | HC/R01 裁决 |
|---|---|
| G1 | FAIL：A/R11 已覆盖核心；原创新评分也不足 8 |
| G2 | FAIL：HFL 学习未产生不可替代计算；SD 未反向改变学习；平坦可模拟 |
| G3 | PARTIAL/FAIL：有限假设下随机采样正确，greedy 失败，无非平凡新理论、无有限精度实现审计 |
| G4 | FAIL：零真实 LLM 实验 |
| G5 | FAIL：无端到端证据，批内组覆盖/KV 为明确风险 |
| G6 | FAIL：无证据支持好结果概率 80%；不提供虚构成功率 |
| G7 | FAIL：仅推导及设计，无运行代码或原始日志 |

**保留资产与下一步：** J 的显式表达、代理切面 DP 的适用边界、期望访问组数公式、随机混合 greedy 反例、强 AR/平坦基线要求可纳入公共限制。禁止仅通过改名、随机选组粒度、普通候选头联邦训练或接受率损失重新提交 A/R11/HC/R01。原本下一轮想让 HFL 主动形成互补层级，但普通分组/混合训练及人工错误抵消不足以产生独立贡献；只有新计算对象、可证明的非退化学习/验证双向必要性以及无法被强基线抹去的真实约束才值得继续。下一轮从共享库所有团队最新结果出发重新创造，不预设继续 HiCut-SD。

### team-hcr-sd/R01｜HCR-SD：消差感知分层精确投机解码（暂定，未立项）

- 日期：2026-11-24；本团队为独立 ChatGPT 研究对话，唯一 ID team-hcr-sd。第一轮报告产生于读取共享库前，归档时已阅读共享库并补充跨团队审查。
- 归档读取 README blob SHA：c609e52f62cbb91d7d83ac1b7f479fbde144f611（文件 blob SHA，非 commit SHA；初次读取 c9a305f316ad5bb371e4ab3c731dd590a5e0b9ae）。
- 原始裁决：Revise / 暂存，不批准正式立项；当前跨团队审查后仍不通过 G1–G7。没有真实 LLM 或 GPU 结果，不宣称顶刊创新或 80% 成功率。

#### 1. 研究场景、目标与信息所有权（H）

采用 client-edge-cloud 个性化 HFL：客户端私有非 IID 数据 D_{i,k}，客户端本地更新共享参数/LoRA/预测头，边缘按 w_{i,k} 聚合，云按 a_i 聚合共享参数。区域保留适配器 φ_i 和内生预测头 ψ_i，冻结共同主干，推理会话锁定版本。训练可写 θ_i^{r+1}=Σ_k w_{i,k}θ_{i,k}^{r+1}，θ_c^{r+1}=Σ_i a_iθ_i^{r+1}；实际 LoRA 聚合和个性化规则仍待实现，不能把因子平均误称为有效矩阵平均。

Edge i 对共享上下文 h 持有目标概率 p_i(x|h) 与轻量内生头 q_i(x|h)，局部 KV cache 不上传。Cloud 持有混合权重、授权的上下文、提交状态，可请求标量概率但不免费获得完整向量。假设相同词表、可靠通信、可信节点，不声称隐私保护；共享 prompt 本身是额外假设。

正式目标为逐位置概率混合 P(x|h)=Σ_i a_i p_i(x|h)，a_i≥0、Σa_i=1；不是 FedAvg 参数平均后的 LLM，须单独证明选择此服务目标有真实应用价值。若所有区域模型相同，跨区消差的独立价值消失。目标序列分布为 ∏_t P(y_t|h,y_<t)。允许本地前向、随机采样、标量/向量消息；优化 E[T_round]/E[N_committed]，同时统计链路字节、GPU 总工作、尾延迟。初步候选用 Medusa 风格固定上下文多步头；EAGLE 仍有轻量额外候选模块，并非完全无草稿计算；LayerSkip 可作替代。真正 HFL 训练—推理双向机制尚无。

#### 2. 计算矛盾、可检查反例与理论资产（T）

标准 SD：X~Q，接受 min(1,P(X)/Q(X))，拒绝时从 (P−Q)_+/Z 采样，Z=TV(P,Q)。概率聚合与取正一般不可交换：

[Σ_i a_i(p_i−q_i)]_+ ≤ Σ_i a_i[p_i−q_i]_+（逐 token），一般不等。

双边缘、a=(.5,.5)，三 token A/B/C：
- p1=(.8,.1,.1)，q1=(.4,.5,.1)
- p2=(.2,.6,.2)，q2=(.5,.4,.1)
- P=(.5,.35,.15)，Q=(.45,.45,.1)
- d1=(.2,−.2,0)，d2=(−.15,.1,.05)。
- 正确全局残差 (.05,0,.05)，归一化 (.5,0,.5)。
- 局部正残差聚合 (.2,.1,.05)，归一化 (4/7,2/7,1/7)。错误地使用该修正分布会输出约 (.507143,.378571,.114286)，不等于 P。
- 全局最大耦合单步接受率 .9；先选节点并局部最大耦合的接受率 .65。此为人工反例，不代表真实 LLM 的接受长度或速度。
- 当逐 token 各节点概率差同号时等式成立；异质模型可产生抵消，但不保证实际误差互补。
- 与 PRE-06、A/R11 FSV 高度相关：二者已经研究概率混合与接受率凸性，故上述不等式不能算独立原创贡献。

#### 3. 初步协议（T/H）：块级候选 + 后验权重 + 标量验证 + 残差包络

(1) 从 π 采样一个隐藏候选节点 I，生成整块 X_1:K~q_I；块分布 Q(X_1:K|h)=Σ_i π_i∏_j q_{i,j}(X_j|h,X_<j)。不允许把整块混合误认为逐位置固定权重混合。

(2) 对已观察候选前缀，计算 b_{i,j}=π_i∏_{l<j}q_{i,l}(X_l)/Σ_m π_m∏_{l<j}q_{m,l}(X_l)。于是 Q_j(x)=Σ_i b_{i,j}q_{i,j}(x)，目标 P_j(x)=Σ_i a_i p_{i,j}(x)。例：初始各 .5，两个头生成 A 概率 .9/.1，观察 A 后后验变为 .9/.1。各边缘收到整块后返回 K 个目标与候选标量概率，云按序验证，接受概率 min(1,P_j(X_j)/Q_j(X_j))。只使用已到达候选前缀与独立随机数，不得利用未来候选或已知 I 改变前面接受条件。即使云在路由时知道 I，也必须忽略它对验证决策的额外信息。

(3) 拒绝后令 d_i(x)=a_i p_i(x)−b_i q_i(x)，D=Σ_i d_i，r_0=[D]_+，e(x)=Σ_i[d_i(x)]_+≥r_0(x)。各边缘本地计算 R_i=Σ_x[d_i(x)]_+，云按 R_i/ΣR_i 选择节点，该节点从局部正残差采样 z，再由各节点返回 d_i(z)；以 β(z)=[Σ_i d_i(z)]_+/Σ_i[d_i(z)]_+ 接受，失败重试。只在残差触发时支付完整本地概率扫描、包络准备与相应 RPC；这些成本不是免费。若所有候选通过，从当前目标混合分布补采样一个 token（EOS/长度边界需特判），失败位置提交修正 token，丢弃其后候选并同步各节点 KV 状态。

(4) 真正层级扩展：对推理参与节点树的任意互不相交覆盖切分 C，e_C(x)=Σ_{G∈C}[Σ_{i∈G}d_i(x)]_+。子树内聚合先消差，再作上层包络，合并子树有 e_C2≤e_C1，且 r_0≤e_C2。高层可能降低重试，但完整子树向量聚合增加通信。动态最优切分仅是 H 级假说，尚无可部署选择算法。初始双边缘模型仅验证树的简单特例，不能假装完成多层部署。

#### 4. 理论证明与边界（T）

- 单步精确性：R=Σ_x e(x)，Z=Σ_x r_0(x)。一次提议并接受 x 的概率为 (e(x)/R)(r_0(x)/e(x))=r_0(x)/R，最终接受分布为 r_0/Z；正式输出概率 min(P,Q)+[P−Q]_+=P。零包络 token 不会被提议；Z=0 时不触发残差。
- 包络单调性：由 [u+v]_+≤[u]_++[v]_+，合并子树后 Z≤R_C2≤R_C1≤1（因正差不超过 a_i p_i）。合并后通信不一定更低。
- 当 Z>0，拒绝条件下尝试次数服从成功率 Z/R 的几何分布，E[N|reject]=R/Z，单个已到达位置无条件 E[N]=Z·R/Z=R≤1。Z=0 时 E[N]=0 而不一定等于 R。该式不是端到端时间或总消息下界，不能忽略预处理。
- 多步正确性证明思路：候选前缀接受事件在给定已观察候选前缀后不再依赖隐藏节点 I；因接受规则不利用 I 或未来候选，后验仍为 b_{i,j}，再逐位置归纳。完整随机过程、EOS、温度/top-p、有限精度、数值下溢与 cache 回滚尚未审计；不保证 greedy 与混合目标一致（另需全局 argmax 算法）。不能因此宣布 G3 通过。
- CRITICAL 尾部反例：Z=.001、R=.5，拒绝后期望 500 次尝试。可用并行预提议或设上限，回退时集中计算并仍从正确残差采样，不得直接从 P 补采。上限降低长尾但最坏通信仍可为完整词表；禁止声称保证只传标量。
- 最强无 SD 目标采样：每个 token 独立按 a_i 选区域再从 p_i 采样，严格得到 P，无需全节点前向。这是 A/R11 已发现的强对照。不能按同一 I 整块路由目标：例如两区 Bernoulli .9/.1，固定区整块的两步联合 [.41,.09,.09,.41]，逐位置独立路由为 [.25,.25,.25,.25]，TV=.32。HCR 候选块可共享 I 仅因使用真实后验、逐位置对混合目标验证。
- 同信息平坦系统能模拟树协议；只有带宽/地域数据驻留等真实约束可产生拓扑优势，当前未证实。HFL 的学习过程没有反向受 SD 影响，G2 失败。

#### 5. 数值实验记录（E-H + 本次归档可复算 CPU 检查）

历史 R01 报告声称 N=200000、seed=20261124 的三 token CPU 模拟：输出 A/B/C=49.892%/35.1655%/14.9425%，接受率90.004%，无条件平均修正次数 .34606，拒绝条件下 3.462，经验 TV=.001655。未保留原始脚本/日志，按 E-H 转录而非独立复现。

本次归档另用确定性的 JavaScript LCG、N=200000、seed=20261124 实际执行相同数学例子的独立 CPU 数值检查（不同 RNG，不应要求历史数字逐位相同）：
- A/B/C counts = 99588/70175/30237；概率 .49794/.350875/.151185；TV=.00206。
- 接受 179867 次（.899335），残差尝试 70627 次，无条件平均 .353135，拒绝条件下平均 3.508021656。理论 .9/.35/3.5。
- 理论与模拟一致，仅检验三 token 单步聚合包络，不代表真实 HFL、真正网络、GPU、block SD 或实际速度；未运行多种子和 CI。LCG 仅用于可复算 sanity check，并非推荐生产 RNG。
- 环境：ChatGPT JS V8 工具运行；未测 wall time；以下为执行脚本（保存为 hcr_probe.js，命令 node hcr_probe.js）。不将其标成完整 R 级 LLM 复现。

~~~javascript
let s=20261124>>>0;
function u(){s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;}
function draw(w){let x=u()*w.reduce((a,b)=>a+b,0);for(let i=0;i<w.length;i++){x-=w[i];if(x<0)return i;}return w.length-1;}
const p=[.5,.35,.15],q=[.45,.45,.1],e=[.2,.1,.05],r=[.05,0,.05];
let n=200000,out=[0,0,0],accept=0,attempts=0;
for(let j=0;j<n;j++){
  let x=draw(q);
  if(u()<Math.min(1,p[x]/q[x]))accept++;
  else{do{x=draw(e);attempts++;}while(u()>=r[x]/e[x]);}
  out[x]++;
}
console.log({n,out,accept,attempts,prob:out.map(x=>x/n),tv:out.reduce((a,x,i)=>a+Math.abs(x/n-p[i]),0)/2,mean:attempts/n,conditional:attempts/(n-accept)});
~~~

#### 6. 文献检索、撞题审查及被否决方向

历史检索日期 2026-11-24，关键词 hierarchical/personalized FL, client edge cloud, self-speculative decoding, distributed/federated inference, multi-draft optimal transport, acceptance length, signed mixtures, residual rejection sampling, remote sampling, LLM mixture ensembling。来源 arXiv、PMLR、ACL、ICLR、IEEE、NeurIPS、统计学期刊。以下链接与内容来自 R01 历史报告；本次上传未逐条重新打开核验，不声称已核实 2026 年论文题名/日期、算法附录或独创性。待重点检索 distributed maximal coupling、signed-measure sampling、通信复杂度下界；有未公开同期工作盲区。

七篇起点：
- [Leviathan et al., Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- [SD Survey](https://aclanthology.org/2024.findings-acl.456/)
- [EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [SpecFed](https://arxiv.org/abs/2604.25777)：历史报告认为已覆盖联邦目标概率聚合、接受/残差及 Top-K 压缩近似，最直接竞争者；不能重新命名为原创。

扩展文献及风险：
- [HierFAVG](https://arxiv.org/abs/1905.06641)、[Personalized HFL](https://link.springer.com/article/10.1186/s13677-024-00721-w)：HFL 层级与区域模型依据；训练后概率混合不是标准 HFL 必然流程。
- [LayerSkip](https://aclanthology.org/2024.acl-long.681/)：共享模型浅层起草。
- [FlexSpec](https://arxiv.org/abs/2601.00644)、[Online SD](https://proceedings.mlr.press/v235/liu24y.html)、[When Drafts Evolve](https://proceedings.mlr.press/v306/qian26k.html)、[DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8766fbc68e1ed1cdef712ce273e0a363-Abstract-Conference.html)：在线适配、蒸馏已有；不将训练轻量头或随目标版本更新作为核心贡献。
- [Decentralized SD](https://arxiv.org/abs/2511.11733)、[DSSD](https://arxiv.org/abs/2507.12000)、[SPADE](https://arxiv.org/abs/2608.13076)：分布式、切分及端云部署已有；不以边缘起草、云验证本身为原创。
- [Acceptance-Aware Draft Training](https://arxiv.org/abs/2609.24150)、[D-REAL](https://arxiv.org/abs/2611.10432)：接受长度或前缀存活优化，不能只换目标函数命名。
- [Multi-Draft Speculative Sampling](https://arxiv.org/abs/2410.18234)、[Optimal Multi-draft SD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0907335ecf28faf15be54485dbcbe70e-Abstract-Conference.html)、[Global Resolution](https://arxiv.org/abs/2511.15898)、[SpecTr](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6034a661584af6c28fd97a6f23e56c0a-Abstract-Conference.html)：多候选最优验证和 OT 先例。
- [DISCO](https://arxiv.org/abs/2610.35957)：历史报告认为可免完整概率通信并精确采样，借助服务端复制草稿；必须核对是否已覆盖标量验证/残差协议。
- [Lossless Hierarchical SD](https://arxiv.org/abs/2601.05724)：候选结构层级及概率质量平衡，非天然 HFL 网络层级；仍需逐式比对。
- [Simulating Signed Mixtures](https://arxiv.org/abs/2401.16828)：正负混合分量配对与拒绝采样，直接威胁核心包络的新颖性。
- [Rethinking LLM Ensembling / Mixture Models](https://proceedings.mlr.press/v306/fu26j.html)：按 token 路由目标节点的强基线。
- [Remote Sampling](https://pmc.ncbi.nlm.nih.gov/articles/PMC7514202/)：分布参数分散的精确远程采样，分布式拒绝采样原理非新。
- 其他历史论文引用必须回到原文；无正式文字查重报告，归档不是出版原创性认证。

本轮已排除：HFL+Medusa 简单训练拼接、在线草稿适配、接受长度直接优化、端边云分层部署、SD+Top-K 压缩、多节点候选统一验证。失败原因是已有相关方法覆盖基础问题或缺乏不可替代机制。

跨团队补充：PRE-06 区域草稿混合、A/R06 FRCS 残差耦合、A/R07 概率层级、A/R10 FRDD 概率抵消、A/R11 FSV 混合目标、局部/全局耦合、强路由基线均已存在。HCR 的相对差异仅在整块隐藏草稿后验及分布式正残差包络取样；前者是标准 Bayes 法则，后者是已有拒绝采样结构的应用，尚不足以支撑 G1。禁止其他团队以同一数学结构换名重新立项。

#### 7. 最小实验、强基线与失败判据（H）

预算 1–2×40GB NVIDIA。冻结 Qwen2.5-1.5B/3B 主干，多 LoRA，8–16 逻辑客户端，2–4 区域，CPU 云协调，2–4 预测头，BF16、512–2048 context；后续考虑 7B。单 GPU 顺序模拟仅测概率、接受长度和正确性，不可称分布式加速；两卡可做真正并行验证，tc/netem 仅模拟网络。真实数据先选数学/代码/通用领域公开数据，至少三非 IID/时延设置、三随机种子、两尺度；具体公开数据和训练命令仍未选定。

最小实验：先训练或加载两个区域 LoRA、训练内生预测头；记录 P,Q,p_i,q_i 的 TV、抵消 R−Z、R/Z 分位数、多步接受长度，比较路由基线所需 KV 补算、完整计算和通信。后续网络带宽 10/100/1000 Mbps、RTT 1/10/30/50ms，记录 TTFT、TPOT、吞吐、字节数、FLOPs、总训练及推理时间、P95/P99、采样质量及失败回退。

公平基线：HFL-AR 全概率聚合；HFL-ME 每 token 随机目标路由；直接 HFL+Medusa-style（同目标精确实现）；完整向量 SD；SpecFed 近似方案（必须报告质量差异）；逐位置 local-coupling；正确 routed target-first SD；同信息平坦委托与完整缓存强基线。所有方法维护合法上下文和 KV，计各模型头、网络与失败开销。消融：取消后验/消差（仅诊断分布错误）、直接用 P 作包络、固定候选节点、固定层级切分、无区域个性化、对照平坦/无 SD 学习。

原始拟定好结果：同目标最强精确推理基线 TPOT 下降≥20%、吞吐提升≥20%、相对完整向量通信下降≥90%、尾延迟无严重退化、多域多种子稳定；这些不是测量结果。共享库 G5 还必须核算训练+推理总 wall time（下降≥15% 或等预算有效工作+20%），不能用解码速度单独替代。性能门槛：TPOT≈(T_d+T_v+T_c)/L，要低于基线 .8T_b，L≥(T_d+T_v+T_c)/(.8T_b)。初始展示例 T_d=8ms,T_v=25ms,T_c=15ms,L=3,T_b=25ms 仅假设，不是 GPU 结果。

否决条件：真实区域不存在明显消差；多模型前向+通信不能击败 HFL-ME/平坦；尾延迟严重；层级不具备独立贡献；近邻覆盖核心机制；没有 HFL↔SD 双向学习机制。先做低成本否决实验，不追加无意义模块。

#### 8. G1–G7 与下一轮

原始主观创新度 6.4/10（问题7、机制6、理论6.5、价值7.5、文献距离5），已低于门槛；跨团队去重后风险更高，不虚构新的精确评分。好结果概率没有数据支持，不填80%。无真实 GPU 实验。

| G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|
| 未通过：签名混合/远程采样、A/R11 重合 | 未通过：HFL 学习不是必需，平坦可替代，缺反向作用 | 部分 T：单步精确、后验、包络；完整序列/greedy/精度未审计 | 未通过：仅 CPU toy | 未通过：无端到端 | 未通过：无80%依据 | 部分：本节 JS 可复算，非完整系统 |

裁决 Revise / 不批准立项；若无改变核心前提的新证据，Reject and Pivot。保留不可交换反例、后验权重约束、几何长尾、精确性证明与路由基线作为共享科研资产。下一轮优先：① 建立真实 HFL 学习改变 SD 计算且 SD 反向改变学习的机制，不重复 PRE-06/A/R11；② 复核最相近原文、通信下界与同信息平坦；③ 小型真实模型收集消差和 KV 成本，若不可能获得净收益立即否决。每轮开始读取默认分支 README 全文、记录 SHA，结束追加正负结果，不覆盖其他团队记录。


### team-himerge/R01｜HiMerge-SD：有限支持集的分层精确投机验证

- **日期 / 读取版本：** 2026-11-24；归档前已读 README 全文，读取时 main commit f93a4c815028bcc3b7104df1e68e84785071bf85，README blob c9a305f316ad5bb371e4ab3c731dd590a5e0b9ae。此轮最初独立探索发生在读取共享库前，现补做跨团队去重；后续必须先读库。
- **裁决：Reject and Pivot（保留局部数学机制及研究资产）。** 独立报告暂评 7.2/10，未达 8/10；无证据支持强结果概率 ≥80%。共享库复核发现与 A/R11 FSV 的混合目标、TV 凸性、局部/全局耦合及强基线高度重合；不能作为新论文立项。当前未执行代码或 GPU 实验，以下数值仅为手算或理论成本。
- **与既有记录关系：** PRE-06、PRE-07、A/R01、A/R06、A/R07、A/R10、特别是 A/R11。A/R11 已报告单组/全组耦合及平坦配对匹配、整块路由反例。本轮可保留的增量是稀疏 q 支持集内的概率聚合、支持集外精确尾部样本的可合并摘要，不是已知全局接受率不等式的重新命名。

#### 1. 系统场景及明确目标（H）

跨机构 cross-silo HFL：客户端持有非 IID 本地数据 D_i，边缘按样本数聚合，云较低频聚合。共享预训练基座和本地 LoRA，保留个性化检查点。可示意写成 θ_i=θ_g+Δθ_i（实际 LoRA 参数化需明确，不能将低秩因子平均等同于有效矩阵平均）。本地优化 θ_i←θ_i−η∇F_i；区域 θ_e=Σ_{i∈C_e}(n_i/N_e)θ_i；云 θ_g=Σ_e(N_e/N)θ_e。

推理时给定已提交前缀 h，每个模型持有 p_i(v|h)=softmax(f_{θ_i}(h))_v。正式目标定义为每个位置的混合概率 p(v|h)=Σ_i w_i p_i(v|h)，w_i≥0、Σw_i=1。它不是参数平均后的云模型，也不是一次选定机构后生成整段序列的模型。需先证明混合模型相对单个聚合模型具有质量/应用价值。同 tokenizer、词表，固定请求内模型快照和聚合规则，节点获授权接收推理前缀，不宣称差分隐私或请求保密。边缘持有子树消息，云提交结果，各节点不能免费获取其他节点的完整概率或模型。现有参与模型的附加头/内部特征/早退出路径生成草稿，不部署另一个独立 LLM。EAGLE 有辅助预测模块，不是完全无额外参数；Medusa 典型宽松验证不能自动宣称精确采样。

#### 2. 第一性原理计算矛盾及反例（T）

标准精确 SD：X~q，接受 a(X)=min(1,p(X)/q(X))；拒绝时从 d(v)=[p(v)−q(v)]_+ 的归一化分布采样。接受质量 min(p,q)+残差质量 [p−q]_+=p。

混合目标中 [Σ_i w_i(p_i−q)]_+ 不等于 Σ_i w_i[p_i−q]_+；min(q,Σ_iw_i p_i) 不等于 Σ_iw_i min(q,p_i)。概率聚合是线性的，局部最大耦合非线性。局部完整 SD 按权重混合仍可保持每步正式分布，但接受率可能降低；不能把局部残差直接拼到全局接受机制。

四词手算：p1=(.9,0,.1,0)，p2=(.3,.5,0,.2)，w1=w2=.5，p=(.6,.25,.05,.1)，q=(.5,.5,0,0)。局部接受 .5/.8，平均 .65；全局接受 .75（+10pp）。全局残差 (.1,0,.05,.1)，平均局部残差 (.2,0,.05,.1)。非交换不是正确混合采样本身的错误。

定义耦合差 G=Σ_i w_i TV(p_i,q)−TV(Σ_iw_i p_i,q)≥0，等于全局接受率减局部平均接受率（TV 凸性，非新定理，A/R11 已有）。逐 token 正负偏差抵消是其来源；Non-IID 不必然提升 G，同方向偏差时 G=0。最大化 G 本身可能人为恶化局部草稿而无益于全局接受率或质量。

#### 3. 初步计算问题与机制（T/H）

HMESV：给定 HFL 树、叶节点分布、固定权重、实际草稿条件分布、词表和网络成本，输出精确接受/拒绝、残差样本和额外目标 token。最小化长期每提交 token 的期望成本 (E[T]+λ_B E[B]+λ_C E[C])/E[L]，约束完整生成序列与相同正式混合目标的 AR 条件分布一致。

HiMerge-SD：
1. 已有参与模型的内部头产生原始分布 q̃。令 S=TopK(q̃,K)，q(v)=q̃(v)1{v∈S}/Σ_{u∈S}q̃(u)。必须先截断归一化再采样，验证用实际 q，不能先按完整 q̃ 抽样后假装稀疏。候选为线性长度 γ，暂不引入动态树。
2. 对每个候选前缀，各验证节点用目标模型批量前向得到 p_i。残差在 S 内为 [p−q]_+，S 外为 p。每个叶子上传摘要 M_i=(a_i,b_i,z_i)，其中 a_i(v)=w_i p_i(v), v∈S；b_i=w_i Σ_{v∉S}p_i(v)；z_i~p_i(.|S^c)（b_i=0 不抽）。
3. 合并子树 A、B：a=a_A+a_B，b=b_A+b_B；以 b_A/b 选择 z_A，否则 z_B。b=0 时忽略样本。递归合并，摘要维度不随子树叶子数增加。
4. 根节点用精确 a(x) 判断候选接受；若拒绝，令 A=Σ_{v∈S}[a(v)−q(v)]_+、R=A+b。以 A/R 从支持集内正残差采样，以 b/R 使用预先合并的尾部样本。R=0 不会拒绝。全接受后额外 token 用普通混合采样（每步随机选模型或各子树合并一个无条件样本）。
5. 按顺序提交接受前缀及修正/额外 token，丢弃拒绝后缀，更新各节点 KV，下一轮用一致前缀及固定快照。支持集须分发，包含消息标签、前缀/版本及权重。所有条件随机数必须独立于相应接受随机数，且不提前利用未来候选信息。禁止整块共享一个机构路由冒充逐位置混合。

#### 4. 理论性质（T；仅实数、严格假设）

- 合并尾部样本：对 v∈S^c，P(z=v)=b_A/(b_A+b_B)P(z_A=v)+b_B/(b_A+b_B)P(z_B=v)=Σ_{i∈A∪B}w_i p_i(v)/(b_A+b_B)。由树归纳可知根样本为 p(.|S^c)。只是分布意义上的结合性，不保证具体样本或随机路径相同。加权随机合并是已有采样原语。
- 单步正确性：v∈S 时输出概率 min(p(v),q(v))+R[p(v)−q(v)]_+/R=p(v)；v∉S 时输出概率 R(b/R)(p(v)/b)=p(v)。b=0 或 R=0 时跳过相应分支。多步在按每个真实条件前缀、正确条件分布、独立随机数及缓存状态运行时可沿用标准 SD 证明，但实现和有限精度未验证。
- 最大单步接受率 A_g=1−TV(p,q)≥Σ_iw_i(1−TV(p_i,q))。上界 A_g≤p(S)，K 小会损失接受。全局最大单步耦合不等于多步/墙钟全局最优。
- 通信：J 条树链路，γ 个候选位置，完整概率消息 O(JγV)，摘要 O(JγK) 个标量/ID，另有 token ID 的 log2 V 编码、下行支持集及协议成本；仍需本地全词表 logits、归一化、尾部采样，不等于 FLOPs 降低。例 V=128000、K=32、γ=4、FP32、4字节 ID，完整 γ+1 分布 2,560,000 B；候选摘要 γ(4K+8) 加额外 token 4 B=548 B，约4672×单链路理论上行概率负载差，未计下行、协议、模型、网络 RTT 或同步；不代表实际总通信或速度。实际浮点实现需审计尾部质量消减误差、随机数、温度、top-p。正式目标若是先混合再 top-p，则不能直接拿局部 top-p 替代全局。greedy 全局 argmax 不能由上述固定支持集摘要保证；G3 未通过。

#### 5. 文献检索线索与原创性审查

2026-11-24 第一轮报告检索：client-edge-cloud HFL、personalized HFL、federated LLM inference、ensemble inference、distributed SD、split verification、lossless communication、exact residual sampling、self-speculative decoding、maximal coupling、optimal transport、distributed weighted sampling、sparse draft support。以下链接为上一轮报告和用户指定的文献线索；本次归档未逐条重新核验题名、版本日期及全文，不将历史引用视为新增验证，投稿前必须复核。

基础：
- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)，2023，精确 SD。
- [SD Survey](https://aclanthology.org/2024.findings-acl.456/)，2024。
- [EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)、[Medusa](https://arxiv.org/abs/2401.10774)、[LayerSkip](https://aclanthology.org/2024.acl-long.681/)：内部特征/头/早退等草稿机制；稀疏草稿本身不新。
- Client-Edge-Cloud Hierarchical Federated Learning (HierFAVG)、Personalized HFL：分层训练与个性化背景，保留模型并集成推理是本方案额外设定。

最邻近（≥5 条待进一步核验）：
- [SpecFed](https://arxiv.org/abs/2604.25777)：报告认为为联邦多模型预测聚合，独立小模型草稿，Top-K 目标概率重建；本方案不截断目标，使用稀疏 q 的精确残差；必须核对全文及版本。
- [DISCO](https://arxiv.org/abs/2610.35957)：历史报告称 2026-10、低通信精确验证，服务器复制草稿模型，需核实能否推广到混合分布多节点。单纯通信减少已非创新。
- [DSSD](https://arxiv.org/abs/2507.12000)：端边验证拆分；可能存在等价的分布式残差协议。
- [Speculative Ensemble](https://arxiv.org/abs/2502.01662)：集成目标的投机验证，重点核查低通信采样、支持集和混合耦合。
- [HSD](https://arxiv.org/abs/2510.19705)、[Lossless Hierarchical SD](https://arxiv.org/abs/2601.05724)：多级验证、层级采样；需区分模型大小级联与联邦概率所有权。
- [SpecTr](https://arxiv.org/abs/2310.15141)、[Block Verification（报告链接待复核）](https://arxiv.org/abs/2403.10444)：最优传输耦合、块级验证；比较强多步算法。
- 其他待查：DIVERSED、PipeSD、Acceptance-Aware Draft Model Training、Speculation at a Distance、分布式加权 reservoir sampling、分布式混合采样。盲区：未完整读的原文、未索引/未公开工作、专利、概率采样等价算法；没有商业查重报告。

#### 6. 最强公平基线与致命缺陷

**CRITICAL 1（混合采样）：** 每位置先 I~Categorical(w)，再 Y~p_I，即 P(Y=v)=Σ_iw_ip_i(v)，只访问一个模型便可精确采样。亦可预先为每候选位置独立选择模型，局部 SD 正确混合；缓存维护和多位置访问需计成本。若使用同一个 I 生成整块则错误（A/R11 已给反例，两个 (.9,.1)/(.1,.9) 模型，整块联合与逐位置混合 TV=.32）。HiMerge 对全局最大耦合的优势未必抵消全模型计算。
**CRITICAL 2（HFL 必要性）：** 扁平分布式聚合在相同网络/信息/区域委托计算条件下能执行同样的摘要合并；HFL 训练只产生模型，没有被 SD 实质改变。双向必要性未建立。
**MAJOR 3：** 核心机制由支持集分解+已有加权采样组成，未证明原创性；局部定理易证，非顶刊级充分贡献。
**MAJOR 4：** 没有证明全局 greedy、全局 top-p 及有限精度严格等价。实际应用需共享请求、模型一致 tokenizer；集成质量优势未证实。
**MAJOR 5：** 全部模型批量验证的 FLOPs、额外头、等待、缓存追赶可能远超基线。若 G≈0、p(S) 小、单模型质量相当或扁平匹配，即否决。

#### 7. 实验计划（H；未执行，无代码/日志/种子/CI）

CPU 最小验证：词表 100/1000/32000/128000，模型数 2/4/8/16，K=4/8/16/32/64，Dirichlet、长尾、反例，先小词表精确枚举，再每组至100万采样，验证误差、接受率、G、拓扑一致性和浮点误差。新增强反例：整块路由、尾部为零、q=p、K=1、全支持、同方向偏差、greedy 最大 token 在 S 外。
GPU：1–2×40GB，Qwen2.5 1.5B/3B 两尺度，余力7B，共享主干、4–8个独立 LoRA adapter，冻结主干训练轻量预测头，领域代码/数学/通用/领域文本，真实 HFL 微调。公开任务、≥3独立种子、≥3非IID/延迟设置须在执行前确定。单卡轮流模拟只证明数学/接受率，不能声称多节点真实速度；双卡测物理通信，其他层级网络仿真标注清楚。
基线：B0 完整概率混合 AR；B1 每位置随机机构 AR（必须）；B2 同内部头+完整概率 SD；B3 SpecFed 原始与同内部头受控；B4 SpecFed 无压缩；B5 局部最大耦合混合（含缓存）；B6 只传候选标量、拒绝才传完整残差；B7 扁平精确摘要/同等委托计算。还应比较随机模型组、渐进验证（A/R11）、预缓存及块级验证。
消融：层级/扁平；尾部精确/近似；K；个性化/同模型；Non-IID；独立/内部草稿；深度与带宽。测 TTFT、TPOT、P50/P95 latency、tokens/s、requests/s、接受长度、无效候选、每token字节与轮次、各链路负载、GPU时间/FLOPs/显存、实际分布误差、质量、G；训练和推理共同总成本含回退与 CI。
原报告暂定成功：概率通信较 dense SD 减95%，最强无损同设定基线端到端≥1.2×，局部耦合接受率+5pp，层级可重复独立收益、三任务、多网络、1–2卡。必须额外遵守共享库 G1–G7，不可用该暂定指标替代全流程 ≥15% wall-time 减少或同成本有效工作+20%、greedy 与随机精确性、双向必要性。
**数值证据：** 仅上述手算 65%/75%、理论消息大小、推导，非实测；无真实 GPU 结果；无实验脚本或命令可上传。

#### 8. G1–G7 与保留结论

| 门槛 | 本轮状态 |
|---|---|
| G1 原创性 | 未通过。原报告主观分项：问题7.5×25%、机制7×25%、理论7×20%、价值8×15%、距离6.5×15%，约7.2；共享库重叠意味着需下调/重新审查，不能凑分。 |
| G2 双向必要性 | CRITICAL 未通过：训练与验证松耦合，公平平坦匹配。 |
| G3 数学正确性 | 局部实数单步证明成立；序列实现、greedy、top-p、有限精度未完全验证，整体未通过。 |
| G4 实验 | 未执行；无两尺度、三任务/种子及真实配置结果。 |
| G5 端到端 | 未通过；仅概率负载理论计算，强基线可能更快。 |
| G6 成功把握 | 无证据支持≥80%，不得给虚假概率。 |
| G7 复现 | 数学和初步流程可复核，无原始脚本、环境、日志及完整实测，未通过。 |

保留：支持集内/外残差分解、可合并加权尾样本、非交换性反例、耦合增益的定义（与 A/R11 已有结果合并理解）、A_g≤p(S)、随机单机构/分组/平坦强基线。禁止：改名重投混合目标的 TV 优势或分层摘要作为 HFL 双向创新；把通信压缩比当速度；仅最大化 G；整块固定路由冒充逐步混合；将新混合目标精确宣称为原全局模型无损。

下一轮（H）：先阅读最新跨团队记录及 Supervisor-Skills，再根据其他团队反例创造真正改变学习与验证计算的机制。原报告提出“让训练塑造全局耦合有利结构”，目前无新学习机制，且不能仅做 acceptance-aware 训练或最大化 G；若继续，必须证明改变核心前提、真实质量/学习净收益以及 SD 反向作用，预实验需与普通 AR、单/分组混合、同信息平坦对照。否则直接 pivot，不继续围绕通用统计估计/通信摘要微调。保留本轮失败记录，后续更正用追加方式。


- HIERASPEC/R01 归档写入基于 README blob SHA：c8ee552a712ea53b0e4af1ee07d7b51fb8b5aa51（文件 SHA，非 commit SHA）。此前已阅读 c9a305f316ad5bb371e4ab3c731dd590a5e0b9ae 及 9fc7bb02fa901f2cddaf3a05b6aba069ade2eeb8 的新增记录。

### HIERASPEC/R01｜分层有符号残差求解与精确自投机验证

- 日期：2026-11-24；独立研究对话 HIERASPEC；资源上限 1–2×40GB NVIDIA GPU。
- 历史裁决：暂存、不立项，创新度 6.4/10。归档交叉审查裁决：**CRITICAL / Reject and Pivot**。A/R11 FSV、HC/R01 与本轮混合目标和 TV 凸性高度重合；team-hcr-sd/R01 更已提出相同的正残差包络、层级单调性和几何重试，且扩展至混合草稿后验。本轮不能独立立项；保留数学资产、负面证据与完整实验设计。
- T=推导；E-H=历史报告数值（无原始脚本/日志）；P=代理/理论成本；H=假说。没有真实 LLM/HFL/GPU 实验。归档补充可复算 CPU sanity check（见下），不冒充完整 R 级证据。

#### 1. 场景、目标及形式化（H/T）

三层 Client–Edge–Cloud：客户端持有私有非 IID 数据 D_i；共享基础模型 θ_0，训练适配系数 φ_i，边缘聚合 φ_e=Σ_i a_ei φ_i，云聚合 φ_c=Σ_e w_e φ_e。可采用固定共享低秩基底、聚合可训练系数，避免直接分别平均 LoRA 两因子的附加误差。保留区域模型，云协调推理，区域模型内生轻量 Medusa 风格多步头 q_j 从目标模型内部特征起草；不部署独立完整草稿 LLM。共享头与一次特征传输重建 q_j 的费用需计入。边缘仅持有局部概率，云协调标量或请求全向量；不承诺 DP/密码学隐私。

正式目标 P(v|s)=Σ_e w_e p_e(v|s)，非负归一化权重，p_e=softmax(f_{θ_0,φ_e})。一般不等于参数平均模型的分布。正式序列为 ∏_t P(y_t|s,y_<t)，推理固定快照、共享词表、可靠节点、已确认前缀与正确候选条件分布。若仅用单个全局模型，普通 Medusa 已足够；混合集成的任务质量价值尚未证明。

HDRR（Hierarchical Distributed Residual Resolution）：给定层级树、各节点概率、权重、草稿分布、上下文和长度 γ，选择通信、验证、修正及回退协议 π，优化 E[T_draft+T_verify+T_comm+T_repair]/E[L_commit]，约束相对于指定混合目标的序列 TV=0（理想算术）。greedy、温度/top-p 施加顺序、有限精度和 EOS 尚未完整审计，不得宣称对原模型无损。

#### 2. 数学矛盾与构造反例（T）

标准 SD 接受 min(1,P(x)/q(x))；拒绝从归一化 R(v)=[P(v)−q(v)]_+ 采样。定义 d_e=p_e−q，g_e=[d_e]_+，m_e=Σg_e=TV(p_e,q)，G=Σ_e w_e g_e，M=Σ_e w_e m_e，Z=ΣR=TV(P,q)。正部函数非线性：[Σ_e w_e d_e]_+≤Σ_e w_e[d_e]_+，通常不等。J=M−Z≥0 是跨节点抵消增益（TV 凸性，非新数学）。各坐标正权节点的残差同号时 J=0。

| 三词示例 | A | B | C |
|---|---:|---:|---:|
| p1 | .70 | .10 | .20 |
| p2 | .10 | .60 | .30 |
| q | .40 | .30 | .30 |
| 等权 P | .40 | .35 | .25 |

d1=(.30,−.20,−.10)，d2=(−.30,.30,0)；G=(.15,.15,0)，M=.30；R=(0,.05,0)，Z=.05。错误局部残差归一化 (.5,.5,0)，正确全局残差 (0,1,0)。若用全局接受规则但错误残差修正，输出 (.425,.325,.25)，并非 P。两局部接受率均 .70，全局 .95，J=.25，仅构造数学例子。

#### 3. HieraSpec 协议与理论（T/H）

1. 内部头生成 γ 个候选，记录真实条件 q_j；各区域批量验证，上传 p_e(x_j) 标量，云聚合 P(x_j) 并执行标准接受判定。
2. 首次拒绝时各区域计算 m_e；云按 w_e m_e/M 选择节点，由该节点按 g_e/m_e 提出 v。
3. 请求各节点 d_e(v) 标量，聚合 R(v)、G(v)，以 R(v)/G(v) 接受；否则重试。零质量节点不抽取。
4. 最多 K 次失败后全量上传必要概率、直接从 R/Z 精确采样。全候选接受时 bonus token 可按 w 抽节点再从 p_e 抽样。首次拒绝后丢弃后续候选，维护 KV；节点掉线不允许未经定义改变目标。

证明：提议 Q(v)=G(v)/M；联合接受质量 Q(v)R(v)/G(v)=R(v)/M；单次成功率 Z/M；条件输出 R/Z。因此 min(P,q)+R=P。逐位置依据实际条件 q_j 和独立随机数、标准 SD 停止规则推广到完整序列；固定多步头的 q_j 可不依赖候选历史，相关候选/树/隐节点混合须重新证明。K 次失败后仍使用正确残差回退，保持精确。

Z>0 时 E[N|reject]=M/Z，E[N]=M≤1；Z=0 时不触发修正，E[N]=0（不能套极限）。有限 K≥1：F_K=Z(1−Z/M)^K≤M/(K+1)·(K/(K+1))^K；有上限平均尝试数 M[1−(1−Z/M)^K]。强抵消可造成巨大条件长尾，平均低不保证 P95/P99 或低 RPC。

层级包络：合法树分区 𝒫，组权重 w_B、p_B=Σ_{i∈B}w_i p_i/w_B，M(𝒫)=Σ_B w_B TV(p_B,q)，合并组由凸性可得 M_client≥M_edge≥M_cloud=Z。粗分组降低重试，但组内聚合可能增加通信；仅保留客户端适配器并执行概率聚合时才有客户端级包络，不能将参数平均区域模型混同于 p_B。曾拟研究最小化通信+计算+残差成本的合法树分区及通信下界，尚未完成；HC/R01 已提供标准树 DP 代理，HCR 已提供同类包络，不能以此另立项。

P：4 区、128000 词、γ=5 且六个位置、FP32 全概率上传 4×6×128000×4=12,288,000 bytes≈12.29 MB/轮；标量验证省概率传输，但必须计入 q 重建、特征传输、RPC、拒绝回退、全模型前向及 KV，不能当实际加速。

#### 4. 历史模拟与归档复核

E-H：历史报告称 CPU 三词、双节点、1,000,000 次独立样本，seed=20261124，另有 5,000 组随机四节点层级单调性检查。原始脚本、stdout、环境、命令与 CI 未保留，故下列历史数字未独立审核，不当作真实 GPU 结果。

| 指标 | 理论 | 历史报告 |
|---|---:|---:|
| A | .400000 | .400270 |
| B | .350000 | .350205 |
| C | .250000 | .249525 |
| 接受率 | .950000 | .949941 |
| 无条件平均修正 | .300000 | .299031 |
| 拒绝后平均尝试 | 6 | 5.973571 |
| 层级单调性违反 | 0 | 0/5000 |

本次归档另实际执行 JS V8 三词局部节点包络抽样（LCG seed 20261124，N=1,000,000，未测 wall time；不同随机数实现不能要求历史数字一致）：
{"n":1000000,"out":[399903,350466,249631],"accept":949738,"attempts":301778,"prob":[0.399903,0.350466,0.249631],"alpha":0.949738,"mean":0.301778,"conditional":6.0040985237356255}
可复算脚本保存为 hieraspec_probe.js 后运行 node hieraspec_probe.js（这里直接用 JS 工具执行同一逻辑）：
~~~javascript
let s=20261124>>>0;
function u(){s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296}
function draw(w){let x=u()*w.reduce((a,b)=>a+b,0);for(let i=0;i<w.length;i++){x-=w[i];if(x<0)return i}return w.length-1}
const p=[.4,.35,.25],q=[.4,.3,.3],g1=[.3,0,0],g2=[0,.3,0],G=[.15,.15,0],R=[0,.05,0];
let n=1000000,out=[0,0,0],a=0,k=0;
for(let i=0;i<n;i++){let x=draw(q);if(u()<Math.min(1,p[x]/q[x]))a++;else{do{const e=draw([.5,.5]);x=draw(e===0?g1:g2);k++}while(u()>=R[x]/G[x])}out[x]++}
console.log({n,out,accept:a,attempts:k,prob:out.map(v=>v/n),alpha:a/n,mean:k/n,conditional:k/(n-a)});
~~~
仅单步正确性 sanity check，无真实模型、三种子、多步或层级随机测试的本次复现；LCG 非生产 RNG。

#### 5. 失败方向、反例、强基线和跨团队去重

本轮初筛否决：HFL+Medusa/EAGLE 简单拼接；仅按反馈联邦训练预测头（在线 SD/接受率优化已有）；按网络动态候选长度（已有调度）；Top-K 分层上传（SpecFed 高度接近）。未执行 GPU 验证。

CRITICAL：P=Σw p_e 本身可逐位置 e~w、v~p_e 精确抽样，不必全节点前向；单次整块路由一般改变联合分布（A/R11 已有反例）。全量目标批量验证后仅在拒绝时全向量上传也为强基线。同信息平坦系统可执行相同树分组和包络协议；SD 没有反向改变 HFL 学习。PRE-06、A/R06、A/R10、A/R11 已覆盖相关抵消/混合/耦合；HC/R01 已明确树切面 DP、批内组覆盖；team-hcr-sd/R01 完全覆盖本轮局部正包络数学，并拓展不同草稿分布及块后验。HieraSpec 仅有限 K 回退上界和明确反例可作为增量数学记录，不能当新 Idea。

MAJOR：同模型/同号残差无 J；Z 极小尾延迟高；掉线/快照改变影响正确性；云已有全部模型时通信优势消失；高 RTT 标量多次 RPC 可能更慢；共享预测头特征成本未知；区域混合集成任务质量未证。禁止把正部凸性或已有拒绝采样原理当原创；无核心新前提前不投入重复 GPU 实验。

#### 6. 文献检索与查重（历史线索，投稿前复核原文）

检索日期 2026-11-24；关键词 hierarchical federated learning, federated speculative decoding, distributed exact speculative sampling, ensemble speculative decoding, signed mixtures rejection sampling, remote sampling communication complexity。来源 PMLR、ACL Anthology、arXiv、NeurIPS、IEEE 索引等。历史报告称查看基础原文和部分近邻正文，归档未逐条复核，盲区包括未公开/未索引工作、付费全文、概率论/通信复杂度。无商业文字查重，规范引用与独立撰写不能解决内容重复。

七篇起点：[Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html)、[SD Survey](https://aclanthology.org/2024.findings-acl.456/)、[EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)、[Medusa](https://arxiv.org/abs/2401.10774)、[SpecFed](https://arxiv.org/abs/2604.25777)。标准精确接受/修正、内部特征/动态树、多头和概率聚合已有；Medusa 宽松接受与精确随机采样区别明确。

| 近邻历史检索链接 | 重合与待核查差异 |
|---|---|
| [SpecFed](https://arxiv.org/abs/2604.25777) | 多节点概率聚合、Top-K 重构，当前区别为精确残差；高风险 |
| [CoS](https://arxiv.org/abs/2502.01662) | 多模型联合投机验证；内生头与信息局部性待比对 |
| [DISCO](https://arxiv.org/abs/2610.35957) | 历史报告记载精确 SD、验证端重建草稿分布及消除词表通信；多目标残差待查 |
| [Two-Stage Rejection Sampling](https://digit-uat.eurekamag.com/research/109/621/109621528.php) | 分布式精确采样；全文及层级/词表分区区别未核验 |
| [Simulating Signed Mixtures](https://arxiv.org/abs/2401.16828) | 正分量提议、负分量抵消、分层优化直接威胁核心数学；高风险 |
| [Remote Sampling](https://www.mdpi.com/1099-4300/21/1/92) | 分散参数精确采样；多步 SD 约束待查 |
| [FedHLM](https://doi.org/10.1109/JIOT.2025.3618095) | 联邦混合模型和分层聚合；无独立草稿/精确性待查 |
| [LayerSkip](https://aclanthology.org/2024.acl-long.681/) | 提前退出自投机，非独立创新 |
| [Lossless Hierarchical SD](https://arxiv.org/abs/2601.05724) | 共享库 A/R11 已列；须区分模型层级与网络层级 |

其他历史检索名称（仅线索）：HierFAVG、Personalized Client-Edge-Cloud HFL、Online Speculative Decoding、Speculative Ensemble、DIVERSED、LK Losses、Acceptance-Aware Draft Model Training、When Drafts Evolve。不要将未重新核实的日期、题名、机制和结论当投稿引用。潜在后续查重：分布式最大耦合/OT、signed mixtures、通信复杂度、分布式拒绝采样及概率复用。

#### 7. 原拟最小与完整实验（H，未执行）

1×40GB：Qwen2.5-1.5B-Instruct，扩展 Qwen2.5-7B-Instruct；8 逻辑客户端、2–4 边缘、固定低秩共享基底、2–4 轻量多步头、γ=2/4/6，通用对话/数学/代码/摘要。单卡切适配器只测概率、正确性和独立算子时间，不冒充并行加速。2×40GB 部署双区域真实并行验证，网络模拟 10–1000 Mbps、1–100 ms RTT。7B BF16 权重约14GB，不含激活/KV/优化器。后续至少三种异质性/时延、三种子、两尺度及真实公开任务，具体数据集和命令尚未落实。

公平基线 B0 全局 HFL AR（质量对照非同目标）；B1 同目标全量混合 AR；B2 逐位置随机目标节点精确 AR；B3 各区域普通 Medusa（目标不同须说明）；B4 集中式精确混合 SD；B5 SpecFed 风格 Top-K（近似需报告误差）；B6 CoS 风格联合 SD；B7 标量接受、拒绝才全量上传；B8 逐位置随机组件精确 SD；另需同信息平坦委托、预缓存、简单直接采样。B2/B4/B7/B8 不得删除。

消融：内部头、跨节点抵消、层级分组、局部包络、有限重试、IID/Non-IID、统一/区域模型。指标：TPOT、tokens/s、P50/95/99、bytes/token、消息/RPC、GPU time/forward 次数/显存、接受率/长度、M/Z/J、采样正确性和任务质量、训练/通信/失败回退总 wall time 及 CI。历史好结果：相对全量 SD 通信降≥90%，最强同目标精确基线效率提升≥20%，P95 不显著恶化，两模型三任务，层级优于公平平坦且混合有质量价值；仍必须满足共享 G5 总时间下降≥15% 或同成本有效工作+20%。原拟否决线：多个任务 J<3pp 且接受长度无明显提高；即使超过也不证明独创性。

#### 8. 评分、G1–G7、保留资产与下一步

历史主观评分：问题6.5×25%、机制6×25%、理论6.5×20%、价值8×15%、文献距离5×15%=6.375≈6.4/10（未达8）；跨团队重合不应上调。历史证据强度（非概率）数学可运行9、硬件可复现8、通信潜力8、端到端3、真实抵消2、HFL 不可替代性4（/10）。缺证据支持80%成功把握。

| G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|
| 未过：重复 | 未过：CRITICAL | 部分 T；greedy/精度未证 | 未过 | 未过 | 证据不足 | 部分 CPU 脚本，完整系统未过 |

裁决 Reject and Pivot：保留正部非交换反例、包络正确性、M/Z 长尾、有限 K 回退上界、树分区凸性及强基线。禁止以改名、普通头训练、动态分组、Top-K 压缩重启 PRE-06/A/R11/HC/HCR。后续若有新前提，先证明 HFL 与 SD 的双向因果及非退化学习、真实约束下不可被同信息平坦复制的增益；再做真实分布及通信下界的区分性实验。否则直接创造新问题。

**每轮协作规则：** 本研究对话每次收到新研究轮次前读取最新默认分支 README 全文、记录 SHA、学习其他团队全部成功和失败；结束追加本轮正负结果、反例、假设、证据、G1–G7。使用最新 SHA，保留他人记录，不强推；不是后台自动执行。

### team-hieracert/R01｜HieraCert-SD：分层不完备信息下的精确投机验证（HIISV）

- 日期：2026-11-24。独立第一轮研究在读共享库前完成；归档时已读初始 README（main commit f93a4c815028bcc3b7104df1e68e84785071bf85）及随后新增 HC/R01、team-hcr-sd/R01、team-himerge/R01；本轮追加前读取 README blob c8ee552a712ea53b0e4af1ee07d7b51fb8b5aa51。
- 裁决：原报告暂定保留，创新6.8/10；跨团队复核发现 PRE-09 高度相似，且平坦/普通 AR 可替代，修订为 **Reject and Pivot（保留局部数学资产，不立项）**。没有真实 LLM/GPU 结果，80% 成功把握未知。
- 固定流程：以后每轮开始读取最新默认分支 README 全文及新增相关文件、记录 SHA、对照全部失败库；结束追加正负结果、反例、代码、G1–G7，不覆盖他人。

#### 1. 文献探索及排除路线（历史检索，投稿前复核）

七篇起点：[Leviathan et al., SD 2023](https://proceedings.mlr.press/v202/leviathan23a.html)、[SD Survey 2024](https://aclanthology.org/2024.findings-acl.456/)、[EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)、[Medusa](https://arxiv.org/abs/2401.10774)、[SpecFed](https://arxiv.org/abs/2604.25777)。EAGLE 有轻量附加预测网络，依赖内部特征，符合本项目“不部署完整独立草稿 LLM”的定义，并非完全无附加模型参数。
排除：仅 HFL 分层训练预测头（个性化 FL/蒸馏）；边缘起草云验证（DSD/SpecFed）；直接优化接受长度（LK Losses/VSD/REAL Loss/Acceptance-Aware Draft Training）；模型更新后适配草稿（FlexSpec）；提前退出云验证（LayerSkip/split SD）；网络自适应候选长度（SpecRelay）。均为历史相关工作线索，不可用改名或常规损失拼接立项。

#### 2. 系统设定与问题（H）

个性化 HFL 子场景：客户端持有非 IID 数据 D_ei、参与区域及共享参数聚合。云掌握共享 Transformer f_θ、输出矩阵 W∈R^(V×d)、共享低秩投影 A∈R^(r×d)、内生候选预测模块 φ；边缘 e 保留私有输出适配器 B_e∈R^(V×r)，不运行完整 backbone。请求内模型、证书版本固定。云不能访问完整 B_e，边缘没有当前完整云 logits；此为部署约束而非隐私证明。令 h=f_θ(s), a=Wh, z=Ah，正式区域目标 l_e=a+B_e z，p_e=softmax(l_e)。云生成候选树 T；边缘利用云发送的验证信息和 B_e 逐节点决定正确 token，沿匹配分支继续，缺失时提交正确 token 并停止本轮。目标是该个性化模型，而非未经修改的原始基座。

HIISV 定义：给定 HFL 层级、上述参数分布、前缀、候选树、网络带宽/RTT/边缘显存预算，设计跨层消息协议 π 与候选树，min E[T_round]/E[committed tokens]；计 draft、verify、network、commit、证书维护和训练摊销。约束：greedy 与同一区域 AR 逐 token 一致，随机采样与同一正式目标序列条件分布一致，不上传完整 B_e，满足通信与显存预算。B_e=0、云缓存完整 B_e、边能运行全模型或通信免费时问题退化；中间层个性化不自动满足可加 logits 分解。

#### 3. 计算矛盾及反例（T/P）

SD 标准接受 min(1,p/q)，拒绝按归一化 (p−q)_+ 采样；在此场景目标信息分散于云边。全局 Top-K 不保证覆盖区域最优：
a=(4,3.8,3.5,1,0)，区域修正 δ=(−.5,−.1,1.2,.2,0)，l=(3.5,3.7,4.7,1.2,0)。只传全局前二 A/B 会误选 B，正确 C 为全局第三。SD 树节点并行使逐节点通信负载放大，但普通 AR 同样存在这一问题。

载荷例（P）：V=100000,d=4096,r=16,N=32 个验证节点，FP16 数值、32-bit token ID，完整 logits 6.4MB，完整 hidden 262144B；若每节点 K=16，证书载荷 N(2r+6K)=4096B。K=16 是假设，未实测。一般稀疏比 hidden 省载荷的条件 K<(d−r)/3；上述 d,r 时 K<1360。未计协议头、版本、数值保护、RPC、KV、证书更新、计算与回退，不能当速度。隐藏状态基线要求边缓存 W，须核算存储、计算，也允许合法压缩 hidden。

#### 4. 初步机制及数学结论（T/H）

若 ||b_ev||₂≤R_e，ε=R_e||z||₂，由 Cauchy–Schwarz 得 L_v=a_v−ε≤l_v≤a_v+ε=U_v。令 S={v:U_v≥max_u L_u}。云发 z 及 {(v,a_v):v∈S}，边缘计算 S 内真实修正并 argmax；稀疏过大则传完整 a+z，或边缓存 W 时传 h。云依次做内生起草、目标树批量前向、区间筛选、选择稀疏/稠密消息，边验证、提交，云更新合法 KV。不得把未经验证的候选提交。
- 精确性定理：若真实最大值索引 v 不在 S，则存在 u 使 l_v≤U_v<L_u≤l_u，矛盾。因此所有最大值索引在 S；统一 tie-break 得精确 greedy。在每个访问前缀重复、树掩码/位置/版本/缓存一致条件下按序列归纳得到 greedy 等价。实际 GPU 需保护浮点边界，不保证不同算子逐位等价。
- 有限最小性：在只知独立区间的模型下，若 U_v>max_{u≠v}L_u，取 l_v=U_v 及其余 l_u=L_u 即使 v 唯一最优。任何必须包含最优的集合都不能安全删 v。这不是一般编码/交互通信下界；相关低秩结构、tie-break 等可改变结论。
- 间隔：S={v:a_v≥a_max−2ε}；top1/top2 margin Δ>2ε 时 |S|=1；平坦 logits 或大 ε 时可达 V。
- 分组证书（T/H）：公共 token 分组 V_g，公开中心 c_eg、半径 R_eg，||b_ev−c_eg||≤R_eg；区间 a_v+c_egᵀz±R_eg||z||。若 B_e=Σ_i w_iB_ei，w_i≥0且Σw_i=1，同组客户端中心/半径按相同权重平均，由三角不等式保持有效。边缘掌握 B_e 时直接重算可能更紧，且平坦线性聚合也可执行，不是 HFL 独有原语。摘要泄露、传输和版本成本待审计。
- 学习假设（H）：Σ_e ω_e[L_task,e+λL_draft,e+μR_cert,e]，R_cert=Σ_v sigmoid((U_v−max_u L_u)/τ) 近似候选规模。需比较简单范数正则、margin 优化及质量匹配；防 B_e→0 人为制造压缩。未验证。
- 随机采样扩展（T/H）：温度 T>0，独立 Gumbel g_v，argmax(l_v/T+g_v) 等价温度 softmax。云对 a_v/T+g_v 构区间、误差 ε/T，边精确取最大值；在实际访问前缀独立抽目标随机变量，若 token 在候选树继续，否则提交后停止。候选构造不得偷看目标随机性。此 target-first tree sampling 可按条件分布归纳，非最大耦合 SD，接受率可能更差。top-p、实际 RNG、有限精度及实现未完成，不宣布 G3 通过。

#### 5. 本次归档实际重跑的 CPU 合成实验（局部可复算）

NumPy 2.3.5，CPU，seed=20261124，400×4096 固定标准正态 logits，每 ε 生成独立逐 token uniform(−ε,ε) 修正。该修正不是实际 HFL 训练出的低秩 B_e z。以下实际运行脚本可直接保存为 probe.py，在安装 numpy==2.3.5 的环境 python probe.py 执行：

~~~python
import numpy as np
rng = np.random.default_rng(20261124)
a = rng.standard_normal((400,4096))
for eps in [.05,.25,.5,1.,2.]:
    delta = rng.uniform(-eps,eps,a.shape)
    keep = a >= a.max(axis=1,keepdims=True)-2*eps
    y = a+delta
    count = keep.sum(axis=1)
    print(eps,count.mean(),np.percentile(count,95),
          np.count_nonzero(y.argmax(axis=1)!=
                           np.where(keep,y,-np.inf).argmax(axis=1)))
print(np.__version__)
~~~

实际输出（ε、均值 K、P95 K、不一致数；本次重跑与第一轮历史报告完全一致）：

~~~text
0.05 1.4525 3.0 0
0.25 5.5175 15.0 0
0.5 24.095 60.0 0
1.0 249.7225 488.0 0
2.0 2648.2125 3275.05 0
2.3.5
~~~

2000 次对比零错误；大扰动导致集合急剧增大。单种子、无 CI、无真实 LLM/HFL/网络/GPU 时间；不能据此推断真实加速或成功概率。仅局部复算证据，不是完整 G7。

#### 6. 原创性审查与跨团队去重

- **PRE-09 Federated Certifiable Computation** 已探索共享 logits+区域残差、Gumbel-Max/界跳过区域；历史6区平均访问1.70、残差占10%时总收益7.2%非本队实测。HieraCert 仅改为单区域个性化目标的词表筛选，区间原语非新，不能证明机制创新。
- A/R07 低秩接口有表达力/通信局限；A/R08 区域编译受全局平坦对照替代；HC/R01、HCR-SD、HiMerge-SD 主要为概率混合、组选择或残差包络，本轮为单区域私有 logits 并非混合目标，但同样缺双向必要性。
- [Threshold Algorithm, Fagin et al.](https://doi.org/10.1016/S0022-0000(03)00026-6)：部分得分及边界下的精确 Top-K 是直接数学近邻；不能把区间剪枝包装为新定理。
- [SpecFed](https://arxiv.org/abs/2604.25777)：历史报告称多节点概率聚合、Top-K 截断/重构误差；此处为私有区域目标的精确验证，需核实是否已有等价协议。
- [HierFAVG](https://arxiv.org/abs/1905.06641)、[FedRep](https://proceedings.mlr.press/v139/collins21a.html)：分层聚合及共享表示/个性化头，模型结构非新。
- [LayerSkip](https://aclanthology.org/2024.acl-long.681/)、[SpecInfer](https://arxiv.org/abs/2305.09781)、[Sequoia](https://arxiv.org/abs/2402.12374)：内生起草和候选树验证已有先例。
- 其他历史检索线索：FedSA-LoRA（共享私有低秩因子）、HAT（U形模型切分和协同 SD）、Privacy-Aware Split Inference（WAN/Lookahead/精确 greedy）、DSSD、FlexSpec、VocabTrim、FlashSampling、distributed top-k aggregation、robust maximum inner product search、communication complexity、certified inference、distributed sampling。缺可确认链接者仅作检索线索，不伪造引用。原报告对 HAT/Privacy-Aware Split Inference/Threshold Algorithm/SpecFed/FedSA-LoRA 均判高撞题风险；DSSD 中高；FlexSpec/FedRep/VocabTrim/SpecInfer/Sequoia 中等。
- 2026-11-24 历史检索关键词：Hierarchical/Personalized FL、Federated/Distributed Speculative Decoding、Acceptance Length Optimization、Draft Training、Split Inference、Private Output Head、Threshold Algorithm、Certified Top-K、Federated LoRA、Shared Private Factors、Vocabulary Pruning、Exact Sampling；历史来源 arXiv、IEEE、Springer、PMLR、OpenReview、ACL、期刊索引。本次归档未逐条重新查原文；所有链接/年份/重合点投稿前核实；有未索引、未公开及数据库算法领域盲区；无正式文字查重，不声称内容已独创。

#### 7. 预实验与完整实验（H，未执行）

1–2×40GB NVIDIA，客户端串行/分批模拟。最小 Qwen2.5-1.5B 冻结主体，2边缘/8客户端、rank8/16；后续7B、4–8边缘/16–32客户端、rank8/16/32/64，Medusa/EAGLE-3。7B BF16 权重约14GB但须计激活/KV/训练状态。数学、代码、通用指令不同区域，公开真实数据、训练测试分离、至少3种子/3异质性或时延配置、2模型尺度。先不训练 SD，收集≥10000真实 token 位置的区域质量、margin、修正范围、最优 token 全局排名、证书K均值/P95、hidden字节与算子时间。若证书或任务质量不达标停止；当前核心否决后不建议投入 GPU 重做，除非出现改变前提的证据。

强基线：HFL个性化AR（含证书AR）、HFL+Medusa完整logits、完整hidden（缓存W）及压缩hidden、固定Top-K（近似，质量单列）、阈值验证+精确回退、HFL+EAGLE-3稠密、集中式个性化SD（放宽约束的参考）、同信息/同边缘委托/同缓存/同证书的平坦FL+SD、随机采样时直接Gumbel与普通target-first tree及最大耦合。控制同一目标/草稿/树/精度/预算；消融统一范数、分组、层级聚合、直接重算摘要、证书损失与常规范数正则、无SD、无回退。指标：端到端延迟/吞吐/TPOT/P95、接受长度、计算量、通信含RTT/回退、质量、训练摊销、GPU峰值和CI。
原报告好结果：greedy 100%一致，采样序列正确；较最强基线延迟降≥20%、相对hidden通信降≥50%、吞吐增≥15%、P95不恶化、个性化质量无显著下降。仓库G5额外强制真实总wall time降≥15%或同成本有效工作增≥20%，计训练、推理及全部开销，不能用理论通信比替代。

#### 8. 失败、G1–G7 与后续

CRITICAL：PRE-09 高度重合；同信息平坦和普通 AR 能使用同样证书（没有证明 HFL 或 SD 双向不可替代）；证书过宽退化；完整 hidden 是不可忽略的强基线；云仍计算全量 W h。MAJOR：区域草稿接受长度过短；摘要维护及可能泄露；参数更新证书过期；中间层适配不可直接分解；浮点误差、tie-break；正则可能破坏个性化；target-first 采样可能降低接受率。以上均无真实系统证据消除。

原主观等权评分：问题8、机制6.5、理论6.5、价值7.5、距离5.5，均值6.8；共享库去重后风险更高，不上调。原复现难度估计（非实测）：输出适配3/10、HFL模拟4、证书3、Medusa5、EAGLE-3/树KV7、端到端8。无依据支持80%好结果概率，不估计发表率；TMC/TPDS/TNNLS 仅可能方向。

| G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|
| FAIL，6.8且重合 | FAIL，平坦/AR可替代 | 局部greedy定理成立；随机及精度实现未完成，FAIL | FAIL | FAIL，仅载荷代理 | FAIL，未知80% | FAIL，仅CPU局部可复算 |

**Reject and Pivot。** 保留反例、区间充分性/限定最小性、margin关系、通信分界、实际CPU数据和退化规律；不得换名复活 PRE-09 或 HieraCert。原拟下一步为利用低秩相关性超越独立区间、证明通信下界及近下界协议、真实1.5B预实验、跨领域查重；现仅在新证据打破平坦/AR替代性时考虑重启。下一轮从全部跨团队最新成果开始，优先创造新的学习对象及真正双向必要机制，不直接投入完整 EAGLE-3 训练。

- team-hiercouple/R01 写入前最新 README blob SHA：ede10bcf2bb7869a637f2e81792b172c88a43042（文件 SHA，非 commit SHA；保留此前所有并发追加记录）。

### team-hiercouple/R01｜HierCouple-SD：分层残差耦合无损自投机解码

- 日期：2026-11-24；独立研究对话第一轮。原研究结束后同步共享库，已阅读 A/R01–A/R13、PRE-01–PRE-11、HC/R01、team-hcr-sd/R01、HiMerge-SD、HIERASPEC/R01 等并发新增记录。
- 裁决：**CRITICAL / Reject and Pivot**。原报告暂定保留、创新度6.6/10；共享库交叉审查发现核心与 HCR-SD、HIERASPEC 高度重合，且 A/R11 已有关键负证据。保留全部数学、历史数值、反例、风险及实验计划，不批准独立立项。
- T=明确假设下推导；E-H=历史对话报告的CPU模拟数字，原始脚本/日志/环境当前不可审核；H=未验证假说；P=代理或理论成本，非实测延迟。无GPU/真实网络结果，无完整R级证据。

#### 1. 场景和计算问题（T/H）

个性化 client-edge-cloud HFL：客户端 i 私有非IID数据 D_i，F_i=E_{D_i}ℓ，边缘 F_g=Σ_i a_{i|g}F_i，云 F=Σ_g W_g F_g，w_i=W_g a_{i|g}。共享冻结LLM基座，保留不同区域/本地LoRA/Adapter，模型实例可部署在边缘而非终端。云全局模型附Medusa风格预测头 q_j=softmax(W_o H_j(h))，按实际q_j采样单路径多token候选，不另部署独立草稿LLM；EAGLE也有轻量草稿组件。首次拒绝后丢弃后缀并回滚KV，树状多路径尚未证明。

固定前缀 s、版本、权重、词表：p_i(v|s)=softmax(f_i(s))_v，区域 p_g=Σ_i a_{i|g}p_i，正式目标 p=Σ_g W_g p_g。概率混合不是普通HFL参数平均模型；联合推理质量未证。云持有q、权重、候选，边缘持有区域概率；跨域完整向量昂贵。无额外隐私保证，prompt与模型预测交换可能泄露。

标准SD：X~q，接受 min(1,p(X)/q(X))；拒绝按 R(v)/ρ 补采，R=(p−q)_+，ρ=ΣR=TV(p,q)。核心矛盾：[Σ_i w_i(p_i−q)]_+≠Σ_i w_i[p_i−q]_+。手算例：q=(.4,.3,.3)，p1=(.7,.1,.2)，p2=(.1,.5,.4)，等权p=q；局部最大耦合接受率均70%，联合100%。由TV凸性，Δ=Σ_i w_i TV(p_i,q)−TV(p,q)≥0，局部独立验证可能损失抵消收益；集中验证全量通信昂贵。此不等式已知，非新定理。

计算问题：给定层级树、分布式p_i、q、权重、带宽、RTT及通信预算，设计局部计算、标量查询、随机采样、有限轮通信及精确回退协议，最小化期望计算+网络+回滚时间，约束 Y|s~p 且到达候选位置接受率=1−TV(p,q)。同信息平坦可委托同样区域计算，必须公平比较。

#### 2. 核心机制与证明（T）

1. 全模型并行验证候选 x_1…x_γ；区域仅返回候选token对应聚合标量p_g(x_j)，云端聚合p(x_j)，顺序执行标准精确接受检验。
2. 拒绝后区域计算 d_g(v)=W_g(p_g(v)−q(v))，S(v)=Σ_g[d_g(v)]_+≥R(v)=[Σ_g d_g(v)]_+；m_g=Σ_v[d_g(v)]_+，M=Σ_g m_g。云按 m_g/M 选区域，区域按 [d_g(v)]_+/m_g 采样Z，提议分布S/M；对Z查询各区域标量d_g(Z)，以R(Z)/S(Z)接受，否则重试。零质量区域不抽；ρ=0无初次拒绝。
3. 正确性：一次成功且输出v的概率=(S/M)(R/S)=R/M，条件分布R/ρ；与初次接受质量min(p,q)相加即p。固定快照、准确概率、独立随机数及真实条件q_j下，对已提交前缀归纳可保持完整单路径随机序列分布。不能宣称相同seed逐位一致；greedy argmax、temperature/top-p顺序、有限精度、EOS和KV实现待审计。
4. 若ρ>0，N|reject~Geom(ρ/M)，E[N|reject]=M/ρ，摊销E[1_reject N]=M；ρ=0时实际次数为0，即使M>0。最多K次失败后必须从真正R/ρ精确回退，不能从p补采。P(reject and fallback)=ρ(1−ρ/M)^K≤M K^K/(K+1)^(K+1)，K≥1，由t=ρ/M最大化t(1−t)^K得到。并行预采K个修正候选并按固定顺序选首个成功者可减串行RTT，须计批量查询。
5. M_leaf=Σ_i w_i TV(p_i,q)≥M_edge=Σ_g W_g TV(p_g,q)≥M_root=TV(p,q)。任意覆盖叶节点的不重叠树切面C有 M_C=Σ_{u∈C}W_u TV(p_u,q)。上移切面减包络但可能增加全向量通信。最优在线切面、通信下界、学习—验证联合优化均未完成；HC/R01已有标准树形可加代理DP。
6. 草稿q的下行通信不可忽略。共享W_o、数值和采样设置完全一致时，云下发头特征z，边缘重建q=softmax(W_o z)。d=1536、V=151936时元素数约1/99，未计重复输出投影计算、数据类型、报头、局域网传输，非实测加速。

#### 3. 历史CPU模拟（E-H）及解析值（T）

q=(.40,.30,.20,.10)，四等权节点 p1=(.70,.10,.10,.10)，p2=(.20,.40,.30,.10)，p3=(.10,.60,.20,.10)，p4=(.50,.15,.25,.10)，分组(1,2)/(3,4)。p=(.375,.3125,.2125,.10)，ρ=.025。历史报告每方法100万次、seed=20261124、K=4，均精确全量回退。无原脚本/输出/环境可上传，按E-H转录，不认定独立复现。

| 指标 | p包络 | 叶节点残差 | 边缘残差 |
|---|---:|---:|---:|
| M（T） | 1 | .2375 | .075 |
| 报告草稿接受率 | 97.50% | 97.50% | 97.50% |
| 拒绝后最多4次平均尝试（E-H） | 3.846 | 3.412 | 2.414 |
| 拒绝后实测回退率（E-H） | 90.22% | 64.09% | 19.95% |
| 理论回退率（T，约） | 90.37% | 64.09% | 19.75% |
| 经验输出TV（E-H） | .000519 | .000314 | .000193 |

历史边缘输出=(.374807,.312530,.212639,.100024)，条件回退相对叶节点约降68.9%。人为构造强抵消，不外推LLM/网络/端到端收益；经验TV差异为有限采样波动，非正确性差异。理论值可按公式复核，历史随机数字需独立重跑。HCR及HIERASPEC各自JS检查不是本团队历史原始日志。

#### 4. 失败经验与跨团队去重

- A/R11 FSV已覆盖混合目标、局部/全局TV凸性、逐token区域路由，平坦配对匹配或胜出。PRE-06区域混合、A/R06有符号差值及残差耦合、A/R10聚合抵消依赖为前序限制。
- HC/R01已否决随机验证切面并给出代理DP及批内覆盖反例。本轮全局最大耦合后对拒绝残差选提议组，不等于随机验证组。但 team-hcr-sd/R01 已提出同样的分散正残差包络、几何摊销、精确回退、任意切面，并扩展异构q_i和块候选后验；本团队固定共享q为其特例。HIERASPEC/R01同样推导有限K回退上界，HiMerge-SD提供稀疏q支持集内外分解的另一强通信基线。多个团队独立发现不构成多个创新。
- CRITICAL：同信息平坦可委托同样区域残差计算和采样；HFL学习没有不可替代性，SD未反向改变HFL学习。普通分布式ensemble亦可用此机制。
- CRITICAL：最强无SD基线先抽 I~Categorical(w_i)，再抽 Y~p_I，即精确单步p，无需全模型前向；整块固定目标索引改变联合分布（A/R11）。逐位置正确路由，公平计KV同步/补算；全模型验证SD未证实胜过它。
- 相同p_i或逐token偏差同向时各层M相同；真实HFL相似数据聚类可能使组内偏差同向，人造互补不构成证据。
- ρ很小M较大时条件重试长尾；多RTT可能比一次全量传输慢；回退非零，不能声称永远不传词表级向量。
- 标量验证+首次拒绝全量回退是简单强精确基线；下行q、区域聚合、额外输出投影、训练与回滚须计入。
- 已排除仅换Medusa、增加边缘层、动态长度、Top-K压缩、切面或普通联邦训练。经典拒绝采样、TV凸性及混合采样非新定理。

#### 5. 近邻文献、来源及检索盲区

原轮检索2026-11-24：client-edge-cloud HFL; personalized HFL; hierarchical federated LLM LoRA; distributed speculative decoding exact sampling; federated inference residual sampling; distributed maximal coupling mixture distributions; signed mixture exact sampling; hierarchical SD verification; edge-cloud SD latency。来源arXiv、PMLR、ACL、IEEE、NeurIPS、机构/出版社。归档补核部分近邻原始来源页面（非全面公式审查）：
- [SpecFed](https://arxiv.org/abs/2604.25777)（2026）：历史报告称多模型联合验证、Top-K概率重建；精确/近似区别待全文复核。
- [DISCO: Distributed LLM Inference via Communication-Efficient Speculative Decoding](https://arxiv.org/abs/2610.35957)（2026-10-25）：精确低通信SD；须比对分散混合目标与单目标、复制草稿及残差协议。
- [Robert & Stoehr, Simulating signed mixtures](https://wrap.warwick.ac.uk/id/eprint/192370/)（2025）：正负成分配对、分层拒绝采样，核心数学高度相关。
- [Coupling without Communication and Drafter-Invariant Speculative Decoding](https://arxiv.org/abs/2408.07978)（2024预印本/2025会议）：无通信耦合，相关共享随机数基线。
- [SpecTr](https://arxiv.org/abs/2310.15141)（2023）：最优传输耦合。
- [PipeSpec](https://aclanthology.org/2025.findings-acl.669/)（2025）：多模型分层异步流水线。
- [Speculation at a Distance](https://arxiv.org/abs/2606.25091)（2026）：网络RTT性能边界。
- 新增归档线索：[Fast collaborative inference via distributed speculative decoding](https://doi.org/10.1016/j.jiixd.2025.12.008)（具体出版日期待核），TSLT低通信策略。
- 基础七篇：[Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html)、[SD survey](https://aclanthology.org/2024.findings-acl.456/)、[EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)、[Medusa](https://arxiv.org/abs/2401.10774)、SpecFed见上。
- 其他历史线索：HierFAVG、PHE-FL、CHPFL、HRL-FLLM、LayerSkip、SpecInfer、Block Verification、Multi-Draft Speculative Sampling、PipeSD、List-Level Distribution Coupling；未逐条核验者仅作检索线索，另见共享库§5及其他团队的CoS、远程采样等近邻。
- 盲区：未公开投稿、付费全文、未索引论文、分布式采样通信复杂度、有符号测度。无正式文字查重；规范引用和独立撰写不替代机制去重。

#### 6. 原实验计划与否决标准（H，未执行）

1–2×40GB NVIDIA：Qwen2.5-1.5B/3B，4–8逻辑客户端、2–4边缘组，共享基座不同LoRA、2–4内生预测头；可扩展7B（BF16权重约14GB，不含激活/KV/训练）。原拟200–500 prompt、数千真实token位置，比较相似/随机/互补误差分组，测η_edge=1−M_edge/M_leaf、η_global=1−M_root/M_edge（零分母特判），对比普通全局与联合模型质量，再离线采样、实测网络校准。逻辑节点非真实物理并行。完整实验需两尺度、三异质性/延迟设置、三种子、公开任务与CI。

原拟基线B0普通全局HFL AR（目标不同，仅质量参照）；B1联合全量AR；B2按模型索引精确采样AR；B3 HFL+Medusa全量SD；B4 SpecFed Top-K近似；B5标量验证+全量回退；B6 p包络；B7局部最大耦合混合；B8 DISCO思想多模型适配；B9本方案；新增B10同信息同区域委托平坦、B11 HiMerge稀疏草稿精确残差（接受率/草稿分布差异须公平）。消融区域残差、切面、批量重试、特征重建、个性化。指标每token字节/RTT、TPOT/tokens/s、P95/P99、GPU时间/FLOPs、总训练+推理耗时、KV/显存、接受长度、正确性、下游质量。

原拟好结果：最强同目标精确基线跨广域字节降≥50%、端到端延迟降≥20%、吞吐升≥25%、尾延迟无明显恶化；必须额外满足共享G5质量/预算匹配下总时间降≥15%或有效工作增≥20%。真实抵消弱、联合质量无优势、平坦/普通混合采样/简单基线匹配或更好即否决。核心重复，现不建议投入GPU重做，除非有改变核心前提的新证据。

#### 7. G1–G7、保留资产及后续

原主观分项：问题7.5×25%、机制6×25%、理论6.5×20%、价值7×15%、已有距离6×15%，合计6.625≈6.6/10；跨团队去重后原评分不代表当前原创性，不虚构新分数；无80%成功把握依据。

| G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|
| FAIL：核心重复且原分不足8 | FAIL：无双向必要性，平坦模拟 | PARTIAL/FAIL：随机采样局部T，greedy/精度/完整实现未审计 | FAIL：无真实LLM | FAIL：无实际总时间 | FAIL：无80%依据 | FAIL：历史原脚本/日志缺失 |

保留全局残差精确性、K次回退概率上界、ρ=0边界、下行q成本、历史四节点数字和强基线。原下一步假说为HFL训练最优分组与SD推理抵消最优分组冲突；共享失败证据说明不能仅重新分组、调度或普通候选头训练就重启。下一轮优先读取最新README全文并记录版本，遵循Supervisor-Skills，去重其他团队正负结果，寻找真正新的学习对象及双向不可替代计算规则。每轮追加正负结果、证据等级和裁决，不覆盖他人、不强推、不降低门槛。仅在收到新研究任务时执行，非后台工作。


### team-hca/R01｜HCA-SD / HCDCP：分层概率耦合与跨层抵消感知投机解码

- 日期 2026-11-24；独立 ChatGPT 研究对话 team-hca；初次读取完整 README blob c9a305f316ad5bb371e4ab3c731dd590a5e0b9ae，后续读取 9fc7bb02fa901f2cddaf3a05b6aba069ade2eeb8 及 a8d64203aed1f6fd46a0816f6caf670a2dd788da 的并发新增记录。
- **修订裁决 CRITICAL / Reject and Pivot**：原报告创新度 7/10，暂存未立项。共享库 A/R11 FSV 已覆盖混合目标、随机验证组、TV 凸性、逐位置路由且已有负结果；HC/R01 已有同样的树切面 DP，team-hcr-sd/R01、HIERASPEC/R01 已有相邻且更紧的正残差包络。不得改名重新立项。保留数学、失败方向、机制与实验计划供其他团队使用。
- T=数学推导，O=完整信息代理，H=假说。**本轮无任何实际 CPU/GPU 实验**，所有数值为解析例子，非真实模型结果；无代码、日志、CI 或独立复现证据。

#### 1. HFL 场景和对象（H）

客户端私有非 IID 数据训练个性化 LoRA，边缘区域聚合，云全局聚合；共享基础 LLM、词表与 tokenizer。有效参数 θ_i=θ_0+Δθ_i，不假设 LoRA 因子分别平均等于有效权重平均；经授权可由边缘代执行客户端模型，不要求每客户端有 GPU。云端 HFL 全局模型的已获得隐藏状态 h_t 供 Medusa 风格轻量多头 q_{t,k}=softmax(H_k(h_t))，初期单候选链、冻结目标；可替换 EAGLE-3/LayerSkip，不需独立完整草稿 LLM，但辅助参数和云主模型前向/KV 不免费。

给定前缀 s，正式目标特意定义为 p(y|s)=Σ_e w_e Σ_i w_{i|e}p_{e,i}(y|s)，各层权重归一，叶权重 w_i。概率混合不同于参数聚合模型 p_{Σw_iθ_i}，必须另证混合服务质量价值。仅用一个全局模型时本问题消失。节点持有各自模型与 KV，边缘协调，云持候选、权重和历史统计；跨层完整概率、隐藏状态与 KV 不可免费获取。允许前向、点概率查询、局部采样、聚合、验证、回滚。请求内版本、前缀、词表、温度/截断顺序须统一；prompt 共享需授权，不宣称天然隐私。

#### 2. 计算矛盾（T）

标准 SD：X~q，接受 min(1,p(X)/q(X))，拒绝从归一化 [p−q]_+ 补采；A=Σ min(p,q)=1−TV(p,q)。正部函数与概率聚合不交换：
R_global(y)=[Σ_i w_i(p_i(y)−q(y))]_+ ≤ R_local(y)=Σ_i w_i[p_i(y)−q(y)]_+。
异号偏差在全局抵消，先局部截断会丢失抵消。经典凸性，不是新定理。
三词 A/B/C：q=(.5,.3,.2)，p1=(.8,.15,.05)，p2=(.2,.15,.65)，等权 p=(.5,.15,.35)。偏差 (.30,−.15,−.15)、(−.30,−.15,.45)；全局残差 (0,0,.15)，局部正残差聚合 (.15,0,.225)。全局接受 .85；局部 .70/.55，加权 .625，相差22.5pp。错误混用全局接受与归一化局部残差，输出 (.56,.15,.29)≠p。完整一致的局部规则则保持精确。二元 q=(.5,.5),p1=(.9,.1),p2=(.1,.9)，等权全局接受1、局部.6。均为手算。
退化：单模型、分布全相同、逐词偏差同号、全量聚合免费。真实 HFL 区域可能同向而非互补。

#### 3. HCDCP 问题、HCA-SD 机制与命题（T/H）

Hierarchy-Constrained Distribution Coupling Problem：合法树切分 P 是覆盖全部叶且互不重叠的子树集合；W_g=Σ_{i∈g}w_i，p_g=Σ_{i∈g}w_i p_i/W_g（忽略零权组），p=Σ_g W_gp_g。根、区域、叶及不对称切分均合法。优化长期 E[总时间]/E[总提交 token]，含起草、验证、通信、补采、缓存同步及回退；显存/通信约束，正式自回归序列 ∏_t p(y_t|y_<t) 不变，训练推理全流程成本另按 G5。

机制：在当前候选随机性前依据已提交前缀/历史统计选切分，每个位置（包括 bonus）按 W_g 独立于当前及未来候选随机性抽 G，仅对 p_G 做完整精确 SD。Uq(X)≤p_G(X) 接受；否则正确残差补采、截断后缀、更新 KV；全通过则 bonus 从新抽组目标采样。不可按当前候选任意选组或整块固定组。

命题1：在合法因果路由、同快照/前缀、真实条件 q、独立随机性及精确补采下，P(Y=y|G=g)=p_g(y)，边缘化得 p(y)。每步完整历史条件正确才能归纳整个序列。预采样候选、停止/EOS、bonus、浮点及 top-p 需另审计；不证明 greedy 一致。A/R11 已有整块固定组反例 TV=.32；HC/R01 已有 greedy 反例。
命题2：A(P)=1−Σ_g W_g TV(p_g,q)。合并子树 h 为父 g 的接受率增益 Δ_g=Σ_h W_h TV(p_h,q)−W_g TV(p_g,q)=1/2[Σ_h W_h||p_h−q||_1−||Σ_h W_h(p_h−q)||_1]≥0。凸性直接结果，与 A/R11/HC 重合。
命题3：**无需完整词表上传的精确残差采样**（已完成所选子树前向时）：拒绝后按 w_i/W_g 抽叶 I，再 Z~p_I，故 Z~p_g。向子树其他叶查询标量 p_i(Z)，聚合 p_g(Z)，以 b(Z)=[1−q(Z)/p_g(Z)]_+ 接受（仅在 p_g(Z)>0 上）。否则重复。一次提议并接受 z 的质量 [p_g(z)−q(z)]_+，最终样本为正确归一化残差。δ=TV(p_g,q)>0 时每次成功率 δ，拒绝条件下 E[N]=1/δ；原候选拒绝概率 δ，每个到达验证位置无条件 E[N]=1；δ=0 时 N=0。m 叶且本地概率已可查询，理想期望标量通信 O(m) 对比全量 O(m|V|)。最多 M 次尝试再回退到完整真实残差，回退概率 δ(1−δ)^M；字节、串行 RTT、局部完整 logits/softmax、KV/前向与长尾均需计入，不保证最坏通信或真实速度。这是经典拒绝采样应用，不是新原语。HCR/HIERASPEC 正残差包络质量 R≤1，条件尝试 R/δ，可能更高效但需付包络准备费用；DISCO/remote sampling 等亦需比较。

#### 4. 树优化及信息循环（O/H）

D_g=W_g TV(p_g,q)，c_g 为选中子树期望成本，代理 J(P)=Σ_g(D_g+λW_gc_g)。已知可加代价下 F(g)=min{D_g+λW_gc_g,Σ_{h child of g}F(h)}；叶仅第一项。O(树节点数)，HC/R01 同样推导。真实多步生存率、KV、并行临界路径、通信竞争并不可加。获取 D_g 本身可能要求全量前向与通信，出现先完成昂贵验证才能选验证粒度的信息循环。原拟用低成本标量查询、历史统计估计 Δ_g 并给误差界；PRE-08/A/R11 已有相邻失败，现不重复，除非新证据改变核心前提。

#### 5. 失败方向、反例及强基线

初筛不采用：联邦训练 Medusa 头（拼接）；接受率加权聚合（接受感知训练/自适应聚合已有）；在线 EAGLE（在线学习/演化目标）；边缘预测云验证（分布式 SD）；词表压缩（SpecFed/DSSD/DISCO）；简单增设验证层（Lossless HSD）。属文献/机制筛选，不是新实验。
CRITICAL：PRE-06/PRE-08/A/R06/A/R10/A/R11 已有相关历史，尤其 FSV 核心相同，E-H/P 表明平坦配对匹配或胜过边缘配对；同信息同委托计算平坦可模拟树。普通 AR 逐 token 按权重选一个模型再采样即精确，不需全量混合，禁止只和低效全量 AR 比。SD 目前不反向改变 HFL 学习；数学也可搭配独立草稿模型，G2 不通过。HC/HCR/HIERASPEC 的同期记录进一步说明概率抵消方向已高度重复。
MAJOR：逐位置抽组导致其他模型补算前缀/KV；高 RTT 残差串行长尾；云候选器前向成本；联合模型任务质量未知；真实区域同号偏差导致 Δ=0；参数聚合非概率聚合；错误混用残差或整块路由破坏分布；greedy/浮点未证。接受率与通信降幅不能代替全流程 wall time。

#### 6. 文献检索与撞题（历史线索，归档未逐条重新核验）

检索日期 2026-11-24，来源 arXiv、PMLR、ACL Anthology、OpenReview、IEEE、ICLR。关键词 Client Edge Cloud Hierarchical Federated Learning; Hierarchical Federated LoRA; Personalized Federated LLM; Multi-target SD; Distributed Speculative Decoding; Cloud Edge Verification; Speculative Ensemble; Collaborative Decoding; Maximal Coupling; Communication-free Coupling; Hierarchical Residual Sampling; Lossless Verification; Acceptance-Aware Training; Online Speculative Decoding; Mixture Distribution Coupling; Partition Verification; Distributed Exact Sampling。盲区：未公开同期、未索引、付费全文、概率论不同术语等价机制。
指定原文：[Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html) 精确接受/残差；[SD Survey](https://aclanthology.org/2024.findings-acl.456/) 分类；[EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840) 特征递推/动态树/多层特征直接 token 预测；[Medusa](https://arxiv.org/abs/2401.10774) 多头树验证（不可默认所有采样策略精确）；[SpecFed](https://arxiv.org/abs/2604.25777)（历史 [HTML v2](https://arxiv.org/html/2604.25777v2)）概率聚合/Top-K 压缩、重建近似待复核，最直接外部近邻。
重点近邻：[CoS](https://arxiv.org/html/2502.01662) 已有多模型联合分布精确 SD，不可声称首次；[Lossless HSD](https://arxiv.org/html/2601.05724) 候选分支层级残差而非物理树，仍需比对数学等价；[DISCO](https://arxiv.org/abs/2610.35957) 低通信分布式 SD，核查草稿复现与标量精确性；DIVERSED 动态集成，固定目标/保证待核；[HierFAVG](https://arxiv.org/abs/1905.06641)、[LayerSkip](https://aclanthology.org/2024.acl-long.681/)、[ResiSpec](https://arxiv.org/abs/2608.24411)。
其他原报告名称线索：Personalized Client-Edge-Cloud HFL、HierFedLoRA、pFedGPT、S2D、DSD、DSSD、PipeSD、FlexSpec、Variational SD、Acceptance-Aware Draft Training、When Drafts Evolve、Coupling without Communication。其他团队已记录部分链接；不补造未知链接。正式投稿前逐篇核查题名、时间、算法和重合点，独立撰写/规范引用不能解决内容重复；无正式文字查重。内部重合已足以否决。

#### 7. 最小及完整实验计划（H，未运行，取消重复核心实验）

P0 CPU 人造分布：词表 10/100/1000/32000，4/8/16/32 客户端，2/4 边缘，Dirichlet 异质性、RTT、K=1/2/4/8；枚举/MC 验证 TV、接受、回退、序列。A/R11 有相似实验。
P1 1–2×40GB NVIDIA，Qwen 1.5B–3B 级，8–16 虚拟客户端，顺序训练、共享主干、按需 LoRA。GSM8K 数学、MBPP 代码、通用 QA/指令、XSum 摘要、多轮对话；训练测试隔离、两级 non-IID。先测真实前缀 Δ_g 分布和层级相关性、混合质量；抵消小或混合无价值即否决。
P2 基线 B0 HFL 全局 AR（不同目标质量参考）；B1 全概率混合 AR；B2 随机选叶精确混合 AR；B3 Medusa 式+全局精确 SD；B4 EAGLE 式+全局验证；B5 SpecFed 压缩（质量/误差单列）；B6 固定叶/区域切分精确 SD；B7 全局标量查询+精确残差。加同信息平坦委托、自由配对、预缓存。消融：去动态切分/Δ估计、随机切分、平坦、打乱区域、完整残差、候选头、异质性。
指标 TTFT、TPOT、P95/P99、吞吐、接受长度、通信字节/轮次、FLOPs/GPU time、KV 补算回滚、任务质量、分布/序列正确性、训练推理累计 wall time/CI；两尺度、三非 IID/延迟设置、三独立种子。单卡模拟不冒充实际分布式速度。原拟好结果：同目标最快强基线≥1.2×推理加速、完整概率 SD 通信下降≥50%、精确、至少三任务、层级消融正收益；仓库 G5 要求全流程真实时间下降≥15%或同成本有效工作+20%。无实际数据，不判80%。

#### 8. G1–G7 和下一轮

原主观等权评分：问题7.5、机制6.5、理论7、价值7.5、距离6.5，均值7.0/10（未达8）；跨团队直接重合后旧分数失效，不虚构新评分。

| G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|
| FAIL 核心重复 | FAIL 无双向必要性 | PARTIAL/FAIL 单步 T，greedy/多步实现未审计 | FAIL 无实测 | FAIL 无全流程耗时 | FAIL 无80%证据 | FAIL 无代码日志 |

保留非交换性反例、残差期望尝试与有界回退、树代理优化及成本边界；不冒用他队数值。原拟 Δ 估计、HFL 特异性、真实概率预实验及重新查重，现 **Reject and Pivot**，禁止复刻 A/R11/HC/HCR/HIERASPEC 或仅追加普通训练/通信模块。下一轮先读最新 README 全文及所有团队正负成果，创造能真正双向改变 HFL 学习和免独立草稿执行的新计算对象，以强平坦及普通 AR/SD 否证；每轮结束无论成败都追加，不覆盖、不强推。

- 本次追加实际使用 README blob SHA：fa9946c112aa63d0d6542b916e5d8124ca451b8c；保留该版本中全部其他团队记录。


### team-himerge/R02｜区域私有内生预测头的联合验证隐私与接受率边界

- 日期：2026-11-24；研究开始读取 main commit 337195f58d2c286d5ce421d1f103183d3ef071fa，README blob 54306a1d276251d11389a5944af2f9364d004e11；该快照文件树仅 README.md（143220 bytes），无额外代码文件。已对照 PRE-01–11、A/R01–13、HC、HCR、HiMerge、HIERASPEC、HieraCert、HierCouple、HCA。证据等级 T（数学推导）、本轮实际 CPU/JS 合成检查（脚本和原始计数见下，非完整系统 R）、P（成本代理）、H（未验证训练假说）。
- 裁决 **Reject and Pivot（保留联合接受隐私边界作为通用限制）**。本轮不重复混合目标、概率抵消、随机验证组、残差包络或证书。新前提：正式目标相同，区域私有学习只改变内生候选头，受保护的是额外验证信息而非正式输出本身。

#### 1. 场景与计算问题（H/T）
客户端非IID私有数据，区域边缘按区域数据训练轻量 Medusa/早退式候选头 q_s，云周期性 HFL 聚合主模型 θ。固定服务快照 p(v|h) 为所有区域共享的正式目标；区域私有数据可改变 q_s 但不改变本次正式 p。候选生成可复用目标隐藏特征，不部署独立完整草稿 LLM。区域 s 表示两个或多个可能的私有头状态，而不必是公开的物理机构身份。可信执行域内可计算 q_s、p、接受判断，候选/概率/随机数不暴露；外部观察者仅见正式输出 Y 和真实接受标志 A（或其块长函数），不观察完整运行时间、候选、内存访问或训练消息。该限制不是全系统DP，也不保证成员隐私；在 p 随私有数据改变时，精确输出本身就可能泄漏。若云必须获得 q_s 或候选，则需另行证明完整消息隐私。
允许单位置先 X~q_s，A=1 仅当复用 X 作为正式输出，A=0 则从修正分布采样并截断后缀；标准无损 SD 是其中一个协议。约束 Y~p，且对于固定公开 h，TV(Law_s(Y,A),Law_t(Y,A))≤δ。优化平均接受率或实际 E[成本]/E[提交token]。δ 是该有限观测模型的统计距离，不是通用训练 DP 参数。

#### 2. 关键命题：共同接受质量（T）
记 m_s(v)=Pr_s(Y=v,A=1)，则 0≤m_s(v)≤c_s(v)=min(p(v),q_s(v))。必要性来自候选边际及正式输出边际。充分性：X~q_s 后以 m_s(X)/q_s(X) 接受；拒绝以 r_s=(p−m_s)/(1−Σm_s) 补采（分母0不触发；q_s(v)=0不抽该v）。Pr(Y=v,A=0)=p(v)−m_s(v)。注意 r_s 一般不是标准 (p−q_s)_+ 残差，因为主动放弃了部分可接受质量。
精确联合观测隐私 δ=0 当且仅当所有 m_s 相同。最优共同 m*(v)=min(p(v),min_s q_s(v))，最大共同接受率 C=Σ_v m*(v)。证明：必要条件逐坐标给上界，充分性构造达界。区域树可递归逐坐标 min 得下界向量，但该摘要可泄露私有头，需可信协调/安全计算；全词表、版本、访问当前前缀 p/q 的计算和通信不是免费。只隐藏 A 边际（相同接受率）不等于隐藏 (Y,A)。
两个等权状态的精确 Pareto 前沿：D=Σ|c1−c2|，C=Σ min(c1,c2)，TV((Y,A)_1,(Y,A)_2)=||m1−m2||_1（两类接受标志贡献相同）。平均接受率≤C+min(δ,D)/2；证明 (Σm1+Σm2)/2=Σmin(m1,m2)+||m1−m2||_1/2≤C+δ/2 且≤(Σc1+Σc2)/2=C+D/2。取 m_i=min(c1,c2)+t(c_i−min(c1,c2)), t=min(1,δ/D)，D=0时直接取共同 c，即达界。多状态/非均权可写 LP，未推导闭式。此为给定单候选复用语义、同 p、可观察真 A 的限定上界，不是任意侧信道或所有SD算法的通用下界。
多步：固定 p_h，若所有状态在每个实际可达公开前缀有相同 m_h，私有候选条件分布按实际 q_s(.|h)，不使用未来候选，随机数正确独立，则逐位置条件 (Y,A) 相同；用链式法则和标准首次拒绝/bonus规则得到正式序列分布正确、逻辑接受轨迹分布相同。EOS和固定长度需一致。实际时钟、消息内容、额外条件随机性、数值误差、缓存未审计。greedy 可单独设 p 为 tie-break 确定点质量作目标先行采样，但没有真实实现，不宣称完整 G3。

#### 3. 解析反例与成本（T/P）
p=(.5,.5)，q1=(.9,.1)，q2=(.1,.9)，c1=(.5,.1)，c2=(.1,.5)。两者标准接受率均 .6，观察 A 的边际相同，但联合TV=.8；只看正式Y的TV=0。完全隐藏 (Y,A) 的最优 m=(.1,.1)，接受率 .2。δ=.4 时 m1=(.3,.1)，m2=(.1,.3)，平均接受率 .4。四候选 γ=4 且条件接受率恒定/独立，期望每块提交 Σ_{j=0}^4 α^j：标准2.3056，δ=.4 时1.6496，完全隐私1.2496。若每块假设成本为1.25个AR token（仅P），对应加速1.84448/1.31968/.99968；完全隐私已输普通AR。一般要达到G5的15%总时间减少，推理本身须有足够节省且含额外训练与安全通信；即使零训练开销且全部成本为推理，此例完全隐私要求块成本≤.85×1.2496=1.06216个AR token。真实模型未知。

#### 4. 实际 CPU 合成检查（非 GPU / 非实际延迟）
本轮实际运行 JS V8，3个种子20261124/25/26；每状态每方法每种子100000独立块，γ=4。共2.4M块；仅统计首位置联合分布和实际逻辑提交长度，bonus token只计长度不采样，未验证完整序列分布或运行时。对照 standard、delta.4、private、public-oracle（q=p，不代表可部署廉价预测头）。原始输出每行 seed,mode, counts_s1,counts_s2（顺序 A=0/Y=0, A=0/Y=1, A=1/Y=0, A=1/Y=1）, meanL_s1/s2, empirical TV：
~~~text
20261124 standard [0,40210,49760,10030] [39748,0,9950,50302] 2.30391 2.31034 .8002
20261124 delta.4 [19913,40084,29975,10028] [40116,19941,10021,29922] 1.64785 1.64674 .40097
20261124 private [40297,39905,9866,9932] [39935,39897,10069,10099] 1.24673 1.25139 .0037
20261124 public-oracle [0,0,50058,49942] [0,0,50252,49748] 5 5 .00194
20261125 standard [0,40034,50019,9947] [40211,0,10046,49743] 2.30228 2.30195 .80007
20261125 delta.4 [20203,39961,29944,9892] [40154,19792,10033,30021] 1.64405 1.6505 .4008
20261125 private [39908,40138,9918,10036] [39755,39994,10259,9992] 1.24893 1.25278 .00341
20261125 public-oracle [0,0,49567,50433] [0,0,49872,50128] 5 5 .00305
20261126 standard [0,39940,50177,9883] [39980,0,10031,49989] 2.3132 2.30354 .80086
20261126 delta.4 [19886,40064,29900,10150] [39867,20172,10000,29961] 1.65181 1.65137 .39792
20261126 private [39755,40181,10113,9951] [39599,40423,10095,9883] 1.25127 1.24943 .00242
20261126 public-oracle [0,0,50107,49893] [0,0,49938,50062] 5 5 .00169
~~~
经验TV非零为有限采样波动，不是理论隐私失效；三种子未计算正式CI。实际执行代码可复制保存为 himerge_r02_probe.js，命令 node himerge_r02_probe.js；LCG仅作可复算检查：
~~~javascript
const seeds=[20261124,20261125,20261126],N=100000,gamma=4;
let results=[];
for(const seed of seeds){
let s=seed>>>0;
const u=()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296};
const draw=(a)=>u()<a[0]/(a[0]+a[1])?0:1;
for(const mode of ["standard","delta.4","private","public-oracle"]){
let counts=[],means=[];
for(let r=0;r<2;r++){
const p=[.5,.5],q=mode==="public-oracle"?p:r===0?[.9,.1]:[.1,.9];
const m=mode==="standard"||mode==="public-oracle"?p.map((v,i)=>Math.min(v,q[i])):mode==="private"?[.1,.1]:r===0?[.3,.1]:[.1,.3];
let c=[0,0,0,0],total=0;
for(let k=0;k<N;k++){let L=0;for(let j=0;j<gamma;j++){let x=draw(q),a=u()<m[x]/q[x];let y=a?x:draw(p.map((v,i)=>v-m[i]));if(j===0)c[2*Number(a)+y]++;L++;if(!a)break;if(j===gamma-1)L++;}total+=L}
counts.push(c);means.push(total/N)}
const tv=counts[0].reduce((t,x,i)=>t+Math.abs(x-counts[1][i]),0)/(2*N);
results.push({seed,mode,counts,means,tv});
}}
console.log(JSON.stringify(results));
~~~

#### 5. 文献核查、原创性与强基线
检索2026-11-24：speculative decoding timing side channel privacy; privacy-aware rejection sampling; differential privacy timing attacks; federated learning user-level DP; private draft acceptance; time-oblivious sampling。打开原始论文页面/部分HTML正文，未逐篇读全附录；未索引/同期未公开/密码学及隐私机制领域盲区。
- [Leviathan et al. 2023, Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)：标准精确SD。
- [LayerSkip, ACL 2024](https://aclanthology.org/2024.acl-long.681/)：共享早退和验证。
- [Wei et al., Privacy Risks of Speculative Decoding in Large Language Models, 2024 preprint](https://arxiv.org/abs/2411.01076)：输入相关接受模式、数据存储等私有信息泄漏、聚合/填充防护；新版本题名When Speculation Spills Secrets，不能声称首次发现侧信道。
- [Carlini & Nasr, Remote Timing Attacks on Efficient Language Model Inference, 2024 preprint](https://arxiv.org/abs/2410.17175)：黑盒时间侧信道。
- [Awan & Rao, Privacy-Aware Rejection Sampling, JMLR 2023](https://www.jmlr.org/papers/v24/21-0870.html)：接受率与运行时间的隐私成本、数据无关时间精确采样，直接数学近邻。当前共同子概率和联合(Y,A)前沿是限定SD复用语义的推导，不保证新颖。
- [Ben Dov et al., Resistance to Timing Attacks for Sampling and Privacy Preserving Schemes, FORC 2023](https://doi.org/10.4230/LIPIcs.FORC.2023.11)：时间不可区分抽样。
- [HierFAVG 2019](https://arxiv.org/abs/1905.06641)：分层部分参数聚合，非本轮验证隐私机制。
- [McMahan et al., Learning Differentially Private Recurrent Language Models, 2017 preprint](https://arxiv.org/abs/1710.06963)：用户级DP联邦训练，若所有私有依赖均来自DP训练发布的头、之后运行只后处理该头，则逻辑/时序输出亦受相同DP保证，不能把本轮零TV要求当作唯一隐私方法。
CRITICAL：普通共享公开内生头 q0 不依赖私有区域，天然不泄漏该区域头状态；给定任一固定公开概率向量 q0≥min_s q_s（例如公开的平均概率，在能合法取得时），标准SD接受率≥C。取得/执行平均分布不免费，oracle q=p仅说明数学上限；现实应训练同容量共享头并实测。平坦FL能按同样约束训练共同头或执行min摘要；若观察者仅见外部流式包而看不到真A，则可信域缓冲/固定大小填充允许保持原接受率，可能更优；若真A、候选与q都被云看到，本机制不保证完整隐私。共享p若来自私有非DP训练本身仍可泄漏，训练、网络、存储侧信道均未处理。核心训练目标拟最大化 E_h Σ_v min(p_h(v),min_s q_s(v|h))，可用软min或受约束优化，但只是鲁棒/一致性训练，且 min 梯度、私有摘要发布与公平平坦均未解决。未建立 SD 改变正式HFL学习机制；故G2失败。不为挽救候选而降低门槛。

#### 6. 计划和验收
只有真实业务要求隐藏可见真A（非仅外部流量）且独立区域头相对同容量公开头具有已测优势，才可重新研究。假设预实验：1–2×40GB，Qwen2.5 1.5B/3B，冻结主模型，4–8逻辑客户端2边缘，私有轻量头，公开数学/代码/通用数据三异质设置及三种子；保留目标p，记录真实前缀 c_s、C、D、实际接受长度、联合泄漏、吞吐、真实训练/推理/通信/回退、安全计算开销。基线无SD HFL AR；普通HFL+SD；同容量共享公开头；DP训练头（同隐私定义需区分）；缓冲/填充/固定节奏（同可见性）；平坦同信息/同委托；隐私感知拒绝采样。先看C与共享头实测、隐私威胁模型是否真实，若共同头匹配即停止。好结果按G5：同目标质量、同明确隐私预算及资源下总真实时间降≥15%或同成本有效工作增≥20%，并满足G1–G7，不能拿无隐私弱基线作为唯一对照。未执行GPU实验。
主观分项（问题7×25%、机制6×25%、理论7×20%、价值7×15%、文献距离5×15%）=6.45/10，不是同行评审；无证据支持80%好结果概率。G1 FAIL（近邻、限定推导）；G2 FAIL/CRITICAL（共享头/平坦、无双向学习）；G3 PARTIAL/FAIL（理想单步与逻辑轨迹，完整greedy/采样/精度未审计）；G4 FAIL（仅CPU合成）；G5 FAIL（无wall time，代理例已否决）；G6 FAIL（证据不足）；G7 PARTIAL/FAIL（内联CPU代码、计数与种子，非完整系统）。无正式文字查重，独立撰写和规范引用不替代内容查重。
保留共同接受质量上界、等接受率仍泄漏联合信息反例、两状态精确TV—接受率前沿、强基线与实验脚本。禁止只用一致接受率宣称验证隐私，或把局部头训练+隐私约束视为HFL×SD双向创新。下一轮优先寻找真实计算约束，正式HFL更新与自投机计算共享不可替代的可训练原语；先用公开小模型或实际trace证明问题存在，再考虑训练，不重做概率混合/残差或单纯隐私填充。



### team-hieracert/R02｜PrivSpec-HFL：私有内生预测头与精确投机验证轨迹的隐私—接受率冲突

- 日期 2026-11-24；开始读取 main commit 337195f58d2c286d5ce421d1f103183d3ef071fa，README blob 54306a1d276251d11389a5944af2f9364d004e11。固定版本全文1039行，递归文件树只有 README；读取内嵌代码和结果，未发现独立历史实验包。已读 PRE-01–11、A/R01–13、HC、HCR、HiMerge、HIERASPEC、HieraCert、HierCouple、HCA。上传前重新读取最新 README，追加时使用 blob cf6da592627a62ce6b397099d1569689c5711c00。
- 裁决 **Reject and Pivot / 保留限定隐私下界与最优接受质量公式**。新问题与既有概率混合/残差包络/缓存不同，但 HFL 双向必要性失败；DP 训练、隐藏真实接受标记、普通缓冲是强替代，邻接图全知为 oracle。未立项。
- 证据 T=限定数学推导；本轮实际 CPU 数值检查附代码输出（局部可复算，不是完整 R）；O/P=有限图全知与理想成本；H=待证假说。未运行预训练 LLM、HFL 训练、GPU 或网络实验。不得把 toy 数字当真实速度。

#### 1. 场景及问题

固定公开共享目标 p(v|h)，区域内生轻量头 q_D(v|h) 由客户端非 IID 私有数据 D 经 client-edge-cloud HFL 训练；冻结请求目标快照，边缘有正式模型和头，可信本地执行，无独立草稿 LLM。研究对象为观察者能看到正式输出 Y 与真实接受标记 A，但不能看到未提交候选 X、私有 q、额外数据依赖元数据的限定接口。训练消息、模型发布、私有 prompt、其他计时、包大小、内存访问均未被此证明保护。固定 p 对 D 不变；若 p_D 不同且精确输出本身违反 DP，则任何协议都不可能保证完整 transcript DP（投影后处理）。若 p 也训练于私有数据，需另行账户与共同条件分析。

标准 SD m_D(v)=min(p(v),q_D(v)) 为 P(Y=v,A=1|D)，拒绝质量 p-m_D。p=(.5,.5), q1=(.5,.5), q2=(1,0)，正式输出都为 p，拒绝概率0和.5，任何有限纯 ε-DP 均不成立（实际可观察事件）。文献已知 SD 侧信道，不声称首次发现。

定义有限无向数据邻接图 G（邻居例如一客户端改变），p固定，u_D(v)=min(p(v),q_D(v))。选择 0≤m_D≤u_D，最大化各数据集接受率 Σ_v m_D(v)，要求 (Y,A) 为 ε-DP，同时 Y~p。允许可信本地抽样、残差计算、公开固定版本与预先获得的邻接约束；不宣称实际能免费获得全部邻居的模型。候选 X~q_D，接受 m_D(X)/q_D(X)，拒绝从 (p-m_D)/(1-Σm_D) 抽样；零拒绝跳过。单步输出严格 p。与 PRE-07（未知未来 p 的公共质量）不同，此处 p固定、q私有且 joint transcript 受约束；与 A/R12 独立接受不同，允许 token 相关接受质量但跨 D 隐私约束。仍使用经典广义耦合及 DP 线性约束，不将其包装为全新数学领域。

#### 2. 理论 T：有限邻接图上的逐坐标最优接受质量

c=exp(ε), F_v(x)=min(c*x,p(v)*(1-1/c)+x/c)。两方向 DP 约束 m_D≤c*m_D' 和 p-m_D'≤c*(p-m_D) 等价 m_D≤F_v(m_D')，对每条无向边双向约束。F 单调，0≤x≤p 时 F(x)≥x。令 d 为图最短距离，m*_D(v)=min_{D'同连通分量} F_v^{d(D,D')}(u_D'(v))，F^0 恒等。
证明：任意可行 m 沿路径递推≤F^d(u_D')，所以 m≤m*。对邻居 D,E，d(D,J)≤1+d(E,J)，F单调且 F≥id，因此 m*_D≤min_J F^{1+d(E,J)}(u_J)=F(m*_E)，反向同理；且包含 J=D 故 m*≤u_D。故 m* 可行且逐坐标支配任意可行解，任意非负加权接受率最优。实现为从 u 初始化、同步邻接 min 传播至图直径步数。ε=0 连通分量公共 min u；若任一节点 u(v)=0，则任何有限纯 ε 下全连通分量 m*(v)=0。两节点 p=.5,q2(B)=0 时 B 接受质量全为0，A 最多.5；即使 ε 很大仍有支持集障碍。仅适用于真实接受 flag 可观察、单候选接受—残差、固定 p、有限图完全信息，不是所有 DP/SD 协议普适下界。若 greedy p 为确定性点质量，预测头确定且邻图任一节点猜错，纯 DP 真实接受 flag 约束使对应 token 在整连通分量接受质量为0。

多步：若每个可达历史下真实条件 q_D、独立接受/残差随机数、因果停止、未观察候选不参与后续决策，按链式条件输出保持 p；每一步联合核统一 ε_t-DP，固定最大 N 步的基本自适应组合上界 Σε_t（停止/bonus/公开元数据只允许是该 transcript 后处理）。全块候选的条件分布、KV、EOS、提前退出时长、浮点及 top-p 尚未实现审计；p,q 截断应预先定义，零 q 会触发上述下界。不能用单步 ε=1 宣称 100 token 整体 ε=1，基本上界100。不能将输出内容 DP 扩展到可见候选：若 X 也公开，其边际必须本身 DP，改变接受规则无法修复。

#### 3. 实际执行 CPU 数值检查（O/P，非 LLM）

Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, scipy.optimize.linprog highs。5节点链、32词，3独立种子 11/23/37，各40试验，15配置=1800个合成分布图。共享 p=softmax(normal logits)，q_i=softmax(logits+σ iid Gaussian)，邻接链仅人为模拟，不是实际相邻训练集。每配置120图；LP 对每种子首个图的 token0 独立验证，共45 LP。记录标准 SD、零隐私公共包络、最优 m* 的节点平均接受率；后两列是最大 DP 约束违反及 LP 差。实际执行脚本如下（保存 probe.py；python probe.py；依赖 numpy==2.3.5 scipy==1.17.0）：

~~~python
import numpy as np, scipy, sys
from scipy.optimize import linprog
def run():
    results=[]
    for eps in [0.,.1,.5,1.,2.]:
        c=np.exp(eps)
        for spread in [.15,.6,1.2]:
            means=[]; errs=[]; lp_errs=[]
            for seed in [11,23,37]:
                rng=np.random.default_rng(seed); per=[]
                for trial in range(40):
                    v=32;n=5
                    z=rng.normal(size=v); p=np.exp(z-z.max());p/=p.sum()
                    l=z[None,:]+spread*rng.normal(size=(n,v))
                    q=np.exp(l-l.max(axis=1,keepdims=True));q/=q.sum(axis=1,keepdims=True)
                    u=np.minimum(p,q); m=u.copy()
                    for _ in range(n-1):
                        old=m.copy()
                        m[1:]=np.minimum(m[1:],np.minimum(c*old[:-1],p*(1-1/c)+old[:-1]/c))
                        m[:-1]=np.minimum(m[:-1],np.minimum(c*old[1:],p*(1-1/c)+old[1:]/c))
                    res=p-m
                    err=max(0.,(m[1:]-c*m[:-1]).max(),(m[:-1]-c*m[1:]).max(),(res[1:]-c*res[:-1]).max(),(res[:-1]-c*res[1:]).max())
                    errs.append(err)
                    per.append([u.sum(1).mean(),u.min(0).sum(),m.sum(1).mean()])
                    if trial==0:
                        a=[];b=[]
                        for i in range(n-1):
                            for j,k in [(i,i+1),(i+1,i)]:
                                row=np.zeros(n);row[j]=1;row[k]=-c
                                a.append(row);b.append(0)
                                a.append(-row);b.append((c-1)*p[0])
                        sol=linprog(-np.ones(n),A_ub=a,b_ub=b,bounds=[(0,x) for x in u[:,0]],method="highs")
                        lp_errs.append(np.max(abs(sol.x-m[:,0])))
                means.append(np.mean(per,axis=0))
            results.append((eps,spread,*np.mean(means,axis=0),max(errs),max(lp_errs)))
    return results
print(sys.version.split()[0],np.__version__,scipy.__version__)
for r in run():print(" ".join(f"{x:.8g}" for x in r))
~~~

实际输出（ε σ 标准接受 公共包络接受 最优隐私接受 最大DP绝对违反 LP最大差）：
~~~text
3.13.5 2.3.5 1.17.0
0 0.15 0.94176055 0.83868783 0.83868783 0 0
0 0.6 0.7722312 0.45859558 0.45859558 0 0
0 1.2 0.57548108 0.17542962 0.17542962 0 0
0.1 0.15 0.94176055 0.83868783 0.86038301 3.469447e-17 6.9388939e-18
0.1 0.6 0.7722312 0.45859558 0.51064737 2.7755576e-17 6.9388939e-18
0.1 1.2 0.57548108 0.17542962 0.20183698 2.0816682e-17 0
0.5 0.15 0.94176055 0.83868783 0.90316326 4.1633363e-17 6.9388939e-18
0.5 0.6 0.7722312 0.45859558 0.64153216 5.5511151e-17 3.469447e-18
0.5 1.2 0.57548108 0.17542962 0.30887505 1.7347235e-17 3.469447e-18
1 0.15 0.94176055 0.83868783 0.92285889 1.3530843e-16 6.9388939e-18
1 0.6 0.7722312 0.45859558 0.7099921 1.2490009e-16 6.9388939e-18
1 1.2 0.57548108 0.17542962 0.41254133 5.5511151e-17 3.469447e-18
2 0.15 0.94176055 0.83868783 0.93600132 1.8691645e-16 6.9388939e-18
2 0.6 0.7722312 0.45859558 0.75421034 1.9428903e-16 6.9388939e-18
2 1.2 0.57548108 0.17542962 0.52213582 3.3306691e-16 6.9388939e-18
~~~
最大DP违反约3.33e-16、LP差6.94e-18，仅浮点容差检查，不是严格浮点DP证明。未测 wall-clock、CI、完整随机序列；接受率是解析质量均值，不是 Monte Carlo 采样频率。与标准非隐私 SD 不同约束，只作性能上界；公共包络是更严格 ε=0 基线，LP是同约束最优对照，结果完全匹配（非优于LP）。中σ=.6 ε=.5，接受.64153 vs标准.77223/公共.45860；ε=.1仅.51065。固定四候选且恒定独立接受率 α 的代理提交数 Σ_{j=0}^4 α^j；实际接受率条件依赖，不可直接代入推断真实速度。完整图若 n 个节点、|E| 条边、直径 H、词表 V，传播 O(H|E|V) 算术/潜在标量跨节点通信；若整个图集中持有 O(nV) 内存。真实邻接数据集数巨大，每个都重新训练不可能；发送私有 q 或图消息也可能泄露。上述实验忽略图构建和训练，oracle 结果不代表可部署。

#### 4. 原始来源检索与查重（2026-11-24）

本轮实际检索关键词 speculative decoding privacy leakage timing side channel draft model acceptance differential privacy; remote timing attacks efficient inference; differential privacy optimal graph mechanisms; DP federated language models; private decoding; speculative decoding padding. 查 arXiv/PMLR/ACL/ICLR/JMLR/Google Research。核对原始论文摘要及部分HTML正文，不声称全部最新论文全文逐式覆盖。
- [Privacy Risks of Speculative Decoding in Large Language Models](https://arxiv.org/abs/2411.01076) 2024：已明确 SD 正误模式泄露、预测数据和参数风险，padding/聚合缓解；直接问题先例。
- [Remote Timing Attacks on Efficient Language Model Inference](https://arxiv.org/abs/2410.17175) 2024：推理效率差异造成计时侧信道；不以侧信道本身为新贡献。
- [Whisper Leak](https://arxiv.org/abs/2511.03675) 2025：网络元数据泄露及 padding、batching、packet injection 缓解；强系统基线。
- [Leaky Language Models](https://arxiv.org/abs/2607.20723) 2026：从 token timing 识别架构和 SD 配置；不是本轮新发现。
- [DP-Fusion](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f5198bc255e1d5f959edd6d1d1a86fab-Abstract-Conference.html) 2026：改变输出分布以保护输入上下文；本轮固定 p，保护实际验证 trace，问题不同。
- [PEARL](https://proceedings.mlr.press/v306/joo26a.html) 2026：RAG 生成的自适应 DP 预算及质量；不同目标，但预算分配非新。
- [Extremal Mechanisms for Local Differential Privacy](https://www.jmlr.org/beta/papers/v17/15-135.html) 2016、[Universally Utility-maximizing Privacy Mechanisms](https://doi.org/10.1137/09076828X)：DP 机制优化、线性约束及最优机制数学先例；本轮闭式解只是特殊结构，非通用 DP 新原理。
- [Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963) 2018：user-level DP FedAvg，训练阶段 DP 是关键强基线；若整个发布头满足 DP，固定 p 与公开输入下全部推理 transcript 仅是 postprocessing，无需每 token 额外付本轮 DP 代价（前提训练及其他可见信息均覆盖）。
- [Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html) 2023、[LayerSkip](https://aclanthology.org/2024.acl-long.681/) 2024：精确验证和内生起草基础。盲区：DP graph mechanism 特殊形式、oblivious traffic shaping、privacy-preserving distributed inference 最新未索引工作、完整正式论文与附录；无正式文字查重，需独立写作规范引用。

#### 5. 强基线、失败及下一轮

CRITICAL: (i) 若客户端级 DP HFL 头训练一次且全部 observable transcript 仅由 DP 模型及公开 p、公开 prompt 和独立随机数计算，则后处理已保护训练数据，无逐token组合损失；本轮未击败。注意 DP 训练的质量代价也须测，不声称必然性能更好。 (ii) 不依赖私有数据的公开内生头天然对 D 为0-DP；普通 AR 也是；若性能匹配，本轮不存在价值。 (iii) 隐藏接受标记、固定大小/时间的输出缓冲、可信执行可改变可观察对象；本轮公式不适用，须同威胁模型计填充、排队、延迟。 (iv) 有限邻接图全知不可部署；实际训练可能随机化，邻接图是训练数据集而非区域ID，不可将区域相邻误称客户端DP。 (v) 普通 FL 或任意私有预测头均可使用，HFL没有改变机制的必要性，SD没有改变有效联邦学习。 (vi) 纯 DP 支持集与长序列预算令接受率大幅退化；实际时间和泄漏未测。若真实 q 有极小非零概率也可能造成严重瓶颈；近似 DP 需单独全事件约束（不能仅逐坐标加 δ）。

若有改变前提的新证据，再做最小真实实验：固定公开 Qwen2.5-1.5B/3B（两尺度），冻结主体，2–4区域8–16逻辑客户端，公开代码/数学/通用数据非IID，训练同架构内部头，比较普通HFL+SD、无SD AR、公开头、客户端级DP-HFL+SD、同信息平坦DP-FL+SD、固定长度/节拍缓冲、同一威胁模型下本机制（仅可证明真实邻接界时）。至少3种子3异质性/网络设定。1–2×40GB，单卡逻辑模拟不声称并行；双卡才测物理链路。质量匹配，固定请求级/用户级 ε,δ 与全部可见信息；报告累计训练+推理耗时、通信含证书及 padding、TPOT/P95、接受率、吞吐、GPU时间、任务质量、实际隐私会计、CI。验收 G5 -15%总wall time或同成本+20%有效工作，不以接受率替代。若 DP训练/公开头/缓冲匹配，或必须全知图、无法证明隐私或质量，立即否决，不浪费GPU。

主观等权创新分：问题7、机制6.5、理论7、价值6.5、距离5.5，均值6.5/10；非审稿人评分。好结果概率无法可靠估计，未达到80%证据门槛。G1 FAIL（6.5且数学/问题先例）；G2 FAIL（CRITICAL 双向必要性无、平坦模拟）；G3 PARTIAL/FAIL（限定实数单步最优与条件序列证明；完整实现/精度/其他元数据未证）；G4 FAIL（仅合成）；G5 FAIL（无真实wall time）；G6 FAIL（无80%证据）；G7 PARTIAL/FAIL（局部代码输出可复算，完整系统未复现）。**Reject and Pivot**。保留 transcript 与输出区别、有限图最优公式及支持集下界、LP 检验、DP训练后处理强基线；不以加噪/边缘隐私模块换名立项。下一轮优先从真正 HFL 学习对象出发：只在数据驻留/计算授权下可形成、同时改变目标模型内部执行规则和学习计算的机制；先证明普通 AR、DP训练和同信息平坦不能替代，再做最小真实模型检验。每轮先读库并记录 commit，后追加正负结果。


### team-hcr-sd/R02｜PC-SD：条件验证状态隐私与精确投机接受率上界

- 日期 2026-11-24；起始 main commit 337195f58d2c286d5ce421d1f103183d3ef071fa，README blob 54306a1d276251d11389a5944af2f9364d004e11，143220 bytes。分段读取完整 README（78082 JS字符），根目录仅 README，相关代码内嵌。提交前检测并发更新，读取 d5f04bafe761c3962a6baac1d19c39fa1b0f663e（blob a44b2cdff41596e3b53b6fdef8bc4ffd16061bd3）全部新增内容并保留原文。
- **CRITICAL / Reject and Pivot。** R01 HCR-SD 正式否决；本轮不重复概率混合。R02 独立探索后并发发现 team-himerge/R02、team-hieracert/R02 均已研究真实接受状态与输出联合隐私。区别：本轮允许私有正式目标 p_i 不同、固定公共 q，条件于正式输出的额外隐私；两队固定 p、私有 q，分别研究联合 TV 前沿、邻接图纯 DP 最优接受质量。将本轮 a=m/p 归一化后，闭式是同类图约束的完全邻接特例；不能当三个独立创新。A/R12 固定 c 接受亦是近邻。保留有限条件最优式、反例与 CPU 检验，未立项。T=数学，P/O=代理/完全信息，H=假说；无真实 LLM/HFL/GPU/网络运行。
- 场景：client 非IID 数据，edge 可信区域聚合并执行私有目标 p_i(.|h) 和内部预测头 q，cloud 聚合共享权重、协调服务，可见正式输出 Y 和真实接受 A/块边界，但看不到未提交 X、私有 logits 和随机数。i 是同一公开路由下的可能秘密模型状态，非已公开的机构标签。固定版本、同词表、正 p_i、公共 q；正式目标各自自回归 p_i，不是混合目标。固定输出长度、温度采样、无 top-p；不承诺完整时间/训练数据 DP。若完整模型、草稿、可见状态均为联合 DP 训练发布的后处理，则推理不额外耗费该训练 DP 预算；本题是未公开模型的条件状态隐私。
- 定义：固定 h,y，对任意 i,j,b∈{0,1}，Pr_i(A=b|Y=y,h)≤exp(ε)Pr_j(A=b|Y=y,h)。标准最大耦合的 a_i(y)=min(1,q(y)/p_i(y))。q=(.5,.5)，p1=(.6,.4)，p2=(.4,.6)，两者总体接受率均 .9，给定 Y=A 拒绝概率分别 1/6、0，条件比值无穷，说明平均接受相同不保证额外隐私。正式输出本身可能泄露，不因条件隐私而隐藏。
- 精确机制（T）：选择非最大接受质量 0≤m_i(y)≤min(p_i(y),q(y))，a_i=m_i/p_i。X~q，以 m_i(X)/q(X) 接受；否则从 (p_i−m_i)/(1−Σm_i) 补采，允许补采同一候选但仍停止本块；零拒绝不调用补采。输出边际 m_i+p_i−m_i=p_i。与标准正残差不同，不得误用 (p−q)_+。
- 完全邻接有限模型族固定 q 的最优闭式（T/O）：令 u_i=min(1,q/p_i)，u_*=min_i u_i，t=exp(ε)。a_i*=min{u_i,t u_*,1−(1−u_*)/t}，逐坐标支配全部可行解。证明：取最小上界节点 k，a_k≤u_*；由接受/拒绝双向比值约束 a_i≤t a_k、a_i≤1−(1−a_k)/t，得到两项上界；构造 a_k*=u_*，所有 a_i*≥u_*，最大与最小的两个伯努利事件比值均≤t，因而全部对可行。任意非负权重单步接受最优；非多步时延最优。u_*=0 则全为0。ε=0 时共同条件 a_i=u_* 可随已释放 token 改变，不是 A/R12 固定 c。二词例 ε=0 接受5/6；ε=ln2 接受.8666667；无约束.9。若公共固定下界 b≤u_*，用 b 替代 u_* 仍可行但非最优；b 获取和发布不可泄露秘密。
- 逐位置因果使用真实条件 q、独立随机数、固定正式长度、合法停止/bonus，若给定全部公开过去仍有 Y~p_i(.|h)，联合核分解为正式序列概率乘验证轨迹核，条件于整个正式序列的额外似然比≤exp(T ε)，不是固定全会话 ε。bonus 的状态为公开历史决定。不能利用未来候选或隐藏路由改变核。未证明真实时间、任意 top-p、greedy 隐私、EOS、数值及 KV 实现；若 X 也公开则本式不保护。
- ε=0 可优化公共 q 的完全信息上界（O）：pbar=Σw_i p_i，pmax(y)=max_i p_i(y)，最大化 Σ_y pbar(y)min(1,q(y)/pmax(y))，Σq=1。按 pbar/pmax 降序分配 q，每词上限 pmax，分数背包交换论证。非新背包算法，非可部署头；HFL 训练拟合该分布仅 H，无法免费取当前全部区域概率，可能泄漏，普通平坦同样可做。
- 实际 CPU 数值检查（P/O，非采样频率/实际时间）：Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0；seed 11/22/33，每个σ每种子200个合成前缀、4区域、64词，共1800个图；共享高斯 logits + 区域独立扰动，q=pbar 是 oracle。3种子 mean±sample SD，列为标准最大耦合、固定 c=min_y,i u_i、ε=0 条件保护、ε=ln2、ε=ln4、ε=0 最优q。100随机单词 LP 检验闭式，最大坐标误差1.1102230246251565e−16。未测 wall time/真实语言模型或CI。脚本内嵌，可复制为 pc_sd_probe.py，运行 python pc_sd_probe.py（依赖版本见上）。
~~~text
sigma=.1 mean [0.966402 0.793221 0.908703 0.940002 0.954040 0.914976]
sd            [0.000126 0.001534 0.000302 0.000192 0.000144 0.000311]
sigma=.5 mean [0.833263 0.411686 0.665604 0.757594 0.797822 0.712097]
sd            [0.000780 0.002784 0.001317 0.001074 0.000901 0.001431]
sigma=1 mean  [0.677368 0.290721 0.499675 0.597728 0.640347 0.572733]
sd            [0.002081 0.001424 0.001932 0.002094 0.002105 0.001939]
LP max coordinate error 1.1102230246251565e-16
~~~
~~~python
import numpy as np
from scipy.optimize import linprog
import scipy, sys
def probe():
    out=[]
    es=[0,np.log(2),np.log(4)]
    for sig in [.1,.5,1.0]:
        vals=[]
        for seed in [11,22,33]:
            rng=np.random.default_rng(seed)
            base=rng.normal(size=(200,64))
            logits=base[:,None,:]+sig*rng.normal(size=(200,4,64))
            p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            q=p.mean(1)
            u=np.minimum(1,q[:,None,:]/p)
            lo=u.min(1,keepdims=True)
            c=lo.min(-1,keepdims=True)
            pmax=p.max(1); order=np.argsort(-q/pmax,axis=1)
            mx=np.take_along_axis(pmax,order,axis=1)
            qsorted=np.minimum(mx,np.maximum(0,1-(np.cumsum(mx,axis=1)-mx)))
            qo=np.zeros_like(q);np.put_along_axis(qo,order,qsorted,axis=1)
            oracle=(q*np.minimum(1,qo/pmax)).sum(-1).mean()
            alpha=lambda a: (p*a).sum(-1).mean()
            vals.append([alpha(u),alpha(c),*[alpha(np.minimum(u,np.minimum(np.exp(e)*lo,1-np.exp(-e)*(1-lo)))) for e in es],oracle])
        arr=np.array(vals);out.append((sig,arr.mean(0),arr.std(0,ddof=1)))
    return out
print(sys.version.split()[0],np.__version__,scipy.__version__)
for s,m,sd in probe(): print(s, np.round(m,6), np.round(sd,6))
rng=np.random.default_rng(7);err=0
for k in range(100):
    u=rng.uniform(0,1,4); e=rng.uniform(0,2); t=np.exp(e); w=rng.uniform(0,1,4)
    A=[];b=[]
    for i in range(4):
        for j in range(4):
            if i==j:continue
            x=np.zeros(4); x[i]=1;x[j]=-t;A.append(x);b.append(0)
            A.append(-x);b.append(t-1)
    res=linprog(-w,A_ub=A,b_ub=b,bounds=list(zip(np.zeros(4),u)),method="highs")
    a=np.minimum(u,np.minimum(t*u.min(),1-(1-u.min())/t))
    err=max(err,np.max(np.abs(res.x-a)))
print("LP max coordinate error",err)
~~~
- 文献：2026-11-24 搜索 speculative decoding timing side channel/privacy/acceptance; differential privacy rejection sampling timing; token streaming padding; model architecture timing; federated privacy。核对原始论文摘要/页面，非全部全文。Wei et al. [When Speculation Spills Secrets](https://arxiv.org/abs/2411.01076) 2024-11初稿、2026-02 v4 已研究接受模式泄露、填充/聚合；Awan & Rao [Privacy-Aware Rejection Sampling](https://www.jmlr.org/papers/v24/21-0870.html) JMLR2023 数据无关运行时间及精确采样；[A Framework for Differential Privacy Against Timing Attacks](https://arxiv.org/abs/2409.05623) 2024 区分输出和时间隐私；[Remote Timing Attacks on Efficient Language Model Inference](https://arxiv.org/abs/2410.17175) 2024；[Leaky Language Models](https://arxiv.org/abs/2607.20723) 2026-07；[From Length to Content](https://www.usenix.org/conference/usenixsecurity26/presentation/li-sijia) 2026 合并token仍存在流量泄露；[Privacy-Aware Split Inference with SD](https://arxiv.org/abs/2602.16760) 2026 保护目标不同；基础 [Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html)、[LayerSkip](https://aclanthology.org/2024.acl-long.681/)、[HierFAVG](https://arxiv.org/abs/1905.06641)。盲区：完整附录、未公开工作、DP图机制/条件隐私耦合OT。不能声称首次发现侧信道；无正式文字查重。
- CRITICAL：可信边缘若无需公开真 A，普通最大耦合 SD+缓冲/固定频率大小/完成后发布可以保持内部接受率且隐藏逻辑边界；真实时间需另外防护和实测。若真A必须公开才有比较意义；若候选也公开则公式不足。公共 u_* 或 b 获取成本/泄漏未解决，模型族很大时 b 可近0。普通平坦、普通个性化服务可执行同样机制；没有学习↔SD双向必要性；并发两团队更证明核心数学重复，不投入重复 GPU 实验。
- 基线：普通HFL AR；HFL+LayerSkip/Medusa；标准SD+同威胁下填充、缓冲/固定节奏/随机延迟；固定c；公共保守包络；DP训练（若保护训练数据）；同信息平坦；同约束LP。无隐私标准SD仅效率上界。未来若有改变前提的业务证据，1×40GB Qwen2.5 1.5B、8客户端2区域、LoRA和内部头，第二尺度3B/7B，2×40GB物理并行，三异质性/网络配置三种子，真实1万独立前缀、公开数据、任务质量、真实泄漏、通信、TTFT/TPOT/P95/P99、训练+推理总时间、显存及CI；单卡模拟非分布式实测。未执行。好结果同目标/保护/资源下 G5 总时间下降≥15%或等成本有效工作+20%，无80%证据。
- 主观等权创新评分6.5/10（问题6.5、机制6.5、理论7、价值7、距离5.5）；并发重合后不得上调。G1 FAIL；G2 CRITICAL FAIL；G3 局部T/整体FAIL（greedy/精度/真实时间未验证）；G4 FAIL（仅合成）；G5 FAIL（无实际总时间）；G6 FAIL（未知80%）；G7 局部可复算/整体FAIL。**Reject and Pivot。** 保留条件似然比反例、闭式最优、公共下界、q分配和LP检验。禁止只优化接受率、隐私填充或重复概率残差；下一轮从数据驻留/计算授权引起的正式模型学习与内生推测执行共享原语出发，先证明普通AR/SD、同信息平坦不能匹配，再投入真实预训练模型实验。


### A/R14｜区域目标异质性下的条件验证隐私与最优接受质量

- 日期：2026-11-24。研究起始 main commit 337195f58d2c286d5ce421d1f103183d3ef071fa，README blob 54306a1d276251d11389a5944af2f9364d004e11；1039 行。已读取该快照 README（含内联程序/结果），与初始 commit f93a4c815028bcc3b7104df1e68e84785071bf85 比较，7 个追加 commit 仅修改 README，未见独立历史代码文件。已审查 PRE-01–11、A/R01–13、七团队 R01（HC、HCR、HiMerge、HIERASPEC、HieraCert、HierCouple、HCA）。
- **并发去重更新：** 上传前 main d5f04bafe761c3962a6baac1d19c39fa1b0f663e 已出现 team-himerge/R02 和 team-hieracert/R02，二者独立研究了固定正式 p、不同私有 q 的联合 (Y,A) 隐私与最优接受质量；本轮为不同正式 p_r、共同 q，并保护给定 Y 后的条件 trace，而非完整 (Y,A) 的绝对隐私。数学同属受限子概率耦合，尤其本轮闭式解等同于对 team-hieracert/R02 的有限完全邻接图公式作条件概率归一化。保留差异和独立实验，不以并行发现声称多项创新。
- **裁决 Reject and Pivot（保留条件隐私数学边界、最优共同草稿及独立数值证据）。** G1/G2 不通过，核心尚未证明优于同隐私预算普通缓冲/固定节奏、公开头、同信息平坦。证据 T=限定数学；E=本轮实际执行 V8 CPU 数值运算和完整有限序列枚举，附脚本及逐种子结果，未独立复跑；P=理想成本模型；O=全信息 oracle；H=未验证假设。无预训练 LLM、实际 HFL 参数训练、GPU、真实通信/计时实验，非完整 R 级复现。

#### 1. 信息边界与目标（H/T）

个性化 HFL 区域 r 持有由私有数据学习得到的适配器及固定请求版本正式分布 p_r(y|h)，共同共享主干内生轻量起草 q(y|h)（假设不随隐藏区域身份变动，不另部署独立草稿 LLM）；本轮不是概率混合目标，区域输出各自保持 p_r。保护目标为隐藏的区域/模型状态 r；观察者已知授权发布的正式输出 Y、公开前缀及逻辑接受事件 A 或块边界，但不能看到未提交草稿 X、私有 p、额外数据相关元数据；训练消息和真实物理时钟均不在当前保证内。不同 p_r 的输出本身可能泄漏区域身份，因此不声称客户端级 DP、成员隐私或完整轨迹 DP。若所有 p_r 对同一输出 y 正质量，约束给定 Y=y 后任意 r,s 的接受与拒绝概率比 ≤exp(eps)；零质量输出只对双方共同支持定义。实验全 p_r 正值。epsilon=0 对应 I(R;A|Y)=0。与 A/R12 的 A 独立 Y（截止偏差）不同，此处允许 a(y) 随 y 变化，但要求给定 y 后跨区域一致。

#### 2. 精确耦合与闭式最优（T）

定义 m_r(y)=Pr(Y=y,A=1|r,h)，0≤m_r≤min(p_r,q)，a_r=m_r/p_r，u_r=min(1,q/p_r)。先 X~q，接受概率 m_r(X)/q(X)，否则从 (p_r-m_r)/(1-Σm_r) 精确补采（分母为零不拒绝）。一般残差不同于标准最大耦合的 (p_r-q)_+；补采偶然等于候选仍标记拒绝，不复用后续草稿。边际 m_r+(p_r-m_r)=p_r。标准接受 a_r=u_r。

令 u_min(y)=min_r u_r(y)=min(1,q(y)/p_max(y))，p_max(y)=max_r p_r(y)。设 c=exp(eps)。同时约束 a_r≤c a_s，1-a_s≤c(1-a_r) 对所有 r,s。**逐坐标最大可行解**：
a_r*(y)=min{u_r(y), c u_min(y), 1-(1-u_min(y))/c}。
证明：选最小 cap 区域 s，则 a_s≤u_min；两组约束给出另外两个上界。构造解最小值恰为 u_min，最大值不超过两界，所以两类概率比全部满足；逐坐标支配可行解，任意非负区域权重下平均接受率最优。该结论仅在给定 q、单候选复用语义、共同支持与可见逻辑 A 下成立，不是普适时序隐私下界。

eps=0 时共同 a(y)=min(1,q(y)/p_max(y))。设 p_bar(y)=Σ_r beta_r p_r(y)，平均接受率为 Σ_y p_bar(y) min(1,q(y)/p_max(y))。进一步在无模型容量限制、全知 p_bar/p_max 的条件下优化 q：按 p_bar/p_max 从大到小分配 q，单词容量 p_max，直到总质量1。因为 Σp_max≥1 且超出 cap 的边际收益为0，这是分数背包的精确 oracle 解，不是廉价神经草稿头训练算法。HFL 树可分别合并逐词 max 与加权 sum；普通平坦同信息也能合并，计算原语非层级独有。

多步：固定请求长度、K、版本、因果候选条件概率、独立接受/补采随机数、未观察的拒绝后缀不影响状态、全通过 bonus 正式采样。每个位置的条件联合核等于 p_r(y|h) 乘逻辑 a_r 或 1-a_r（bonus 为1）；沿轨迹相乘。故正式序列分布准确，给定完整 Y 后 trace 条件概率比至多 exp(T eps)，eps=0 时条件 trace 分布相同。若公开所有候选/额外时钟、网络或缓存行为，不适用。实际 greedy、top-p 和浮点边界未审计。正式输出本身的 p_r 比值不被约束。

#### 3. 本轮实际执行的实验 E/P/O

源代码：[experiments/A/R14/privacy_probe.js](experiments/A/R14/privacy_probe.js)；原始输出（36 个 seed×异质性组合、每组8上下文汇总及完整枚举检查）：[experiments/A/R14/results.json](experiments/A/R14/results.json)。命令 node experiments/A/R14/privacy_probe.js；实际执行环境为 JS V8 工具，无外部库，LCG 随机数（仅可复算，不用于生产）。实际计算 12 seeds × 3 异质性 × 8上下文=288 配置；8 区域/4边缘、64词，各区域每上下文1024个合成条件观测，.5 加性平滑拟合 p_r，普通共享 q 为 pooled mean。生成 logits 含共享基底、边缘和区域噪声。不是 Transformer 或真实 HFL 优化；q 学习/目标更新间相互作用未验证。单步精确概率计算而非 MC 接受频率；条件互信息 I(R;A|Y) 以 bit 衡量，均匀区域先验。

| 异质 σ | 标准 SD | eps=0 pooled q | eps=.25 | eps=1 | eps=0 optimal q O | A独立Y常数 c 基线 |
|---|---:|---:|---:|---:|---:|---:|
| .15 | 90.10% | 72.97% | 77.63% | 85.11% | 76.77% | 34.52% |
| .5 | 78.12% | 51.64% | 58.54% | 70.48% | 58.18% | 26.28% |
| 1 | 60.13% | 34.06% | 39.20% | 51.94% | 42.34% | 16.38% |

标准条件互信息 .11118/.22503/.30726 bit；eps=.25 为 .002427/.005857/.005685；eps=1 为 .02582/.06355/.09248；eps=0 数值在1e-16内为0。最大条件 log ratio 标准无穷（JSON null，非缺测），eps=.25/1 数值符合上界；eps=0 为0。单步质量约束/边际最大误差1.11e-16（只校验解析核，不是实际 sampler 浮点安全证明）。优化 q 用所有区域正式分布，不能视作可部署或无需通信；高异质仍损失标准接受约17.8pp，较同隐私 pooled q 改善8.28pp。

完整枚举：p1=(.7,.2,.1),p2=(.2,.3,.5),q=(.45,.25,.3)，长度5，K=3，枚举所有243个正式序列及相容逻辑 trace（含 bonus）。所有方法最大正式序列质量误差1.39e-16；eps=0 最大给定序列 trace TV 5.83e-16；eps=.25 TV .161909、最大log ratio1.25（5×.25）；标准TV .8704、最大log ratio无穷。实际枚举逻辑联合核而非执行 Transformer 解码或真实分布随机采样；结果支持数学检查，不能外推物理计时隐私。

P：仅在固定每区域 iid 条件接受率时，K=4，每块期望提交 Σ_{j=0}^4 alpha^j。假设整块成本1.4个AR token，使用1.4/平均块长度作为聚合吞吐成本代理；低/中/高异质，标准无隐私 SD .3411/.4307/.5999，eps=0 pooled .4768/.7012/.9244，eps=0 oracle .4427/.6201/.7940，A独立Y .9184/1.0324/1.1705，AR=1。不同隐私约束不可直接宣布胜负；AR固定长度/节拍为严格逻辑隐私对照。真实条件相关、请求启动、GPU 负载、额外计算与训练通信未知，未测实际时间或CI。oracle高异质推理成本代理仅比AR低20.6%，若推理只占总成本50%，且其余不变，则总成本最多降10.3%，未到G5。

#### 4. 强基线、通信与致命缺陷

- 信息摘要：在每个实际前缀，边缘需得到区域逐词 p_r 才能算 p_max 与 p_bar；四边缘每次上行2V个FP32/边缘，仅 V=128000 时总载荷约4.096MB/前缀（另有下发、客户端边缘、模型前向、版本及安全聚合）。这还要求潜在全部区域执行深层正式模型，可能直接抵消推测节约；离线缓存只对相同前缀/快照有效。不能把全信息 oracle 当作部署速度。使用已有安全聚合/范围界也有独立成本，尚未证明足够紧。
- 如果观察者只看外部流量而不看真 A，普通 SD + 可信输出缓冲/固定节拍/固定大小填充可在改变可观察对象后保留内部接受率；要在同隐私威胁模型下实测延迟和吞吐。若观察者确实看到真实 A，本机制仅有局部数学价值。普通 AR 固定长度节拍、共同公开头、区域个性化头、同信息平坦 max+sum、DP训练后处理（仅在完整可见信息满足相应DP前提时）均为必要强基线，不声称已经实测。
- CRITICAL：缺失 HFL↔SD 双向必要性。此机制适用任何多模型系统，平坦可以复制 max+sum 和闭式接受；没有证明隐私约束改变原 HFL 有效学习计算。与其他团队并发 R02 高度邻近，基础耦合及机制优化已有先例。没有立项依据。MAJOR：精确输出自身泄漏不可消除、完整时间元数据未覆盖；当前前缀全信息的获取可能比直接 SD 贵；高异质接受率下降，严格序列隐私组合限制预算。

#### 5. 本轮核对的原始来源与文献重合

检索日期2026-11-24；关键词 speculative decoding timing side channel privacy, conditional information leakage, acceptance, Pufferfish, model architecture timing, federated learning。核对原始页面/摘要及可用HTML片段，未穷尽全文与附录，不能证明不存在其他等价公式。
- [Wei et al., Privacy Risks of Speculative Decoding in Large Language Models (2024)](https://arxiv.org/abs/2411.01076)：SD 接受模式导致信息泄露、填充聚合；直接问题先例。
- [Majidi et al., Leaky Language Models (2026)](https://arxiv.org/abs/2607.20723)：token timing 识别模型优化/架构。
- [Li et al., From Length to Content, USENIX Security 2026](https://www.usenix.org/conference/usenixsecurity26/presentation/li-sijia)：聚合输出仍有长度侧信道。
- [Kifer & Machanavajjhala, Pufferfish (2014)](https://scholars.duke.edu/publication/1025442)：秘密及条件信息隐私的通用数学框架。
- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding (ICML 2023)](https://proceedings.mlr.press/v202/leviathan23a.html)：标准精确 SD。
- [Elhoushi et al., LayerSkip (ACL 2024)](https://aclanthology.org/2024.acl-long.681/)：内生早退与正式验证。
- [Liu et al., Client-Edge-Cloud HFL (2019 preprint)](https://arxiv.org/abs/1905.06641)：分层部分聚合基础。
- 并发团队 R02 另列 Awan & Rao Privacy-Aware Rejection Sampling、Ben Dov 等 Resistance to Timing Attacks、DP训练及图优化，需后续复核原文；本轮不冒称已重新核对其全文。不存在正式文字查重报告，须规范引用、独立写作。

#### 6. 验收与后续

G1 未通过：暂评约6/10（新的具体受保护对象有区别，但已知SD侧信道、受限接受耦合与并发两团队 R02 高重合，max/sum与背包为经典原语）；G2 未通过（CRITICAL）；G3 部分（条件随机生成证明与枚举，完整greedy/浮点/真实元数据未证）；G4 未执行（无两预训练模型）；G5 未验证（无真实总时间）；G6 未验证（无可靠80%概率）；G7 部分（本轮CPU脚本/输出可复算，完整GPU系统无）。依 Supervisor-Skills idea-evaluator 致命缺陷短路，不对否决实现继续五维乐观评分。好结果预定义仍为同质量、同真实隐私约束及硬件预算下总wall time下降≥15%或同成本有效工作增≥20%，全部 G1–G7 达标。不能报发表概率或虚假成功率。

下一轮：避免重复隐私接受质量/混合/残差/证书路线。优先先测真实小型预训练模型（如 1.5B/3B、冻结共享主体、2–4边缘/8客户端、多域公开数据）是否存在足够显著且不可由普通系统调度或训练消除的**双向计算瓶颈**；若真实业务只需隐藏外部流量，直接采用同隐私预算缓冲/填充基线，不继续本候选。若确需真A条件隐私，先比较同容量公开与个性化头、完整前缀通信和实际GPU时间，按三种异质性/时延、三独立种子、两尺度、真实任务与CI，计训练摊销，1–2×40GB NVIDIA；未执行。不建议在缺独有机制时投入完整训练。保留本轮理论和反例，供其他团队去重。

- 本次追加基于 main commit dcb0236adf18ca4509d736a01733e6c7ab2406b0，README blob 6aee3ff7a86614bc0e5772a48a2cada7890ee615；保留该版本全部内容。


### team-hiercouple/R02｜FPCS：联邦短语级起草的前缀一致性与似然—接受长度冲突

- 2026-11-24，起始 main commit 337195f58d2c286d5ce421d1f103183d3ef071fa，README blob 54306a1d276251d11389a5944af2f9364d004e11，1039行，文件树仅README。读取完整快照并对照PRE-01–11、A/R01–13、7团队R01及内嵌实验；追加前重新读取 blob a5cbc3ee8e8d9d1bac114fb1c5fb7779eed690c7，保留并发新增 HiMerge/HieraCert/HCR 的R02隐私研究（与本轮短语问题不同）。
- **Reject and Pivot**。新对象是每区域固定单一正式目标下的短语预测分布，不再研究混合目标、概率抵消、缓存、梯度修正或奖励分解。短语头已有2026-11-09论文；局部截断oracle为已有采样原语，平坦及普通受限AR草稿可替代，正式HFL学习无反向改变。
- T=数学推导；O=完整信息oracle；P=接受长度/成本代理；本轮实际JS V8有限树枚举，附脚本原始输出，非真实HFL/LLM/GPU或完整R。H=未验证假说。

#### 1. 场景、机制及正确性（T/H）
客户端非IID文本，区域训练轻量内部短语头，可选目标LoRA，云聚合共享参数/短语索引；固定服务区域目标p_e(v|h)，同目标原始tokenizer，D为原始token ID串。目标不修改tokenizer，不做字符串重新分词，不做概率混合。边缘持有私有数据及头，云持公共字典及聚合参数；短语统计可能泄露，无DP保证。一次MLP从已有目标hidden state输出短语分布Q，采样一条短语展开为L个token并批量验证，无独立草稿LLM。优化固定质量和资源下总训练+推理成本/有效工作；完整目标概率、其他区域数据不可免费获取。
固定L，D⊂V^L，|D|≤K。令 M_Q(u)=Σ_{s∈D:u≼s}Q(s)，真实草稿条件 q(a|u)=M_Q(ua)/M_Q(u)。不能以某单条短语概率或独立头概率替代。按标准接受min(1,p/q)，拒绝按归一化[p−q]_+修正并截断；全接受后从p采bonus。决策只依赖已访问前缀及独立随机数，不利用完整短语身份/未来候选；min(p,q)+[p−q]_+=p，沿条件前缀归纳可保随机序列分布（实数、固定快照、正确KV/EOS及采样设置）。greedy展开后按目标argmax验证，tie-break一致。未实现有限精度和KV。
变长扩展H/T：若u本身也是短语且有后代，停止概率Q(u)/M_Q(u)，继续条件q(a|u)=M_Q(ua)/(M_Q(u)−Q(u))。停止抽目标bonus；继续标准SD。不能把未归一子质量作token概率；不提前按已知宏长度/身份更改验证。多短语拼接的潜在分段状态需额外边缘化，本轮未实验。

#### 2. 新的理论资产（T）
设T(D)为前缀树、C(u)为允许下一token集合、P(u)为目标前缀概率。W(ε)=1，W(ua)=W(u)min(p(a|u),q(a|u))。E[接受长度]=Σ_{u≠ε}W(u)，全接受率Σ_{|u|=L}W(u)。因为W(u)≤P(u)，且树外W=0，上限B(D)=Σ_{u∈T(D),u≠ε}P(u)。任何精确耦合且复用字典候选前缀者也受此上界（正式输出前缀必须在树内）。
局部截断oracle q*(a|u)=p(a|u)/Σ_{b∈C(u)}p(b|u)（仅允许子节点），于是q*≥p且W(u)=P(u)，达B(D)、全接受率P(D)。若允许子节点总p质量0，则该目标不可达前缀可任取合法q。短语head可拟合Q*(s)=∏q*，但获取各前缀教师概率不是免费；同支持的受限AR草稿完全等价，target-first全树也达B(D)但评分节点更多。
反例：P(A)=P(B)=.5，P(A|A)=.1，B后A/B各.5；D={AA,BA,BB}，P(D)=.55。整短语NLL理想Q=P(s|D)，Q(A)=1/11,Q(B)=10/11；一阶接受质量.590909、二阶全接受.509091，E[L]=1.10。q*根A/B=.5/.5，A下仅A、B下各.5；E[L]=1+.55=1.55。固定Q的任意合法块耦合E[L]≤Σ_{u≠ε}min(P(u),M_Q(u))（共同前缀事件受双方边际约束），NLL例上界1.140909，仍小于1.55。并非声称TokenVerify优于Block Verification；后者是强基线。
拟训练：区域教师按允许子节点归一构造Q*_e，CE拟合头，联邦同步字典和共享部分、保留区域头。不是实测，不证明正式HFL学习改变。HFL异质性可能让共享头冲突。

#### 3. 实际合成数值检查（O/P）
JS V8，种子11/29/47，σ=0/.75/1.5，各100树/seed/σ，4区域、4词×深4（256路径）；各前缀公共正态logit+区域独立σ正态扰动，softmax；按四区平均路径概率取top24字典（oracle）。900树、3600区域-树，每个四方法。local NLL、local prefix oracle为完整信息，shared NLL/共享oracle为区域短语概率算术平均，不是FedAvg参数训练。实际执行精确前缀枚举，不是Monte Carlo接受率或GPU时间。
| σ | local NLL E[L] | local oracle E[L] | shared NLL E[L] | shared oracle E[L] | local NLL块耦合上界 |
|---|---:|---:|---:|---:|---:|
|0|2.692945|3.072598|2.692945|3.072598|2.887919|
|.75|2.705444|3.086535|2.171978|2.232437|2.882152|
|1.5|2.959601|3.292013|1.743917|1.741195|3.091076|
local oracle达树上界；单步代数误差≤1.11e−16，不代表完整序列实现。高异质共享oracle略低于共享NLL。σ=.75时提交长度比(1+3.086535)/(1+2.705444)≈1.103，同块成本时才是理想吞吐比，尚不足15%时间减少；未测成本、训练、CI或真实模型。
实际重跑代码（复制为 probe.js，node probe.js；LCG非生产RNG）：
~~~javascript
// node probe.js
const A=4,L=4,K=24,R=4,seeds=[11,29,47],sigmas=[0,.75,1.5],trials=100;
let state;
function rand(){state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296}
function normal(){return Math.sqrt(-2*Math.log(rand()))*Math.cos(2*Math.PI*rand())}
function soft(z){let m=Math.max(...z),a=z.map(x=>Math.exp(x-m)),s=a.reduce((x,y)=>x+y,0);return a.map(x=>x/s)}
const prefixes=[['']];for(let d=1;d<=L;d++)prefixes.push(prefixes[d-1].flatMap(s=>Array.from({length:A},(_,a)=>s+a)));
const paths=prefixes[L],interior=prefixes.slice(0,L).flat();
function masses(leaves){let m={'':0};for(let i=0;i<paths.length;i++)if(leaves[i]>0)for(let d=0;d<=L;d++){let s=paths[i].slice(0,d);m[s]=(m[s]||0)+leaves[i]}return m}
function target(base,sigma){let p={'':1};for(let s of interior){let t=soft(base[s].map(x=>x+sigma*normal()));for(let a=0;a<A;a++)p[s+a]=p[s]*t[a]}return p}
function evaluate(p,q){let m=masses(q),v={'':1},length=0,full=0,overlap=0,ceiling=0,err=0;for(let d=1;d<=L;d++)for(let s of prefixes[d]){let h=s.slice(0,-1),pt=p[s]/p[h],qt=m[h]>0?(m[s]||0)/m[h]:0;v[s]=v[h]*Math.min(pt,qt);length+=v[s];if(d===L)full+=v[s];overlap+=Math.min(p[s],m[s]||0);if(m[s]>0)ceiling+=p[s];if(m[h]>0)err=Math.max(err,Math.abs(Math.min(pt,qt)+Math.max(0,pt-qt)-pt))}return [length,full,overlap,ceiling,err]}
function proposal(p,D,mode){let z=paths.map((s,i)=>D.has(i)?p[s]:0),m=masses(z);if(mode==='nll')return z.map(x=>x/m['']);let q={'':1};for(let h of interior){let allowed=Array.from({length:A},(_,a)=>h+a).filter(s=>m[s]>0),den=allowed.reduce((a,s)=>a+p[s],0);for(let s of allowed)q[s]=(q[h]||0)*p[s]/den}return paths.map(s=>q[s]||0)}
const result=[];
for(let sigma of sigmas){let per=[];for(let seed of seeds){state=seed;let sums=Array.from({length:4},()=>[0,0,0,0,0]),n=0;for(let j=0;j<trials;j++){let base=Object.fromEntries(interior.map(s=>[s,Array.from({length:A},normal)])),ps=Array.from({length:R},()=>target(base,sigma)),rank=paths.map((s,i)=>[i,ps.reduce((a,p)=>a+p[s],0)/R]).sort((a,b)=>b[1]-a[1]),D=new Set(rank.slice(0,K).map(x=>x[0])),qN=ps.map(p=>proposal(p,D,'nll')),qO=ps.map(p=>proposal(p,D,'oracle')),avg=qs=>paths.map((s,i)=>qs.reduce((a,q)=>a+q[i],0)/R),qSharedN=avg(qN),qSharedO=avg(qO);for(let r=0;r<R;r++){let qs=[qN[r],qO[r],qSharedN,qSharedO];for(let k=0;k<4;k++){let x=evaluate(ps[r],qs[k]);for(let i=0;i<4;i++)sums[k][i]+=x[i];sums[k][4]=Math.max(sums[k][4],x[4])}n++}}per.push(sums.map(s=>s.map((v,i)=>i===4?v:v/n)))}result.push({sigma,per,mean:per[0].map((s,k)=>s.map((v,i)=>i===4?Math.max(...per.map(x=>x[k][i])):per.reduce((a,x)=>a+x[k][i],0)/per.length))})}
console.log(JSON.stringify({methods:['local_phrase_NLL','local_prefix_oracle','shared_phrase_NLL','shared_prefix_oracle_mix'],metrics:['E_accepted','P_full','block_overlap_upper_bound','tree_ceiling','max_one_step_identity_error'],seeds,trials,R,A,L,K,result}));
~~~
原始输出（per依种子顺序，四方法五指标）：
~~~json
{"methods":["local_phrase_NLL","local_prefix_oracle","shared_phrase_NLL","shared_prefix_oracle_mix"],"metrics":["E_accepted","P_full","block_overlap_upper_bound","tree_ceiling","max_one_step_identity_error"],"seeds":[11,29,47],"trials":100,"R":4,"A":4,"L":4,"K":24,"result":[{"sigma":0,"per":[[[2.6915156729198655,0.5044958084286958,2.885204454797164,3.061195205125159,1.3877787807814457e-17],[3.061195205125159,0.5643526887919164,3.061195205125159,3.061195205125159,0],[2.6915156729198655,0.5044958084286958,2.885204454797164,3.061195205125159,1.3877787807814457e-17],[3.061195205125159,0.5643526887919164,3.061195205125159,3.061195205125159,0]],[[2.6711128470437844,0.4997552296927361,2.8731891036635284,3.067766690593792,2.7755575615628914e-17],[3.067766690593792,0.5613515041296304,3.067766690593792,3.067766690593792,0],[2.6711128470437844,0.4997552296927361,2.8731891036635284,3.067766690593792,2.7755575615628914e-17],[3.067766690593792,0.5613515041296304,3.067766690593792,3.067766690593792,0]],[[2.716206069512358,0.5111415600747905,2.9053629503666034,3.088832331714055,2.7755575615628914e-17],[3.0888323317140536,0.5699063626231322,3.088832331714055,3.088832331714055,0],[2.716206069512358,0.5111415600747905,2.9053629503666034,3.088832331714055,2.7755575615628914e-17],[3.0888323317140536,0.5699063626231322,3.088832331714055,3.088832331714055,0]]],"mean":[[2.692944863158669,0.5051308660654074,2.8879188362757655,3.0725980758110025,2.7755575615628914e-17],[3.0725980758110016,0.565203518514893,3.0725980758110025,3.0725980758110025,0],[2.692944863158669,0.5051308660654074,2.8879188362757655,3.0725980758110025,2.7755575615628914e-17],[3.0725980758110016,0.565203518514893,3.0725980758110025,3.0725980758110025,0]]},{"sigma":0.75,"per":[[[2.7096069913527883,0.5111478457573717,2.88614257316934,3.0899494324274723,5.551115123125783e-17],[3.089949432427472,0.5680855252693886,3.0899494324274723,3.0899494324274723,0],[2.169399067825374,0.3361457931874077,2.5140879624347745,3.0899494324274723,1.1102230246251565e-16],[2.2298625138112795,0.33588156953069415,2.5232394055625385,3.0899494324274723,1.1102230246251565e-16]],[[2.686563202459175,0.5045827072002044,2.863170509688828,3.07041528796576,5.551115123125783e-17],[3.07041528796576,0.5606195485786147,3.07041528796576,3.07041528796576,0],[2.169832649129225,0.3378440327171389,2.5066159136743154,3.07041528796576,1.1102230246251565e-16],[2.2310451523125776,0.336836398470821,2.523318870627361,3.07041528796576,1.1102230246251565e-16]],[[2.7201631844726,0.512792829461436,2.897142074049341,3.0992411210556283,5.551115123125783e-17],[3.0992411210556283,0.5701587898813741,3.0992411210556283,3.0992411210556283,0],[2.176701550955447,0.3376636187328795,2.521182223163601,3.0992411210556283,1.1102230246251565e-16],[2.236403621185639,0.33753248762314086,2.536888049381617,3.0992411210556283,1.1102230246251565e-16]]],"mean":[[2.705444459428188,0.5095077941396707,2.8821517189691694,3.086535280482954,5.551115123125783e-17],[3.0865352804829533,0.5662879545764592,3.086535280482954,3.086535280482954,0],[2.171977755970015,0.33721781487914204,2.513962033090897,3.086535280482954,1.1102230246251565e-16],[2.232437095769832,0.3367501518748854,2.527815441857172,3.086535280482954,1.1102230246251565e-16]]},{"sigma":1.5,"per":[[[2.960656348783743,0.6011537325075953,3.0923747342356442,3.2972268796626913,1.1102230246251565e-16],[3.2972268796626913,0.6483095970063333,3.2972268796626913,3.2972268796626913,0],[1.7442207390344344,0.24833563425573504,2.0297791312647435,3.2972268796626913,1.1102230246251565e-16],[1.743339003990335,0.24302246437350003,2.020850124238561,3.2972268796626913,1.1102230246251565e-16]],[[2.951399196143772,0.5996700103439426,3.0823155925016876,3.28082336204829,1.1102230246251565e-16],[3.28082336204829,0.6464531242154685,3.28082336204829,3.28082336204829,0],[1.7478186619529332,0.24829572504534073,2.0330407987298176,3.28082336204829,1.1102230246251565e-16],[1.7473894001542332,0.24306801388273577,2.025264925917174,3.28082336204829,1.1102230246251565e-16]],[[2.9667465427250987,0.6013658930092539,3.0985363230078065,3.2979886305469406,1.1102230246251565e-16],[3.2979886305469406,0.6483196537456186,3.2979886305469406,3.2979886305469406,0],[1.7397125013544121,0.2445680184910294,2.0382814531321705,3.2979886305469406,1.1102230246251565e-16],[1.7328560549915744,0.23892594625362726,2.027048304777689,3.2979886305469406,1.1102230246251565e-16]]],"mean":[[2.959600695884205,0.600729878620264,3.0910755499150464,3.2920129574193076,1.1102230246251565e-16],[3.2920129574193076,0.6476941249891401,3.2920129574193076,3.2920129574193076,0],[1.7439173007805933,0.24706645926403506,2.0337004610422436,3.2920129574193076,1.1102230246251565e-16],[1.7411948197120477,0.24167214150328767,2.024387784977808,3.2920129574193076,1.1102230246251565e-16]]}]}
~~~

#### 4. 近邻核查
2026-11-24搜索 speculative decoding different tokenizers exact sampling; heterogeneous vocabularies; federated tokenizer training; phrase-level speculative decoding frozen target; variable-length prefix trie; tokenization marginalization; acceptance-aware draft training。原始摘要/部分正文，非逐篇全部附录；arXiv短语论文直连失败，原PDF镜像第一页确认2026-11-09。盲区未公开/未索引论文及随机短语采样理论。
- [Phrase-Level Speculative Decoding on a Frozen Target](https://arxiv.org/abs/2611.12464) 2026-11-09：冻结目标hidden上的MLP短语词表、prefix tree greedy验证，且讨论设备端适应。**直接架构撞题**，随机分析差异不足以独立立项。
- [Lossless SD for Heterogeneous Vocabularies](https://proceedings.mlr.press/v267/timor25a.html) ICML2025：异词表无损SD。
- [Block Verification](https://arxiv.org/abs/2403.10444) 2024/2025：块级最优耦合，强基线。
- [Acceptance-Aware Draft Model Training](https://arxiv.org/abs/2609.24150) 2026：greedy与采样接受长度训练，直接威胁训练目标创新。
- [Variational Speculative Decoding](https://arxiv.org/abs/2602.05774) 2026：序列接受导向草稿训练。
- [Where is the signal in tokenization space?](https://arxiv.org/abs/2408.08541) 2024：非规范tokenization边缘化困难；本轮只处理单短语固定原token序列。
- [Exact Byte-Level Probabilities](https://arxiv.org/abs/2410.09303) 2024：跨词表字节概率。
- [Training a Tokenizer for Free with Private Federated Learning](https://machinelearning.apple.com/research/training-a-tokenizer) 2022：联邦词表训练已有；本轮无其DP保证。
- [SpecVocab](https://aclanthology.org/2026.findings-acl.2000/) 2026：动态草稿词表。[SpecFed](https://arxiv.org/abs/2604.25777) 混合联邦推理。[Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html)、[LayerSkip](https://aclanthology.org/2024.acl-long.681/)。另须比较REST/n-gram/prompt lookup。没有正式文字查重或完整原创性认证，论文规范引用。

#### 5. 强基线、成本、否决及后续
CRITICAL：短语头已有直接论文；受限AR草稿同Q及验证完全匹配、同信息平坦可复制训练和字典共享；仅草稿目标改变而非正式HFL有效学习改变。区域oracle不能推出共享收益。K条长度L短语、d维输入，线性头约Kd乘加、前缀质量O(KL)；受限AR头最多L次较小输出，不能用全词表dV作为唯一基线。单路径仍验证L位置，全树≤KL节点。教师oracle可能需全部树内前缀前向，训练代价高；字典下发K L个token ID（K=8000,L=4,32bit ID为128000字节，未计版本和统计），同步K×d头参数及训练通信均需计。没有真实wall time。
若出现改变核心前提的新证据：1–2×40GB，Qwen2.5-1.5B/3B、8逻辑客户端2边缘，公开数学/代码/通用数据，原始tokenizer，采数千真实前缀，对比共享/私有字典覆盖、教师代价、头拟合误差。基线普通HFL AR、HFL+Medusa/EAGLE、现有Phrase-Level SD greedy及同支持标准随机TokenVerify、Block Verification、受限AR草稿、全树target-first、REST/prompt lookup、同信息平坦个性化FL。两尺度、三异质/网络、三种子、真实任务及CI；计TPOT/P95/吞吐、训练+推理总GPU/网络时间、头同步/失败回退及显存。单卡模拟不冒充物理分布式速度。当前不投入重复GPU实验。
好结果：同质量同资源最强公平基线总wall time−15%或有效工作+20%，greedy及随机精确性、两尺度多设置。无80%证据，不虚构概率。主观分项问题6.5×25%、机制5.5×25%、理论6.5×20%、价值6.5×15%、文献距离4.5×15%=5.95/10，非同行评审。
G1 FAIL近邻及不足8；G2 CRITICAL FAIL平坦、正式HFL无反向作用；G3局部T/整体FAIL（greedy、随机有限精度/完整实现未审计）；G4 FAIL仅有限树；G5 FAIL无wall time；G6 FAIL无80%证据；G7局部脚本输出可复算、完整系统FAIL。**Reject and Pivot**。
保留NLL与前缀最优反例、局部截断oracle和树上界、块共同前缀界、变长停止归一、共享异质性负结果。禁止改名重复短语头/联邦词表/接受损失。下一轮放弃通用草稿优化，先寻找改变正式HFL训练计算的可执行规则及数据驻留/授权约束，证明同信息平坦及AR不可替代再投入真实模型。每轮先读最新commit，结束追加正负结果。


### team-hca/R02｜FJF-SD：分层共同后缀可行性与证书辅助自投机解码

- 日期 2026-11-24；开始读取 main commit 337195f58d2c286d5ce421d1f103183d3ef071fa，README blob 54306a1d276251d11389a5944af2f9364d004e11；当时树中仅 README。写入前 main 60d10d775dcbd098fb5c2c14159d9a91c5aa49d3，README blob d51362c0f5f8bebf4eb78aba4b9c6b0abf78b62e。
- 裁决：**CRITICAL / Reject and Pivot（保留数学反例及实际 CPU 合成证据）**。不是概率混合/残差抵消/切面重启。与 A/R13 奖励证书、PRE-09/HieraCert 证书以及 A/R03 跨版本缓存有概念近邻，但本轮对象为不同机构约束语言的共同后缀存在性。未证明真实 HFL 应用需求、双向必要性，普通 AR 可复用完全相同的证书。
- 证据 T=数学推导；本轮实际执行 JS V8 CPU 合成模拟（局部可复算，非完整 R）；P=载荷/查询代理；H=待验证假说。无真实模型、训练、网络、GPU 时间。

#### 1. 系统场景与数学对象（H/T）
Client-edge-cloud 常规参数 HFL 学习一个共享自回归模型 p_theta，客户端持有私有非 IID 数据和版本化约束语言 L_i（例如跨机构工作流/代码规范）。区域协调 AND 查询，云有共享模型、前缀和内生 Medusa/LayerSkip 类起草路径；约束不直接上传，允许授权的候选后缀查询及必要时精确交集求解（若不允许精确回退则不保证协议）。约束可能来自本地规则或学习出的有限状态模型，后者质量/学习机制尚未证明，不能把 AND 当作 FedAvg。可信节点，固定会话版本、公共 token 字母表、最大长度及 EOS；不宣称 DP/密码学隐私。目标为所有约束的交集 L，且起始有可完成序列。定义有限剩余长度下 S_i(s)={z:sz∈L_i}，共同可行性 I(s)=1{交集 S_i(s)非空}。局部掩码 AND=AND_i 1{S_i(s)非空} 是上界而非精确 I。反例 S_1(A)={X},S_2(A)={Y}，双方局部可行但无共同后缀；可取 L1={AX,BX}, L2={AY,BX}，全局只允许 BX，错误局部掩码接受 A 后必死。无约束或共享同一后缀时退化；仅有前缀局部禁令且保证共同可扩展时不存在该障碍。

#### 2. 正式目标与计算问题（T/H）
固定前缀 s，定义 M(s)={v:I(sv)=1}，r(v|s)=p_theta(v|s)1{v∈M(s)}/sum_{u∈M(s)}p_theta(u|s)，假设可行 token 概率正、EOS/长度规则统一。目标是该逐步掩码 AR 的序列分布，不是原始无约束模型，更不是 p_theta(Y|Y∈L) 的全序列条件分布。例 L={AX,BX}, p(A)=p(B)=.5, p(X|A)=.9,p(X|B)=.1，则逐步掩码第一步 .5/.5，真正条件分布 .9/.1。不得混称无损。
输入：树、各机构私有约束、模型版本、前缀、实际 logits、候选链、公共随机性、带宽/RTT、历史证书；输出同目标精确 greedy 或随机序列。允许成员查询、AND 归约、完整交集求解或可审计精确否定证明。优化质量匹配的总训练+生成 wall time，计证书维护/获取、训练、验证、通信、失败回退和 KV。

#### 3. 证书机制与理论（T/H）
对候选 sv，保存旧快照共同合法后缀 z；向所有节点查询 1{svz∈L_i(current)} 并在边缘 AND 后上传。全为真即正证书，失败/无缓存仅表示未知，不能判全局不可行；调用精确交集求解，若权限禁止则停止、不假装正确。固定公共随机数 g_v iid Gumbel(0,1)，对目标 logits/T+g_v 降序，按顺序查询可行性，首个 I=1 的 token 即 masked Gumbel-Max；greedy 不加噪声并统一 tie-break。内部头可在一条候选链上起草，目标批量验证各候选前缀并生成该前缀独立目标随机数；验证顺序仅使用已访问前缀的随机数，命中则继续，否则提交目标 token 并停止、回滚。可用已确定随机性做有条件耦合但不得改变目标噪声的边际或偷看未来。该 target-first 验证不等于最大耦合，接受率未证。按条件分布归纳，可保证理想算术下相对于指定 r 的序列正确；top-p 需对精确掩码后的分布计算，浮点/终止/实际实现未审计。
证明：共同后缀满足所有语言 => I=1；证书失败并不推出 I=0。精确回退使 oracle 与真实 I 相同，故降序首个合法项等于合法集合中 Gumbel argmax，服从 r；greedy 同理。多步按标准目标优先验证与条件链式法则。已有 Gumbel-max/可行性检查，不是新原语。
复杂度边界：一般多个 DFA 的交集非空为 PSPACE-complete（无界长度且 automata 数目可变）；把新起始 token 接到原 DFA 可将该决策嵌入 I(sv)。不能将此结论直接套到固定短有限长度；有限显式后缀域 B 的 bitset 交集为 O(mB) 工作。只允许黑盒后缀 membership 时，最坏无解与最后一个未查询后缀共同可行不可区分，需检查 B 个后缀（非一般通信下界）。
成本代理：m 个节点、B 个显式后缀、旧证书存在比例 c、所有查询中证书成功比例 h，完整位图每次 mB bit；缓存查询每节点 ceil(log2 B)+1 bit，未命中全量回退。比值 1-h+c*(ceil(log2 B)+1)/B，理想串行 RPC 轮数 1+c-h 对比直接位图 1；不含缓存初建/版本/协议/前缀/聚合拓扑等。位图压缩、短路、批量查询、普通 AR 共享证书、预编译 product automaton 都是强基线。
学习设想（H）：HFL 聚合共享候选/证书预测器，SD 不可行或未知案例反馈本地预测头训练；但普通主动学习/CEGIS/AR 也提供相同标注，尚无独立学习机制或净收益，G2 CRITICAL。

#### 4. 实际执行 CPU 合成实验（P）
脚本：[experiments/team-hca/R02/witness_probe.js](experiments/team-hca/R02/witness_probe.js)，命令 node experiments/team-hca/R02/witness_probe.js。本次 JS V8 执行相同逻辑，LCG seeds 11/29/73；每密度×漂移×种子 4000 个独立状态，m=4,V=16,B=64；共 72000 状态。旧成员 bit 独立 Bernoulli(density)，当前以 drift 概率翻转；预先知道旧共同后缀（未计构建费用），随机 token 排序对应均匀目标的排名。对同一排序比较直接逐 token 全量交集与旧证书+精确回退，全部状态零输出差异（无解时双方都返回 undefined，非成功生成）。没有真实 HFL 参数训练/模型/GPU、实际延迟或置信区间；以下为三个种子指标算术平均，载荷只是代理，非真实通信。局部假阳性=各机构分别存在后缀的 token 中没有共同后缀比例；命中/回退以实际被查询 token 为分母。
| 旧密度 | 翻转率 | 局部假阳性 | 证书命中 | 回退 | 载荷比(证书/直接) | 无解状态比例 | 平均查询 token |
|---|---|---|---|---|---|---|---|
| .2 | .02 | .878771 | .089152 | .910848 | .921422 | .124750 | 7.221750 |
| .2 | .15 | .637745 | .048368 | .951632 | .961934 | .001167 | 2.780667 |
| .5 | .02 | .016016 | .908429 | .091571 | .199248 | 0 | 1.014667 |
| .5 | .15 | .016333 | .515075 | .484925 | .592615 | 0 | 1.017333 |
| .8 | .02 | 0 | .917917 | .082083 | .191458 | 0 | 1 |
| .8 | .15 | 0 | .524667 | .475333 | .584708 | 0 | 1 |
解释：稀疏约束时局部可行性误导明显，而旧证书收益仅约4–8%；宽松低漂移时约80%代理载荷降低，但局部假阳性极低，且普通缓存 AR 完全匹配该协议的证书访问成本。提高漂移使旧密度 .2 的当前密度升高，假阳性下降不是证书鲁棒性证据。旧证书 oracle 构建成本未计，非部署可行证据。没有训练或多步实验，不外推实际生成性能。最强同信息平坦可执行同样 AND/精确回退。可否决，不追加 GPU 以凑数。

#### 5. 原始文献核查与撞题（2026-11-24）
本轮检索 arXiv、ACL Anthology、PMLR、NeurIPS 官方页面，关键词 speculative constrained decoding grammar masks; distributed private constraints automata intersection; grammar aligned decoding future validity; federated automata learning; compositional active learning; specification guided generation。读取以下原始论文页面/摘要（部分全文未核查，不宣称完整逐式去重）：
- [DOMINO, ICML 2024](https://proceedings.mlr.press/v235/beurer-kellner24a.html)：语法约束+预计算+SD 已有；不是首次结合。
- [Grammar-Aligned Decoding, NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/2bdc2267c3d7d01523e2e17ac0a754f3-Abstract-Conference.html)：局部掩码与全局条件目标区别、ASAp 近似未来概率。核心目标区分已有。
- [Constrained Decoding with Speculative Lookaheads, NAACL 2025](https://aclanthology.org/2025.naacl-long.239/)：草稿/目标+奖励验证，非本轮共同后缀交集。
- [Flexible and Efficient Grammar-Constrained Decoding, ICML 2025](https://proceedings.mlr.press/v267/park25l.html)：高效 mask 编译，必须对比预编译而非弱逐词扫描。
- [Future Validity is the Missing Statistic, 2026-05-08](https://arxiv.org/abs/2605.07698)：约束 SD 的局部/全局目标差异，近邻高风险。
- [Specification-Guided Generation, 2026-10-30](https://arxiv.org/abs/2610.41511)：符号推理剔除不能延伸满足规范的 token，高度接近共同可行性 oracle。
- [Finite Automata Intersection Non-Emptiness, 2021](https://arxiv.org/abs/2108.05244)：经典 PSPACE 障碍；不是本轮新定理。
- [Compositional Active Learning of Synchronous Systems, 2025](https://arxiv.org/abs/2504.16624)：组合自动机学习近邻，反馈学习不能直接宣称原创。
- 其他待复核：Decode-Time Grammars (2607.18357)、CASD、XGrammar、分布式 CSP/私有集合交集、CEGIS、增量模型检查、符号自动机；未公开/未索引/术语差异为盲区。无实际文字查重，需规范引用；原理重复不能靠改写解决。

#### 6. 强基线、门槛、否决与后续
B0 普通 HFL AR+完整精确交集，B1 同样缓存证书 AR（最关键，证书通信完全匹配），B2 普通 HFL+内生头 SD+精确交集/预编译，B3 同信息同委托平坦 AR/SD，B4 DOMINO/现有 grammar SD 在同约束可访问时的强实现，B5 共享精确约束 product 编译（在权限允许时），B6 惰性位图交集/压缩/批量请求。质量与目标相同，近似方案另列。若未来有新机制，1–2×40GB NVIDIA 上 Qwen2.5 1.5B/3B LoRA、2–4 边缘8–16逻辑客户端、公共工作流或结构化代码真实任务、三异质性/延迟/种子；测合法率、质量、接受长度、TPOT、TTFT、P95、吞吐、字节/RTT、GPU time、证书维护成本、总训练推理 wall time 与 CI。未执行。好结果按仓库 G5：真实总时间降≥15%或等预算有效工作增≥20%，且同目标精确、质量匹配、双向因果及同信息平坦强基线优势。无80%把握的依据。
主观创新评分（等权、非同行评审）：问题7、机制5.5、理论6、价值6.5、与既有距离5.5，合计6.1/10。G1 FAIL；G2 FAIL/CRITICAL（AR 与平坦替代，HFL 训练只提供模型）；G3 局部实数定理成立，完整实现/top-p/有限精度未验证，整体未通过；G4 FAIL（仅合成）；G5 FAIL（无真实时间）；G6 未验证；G7 局部脚本可复算、完整机制未过。
**Reject and Pivot。** 保留量词顺序反例、目标定义区分、正证书与否定证据不对称、缓存代理成本式及真实执行的合成结果。不要再以普通共同证书缓存、分布式 AND 或标准语法 SD 作为 HFL 双向创新。下一轮优先从共享模型学习本身的可执行计算对象出发，寻找一种正式训练目标或更新过程必然产生的 SD 可复用结构，并先构造普通 AR/同信息平坦不可匹配的因果假设；若不能成立，在 CPU 阶段停止。先读最新全库，再研究，失败也追加。


### HIERASPEC/R02｜PFSC：公开草稿和私有目标的条件隐私约束耦合

- 日期2026-11-24；研究起始 main commit 337195f58d2c286d5ce421d1f103183d3ef071fa、README blob 54306a1d276251d11389a5944af2f9364d004e11；全文1039行、递归树仅README，代码内嵌。对照PRE、A/R01–13及全部R01。提交前读并发HiMerge/R02、HieraCert/R02、HCR/R02、A/R14，读取A/R14实验脚本及结果摘要。前两队固定p、私有q；后两队固定公开q、私有p但隐藏未提交X，本轮额外公开X，故其仅对(Y,A)的闭式不可直接用。本轮相邻但有不同约束，不能把同轮多个隐私边界当独立顶刊创新。
- 裁决 **CRITICAL / Reject and Pivot**，保留数学边界。T=限定推导，实际CPU LP为合成O/P证据，非真实加速；H=假说。无真实预训练LLM、HFL训练、GPU、网络或wall time。

#### 1. 场景、问题、数学结果

客户端非IID数据经HFL学习公共主干/内部候选头，区域保留私有目标适配器。固定公开前缀h、快照和q，秘密e为同一授权边缘服务可能的适配器状态（非已公开路由ID），目标p_e全支持且非混合。观察者见X、真实复用A、正式Y，不见私有随机数/logits。不保护训练消息/真实时序等额外通道，不宣称训练数据DP；若q受秘密影响须另外保护或条件固定。
Γ_e(x,y)≥0，行边际q、列边际p_e；A=1 iff X=Y（相同token可复用），给定Y后A由X决定。对所有e,f,x,y要求 Γ_e(x,y)/p_e(y)≤exp(ε)Γ_f(x,y)/p_f(y)，最大化Σ_e w_eΣ_xΓ_e(x,x)。固定p,q为LP，E V²变量、O(E²V²)约束；须全知私有分布，仅oracle。给定X从Γ_e(X,·)/q(X)采样保持边际。若Y本身对相同邻接满足ε_out-DP，联合(Y,X,A)满足(ε_out+ε)-DP；未证明ε_out，不等于用户级DP。

T1：标准最大耦合若p_e(x)≥q(x)>p_f(x)，事件X=x,A=0在e为零、f为正，有限ε条件隐私不成立。二元p+=(.51,.49),p-=(.49,.51),q=(.5,.5)，标准接受.99，输出最大概率比仅.51/.49≈1.040816；公开拒绝支持不同。
T2（秩障碍）：ε=0时K(x|y)=Γ_e(x,y)/p_e(y)对e共同。行边际给出Σ_y(p_e(y)−p_1(y))K(x|y)=0。若差向量张成V−1维零和空间，K(x|·)只能常数，由边际该常数=q(x)，唯一Γ_e=q⊗p_e，最优接受Σ_e w_e p_e·q。独立target-first匹配达界。二元例.5；任意小满秩扰动仍成立，完全相同分布则普通最大耦合可达。E≪V一般不满秩，不能外推真实LLM。线性代数非全新隐私原理。
T3（低秩构造）：p_e(v)=a_e(g(v))r_g(v)，q(v)=b(g(v))r_g(v)，组内r_g严格共享。给定X独立采样G~a_e，若G=g(X)提交X，否则Y~r_G。边际Y精确；给定Y=y，X以b(g(y))概率等于y，其余组h按b(h)r_h(x)，与e无关。ε=0，接受Σ_g a_e(g)b(g)。p1=(.4,.4,.2),p2=(.3,.3,.4),q=(.35,.35,.3)，{A,B}/{C}组接受.62/.54，均值.58；独立.335，无约束.9，LP确认.58最优。共享组内比例必须严格成立，近似低秩参数不保证；普通分层AR也可用因子分解。若改变目标架构，精确仅针对新目标，需测质量。
真实条件核、因果草稿和固定目标可逐位置归纳序列边际；真实块预采样、未提交后缀、EOS、bonus、greedy、top-p、浮点、物理计时及完整trace未审计；单步ε不等于请求级ε（基本组合Σ ε_t）。若完整可见依赖是DP发布模型的后处理，训练DP预算不另耗；与本轮输出条件要求不同。

#### 2. 实际CPU实验（O/P）

Python3.13.5 NumPy2.3.5 SciPy1.17.0，HiGHS；两个固定例及5词5模型合成Dirichlet(1)，p=(1−δ)q+δ随机分布，均匀q，种子20261124/25/26。无wall time、真实数据或正式CI。固定例最大边际误差1.67e−16、约束违反2.78e−17，非严格浮点隐私证明。以下实际执行代码的等价版本，复制为pfsc_probe.py，运行python pfsc_probe.py，依赖上述版本：

~~~python
import numpy as np, scipy, scipy.optimize as so, sys
def solve(p,q,eps):
    E,V=p.shape
    eq=[];rhs=[]
    for e in range(E):
        for x in range(V):
            a=np.zeros((E,V,V));a[e,x,:]=1;eq.append(a.ravel());rhs.append(q[x])
        for y in range(V):
            a=np.zeros((E,V,V));a[e,:,y]=1;eq.append(a.ravel());rhs.append(p[e,y])
    ub=[]
    for e in range(E):
        for f in range(E):
            if e==f:continue
            for x in range(V):
                for y in range(V):
                    a=np.zeros((E,V,V));a[e,x,y]=p[f,y];a[f,x,y]=-np.exp(eps)*p[e,y];ub.append(a.ravel())
    c=np.broadcast_to(-np.eye(V)/E,(E,V,V)).copy()
    r=so.linprog(c.ravel(),A_ub=ub,b_ub=np.zeros(len(ub)),A_eq=eq,b_eq=rhs,bounds=(0,None),method="highs")
    assert r.success,r.message
    g=r.x.reshape(E,V,V)
    return -r.fun,max(np.max(abs(g.sum(1)-p)),np.max(abs(g.sum(2)-q))),max(0,np.max(np.asarray(ub)@r.x))
print(sys.version.split()[0],np.__version__,scipy.__version__)
eps=[0,.01,.05,.1,.25,.5,1]
for p,q in [(np.array([[.51,.49],[.49,.51]]),np.array([.5,.5])),(np.array([[.4,.4,.2],[.3,.3,.4]]),np.array([.35,.35,.3]))]:
    print("ind",np.mean(p@q),"max",np.minimum(p,q).sum(1).mean(),"rank",np.linalg.matrix_rank(p[1:]-p[0]))
    print([(e,*solve(p,q,e)) for e in eps])
for delta in [.01,.1,.3]:
    vals=[]
    for seed in [20261124,20261125,20261126]:
        rng=np.random.default_rng(seed);V=5;q=np.full(V,1/V)
        p=(1-delta)*q+delta*rng.dirichlet(np.ones(V),size=V)
        vals.append([np.mean(p@q),np.minimum(p,q).sum(1).mean()]+[solve(p,q,e)[0] for e in [0,.01,.1,.5,1]])
    print(delta,np.round(np.mean(vals,axis=0),6),"range",np.round(np.min(vals,axis=0),6),np.round(np.max(vals,axis=0),6))
~~~

实际LP最优平均接受率：
|ε|二词|三词|
|---:|---:|---:|
|0|.500000|.580000|
|.01|.599959|.585994|
|.05|.777641|.609862|
|.1|.856915|.639500|
|.25|.930553|.715106|
|.5|.962068|.780044|
|1|.979066|.840864|
|无约束|.990000|.900000|

5词5模型三种子均值（独立/无约束/ε=0/.01/.1/.5/1）：
- δ=.01: .200000/.996922/.200000/.626905/.918078/.982585/.991550。
- δ=.1: .200000/.969225/.200000/.302723/.627539/.865537/.926435。
- δ=.3: .200000/.907674/.200000/.237895/.443313/.717220/.820306。
这些是LP代理接受率，不是采样频率、真实模型接受长度或加速。

#### 3. 原始文献（2026-11-24）

检索 speculative decoding privacy leakage acceptance timing; privacy-aware rejection sampling; output-conditioned timing privacy; perfect privacy null space; private GPT speculative verification; federated personalization differential privacy; conditional optimal transport。实际打开原始页面/摘要及部分正文，未穷尽附录。
- [Privacy Risks of Speculative Decoding in Large Language Models](https://arxiv.org/abs/2411.01076) 2024-11：接受模式泄漏、聚合/填充。
- [Privacy-Aware Rejection Sampling](https://www.jmlr.org/papers/v24/21-0870.html) 2023：拒绝计时泄漏、数据无关时间采样。
- [A Framework for Differential Privacy Against Timing Attacks](https://arxiv.org/abs/2409.05623) 2024-09：明确输出条件timing privacy；本轮借用定义思想，额外固定边际对角耦合。
- [An Efficient Private GPT Never Autoregressively Decodes](https://proceedings.mlr.press/v267/li25q.html) ICML2025：公开起草+私有安全验证、蒸馏；密码学强基线。
- [On Perfect Privacy](https://arxiv.org/abs/1712.08500) 2017预印/2021修订：披露核、完美隐私效用及零空间近邻。
- [SpecTr](https://arxiv.org/abs/2310.15141) 2023：SD最优传输；新增跨秘密条件约束但未证明原创性。
- [Personalization Improves Privacy-Accuracy Tradeoffs in Federated Learning](https://proceedings.mlr.press/v162/bietti22a) ICML2022：局部/全局个性化隐私。
- [Privacy-Aware Split Inference with Speculative Decoding for Large Language Models over Wide-Area Networks](https://arxiv.org/abs/2602.16760) 2026-02：隐私切分和lookahead。
盲区：近期安全SD、Pufferfish、密码学oblivious verification、OT隐私约束、未公开未索引工作；无商业文字查重。不能以改写解决内容重复。A/R14及HCR/R02均有隐藏X的条件接受闭式和oracle q优化，本轮全公开X导致更强约束；两者不得直接比较接受率。

#### 4. 成本、强基线、验收

CRITICAL：满秩零额外泄露普通独立target-first达界；低秩分组被普通分层AR及同信息平坦学习替代，无SD反向学习贡献。E=4,V=128000时LP变量65,536,000,000，不可部署。可信边缘若能隐藏X/A并固定可见时序，普通SD保留内部接受率，需公平计缓冲、填充、排队及安全通信。公共q获取、组内严格共享的个性化质量代价、跨位置隐私预算未解决。四候选且恒定独立α=.5时理想提交1.9375，α=.99时4.900995，仅代理。
H（未执行）：若新证据改变前提，在1–2×40GB、Qwen2.5-1.5B/3B、8逻辑客户端2区域、公共内生头/私有LoRA、GSM8K/MBPP/公开文本三域三种子三非IID或RTT设置，测真实概率差谱、组内比例偏差及输出条件trace，比较HFL AR、独立匹配、普通HFL+Medusa/LayerSkip、固定节拍/缓冲、同目标分层AR、安全验证及同信息平坦。计任务质量、全部训练推理时间、字节/RPC、接受长度、P95、隐私会计、CI；单卡逻辑模拟不称并行。若普通基线可替代立即否决。
主观评分：问题7.5×25%、机制6×25%、理论7×20%、价值7×15%、距离5×15%=6.575≈6.6/10；并发近邻使风险更高，不上调。好结果须同目标/威胁模型/质量/隐私预算下总真实wall time降≥15%或同成本有效工作增≥20%，其余G门槛不降低；无80%把握依据。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（单步T，greedy/完整多步/精度未验）；G4 FAIL（CPU合成）；G5 FAIL（无真实总时间）；G6 FAIL（未知80%）；G7 PARTIAL/FAIL（内联脚本和数值可复算，完整系统未复现）。
**Reject and Pivot。** 保留固定边际条件隐私LP、支持反例、满秩零泄露障碍、共享组构造及强基线。下一轮从真正HFL学习对象和可执行规则出发，先证普通AR、同信息平坦、本地执行不可替代；用真实小模型或trace证实问题再投入训练。禁止以隐私模块换名重启。本轮追加使用最新blob保护并发，不覆盖原文。

- 实际写入前 README blob：db2f19d6f3a13d5d63c1501be8fd0d8a06df66f2；保留原文全部内容。


### HC/R02｜TJSA：目标保持的联合投机特征子空间与分层截断障碍

- 2026-11-24；起始 main commit 337195f58d2c286d5ce421d1f103183d3ef071fa，README blob 54306a1d276251d11389a5944af2f9364d004e11（1039行，143220字节）；完整读取该快照及随后并发追加。PRE/A/R01–13、HC/R01、所有R01均已对照；后来读取HiMerge/HieraCert/HCR/A/R14的隐私结果、HierCouple/R02短语头前缀目标、HCA/R02共同后缀证书、HIERASPEC/R02公开草稿的条件隐私秩障碍。后者的概率差零空间与本轮参数算子行空间不同；不据此宣称新颖。 
- **CRITICAL / Reject and Pivot**。T=有限数学推导，E=实际执行CPU合成数值实验，P=代理指标，H=未验证假设。没有GPU、真实HFL训练、真实模型生成、wall time或CI；不外推速度或成功概率。

#### 1. 问题、机制、理论

客户端持私有非IID文本；边缘聚合有效矩阵更新，云低频聚合一个全局目标LLM及其内生预测头；服务时冻结目标版本。目标模型的局部线性适配块T∈R^(m×d)，多步草稿投影D∈R^(n×d)，共享输入h∈R^d及低秩投影A∈R^(r×d)，目标BAh、草稿CAh。不是完整Transformer的线性化等价，不另部署独立LLM。T*=Σw_e T_e，D*=Σw_e D_e；目标BA=T*严格不改，优化D近似及全流程成本。云不知道各区域算子，区域持自身算子，通信计费。可信聚合，不宣称隐私。真实HFL学习仍是普通NLL与草稿蒸馏，未发明反向学习原语。

坐标反例：标量A1=B1=C1=1、A2=B2=C2=-1，各端BA=CA=1，但逐因子平均为0。一般(BR^-1)(RA)=BA。已有精确产品聚合/对齐可解，非新问题。

T1：t=rank(T*)≤r，P_T为目标行空间正交投影，最小目标保持草稿Frobenius误差平方=Σ_{j>r-t}σ_j²(D*(I-P_T))。因为S=row(A)必须包含row(T*)，固定S最优Dhat=D*P_S；S=row(T*)⊕U，dim U≤r-t，Eckart–Young给出残差最佳奇异子空间。A取正交行基，B=T*A^T，C=D*A^T。联合精确最小秩rank([T*;D*])；t>r不可行。Frobenius在零均值各向同性输入时等于输出MSE，不适用于任意真实隐藏分布。经典线代而非全新定理。

T2：目标正交补中M1=diag(2,1)、M2=diag(-2,1)，等权。局部rank1截断后平均0，原平均diag(0,1)的全局rank1误差0，先截断误差平方1。故该局部截断算法不存在相对全局最优有限乘法界；不是所有通信协议下界。绝对界||M-Mhat||_F≤Σw_e||M_e-C_b(M_e)||_F+||Σw_e C_b(M_e)-C_k(Σw_e C_b(M_e))||_F。尾范数在抵消时可能松。

T3：温度τ>0、无截断的softmax，logit误差δ时TV≤min(1,||δ||∞/τ)，沿路径概率导数p_i(δ_i-E_pδ)/τ积分；若原草稿并非教师需加原误差，不能推多步长度。目标保持，实际q及精确残差采样在实数和正确因果前缀下保留正式随机序列，greedy须用目标逐步argmax；未审计实际KV、EOS、top-p、有限精度实现。

TJSA：云发送P_T（计获取及广播），边缘分解D_eP_T与D_e(I-P_T)，残差先过采样rank b，云聚合取剩余k=r-t；尾范数不足则补传或退回独立投影。强替代：共享线性sketch先聚合后恢复；精确有效矩阵/LoRA因子堆叠后全局SVD；同信息平坦、固定公共空间、独立头、普通多任务共享子空间。每边dense残差nd标量，rank b因子约b(n+d)，另计保护部分、基底同步、补传和协议；现有局部低秩因子可直接堆叠，不得只比较dense弱基线。若可消除的重复投影占全流程不到15%，即使消除也达不到G5时间门槛。真实成本未知。

#### 2. 实际实验（CPU E/P）

[脚本](experiments/HC/R02/probe.js)，[逐种子汇总结果CSV（六位精度）](experiments/HC/R02/results.csv)。命令 node experiments/HC/R02/probe.js；本轮实际执行JS V8工具，LCG+Box-Muller，seed11/23/37，4边缘、16维（4保护/12残差），每配置100对角矩阵组×64高斯上下文，3异质性共900组、57600上下文/方法。一致信号位于残差前4维，另8维为同向/独立/配对抵消噪声；未压缩平均矩阵产生toy教师logits，非真实LLM。比较局部b=4/8/12后根k=4 vs 精确聚合后根k=4(root)。NMSE为每组平方误差/真实范数平方；alpha为解析单步1-TV，不是实际采样频率。b12所有组等root。

|异质性|b4 alpha|b8 alpha|root alpha|
|---|---:|---:|---:|
|同向|67.67%|67.67%|67.67%|
|独立|79.82%|83.97%|85.16%|
|抵消|83.46%|83.46%|100%|

抵消组b4/b8 NMSE=1、root=0；双倍秩仍丢失一致信号。root只是Frobenius最优而非一般接受率最优。该极端人为构造不是现实HFL出现频率证据；未运行模型训练或GPU。代码可复算逐种子完整精度输出。

#### 3. 原始文献与致命基线

2026-11-24检索 exact federated LoRA aggregation, subspace alignment, hierarchical low-rank truncation, SD feature representation, signed matrix sum sketches。核对原始页面/摘要，EAGLE-3正文；部分2026 arXiv HTML不可用，仅摘要，不声称穷尽全文。
- [FedEx-LoRA ACL2025](https://aclanthology.org/2025.acl-long.67/) 精确产品聚合；[CFLoRA](https://arxiv.org/abs/2609.33147) 互补因子；[FedLore](https://arxiv.org/abs/2610.01620) 共享低秩投影/子空间碎片化；[Rethinking Factor Sharing](https://arxiv.org/abs/2608.09742) rank-aware共享空间；[Subspace-Constrained FL](https://arxiv.org/abs/2606.22724) 破坏性聚合。四篇2026 arXiv仅摘要。
- [FedPissa ICML2026](https://proceedings.mlr.press/v306/he26p.html)、[SDFLoRA IJCAI2026](https://ijcai.org/proceedings/2026/533)、[FedRoRA](https://aclanthology.org/2026.findings-emnlp.608/)、[FedSVD](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ad922aa85d4027ff3502e8e5f406e828-Abstract-Conference.html) 选择性聚合、对齐及全局SVD。
- [Git Re-Basin](https://arxiv.org/abs/2209.04836) 对称性；[EAGLE-3](https://arxiv.org/html/2503.01840v3) 内生多层特征；[ReeFL](https://proceedings.mlr.press/v235/lee24h.html) 联邦早退/蒸馏；[Practical Sketching Algorithms](https://doi.org/10.1137/17M1111590) 通用矩阵sketch；[Frequent Directions](https://arxiv.org/abs/1501.01711) 合并式协方差/行流，不直接等价有符号矩阵求和；[Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html) 精确采样。
盲区：真实EAGLE联合训练、非线性、多任务共享投影、未索引和全文公式；无实际文字查重。PRE-05/A/R02–03/A/R07–08已有子空间/接口近邻；本轮不同于概率混合和隐私，但与通用低秩矩阵压缩高度重合。普通AR、普通HFL+内生头、产品精确聚合+根SVD、共享sketch及同信息同委托平坦皆为强基线。完整聚合在压力测试恢复全部方向；未实测同预算sketch或独立头实际时间，不能声称优于它们。CRITICAL：HFL没有不可替代性，SD没有反向改变学习；标准矩阵工具可复制主要机制。

#### 4. 验收和下一轮

主观分项问题6.5×25%、机制5×25%、理论6×20%、价值6×15%、文献距离4×15%=5.575≈5.6/10，非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（局部数学、完整精确解码未验）；G4 FAIL仅合成；G5 FAIL无真实总时间；G6 FAIL无80%依据；G7 PARTIAL/FAIL（局部可复算，完整系统未复现）。好结果：同目标/质量/预算下训练推理全流程真实时间降≥15%或同成本有效工作+20%，两预训练尺度、三种子三非IID/时延设置及CI，全部G门槛。未执行的最小实验H：1×40GB Qwen2.5-1.5B、后扩3B/7B，8客户端2边缘、rank8/16/32、数学代码通用公开语料，先测真实联合秩谱、跨边缘截断误差、共享投影真实耗时占比；计训练、通信、TPOT、质量、显存、回退与CI，与上述强基线公平比较。单卡逻辑模拟不是物理并行。因致命缺陷不建议为本方案投入GPU。
**Reject and Pivot**。保留目标保持联合秩界、截断反例、绝对误差界、CPU合成结果。禁止改名复刻普通低秩对齐、压缩、概率混合或隐私耦合。下一轮先从真实小模型trace验证数据驻留、跨层异步屏障、SD流水线的实际瓶颈，预设普通AR/SD与公平平坦，寻找学习和执行双向不可替代原语。先读最新库再研究，正负均追加。

- 写入前 README blob：4e929ec4c7d31767be44ccb761272b290875e31c；完整保留旧内容。



### team-hieracert/R03｜FPQ-SD：联邦预测状态商空间与验证树精确合并的能力—复用障碍

- 日期2026-11-24；研究开始 main commit a64930b8caf002a13dd4d293661b54725371bd2d，README blob 4e929ec4c7d31767be44ccb761272b290875e31c，1571行；验证前轮已读内容为原样前缀，读取新增HiMerge/R02、HCR/R02、A/R14、HierCouple/R02、HCA/R02、HIERASPEC/R02及内联程序；读独立文件 experiments/A/R14/privacy_probe.js、results.json（解析汇总及部分逐种子结果）、experiments/team-hca/R02/witness_probe.js。提交前读新增HC/R02目标保持联合子空间、局部SVD截断抵消反例；该方向与本轮预测状态等价不同。未独立复跑其他团队实验。读取及写入版本分开记录：上传前 README blob 11af5f16375b12850c7e97fe8bafc8bbc1e34e14，已保留并发记录。
- **Reject and Pivot（CRITICAL）**。有限状态可精确合并不同历史的验证计算，但普通AR精确状态缓存访问集必为本轮预验证访问集的子集，且小状态空间直接预计算即可；平坦聚合完全匹配。本轮没有证明HFL与SD双向必要性，不批准立项。T=明确假设下理论；E=本轮实际CPU合成拟合及配对抽样；P=访问计数/通信载荷代理；H=未验证研究假说。无GPU、真实Transformer训练、实际网络及wall time；不冒称独立完整R级。

#### 1. 场景、计算对象及机制（T/H）

明确替换目标架构：历史h经φ变为确定性预测状态s∈有限集合S，pθ(v|h)=Pθ(v|s)，s'=T(s,v)。版本内T、P固定，词表、采样策略、位置/EOS/停止条件明确；若存在位置或额外上下文依赖须并入状态。客户端私有非IID数据，区域汇总状态条件充分统计量，云学习共享转移/输出模型（本轮实验只拟合输出表，转移固定为后缀shift）。允许授权共享计数、模型参数及状态ID，非隐私证明；本轮不使用概率混合、区间证书、跨版本缓存、DP或短语头。低成本内生起草函数从较短状态预测top-B候选；目标表或重模型头验证。目标精确仅相对于定义的有限状态模型，不能称原始Transformer无损。模型若能在固定完整状态下定义预测头，可直接按状态/版本缓存输出；不得将不同KV仅凭向量相似合并。

FPQ-SD：基于严格等价类构建分层候选DAG（状态可循环，按深度展开为DAG；跨深度缓存同一平稳状态），每个独立状态最多计算一次目标分布，随后 target-first 逐位置采样 p（不是标准最大耦合拒绝采样）。若正式token在候选树对应节点出边，沿其继续，否则提交正确token并停止该块；全匹配后bonus正式采样。真实采样随机数按提交位置独立，不能把相同状态的一次随机输出跨位置复用；相同状态概率可缓存，若一个位置只访问一条路径可共享其计算。greedy取目标argmax统一tie-break；top-p需同一精确P和规则，有限精度真实执行未审计。优化总学习+推理时间、通信、峰值存储与质量，而非仅树节点减少。拟学习共享状态划分，利用SD访问频率约束状态压缩（H），但这仍是普通状态抽象/压缩目标，尚未证明独立反向学习收益。

#### 2. 数学资产与可证伪边界（T）

T1 充分性：若等价关系 s~t 满足 P(.|s)=P(.|t) 且所有token v 的 T(s,v)~T(t,v)，则对任意未来序列 y_1:n 两状态概率相同；逐步按条件概率乘积归纳。商空间可精确执行，同一等价类的输出分布可共用。当前预测相同不充分：两个状态均给A概率.5，A之后一个给C概率.9另一个.1，则AC概率.45/.05。软相似、共享后缀或近似隐藏向量不能自动证明完全预测等价。经典概率自动机/预测状态/双模拟，非首次发现。

T2 聚合闭合（严格条件）：各区域共享T、φ，聚合的是有效输出logits z_g(s)=Σw_e z_e(s)。若每一区域 z_e(s)−z_e(t)=c_e·1 且后继仍在共同等价类，则聚合差=(Σw_ec_e)·1，softmax不变。局部等价关系交集为安全共同划分，根的真实等价关系可能更粗（抵消），所以交集未必最优。参数FedAvg经过非线性模型不等于logit平均，不能套用；区域模型局部状态编号/转移不同需先对齐，普通平坦亦可求交及聚合。计数聚合与参数FedAvg不同。有限状态最小化/partition refinement 是强基线。

T3 压缩质量下界：令随机历史H及下一token Y来自同一固定评价分布，S=φ(H)确定。对任意仅用S的条件预测r，E[-log r(Y|S)]≥H(Y|S)，相对使用完整历史的Bayes最优损失H(Y|H)增加至少 I(Y;H|S)，等号在r=P(Y|S)时成立。这是标准信息论，不是对实际Transformer损失的普适下界，也不声称模型必然达到Bayes最优。训练可能产生新目标，但必须测任务质量。相同一步分布但不同未来的合并仍可能失败。

T4 最强缓存反例：固定同一目标、初始状态及每个实际提交位置的随机数U_t；每轮先预验证全部候选状态，随后严格按目标分布抽样并在树外停止。普通AR每步访问的状态集合A，SD实际预验证集合D必满足 A⊆D（包括发生拒绝/bonus所在的父状态；按每块和时间归纳）。故同容量足以容纳完整访问集、同版本、无逐出、每个状态一次成本相同的精确缓存下，AR目标分布计算次数≤SD，跨请求/全局缓存只需相同版本和状态键。这不是任意GPU延迟下界：批量并行、通信屏障、memory-bound模型可以以更多工作换更短墙钟。不能因此声称所有SD都不快。小S时直接预计算全部分布/转移也可替代。更强精确平行采样对照（T/H，未执行）：预先为每一位置抽独立U_t，构造随机映射 F_t(s)=T(s,InvCDF(P(.|s),U_t))；映射合成结合，可对F_t做parallel prefix scan，以O(L|S|)工作（工作有效scan、表查及构图另计O(L|S|V)朴素CDF）和O(log L)合成依赖深度得到所有实际状态，按U_t输出token，和AR使用相同随机数完全相同。空间和状态数可能大，且不是普通Transformer方案；小状态空间必须纳入该强基线，不能人为限制AR只串行计算。该分析基于经典并行前缀，不声称首次提出。

T5 贪心/随机正确性：固定目标状态与真实概率，DAG target-first逐位置输出同AR，候选只决定何时结束块、不决定输出；独立按提交位置采样且所有已访问目标状态在D中可查，链式法则成立。greedy同理。对不同目标或缓存过期无保证。验证树节点减少与最大耦合接受率不等价。

#### 3. 实际执行 CPU 合成拟合及模拟（E/P）

代码：[experiments/team-hieracert/R03/probe.py](experiments/team-hieracert/R03/probe.py)；逐种子原始计数：[experiments/team-hieracert/R03/results.json](experiments/team-hieracert/R03/results.json)（由实际stdout JSON转存）。运行 python experiments/team-hieracert/R03/probe.py；本次实际环境 Python3.13.5 NumPy2.3.5 CPU。种子11/29/73，V=4，k=1/2/4/6，S=4^k；每状态每客户端16个独立合成条件观测，4客户端2边缘，公共高斯logits+区域和本地噪声（σ=.5），先边缘后云加计数，根加.5平滑，p归一化；q为同计数表按最后k−1 token聚合（内部简化模型）。固定转移T(s,v)=(4s+v) mod S，top2分支、候选深度4含深度4 bonus分布共31节点/块。20请求×128token×12配置=240请求、30720配对token。每请求随机初态及相同独立均匀随机数，AR vs DAG target-first；两者输出逐token一致，AR访问集合为SD子集的assert全部通过。模拟在内存中已知完整p，计数是如果每个状态需要真实目标计算时的访问代理，并非实际执行了相应次数的神经网络前向。无GPU、wall time、CI、真实数据或真实训练。DAG每块计独立状态（允许跨深度合并），持久缓存每请求重置（不跨请求），两者公平；未执行parallel scan。

下表三个种子的均值，分母各2560个提交token：
|k|状态数|均匀状态平均 KL(p||q),nat|SD块数|普通树节点|DAG块内独立节点|SD每请求缓存合计|AR每请求缓存合计|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|4|.198731|1061.333|32901.333|3079.333|80|80|
|2|16|.217465|1012.667|31392.667|8312.333|317.333|317.333|
|4|256|.217331|1018.667|31578.667|29526.333|3243.667|1923.667|
|6|4096|.207374|1038|32178|32000.333|21285|2518|

DAG相对未合并树节点减少约90.64%、73.52%、6.50%、0.55%。较缓存AR，SD独立目标访问比1、1、约1.686、8.453。块数降低不能直接换算时间；在小状态空间AR每请求已缓存几乎全表，直接全版本预计算可能更优。大状态空间DAG合并收益消失，投机产生大量未使用的状态。均匀状态KL是将模型状态由k压至k−1时在合成拟合目标p上的损失代理，不是真实语言任务perplexity；不同k独立模型不能直接比较质量。该实验仅否决此具体状态与访问计数机制，非所有分支合并方案的否定。

成本P：若直接以FP32缓存所有V维分布，空间4|S|V字节（另需状态索引/转移），4词、4096状态仅65536B；对真实词表V=128000，即使仅4096状态也约2.10GB，且状态压缩质量未知。两边缘各上传32位S×V计数时总8SV字节/聚合轮，4096×4例131072B；若真实V=128000则约4.19GB，尚未计4客户端→边缘上传、下发、版本、状态划分、重训练及通信协议，稀疏计数/压缩为强基线。普通平坦在同委托下可用完全相同充分统计量；本轮无网络计时。理论预验证计算工作比不能当端到端速度。

#### 4. 原始文献核查与原创性（2026-11-24）

检索词 speculative decoding merging branches DAG equivalent states; suffix automaton; finite state autoregressive predictive equivalence; probabilistic automata learning state merging; recurrent/hybrid self-speculation; parallel sampling. 实际打开下列原始论文页面、摘要及部分可用正文；未穷尽全文、专利、未索引工作：
- [Graph-Structured Speculative Decoding](https://aclanthology.org/2024.findings-acl.677/) ACL Findings2024：利用DAG合并草稿重复序列，DAG不是新对象；本轮是正式预测状态精确等价，仍高重合风险。
- [SAM Decoding](https://aclanthology.org/2025.acl-long.595/) ACL2025：后缀自动机用于快速检索草稿，区别于目标状态等价。
- [SuffixDecoding](https://papers.nips.cc/paper_files/paper/2025/hash/b7aea253ab34a773967f1e4cdea9e4fb-Abstract-Conference.html) NeurIPS2025：利用缓存文本和可预测序列，相关强基线。
- [FlexFringe](https://doi.org/10.46298/lmcs-21(3:31)2025) LMCS2025：概率确定性自动机的经典状态合并与学习实现，直接威胁学习机制原创性。
- [Component-Aware Self-Speculative Decoding in Hybrid Language Models](https://arxiv.org/abs/2605.01106) 2026-05：利用SSM/线性注意力子图作内部起草，不能声称首次使用内部递归状态。
- [H-Spec](https://arxiv.org/abs/2609.24197) 2026-09：Mamba内部草稿及parallel scan，去除草稿KV，需区分正式状态复用。
- [Reviving Any-Subset Autoregressive Models with Principled Parallel Sampling and Speculative Decoding](https://arxiv.org/abs/2504.20456) 2025-04：调整目标架构和训练实现精确并行生成，已有机制级先例。
- [Parallel Sampling via Autospeculation](https://arxiv.org/abs/2511.07869) 2025-11：通过计数实现平行采样，相关非普通SD基线；[SpecMamba](https://arxiv.org/abs/2509.19873) 2025-09：SSM投机验证和回滚。另有概率双模拟、因果状态、有限自动机最小化及prefix scan经典理论。没有实际文字查重报告，独立撰写/规范引用不等同于内容原创。去重：A/R07–08低秩接口/仿射编译、A/R10依赖分解为数学近邻，前者不同于离散状态等价，后者不同于验证图合并；HierCouple/R02的短语字典只改草稿，HCA/R02为约束共同可行性，HC/R02为矩阵目标保持压缩。本轮并未证实新的HFL独特计算原语。

#### 5. 强基线、失败、验收与后续

CRITICAL：缓存AR访问工作不超过本方案；小状态可预计算，普通并行prefix scan是强基线；真实Transformer完整状态一般不能按有限后缀严格合并，强制压缩存在条件互信息损失；HFL计数聚合可被同信息平坦精确替代，SD没有实质改变正式学习更新。GSD/概率自动机学习已有先例；不能把DAG、普通状态缓存或常规压缩正则作为新机制。此处否决的是立项，不是宣称所有有限状态SD墙钟必慢。
若新证据改变前提才做真实预实验（H，未执行）：1–2×40GB NVIDIA，选可获取1.5B/3B两尺度预训练模型或具有明确有限状态语义的混合架构，先在公开代码/数学/结构化任务上收集真实分支状态，检查是否有非平凡严格相等的完整预测状态（近似状态需另报误差，不可当精确），固定目标与版本，测质量/状态表可行性。三种子、三种非IID/时延，8逻辑客户端2边缘，真实训练与通信；对照普通HFL AR、精确状态缓存AR、全表预计算/并行scan（适用时）、HFL+LayerSkip/Medusa、GSD、H-Spec/组件式自推测（适用架构）、同信息同委托平坦。计全流程训练+推理/失败回退/状态划分与版本同步时间、TTFT/TPOT/P95/吞吐、bytes/RPC、GPU time、显存、质量/CI；单卡模拟不是物理并行。优先测真实问题存在性，不建议为已否决有限状态构造投入GPU。
主观等权创新评分：问题6.5、机制5.5、理论6、价值6.5、距离4.5=5.8/10，非同行评审。好结果预定义：同正式目标、任务质量、资源下对最强公平基线总真实wall time下降≥15%或同成本有效工作增加≥20%，且全部G门槛。现无证据支持80%概率。G1 FAIL（5.8及直接近邻）；G2 CRITICAL FAIL（AR与平坦替代、无双向机制）；G3 PARTIAL/FAIL（有限状态理想证明、配对随机实验，实际greedy/top-p/有限精度/LLM未审计）；G4 FAIL（仅合成条件计数拟合）；G5 FAIL（只有工作代理）；G6 FAIL（证据不足）；G7 PARTIAL/FAIL（实际脚本与逐种子输出可复算，完整系统未复现）。**Reject and Pivot。** 保留等价的递归条件、聚合闭合严格前提、信息损失下界、AR缓存集合反例、parallel scan强基线及实测合成结果。禁止将相似状态当精确状态、重复普通DAG/状态压缩或用树节点数冒充时间。下一轮优先读取真实可获得的预训练模型执行trace，测量异步学习版本、实际网络/计算瓶颈与推理占总成本；寻找正式学习更新不可避免且能够被自投机机制反向减少的计算工作，在真实测量前不继续堆叠toy模块。每轮读最新commit，正负结果均追加。


### team-hca/R03｜FSRA：联邦参数坐标一致性、联合低秩表示与接受率虚高反例

- 日期 2026-11-24；起始 main commit 929708a994b8557b39c532369aeb420cd131ed0a，README blob 4e929ec4c7d31767be44ccb761272b290875e31c（1571行）；读取该快照，核对既有记录及递归树。与前轮新增 HiMerge/R02、HieraCert/R02、HCR/R02、A/R14、HIERASPEC/R02 隐私边界和 HierCouple/R02 短语头、HCA/R02 共同后缀证书不同。
- **并发去重：** 提交前读取新增 HC/R02 TJSA 及 experiments/HC/R02/probe.js、results.csv。HC 独立提出相同坐标反例与联合行空间秩界，并进一步证明目标保持的最优草稿残差及局部截断不可交换；本轮不重复宣称联合低秩原创。独立增量仅为显式接受率虚高反例、普通坐标对齐对照与共享初始化近邻条件下的数值证据。
- **裁决 CRITICAL / Reject and Pivot**。证据 T=限定数学；E=本次真实 CPU 数值运行（非 HFL 训练、非 LLM/GPU 实验）；P/O=解析接受率、通信/运算代理或完整信息；H=未验证。无实测端到端时间、实际 SD 生成或完整 R 级复现。

#### 1. 场景与计算矛盾（T/H）

客户端私有非IID数据、共享初始化；区域/云分层聚合。抽象线性子模块 A_i∈R^(r×d)，正式 B_i、内生草稿 C_i∈R^(V×r)，p_i=softmax(B_i A_i x)，q_i=softmax(C_i A_i x)。固定请求模型版本、词表、前缀，使用共享表示，不另部署独立完整草稿模型。仅线性子模型，不声称 LayerSkip/EAGLE 真实非线性计算等价；真正轻量路径的成本未测。客户端持 A/B/C 及私有数据，边缘/云允许交换参数或有效矩阵；不宣称 DP。任意可逆 G_i 下 (G_i A_i,B_i G_i^-1,C_i G_i^-1) 保持 p_i、q_i、局部损失和接受率，但逐因子 FedAvg 不具坐标不变性。目标 P*=Σw_i B_i A_i、Q*=Σw_i C_i A_i 是本轮明确定义的有效logit算子平均参考，不等于概率混合，不是一般 HFL 参数平均的正式目标，需单独测任务质量。
反例：两个等权客户端 A1=1,A2=-1,B1=(2,-2)^T,B2=-B1,C1=(1,-1)^T,C2=-C1，x=1。两端 p=(.98201379,.01798621)，q=(.88079708,.11920292)，最大耦合接受率 .89878329；直接平均 A=B=C=0，正式/草稿均均匀，接受率1，正式分布TV偏差.48201379。对正确token A 的NLL从约.01815升至.69315。说明 SD 高接受不保证聚合质量，绝非证明真实预训练HFL经常产生随机旋转。若同初始化、小步长且坐标稳定，该矛盾可很弱；一般 Transformer 的对称群受非线性、归一化、残差和权重绑定约束，不应任意插入旋转。

#### 2. 机制与理论（T/O/H）

形式问题：固定线性算子参考 P*,Q*，选择共享 r 维 A、B、C 和合法层级聚合协议，最小化 ||BA-P*||_F²+λ||CA-Q*||_F² (λ>0)，同时约束任务质量、通信预算及全流程成本。各区域先计算完整有效矩阵加权和，云再合并；因线性和可合并且坐标不变。堆叠 Z=[P*;sqrt(λ) Q*]，用 rank-r SVD Z_r=U_r Σ_r V_r^T，取 A=V_r^T，B=Z_r 上V行乘A^T，C=Z_r下V行乘A^T/sqrt(λ)。Eckart-Young：最小损失 Σ_{j>r}σ_j²(Z)，联合精确 iff rank([P*;Q*])≤r。证明任意 [B;sqrt(λ)C] A 的秩≤r，反过来任意rank≤r矩阵均可因子分解。Frobenius 只对各向同性输入对应期望 logit MSE；真实输入需协方差加权，真实接受率不是此目标。
目标必须优先：若 rank(P*)>r，不可能严格恢复；若 rank(P*)≤r，可固定目标行空间，再分配剩余秩给草稿残差，HC/R02 已给更强闭式最优结果（不重复声称）。反例 P*=(1,0),Q*=(0,1),r=1，任何共享 A 均不能同时精确。若λ>1，普通联合SVD会牺牲目标而保留草稿，故不能用接受率替代质量约束。采用更稳妥目标保持式或回退独立草稿头，普通多任务表示学习/低秩压缩可实现。
对齐强基线：公共旧表示 A0，各客户端解 min_{R orthogonal}||RA_i-A0||_F，若 SVD(A0 A_i^T)=UΣV^T，则 R=UV^T，同步变换 B_iR^T,C_iR^T；再普通 FedAvg。纯正交坐标扰动、相同函数下可恢复；非正交/不同子空间不保证。FedRot-LoRA已有正交对齐，FLoRG已有Procrustes及Gram聚合；固定共同基底、精确产品聚合、普通蒸馏亦为强基线。
标准 SD 接受 min(1,p/q) 与拒绝按(p-q)_+修正保持所部署 p 的采样分布；若聚合近似改变 p，精确性只针对新 p，不能声称保留 P* 教师。greedy 用正式目标argmax并匹配候选。完整随机序列、top-p、有限精度、KV/EOS及实际批验证未实现。拟以正式验证误差/质量约束指导共享子空间更新（H），但普通监督蒸馏/多任务学习同样可利用，不构成反向学习新原语。
成本（P）：每客户端传 A,B,C 共 r(d+2V) 数值；两有效矩阵 dense 2Vd，d=V=4096,r=16 时分别196608与33554432数值（FP16约.393MB vs67.109MB，未计协议）；不是强基线必需发送dense：可传原因子并在边缘/云堆叠，精确加权算子分解秩可达nr；对齐额外矩阵乘与r×r SVD，固定参考可本地执行。共享A相比两独立rank-r投影仅在同输入、两路径都需计算时最多节省rd乘加（独立2r(d+V)、共享r(d+2V)），已有普通共享表示基线也节省；计全Transformer、KV、训练及通信后收益未知。相同信息同委托平坦可复制所有聚合、对齐及SVD。层级局部截断可能丢失跨区抵消方向，HC/R02 已有反例与合成数据；不得把根SVD代理最优当层级贡献。

#### 3. 本轮实际 CPU 实验（E/P/O）

脚本 [experiments/team-hca/R03/probe.py](experiments/team-hca/R03/probe.py)，逐种子输出汇总（8位小数）[results.txt](experiments/team-hca/R03/results.txt)。命令 python experiments/team-hca/R03/probe.py，Python3.13.5、NumPy2.3.5。seed11/29/73；2种坐标条件×3漂移(.02/.2/.6)×3种子×80矩阵组=1440组，每组8客户端、d=16,r=4,V=8、128个Gaussian输入；184320输入/方法。A0为正交基，B0随机，C0=B0+.15 Gaussian；本地因素加高斯扰动，坐标条件为恒等或独立QR正交G，人为stress test。未训练任何模型，未模拟真实异步网络。五方法：直接因子平均、对齐后平均、完整有效算子加权后联合rank4 SVD、分别rank4 SVD（双输入投影，容量更高，不宣称等成本）、完整参考oracle。指标相对Frobenius误差、内部接受α=1-TV(p_method,q_method)、正式参考重叠β=1-TV(p_ref,p_method)；均为解析概率，不是实测采样/任务准确率/速度。三种子等权均值如下（每格 target error / α / β，四位小数）：
|随机坐标|漂移|直接|普通对齐|联合SVD|分别SVD|全参考α|
|---|---|---|---|---|---|---|
|否|.02|.0001/.9072/1.0000|.0001/.9072/1.0000|.0001/.9072/1.0000|.0001/.9072/1.0000|.9072|
|否|.2|.0135/.9027/.9956|.0134/.9027/.9956|.0102/.9027/.9967|.0083/.9026/.9974|.9026|
|否|.6|.1159/.8688/.9610|.1155/.8679/.9611|.0877/.8668/.9703|.0710/.8621/.9765|.8573|
|是|.02|.7472/.9695/.7707|.0001/.9070/1.0000|.0001/.9070/1.0000|.0001/.9070/1.0000|.9070|
|是|.2|.7484/.9647/.7692|.0135/.9022/.9956|.0103/.9021/.9967|.0083/.9020/.9974|.9019|
|是|.6|.7601/.9355/.7556|.1162/.8671/.9611|.0881/.8650/.9705|.0708/.8603/.9766|.8559|
随机坐标下直接聚合接受率虚高同时正式输出明显偏离参考；普通对齐在小漂移基本消除错误。无随机坐标/中低漂移时普通平均本来已很接近参考；高漂移联合SVD仅改善代理误差，分别SVD质量更好（额外容量），全参考平坦可完全匹配。随机与非随机条件使用相同seed但QR额外消耗随机数，非逐样本配对；不得将二组差直接解释为严格因果估计。无正式CI、实际wall time或真实质量；合成数据无法估计实际坐标漂移出现频率。未独立重跑代码；逐种子记录为实际输出的舍入值，脚本可重跑完整精度。

#### 4. 近邻原始文献核查（2026-11-24）
关键词 federated LoRA rotational misalignment, gauge alignment, FedMA, model merging permutation symmetry, shared low rank regression, early exit self speculative decoding, hidden representation alignment, multihead merging。实际检索原始论文页面/摘要和Git Re-Basin部分正文；非穷尽公式/全文。直接撞题：
- [FedRot-LoRA, ICML2026](https://proceedings.mlr.press/v306/zhang26cc.html)：正交Procrustes对齐客户端低秩更新，直接覆盖对齐机制；不是本队新原语。
- [FLoRG, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/85ce42cbc4842572cfe7a9a9b8fc95ec-Abstract-Conference.html)：Gram矩阵聚合及Procrustes。
- [FedEx-LoRA, ACL2025](https://aclanthology.org/2025.acl-long.67/)、[FLoRA, NeurIPS2024](https://arxiv.org/abs/2409.05976)、[FlexLoRA, NeurIPS2024](https://arxiv.org/abs/2402.11505)：精确产品/堆叠/低秩重分配。
- [FedMA, ICLR2020](https://arxiv.org/abs/2002.06440)、[Git Re-Basin, ICLR2023](https://arxiv.org/abs/2209.04836)：参数/隐单元对齐先例。
- [FedRep, ICML2021](https://proceedings.mlr.press/v139/collins21a.html)：共享低维表示及个性化头。
- [LayerSkip, ACL2024](https://aclanthology.org/2024.acl-long.681/)、[EAGLE-3, 2025](https://arxiv.org/abs/2503.01840)：自投机、内生特征预测基础。
- [Single-Input Multi-Output Model Merging](https://arxiv.org/abs/2504.11268) 2025：共享编码器与任务解码器表征对齐，近邻；[DySD](https://aclanthology.org/2026.findings-emnlp.45/) 2026：目标变化下草稿适应，不能将适配本身当原创。HC/R02列出更广泛近邻，包括ReeFL、FedLore、FedSVD及sketch；先读其记录。盲区：未公开、未索引工作、真实多层模型对称群、最新合并与多任务低秩论文全文。无正式文字查重，规范引用不能解决机制重复。

#### 5. 强基线、G1–G7及后续
普通HFL AR（质量参照）、直接HFL+内生头SD、同容量固定共同基底、FedRot式对齐、FedEx/FLoRA式完整有效更新、目标保持HC/TJSA、各自头蒸馏、普通共享表示、同信息平坦FL+SD。只有在出现新计算对象且真实trace证实问题后才考虑1–2×40GB Qwen2.5-1.5B/3B两尺度，8–16逻辑客户端/2–4区域、rank8/16/32、三非IID/时延设置三种子、公开数学代码通用文本；先用共享初始化的真实LoRA检查对齐残差、正式质量、接受率、训练和解码总时间。单卡模拟不宣称物理多节点加速。好结果同正式目标/质量/预算下全流程实际wall time下降≥15%或有效工作+20%，且双向必要性及其余G通过；无80%证据。
主观等权创新分（问题6.5、机制4.5、理论5.5、价值6.5、文献距离3.5）=5.3/10，非同行评审；并发HC及FedRot直接重合，不上调。G1 FAIL；G2 CRITICAL FAIL（同信息平坦、普通AR/多任务对齐替代、无新反向学习）；G3 PARTIAL/FAIL（限定线性理论与标准SD，完整实际greedy/随机精度未证）；G4 FAIL仅CPU合成；G5 FAIL无真实总时间；G6 FAIL证据不足；G7 PARTIAL/FAIL局部代码及逐种子数据可复算，完整系统未复现。**Reject and Pivot**。
保留明确的接受率虚高反例、独立合成证据、共享初始化对照、目标保持优先约束；不得改名重投普通坐标对齐/联合低秩/层级截断。下一轮不要再先构造抽象算子后寻找应用；优先用真实预训练小模型或公开可复核执行trace，测量 HFL 训练/生成共同出现的实际跨层瓶颈和工作占比；先建立同信息平坦、普通AR和SD强基线及可证伪的双向因果，再创造新机制。若真实瓶颈比例不足以达到G5立即停止。
- 追加前 main commit e1295ab7f355deadd7cf299fd8e54642527a5a7c，README blob 87f5125bce40de27159631196a242f9a97e11f4e；保留并发 HC/R02 及历史内容。


### team-hcr-sd/R03｜FBE-SD：投机验证块专家集合的高阶信息障碍与联邦部署

- 2026-11-24；起始 main commit 23d38083e8db9d9da2274f59a3f5ee343d776188，README blob 11af5f16375b12850c7e97fe8bafc8bbc1e34e14，243484 bytes。核实上一轮完整已读版本12fd3bf8b1b7f8d5c8a186d6e552d2457f0e33b1为原样前缀，阅读新增A/R14、HierCouple/R02、HCA/R02、HIERASPEC/R02、HC/R02全文及HC/R02脚本/结果；检查文件树。并发HieraCert/R03代码和结果已读；写入前再读HieraCert/R03 FPQ-SD（状态商空间、AR缓存强对照）、HCA/R03 FSRA（参数坐标对齐、接受率虚高）的完整新增记录，与本轮专家集合不同。未独立复跑他队实验。追加前main d27b72ada362b76c92c4800fe3731333a39d73e0，blob b1d2d50b049522040a80d0433516ba43360d5a98；第一次写入前检查发现并发更新并停止，保留新增原文。
- **CRITICAL / Reject and Pivot**。不重启R01概率混合、R02隐私、共同后缀或低秩。与A/R04/A/R06联合轨迹/统计有思想近邻，但对象为固定目标MoE验证块专家集合及启动成本。EcoSpec已研究专家集合；超图通信分区是直接先例。T=限定数学，E=实际CPU数值，P=成本代理，O=已知分布oracle，H=假设；无真实LLM/HFL训练/GPU/网络/wall time。
 
#### 1. 场景和机制（H/T）
Client私有非IID数据，edge聚合MoE专家/适配器和内生头，cloud聚合共享主干、协调专家存储与版本。假设各区域获授权使用同一固定正式模型p_theta，不修改目标router/专家/权重/tokenizer；非独立完整草稿LLM。单层实际验证块S含全部候选（包括后续被拒绝的分支）激活的专家集合。区域e容量B_e、本地集合C_e，其余专家远程执行/加载，假设单远程owner且同层能合并请求。启动成本lambda_e，线性远程token-expert调用成本b_e，J_e(C)=lambda_e Pr(S not subset C)+b_e E[N_remote]。优化同质量下总训练推理时间；本轮只研究固定快照、候选策略的一层启动成本。多owner可求各owner非空访问指示的加权和，但实际并行、拥塞、权重读取、数据传输不同，非完整延迟模型；若all-to-all固定启动或lambda=0，则退化。无DP保证。
机制：客户端统计真实验证块集合n_i(S)，边缘累加n_e(S)=Σ_i n_i(S)，求argmax_{|C|≤B}Σ_{S⊆C}n_e(S)（相同容量、线性项相同的特例），多owner使用超图连接成本；同步部署版本后测新trace。普通平坦同信息可累加并运行相同优化；普通batch超图分区也可使用。拟让SD块统计影响HFL专家部署/训练，尚无新正式梯度机制；若改变router则正式模型变了，须另证质量。本轮仅物理位置调整。若保持同版本、真实路由、目标logits、实际q、合法KV和数值规则，则标准SD接受min(1,p/q)与正残差保持随机序列，greedy用目标argmax；没有实现或审计浮点dispatch、EOS、top-p，不能宣布G3通过。
 
#### 2. 高阶信息障碍（T）
Z_i=1{专家i在块内出现}。Pr(远程启动)=1−E[∏_{i∉C}(1−Z_i)]，一般依赖所有阶共现；纯线性调用数则边际足够，不能把启动率当字节数。
反例：3位置，各二选一专家(0,1),(2,3),(4,5)，块必有3不同专家。x∈{0,1}³，P_±(x)=(1±ρ(−1)^(x1+x2+x3))/8，0≤ρ≤1。对剩余自由位求和，奇偶项消失，故两个分布的所有1/2阶边际完全相同。容量3缓存若每对取一个（transversal），对应唯一x_C，Pr(S⊆C)=(1±ρ paritysign(x_C))/8；否则命中0。ρ>0时两个分布的最优缓存互不相交；正确奇偶启动率(7−ρ)/8，错误(7+ρ)/8。ρ=1时同边际及pair统计，启动率.75 vs 1，相差25pp；任意3专家的期望远程调用数均1.5，线性字节项不变。任一只使用精确population一二阶统计的确定性选择至少一分布regret≥ρ/4；随机最坏期望regret≥ρ/8（均匀先验两个分布时任何transversal平均成本7/8）。一般K对的偶/奇分布低于K阶边际完全相同，正确命中2^(1−K)、错误0，随机最坏regret≥2^(−K)。这不是所有有限样本二阶算法的下界：二阶估计量联合波动可间接携带高阶信息，实际实验已观察此现象。亦非所有通信协议下界；经典包含排除/超图/奇偶分布数学。
 
#### 3. 本轮实际CPU实验（E/P/O）
[源代码](experiments/team-hcr-sd/R03/probe.js)；[逐种子原始汇总](experiments/team-hcr-sd/R03/results.json)。命令 node experiments/team-hcr-sd/R03/probe.js；实际JS V8运行、无外部依赖，LCG仅用于可复算。6专家、3位置、容量3、4区域×2客户端、每客户端n=16/64/256块；奇偶符号交替，ρ=0/.4/1，seed11/29/73，每配置200四区域试验，共5400试验21600区域决策。训练计数枚举20种缓存，最大化专家命中数、共现对数、完整块命中数；另对照全局共享缓存、已知分布oracle、平坦同区域计数。测试直接按已知分布计算期望启动率；不是实际网络调用频率/时间，未运行真实SD/LLM、训练、正式CI。三种子均值：
|ρ|n|边际|pair|区域完整集合|全局共享|oracle|同信息平坦|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0|16|.881510|.875000|.875000|.875000|.875000|.875000|
|0|64|.877240|.875000|.875000|.875000|.875000|.875000|
|0|256|.875729|.875000|.875000|.875000|.875000|.875000|
|.4|16|.879979|.854625|.828625|.875000|.825000|.828625|
|.4|64|.875552|.853958|.825000|.875000|.825000|.825000|
|.4|256|.874937|.854958|.825000|.875000|.825000|.825000|
|1|16|.866250|.751250|.750000|.875000|.750000|.750000|
|1|64|.869063|.750104|.750000|.875000|.750000|.750000|
|1|256|.874896|.750000|.750000|.875000|.750000|.750000|
27配置中edge_exact与flat_same_info最大差0（同计数同算法构造性等价，不是独立物理实现）。ρ=1时有限样本pair几乎完全匹配本方法；population不可辨识不保证有限样本优势。ρ=.4,n=64，启动代理相对pair改善约3.39%，对共享缓存5.71%（共享不是最强基线）。若其余成本相同，端到端收益至多启动成本占比×该相对改善；平坦差0。极端反例25%启动节约在启动占总成本50%时仅降12.5%，不足G5。无真实延迟结果。
通信代理：toy八集合32位计数32B/客户端，6个边际24B，15个pair60B（不含ID/版本/协议）；M专家集合最多2^M，最多k个活跃不同专家时Σ_{j=0}^k C(M,j)，稀疏计数可远少但成本随语料与版本变化。完整trace采集、模型前向、合并与存储迁移/回退、训练费用必须计。 
   
#### 4. 文献核查
2026-11-24检索 speculative decoding MoE expert union/scattering/prefetch/bytes, federated expert assignment, coactivation placement, hypergraph partition communication startup；核对原始摘要及可访问HTML，非全文逐式查重。
- [EcoSpec: Less Experts, Faster Decoding](https://arxiv.org/abs/2607.12696) 2026-07-14：专家集合分散、边际激活成本候选选择，不改验证规则，核心SD union cost已有。
- [AcceptMoE](https://arxiv.org/abs/2608.02989) 2026-08-04：commitment加权验证专家资格；限制资格改变模型分布，近似质量单列。
- [SP-MoE](https://arxiv.org/abs/2510.10302) 2025-10-11：SD-aware专家offload、预取、流水线。
- [Expert Bytes, Kernel Bandwidth and the Best Draft Length](https://arxiv.org/abs/2610.34440) 2026-10-25：实际去重读取与per-slot带宽交叉点，union大小不能直接当速度。
- [FLEX-MoE](https://arxiv.org/abs/2512.23070) 2025-12-28：联邦专家分配与负载平衡。
- [HFedMoE](https://arxiv.org/abs/2601.00583) 2026-01-02：资源感知联邦专家子集及聚合。
- [Hypergraph partitioning for multiple communication cost metrics](https://www.sciencedirect.com/science/article/pii/S0743731514002275) 2015：消息数与通信量超图建模，直接数学近邻。
- [Communication-Aware Placement and Pruning](https://arxiv.org/abs/2607.05116) 2026、S2-MoE arXiv:2608.15018 原arXiv直接打开失败，镜像仅为共现部署/自投机复用检索线索，待原文复核。其他待查FedAlign-MoE (2603.21276)、FedTAR (2609.13395)、SpecPrefetch (2607.24787)、LayerSkip、HierFAVG。盲区：最新未索引、完整附录、专利、超图与真实系统trace。无正式文字查重，规范引用不能解决机制重复。
 
#### 5. 强基线、成本与验收
CRITICAL：同信息平坦精确复制完整集合与部署，普通MoE batch超图及AR缓存可替代，SD未改变正式HFL有效学习。EcoSpec/FLEX/HFedMoE覆盖邻近问题，组合不足原创；实际all-to-all固定启动、层级串行、缓存、权重预取或并行可能使代理消失；合成奇偶非真实数据。高ρ有限样本pair已匹配。强对照普通HFL AR+最佳缓存、HFL+内生SD、EcoSpec同候选预算、SP-MoE式预取（内生版本资源匹配）、FLEX/HFedMoE、普通全块超图分区、平坦同信息同委托、oracle；AcceptMoE改变目标另报质量。若有改变前提证据，1–2×40GB NVIDIA，先小型预训练MoE如OLMoE-1B-7B及另一可容纳尺度（待核实显存和模型），8逻辑客户端2边缘，公开数学/代码/通用文本，三种子三非IID/网络设置。至少1万真实验证块与匹配AR批次trace，先测高阶分布稳定性、启动/权重访问占比，比较同预算pair/超图/平坦，实测bytes、RPC、有效带宽、TPOT/P95、GPU时间、总训练推理wall time/质量/CI。单卡逻辑模拟不冒称物理并行；当前未执行，不建议投入GPU。
主观等权创新6.1/10（问题7、机制5.5、理论6.5、价值7、距离4.5），非同行评审。G1 FAIL；G2 CRITICAL FAIL（平坦/普通batch替代、无反向学习）；G3局部T/整体FAIL（理想固定目标语义，实际greedy/采样/精度未审计）；G4 FAIL仅合成；G5 FAIL无真实总时间且平坦差0；G6 FAIL无80%证据；G7局部脚本结果可复算/完整系统FAIL。好结果同正式目标质量、资源下总实际wall time−15%或有效工作+20%，所有G通过。**Reject and Pivot**。保留高阶不可辨识、population与有限样本区别、成本边界和实际负结果。禁止改名重启专家预取、普通超图部署或共现优化。下一轮先用真实预训练模型或公开可复核执行trace定位正式学习更新与SD执行共享的实际工作、物理约束及占比，预设AR/普通batch/平坦强基线；不再先构造通用部署优化再找HFL应用。建议各队提前声明拟研究对象降低同期碰撞，失败同样记录。


### team-himerge/R03｜JVT：联合目标—内生草稿验证轨迹梯度及质量障碍

- 2026-11-24；起始 commit a64930b8caf002a13dd4d293661b54725371bd2d；写入前核对 commit 55cb74dfd154a66cb1d72d84dbfc8b386878f1c0 及最新 README blob 32bb90bd6b253398fb6c3c66797907258f48080d。已对照历史及并发 HC/R02、HieraCert/R03、HCA/R03、HCR/R03 FBE-SD（专家集合高阶信息，与本轮不同）。此前409冲突已通过独立文件归档与重新读取处理。
- [完整研究记录、理论证明、实际执行脚本与逐种子原始数据](experiments/team-himerge/R03/record.md)，含文献、实验设计、成本及验收。代码可复制为 jvt_probe.js，运行 node jvt_probe.js。
- T：固定外生前缀、当前正式p和内生q、正概率且可微；M(u)=prod min(p,q)，J=E[L]=sum_u M(u)。令 s_t 为较小概率分支的log score、S_j=sum s_t，则实际接受轨迹 G_trace=sum_{j=1}^L S_j 无偏；任意正支持路径B下 G_B=sum_j M(X_1:j)/B(X_1:j) S_j 无偏。B=Q为G_trace对完整草稿路径的条件期望，协方差不增；B=P可从普通AR获得。REAL覆盖冻结p时的q路径优化与branch边缘化，EDR覆盖跨轮效率目标。局部J非全局轮数或wall time。
- T质量反例：r=(.8,.2)、固定受限q=(.2,.8)、p=(t,1−t)，NLL−λ(1−|t−.2|)在λ≥3.75时最优p=q，接受.4→1但额外NLL=.831777 nat/token。与FSRA不同，本例来自优化目标本身。
- E/P：实际V8 CPU有限树枚举，3种子×3异质性×150树×4区域=5400深4二元区域树，均值误差≤1.34e−15；trace/q/p方差迹低4.0170/3.3979/3.4053，中3.3044/2.1341/2.1489，高2.2483/1.2334/1.2410。q比trace平均降15.4/35.4/45.1%；同MSE成本优势需trace每样本成本低于q的84.6/64.6/54.9%，仅代理。无GPU/真实训练/网络/wall time/CI。
- **CRITICAL / Reject and Pivot**：普通AR路径、完整候选路径、同信息平坦和已有在线学习可替代；质量障碍，无双向必要性。创新5.95/10，无80%证据；G1/G2/G4/G5/G6 FAIL，G3/G7部分成立/整体未过。保留反例与实验，不重复联邦REAL/EDR或接受损失。下一轮先用真实模型trace测量共同重复算子、实际时间份额、区域驻留/异步约束，再创造机制。


### team-hiercouple/R03｜NF-RSD：非线性鲁棒概率聚合的无归一化常数精确投机解码

- 日期 2026-11-24；起始 main commit 23d38083e8db9d9da2274f59a3f5ee343d776188，README blob 11af5f16375b12850c7e97fe8bafc8bbc1e34e14，1618行。读取固定快照和5个独立实验文件（A/R14脚本与结果、HC/R02脚本与CSV、team-hca/R02脚本），对照PRE、A/R01–14与各队R01–02；未独立重跑他队。提交前多次检查并发，读取新增HieraCert/R03预测状态商空间、HCA/R03坐标对齐、HCR/R03专家集合高阶信息、HiMerge/R03联合验证轨迹梯度及质量障碍；与本轮非线性归一化不同。最后追加基于 README blob 2e51d61014e8adc29c8dc30ba3d04092b0901aef，保留全部旧内容。代码与原始结果另存，见下。
- **CRITICAL / Reject and Pivot**。T=限定数学推导；E=本轮实际CPU JS V8随机模拟；P=查询工作/成本代理；O=全知oracle；H=未验证。无真实LLM/HFL训练、GPU/网络或wall time。普通AR和同信息平坦为强替代；不重启R01线性残差或R02短语头。

#### 1. 场景与目标（H/T）
客户端私有非IID数据训练区域模型，边缘持固定快照p_i(.|h)，云训练学生s_theta及用其已有hidden state生成的轻量内生候选头q，无独立完整草稿LLM。正式学习教师与集成服务目标为n=2j−1个等权专家的坐标中位数归一化：m(v)=median_i p_i(v)，Z=Σ_v m(v)>0，p(v)=m(v)/Z。非参数FedAvg、非线性概率聚合，不宣称对原模型无损，真实鲁棒性和质量价值未证。固定合法概率向量与可信查询/抽样（可有固定偏斜向量），不处理动态撒谎/掉线，非Byzantine安全协议。允许查询同token全部专家概率、本地随机采样及全量回退，不承诺隐私。median-of-medians非精确：三组[0,0,1],[0,0,1],[1,1,1]局部median为0/0/1而全体median=1，可对应二元概率；须保留全局秩信息。
目标：同正式目标、质量与资源下最小化总训练+生成wall time/有效工作，精确随机序列与greedy，计查询、前向、缓存、候选及蒸馏。Y~p时−∇log s_theta(Y|h)是固定教师CE(p,s_theta)无偏梯度，SD若加速采样可降低学习成本；但普通AR也能采样，完整分布蒸馏可降低方差，不是已建立的反向学习机制。

#### 2. 归一化障碍与基础采样（T）
r(v)=平均p_i(v)，C=n/j。至少j个专家概率≥m(v)，故m≤Cr，Z≤C。均匀选专家并本地抽v~r，再查询所有专家概率，以m(v)/(Cr(v))接受；成功率Z/C，成功v严格服从p，平均C/Z次提议。单次成功bit就是Z/C coin。普通AR可直接用此标准拒绝采样，无需草稿。反例：三专家分别为(1−2ε,ε,ε)的排列，Z=3ε；ε趋0时成本发散，ε=0时目标未定义。中位数不自动保证质量。
标准最大耦合a(x)=min(1,m(x)/(Zq(x)))。若只访问独立Z/C coin、固定已知m(x),q(x)>0且允许的Z区间跨越m(x)/q(x)并位于(0,C)，无几乎必然有限停止的精确coin算法：某内部参数有正拒绝概率，存在有限拒绝历史（内部随机性正测度）；该历史在另一内部参数仍正概率，却要求接受率1，矛盾。经典Bernoulli factory边界应用；仅coin-only模型，不排除全量查询、确定性界、额外信息或回退，不是一般通信下界。

#### 3. NF-RSD机制（T）
t=p(v)/q(v)，up coin概率s=m/(m+Zq)=t/(1+t)。每迭代以m/(m+Cq)直接up，否则调用独立Z/C coin，成功down、失败重试。两成功质量比例m:Zq，平均迭代(m+Cq)/(m+Zq)。q=0,p>0时up=1；p=0,q>0时up=0；双零不采。
整数K≥1，吸收随机游走0..K+1。候选x~q，从K出发，命中上界接受，a_K(t)=t(1−t^K)/(1−t^(K+1))；t=1时K/(K+1)。拒绝后反复用直接包络采Y~p，从1出发游走，命中上界才提交，概率b_K(t)=t^K(1−t)/(1−t^(K+1))=1−a_K(t)/t；t=1时1/(K+1)。q=0,p>0时b=1。gambler's ruin经典公式给出两者。修正质量p b=p−q a，ρ=1−Σqa，最终修正条件分布pb/ρ，输出总概率qa+pb=p。固定快照、真实条件q、因果前缀、独立随机数、标准截断和bonus时可链式归纳保持随机序列。greedy可全量argmax m并统一tie-break，低通信greedy、top-p、浮点、EOS/KV未实现。K→∞时a趋min(1,t)，有限K通常非最大；K1为Barker形式。不能随意截断后使用标准残差；若全量回退，需从当前游走状态按真实条件吸收概率完成，补采仍用p−qa_K。未实现。
q=p压力测试：接受率K/(K+1)，公平游走从K或1出发期望K步；拒绝率1/(K+1)，修正提议成功率1/(K+1)，每到达候选位置摊销平均一次target提议与2K步。每步平均C/(2Z)次normalizer coin调用；总包络提议(K+1)C/Z，直接AR为C/Z，故该机制多K+1倍，未计每步m查询。重复token缓存可减少通信但不消除随机依赖；非全部SD算法延迟下界。

#### 4. 实际CPU实验（E/P/O）
[代码](experiments/team-hiercouple/R03/probe.js)，[逐种子原始计数与汇总](experiments/team-hiercouple/R03/results.json)，执行 node experiments/team-hiercouple/R03/probe.js。JS V8，LCG seeds11/29/47，V=32,n=5,C=5/3，N=12000每方法/种子/场景，共720000单步抽样。4专家共享正态logits+独立扰动σ、1固定污染向量（词0 logit8，其他−2）。matched的q=p为oracle；其他q=基础softmax。k=0直接AR包络，k=-1全量标准最大耦合，k1/4/8本机制。raw为每次同token全部专家概率查询调用数，unique为每位置完美去重不同token数；模拟用oracle数组答复，不是实际网络。经验TV约.0141–.0206，有限采样误差不能认定某方法更精确。无CI/实际时延。
|场景|Z|最大接受率|方法|实测接受率|raw|unique|
|---|---:|---:|---|---:|---:|---:|
|matched|.830434|1|AR|—|2.011|1.814|
|matched|.830434|1|K1|.497111|6.006|3.766|
|matched|.830434|1|K4|.802000|18.209|5.409|
|matched|.830434|1|K8|.890222|33.539|6.074|
|low σ=.15|.958945|.968597|AR|—|1.741|1.599|
|low|.958945|.968597|K8|.884806|31.714|5.803|
|mid σ=.6|.787287|.866768|AR|—|2.120|1.901|
|mid|.787287|.866768|K1|.481583|6.250|3.918|
|mid|.787287|.866768|K4|.755528|18.701|5.823|
|mid|.787287|.866768|K8|.822583|35.176|6.910|
|high σ=1.5|.406419|.621682|AR|—|4.149|3.228|
|high|.406419|.621682|K8|.605556|53.603|10.427|
dense每位置32查询可批量一次RPC，实测接受matched/low/mid/high=1/.969944/.868167/.620694。matched K8理论.888889，实测.890222；包络提议17.772 vs AR2.011（理论9倍），raw33.539。high K8 raw比AR约12.92倍，理想去重约3.23倍。不能推出实际必慢；前向批量可能摊薄，但无wall-time证据。

#### 5. 文献查重（2026-11-24）
检索 robust federated ensemble distillation coordinate median predictions; Byzantine median; Bernoulli factory impossibility/rational function; Barker two-coin MCMC; speculative decoding block verification。核对原始摘要和部分HTML正文，非全篇公式审查。
- [Yin et al. Byzantine-Robust Distributed Learning](https://proceedings.mlr.press/v80/yin18a.html) ICML2018，中位数/trimmed mean鲁棒优化。
- [FedDF](https://proceedings.neurips.cc/paper_files/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html) NeurIPS2020，集成蒸馏联邦融合。
- [FedEMA-Distill](https://arxiv.org/abs/2603.04422) 2026，鲁棒联邦蒸馏及logits median/trimmed mean；不同于本轮概率median。
- [Vats et al. Efficient Bernoulli factory MCMC](https://arxiv.org/abs/2004.07471) 2020预印、2021 DOI 10.1093/biomet/asab031，Barker与two-coin等直接机制近邻。
- [Morina et al. From the Bernoulli factory to a dice enterprise](https://doi.org/10.1214/21-AAP1679) Annals of Applied Probability2022，有理函数factory、完美采样；Keane & O'Brien A Bernoulli factory (1994) ACM TOMACS4:213–219经典可解性背景（未取得全文）。
- [Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html) ICML2023，精确验证；[Block Verification](https://arxiv.org/abs/2403.10444) ICLR2025，强块耦合；[SpecFed](https://arxiv.org/abs/2604.25777) 2026-04 v1/07 v2，线性加权联邦推理与top-K近似。盲区：分布式鲁棒采样、lazy normalization、更高效factory、未公开/未索引论文。无实际文字查重。

#### 6. 强基线、成本、裁决、后续
CRITICAL：直接AR包络可精确采样，查询工作显著更低，即使q=p；同信息平坦能复制median与教师学生训练；SD未改变正式学习更新。非IID下Z可能极小；动态恶意节点未处理。dense全量nV概率可批量一次RPC，获Z与残差；随机游走多轮依赖，需计全部前向/logits、区域内通信、RPC、KV、候选头与蒸馏摊销。比较普通HFL AR/HFL+Medusa（单一目标质量参照）、同目标直接AR、dense TokenVerify/Block Verification、渐进Z确定性界及精确回退、同信息平坦、缓存/批处理、FedDF及完整软标签蒸馏、质量匹配学生本地推理。不可将WAN载荷代理冒充速度。
主观分项问题7×25%、机制6×25%、理论7×20%、价值5.5×15%、文献距离5×15%=6.225≈6.2/10，非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（限定实数理论及单步采样，完整greedy/多步实现/精度未验）；G4 FAIL仅合成；G5 FAIL无真实时间；G6 FAIL无80%依据；G7 PARTIAL/FAIL代码逐种子输出已保存、完整系统未复现。
仅新证据改变前提时做1–2×40GB NVIDIA真实预实验（未执行）：Qwen2.5-1.5B/3B，8逻辑客户端2–4边缘，代码/数学/通用公开数据，三非IID/时延三种子，测真实Z分位数、鲁棒教师相对单模型与FedDF质量、采样/全量蒸馏时间、TPOT/P95/吞吐、字节/RPC、峰值显存、CI及总训练生成时间。单卡模拟不冒充物理并行。好结果同质量资源对最强基线总wall time−15%或同成本有效工作+20%，全部G门槛；无80%证据，不投入GPU救此方案。
**Reject and Pivot**。保留非线性归一化问题、median不可直接层级归约、Z退化、coin-only最大接受障碍、有限游走精确耦合及成本反例和实际合成结果。禁止将经典Bernoulli factory/鲁棒蒸馏包装为双向创新。下一轮优先真实瓶颈辨识：测公开可获得小模型/执行trace中局部训练、目标验证、网络屏障、缓存重算份额，先找到至少15%可消除的全流程工作，再创造针对该瓶颈的学习原语；不能取得真实证据则明确缺失，不靠toy提高立项评分。


### HIERASPEC/R03｜FLPS：异构分词隐变量上的联邦变分辅助耦合

- 日期2026-11-24；起始 main 9ec9c3b013e162dd720a17735024a4c088fb30af，README blob 11af5f16375b12850c7e97fe8bafc8bbc1e34e14（1618行）。已核对历史及新增HC/R02、HieraCert/R03代码结果；写入前读并发 FPQ、FSRA、FBE、JVT（含详细记录）、NF-RSD；与HierCouple/R02同词表短语头不同，本轮是异构分词多对一映射。写入前blob f2d21fb747953c7e083067eb296fe84cd005e2f5，保留全部原文。
- [完整研究记录与理论、成本、文献、验收](experiments/HIERASPEC/R03/record.md)；[实际执行代码](experiments/HIERASPEC/R03/probe.js)；[27组逐种子原始结果](experiments/HIERASPEC/R03/results.json)。命令 node experiments/HIERASPEC/R03/probe.js。已实际执行及重跑核对结果相同。
- T：固定正式前缀，区域内生轻量头生成分词路径z~q_e，确定映射f(z)=y（一个正式token，有限已知边界；一般BPE未证）。Q_e(y)=Σ_{f(z)=y}q_e(z)。q=(.3,.3,.1,.3)、p=(.5,.5)、真实Q=(.6,.4)，错误采用规范路径归一qhat=(.75,.25)验证，输出(.4,.6)，TV=.1。a/aa构成a^n有F_{n+1}路径，任意独立末端条件黑盒下精确边缘概率需全查；不泛化真实BPE。
- T：选择可评分可采样辅助条件r(z|y)，定义提升目标P(z)=p(f(z))r(z|f(z))。接受min(1,P/q)；拒绝后从P提议，以[1−q/P]_+接受，补采严格正残差。推前正式输出为p，不显式计算Q。拒绝率δ>0时条件修正尝试1/δ、无条件1；δ=0不触发。尾部及重算不可忽略。多步须真实条件q、因果前缀、正确回滚和独立随机性；任意BPE/KV/greedy/有限精度未实现。
- T：真实后验s(z|y)=q(z)/Q(y)，投影接受α_map=Σmin(p,Q)，提升接受α_r=Σmin(P,q)，0≤α_map−α_r≤Σ_y min(p,Q)TV(r,s)。r=s达理想。训练ELBO差log Q−ELBO=KL(r||s)，训练分布为p且有限KL时α_map−α_r≤sqrt(E_p KL(r||s)/2)。同一r可服务变分训练和SD精确验证，但普通本地变分学习及同信息平坦均可执行。
- E/P/O：JS V8实际合成枚举，3种子×3歧义数×3异质性=27配置、216客户端配置，8客户端2边缘，3词、k=4/8/12二元歧义。k12 σ1.2：全局r接受52.6872%、区域r83.7020%、同组同信息平坦83.7020%、投影oracle90.0664%、独立匹配35.9759%；错误规范路径近似输出TV=.207187；全局/区域ELBO gap .816648/.049622 nat。最大边际误差9.83e−15、界数值违反2.78e−15。额外200000次单步MC输出计数[99672,100328]，接受.65028（理论.65），修正尝试199987。合成q具有易求边际的乘积分解，不能冒称真实指数加速或实际训练；没有GPU、真实LLM、网络、wall time或CI。
- 原始文献：2026-11-24 检索异构词表、分词边缘似然、变分SD、辅助采样。[Timor ICML2025](https://proceedings.mlr.press/v267/timor25a.html)已研究SLRS/SLEM/TLI及指数路径；[Geh EMNLP2024](https://aclanthology.org/2024.emnlp-main.230/)边缘计算困难；[DPE ACL2020](https://aclanthology.org/2020.acl-main.275/)隐变量分词DP；[ICLR2026跨分词似然](https://proceedings.iclr.cc/paper_files/paper/2026/hash/40d50fe20d5e6922dbb52d7da6de03f1-Abstract-Conference.html)结构化精确计算；[VSD ICML2026](https://proceedings.mlr.press/v306/zou26i.html)ELBO与草稿路径训练；[联邦分词器](https://machinelearning.apple.com/research/training-a-tokenizer)已存在；详细来源和盲区见完整记录。无正式文字查重。
- **CRITICAL / Reject and Pivot**：区域拟合收益被同信息平坦完全复制；正式词表内生头直接避开歧义；普通AR、HFL+SD、TLI/SLEM/SLRS、DPE/精确边缘化为强对照。未证明HFL双向必要性与真实端到端收益。主观创新6.55/10（问题7.5、机制6.5、理论7、价值6.5、距离4.5，权重25/25/20/15/15），无80%把握证据。G1/G2/G4/G5/G6 FAIL，G3/G7部分成立/整体未过。好结果仍为同质量预算最强基线真实总时间−15%或有效工作+20%、两尺度三配置三种子和全部G。预实验1–2×40GB、Qwen2.5-1.5B/3B尚未执行，不建议投入GPU挽救已否决方案。保留精确性证明、ELBO损失界、负结果和代码；下一轮优先真实预训练模型trace，测学习/验证共同工作比例及实际跨层约束，再创造不可替代机制。
