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
