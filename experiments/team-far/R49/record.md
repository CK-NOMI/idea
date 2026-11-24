# team-far/R49｜HVC-SD：分层正式验证可补偿子空间与生存目标更新
日期：2026-11-24

## 0. 起始快照、读取范围与证据标记
- 研究开始时 main commit：d505fc2920c10622e1928520851d6c98957cc5cb。
- README blob：7026195e84c98d7ca4e70f8e48c73ed163f1fa48，3621 行、510468 字符；已完整取得当前 README，并筛查 G1–G7 与否决库。
- 重点读取 R43–R48（含 R48 integrated.js/results.json）、team-himerge/R08 FDC 完整报告，以及 README 中 R06 FCF、R40 SPAV、其他团队近期记录。不是对所有团队每个原始代码进行独立复跑；不声称全库逐项复现。
- 关键既有约束：R08 FDC 已构造“接受率相同、深层 Jacobian/正式梯度相反”的两世界反例；因此不能把接受率当作深层补偿信息。本轮明确使用实际 Jacobian/JVP/VJP 对象，而不以接受率代理深层梯度。
- 证据等级：T=限定条件下推导；E-SYN=本轮实际 CPU 运行的合成线性模型；P=随机矩阵秩压力测试；H=真实 Transformer/HFL/生产 SD 待验证。未运行 GPU、未训练预训练 Transformer、未执行物理 HFL 或端到端推理。

## 1. R49 研究问题与否决的朴素版本
朴素方向“按接受率或层级敏感度动态分配 Transformer 各层的全局/区域/客户端参数”不够新：
- MT-SFL（WCNC 2026）已将 multi-tier split FL 与 multi-exit model architecture 结合；
- Adaptive layerwise personalized FL（Neurocomputing 2026）已做共享前层/本地保留后层与超网络层重要性选择；
- LayerSkip 已有早退自投机与共享 Transformer 训练；
- AlphaEdit 已用零空间投影减少参数更新对受保护输出的干扰。
因此 R49 不把层划分、layer-wise aggregation 或普通 null-space projection 单独宣称为原创。

**新问题：** 对每个客户端 c，能否共享/更新 Transformer 浅层参数以改善内生 draft q_c，同时由该客户端的深层正式路径做最小补偿，使其原正式分布 p_c 在校准前缀集合上保持不变或有可控漂移？再把“对哪些客户端可补偿”形成 Cloud/Edge/Client 的嵌套可行子空间，并以多 Token EAL 梯度选择更新方向。

## 2. 系统对象与信息边界
- 单个 decoder-only Transformer；draft 使用前 d 层 early exit 和同一 LM unembedding；verifier 使用完整 L 层。无独立 draft LM。
- 对客户端 c，浅层参数记 θ_s，深层补偿参数记 φ_c。可采用现有深层权重的低秩可训练子空间/adapter；秩 r_d 是显式预算，不把任意全量深层更新视为免费。
- 仅对授权校准上下文/合法前缀集合 H_c 计算正式输出的 Jacobian。客户端原始训练数据不离开本地；Edge/Cloud 接收经协议允许的投影梯度、低秩统计或参数更新。真实隐私泄露/安全聚合未实现。
- q_c 由浅层 θ_s 产生，p_c 由 θ_s 与深层 φ_c 共同产生。参数更新后标准 SD 必须使用更新后的真实 p_c、q_c 做验证。

## 3. 核心数学对象：深层补偿核
将每个校准上下文的正式 logits 投影到去除 softmax 常数平移的中心子空间，堆叠得到 z_c(θ_s,φ_c)。局部线性化：
\[
\Delta z_c=J_{s,c}v+J_{d,c}u_c+O(\|(v,u_c)\|^2),
\]
其中 J_s 对浅层参数求导，J_d 对深层补偿参数求导。

定义
\[
B_c=(I-J_{d,c}J_{d,c}^{\dagger})J_{s,c},\qquad
\mathcal S_c=\ker B_c.
\]

**定理 1（最小残差与精确一阶可补偿性，T）。**
\[
\min_{u_c}\|J_{s,c}v+J_{d,c}u_c\|_2
=\|(I-J_{d,c}J_{d,c}^{\dagger})J_{s,c}v\|_2=\|B_cv\|_2.
\]
最小范数补偿器为
\[
u_c^*(v)=-J_{d,c}^{\dagger}J_{s,c}v.
\]
故对该校准集合，浅层更新 v 能被深层参数完全一阶抵消，当且仅当 v∈S_c。注意这是局部一阶命题；不是对所有未来前缀、有限步更新或模型能力的全局不变性证明。

若使用 Fisher 度量 F_c 衡量正式分布漂移，令 A_c=F_c^{1/2}J_{d,c}，则加权最小残差由
\[
B_c^F=(I-A_cA_c^{\dagger})F_c^{1/2}J_{s,c}
\]
给出。若 v 取 B_c^F 的右奇异向量中 σ≤τ 的子空间，则一阶 Fisher 漂移不超过 τ\|v\|；局部
\[
D_{KL}(p_c\|p_c')\approx\tfrac12\|B_c^Fv\|^2
\le\tfrac12\tau^2\|v\|^2.
\]
这是近似谱过滤的局部界，需参数漂移后重估 Jacobian；未在本轮真实网络验证。

## 4. HFL 嵌套可补偿空间与更新算子
定义
\[
\mathcal S_0=\bigcap_c\mathcal S_c,\quad
\mathcal S_e=\bigcap_{c\in e}\mathcal S_c,\quad
\mathcal S_c=\ker B_c.
\]
必有 \(\mathcal S_0\subseteq\mathcal S_e\subseteq\mathcal S_c\)。设正交投影为 P_0、P_e、P_c，则 P_0P_e=P_0，P_eP_c=P_e；P_e−P_0 与 P_c−P_e 分别投影到区域新增和客户端新增的可补偿方向。

令 g_c 为“沿最小范数深层补偿路径”的 EAL 梯度：
\[
g_c=\nabla_{\theta_s}A_c-
J_{s,c}^{T}(J_{d,c}^{\dagger})^{T}\nabla_{\phi_c}A_c.
\]
对 K-token sampling EAL 可用精确枚举（小词表）、R46 生存下界或经验证的采样估计计算。记 \bar g 为全局平均，\bar g_e 为区域平均。局部二次代理
\[
\max_{v_0,v_e,v_c}\sum_c
\left[g_c^T(v_0+v_e+v_c)-\frac{\lambda}{2}\|v_0+v_e+v_c\|^2\right]
\]
约束 v_0∈S_0、v_e∈S_e∩S_0^⊥、v_c∈S_c∩S_e^⊥。其解为
\[
v_0=\frac{P_0\bar g}{\lambda},\quad
v_e=\frac{(P_e-P_0)\bar g_e}{\lambda},\quad
v_c=\frac{(P_c-P_e)g_c}{\lambda}.
\]
客户端浅层总更新为 v_0+v_e+v_c，深层更新为 u_c^*(v_0+v_e+v_c)。

**定理 2（嵌套空间局部最优性，T）。** 上述解是所给局部二次代理和嵌套约束下的最优解；增加 Edge 空间相对 Cloud-only 的代理目标增益为
\[
\sum_e\frac{n_e}{2\lambda}\|(P_e-P_0)\bar g_e\|^2\ge0,
\]
增加 Client 空间的额外增益为
\[
\sum_c\frac{1}{2\lambda}\|(P_c-P_e)g_c\|^2\ge0.
\]
若相应投影梯度非零则严格为正。这不意味着真实非凸 Transformer 的全局最优，也不意味着优于不施加正式质量约束的个性化基线。

## 5. SD 正确性边界
本机制不改变标准 SD 的验证核。更新后的模型定义新的正式分布 p_c' 和草稿 q_c'；若使用真实条件 q_c'、接受率 min(1,p_c'/q_c')、拒绝正残差、首拒绝截断、合法 bonus/EOS 与正确随机数，则标准拒绝采样在理想算术下仍精确采样 p_c'；greedy 路径用正式模型 argmax 验证。**这只保证输出匹配更新后的正式模型，不保证 p_c' 等于训练前 p_c。** 生产 top-p、FP16/BF16 边界、KV/cache 与数值路径均未验证。

## 6. 实际合成实验 E-SYN
脚本 [toy.py](toy.py)，结果 [results.json](results.json)。Python 3.13.5 / NumPy 2.3.5。100 个独立随机种子；4 个客户端、2 个区域、二元词表、2-token EAL 精确枚举、浅层参数维度 12、深层补偿秩 2。所有方法匹配四个客户端副本上的浅层总更新范数 0.1。EAL 梯度用中心有限差分计算；对每个客户端执行最小范数深层补偿。

| 方法 | 平均 ΔEAL | 正式分布平均 KL | 100 seeds 中最大线性残差 |
|---|---:|---:|---:|
| Global unconstrained | 0.013122 | 4.76e−5 | 0.04528 |
| Edge unconstrained | 0.018610 | 7.32e−5 | 0.05539 |
| Client unconstrained | 0.025734 | 1.21e−4 | 0.07426 |
| Global safe | 0.009523 | ≈0 | 8.95e−17 |
| Edge safe | 0.011494 | ≈0 | 5.86e−17 |
| HVC hierarchy safe | **0.013531** | **≈0** | **3.53e−16** |

HVC hierarchy safe 的 ΔEAL 比 Global safe 高 42.1%，比 Edge safe 高 17.7%，同时线性模型中的正式分布 KL 约为数值零。可是 unconstrained client update 的 EAL 增益仍高约 90.2%，代价是正式 KL 漂移。因此结果只支持“有一个可计算的 EAL—正式分布保持折衷”，不支持整体优越性。线性 logits 模型没有真实 Transformer 的层间非线性、注意力、训练动态或通信开销。

## 7. 关键反例/风险：可补偿空间会秩坍塌（P）
另做随机 Jacobian 秩压力测试：早层维度 24、每上下文正式 logits 维度 6、4 客户端/2 区域、每设置 20 seeds。表中为 client/edge/cloud 平均零空间维度：

| 深层补偿秩 | 1 context | 2 contexts | 3 contexts | 4 contexts |
|---:|---:|---:|---:|---:|
| 3 | 21 / 18 / 12 | 15 / 6 / 0 | 9 / 0 / 0 | 3 / 0 / 0 |
| 6 | 24 / 24 / 24 | 18 / 12 / 0 | 12 / 0 / 0 | 6 / 0 / 0 |
| 12 | 24 / 24 / 24 | 24 / 24 / 24 | 18 / 12 / 0 | 12 / 0 / 0 |

解释：上下文增多时，要求多个客户端同时保持正式输出会让共同 Cloud 子空间迅速消失；提高 suffix rank 能恢复可补偿空间，但会增加参数、计算和可能的通信。过低秩→共享空间为空；过高秩→几乎所有方向都可补偿，层级区分又失去意义。真实网络的 Jacobian 谱可能完全不同。**这是当前最大致命风险，不应隐藏。**

## 8. 相关工作与差异边界
- [MT-SFL, WCNC 2026](https://doi.org/10.1109/WCNC65185.2026.11555087)：多层级 split FL + multi-exit personalization；直接威胁普通 HFL+多出口组合。本轮仅核对 Princeton/IEEE 摘要页，未取得完整公式，不能宣称已排除撞题。
- [Adaptive layerwise personalized FL, Neurocomputing 2026](https://doi.org/10.1016/j.neucom.2026.134290)：共享前层、本地后层、超网络层重要性；本轮核对摘要与可见引言，未阅读全文。
- [LayerSkip, ACL 2024](https://aclanthology.org/2024.acl-long.681/)：共享主干早退自投机、layer dropout、early-exit loss。
- [AlphaEdit, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html)：零空间保护输出；零空间本身不是本轮原创。
- [FLAYER / adaptive layer-wise pFL, IJCAI 2025](https://ijcai.org/proceedings/2025/541)：层级参数选择与个性化聚合。
- [R08 FDC 内部记录](https://github.com/CK-NOMI/idea/blob/main/experiments/team-himerge/R08/record.md)：接受率不能识别隐藏深层 Jacobian 的两世界反例。本轮需要实际 JVP/VJP，不能把接受率当作补偿信息。
- 关键差异候选是“以早层 draft—深层 verifier 的可补偿 Jacobian 构造跨客户端嵌套空间，再以 EAL 梯度优化”；但它与 AlphaEdit 零空间、MT-SFL 多层级出口及普通 pFL 的组合边界尚未通过全文级审查。

## 9. 端到端成本与可行性
若显式形成 J_s/J_d，LLM 规模不可接受。候选实现必须限制在早层 LoRA/adapter 和低秩深层补偿器上，使用 matrix-free JVP/VJP、CG/LSQR 与小维 sketch；不能构造完整参数 Jacobian。需计入每轮 Jacobian 估计、CG 迭代、深层补偿反向、额外 LoRA 参数、Edge/Cloud 通信、anchor refresh 和 guard 回退。若这些开销吃掉生成阶段收益，机制应终止。当前没有任何真实时间/通信数据。

## 10. 主观评分与 G1–G7
候选内部评分：问题 8.0、机制 7.7、数学 8.4、文献差异 6.8、HFL×SD 融合 8.0；权重 25/25/20/15/15，加权 **7.825/10**。非同行评审，不代表已达到 G1。历史 R44 8.14、R30 窄数学 8.35 保留，不能把 R49 分数包装成全项目新纪录。

- G1 原创性：**未通过**。机制候选较具体，但 nullspace、层级多出口和 layer-wise pFL 已有近邻；需全文公式比对与新原语审查。
- G2 双向必要性：**未通过**。HVC 把 SD 生存目标写入 HFL 更新，但同信息平坦系统可以计算相同嵌套投影；尚无物理拓扑/通信证据说明 HFL 不可替代。
- G3 数学正确性：**部分通过**。局部补偿定理与二次代理最优解在限定假设下成立；非线性有限步仅局部，SD 精确性是标准核的条件性继承，不是新定理。
- G4 实验：**未通过**。只有线性 toy 与随机矩阵压力测试；无预训练 Transformer、真实任务、三种非IID/延迟和独立训练种子。
- G5 端到端：**未验证**。无 GPU、通信或 wall time，不能声称 15% 加速/20% 工作增益。
- G6 好结果概率 ≥80%：**无证据，不给数值概率**。
- G7 复现：**部分通过**。脚本与聚合结果已附；无真实模型配置、训练/推理 trace、GPU 环境及全量实验。

## 11. 本轮裁决与下一轮任务
**裁决：保留为理论候选，不独立立项；优先验证可补偿子空间是否在真实 LayerSkip Transformer 上非空且有足够谱间隙。** 若在合理低秩 suffix budget 下 Cloud/Edge 可用空间在 1000 个真实前缀上迅速坍塌，或 matrix-free Jacobian/补偿开销无法达到 G5，立即否决，不继续堆模块。

下一轮最具体任务：
1. 选 1B 与 3B 两个支持 early-exit 的预训练模型；1–2×40GB GPU，8 客户端/2 Edge，至少 3 个非IID与延迟划分、3 独立种子。
2. 只训练早层 LoRA 与深层低秩补偿器；采样真实前缀，估计 Fisher 加权 B_c^F 的谱、client/edge/cloud 近似零空间维数及稳定性。
3. 比较 LayerSkip、普通 HFL+LayerSkip、MT-SFL 可复现版本（若取得代码）、adaptive layerwise pFL、AlphaEdit-style null projection、HVC-SD、无补偿与 full-personalized。
4. 同时测正式 NLL/任务质量、EAL、精确采样一致性、训练成本、补偿成本、通信、推理端到端 wall time 和 CI；必须同质量同预算。
5. 若无法在不超过质量容差的条件下实现端到端 wall time −15% 或同成本有效工作 +20%，Reject。

## 12. 文件
- [本轮报告](record.md)
- [可复现实验脚本](toy.py)
- [聚合原始结果/秩压力测试](results.json)
