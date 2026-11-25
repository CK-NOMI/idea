# team-anchor / R119-P0 — 多层稀疏后缀可行性初筛

**裁决：双门槛均未通过；保留本轮失败结果，停止将当前原型视为可用加速实现。** 本记录只报告实验，不评估创新度或论文前景。

## 可复现身份

- 开始时知识库 commit：`80f56a396b36c8148fc1bff3ae12bca1e87c13f6`（R118）。实验前 `git fetch origin main` 确认当时 `origin/main` 仍为同一 commit。
- 团队与轮次：`team-anchor / R119-P0`。
- 设备：NVIDIA GeForce RTX 3090 24 GB，单卡；运行时 PyTorch 报告 24,124 MiB。环境：Python 3.11、PyTorch 2.5.1+cu124、CUDA 12.4、Transformers 4.48.3。完整 `pip freeze` 在 `results/environment.txt`。
- 模型：R118 已验证的 LayerSkip 1B 本地快照；权重 SHA-256 `ba9a27bc1fadc74d7ca576aa618e464aa006b1c56b4a940c668c980a302708ee`。仓库不重复存放模型权重。
- 数据：固定 AG News CSV 版本 `555590db4219b1243abb1918effd6a7425a2d75f`。本轮用 8 篇来自训练源的独立校准文档和 8 篇测试文档，原始文本及划分在 `results/prompts.json`，来源元数据在 `results/prepared.json`。
- 参数预算：冻结主干，0 个可训练参数、0 步、0 秒训练；不执行 HFL。没有 adapter，因此各稀疏对照适配预算均为零。
- 正式复跑：插桩预热后重复 6 次；剖析覆盖 64/256/512 输入，AR 与 Transformers 集成 LayerSkip（exit 4、草稿长度 2）。稀疏质量与计时使用 64-token 前缀、16 篇文档；每种后缀 2/4/8 层、间隔 B=2/4，共 18 个稀疏配置。Bootstrap 以 8 篇测试文档为配对单位、10,000 次重采样，随机种子 119。

## 计算路径与测量方法

稠密对照运行全部 16 层。周期深算只在锚点执行深层查询和 MLP，非锚点保持浅层输出；锚点历史对照只让注意力读取因果可见的锚点状态；共享记忆对照读取冻结的全 token 浅层记忆并拼接当前层输入处已提交的锚点深状态。每个锚点只能读取当前位置及过去位置；测试中没有把后续 token 或未提交的未来深层状态加入当前查询。

自定义稀疏路径确实跳过了非锚点的深层 Q、注意力输出和 MLP，但仍需要建立相应 K/V 记忆，且当前实现通过 Python 按锚点循环。记录的是同步后的完整稀疏前向墙钟时间，与同一前缀的完整稠密前向墙钟时间配对。该实现不是优化过的生产内核。

逐层、Attention、MLP、Q/K/V、O 投影和 LM Head 由模块 hook 与 CUDA event 插桩。插桩测得的层/子模块事件会受 Python 启动间隔和 hook 开销影响，模块区间相互嵌套且不可相加。插桩墙钟与独立不插桩墙钟同时保存。另运行 `stage_profile.py`，对每个输入长度的一个固定测试请求重复 5 次，分别记录 AR/LayerSkip 顶层 forward、Cache.update、greedy argmax、候选生成及拒绝后 cache crop。**不能把 event 时间当作可消除墙钟时间，也不能将重叠的阶段值相加成总成本。**

## 实际结果

### 1. AR 与 LayerSkip 剖析

下表为不插桩的完整 generate 墙钟均值和 8-token 输出吞吐；每个输入长度与解码方式均有 6 次记录。吞吐仅适用于这次固定 prompt / 8-token 短生成，不代表长生成或请求分布下的服务能力。

| 输入长度 | AR 墙钟 | AR tok/s | LayerSkip 墙钟 | LayerSkip tok/s |
|---:|---:|---:|---:|---:|
| 64 | 0.214 s | 37.47 | 0.170 s | 47.13 |
| 256 | 0.220 s | 36.36 | 0.181 s | 44.22 |
| 512 | 0.195 s | 35.86 | 0.252 s | 27.81 |

本轮条件下 LayerSkip 在 64/256 较快、512 较慢；这组短请求不足以总结通用拐点。插桩后的总墙钟相对不插桩平均高约 1.29–1.42 倍，说明 hook 剖析本身并不轻。`results/profile.json` 保留 36 条逐请求插桩记录及每次模块 CUDA-event 值；`results/summary.json` 保留 36 条配对不插桩记录。

CUDA-event 的层/模块均值例（毫秒；覆盖该次 generate 中各次模块调用，层总时长和其子模块不应相加）：

| 输入 | 路径 | 16 层区间合计 | Attention 区间合计 | MLP 区间合计 | Q/K/V 区间合计 | LM Head |
|---:|---|---:|---:|---:|---:|---:|
| 64 | AR | 274.17 | 154.89 | 38.55 | 35.95 | 5.22 |
| 64 | LayerSkip | 184.81 | 106.05 | 25.64 | 24.65 | 5.99 |
| 256 | AR | 276.13 | 156.29 | 42.84 | 35.95 | 5.23 |
| 256 | LayerSkip | 196.76 | 112.38 | 33.41 | 26.59 | 6.62 |
| 512 | AR | 244.56 | 132.31 | 45.83 | 30.22 | 4.51 |
| 512 | LayerSkip | 262.00 | 142.45 | 50.74 | 32.49 | 9.14 |

逐个 16 层的原始值在 JSON 中。LM Head 的 CUDA-event 区间包含在上面 `profile.json`。阶段性补测结果见 `results/stage_profile_final.jsonl`：下表为请求均值；Stage CUDA-event 包含流上等待/排队区间，host 列是 Python 调用区间，二者不是可相加的独立墙钟分量。

| 输入 | 解码 | 总墙钟 | 草稿候选函数 host | 草稿模型 forward event | 正式模型 forward event | Cache.update event | greedy argmax event | KV crop host |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | AR | 0.262 s | — | — | 0.252 s | 13.1 ms | 0.282 ms | — |
| 64 | LayerSkip | 0.191 s | 100.8 ms | 45.8 ms | 121.6 ms | 7.6 ms | 0.354 ms | 0.944 ms |
| 256 | AR | 0.236 s | — | — | 0.229 s | 12.2 ms | 0.282 ms | — |
| 256 | LayerSkip | 0.192 s | 111.6 ms | 51.5 ms | 119.0 ms | 7.9 ms | 0.351 ms | 0.934 ms |
| 512 | AR | 0.211 s | — | — | 0.203 s | 10.2 ms | 0.249 ms | — |
| 512 | LayerSkip | 0.266 s | 172.5 ms | 79.7 ms | 153.2 ms | 10.7 ms | 0.506 ms | 3.472 ms |

“草稿候选函数 host”包含候选生成期间的模型调用，和草稿 forward event 重叠；“正式模型 forward event”也不等于端到端可删耗时。greedy argmax 已单独记时，但没有 multinomial sampling，因为本轮固定 greedy。拒绝的草稿 KV 通过 Cache crop 作废，作废计算并无单独 AR 重算阶段；crop host 是回退整理耗时，不是拒绝后的完整增量生成时间。每个长度仅一个文档、5 次请求，接受/拒绝计数只作轨迹核验：64/256/512 的提议 token 数为 25/30/45，已接受 20/20/10、拒绝 5/10/35，不能当总体接受率结论。

### 2. 多层稀疏后缀质量

最佳质量配置是 2 层后缀、B=2、周期深算：测试文档宏平均 NLL 相对稠密目标增加 **1.389%**，文档配对 bootstrap 单侧 95% 上界 **2.089%**。均值与上界都超过 1%，质量门槛失败。18 个稀疏设置全部未通过双项质量门槛。

共享记忆+锚点历史在 2 层/B=4 时平均 NLL 增幅 2.40%、上界 3.34%；4 层/B=4 为 7.60%、9.02%；8 层/B=4 为 18.00%、21.39%。`results/per_document.jsonl` 保存每篇文档的稠密/稀疏 NLL、相对变化、KL、概率重叠率及分割；没有筛掉测试文档。

### 3. 实际稀疏前向性能

最佳实测稀疏墙钟比为 0.416（2 层、B=4、周期深算）：稠密前向均值约 27 ms，稀疏前向约 65 ms，净关键路径变化约 **−140%**（即该 Python 原型约慢 2.4 倍）。性能初筛要求至少 +11.11%，故失败。深层稀疏查询虽实际执行，但逐锚点循环、额外 K/V 构建和未融合算子开销超过省下的计算。所有配置的稀疏前向均慢于稠密参照。

这只是该原型实现的实测结论，不是对融合 CUDA 内核或任意稀疏后缀的理论上限。当前没有为新稀疏正式目标实现自回归 KV Cache 解码，也没有把 LayerSkip SD 运行在该新目标上，故新目标 AR 与适用 SD 的端到端吞吐尚未测得。

### 4. 显存与未完成成本项

GPU 运行时总显存 24,124 MiB；`sparse_timing.jsonl` 中记录了 PyTorch allocator 高水位，但现版本在每个设置前没有重置 high-water mark，因此该字段不是独立配置的峰值，不能用于配置间比较。`gpu_snapshot_postrun.csv` 是任务结束后的空闲快照（1 MiB、0%），不是任务峰值。逐请求峰值显存仍未验证。greedy LayerSkip 的 draft、target verification、argmax、KV update 和 KV crop 都有实际阶段记录，但拒绝后的独立重算耗时不适用/未单独形成阶段；新稀疏目标 AR/SD 吞吐仍未测。

## 失败迭代与数据完整性

所有失败均保留在 `results/attempt-*`：

1. `attempt-1-rope-failure`：稀疏前向未传 `position_embeddings`，16 条评测记录失败，已修正 Transformers 4.48 接口。
2. `attempt-2-aliased-memory-control`：共享记忆张量与锚点状态别名，两个对照实际重合；标记为无效对照，不作为结果。
3. `attempt-3-prewarmup-event-contamination`：模块 hook 预热事件进入首条插桩统计；已在预热后清空事件并以 6 次正式重复复跑。该次数据只保留审计，不用于上表。
4. `attempt-stage-profile-buffer-alias`：阶段插桩把可变列表引用直接存进逐请求记录，后续重复覆盖了前序接受/候选计数；该文件不用于汇总。
5. `attempt-stage-profile-argmax-missed`：修正列表快照后发现 AR 使用 `torch.argmax` 函数入口而非 Tensor 方法，原插桩漏计 AR greedy selection。最终版同时插桩两个入口，采用 `stage_profile_final.jsonl`；共 30 条记录（3 长度 × 2 路径 × 5 重复），失败数 0。

最终复跑 `failures.json` 为空。没有 GPU 训练；`results/run.log` 与 `results/stage_profile_final.log` 是实际运行日志，训练日志记录在 `results/training.log`。固定输入、模型标识和 hash、依赖、逐请求输出、代码及 SHA-256 清单一起归档。

## 结论与下一步实验边界

当前冻结 1B 检查点上，未经适配的多层稀疏后缀在质量和原型速度两侧都失败；浅层记忆+锚点历史没有改善该结论。下一轮若继续执行，按负责人指示应先补齐本轮未测的成本拆分和稀疏目标真实解码，并先修复计时/显存归因；在质量门槛没有可行候选前，不扩大 HFL 训练或超参搜索。本记录不替负责人更换方向。
