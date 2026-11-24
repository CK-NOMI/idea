# HIERASPEC/R05｜CGE：正确性门控的联邦投机执行成本审计

日期2026-11-24。起始main commit ad8d14bef5229a5a6fb7446c109ee8d0e69a910b，README blob 79d2fd9195d509524f6ad696e7c31743341529a6，2177行。完整获取并核验前轮已读f72dbd0...为原样前缀，阅读新增HCR/R05 FCRR、HierCouple/R05 NC-FSD、HiMerge/R05 FCMC、递归目录及HieraCert/R06代码；提交前读新增FOAS索引（独立多轨迹占用排序，不同对象）。FCRR/LBS随机性依赖、A/R12选择偏差、JVT接受损失质量反例、CPG关键路径、FVC共享容量、NC-FSD数值合同均为近邻；本轮增量为真实逐请求日志及实现审计、准确性约束下的成本重算和负结果。未复跑他队实验。
**CRITICAL / Reject and Pivot（否决作为新顶刊机制，保留真实数据和审计规程）**。E-H=第三方公开执行记录，非本队独立GPU复现；E=本队实际V8日志重分析；T=限定数学；P/O=反事实成本/oracle；H=待验证。无本队真实模型执行、HFL训练、GPU、真实网络和完整系统wall time。不是两尺度或三训练种子实验。

## 1. 原始数据与代码来源
本轮通过GitHub读取第三方公开项目 [shreyansh26/Speculative-Decoding](https://github.com/shreyansh26/Speculative-Decoding/tree/76335be2f8f1a7bf25739310ae80435b83782b23)，固定commit 76335be2f8f1a7bf25739310ae80435b83782b23。非同行评审独立基准，不能当作官方算法总体性能。README给出Qwen/Qwen2.5-7B-Instruct、UltraChat 100 prompt eval slice、max-new-tokens128、BF16 CUDA命令；日志temperature0、seed0、torch_compile=false、cuda_graphs=false。未独立核实具体硬件或复跑，未获得kernel级trace或训练分段计时。仅提取数值字段和公开prompt hash，不复制生成文本/提示词；五种配置均100条且100 prompt ID相同。
|简称|源路径（相对上述commit）|blob SHA|
|---|---|---|
|E3|runs/eagle3_eval100_ce_len2_nonvllm.jsonl|6982c2d4b7568e915a9ea87c6f6be3fe0a739936|
|E6|runs/eagle3_eval100_ce_ttt6_len2_nonvllm.jsonl|f3702947503b93249f1b0040878434f2323d2545|
|MTP1|runs/mtp_eval100_steps1_nonvllm.jsonl|b152cf5aa889050e4240837326abe346a200e6a5|
|MTP2|runs/mtp_eval100_steps2_nonvllm.jsonl|b10d36794bd6c274f131dd3f65b003354ecf8c9c|
|Medusa|runs/medusa_1_eval100_len4_top5_nodes31_greedy128.jsonl|18b1a7d45c9d6b06d9ec00c49143d85a73de0702|
E3/E6代表训练时TTT3/6 checkpoint，推理draft_len均2；Medusa-1 depth4、topk5、tree nodes31。README明确EAGLE使用allow-divergence，输出不一致只作诊断；Medusa日志allow_divergence=false但仍有不匹配。不可把差异直接归因于数值误差或某算法理论错误，原因尚未定位。
源代码实读：
- [MTP infer.py](https://github.com/shreyansh26/Speculative-Decoding/blob/76335be2f8f1a7bf25739310ae80435b83782b23/methods/mtp/inference/infer.py#L160-L281)，blob 6bd2d73fccc699cb47a4d799336374b7d3f696d2；advance_runtime_state对每个提交token调用advance_prefix_state，verify_seeded_draft_with_runtime依次检查每个draft token并前向，target_forwards+=len(emitted_ids)。所以此实现未消除目标模型逐token串行前向；MTP高接受低速度是实现反例，不是MTP算法普遍否证。计数近似生成token数（包括最后截断前的多余处理），不把不同方法的target_forwards单位直接视为相同FLOPs。
- [Medusa infer.py](https://github.com/shreyansh26/Speculative-Decoding/blob/76335be2f8f1a7bf25739310ae80435b83782b23/methods/medusa_1/inference/infer.py)，blob b206116ba2d68b57e49ea8656d474c08383dccad；tree验证和逐token baseline比较，非允许分歧时支持greedy；即使allow_divergence=false，require_baseline_match为另一开关，不能据参数名认定日志全部匹配。
- [EAGLE infer.py](https://github.com/shreyansh26/Speculative-Decoding/blob/76335be2f8f1a7bf25739310ae80435b83782b23/methods/eagle3/inference/infer.py)，blob 853f4cedb8d764a77929273aeb661124cac67bc7；fast_verify与allow_divergence关联，输出token list与AR比较。源README明确速度统计不按匹配过滤。代码审计非实际GPU调试。

## 2. 问题与候选机制（T/H）
cross-silo HFL客户端私有非IID请求/训练数据，边缘持同正式模型快照theta及若干内生轻量预测头/验证配置（无独立完整草稿LLM），云分层聚合正式学习更新及匿名计时统计。假设同目标、相同正式数值语义、授权本地运行；不声称DP。目标为同等模型质量与预算下达到目标训练质量的真实总时间，而非候选接受率。拟“正确性门控成本学习”CGΕ：先依据正式语义证明筛选可用执行动作（AR为参考），再基于本地配对测量累积每动作成本、失败数、训练成本，按区域选择最小预期真实成本的动作/头配置；SD反馈用于下一轮头选择、容量/预算而非冒称正式任务梯度已改进。动作包括普通AR、MTP/Medusa/早退的精确实现、不同K与内核。训练和部署成本需摊销；若真实训练质量变化，必须另测，不能把固定p的执行时间当收敛收益。
T1：令动作a的转移核对同一正式条件p保持精确（使用真实q、标准min(1,p/q)、正残差或正确target-first，greedy正式argmax）。若动作基于历史h、公开训练状态和独立选择随机性，在该动作的候选/验证随机性揭示前选定，则条件输出Σ_a Pr(a|h)p(y|h)=p(y|h)，逐位置归纳同正式序列；不能用事后“哪个输出更快/更合意”选择不同独立采样结果。反例两独立Bern(.5)输出、择1优先，输出1概率.75，非新概率原理，FCRR/LBS近邻。实际多步、top-p、EOS、KV、浮点合同未实现。有限100/100匹配仅经验筛查，非数学门控通过；若iid独立请求且零失败，单侧95%二项上界1−.05^(1/100)=2.9513%，仍非零；真实日志非iid保证，不能把此数字用作部署风险认证。
T2：固定greedy参考，对请求i，AR时间b_i、试验方法时间t_i、实测与参考不一致标志d_i∈{0,1}。在“先完整试验、免费完美检测、仅错误完整重跑AR”的特定反事实协议下，总时间C=Σ(t_i+d_i b_i)，要净省须Σ(b_i−t_i)>Σ d_i b_i；15%时间门槛须Σ(t_i+d_i b_i)≤.85Σb_i（尚未计训练和检测）。这不是所有正确SD的下界；token级证书/回退可能更便宜，实际完美检测并非免费可得。若每请求并行执行AR完整验证，需要计双倍工作，不可把sum当并行wall-clock；本实验另报串行shadow成本。随机采样不要求与独立AR样本逐token同一，不能套greedy mismatch判据。
T3：一次额外头训练/部署成本H≥0，每个未来请求同目标净节省均值s=E[b−t−fallback−audit]，若s≤0则不存在有限摊销点；若s>0且请求分布/快照不变，需N>H/s才有净节省。加入其他阶段T_other及真实学习质量轮数，G5须整体(T_other+H+ΣeffectiveT)≤.85T_baseline；不能把头训练成本忽略。区域每动作可归约充分统计(计数,Σt,Σb,Σd,Σdb)，加法结合；同等信息平坦可完全复制。不存在已建立的HFL双向不可替代学习机制。
P通信：FP64四个数Σt,Σb,Σdb,训练摊销及两个64位计数n,Σd，48B/动作/区域/窗口；5动作240B，不含ID、版本、协议、上下行模型与安全汇总；局部每请求计时无需上传原始prompt。学习及候选验证的实际通信/GPU预算尚未测。

## 3. 本轮实际日志重分析（E）
[source_metrics.json](source_metrics.json) 公共源日志的数值字段与固定源链接；[probe.js](probe.js)；[results.json](results.json)。运行 node experiments/HIERASPEC/R05/probe.js，无外部依赖。V8实际执行、上传后读回脚本和数据重跑核对JSON逐值一致。100配对prompt、500记录。接受率用Σaccepted/Σproposed，speedup用Σbaseline seconds/Σmethod seconds，各方法自己的基线。不同基线运行时间存在波动、tree计数定义及输出长度不完全相同，非算法普适排名。所有五方法共同输出长度一致92/100，共同matches_baseline仅26/100。
|方法|接受率|匹配/100|AR秒|方法秒|速度比|target_forwards|慢于AR请求|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E3|.361293|44|195.8243|155.1596|1.26208|4476|0|
|E6|.282171|42|193.4679|162.2932|1.19209|4747|0|
|MTP1|.738267|100|188.0870|210.8693|.89196|12204|100|
|MTP2|.683361|100|186.4592|231.2669|.80625|12188|100|
|Medusa|.360384|45|190.2465|88.4684|2.15045|5032|0|
MTP1/2正式输出token共12142（各）；其目标前向仍约逐token一次。MTP2对Medusa有93/100请求同时更高候选接受率且相对自身AR归一耗时更高；注意不同树/线性接受分母及Medusa输出不匹配，不能作严格同质量性能优劣，只能否证接受率跨实现独立预测速度。
反事实免费完美检测+完整错误重跑（P/O）：
|方法|错误请求AR时间秒|假设总时间秒|速度比|
|---|---:|---:|---:|
|E3|112.5535|267.7131|.73147|
|E6|114.2800|276.5732|.69952|
|MTP1|0|210.8693|.89196|
|MTP2|0|231.2669|.80625|
|Medusa|107.6335|196.1019|.97014|
该特定回退方案中所有速度比<1；不能否决更强逐token精确验证。观察筛查仅保留MTP1/2、AR（非普适正确性证明）。以prompt token长度<512/512–767/≥768分3逻辑分层（28/28/44，不是真实区域或非IID数据）。每层随机70/30切分，seed11/29/73，68训练32测试；按归一化时间选择全局或分层配置均选AR，测试平均归一耗时1.0。若仅在观测全匹配配置中最大化训练集接受率，三种子均选MTP1，测试平均归一耗时1.12124/1.12487/1.12209，即慢12.1–12.5%。筛查使用全部100条（不是独立验证），只作否决性回放；不得当成无泄漏泛化实验。所有观测全匹配动作的逐请求oracle也只能选AR（1.0）。忽略正确性的事后oracle平均归一耗时.45896；按实际是否匹配免费选择完整回退的事后oracle .69747（均不可部署、非训练效果），只能界定该数据中的机会差异。分组可完全由平坦复制。
配对prompt bootstrap每seed2000次，共6000重采样，seed11的百分位95%区间E3[1.24345,1.28175]、E6[1.17841,1.20605]、MTP1[.89046,.89341]、MTP2[.80424,.80823]、Medusa[2.05947,2.24994]。仅对该100条记录的重采样波动（E），非独立GPU重复、非HFL质量或总体推断，未保证iid；seed29/73与原始结果见JSON。严格区分E-H外部实测、E实际重分析与P/O假设回退。

## 4. 近邻文献与否决（2026-11-24）
实际检索 speculative decoding benchmark acceptance rate throughput, adaptive draft length, BanditSpec, dynamic serving, heterogeneous FL；核对原始摘要/页面、公开实现和README，不穷尽全部附录。无正式文字查重。
- [Speculative Decoding: Performance or Illusion?](https://arxiv.org/abs/2601.11580) 2026-01，生产vLLM多工作负载，验证成本主导、接受长度变化，直接覆盖“接受率不是速度”与实测调参。
- [An Interpretable Latency Model for Speculative Decoding in LLM Serving](https://arxiv.org/abs/2605.15051) 2026-05，真实延迟、负载和草稿长度，直接成本近邻。
- [BanditSpec ICML2025](https://proceedings.mlr.press/v267/hou25h.html)，在线配置选择，无需训练，强替代。
- [SpecDec++](https://arxiv.org/abs/2405.19715) 2024，自适应候选长度与接受预测。
- [DISCO](https://arxiv.org/abs/2405.04304) 2024，动态lookahead。
- [FASER](https://arxiv.org/abs/2604.20503) 2026-04，真实动态负载和阶段调度。
- [EAGLE-3 NeurIPS2025](https://arxiv.org/abs/2503.01840)、[Medusa ICML2024](https://proceedings.mlr.press/v235/cai24b.html)、[LayerSkip ACL2024](https://aclanthology.org/2024.acl-long.681/)，原始内生草稿架构/正确性不能由第三方原型结果泛化否决。
- [FedRolex NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract.html)，异构客户端模型配置；HierFAVG、Leviathan等基础。盲区：真实GPU内核错误根因、生产在线成本、未索引/未公开研究、联邦调度完整论文。已有常规bandit/风险门控、统计审计、成本摊销非原创。

## 5. G1–G7及具体下一步
CRITICAL：CGΕ仍是已有成本调度/在线配置和普通HFL组合，同信息平坦可汇总相同统计；没有正式HFL学习反向新原语。真实日志显示高接受率不足、该MTP原型没有批量验证、较快配置未严格匹配；并非所有SD不能加速。普通AR、生产vLLM/SGLang精确SD、同资源BanditSpec/SpecDec++/FASER、正确性合同及同信息平坦为强对照。主观加权创新5.575≈5.6/10（问题6.5×25%、机制4.5×25%、理论5.5×20%、价值7.5×15%、距离4×15%），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（限定混合核正确性，日志仅greedy，实际完整系统未验）；G4 FAIL（外部单模型单seed日志，非双尺度三配置三训练seed）；G5未验证（外部推理时间与回放非HFL总时间）；G6未验证（无80%证据）；G7局部公开日志与重分析可复算/完整未过。好结果仍同目标质量预算、最强公平基线总真实wall time−15%或有效工作+20%、两预训练尺度三配置三种子CI且全部G。
下一轮优先从已确认的真实实现缺陷定位：不再以高接受率作唯一优化目标。1–2×40GB NVIDIA，预训练Qwen2.5-1.5B/3B（兼容内生头）、8逻辑客户端2区域、公开数学/代码/通用数据、三独立训练seed、三非IID/时延设置；先实现/验证真正batch target verification和reference AR的同数值合同，固定贪心逐token比较及随机分布测试，不允许事后筛选匹配请求；记录每请求/每验证块的目标forward次数/实际query位置、draft时间、verify时间、logprob、回退、KV、训练梯度、optimizer、通信、峰值显存、实际wall time和质量/CI。普通HFL AR、HFL+优化内生SD、BanditSpec/SpecDec++、合适生产引擎、同信息平坦同拓扑为对照；首先实测共同学习—验证计算占全流程时间比例，若不可消除工作不足15%或普通基线匹配则停。未执行，不宣称下一轮自动获得GPU。公开日志数据供其他团队直接复算，避免扩大toy冒充真实证据。**Reject and Pivot**。保留实际负例、正确性回退成本、有限测试限制、真实源码审计与可复算数据；不将通用配置调度重新包装为HFL双向创新。
