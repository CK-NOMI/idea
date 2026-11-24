# team-hca/R07｜LEA-SD：区域自主授权、执行权限分离与真实PipeSD控制面成本审计

日期2026-11-24；起始 main 30247fc69d0bf0e61c3544aeaa658a3a7e5b2352，README blob 4c8bf0d22364a2b9bbe7b918a3d5715178c1c46c，2384行。获取完整README并核验R06提交a95186cd8994226a1daad2a688a4251a9d26aa58原样前缀；读新增FEC，与CFA、CPG、R12选择偏差等区分。本轮重点为执行权限与每次云授权的串行开销，不是传输熵、梯度或关键路径超加性。写前读取并发FNSC（top-p证书）、HMC-SD（多级耦合）、FCIP（晚绑定查询）。FCIP独立同期提出相同权限/消息图下逐事件模拟和中间反馈限制；本轮不宣称该等价性首创，增量是权限约束的两世界反例、PipeSD官方源码及实际逐请求日志、MsgPack实测和额外控制门槛。最新读 main 044b4ad221e6d3f9b13462b7af8d1def14564550，blob 201e58e52e3f1490e36a5637bf70ed34cc276c43。未独立复跑他队。

**CRITICAL / Reject and Pivot**。T=限定理论；E=实际源码审计、V8公开日志重分析、CPU序列化实测；E-H=作者公开模型推理记录，未独立硬件复现；P=反事实成本代理；H=假说。实查Python3.13.5 torch2.10.0+cpu、CUDA false/0设备、transformers未安装、HF缓存和/opt/models不存在，huggingface.co、raw.githubusercontent.com、github.com DNS Errno -3；GitHub连接器可读公开源码。无本队预训练模型、GPU、HFL训练、实际网络、端到端学习时间或CI。

## 1. 计算问题、信息边界、候选机制 H/T
客户端有私有训练数据，边缘有授权的区域状态s_e（模型、优化器、内生早退/MTP草稿状态、队列、版本、局部随机数），cloud有正式全局模型快照与聚合状态。区分物理设备图G和逻辑协调层级。授权函数A(e,action,state)决定可在何处执行模型计算、访问数据、选择调度、提交本地更新；通信允许集、链路、算力、内存、时钟、RNG均需固定。真正的HFL可能存在多区域局部学习反馈；不能直接把普通平坦单次FedAvg等同于所有可编程委托协议。

LEA-SD假设云可在轮开始为边缘预授权版本绑定的局部策略（允许动作/预算/版本/失效条件），边缘执行内生候选、合法精确验证或请求远端验证、在当前正式输出轨迹上按预定规则计算本地训练更新，最后上报聚合统计。不部署独立完整草稿模型；真实授权、签名、原子事务、故障恢复未实现，不声称安全协议/DP。正式p对一个生成请求固定版本，q可由允许的局部状态产生但验证必须使用真实因果条件提议q；对X~q按min(1,p_X/q_X)接受、拒绝从正残差采样，首拒绝截断错误后缀；每步正确条件核和独立随机数、bonus/EOS/KV、温度、top-p及有限精度条件才可保证完整序列，greedy正式argmax统一tie-break。本轮未实现完整SD。正式训练使用全部合法正式输出（含修正和bonus）及其预定权重，不能用接受样本选择改变训练语义；同快照一次加权更新与多步区域反馈须分别指定，不把冻结p的草稿适配当正式HFL学习机制。
设严格集中授权模式C：每个依赖最新私有事件的局部操作必须在云收到事件并返回许可后才能执行；本地自治模式L：允许事先部署相同策略在边缘执行。目的是减少真正暴露的控制延迟，而非篡改p/q或学习目标。若原系统无每次额外授权或可用普通预授权，候选增量为0。若监管真正禁止本地自治，候选本身不可部署，不能只给HFL放宽权限。真实此类制度约束未取得证据。

## 2. 理论 T
(1) **权限受限的两世界反例。** 边缘观察私有事件b∈{0,1}，云在此前公共历史下不能区分两个世界，唯一合法下一动作a=b（可解释为本地二元安全约束或应按奖励/损失选择的两个不同训练操作；不是所有LLM操作均具有此结构）。若云须逐次决定a且没有获得b的信息，其决策分布相同；均匀b下平均成功率≤1/2，无法对两世界零错。故需要包含区分信息的边到云通信及许可回传（或合法预授权计算）。若每个新b必须在上一步许可执行后才产生、云独占决定且串行链无并行/重叠，N次决策需要N次往返的暴露控制延迟；这是特定依赖/权限模型的下界，不是一般网络/SD下界。若各b独立预先可得，则可合并消息；若云许可可预计算成策略，直接委托即可绕开。真实带宽、延迟、事件信息量与授权开销需测。
(2) **同权限逐事件模拟（FCIP独立同期相同结论，非首次）。** 形式化协议为物理节点状态s_v、事件e=(接收消息/局部数据/时钟/随机流/完成通知)、转移F_v(s_v,e)→(s'_v,动作,输出消息)。假设对照类允许完全相同的各节点状态、授权、资源与执行/消息图、条件转移、调度与随机流；初始化一致，将层级协议的状态机及通信端点原样映射到一个逻辑协调器和授权边缘委托者。按相同物理事件顺序归纳：初始状态同；若前t事件状态/消息/RNG一致，则第t+1事件输入同，转移、输出、时刻及下一个允许事件相同。故相同执行轨迹、正式token、训练更新、字节、计算、关键路径、故障/恢复（前提故障模型相同）。随机化通过共同随机流耦合，联合分布相同。无需聚合线性/交换律，也不限同步。结论针对可以执行完全相同程序的能力类，不代表普通FedAvg、没有区域状态或单次上传终点的具体算法可复制；物理上复制树消息后它本来就是层级式执行，标签差异不能当科研优势。若目标要超越普通平坦算法，G2应明确不包含候选新原语但允许同等资源/授权的合理强基线，不要求打败能逐行复制候选的全算法集合。仅改变权限不能作为公平算法收益。该形式化是对库内1.3、CFA和FCIP限制的操作性解释，非全新分布式计算定理。
(3) **控制延迟门槛。** 给定同一实际算法与工作负载的记录总时间T（此处5个串行样本时间和），人为增加N个串行且完全暴露的授权往返R，C=T+NR；若预授权成本H，则理想节约比例(NR−H)/(T+NR)。若要求≥15%下降且H=0，需R≥.15 T/(.85N)。若对照已有同一预授权，额外授权数为0，收益0；若授权复用同一验证RPC、异步重叠/合批，则不能用该式断言真实成本。对同工作量的假设反事实而非实测优势。正式HFL更新规则未因该机制本质改变，G2仍失败。

## 3. 原始源码审计 E/E-H
- [MTGC](https://github.com/wenzhifang/MTGC/tree/a48a00c61476ba6343d9ec475202272e4df59216)，MTGC_Implementation/utils_methods.py blob 7bae83f721d9c50e889038da05f33180e2c635cf L75–151：云轮循环、区域循环、区域内E次同步、每客户端训练、区域均值、Z客户端修正、Y区域修正、云平均。实际公开实现为单进程数组和模型调用的实验代码；本次仅源码读取，不运行、不否定算法的理论贡献、不把其当物理网络实测。说明普通HFL算法确有区域状态/中间反馈，不能仅用单次平均概括；计算状态在同权限可被其他调度实现。
- [PipeSD](https://github.com/Ghanyunhe/PipeSD/tree/dac0f52eae7ba55e5def2d82003d8413cb58340c)。edge/docs/multiclient-implementation-summary.md blob 7d58912ec86c149ba853ec570d0476aaa71245e7：作者文档描述单共享云target权重，每task独立state，云验证串行无跨task batching；多个edge进程同一机器CPU竞争，非多独立设备，报告cloud GPU能耗而非全部端侧能耗；当时GPU offload不可用。该文档记录2026-03-26状态，不能泛化其他试验或论文全部结果。
- edge/src/engine.py blob 2e0613612df35067988ab6240b2010136947671e L188–211用局部概率阈值决定是否验证，L370–416/L652–666在边缘生成propose payload (tokens, probs整词表转list, task_id,n_past,index,should_verify)，msgpack.packb，无额外逐次云授权。等待云验证期间继续起草。edge/src/comm.py blob 27e2d5dbe57908faeaf84fda74c0f7e5433dd61f：单sender线程，quota=payload_size/bandwidth+base_latency，真实post后若elapsed<quota则sleep差值；不是恒把传输时间和真实HTTP时延相加，也不是物理吞吐测量。cloud/src/speculative_server.py blob 1fab3ac2bc2dff21fa2423769298aa0ce85d39ee L240–393验证、L394–431按task状态恢复和保存；L309以seed+global_idx生成随机数、L299分母加EPSILON、目标softmax与bonus配置等需另审计真实随机语义。这里只记录检查点，不认定源算法/论文错误或日志具备G3证据。PipeSD使用独立草稿模型，作为分布式调度/通信强基线，不称免独立草稿HFL实验。
- edge/src/multiclient.py blob 26612225cc2ccdfe5749b8b108f282e38bbcbc9d：sample-window和进程级makespan不同；本轮单请求时间和不是并发makespan。原始日志和训练trace不存在联结，不能据此证明HFL收益。

## 4. 真实公开日志重分析 E/E-H/P
[源数值字段及固定版本](source_metrics.json)，[代码](probe.js)，[完整结果](results.json)。node experiments/team-hca/R07/probe.js。实际GitHub读取PipeSD固定commit dac0f52eae7ba55e5def2d82003d8413cb58340c的edge/exp/exp__gsm/humaneval下nav_diag_pilot_v3四种方法原始JSON，路径和blob见source_metrics.json；每种5个任务ID 0–4，总20条。保留数值字段不复制生成文本；未独立核实硬件、输出质量/随机正确性、训练、所有论文配置。相同任务ID但生成长度不同，不能宣称质量匹配或普适排名。带宽配置2.5MB/s，单请求时间和；没有真实额外授权。真实字段汇总：
|方法|样本数|T秒|输出token|verify_num|逐次诊断事件数|逐次诊断候选token|sent计数|generated计数|加权TPOT秒|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Vanilla|5|124.0725|659|92|92|736|736|736|.188274|
|PipeSD|5|91.1569|727|129|126|625|673|769|.125388|
|HSL|5|97.4634|649|166|166|528|528|528|.150175|
|EdgeLLM|5|107.4974|774|152|148|702|787|880|.138886|
发现PipeSD task1/3和EdgeLLM task0/1/4的verify_num与诊断数组长度不一致，共5/20；未确认原因，保留两计数，不视为论文结果错误。原作者summary用PipeSD126/EdgeLLM148个诊断事件。sent为独特索引计数，不是封包日志，generated/verified/sent不同；不能将任一计数直接称实际RPC次数/总字节。N=verify_num仅反事实代理。
|方法|额外R=40ms时理想节省|65ms|100ms|200ms|达到15%所需R(ms)|
|---|---:|---:|---:|---:|---:|---:|
|Vanilla|2.88%|4.60%|6.90%|12.91%|237.99|
|PipeSD|5.36%|8.42%|12.40%|22.06%|124.70|
|HSL|6.38%|9.97%|14.55%|25.41%|103.61|
|EdgeLLM|5.35%|8.42%|12.39%|22.05%|124.80|
若PipeSD以126个诊断事件则阈值约127.67ms；EdgeLLM以148个约128.18ms。无额外控制的同信息自治平坦基线增量0。上述均为反事实，不是实际HFL改善或CI。

## 5. 实际序列化测量 E 与通信代理 P
当前本队Python3.13.5/msgpack1.1.2，按PipeSD源码payload字段构造固定token ID=1,task_id=0,n_past=10,index=0,should_verify=True,type=propose，每向量V32256相同合成浮点值1/V，实际执行：
```python
import msgpack
V=32256
for k in [1,4,8]:
    p={'type':'propose','tokens':[1]*k,'probs':[[1/V]*V for _ in range(k)],'task_id':0,'n_past':10,'index':0,'should_verify':True}
    print(k,len(msgpack.packb(p)),len(msgpack.packb(p,use_single_float=True)))
```
实际stdout：
```
1 290376 161352
4 1161300 645204
8 2322532 1290340
```
默认每float64 9B（类型标记+8数据），float32 5B；float32是仅字节对照，不保证改变精度后精确采样。默认单向量约290.4KB，2.5MB/s quota约116.15ms，非实际传输。按诊断候选token×9V得到四种总代理213.664/181.440/153.281/203.793MB；PipeSD按sent计数代理195.375MB。PipeSD仅验证向量quota代理72.576s，占日志时间91.157s的79.61%，不等于暴露通信占比：流水可重叠、quota是软件整形、真实发送重传/取消/封包元数据、版本及字段计数均未记录。不可由此声称15%总时间必可省。真实强基线包括固定二进制编码、DSSD按拒绝传输、DISCO服务端草稿副本、ERSS/直接精确残差抽样、共置内生SD及真实权限许可。训练的模型同步字节、版本更新、额外部署、真实传输与等待均未知。G5需全流程测量。

## 6. 原始文献核查与重合
2026-11-24实际检索 hierarchical topology learning, multi-timescale correction, distributed SD WAN, exact communication-efficient SD；读取原始论文页面/摘要，SpecFed v1及Speculation at a Distance、DISCO可读正文，PipeSD/MTGC原始代码。非全文附录/未索引研究穷尽，无正式文字查重。
- [HierFAVG](https://arxiv.org/abs/1905.06641) 2019/2020，Client-Edge-Cloud；[MTGC](https://papers.nips.cc/paper_files/paper/2024/hash/8fb96e8d0fbf591b1fa1ad85653d8417-Abstract-Conference.html) NeurIPS2024，区域/客户端控制变量。
- [Unified Analyses for HFL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b444767c015c50c14663a0ca9c3de58c-Abstract-Conference.html) ICLR2026，不同物理/逻辑聚合拓扑与异质性收敛，否定“所有层级学习等同普通一次FedAvg”的过宽说法。
- [PipeSD](https://proceedings.mlr.press/v306/han26k.html) ICML2026，边云流水和本地触发；作者报告相对其基线1.16–2.16×加速，非本队复现。
- [Speculation at a Distance](https://arxiv.org/abs/2606.25091) 2026-06-23，真实/解析WAN延迟与共置基线和多租户容量分析；已有延迟门槛，不能宣称本轮简单RTT式为原创。
- [DISCO](https://arxiv.org/abs/2610.35957) 2026-10-25，服务端复制草稿而避免全词表通信；[SpecFed](https://arxiv.org/html/2604.25777v1) 2026-04-28，独立小草稿+分布式模型概率加权及top-K重建、分析偏差，非HFL参数学习且非保证原始目标无损的通用替代。
- [SPADE](https://arxiv.org/abs/2608.13076) 2026-08，分布式边云推理；[GoodSpeed](https://doi.org/10.1109/INFOCOM59046.2026.11571335) INFOCOM2026，分布式自适应SD公平goodput（检索摘要，未完整核查）。
库内CFA、FCIP、FEC、CPG、QCRA/CGE为直接近邻；普通权限委托、流水调度与逐事件模拟均非新基础数学。

## 7. G1–G7及下一轮
主观加权创新5.425≈5.4/10（问题7×25%、机制4×25%、理论5.5×20%、价值7×15%、距离3.5×15%），非同行评审。G1 FAIL，已有边缘自治/调度与FCIP重合；G2 CRITICAL FAIL，真正权限区别未证且普通授权平坦和已有PipeSD可复制；G3 PARTIAL/FAIL，限定模拟/信息因果定理，实际完整精确SD/数值未验证；G4 FAIL（第三方20条推理记录非本队两尺度真实HFL）；G5未验证；G6未验证无80%依据；G7局部代码、来源与实测可复算/整体未过。好结果同质量资源对最强公平基线真实全流程−15%或有效工作+20%，两预训练尺度、三非IID/时延、三独立训练seed及CI、全部G。
未执行1–2×40GB：先部署真实Qwen2.5-1.5B/3B或带内生头两尺度模型与8逻辑客户端2区域，用预定公开任务、固定正式快照、合法区域学习；记录每个真正跨层事件的状态归属、授权来源、可委托性、依赖、请求/响应时间和字节、版本、训练梯度、计算/通信重叠；区分严格政策要求与人为设计。对照优化普通HFL+内生SD、普通强平坦FedAvg/局部多步委托、PipeSD/DISCO适用通信子系统、同节点共置及同信息同权限委托。若根本没有额外串行授权，直接否决LEA。若存在，可在同样合法权限下研究真实减少授权依赖的新数学原语，而非给予候选特权。单卡逻辑模拟非物理多设备。优先要求真实权重与学习-验证联合trace或部署方可验证约束；无资源时进行有限源码审计，不再重复构造假设权限壁垒。
**Reject and Pivot**。保留权限两世界例、模拟边界、真实PipeSD源码与原始数值、计数差异、实际MsgPack序列化及RTT反事实阈值。禁止把普通预授权/分布式调度或全量概率通信弱基线改名当双向创新。
