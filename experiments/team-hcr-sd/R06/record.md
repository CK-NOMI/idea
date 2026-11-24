# team-hcr-sd/R06｜SCA：SPEC-RL真实源码审计及不可修复接受质量下界

日期2026-11-24。起始main 773b453d0483e5059618e0ca245c730d6e6dd515，README blob 994dbc520ccbaf7d88febee50f25105d650301f6。验证上轮已读838720eb37d1784df04c4da13e60e1385ebabfe9原样前缀，读新增NC-FSD、FCMC、FOAS及递归文件树；追加前读并发FCKG、CGE、QCRA、FCF、FRGB、CFA。CGE/QCRA为公开日志正确性及成本审计，CFA为表观协同，其他为梯度、曲率和流水线，区别于本轮采样源代码。未独立重跑他队。多次检查发现README并发变化即停止旧写入。R05条件依赖/组边际问题不重复声称原创。

**CRITICAL / Reject and Pivot**（保留源码审计、数学边界、局部复现）。T=限定理论，E=实际CPU运算及源码审计，P=成本代理，H=假说，E-H=论文报告。无真实LLM/HFL训练、vLLM运行、GPU、网络、wall time或CI；未向上游提交issue/patch，不声称论文任务质量结果错误。

## 1. 官方源代码证据
固定 ShopeeLLM/Spec-RL main 97da99ed54d663b238c417c728bc824c79cfab6f，见 https://github.com/ShopeeLLM/Spec-RL/tree/97da99ed54d663b238c417c728bc824c79cfab6f 。阅读README、完整递归树、verl/utils/speculative_decoding.py全文以及相关trainer、worker、actor、脚本。
- verl/trainer/ppo/ray_trainer.py L1623–1686：同batch先advantage/update_actor后保存response和更新前重新评分的old_log_probs；L1270–1316下一bucket读取历史缓存并当前actor评分。历史训练参与者被回收，R05条件依赖的前提获源码支持；实际后验/偏差未测。
- L1321–1353固定seed=1234；正spec_bias转负bias。verl/utils/speculative_decoding.py L5–14重置generator，L30–48/89–103接受a=min(1,exp(new_logp−old_logp+b))（scale=1,b为正宽容强度）；training_scripts/spec-rl/1.7B-grpo-lenience-0.5.sh L29–30例b=.5。论文明确宽容权衡，不将其等同无损。
- ray_trainer.py L1355–1404构建只含接受前缀的ctx、传剩余长度调用generate_sequences；verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py L318–357同步路径普通SamplingParams/max_tokens及generate；trainer L1436–1457拼接。该路径未传旧分布或显式正残差；理想普通目标采样下首拒绝补p非(p−q)_+。未审计全部异步及外部引擎。
- trainer L1550–1567按当前actor重算old_log_probs，L1675–1686缓存；模型评分未必真实混合行为律，重新评分不能纠正已偏采样。dp_actor.py L259–263普通非融合路径按温度缩放选中token评分；若实际采样含top-p等processor另查，未证实际配置错配。
- speculative_decoding.py L240–296 build_ctx，L270–280：N>0且待续写全部cut=0时max_k=0，resp_cut=[N,0]，clamp(0,-1)后gather越界；CPU最小同式测试复现。建议max_k==0跳过prefix gather、直接保留prompt；未测真实训练触发频率。
- 固定seed同shape/dtype/device重复流。共享U反例：固定p(1)=.5,q(1)=.8，两次独立候选却共用U，标准最大耦合且正确残差补0；U≤.625时两输出各Bern(.8)，否则均0。边际均.5，联合(00,01,10,11)=(.4,.1,.1,.4)，相对独立均匀TV=.3。非实际跨轮偏差量级。需持久RNG/计数器按轮、请求、位置分离，seed仅实验初始设定。

## 2. 理论及机制（T）
HFL客户端私有任务与奖励，区域更新，云聚合；正式目标为新固定快照p的预定独立rollout组，缓存或内生头无独立完整草稿LLM。允许合法历史缓存本地使用，不承诺DP。R05已证真实条件提议Q可能非旧q。本轮先隔离依赖，固定p及真实独立q；接受a(v)，不可撤回质量m(v)=q(v)a(v)，拒绝δ=1−Σm；任意补采r输出s=m+δr。
**min_r TV(s,p)=E=Σ[m−p]_+。** 超额不可被非负补采减少；不足总量D=Σ(p−m)_+=δ+E，δ>0取r=(p−m)_+/D即达下界；δ=0直接输出m。精确iff m≤p，δ>0时唯一r=(p−m)/δ。宽容ell>1且p/q全支持、p≠q时存在q>p项m=min(q,ell*p)>p，任何不撤回接受的补采均不精确；非任务质量下界。ell=1普通p补采仅在m=(1−δ)p时精确。p(1)=.8,q(1)=.5时s(1)=.74、TV=.06；ell=e^.5时s(1)=.6362046、TV=.1637954，任意补采至少TV=.1297443。可撤回/重新验证另论。
固定缓存x或未知实际Q（允许点质量），对所有Q精确且不可撤回的算法必有P(Y=x|x)=p(x)，故接受≤p(x)。target-first直接采正式Y、匹配继续达到上限，无需后验，但SRT/SuffixDecoding已有。真实Q可知才用最大耦合+残差；旧评分非Q。任意自适应缓存在当前独立目标随机性之前固定时，target-first给定训练状态及缓存逐步保持p；greedy正式argmax，processor/top-p、EOS/KV/bonus、组独立、浮点另审计。普通AR、SD及同信息平坦可执行，正式HFL学习无不可替代反向作用。
训练解析例：Bern(p)奖励Y、score=Y−p，正式期望p(1−p)，实际Bern(s)则s(1−p)；p=.8,s=.74时.148 vs .16。外生单步REINFORCE非真实GRPO，重新评分不能改变样本律。

## 3. 实际CPU证据（E）
Python3.13.5/PyTorch2.10.0+cpu/NumPy，单线程CUDA false。手工摘录源代码同式RNG、截断、gather子表达式，非完整仓库import。4096×8 FP64 seed1234两次cut完全一致；seed1235相同cut率.13818359375。gather两行3token、prompt长4：[0,0] RuntimeError index -1 is out of bounds for dimension 1 with size 0；[0,1]、[1,2]成功。
枚举iid Bern(q=.5)候选、固定Bern(p=.55/.65/.8)、b=0/.2/.5、L=1/4/8/12共36配置，无随机seed。首拒绝普通p续写序列律 prod_j m(y_j)+δΣ_j prod_{i<j}m(y_i)prod_{i≥j}p(y_i)；标准正残差以m0=min(q,p)及(p−m0)(y_j)替代拒绝项。36配置正确序列TV最大5.430277105085464e−17。L12实际输出（非LLM）：
|p(1)|b|首tokenTV|任意补采下界E|序列TV|平均接受前缀|
|---:|---:|---:|---:|---:|---:|
|.55|0|.022500|0|.048963|8.733158|
|.55|.2|.050000|.050000|.139723|12|
|.55|.5|.050000|.050000|.139723|12|
|.65|0|.052500|0|.082279|4.860630|
|.65|.2|.102869|.077491|.203069|7.607757|
|.65|.5|.150000|.150000|.400058|12|
|.8|0|.060000|0|.082608|2.301037|
|.8|.2|.095424|.044281|.135849|2.826432|
|.8|.5|.163795|.129744|.256469|4.354510|
固定q=.5时target-first平均匹配Σ .5^k，L12约.999756；正确最大耦合Σα^k，p=.8时2.301037。前缀长度不是时间。实验隔离补采偏差，不测真实训练依赖。以下为实际执行代码等价整理，复制为probe.py运行python probe.py复算全部36行及张量输出：
~~~python
import torch, math, itertools, numpy as np, json
torch.set_num_threads(1)
def rng(t,seed):
    g=torch.Generator(device=t.device if t.device.type=="cpu" else "cpu")
    g.manual_seed(seed)
    try:return torch.rand(t.shape,device=t.device,dtype=t.dtype,generator=g)
    except TypeError:
        torch.manual_seed(seed)
        return torch.rand_like(t)
def cut(old,new,mask,seed=1234,bias=0.):
    valid=mask.bool(); length=valid.sum(dim=1).long()
    delta=(new-old).masked_fill(~valid,0.)-bias
    accept=(~valid)|(delta>=0)|(torch.log(rng(new,seed).clamp_min(1e-12))<=delta)
    bad=valid&(~accept)
    return torch.where(bad.any(dim=1),torch.argmax(bad.to(torch.int8),dim=1),length)
z=torch.zeros(4096,8,dtype=torch.float64)
q=z+math.log(.5);p=z+math.log(.4);mask=torch.ones_like(z,dtype=torch.bool)
a=cut(q,p,mask);b=cut(q,p,mask);c=cut(q,p,mask,seed=1235)
print("fixed_seed",torch.equal(a,b),"fresh_seed_equal_rate",(a==c).double().mean().item())
def gather_prefix(response_ids,cut_idx,P):
    N,R=response_ids.shape;k=cut_idx.clamp(min=0,max=R)
    max_k=int(k.max().item()) if N>0 else 0
    ctx_len=P+max_k;col=torch.arange(ctx_len).unsqueeze(0)
    Lp=torch.full((N,1),P,dtype=torch.long);start=(ctx_len-(Lp.squeeze(1)+k)).unsqueeze(1)
    return torch.gather(response_ids[:,:max_k],1,(col-(start+Lp)).clamp(0,max_k-1))
for cuts in [[0,0],[0,1],[1,2]]:
    try:print(cuts,gather_prefix(torch.tensor([[1,2,3],[4,5,6]]),torch.tensor(cuts),4))
    except Exception as e:print(cuts,type(e).__name__,str(e))
rows=[]
for p in [.55,.65,.8]:
    q=np.array([.5,.5]);pp=np.array([1-p,p])
    for b in [0,.2,.5]:
        m=np.minimum(q,np.exp(b)*pp);delta=1-m.sum();r=m+delta*pp
        for L in [1,4,8,12]:
            Y=np.array(list(itertools.product([0,1],repeat=L)))
            a=m[Y];t=pp[Y]
            pre=np.concatenate([np.ones((len(Y),1)),np.cumprod(a,axis=1)],axis=1)
            suf=np.concatenate([np.cumprod(t[:,::-1],axis=1)[:,::-1],np.ones((len(Y),1))],axis=1)
            out=pre[:,-1]+delta*(pre[:,:-1]*suf[:,:-1]).sum(axis=1)
            true=suf[:,0];assert abs(out.sum()-1)<1e-12
            m0=np.minimum(q,pp)
            ep=np.concatenate([np.ones((len(Y),1)),np.cumprod(m0[Y],axis=1)],axis=1)
            exact=ep[:,-1]+(ep[:,:-1]*(pp-m0)[Y]*suf[:,1:]).sum(axis=1)
            rows.append(dict(p=p,b=b,L=L,accepted_mass_excess=float(np.maximum(m-pp,0).sum()),
                firstTV=float(np.abs(r-pp).sum()/2),seqTV=float(np.abs(out-true).sum()/2),
                exactTV=float(np.abs(exact-true).sum()/2),expected_prefix=float(sum(m.sum()**k for k in range(1,L+1))),
                exact_prefix=float(sum(m0.sum()**k for k in range(1,L+1))),targetfirst_prefix=float(sum(.5**k for k in range(1,L+1))),output1=float(r[1])))
print(json.dumps(rows,indent=2))
print("max corrected seq TV",max(r["exactTV"] for r in rows))
~~~

## 4. 文献、成本、验收
2026-11-24核对原始页面/摘要及可读正文：[SPEC-RL](https://arxiv.org/abs/2509.23232) Algorithm1/§3.1/§4.3（作者报告ell=1加速1.22x、ell=e^.5准确率48.0、极宽容40.1，非本队复现）；[SRT](https://arxiv.org/abs/2601.09083) 2026-01历史树model-free精确验证，作者报告最高2.08x rollout；[SpecRoll](https://arxiv.org/abs/2608.04962) 2026-08在线内生头；[AReaL](https://arxiv.org/abs/2505.24298) 2025异步；[Leviathan](https://proceedings.mlr.press/v202/leviathan23a.html) ICML2023；[SuffixDecoding](https://arxiv.org/abs/2411.04975) 2024；[BADC](https://arxiv.org/abs/2611.04042) 2026-11-02原始摘要检索可得、abs打开失败，镜像检索指出真实提议、残差、新鲜随机性、点质量契约，直接原创性威胁，全文待核；[WAR](https://arxiv.org/abs/2607.17299) 2026。未穷尽其他分支/issues/全文，无正式文字查重，不声称首次发现。
P：旧token+选中FP32 logprob最低8B/token；精确残差需实际前缀p,q完整支持或精确采样接口，V151936每FP32分布607744B/位置，非必须网络上传（共置、首拒绝重算、流式强基线）。target-first无需旧模型但需正式抽样；计全部重评分、失败后缀、通信、奖励、反向、更新、质量；强基线生成占f、额外改进r则净节约≤fr−额外开销比例。近似复用速度非无损性能。
CRITICAL：SRT/普通target-first替代，普通AR、SD、同信息平坦可执行，无HFL双向原语。主观等权创新6.5/10（问题8、机制5、理论7、价值8、距离4.5），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL限定定理、源码和局部测试，真实greedy/随机/processor/浮点未验；G4 FAIL；G5未验证；G6无80%依据；G7局部复算/整体FAIL。好结果同质量预算最强公平基线真实总时间−15%或有效工作+20%，两预训练尺度三非IID/时延三种子CI及全部G。
H未执行：1–2×40GB，Qwen3-1.7B/4B或可用两尺度，固定模型/processor，SPEC-RL零/正bias、独立候选正残差、SRT目标优先、普通AR、最优内生SD、同信息平坦；先冻结模型重复抽样、行为概率/随机流/零截断测试，再8逻辑客户端2区域公开数学/代码任务三种子三非IID/时延，计质量、生成/训练/通信/全流程及CI，独立重复训练实例估计真实缓存依赖；单卡模拟非物理并行。先核对上游更正、与作者沟通前保持限定结论。若无HFL特异机制，停止缓存路线，保留正确性审计规范，转向真实物理约束下新计算问题。**Reject and Pivot**。
