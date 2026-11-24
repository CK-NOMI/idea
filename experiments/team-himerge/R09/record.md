# team-himerge/R09｜FMR：联邦随机掩码与精确投机验证的联合学习分布
2026-11-24；起始 main e913dfd74ead040fbf95eb34a9f62b78c47e378c、README blob 4d533746be7909cd7a4c96250636b5c92f24bc1c，308919 JS chars。核对R08已读快照00cd7ff26f44fe4c4fc9048b3124b4ca309057b3原样前缀，阅读新增FIA-SD全文并重查PRE-11、HCR/R04反向路径、FCRR/LBS、FOAS、FUDG及dropout相关记录。未独立复跑他队。区别：固定前缀下正式无dropout目标与训练随机掩码的联合分布，非深层梯度随机抽样或新隐私协议。已有标准耦合和consistent dropout，不声称新基础概率数学。
**CRITICAL / Reject and Pivot**。T=限定理论，E=实际CPU公开文本线性模型训练与精确掩码枚举，P=成本代理，H=假说。实际检查Python3.13.5 torch2.10.0+cpu CUDA false、无HF缓存；本轮未尝试联网下载权重。无预训练Transformer/真实HFL/多token SD生成/GPU/物理网络/总wall time/CI。

## 1. 场景及数学
固定授权前缀h、参数快照，p为正式无dropout的分布（冻结teacher/stop-gradient）；q_m为同模型随机失活预测路径，M~mu为事先规定的训练掩码分布，q可来自共享主干的轻量随机头，但本轮具体实验是同一线性头加掩码，未证起草比正式路径更便宜。不使用独立完整草稿LLM。客户端保留数据与合法计算图，边缘/云按固定样本权重汇总更新。目标不是改变正式p为MC dropout混合，而是对正式Y~p且M~mu独立的固定前缀学习目标 E_{h,Y,M}[−log q_M(Y|h)] 估计梯度（教师p停止梯度）；不是完整RL策略梯度、同轨迹PPO或所有监督训练的目标。普通真实标签独立于M时需另审计，不宣称都受此反例影响。
T1：若M~mu后直接使用X~q_M生成并用X训练，联合律为mu(m)q_m(y)，而目标为mu(m)p(y)。联合TV = E_M TV(q_M,p)；边际TV(Y,p)=TV(E q_M,p)≤联合TV，由三角不等式。边际正确不足以认证训练联合律。反例M=±1等概率，z_m=theta*m，theta=log9，p=(.5,.5)，q_+(1)=.9,q_-(1)=.1，生成边际仍为p，但联合TV=.4。令梯度g(m,y)=m(q_m(1)−1{y=1})，独立目标Eg=.4，自采样Eg=0。该例为一般随机特征符号掩码，不是原始非负Bernoulli dropout；真实dropout实验另见下。
T2：固定M=m执行标准SD单步最大耦合：X~q_m，a=min(1,p(X)/q_m(X))，接受Y=X，拒绝Y~(p−q_m)_+/delta_m，delta_m=TV(p,q_m)。对每个m，接受质量min(p,q_m)，残差质量(p−q_m)_+，其和p；故联合(M,Y)恰为mu×p。拒绝时可在同前缀使用原掩码的完整q_m分布计算修正标签的loss/梯度，接受时同样；不可使用原被拒绝后缀的错误历史；若需要深层训练目标还需相应深层训练计算图。若训练只选择接受样本，令A_m=Σmin(p,q_m)，则M|accept分布mu(m)A_m/E A_m，Y|M,accept分布min(p,q_m)/A_m，两者一般偏离目标。固定h亦成立；不同h按接受事件选择还改变h的权重。上例A_m=.6，接受条件梯度(.6*.9−.5)/.6=.0666667，非.4。整体接受率alpha时，若梯度范数≤G，||E[g|accept]−Eg||≤2G(1−alpha)，是绝对误差上界；当目标梯度很小时相对误差仍可能大，不宣称实际训练崩溃。
T3：对于任意固定外生h，若新掩码与既有历史独立且逐位置真实条件q、p、正残差、合法首拒绝截断/bonus/EOS/KV均正确，正式序列的条件律为p；训练目标掩码的独立性需对实际提交位置及完整历史条件检查，不能把条件于接受/树选择的mask当无条件样本。共享跨序列mask改变联合采样噪声；如果mask在当前正式输出后再选择或参与参数更新，需重新证明。greedy取正式argmax及统一tie-break，训练mask独立采样，不从随机证明推出greedy生产等价。temperature/top-p须使用实际处理后分布，完整block/浮点/processor/真实生成未验证。
机制仅尝试保留随机内生头的前向计算，并通过真实条件标准SD保证正式输出、掩码联合正确。需要记录mask/版本和梯度tape，拒绝修正位置亦计算原mask loss，bonus独立新mask或普通训练。普通AR直接从p生成再独立采M并计算loss即达到相同联合律，完整p软标签蒸馏也能直接算目标期望；若q_m与p共享冻结特征，普通特征缓存可复用。不允许以无dropout训练代替有dropout目标后声称无损；但许多真实RL系统本来关闭dropout或选择consistent dropout行为策略，公平基线须匹配预先定义目标。HFL固定快照线性聚合没有新原语，同信息平坦可复制。

## 2. 实际CPU实验 E
Python3.13.5 NumPy2.3.5，OPENBLAS_NUM_THREADS=1 python probe.py。实际读取本机Python标准库email/json/urllib三域31个顶层.py文件（公共CPython源码，https://github.com/python/cpython/tree/v3.13.5/Lib ；发行版可能有补丁）。逐域按文件名连接解码、小写，非27字符映射空格，前80%训练、后20%留出。训练字符298190/39075/129737，留出74548/9769/32435（预测位置74547/9768/32434）；固定平滑全局bigram底座，4维随机固定特征+可训练4×27线性适配头，dropout在4维特征上（非标准LoRA实现、非Transformer）；seed11/29/73与drop .1/.3/.5，9次独立初始化及每次150次实际监督SGD（3域轮替、batch128、lr.2），共1350步。训练是集中CPU、域轮替非HFL；评估对全部16个Bernoulli mask精确枚举及留出前缀频数加权，9模型×3域=27行。正式p为当前不加mask头，冻结教师；q_m为相同头带inverted dropout。评估目标为冻结p的自蒸馏CE期望梯度，不是实际随机生成或RL，未测NLL/最终训练质量、实际时延或CI。三域共享同一训练模型，不是三独立训练seed。所有梯度解析计算。
|drop|alpha|marginal TV|joint TV|accepted mask TV|true gradient norm|accepted gradient absolute bias|mean relative bias|
|---|---|---|---|---|---|---|---|
|.1|.9962903001|.0000504599|.0037096999|.0012659756|.0013813153|.0010519875|.7382328818|
|.3|.9901787433|.0001937588|.0098212567|.0011116654|.0052011604|.0035017110|.6583002564|
|.5|.9859464122|.0003502063|.0140535878|.0004354571|.0101738061|.0063239183|.6070534368|
均为9个seed×domain行等权均值，非位置微平均；relative为每行bias/norm再平均。自采样q_m的CE期望梯度严格0，因此相对偏差1（当目标非零）。标准SD接受+残差联合质量的最大逐坐标数值误差0（实际FP64数组运算；不是生产采样位级认证）。高接受率下相对偏差仍约61–74%，但绝对梯度较小，未证实际训练损害。无MC抽样结果，概率由精确枚举得到。
逐配置实际原值（9有效数字舍入，代码重跑完整JSON含原文manifest，maxError每行0、naiveRelativeBias每行1）：
~~~text
seed d region alpha margTV jointTV maskTV norm bias rel
11 0.1 0 0.99492428 7.09494374e-05 0.00507571957 0.00171746938 0.00202084314 0.00160207046 0.792773287
11 0.1 1 0.994962258 7.3945712e-05 0.00503774239 0.00170441882 0.00213308547 0.00180594612 0.846635609
11 0.1 2 0.994909095 7.38805671e-05 0.00509090451 0.00171523967 0.00209837788 0.001733171 0.825957526
11 0.3 0 0.987170267 0.000239832348 0.012829733 0.00159951004 0.00733632041 0.0051164182 0.697409316
11 0.3 1 0.98724474 0.000249247638 0.0127552596 0.00168474431 0.00774145335 0.00565954563 0.731070172
11 0.3 2 0.98714495 0.000249533896 0.0128550499 0.00165100881 0.00762734193 0.00545905225 0.715721453
11 0.5 0 0.982447605 0.000412880633 0.0175523948 0.00051798654 0.0124063078 0.00879175754 0.708652217
11 0.5 1 0.982945411 0.000423438915 0.017054589 0.00063877239 0.0125611861 0.00942155616 0.750053064
11 0.5 2 0.982646513 0.000426236201 0.0173534872 0.000645716859 0.0126516435 0.00930777207 0.735696678
29 0.1 0 0.997107666 4.04578554e-05 0.00289233425 0.00103410086 0.00105737499 0.000593708887 0.561493217
29 0.1 1 0.997360273 3.54206035e-05 0.00263972654 0.000947030272 0.000922406165 0.000517183519 0.560689573
29 0.1 2 0.997287316 3.67538916e-05 0.00271268386 0.000975359082 0.000965282163 0.000565276763 0.585607799
29 0.3 0 0.992215403 0.000174113531 0.00778459736 0.000773846915 0.00410880618 0.00212053701 0.516095654
29 0.3 1 0.992891092 0.000153807381 0.0071089083 0.000720224857 0.00362027041 0.00187569931 0.518110277
29 0.3 2 0.992742667 0.000162002341 0.0072573326 0.000723701269 0.00375871225 0.0020160663 0.536371546
29 0.5 0 0.987502975 0.000370901164 0.0124970249 0.000140507766 0.00956544058 0.004918994 0.514246464
29 0.5 1 0.988577546 0.000327750423 0.011422454 0.000116216071 0.00838216081 0.00430645777 0.513764632
29 0.5 2 0.988351959 0.000342616241 0.0116480406 0.000135678084 0.00873026011 0.00458371543 0.525037671
73 0.1 0 0.996460723 4.39828778e-05 0.00353927677 0.00117826078 0.00120474444 0.0009114132 0.756519944
73 0.1 1 0.996816869 3.89823155e-05 0.00318313106 0.00105676597 0.000998033834 0.000878676474 0.880407502
73 0.1 2 0.99678422 3.97660076e-05 0.00321577974 0.00106513533 0.00103168997 0.000860441279 0.834011479
73 0.3 0 0.990132641 0.000183519346 0.00986735923 0.00103036104 0.00463391423 0.0032361075 0.698352914
73 0.3 1 0.991087664 0.000164250502 0.00891233557 0.000897742761 0.00392905359 0.00303009156 0.771201382
73 0.3 2 0.990979265 0.000167522537 0.00902073495 0.000923848921 0.00405457141 0.00300188139 0.740369594
73 0.5 0 0.986371383 0.000299178479 0.0136286165 0.000581763552 0.00964930637 0.00539644194 0.55925698
73 0.5 1 0.987415539 0.00027034942 0.0125844611 0.000579205891 0.0086839418 0.00507185629 0.584050009
73 0.5 2 0.987258778 0.000278505133 0.0127412221 0.000563266611 0.00893400797 0.00511671377 0.572723215
~~~
## 3. 实际执行脚本
提取为probe.py；OPENBLAS_NUM_THREADS=1 python probe.py。脚本计算31文件SHA256 manifest及完整JSON，不依赖网络。
~~~python
import numpy as np, sysconfig, pathlib, hashlib, itertools, json, platform
V=27; D=4; alphabet=' abcdefghijklmnopqrstuvwxyz'; seeds=[11,29,73]; drops=[.1,.3,.5]
root=pathlib.Path(sysconfig.get_path('stdlib')); corp=[]; manifest={}
for domain in ['email','json','urllib']:
 files=sorted((root/domain).glob('*.py')); raw=[]
 for f in files:
  b=f.read_bytes();manifest[str(f.relative_to(root))]=hashlib.sha256(b).hexdigest();raw.append(b.decode('utf-8',errors='replace'))
 s=''.join(raw).lower(); a=np.array([alphabet.index(c) if c in alphabet else 0 for c in s]);n=int(.8*len(a));corp.append((a[:n],a[n:]))
counts=np.ones((V,V))*.5
for a,_ in corp:np.add.at(counts,(a[:-1],a[1:]),1)
base=np.log(counts/counts.sum(1,keepdims=True)); bits=np.array(list(itertools.product([0.,1.],repeat=D)))
def soft(z):
 a=np.exp(z-z.max(axis=-1,keepdims=True));return a/a.sum(axis=-1,keepdims=True)
rows=[]
for seed in seeds:
 for d in drops:
  rng=np.random.default_rng(seed); features=rng.normal(size=(V,D)); W=np.zeros((D,V));mu=np.prod(np.where(bits==1,1-d,d),axis=1)
  for t in range(150):
   a=corp[t%3][0];idx=rng.integers(0,len(a)-1,128);h=features[a[idx]]*(rng.random((128,D))>=d)/(1-d);p=soft(base[a[idx]]+h@W);p[np.arange(128),a[idx+1]]-=1;W-=.2*h.T@p/128
  for e,(_,a) in enumerate(corp):
   freq=np.bincount(a[:-1],minlength=V)/len(a[:-1]);h=features; p=soft(base+h@W);hm=h[:,None,:]*bits[None,:,:]/(1-d);q=soft(base[:,None,:]+hm@W);m=np.minimum(q,p[:,None,:]);A=m.sum(-1);joint=np.einsum('h,m,hmd,hmv->dv',freq,mu,hm,q-p[:,None,:]);accept=np.einsum('h,m,hmd,hmv->dv',freq,mu,hm,A[:,:,None]*q-m);alpha=np.einsum('h,m,hm->',freq,mu,A);mix=np.einsum('m,hmv->hv',mu,q);margtv=np.einsum('h,h->',freq,.5*np.abs(mix-p).sum(1));jointtv=1-alpha;acmask=np.einsum('h,hm->m',freq,A)*mu/alpha;masktv=.5*np.abs(acmask-mu).sum();g=np.linalg.norm(joint);bias=np.linalg.norm(accept/alpha-joint);corrected=m+np.maximum(p[:,None,:]-q,0);err=np.max(np.abs(corrected-p[:,None,:]));rows.append(dict(seed=seed,drop=d,domain=e,positions=len(a)-1,alpha=float(alpha),marginalTV=float(margtv),jointTV=float(jointtv),acceptedMaskTV=float(masktv),trueGradNorm=float(g),acceptedGradBias=float(bias),acceptedRelativeBias=float(bias/max(g,1e-20)),naiveRelativeBias=1. if g>0 else 0.,correctedMaxError=float(err)))
print(json.dumps(dict(env={'python':platform.python_version(),'numpy':np.__version__},seeds=seeds,drops=drops,train_steps=150,domain_lengths=[(len(a),len(b)) for a,b in corp],manifest=manifest,rows=rows),indent=2))
~~~
## 4. 成本、文献与裁决
若随机mask影响全部d维特征，每位置直接保存mask为d bit（例如1536 bit=192B，1024位置196608B），还需RNG状态、版本/执行顺序、激活及训练图；可用counter-based RNG或种子重建，不能假定不同kernel随机数消费一致。冻结特征BF16 d1536一位置3072B；真实激活tape远大于mask。掩码q的计算成本未测，正式p验证仍需完整计算。普通AR+独立mask+冻结特征缓存、完整软标签蒸馏、普通HFL+LayerSkip、consistent dropout、PrefixFlow、同信息平坦均是强对照；若已有真实训练使用dropout0，此问题消失，不得人工增加dropout制造优势。对固定工作量，若可复用训练前向占总时间f，真正可复用比例c，额外成本h（相对总时间），理想节约≤fc−h，未测f/c/h；不能把alpha当c或时间。训练随机mask可能增加验证和显存成本。
2026-11-24实际检索并核对原始论文摘要/页面（非穷尽全文）：[LayerSkip ACL2024](https://aclanthology.org/2024.acl-long.681/) layer dropout与早退自投机；[Don't Drop Dropout ICML2026](https://proceedings.mlr.press/v306/elhoushi26a.html) 训练稀疏与推理跳层；[Consistent Dropout for Policy Gradient RL](https://arxiv.org/abs/2202.11818) 2022 同mask策略学习（目标不同，本轮p无dropout且M独立）；[LoRA](https://arxiv.org/abs/2106.09685) 2021低秩适配；[Dropout as Bayesian Approximation ICML2016](https://proceedings.mlr.press/v48/gal16.html) 混合随机预测；[When RL Meets Adaptive Speculative Training ICML2026](https://proceedings.mlr.press/v306/wang26j.html) 训练服务一体化；[Diagnosing Training Inference Mismatch](https://arxiv.org/abs/2605.14220) 2026-05；[On the Origin and Evolution of Training-Inference Mismatch](https://arxiv.org/abs/2610.40945) 2026-10；[PrefixFlow](https://arxiv.org/abs/2606.01143) 2026共享前缀训练反传；基础标准SD [Leviathan](https://proceedings.mlr.press/v202/leviathan23a.html)。原始摘要检索支持，不声称完成全文公式对比或正式文字查重。已有随机性、训练推理一致性、标准SD最大耦合和激活缓存覆盖核心。
CRITICAL：条件标准SD本身即实现所需联合律，普通AR+独立训练mask/全软标签、平坦同信息可替代；没有真实训练要求随机mask与验证前向共享的证据。主观创新5.8/10（问题7.5、机制4.5、理论6.5、价值6.5、文献距离4，25/25/20/15/15%=5.875），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL（单步联合律，完整实际SD未验证）；G4 FAIL（公开文本低秩线性模型）；G5/G6未验证，无80%证据；G7局部复算/整体未过。好结果同正式目标、质量、预算最强公平基线真实全流程−15%或有效工作+20%，两预训练尺度三非IID/时延三独立种子及CI、全部G。未执行1–2×40GB：先核查真实训练是否启用dropout、是否允许共享训练tape、mask与实际目标/采样的关系，两个预训练尺度（如1.5B/3B）、8逻辑客户端2区域，≥10000真实逐位置mask/版本/p/q/accept/gradient及阶段时间；对照普通AR独立mask、完整soft target、普通HFL+生产SD、consistent dropout适用目标及同信息平坦。仅在真实共享计算需求成立且有G2双向介入及剩余可优化时间份额时继续。不为本否决机制投入GPU。保留联合TV恒等式、边际正确但梯度错误反例、accept-only选择偏差及实际结果。Reject and Pivot。
