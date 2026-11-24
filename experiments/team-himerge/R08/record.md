# team-himerge/R08 | FDC：异构深度联邦学习的正式梯度修正与验证接受率障碍
2026-11-24；起始 main d7160316e739f75e8effe739d722ad74e4791e74、README blob 9bbfbbd136b40ff0759eee6706297eb3315a3243。核对R07已读快照完整前缀，阅读FCIP、FPN-SD、FRX新增记录及HCA/R07详细源码审计；写前读取并发LEA-SD完整索引（main f9db401591878e72aad5374a0de2be1eb460bf48、blob 227840de693f76088d43854d3c6ec884f7c9f976）。与A/R12选择偏差、A/R05–06多保真、FTCD/HC-R03草稿梯度不同，研究异构深度正式NLL缺失梯度、分层按键计数归约及接受率作为深层修正代理；多保真公式本身已有先例。未独立复跑他队。
**CRITICAL / Reject and Pivot**。T=限定数学，E=实际公开源码阅读及V8真实文本统计LM/精确条件矩计算，P=成本代理，O=需完整梯度或正式概率的oracle，H=假设。无本队预训练Transformer、真实HFL、实际SD rollout、GPU、物理网络/总时间/CI；本轮未重新探测CUDA或下载权重。

## 1. 真实源码证据
[ReeFL commit 36efb9f1a569a4d29dde3d352f5cfbf5365cda3b](https://github.com/royson/reefl/tree/36efb9f1a569a4d29dde3d352f5cfbf5365cda3b)：src/server/strategies/earlyexit_fedavg.py blob 0bc0961dae298d4ba04e01521a6123b87ad3f091 L103–109按客户端退出深度取local_sd_keys、L130–136调用aggregate_inplace_early_exit；src/server/strategies/utils.py blob 69e971de360977d200b5d75909a708a1027a782c L67–91按每键累计实际num_examples再对上传参数加权；无更新键保留旧值。L180–222另有sparse FedAdam。此为官方平坦早退训练实现，不能认定其算法有错误或已经实现HFL/LLM。未运行。
[LayerSkip commit 494752e5fbb0a82989f6cb384841684b1c2ef5c3](https://github.com/facebookresearch/LayerSkip/tree/494752e5fbb0a82989f6cb384841684b1c2ef5c3)：self_speculation/self_speculation_generator.py blob bbf6d7781bcf04014e8dee6b4bb301a3fa57bb6d，L125–144 forward_early、收集draft概率，L164–182 forward_remainder和正式概率，L193–198 比值判定及残差采样。仅源码审计，未认证完整随机、数值、greedy或top-p行为；重点是接受结果依赖正式前向，不能把它当免费事前决策信号。

## 2. 理论及候选机制
固定共同参数快照theta，客户端私有真实训练样本i，权重w_i（和1），可计算浅层损失及梯度c_i（缺失深层坐标零填充），完整正式NLL梯度g_i仅在授权执行全深度时可得。目标G=Σ w_i g_i；若另有固定浅层辅助目标可精确另加，不把异构局部目标冒充完整NLL。边缘可在授权下执行深层计算；若数据/激活不允许移动则需要本地执行，未证真实权限和成本。内生草稿与正式共享模型，固定当前正式p，标准SD真实条件q、min(1,p/q)、正残差、拒绝截断、独立随机性及合法bonus/EOS/KV可保持理想随机序列；greedy正式argmax，实际完整实现、温度/top-p/有限精度未验证。本轮无新采样核。
T1 按键聚合：对键k，区域e报告 S_ek=Σ_{i∈e:k有效}n_i Δ_ik、N_ek=Σ n_i。目标为实际有此键的样本加权平均时，云必须用Σ S_ek/Σ N_ek，而非区域全样本量N_e加权区域均值。例区域1全样本100、键有效1、局部均值0；区域2全样本10、键有效10、局部均值1；普通区域全样本加权10/110=.09091，正确键加权10/11=.90909。加和对任意树拓扑结合，若无贡献保持旧键。此为分层移植可能错误，不是ReeFL原代码bug。对于固定一次相同学习率梯度步，若缺失键意味着本地浅层损失不依赖该键，则每键归一化是异构局部损失总和梯度的对角预条件 D_kk=Σ n_i/Σ_{i:k有效}n_i，不必是欧氏保守场；但不能据此声称优化完整正式NLL。多步本地训练、momentum等另审计。普通加权sum/count、平坦同样可做。
T2 信息反例：二元q=sigmoid(x), p=sigmoid(x+b k)，x=b=0，真实标签1，k∈{−1,+1}为不可见的深层特征/Jacobian。两世界p=q=.5、标准SD接受率1，浅层梯度−.5，完整深层梯度−k/2相反。如果允许观测只含q,p及接受轨迹和浅层梯度而不含k或其等价JVP，则不能同时恢复两世界g_b，任何估计最坏平方误差至少.25。若k可本地获得并计算，则该信息限制不适用。接受率只约束概率，不约束深层Jacobians或真实训练标签；高接受不能保证完整梯度可省。
T3 采用普通多保真修正：在同一固定样本和快照下，I_i~Bernoulli(pi_i)，条件于样本和所有采样前信息，pi_i>0且真实概率已知。定义 H_i=c_i+(I_i/pi_i)(g_i−c_i)。E H_i=g_i，条件平方误差 E||H_i−g_i||²=(1/pi_i−1)||g_i−c_i||²。各I独立时聚合迹方差Σ w_i² (1/pi_i−1)||g_i−c_i||²；相关采样需额外协方差。预算Σ pi_i C_i≤B，最优pi_i=clip(w_i||g_i−c_i||/sqrt(lambda C_i),lower,1)，从凸目标的KKT求出。等权等成本为按残差范数分配；是已有重要性抽样与多保真控制变量。若某区域无法合法取得g且pi=0、残差可能非零，无一般无偏保证；不能拿成功接受的样本替代缺失正式梯度。普通AR/全概率teacher forcing也可选择此修正。用当前实际SD接受结果决定是否执行完整前向存在因果循环；若已做正式前向，最多尝试省反向，不得重复计前向收益；历史代理须校准且保证正概率。
T/P 等计算预算反例：归一化完整梯度成本1，浅层控制量成本s，假设抽中时浅层工作可全部复用，CV成本s+pi(1−s)，直接均匀抽样成本b。相同b时pi=(b−s)/(1−s)，b>s；令k=E||g−c||²/E||g||²（等权固定数据、独立抽样），CV与直接的条件方差比 k b/(b−s)。因此CV胜出需s<b(1−k)。若浅层不能复用、通信/固定成本不同则需重新计算。此为固定数据采样误差模型而非优化达到同质量时间定理。

## 3. 实际公开文本计算 E/P/O
固定原文[Wallpaper](https://github.com/GITenberg/The-Yellow-Wallpaper_1952/blob/ce7c5bc1a5cf0fe23bcb265545a900365dbae29c/1952.txt) blob 3a2b5f55c6f1d13a7ae623282d38fde5a6ff352b、[Alice](https://github.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/blob/7f18748e114b994d5d9454d5793581bc0b9cf9d7/11.txt) blob 9d73b1d7447b7ddc158b1b157f5f700d21bd2fb7；实际GitHub取文本，与HCA/R06及本队R07同源。27字符，按Gutenberg marker截取、小写及空格清洗（可能含制作说明），Alice按中间空格拆2区、Wallpaper一区，各80%训练20%留出。132382训练字符，33097测试字符，33091有效预测位置。q为全局bigram+.5平滑，p为全局trigram+5*q平滑。将统计分布写为共享可训练加性头z_q=A_b=log q_b、z_p=A_b+B_s=log p_s。对真实标签y，g=(p−e_y,p−e_y)在A与B不同行块，c=(q−e_y,0)，故||g||²=2||p−e_y||²、||g−c||²=||p−q||²+||p−e_y||²。真实文本拟合统计模型及计算梯度矩，不是实际神经网络SGD、HFL或SD。固定留出数据、精确条件矩，不需MC或种子；三个区域不是三个独立训练seed。
各区独立比较同完整梯度比例b=.25/.5/.75、pi floor=.05：direct直接均匀抽完整梯度；uniform浅层控制量+均匀深层修正；entropy按q熵、delta按真实TV、oracle按真实残差范数选择pi并二分归一化期望预算。delta需完整p而oracle需g，均不可视为免费部署方法。表为每位置平均条件平方误差，非收敛或速度。
e N b overlap fullNorm resNorm direct uniform entropy delta oracle
0 6067 .25 .6418571794255902 1.431261820096685 .8388055399521882 4.293785460290055 2.5164166198565647 2.578956383520487 3.949355082039066 2.3024375272568514
0 6067 .5 .6418571794255902 1.431261820096685 .8388055399521882 1.431261820096685 .8388055399521882 .8700754217841498 1.5490112430616216 .7316844060561545
0 6067 .75 .6418571794255902 1.431261820096685 .8388055399521882 .4770872733655615 .27960184665072935 .30044843453874154 .6276177144746818 .20909446567320686
1 13512 .25 .6404651418333176 1.3866582940187417 .8203680016574099 4.1599748820562255 2.4611040049722295 2.508873030310283 3.8405133615911 2.2440945130426653
1 13512 .5 .6404651418333176 1.3866582940187417 .8203680016574099 1.3866582940187417 .8203680016574099 .8442525143264435 1.4991244777229857 .7116758216172188
1 13512 .75 .6404651418333176 1.3866582940187417 .8203680016574099 .4622194313395805 .2734560005524699 .2893790089984129 .5973793171001859 .20205643427818923
2 13512 .25 .6442548298493496 1.38083835170808 .8141494900608737 4.14251505512424 2.442448470182621 2.485078965025416 3.76674963511391 2.213560496528438
2 13512 .5 .6442548298493496 1.38083835170808 .8141494900608737 1.38083835170808 .8141494900608737 .8354647374822517 1.4674484049365206 .6995309944925273
2 13512 .75 .6442548298493496 1.38083835170808 .8141494900608737 .46027945056935987 .2713831633536245 .28559332830114037 .5820274897088197 .19612731622574534
uniform比direct同完整计算比例方差低约40.8–41.4%，但额外浅层成本未计；delta全部9组差于uniform，b=.5时甚至差于direct；entropy亦全部差于uniform。oracle在同完整比例改善但需完整残差。b=.5等计算成本下需s<约.204–.207才可能保住uniform方差优势（理想全复用）；未实测s。不可用这些条件矩预测收敛、真实接受率、GPU时间或CI。
## 4. 复现脚本（实际执行函数）
下载上述固定文本，保存wallpaper.txt alice.txt，提取为probe.js，node probe.js wallpaper.txt alice.txt。V8 deterministic，无依赖。
~~~javascript
function probe(docs){
const V=27,D=V*V,alphabet=" abcdefghijklmnopqrstuvwxyz",sum=a=>a.reduce((s,x)=>s+x,0),norm=a=>{let s=sum(a);return a.map(x=>x/s)};
const clean=d=>d.text.split(/\*\*\* START OF THIS PROJECT GUTENBERG EBOOK[^\n]*\n/)[1].split(/\*\*\* END OF THIS PROJECT GUTENBERG EBOOK/)[0].toLowerCase().replace(/[^a-z]+/g," ").trim();
let a=clean(docs[1]),w=clean(docs[0]),cut=a.lastIndexOf(" ",Math.floor(a.length/2)),regions=[w,a.slice(0,cut),a.slice(cut+1)].map(x=>Array.from(x,c=>alphabet.indexOf(c)));
let train=[],test=[],bi=new Float64Array(D),tri=new Float64Array(V*D);
for(let x of regions){let n=Math.floor(x.length*.8),t=x.slice(0,n);train.push(t);test.push(x.slice(n));for(let i=2;i<t.length;i++){let j=t[i-1]*V+t[i];bi[j]++;tri[t[i-2]*D+j]++}}
let q=Array.from({length:V},(_,b)=>norm(Array.from(bi.slice(b*V,(b+1)*V),x=>x+.5))),p=Array.from({length:D},(_,s)=>norm(Array.from(tri.slice(s*V,(s+1)*V),(x,j)=>x+5*q[s%V][j])));
let rows=[];for(let e=0;e<3;e++){let r=test[e],items=[],N=r.length-2;for(let i=2;i<r.length;i++){let P=p[r[i-2]*V+r[i-1]],Q=q[r[i-1]],y=r[i],g=sum(P.map((x,j)=>(x-Number(j===y))**2)),diff=sum(P.map((x,j)=>(x-Q[j])**2)),delta=.5*sum(P.map((x,j)=>Math.abs(x-Q[j]))),entropy=-sum(Q.map(x=>x*Math.log(x)));items.push({full:2*g,res:g+diff,delta,entropy});}
function calc(key,budget){let a=items.map(x=>key==="oracle"?Math.sqrt(x.res):key==="delta"?Math.max(x.delta,1e-8):Math.max(x.entropy,1e-8));let lo=0,hi=1;const mean=k=>sum(a.map(x=>Math.min(1,Math.max(.05,k*x))))/N;while(mean(hi)<budget)hi*=2;for(let j=0;j<50;j++){let m=(lo+hi)/2;if(mean(m)<budget)lo=m;else hi=m}let probs=a.map(x=>Math.min(1,Math.max(.05,hi*x)));return sum(items.map((x,i)=>(1/probs[i]-1)*x.res))/N}
for(let budget of [.25,.5,.75]){let full=sum(items.map(x=>x.full))/N,res=sum(items.map(x=>x.res))/N;rows.push({e,N,budget,overlap:1-sum(items.map(x=>x.delta))/N,fullNorm:full,resNorm:res,direct:(1/budget-1)*full,uniform:(1/budget-1)*res,entropy:calc("entropy",budget),delta:calc("delta",budget),oracle:calc("oracle",budget)})}}
return {train:train.map(x=>x.length),test:test.map(x=>x.length),rows}
}
const fs=require("node:fs");console.log(JSON.stringify(probe(process.argv.slice(2,4).map(p=>({text:fs.readFileSync(p,"utf8")}))),null,2));
~~~
## 5. 文献与成本审查
2026-11-24实际检索、核对原始摘要/页面、上述固定源码：
- [ReeFL ICML2024](https://proceedings.mlr.press/v235/lee24h.html) 共享早退头和异构FL。
- [LayerSkip ACL2024](https://aclanthology.org/2024.acl-long.681/) 早退损失、layer dropout、共享计算。
- [FedNova NeurIPS2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) 异构更新目标不一致。
- [FedRolex NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html) 滚动子模型。
- [FedPLT 2026](https://arxiv.org/abs/2605.02337) 部分层训练。
- [DELTA NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/949c57d30f8791e3ae42646081b3c102-Abstract-Conference.html) 最优无偏采样。
- [Optimal Client Sampling](https://arxiv.org/abs/2010.13723) 更新范数采样。
- [MLMC unbiased gradient AISTATS2021](https://proceedings.mlr.press/v130/shi21d.html) 多层无偏梯度。
- [Bias-Variance-Cost Tradeoff NeurIPS2021](https://proceedings.nips.cc/paper/2021/hash/b986700c627db479a4d9460b75de7222-Abstract.html) 多层成本/方差。
- [MFPG TMLR2026](https://xinjie-liu.github.io/mfpg-rl/) 多保真控制变量梯度。
未穷尽全文、最新异构层梯度估计/专利、未正式文字查重。直接最接近为ReeFL/LayerSkip+普通重要性采样及多保真。
P：每个参数组上传sample count 64位示意8B，L组额外8L B/区域，非必需下界（共享mask可压缩）；梯度本身D个FP32标量4D B。浅层+深层差分若分离上传可能重复共享坐标，优先边缘合并；若需要远端深层计算，BF16 d1536 hidden单位置3072B、反向需类似传输，另计版本/RTT/缓存。完整p验证前向已发生不能免费省；若本地没有合法深层计算权限，采样正概率不可实现。普通完整模型抽样、普通多保真均匀/范数代理、ReeFL/FedRolex/FedPLT、普通HFL+LayerSkip、同信息平坦均为强对照。未测实际计算、通信和wall time。
## 6. 裁决
CRITICAL：标准多保真与普通AR/平坦可复制；接受率不是深层梯度残差充分统计，真实文本按TV选预算全部劣于均匀修正；已有官方早退和异构聚合；无正式HFL↔SD双向不可替代原语。主观创新6.075/10（问题7、机制5、理论7、价值6、距离4.5，25/25/20/15/15%，实际加权5.975；以5.975为准），非同行评审。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL（限定数学，完整SD未实现）；G4 FAIL（真实文本统计模型，非两尺度预训练）；G5/G6未验证，无80%证据；G7局部可复算/整体未过。好结果同正式目标质量预算对最强基线真实全流程−15%或有效工作+20%，两预训练尺度三非IID/时延三独立训练种子CI及全部G。未执行1–2×40GB。下一轮先取得真实早退模型梯度/算子trace和合法层归属，1–2×40GB、两种LayerSkip或可训练早退预训练尺度、8客户端2边缘、≥10000位置记录p/q、真实标签、每块g/c及实际前向/反向时间、通信；测s与残差比，比较同预算普通抽样、历史梯度范数代理、ReeFL/LayerSkip及同信息平坦。若接受信息无法在昂贵计算前合法获得或无法超越普通多保真即停。不建议投入GPU挽救本机制。保留按键分层权重反例、接受率与深层梯度信息障碍、等成本门槛和真实文本负结果。Reject and Pivot。
