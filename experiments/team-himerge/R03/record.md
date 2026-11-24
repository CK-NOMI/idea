# team-himerge/R03 | JVT 联合目标—内生草稿验证梯度及质量障碍
2026-11-24；起始 main a64930b8caf002a13dd4d293661b54725371bd2d，README blob 4e929ec4c7d31767be44ccb761272b290875e31c。核对此前已读完整前缀及全部新增记录；读A/R14与HCA/R02脚本/结果摘要。上传前读并发HC/R02 TJSA、HieraCert/R03 FPQ-SD（脚本及结果）、HCA/R03 FSRA（坐标对齐及接受率虚高，区别于本轮直接联合目标训练反例）；最新已核查 main d27b72ada362b76c92c4800fe3731333a39d73e0。README 写入发生409冲突，先独立归档再追加索引。
**CRITICAL / Reject and Pivot**。T=限定数学；E=实际CPU概率矩枚举；P=代理；H=未验证假说。无真实LLM/HFL训练/GPU/实际网络与时间。不同于R01混合、R02隐私、A/R05控制变量、A/R09近似反向、A/R12选择偏差、A/R13奖励分解：本轮固定块起始前缀，对正式p和内生q同时求真实接受长度梯度。
## 数学机制
跨机构HFL：区域持非IID数据及授权轨迹、共享正式p_theta和内部轻量q_phi，可共享主干，无独立完整草稿模型。边缘/云聚合梯度，私有数据不上传，不保证DP。固定快照、同词表、真实条件q、独立随机数、标准min(1,p/q)及正残差、合法停止/bonus/回滚保持当前p随机序列；greedy由目标argmax。实际KV/EOS/top-p/有限精度未验证；新目标不等于旧模型无损。
拟目标 F=sum_i w_i E[NLL(p)+mu CE(q)−lambda J_K]，J_K为固定外生起始前缀下SD连续接受长度L的期望（不含bonus）。若前缀分布依赖p，另需其导数；跨块占据也未计，非全局轮数或wall time梯度。
固定h、正p/q、有限K、无p=q平局的可微点。M(u)=prod_t min(p(x_t|h,u_<t),q(x_t|h,u_<t))，Pr(L>=j,接受前缀=u)=M(u)，J_K=sum_{1<=|u|<=K} M(u)。令s_t=grad log p若p<q，grad log q若q<p；S_j=sum_{t<=j}s_t。grad J=sum_u M(u)S_|u|。真实轨迹 G_trace=sum_{j=1}^L S_j=sum_{t=1}^L(L−t+1)s_t 无偏，不能仅用最终score；L、系数、分支detach，共享参数链式求导。任意正支持完整路径分布B，G_B=sum_{j=1}^K M(X_1:j)/B(X_1:j) S_j，X~B，无偏；权重detach，不能仅autodiff比率而遗漏采样分布score。B=Q时G_Q=E[G_trace|完整草稿路径]，全协方差公式Cov(G_Q)≼Cov(G_trace)；B=P为普通AR目标轨迹版本，固定p仅训练q时与REAL path数学一致，REAL另有branch边缘化。联合p梯度为直接乘积求导，非新通用定理。
**质量反例：** 数据r=(.8,.2)，容量受限固定q=(.2,.8)，p=(t,1−t)，J=1−|t−.2|；F=−.8ln t−.2ln(1−t)−lambda J。lambda=0最优t=.8，接受.4；t=.2左右导数−3.75−lambda、−3.75+lambda，lambda>=3.75时凸目标全局最优t=.2、接受1，额外NLL=KL(r||q)=.6ln4≈.831777 nat/token。否定“提高接受率必改善正式学习”，非所有联合训练必退化。质量硬约束/冻结p必要；后者回归已有草稿训练。
## 实际CPU实验
JS V8，seed11/29/73，sigma=.2/.7/1.4，150公共logit树/seed/sigma，每树4区域独立p/q高斯扰动，深4二元树；1350组、5400区域树。两个共享标量logit bias，score=x−p或x−q。精确枚举拒绝/接受与完整p/q路径梯度矩，非MC、训练或真实时间；后两路径不是完整REAL/EDR实现。均值：
|sigma|E[L]|trace方差迹|q路径方差迹|p路径方差迹|
|---|---|---|---|---|
|.2|3.555627|4.017026|3.397861|3.405314|
|.7|2.675033|3.304450|2.134072|2.148930|
|1.4|1.939003|2.248255|1.233447|1.240971|
最大均值误差1.34e−15；trace 0/5400次方差最小，q/p互有胜负。q平均方差降15.4/35.4/45.1%；同MSE独立样本线性成本时trace每样本成本须低于q的84.6/64.6/54.9%才有方差×成本优势（P）。普通SD已批验证完整候选，概率可能已存在，仍需反向激活保留/重算。无实际时间/CI。
逐种子实际输出（sigma seed E[L] 三方差 梯度均值范数平方 最大误差 最低方差次数）：
```text
.2 11 3.5600168901103872 [3.9831444667822997,3.3837369140926468,3.378538980563097] 1.8043175376864455 1.1102230246251565e-15 [0,294,306]
.2 29 3.5599636525625624 [4.0354963793562195,3.4183723444962832,3.437905115267792] 1.72429005614489 1.3322676295501878e-15 [0,308,292]
.2 73 3.546900402140292 [4.032437640969386,3.391474553814916,3.3994978780140332] 1.8786683101971686 1.3322676295501878e-15 [0,319,281]
.7 11 2.6855711187872027 [3.2875861347326287,2.154085143235394,2.130969245986858] 1.1622922567383565 8.881784197001252e-16 [0,285,315]
.7 29 2.6862555205223155 [3.308978612738554,2.134371574872856,2.1800604051485575] 1.1248856944892471 1.1102230246251565e-15 [0,306,294]
.7 73 2.6532722624501077 [3.3167843291353023,2.1137593115194435,2.135761730839955] 1.2022650160550596 8.881784197001252e-16 [0,312,288]
1.4 11 1.955785712488853 [2.234876944313307,1.2526840392468876,1.211341990480639] .6625793881382355 8.881784197001252e-16 [0,278,322]
1.4 29 1.9478319865767328 [2.2642736764925804,1.2220962323850104,1.281812879534293] .6454875973944183 8.881784197001252e-16 [0,322,278]
1.4 73 1.9133921040301536 [2.2456143456544235,1.2255603954627419,1.2297575563882084] .665426507231188 7.771561172376096e-16 [0,292,308]
```
实际执行代码复制为 jvt_probe.js，node jvt_probe.js；LCG仅复算：
```javascript
function experiment(){
const seeds=[11,29,73], sigmas=[.2,.7,1.4], N=150,K=4,R=4;
let state; function u(){state=(Math.imul(1664525,state)+1013904223)>>>0;return (state+.5)/4294967296}
function normal(){return Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u())}
const sigmoid=x=>1/(1+Math.exp(-x)), add=(a,b)=>a.map((x,i)=>x+b[i]), scale=(a,s)=>a.map(x=>x*s), norm=a=>a.reduce((s,x)=>s+x*x,0);
function evalTree(tree){
 let truth=[0,0], obj=0, second=[0,0,0],means=Array.from({length:3},()=>[0,0]);
 function moment(i,g,w){means[i]=add(means[i],scale(g,w));second[i]+=w*norm(g)}
 function go(h,P,Q,W,S,T,Gp,Gq){
  if(h.length===K){moment(0,T,W);moment(1,Gq,Q);moment(2,Gp,P);return}
  let {p,q}=tree[h];
  for(let x=0;x<2;x++){let a=x?p:1-p,b=x?q:1-q,m=Math.min(a,b),z=W*m;
   let s=add(S,a<b?[x-p,0]:[0,x-q]);
   obj+=z;truth=add(truth,scale(s,z));
   moment(0,T,W*(b-m));
   go(h+x,P*a,Q*b,z,s,add(T,s),add(Gp,scale(s,z/(P*a))),add(Gq,scale(s,z/(Q*b))));
  }
 }
 go("",1,1,1,[0,0],[0,0],[0,0],[0,0]);
 return {obj,truth,variance:second.map((v,i)=>v-norm(means[i])),error:Math.max(...means.flatMap(m=>m.map((x,i)=>Math.abs(x-truth[i]))))};
}
const rows=[];
for(let sig of sigmas)for(let seed of seeds){state=seed;let v=[0,0,0],obj=0,err=0,grad2=0,wins=[0,0,0];for(let t=0;t<N;t++){
 const base={};for(let d=0;d<K;d++)for(let j=0;j<2**d;j++)base[d?j.toString(2).padStart(d,"0"):""]=normal();
 for(let r=0;r<R;r++){let tree={};for(let h in base)tree[h]={p:sigmoid(base[h]+sig*normal()),q:sigmoid(base[h]+sig*normal())};
 let z=evalTree(tree);obj+=z.obj;grad2+=norm(z.truth);err=Math.max(err,z.error);v=add(v,z.variance);wins[z.variance.indexOf(Math.min(...z.variance))]++}
}
rows.push({sig,seed,obj:obj/(N*R),variance:v.map(x=>x/(N*R)),grad2:grad2/(N*R),error:err,wins})}
return {seeds,sigmas,N,K,R,rows};
}
console.log(JSON.stringify(experiment(),null,2));
```
## 原始文献与否决
2026-11-24检索 speculative decoding expected acceptance length gradient training on policy; joint target draft optimization self-speculative; likelihood ratio gradient; federated policy gradient; online verification feedback。原始摘要、REAL/EDR部分正文，盲区未索引/未公开论文及全部附录。无正式文字查重。
- [REAL Loss](https://arxiv.org/abs/2611.14399) 2026-11-10：真实EAL路径权重、固定p的q无偏梯度、branch Rao–Blackwell，直接近邻。
- [EDR](https://arxiv.org/abs/2610.10411) 2026-10-07：跨轮MRP、TD梯度、局部EAL最优可非全局最优。
- [Acceptance-Aware Draft Model Training](https://arxiv.org/abs/2609.24150) 2026-09-21：greedy EAL、sampling WTV、GRPO。
- [VSD](https://arxiv.org/abs/2602.05774) 2026-02-05：路径接受变分、EM。
- [When Drafts Evolve](https://proceedings.mlr.press/v306/qian26k.html) ICML2026：在线验证反馈；[Online SD](https://proceedings.mlr.press/v235/liu24y.html) ICML2024：在线蒸馏。
- [LayerSkip](https://aclanthology.org/2024.acl-long.681/) ACL2024：联合早退训练、自投机；[HierFAVG](https://arxiv.org/abs/1905.06641) 2019、[Leviathan SD](https://proceedings.mlr.press/v202/leviathan23a.html) ICML2023。
CRITICAL：同信息平坦、本地学习、普通AR目标路径G_P可替代；冻结p变REAL/在线SD，联合p有质量障碍；J_K非全局轮数（EDR）；无HFL双向必要性。MAJOR：推理无梯度与反向激活重算、共享主干、快照漂移、训练许可、通信/截止偏差。候选ID+两FP32概率12B/位置，V128000,K4时相对于全量FP32 teacher仅消息2.048MB vs48B；本地训练本来不上传完整概率，须与只传梯度/标量/共置计算比较，非实测通信收益。
若新证据改变前提才执行1–2×40GB Qwen2.5-1.5B/3B、8逻辑客户端2边缘、公开数学/代码/通用语料、三异质/时延、三种子，LayerSkip或内生头+LoRA；测真实梯度、反向激活、质量退化；对照HFL AR+teacher forcing、普通HFL+SD、REAL path/branch、EDR、在线蒸馏、固定p、同信息平坦/委托；质量预算匹配，计NLL/任务、实际轮数、TPOT/P95、吞吐、训练推理总时间、通信、GPU和CI。单卡模拟非真实分布式。未执行。好结果为同质量预算最强公平基线真实总时间−15%或有效工作+20%，且其余G门槛；若可消除部分不足总时间15%，单独消除不能达到时间门槛。
主观创新度5.95/10（问题7×25%、机制5.5×25%、理论6.5×20%、价值6.5×15%、文献距离4×15%），无80%成功把握证据。G1 FAIL；G2 CRITICAL FAIL；G3 PARTIAL/FAIL（固定快照标准SD及梯度T，完整greedy/随机/精度未验）；G4 FAIL（CPU合成）；G5 FAIL（无实际时间）；G6 FAIL；G7 PARTIAL/FAIL（内联代码及逐种子原始结果，非完整系统）。
**Reject and Pivot。** 保留联合梯度、条件方差界、质量反例、实验与强对照。禁止仅追加接受损失、联邦REAL/EDR或反馈训练换名。下一轮先获取真实模型trace，定位训练与验证共同重复算子及实际时间份额、区域驻留/异步约束；先否证普通AR/SD和平坦再创造机制。
