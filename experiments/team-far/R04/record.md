# team-far/R04 — TriSpec-ZO: learning-priority trivariate speculative coupling
2026-11-24. Start/prewrite 9dd9dd1a5e5db6c14397d7867dbe44a34dbe2b20, README blob ad7be2c7d09f180fd8657cb28e4d6191af561db3, 3181 lines. Full README fetched, all headings screened, targeted review A/R02/04/06/09/10/11, HMC, FSDC, FLPV, FSOT, FCRV, FQAC, FCIP, FAR R01-03. No exhaustive proof audit or independent rerun of others. User asks mechanism >=8 and all dimensions higher. No grade inflation.

## New problem and mechanism
Federated black-box reward optimization for stochastic autoregressive models on memory-limited clients. Fix a shared seeded direction u and perturbation eps before sampling, evaluate each region's J(theta+eps u), J(theta-eps u) via paired generated trajectories. Directional finite-difference estimate (R(Y+)-R(Y-))/(2eps) times u (for Gaussian directions, expected estimate is gradient of Gaussian-smoothed objective under integrability; Rademacher directional FD not generally exact gradient). Pair coupling can change variance while leaving mean of each expected reward and finite difference intact. We require dependence within an authorized +/- training pair, NOT independent rollout group GRPO or service samples. Region aggregates seed/scalar and learns a lightweight endogenous early-exit head on its own paired target distribution. Draft X~q shared across both target distributions p+,p- at shared prefix. New learning-priority coupling: simultaneous marginal correctness, maximal +/- token agreement, maximal achievable total draft agreement under that constraint. After pair diverges, separate histories and independent/legal SD or AR fallback. No separate draft LLM. Need actual common backbone/final-head perturbation and GPU prefix tree verifier to realize savings; full parameter ZO perturbation invalidates backbone/KV sharing, do not claim. Shared head learned using midpoint distillation plus overload penalty D (below), not executed. HFL formal update changes p+/p- and proposal/head adapts; verifier changes estimator covariance, but same-info delegated flat and paired AR+SD are strong alternatives.

## T: exact three-marginal frontier at one fixed prefix
Index 0 draft q, 1/2 targets. For each token v, t_v=min(p0,p1,p2); T=sum t_v. M_ab=sum_v min(p_a(v),p_b(v)) is pairwise maximum agreement. Define pair-only capacity c_ab(v)=min(p_a(v),p_b(v))-t_v, nonzero only for two largest coordinates (ties harmless). Define D=max(0,T+sum_ab sum_v c_ab(v)-1)=max(0,sum_v median(p0(v),p1(v),p2(v))-1). For any coupling, sum of the three pair agreements equals Pr(any equality)+2 Pr(all equal) <=1+2T, and also <=sum M_ab. Thus total pairwise deficit >=D. All three maxima simultaneously feasible iff D=0. Construct triple events (v,v,v) of mass t_v, pair events ab with matching token v and mass c'_ab(v)<=c_ab(v), reduce total pair mass by D, leave target pair 12 capacities intact. Feasible since M12<=1, hence draft capacities suffice. For missing variable j draw from its remaining marginal r_j(v)=p_j(v)-t_v-sum_{k!=j} c'_jk(v), normalized by total r_j. If D>0, all mass assigned to triple/pair events; missing variable cannot equal pair token because its full mass at that token was consumed by triple. If D=0, no-equality event N=1-T-sum c>=0, sample each residual independently; their supports are disjoint across variables, so no accidental equalities. Allocate residual proportionally across all events missing j; gives exact marginals. Consequently target agreement=M12 and sum draft agreements=M01+M02-D, sharp. For prioritized draft pairs reduce target pair first when feasible; generic weighted objective can allocate reductions by pair costs only within this construction, no claim full weighted optimum outside. q=(p++p-)/2 gives sum median=1, simultaneous maxima feasible, but computing it for drafting may require full target and defeat speed. Three categorical example q=(.5,.5,0), p+=(0,.5,.5), p-=(.5,0,.5): each pair cap .5, triple0, D=.5; insisting both draft maxima forces target agreement0, insisting target max .5 leaves draft sum .5. No such conflict for binary alphabet. Conditional sample after X drawn: choose event with probability proportional to its mass times X conditional probability; if X is a missing variable use its residual factor, then sample other missing variables. This is feasible via cumulative arrays O(V) preprocessing per prefix and finite event categories, not O(V^3) enumeration (audit enumerates). Need true p/q distributions including postprocessing and common-prefix only. Draft path generated causally in advance; verification uses only current candidate and joint history; unused future proposals discarded on either rejection for paired common block, no unproven bonus reuse. For full sequential marginal exactness, at each joint history conditional target marginal is the designated p_i(.|h_i), draft conditional q must be actual distribution after conditioning; independent draft RNG across positions and do not peek ahead. Greedy each target independently uses its true argmax; triple stochastic agreement formula not a greedy speed proof. Full multi-token implementation, EOS, numeric and top-p not tested.

## T: effect on learning
Given fixed direction, exact marginals guarantee expected contrast equals J(theta+eps u)-J(theta-eps u), irrespective of coupling. For deterministic common reward bounded [0,1] at a single token, E[(R(Y+)-R(Y-))^2]<=Pr(Y+ != Y-); maximal target coupling minimizes this upper bound, not necessarily actual contrast variance. For reward that is class-equivalent, reward-aware transport and ordinary paired AR can improve further. At multi-token divergence, coupling cannot guarantee reward equality from same prefix; no global variance theorem. Perturbation scaling 1/(4eps^2) amplifies reward differences, so small per-token mismatch matters. Pair dependence not permissible where objective assumes independent group samples. Formal training/serving separate. Ordinary maximal paired AR with same joint reward oracle can replicate target pairing, and can use this verifier if same resources. D is a mathematically sharp tradeoff, not proof of HFL necessity.

## E/P: actual external model moment audit
microGPT fixed aa28d7c0aa66704ce56b263dfd3270579f3d5f6d, weights df323b7e714ceeb10ca070f9945a82c51b2c233c, checkpoint 7f0b4968c56290ebad706ab740de942c369f42a5, input 736d88e18b70b292750a30e5cca6a92b0b333bd2. Forward derived from team-hca/R09 (inherits FEIA R11); two same 7264-parameter models, frozen backbone, Rademacher output-head direction, 4 logical clients grouped in 2 regions by initial char mod4, 16 sampled names/client, 4 teacher-forced prefixes/name=256 prefix instances/config, seeds11/29/73, eps .005/.02/.08, q shallow/center/midpoint. 54 configs (2x3x3x3), 13824 prefix-config instances with repeated context. Gold next character binary reward proxy, not real online black-box task. Exact enumeration of triple event mass and pair marginals, plus independent targets, standard independent verify given shared q, strongest conditional maximal target coupling given fixed standard q-target kernels, paired AR maximal and TriSpec. No actual parameter update, HFL communication, multi-token SD, full ZO training, GPU, timings or CI. Center uses full target unperturbed logits, midpoint uses both full perturbed target logits; neither is cheap baseline. Conditional agreement and FD variance per fixed context/direction, averaged; not global gradient variance. Proportional reduction of draft capacities, target pair full. Draft-priority illustrative reduces target capacities first; not optimal reward transport. Actual code node experiments/team-far/R04/probe.js weights.json checkpoint.json input.txt. 
[
  {
    "eps": 0.005,
    "qtype": "shallow",
    "overload": 0.0008812601877691132,
    "draft": 0.8915689034998194,
    "draftMax": 0.8920095335937042,
    "draftPriorityTarget": 0.9846269669868443,
    "targetMax": 0.9855082271746133,
    "ind": 2138.7449299386813,
    "shared": 423.2998531374969,
    "conditionalMax": 45.26509855966176,
    "ar": 37.8065738613326,
    "ours": 37.8065738613326,
    "indAgree": 0.16167474523312228,
    "sharedAgree": 0.8328182552352722,
    "conditionalAgree": 0.9829376831056545,
    "arAgree": 0.9855082271746135,
    "ourAgree": 0.9855082271746135
  },
  {
    "eps": 0.005,
    "qtype": "center",
    "overload": 7.546047120499111e-17,
    "draft": 0.9927539291065383,
    "draftMax": 0.9927539291065383,
    "draftPriorityTarget": 0.9855082271746135,
    "targetMax": 0.9855082271746135,
    "ind": 2138.7449299386813,
    "shared": 37.807506095488975,
    "conditionalMax": 37.80657386133263,
    "ar": 37.8065738613326,
    "ours": 37.8065738613324,
    "indAgree": 0.16167474523312228,
    "sharedAgree": 0.9855078582954433,
    "conditionalAgree": 0.9855082271746135,
    "arAgree": 0.9855082271746135,
    "ourAgree": 0.9855082271746135
  },
  {
    "eps": 0.005,
    "qtype": "midpoint",
    "overload": 6.404020832147712e-17,
    "draft": 0.9927541135873065,
    "draftMax": 0.9927541135873067,
    "draftPriorityTarget": 0.9855082271746135,
    "targetMax": 0.9855082271746135,
    "ind": 2138.7449299386813,
    "shared": 37.80657386133261,
    "conditionalMax": 37.80657386133261,
    "ar": 37.8065738613326,
    "ours": 37.80657386133243,
    "indAgree": 0.16167474523312228,
    "sharedAgree": 0.9855082271746135,
    "conditionalAgree": 0.9855082271746135,
    "arAgree": 0.9855082271746135,
    "ourAgree": 0.9855082271746135
  },
  {
    "eps": 0.02,
    "qtype": "shallow",
    "overload": 0.002852265291092194,
    "draft": 0.8869927039197162,
    "draftMax": 0.8884188365652624,
    "draftPriorityTarget": 0.9392472816603035,
    "targetMax": 0.9420995469513959,
    "ind": 133.5278788587507,
    "shared": 27.961275284583404,
    "conditionalMax": 10.856970897715371,
    "ar": 9.207282489948073,
    "ours": 9.207282489948069,
    "indAgree": 0.16093181503806583,
    "sharedAgree": 0.82264480350651,
    "conditionalAgree": 0.9331841956915422,
    "arAgree": 0.9420995469513959,
    "ourAgree": 0.9420995469513959
  },
  {
    "eps": 0.02,
    "qtype": "center",
    "overload": 4.640385298237959e-17,
    "draft": 0.9710366501626733,
    "draftMax": 0.9710366501626734,
    "draftPriorityTarget": 0.9420995469513959,
    "targetMax": 0.9420995469513959,
    "ind": 133.5278788587507,
    "shared": 9.211169963290482,
    "conditionalMax": 9.207282489948074,
    "ar": 9.207282489948073,
    "ours": 9.207282489948065,
    "indAgree": 0.16093181503806583,
    "sharedAgree": 0.9420734084777416,
    "conditionalAgree": 0.9420995469513959,
    "arAgree": 0.9420995469513959,
    "ourAgree": 0.9420995469513959
  },
  {
    "eps": 0.02,
    "qtype": "midpoint",
    "overload": 6.085988194885299e-17,
    "draft": 0.9710497734756981,
    "draftMax": 0.9710497734756981,
    "draftPriorityTarget": 0.9420995469513959,
    "targetMax": 0.9420995469513959,
    "ind": 133.5278788587507,
    "shared": 9.207282489948074,
    "conditionalMax": 9.207282489948074,
    "ar": 9.207282489948073,
    "ours": 9.207282489948064,
    "indAgree": 0.16093181503806583,
    "sharedAgree": 0.9420995469513959,
    "conditionalAgree": 0.9420995469513959,
    "arAgree": 0.9420995469513959,
    "ourAgree": 0.9420995469513959
  },
  {
    "eps": 0.08,
    "qtype": "shallow",
    "overload": 0.003407824605883959,
    "draft": 0.8405682100714599,
    "draftMax": 0.842272122374402,
    "draftPriorityTarget": 0.7691346335277155,
    "targetMax": 0.7725424581335996,
    "ind": 8.207111588175314,
    "shared": 2.602396741069615,
    "conditionalMax": 2.2025355445922794,
    "ar": 2.0352805351859726,
    "ours": 2.035280535185973,
    "indAgree": 0.14969456545974294,
    "sharedAgree": 0.7159806538073002,
    "conditionalAgree": 0.7597071673914626,
    "arAgree": 0.7725424581335996,
    "ourAgree": 0.7725424581335996
  },
  {
    "eps": 0.08,
    "qtype": "center",
    "overload": 1.1564823173178714e-17,
    "draft": 0.8854434304032665,
    "draftMax": 0.8854434304032667,
    "draftPriorityTarget": 0.7725424581335996,
    "targetMax": 0.7725424581335996,
    "ind": 8.207111588175314,
    "shared": 2.0503240672146554,
    "conditionalMax": 2.035280535185973,
    "ar": 2.0352805351859726,
    "ours": 2.0352805351859726,
    "indAgree": 0.14969456545974294,
    "sharedAgree": 0.7709882980088469,
    "conditionalAgree": 0.7725424581335996,
    "arAgree": 0.7725424581335996,
    "ourAgree": 0.7725424581335996
  },
  {
    "eps": 0.08,
    "qtype": "midpoint",
    "overload": 5.449922920360469e-17,
    "draft": 0.8862712290668001,
    "draftMax": 0.8862712290668,
    "draftPriorityTarget": 0.7725424581335996,
    "targetMax": 0.7725424581335996,
    "ind": 8.207111588175314,
    "shared": 2.0352805351859726,
    "conditionalMax": 2.035280535185973,
    "ar": 2.0352805351859726,
    "ours": 2.035280535185972,
    "indAgree": 0.14969456545974294,
    "sharedAgree": 0.7725424581335996,
    "conditionalAgree": 0.7725424581335996,
    "arAgree": 0.7725424581335996,
    "ourAgree": 0.7725424581335996
  }
]
Errors: max marginal 9.992007221626409e-16; max total-agreement bound error 3.1086244689504383e-15; max target-max error 1.5543122344752192e-15. 3-token stress {"M":[0.5,0.5,0.5],"D":0.5,"actual":[0,0.5,0.5],"err":0}.
Caveat: paired AR achieves same target agreement at no drafting cost, and actual reward contrast variance can differ from AR due to different residual transport. Need strong equal-info pairing baseline, real GPU and cost-sensitive comparison. User wants mechanism 8, but mathematical novelty only limited: multimarginal couplings, pairwise optimality and CRN already known; implementation of three-way frontier within ZO-SD potentially distinct.

## Literature search 2026-11-24
- Pairwise optimal coupling of multiple random variables Angel & Spinka arXiv:1903.00632 https://arxiv.org/abs/1903.00632 (related general incompatibility)
- FedKSeed ICML24 https://proceedings.mlr.press/v235/qin24a.html (seed-coded ZO FL)
- MeZO NeurIPS23 https://arxiv.org/abs/2305.17333; Sparse MeZO NeurIPS25 https://proceedings.neurips.cc/paper_files/paper/2025/hash/1e5c2efbddc02c1d971e2f19ccdb07d0-Abstract-Conference.html
- LLM Zeroth-Order Fine-Tuning is an Inference Workload May 2026 https://arxiv.org/abs/2605.28760 (shared inference runtime, important strong alternative)
- Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs ICML26 https://proceedings.mlr.press/v306/zhang26da.html
- Global Resolution ICLR26 https://proceedings.iclr.cc/paper_files/paper/2026/hash/83995c2acef585ae0f0d647154cfdd85-Abstract-Conference.html; List-Level Distribution Coupling NeurIPS25 https://proceedings.neurips.cc/paper_files/paper/2025/hash/088bafc6da5f934cf9acbb361a71bfe5-Abstract-Conference.html (multi-draft single target, differs from single draft/two target FD)
- SpecRoll Aug 2026 https://arxiv.org/abs/2608.04962; SPEC-RL https://arxiv.org/abs/2509.23232; Acceptance-Aware Draft Model Training Sep 2026 https://arxiv.org/abs/2609.24150.
Search results/abstracts, not exhaustive full text or proof-level novelty search. Prior repo A/R04 multi-client dependence, A/R06 residual transport, HMC multi-level max coupling, FSDC shared candidate variance, FAR R03 conditional marginalization: related but no exact same learning-priority 3-marginal frontier located in screened records. No formal plagiarism check.

## Gate and decision
Internal provisional rubric scores problem 8, mechanism 7, theory 8, value 7, literature difference 6 (weights 25/25/20/15/15 = 7.3). Mechanism NOT 8 yet; no G1>=8, no G2 proof; G3 partial theoretical, G4 fail, G5/G6 unverified, G7 partial. Score subjective and reflects stronger problem formulation not result that meets acceptance. No 80% probability. Retain as conditional candidate only, not top journal approval. Before GPU training, design actual joint decoder conditional X, multi-token exact paired distribution tests, measure whether true contrast variance and shared deep compute wins over paired AR+same shared backbone, independent production self-SD, FedKSeed+CRN and delegated flat; compare reward-aware optimal coupling. Train actual regional q with differentiable/hinge median overload and matching term, and require cross interventions q/learning. 1–2x40GB, two pretrained models, 3 heterogeneity and latency conditions, 3 independent training seeds, total -15% or effective work +20%, fixed quality and CI. If strongest AR/SD combination matches or gains only in artificial cases, reject; never claim observed variance as wall time. No full GPU experiment. 
