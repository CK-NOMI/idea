# team-far/R19 — FLDN: learned multi-rank density states, aggregation and correction audit
2026-11-24; start 991c76676aa2570585ded1c33a5bcb43bffc213e; prewrite e8e31f0c0c9bd3f35e3906306c0a272a3b1a5dee; README blob c511797841ea8140487e050c060ab920b0103ec0; 3312 lines, 449362 chars. Full README fetched, R18 record reread, density/Born/spherical terms searched, relevant earlier compositional representation checked. No independent audit of all history. T=conditional proof; E=actual microGPT head training and spectral checks; P=proxy; H=untested. First attempt at larger 36-configuration experiment timed out before producing a verifiable result or writing files; no figures from that attempt claimed. Bounded completed experiment below. Code and results first committed at e8e31f0c0c9bd3f35e3906306c0a272a3b1a5dee.
## Formal architecture
Common fixed W in R^(V x r), G=W^T W, K learned r x r matrices B_k, frozen backbone feature h with affine bias; F=[B_1 h,...,B_K h], A=FF^T/tr(GFF^T), p_i=(1-eps)w_i^T A w_i+eps/V, eps=.01. Shallow feature s same W,B. Output-space rho=WAW^T PSD trace1. New formal target, not original softmax exact conversion. For same shared dictionary and weights alpha_e, normalized A_global=sum alpha_e A_e gives exactly ordinary weighted probability ensemble. Raw factors are not average-closed, rank may grow to min(r,sum ranks). Per-request density aggregation not FedAvg; same-info flat or transmitted factor models can reproduce. No DP. For amplitudes z_ki=w_i^TB_k h, Z=sum z², teacher t_i, c_i=-(1-eps)t_i/p_i, teacher CE gradient dL/dz_ki=2z_ki/Z*(c_i-sum_j c_j sum_l z_lj²/Z). Exact analytical gradient used with Adam; no new spherical loss, dense teacher and output computation used in actual experiment.
## T: positive spectral envelope and hierarchy
For normalized formal rho and shallow sigma, Delta=rho-sigma, positive and negative spectral parts Delta_+,Delta_- PSD, same trace; [p_i-q_i]_+ <= (1-eps)(Delta_+)_{ii}, total envelope M=(1-eps)||Delta||_1/2<=1-eps. In whitened r-dimensional coordinates measurement effects E_i=G^{-1/2} w_i w_i^T G^{-1/2} sum to I (SPD or quotient support); use tr(E_i Delta_+) there. rank(Delta_+)<=rank(rho) since ker rho has nonpositive quadratic form; similarly negative part. Select eigenvector with weight eigenvalue, sample squared amplitude via R18 tree, accept ratio actual positive residual/envelope, repeat. Standard maximal coupling gives expected correction proposals M/TV conditional on rejection, unconditional M when TV>0; no correction if TV=0. Exact real arithmetic for defined target with same measurement and smoothing, causal application at actual prefixes, independent RNG, proper EOS/bonus; production top-p/temperature/precision and full sequence not implemented. Greedy dense target argmax fallback. No R19 actual SD samples (R18 samples not reused as evidence). For same alpha, M_global <= sum alpha M_e by trace norm triangle inequality, opposite signed residuals may cancel. Not new quantum trace norm theory. Ordinary AR and probability mixtures may obtain related benefits; no hierarchy-specific necessity.
## T: compression nonclosure
For trace1 PSD state with sorted eigenvalues lambda, normalized top-K spectral truncation attains minimum trace distance among rank<=K states, tau=1-sum_{j<=K}lambda_j; any rank-K support P gives test I-P with mass >=tau (Ky Fan); truncation attains bound. Known result, see Ezzell et al arXiv:2203.00811. Measurement TV <=(1-eps)tau; not necessarily optimal under this particular measurement. Counterexample with equal regional densities diag(.51,.49,0) and diag(0,.49,.51): local rank1 truncation retains e1/e3 only; uncompressed global state diag(.255,.49,.255) has largest eigenvalue on e2; locally compressed mixture and further rank1 truncation cannot recover e2. Coordinate measurement TV between disjoint pure outputs 1 (1-eps after smoothing). Rank reduction not associative, must charge each lossy step. This is classical spectral theory, not a new foundational theorem.
## E: completed bounded experiment
Source {"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}],"prior":"experiments/team-far/R18/probe.js"}. External two same-scale 7264-param microGPT checkpoints (2-layer d16, V27), frozen backbone, possible upstream pretraining overlap. Four nonIID first-letter groups, three seed splits 11/29/73, 16 names/group, 2 positions/name (128 dictionary training contexts per split); last 2 names/group, two positions (16 fixed heldout contexts per checkpoint). r=17 includes bias, K=1/2/4, 18 completed configs. Fit common W via R18 ridge sqrt teacher; learn B_k with 48 pooled stratified minibatch Adam steps then four separate local heads each 8 local steps, batch8; 80 steps/config, 1440 optimizer updates, 11520 sampled example uses; 288 eval-context uses across configurations, not 288 independent contexts. Same fixed initial feature model for all; B1 identity, others small random. All inference outputs computed dense for audit. Actual original teacher, single-rank and multi-rank trained models, local factor average, exact equal-probability ensemble, global normalized top-K compression vs local rank floor(K/2) (at least1) then top-K global compression. Original softmax has 432 output params and gives strong quality reference with fewer total head params than new W+B (748/1037/1615), not equal training budget nor retrained baseline. No exact equal-parameter control, no actual physical HFL, multi-round FedAvg, GPU, wall time, CI, real two pretrained scales, or multi-token SD. Three data splits not independent pretrained seeds. Teacher KL is lower as K grows, heldout label NLL does not improve; no quality superiority. Max envelope violation 0; mix error 8.326672684688674e-17; tail-bound violation 0; cancellation-bound violation 0. Spectra calculated on V=27 output-space matrices via Jacobi for correctness, not scalable implementation. Reproduce: node experiments/team-far/R19/probe.js weights.json checkpoint.json input.txt.
[
  {
    "K": 1,
    "n": 6,
    "trainable": 289,
    "dictionary": 459,
    "softParams": 432,
    "origNll": 2.8138150271278444,
    "initNll": 2.8672613509628238,
    "nll": 2.8621314950375534,
    "kl": 0.0272025008424725,
    "accept": 0.9234951054588642,
    "env": 0.10796717450682224,
    "residual": 0.07650489454113588,
    "envRank": 1,
    "ensembleNll": 2.85439105126169,
    "avgNll": 2.8565303182204436,
    "ensembleAccept": 0.9238483137636079,
    "mergeMass": 0.10751324840426386,
    "localMass": 0.10798334574637458,
    "tail": 0.001913073295797427,
    "compressTV": 0.0010040629319699084,
    "compressNll": 2.857193180273702,
    "preTV": 0.0010040629319699585
  },
  {
    "K": 2,
    "n": 6,
    "trainable": 578,
    "dictionary": 459,
    "softParams": 432,
    "origNll": 2.8138150271278444,
    "initNll": 2.856054258530126,
    "nll": 2.8633246720130887,
    "kl": 0.023274282352929054,
    "accept": 0.9240950154137781,
    "env": 0.11328023023482858,
    "residual": 0.07590498458622184,
    "envRank": 2,
    "ensembleNll": 2.862752580979326,
    "avgNll": 2.8708693386369597,
    "ensembleAccept": 0.9244409027422739,
    "mergeMass": 0.11288979540444677,
    "localMass": 0.11382455389819164,
    "tail": 0.004715365307332249,
    "compressTV": 0.002459674134839741,
    "compressNll": 2.8696141123968766,
    "preTV": 0.012728947184494412
  },
  {
    "K": 4,
    "n": 6,
    "trainable": 1156,
    "dictionary": 459,
    "softParams": 432,
    "origNll": 2.8138150271278444,
    "initNll": 2.833314480040325,
    "nll": 2.865558136525946,
    "kl": 0.022248605635367732,
    "accept": 0.9249015984953486,
    "env": 0.11713889584844167,
    "residual": 0.07509840150465145,
    "envRank": 4,
    "ensembleNll": 2.853593134294966,
    "avgNll": 2.879113757999777,
    "ensembleAccept": 0.9244269076629559,
    "mergeMass": 0.1169173656347261,
    "localMass": 0.11880084431527965,
    "tail": 0.00664612741028818,
    "compressTV": 0.0030489032188604436,
    "compressNll": 2.864808543993527,
    "preTV": 0.010121947428928397
  }
]
## P and strong baselines
State factor rank K and latent width r distinct. At r64,K4 factors 256 floats vs symmetric density 2080 floats, FP32 1024B/8320B per request, four regional factors 4096B; no metadata/downlink/dictionary/feature traffic included. Shipping model factors once may be better. Regional rank growth; thin QR/eigen O(rK²+K³) for two rank-K states, full r-dimensional eig O(r³), no actual timing. Compare ordinary AR+same probability mixture, plain HFL+SD, same information flat, MoS/Mixtape/FedDF and spherical loss. No G5 evidence.
## literature actually searched 2026-11-24 (primary abstracts/venue pages; not exhaustive)
Expressive power of tensor-network factorizations for probabilistic modeling, NeurIPS2019 https://proceedings.neurips.cc/paper/2019/hash/b86e8d03fe992d1b0e19656875ee557c-Abstract.html (Born machines and locally purified states); Latent Space Purification via Neural Density Operators PRL2018 https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.240503; The quantum low-rank approximation problem https://arxiv.org/abs/2203.00811 (direct prior trace-distance rank theorem); Quantum-inspired semantic matching based on neural networks with the duality of density matrices https://www.sciencedirect.com/science/article/pii/S0952197624018256; Breaking the Softmax Bottleneck ICLR2018 https://arxiv.org/abs/1711.03953; Mixtape NeurIPS2019 https://papers.nips.cc/paper/2019/hash/512fc3c5227f637e41437c999a2d3169-Abstract.html; FedDF NeurIPS2020 https://proceedings.neurips.cc/paper_files/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html; Exact gradient updates spherical family https://arxiv.org/abs/1606.08061. No claim first on trace distance, density mixture, low-rank approximation or foundational spectral theory. No formal text similarity test.
## Verdict
Subjective weighted innovation 7.000/10 (8/7/8/6.5/4.5,25/25/20/15/15), lower due to direct prior art and weak quality/aggregation effects. G1/G2/G4 fail, G3/G7 partial, G5/G6 unverified/no80% evidence. Reject standalone. Retain positive envelope rank generalization and compression counterexample, actual trained head negative results. Do not keep increasing rank without matched-quality evidence. Next pivot requires an explicit genuine non-flat information/compute constraint AND actual SD-dependent formal learning, preregister causal 2x2 ordinary HFL+SD/AR/flat comparisons before constructing a large system. If resources available 1–2x40GB and 1.5B/3B two pretrained scales, three heterogeneity/delay regimes, three independent training seeds, CI, real total time -15% or effective work +20% at matched quality/budget. No GPU executed.