# team-far/R16 — FIDA-SD: fixed-acceptance inclusion design and reward residual frontier
2026-11-24; start/prewrite ef2552f1917a3c5fe1a8ab8fccc32f90ffce3858, README blob 3e065cb3bc12c332a3bd94d1594cc253a58f92b0; 3285 lines, 445172 chars. Full README retrieved, R15 and A/R05/06, R09, HMC-SD, HC/R09, R13 reexamined. Not a complete line-by-line proof audit or independent reproduction. New object is diagonal mass allocation of a coupling under fixed total matching, not A/R06 residual transport with fixed maximal diagonal, and centered counterfactual rewards not ordinary raw HT. T=conditional math; E=external checkpoint forward + deterministic numeric audit; O=oracle; P=proxy; H=untested hypothesis.
## Formal problem
Fixed legitimate state and model snapshot, terminal action environment independent resettable sandbox, p formal distribution, q internal shallow predictor, X~q,Y~p, joint matrix Gamma rows q cols p. Sample X then Y by Gamma row / q_x; choose coupling before environment reward, no outcome-dependent selection. Commit only Y. Distinct action set U={X,Y}; use one independent R_i per included action; mu_i=E R_i, sigma_i²=Var R_i. Score s_i=grad log p_i. Deterministic pre-sample reward prediction m_i (possibly region-specific, calibrated on prior data), stop-gradient. G= sum_i p_i m_i s_i + sum_{i in U} p_i/pi_i (R_i-m_i)s_i, pi_i=p_i+q_i-Gamma_ii. For any m, expectation grad J by inclusion. Not general PPO/GRPO or multistep guarantee; nonterminal requires unbiased target continuation, extra cost. Formal action marginal p, correct outcome kernel given independent environment; for full autoregressive sequence must apply each conditionally at actual causal history with fresh independent randomness, discard wrong suffix, correct bonus/EOS/processed probabilities, no dependence on future reward. This is real arithmetic single-action result; no production sampler/greedy/top-p numerical tests. Greedy uses target argmax; learned coupling is for stochastic sampling only.
## T1: exact feasible diagonal polytope
Fix desired token agreement D=trace Gamma, T=1-D. Write d_i=Gamma_ii, a_i=q_i-d_i, b_i=p_i-d_i. Nonnegative diag with sum d_i=D and d_i<=min(p_i,q_i). Residual transport between distinct indices exists iff a_i+b_i<=T for all i: necessity by capacity of columns excluding i; sufficiency Hall/max-flow for complete bipartite graph minus diagonal, singleton row subsets have neighbor mass T-b_i, row subsets of size >=2 see all columns, zero supply rows harmless. For vocab>=2. Thus feasible iff max(0,(p_i+q_i-T)/2)<=d_i<=min(p_i,q_i), sum d_i=D. Equivalently inclusion bounds l_i=max(p_i,q_i), u_i=p_i+q_i-max(0,(p_i+q_i-T)/2), sum pi_i=2-D. Restrict to feasible D, e.g. convex mixture of maximal and independent couplings. This is not arbitrary coupling lower bound, is exact two-draw marginals characterization.
## T2: sharp centered variance frontier
If m=mu (oracle), independent reward noise for distinct actions and conditional zero means give zero cross term and exact trace variance V=sum_i p_i² sigma_i² ||s_i||²/pi_i. Same p,q,D and any residual coupling with fixed diag have identical variance. Thus A/R06 style residual rematching cannot improve this centered oracle criterion at maximal D. For strictly positive weights w_i=p_i² sigma_i²||s_i||², minimize separable convex sum w_i/pi_i subject to bounds and sum. KKT gives pi_i*=clip(sqrt(w_i/lambda),l_i,u_i) with scalar lambda chosen by monotone bisection; endpoints and zero weights handled by feasible allocation of remaining mass (experiment positive). Off-diagonal transport via max-flow realizes exact chosen inclusions. This is global optimum within fixed p,q,D, one reward per distinct action, oracle-centered HT estimator; not globally optimal RL or unrestricted estimator. At maximal D all diagonals min(p,q), no freedom. At fixed D, expected distinct tool count 2-D is fixed; hence strict variance improvement is same proxy tool budget. If m inaccurate, residual means e_i=mu_i-m_i produce additional variance sum_i p_i² e_i²||s_i||²/pi_i + sum_{i!=j}(Gamma_ij+Gamma_ji)(p_i p_j/pi_i pi_j)e_i e_j <s_i,s_j> - ||sum_i p_i e_i s_i||². Oracle optimum not guaranteed. Coefficients use m for Bernoulli sigma estimate; for other rewards need separate variance model. Bias unaffected by incorrect m if fixed before inclusion. Region-specific variance estimates can be aggregated when shared p/q/state and weighted sum of conditional variances is goal (use sum of region weights times w_i); minimizing weighted local variance not generally variance of correlated global gradients. Identical-information flat can reproduce. No proof of hierarchy necessity or of learning-mediated draft optimization; no DP. This is standard Hall and convex allocation math applied to coupling, not new basic theory.
## E/O: actual test
Source {"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}],"forwardSource":"experiments/team-far/R15/probe.js"}. Two same-scale externally trained microGPT checkpoints (7264 params, d16, 2 layers, 4 heads, 27 character vocab). Source training history unverified, possible upstream train overlap. Four first-letter groups, last16 each, 128 fixed contexts across checkpoints, shallow/deep distributions with actual model forwards. Three synthetic independent Bernoulli reward profiles same R15: true char .85 else .05; dense .1+.8(1-|i-y|/26); constant .5. For each case three t=.25,.5,1 values interpolating maximal D to independent D; cost-matched baseline Gamma=(1-t)maximal+t independent(p,q). Three reward calibration conditions m=.5+calibration*(mu-.5), calibration1 oracle, .5 and 0 artificial misspecification, not learned prediction. Allocator uses m(1-m), logit score norm 1-2p_i+sum p². Actual Dinic residual max-flow constructs matrix, exact finite-sum conditional moments (no simulated tool calls, no actual generation or training), evaluate both designs with SAME m, compare maximal SD and independent p/p with same m at own distinct cost via V*C. 1152 context/reward/budget cases, 3456 allocation/transport audits. Error flow 1.7763568394002505e-15, bias 1.3877787807814457e-16, closed form 5.551115123125783e-17, sum 1.1102230246251565e-15, oracle violation 0. Reproduce node experiments/team-far/R16/probe.js weights.json checkpoint.json input.txt. No GPU, physical HFL, realistic rewards, latency, CI, independent training seeds or two scales. V*C is variance times expected tools only, not latency or convergence. Results:
[
  {
    "type": "sparse",
    "t": 0.25,
    "n": 128,
    "D": 0.7127497028843044,
    "cost": 1.2872502971156963,
    "c1": {
      "v": 0.036144767801545394,
      "base": 0.03745895007086111,
      "sd": 0.04423887271813859,
      "pp": 0.025652605248374243,
      "cv": 0.04647496017412853,
      "basecv": 0.048172952673367526,
      "sdcv": 0.048385427823718415,
      "ppcv": 0.04751852608984755,
      "winMix": 1,
      "winPP": 0.9140625
    },
    "c0.5": {
      "v": 0.0652528628161672,
      "base": 0.06593854457971941,
      "sd": 0.07895896636751062,
      "pp": 0.04363148981551301,
      "cv": 0.08398332519718692,
      "basecv": 0.08488155044484202,
      "sdcv": 0.08639855202342868,
      "ppcv": 0.08099643928673256,
      "winMix": 0.8046875,
      "winPP": 0.0546875
    },
    "c0": {
      "v": 0.1496884687632377,
      "base": 0.1513773281062943,
      "sd": 0.1831192473156267,
      "pp": 0.09756814351692912,
      "cv": 0.19281420674564137,
      "basecv": 0.19500734375926537,
      "sdcv": 0.20043792462255944,
      "ppcv": 0.18143017887738738,
      "winMix": 0.6640625,
      "winPP": 0.0703125
    }
  },
  {
    "type": "sparse",
    "t": 0.5,
    "n": 128,
    "D": 0.5217161398276186,
    "cost": 1.478283860172382,
    "c1": {
      "v": 0.03134948286277809,
      "base": 0.03251947803984049,
      "sd": 0.04423887271813859,
      "pp": 0.025652605248374243,
      "cv": 0.04632835093141751,
      "basecv": 0.04806104372698586,
      "sdcv": 0.048385427823718415,
      "ppcv": 0.04751852608984755,
      "winMix": 1,
      "winPP": 0.9453125
    },
    "c0.5": {
      "v": 0.056004125982669566,
      "base": 0.05655358303210626,
      "sd": 0.07895896636751062,
      "pp": 0.04363148981551301,
      "cv": 0.08286946049367455,
      "basecv": 0.08369785550484399,
      "sdcv": 0.08639855202342868,
      "ppcv": 0.08099643928673256,
      "winMix": 0.703125,
      "winPP": 0.0703125
    },
    "c0": {
      "v": 0.12727117203425706,
      "base": 0.12865589800890362,
      "sd": 0.1831192473156267,
      "pp": 0.09756814351692912,
      "cv": 0.18854138058157216,
      "basecv": 0.1906082908384182,
      "sdcv": 0.20043792462255944,
      "ppcv": 0.18143017887738738,
      "winMix": 0.6171875,
      "winPP": 0.109375
    }
  },
  {
    "type": "sparse",
    "t": 1,
    "n": 128,
    "D": 0.13964901371424812,
    "cost": 1.8603509862857515,
    "c1": {
      "v": 0.025358508849835162,
      "base": 0.025778504115661575,
      "sd": 0.04423887271813859,
      "pp": 0.025652605248374243,
      "cv": 0.047195522895904236,
      "basecv": 0.047971687230588485,
      "sdcv": 0.048385427823718415,
      "ppcv": 0.04751852608984755,
      "winMix": 1,
      "winPP": 0.6875
    },
    "c0.5": {
      "v": 0.043747440673911536,
      "base": 0.04388741561521813,
      "sd": 0.07895896636751062,
      "pp": 0.04363148981551301,
      "cv": 0.08156078706743985,
      "basecv": 0.08183849747637138,
      "sdcv": 0.08639855202342868,
      "ppcv": 0.08099643928673256,
      "winMix": 0.6328125,
      "winPP": 0.421875
    },
    "c0": {
      "v": 0.09800943285733973,
      "base": 0.09821415011388779,
      "sd": 0.1831192473156267,
      "pp": 0.09756814351692912,
      "cv": 0.18300830027515827,
      "basecv": 0.18343892821372,
      "sdcv": 0.20043792462255944,
      "ppcv": 0.18143017887738738,
      "winMix": 0.640625,
      "winPP": 0.4609375
    }
  },
  {
    "type": "dense",
    "t": 0.25,
    "n": 128,
    "D": 0.7127497028843044,
    "cost": 1.2872502971156963,
    "c1": {
      "v": 0.12096111838897627,
      "base": 0.12530578761554095,
      "sd": 0.14836825609913834,
      "pp": 0.08537604848262671,
      "cv": 0.15579935592969602,
      "basecv": 0.16140268645184414,
      "sdcv": 0.16242391980288912,
      "ppcv": 0.15861625644849792,
      "winMix": 1,
      "winPP": 0.921875
    },
    "c0.5": {
      "v": 0.13126167488598567,
      "base": 0.13434175165690262,
      "sd": 0.15950163942414836,
      "pp": 0.09095399447220685,
      "cv": 0.1690353304722566,
      "basecv": 0.17301890496030276,
      "sdcv": 0.1745921311389156,
      "ppcv": 0.1689609079936092,
      "winMix": 1,
      "winPP": 0.578125
    },
    "c0": {
      "v": 0.15879244777477566,
      "base": 0.16144964378098742,
      "sd": 0.19290178939917837,
      "pp": 0.10768783244094718,
      "cv": 0.20440594729369915,
      "basecv": 0.20786756048567862,
      "sdcv": 0.2110967651469952,
      "ppcv": 0.19999486262894298,
      "winMix": 0.9609375,
      "winPP": 0.0546875
    }
  },
  {
    "type": "dense",
    "t": 0.5,
    "n": 128,
    "D": 0.5217161398276186,
    "cost": 1.478283860172382,
    "c1": {
      "v": 0.10432705772473404,
      "base": 0.10857278097246048,
      "sd": 0.14836825609913834,
      "pp": 0.08537604848262671,
      "cv": 0.15452573985945722,
      "basecv": 0.16080376370311963,
      "sdcv": 0.16242391980288912,
      "ppcv": 0.15861625644849792,
      "winMix": 1,
      "winPP": 0.96875
    },
    "c0.5": {
      "v": 0.11326604956333321,
      "base": 0.1161342059845287,
      "sd": 0.15950163942414836,
      "pp": 0.09095399447220685,
      "cv": 0.16772759379422336,
      "basecv": 0.17197914345022727,
      "sdcv": 0.1745921311389156,
      "ppcv": 0.1689609079936092,
      "winMix": 1,
      "winPP": 0.8359375
    },
    "c0": {
      "v": 0.13618621926812058,
      "base": 0.13881848102073346,
      "sd": 0.19290178939917837,
      "pp": 0.10768783244094718,
      "cv": 0.2015761339359332,
      "basecv": 0.20550528269155038,
      "sdcv": 0.2110967651469952,
      "ppcv": 0.19999486262894298,
      "winMix": 0.9765625,
      "winPP": 0.2578125
    }
  },
  {
    "type": "dense",
    "t": 1,
    "n": 128,
    "D": 0.13964901371424812,
    "cost": 1.8603509862857515,
    "c1": {
      "v": 0.08448502506322948,
      "base": 0.08583347629108529,
      "sd": 0.14836825609913834,
      "pp": 0.08537604848262671,
      "cv": 0.1576896565862486,
      "basecv": 0.1601985199540205,
      "sdcv": 0.16242391980288912,
      "ppcv": 0.15861625644849792,
      "winMix": 1,
      "winPP": 0.703125
    },
    "c0.5": {
      "v": 0.09031307418884972,
      "base": 0.09146203456114073,
      "sd": 0.15950163942414836,
      "pp": 0.09095399447220685,
      "cv": 0.16854579107861709,
      "basecv": 0.17068137322158616,
      "sdcv": 0.1745921311389156,
      "ppcv": 0.1689609079936092,
      "winMix": 0.9921875,
      "winPP": 0.5703125
    },
    "c0": {
      "v": 0.10635600914259481,
      "base": 0.10834770937130665,
      "sd": 0.19290178939917837,
      "pp": 0.10768783244094718,
      "cv": 0.19842808875971793,
      "basecv": 0.20212993302428278,
      "sdcv": 0.2110967651469952,
      "ppcv": 0.19999486262894298,
      "winMix": 0.953125,
      "winPP": 0.671875
    }
  },
  {
    "type": "constant",
    "t": 0.25,
    "n": 128,
    "D": 0.7127497028843044,
    "cost": 1.2872502971156963,
    "c1": {
      "v": 0.16314235050169695,
      "base": 0.16604576285261818,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.20997953758120608,
      "basecv": 0.21374353838410567,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.7109375
    },
    "c0.5": {
      "v": 0.16314235050169695,
      "base": 0.16604576285261818,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.20997953758120608,
      "basecv": 0.21374353838410567,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.7109375
    },
    "c0": {
      "v": 0.16314235050169695,
      "base": 0.16604576285261818,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.20997953758120608,
      "basecv": 0.21374353838410567,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.7109375
    }
  },
  {
    "type": "constant",
    "t": 0.5,
    "n": 128,
    "D": 0.5217161398276186,
    "cost": 1.478283860172382,
    "c1": {
      "v": 0.14137289000772227,
      "base": 0.1439489310576497,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.20919618769022935,
      "basecv": 0.213021543494121,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.8515625
    },
    "c0.5": {
      "v": 0.14137289000772227,
      "base": 0.1439489310576497,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.20919618769022935,
      "basecv": 0.213021543494121,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.8515625
    },
    "c0": {
      "v": 0.14137289000772227,
      "base": 0.1439489310576497,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.20919618769022935,
      "basecv": 0.213021543494121,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.8515625
    }
  },
  {
    "type": "constant",
    "t": 1,
    "n": 128,
    "D": 0.13964901371424812,
    "cost": 1.8603509862857515,
    "c1": {
      "v": 0.11307798279289262,
      "base": 0.11388553616653166,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.21081064713129355,
      "basecv": 0.2123153241800643,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.4140625
    },
    "c0.5": {
      "v": 0.11307798279289262,
      "base": 0.11388553616653166,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.21081064713129355,
      "basecv": 0.2123153241800643,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.4140625
    },
    "c0": {
      "v": 0.11307798279289262,
      "base": 0.11388553616653166,
      "sd": 0.19646636653897795,
      "pp": 0.11325106511121716,
      "cv": 0.21081064713129355,
      "basecv": 0.2123153241800643,
      "sdcv": 0.21499590191449847,
      "ppcv": 0.21016878469779773,
      "winMix": 1,
      "winPP": 0.4140625
    }
  }
]
## P: cost/limits
Allocation O(V log tolerance), full general flow graph O(V²) edges (can exploit missing-diagonal structure for more efficient construction, not implemented); at vocab128k a dense FP32 coupling table 65.536GB, exceeds one 40GB GPU. Actual large vocab factorized or implicit transport needed. Classical maximal residual factorized O(V) and ordinary independent p/p no dense graph; both strong. Reward predictor m requires full vector or sparse structure; prior labels, calibration and execution, policy score, tool and state clone, communications, optimization and rollback all charged. A 15% actual total improvement / +20% effective work not established. Comparison matched fixed D vs interpolated design only, not strongest unrestricted sampler or equal end-to-end time. Same-information flat and ordinary AR+second action with same design reproduce. Reject if no real tool or cost advantage.
## literature actually searched (primary abstracts/publisher pages, not exhaustive)
SpecTr https://arxiv.org/abs/2310.15141 (2023 optimal transport); SpecTr-GBV https://aclanthology.org/2026.emnlp-main.1340/ (EMNLP2026 multi-draft transport); Acceptance-Aware Draft Model Training https://arxiv.org/abs/2609.24150 (2026 acceptance objective); Variational Speculative Decoding https://arxiv.org/abs/2602.05774; When RL Meets Adaptive Speculative Training https://proceedings.mlr.press/v306/wang26j.html; When Drafts Evolve https://proceedings.mlr.press/v306/qian26k.html; Expected Policy Gradients https://www.jmlr.org/papers/v21/18-012.html; Mirage of Action-Dependent Baselines https://proceedings.mlr.press/v80/tucker18a.html; Optimal Multiple Importance Sampling https://graphics.cg.uni-saarland.de/publications/kondapaneni-2019-siggraph-optimal-mis.html; Efficient balanced sampling cube method https://academic.oup.com/biomet/article-abstract/91/4/893/459329; Optimal sampling and estimation strategies under linear model https://academic.oup.com/biomet/article-abstract/95/3/521/217377. No formal text similarity or exhaustive full-text/patent check. Novelty threat: known optimal transport, sampling allocation, model-assisted estimation and action-dependent baselines. Distinct application diagonal polytope and fixed acceptance frontier may be useful theory, not proven first.
## verdict
Subjective 7.375/10 (problem8.5, mechanism7, theory8.5, value6.5, distance5.5; 25/25/20/15/15). G1 below8; G2 critical fail (no causal HFL↔SD, AR and delegated flat reproduce); G3 partial (conditional T, no production decoding); G4 fail; G5/G6 unverified/no 80% evidence; G7 partial. Retain sharp feasible polytope, oracle dominance at fixed agreement, calibration sensitivity. Do not designate independent top-journal candidate. Next only if actual sandbox profile indicates tools significant and small action alphabet, train reward predictors out-of-sample, compare ordinary HFL+production SD, optimal AR multi-action/stratified/IS, max coupling+same control variate, equal-resource flat, two pretrained 1.5B/3B scales on 1–2x40GB, 3 nonIID/delay regimes and seeds, real end-to-end timing/quality/CI. No GPU done.