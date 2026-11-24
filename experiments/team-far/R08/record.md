# team-far/R08 — SDFD: survival-dominated first-divergence sampling for federated zeroth-order policy learning
2026-11-24. Starting/prewrite 0c614f8723eead14d487d3a45863d8d8c45ada1a, README blob e06a992a75645df95f706a1ecaf7c8c3a20e70e2, 3215 lines. Fetched entire README and screened headings; detailed review R04/R06/R07, A/R04/R06/R13, FTCD, FBSG. No independent audit of all proofs/experiments. User target 8.35. This is a new estimator, not another zero-certificate threshold. No new GPU or physical HFL experiment.

## T: problem and survival domination
Fixed prompt and model snapshot, frozen backbone and untied output W, direction bank U_j supported on disjoint row blocks B_j with Rademacher entries, eps>0, full softmax fixed temperature T. Let p0(v|h)=softmax(Wh/T), p_j±(v|h)=p0(v|h) exp(±d_j(v,h))/Z_j±(h), d_j=eps U_j h/T. Let m_j=min(p_j+,p_j-), delta_j=1-sum m_j=TV(p_j+,p_j-). By Cauchy Schwarz, Z+Z- >= (sum p0)^2=1, hence p+ p- =p0²/(Z+Z-) and m_j<=sqrt(p+p-)<=p0 pointwise. The common subprobability path of sequential maximal coupling before first disagreement at t has mass prod_{s<t} m_j(y_s|h_s). For base anchor A~P0, W_jt=prod_{s<t} m_j(A_s|A_<s)/p0(A_s|A_<s) in [0,1]. No exponential amplification of survival importance weight, for any fixed horizon. Nontruncated softmax crucial; top-p/k or changed shared hidden require separate proof/fallback. Not a claim about arbitrary two policies.

For a direction j, couple +/- transitions maximally while prefixes match. At first disagreement, residual token laws (p+−m)/delta and (p−−m)/delta are sampled independently (disjoint support), and each suffix uses its own lawful policy, independently in this implementation. If no disagreement by L, identical final outputs, reward contrast zero under deterministic common R or identical exogenous reward noise; not parameter-dependent reward or independent reward calls. For any bounded R in [0,1], difference J_j+−J_j-= E_A sum_t W_jt delta_jt E[DeltaR|A_<t,first split at t]. Proof expand by first split: shared-prefix survival product times delta times contrast; no-split term zero; replacing shared-prefix measure by P0 times W gives identity. Can draw anchor using exact self-spec target-first with shallow q and logical-position independent Gumbels, proper rejection/truncation/bonus/EOS; this is conceptually exact but not actually executed here. Branches are deliberately drawn from conditional residual laws, NOT ordinary on-policy samples and MUST NOT be fed directly to GRPO/independent group learning. Same finite-difference expectation, not equality of original paired rollout joint law or of optimizer updates. Greedy inference separate target argmax, no stochastic estimator claim for discontinuous greedy.

## T: lazy multi-direction envelope proposal
Let b_t=eps||h||_1/T, P_B=sum_{v in B_j}p0(v). With support block, sum_v p0|exp(±d)-1|<=P_B(e^b−1), Z±>=e^-b. TV(p±,p0)<=P_B e^b(e^b−1) by triangle on normalized/unormalized masses. Thus delta<=H_jt=min(1,2 e^b(e^b−1) P_B). Choose ONE (j,t) conditional on completed anchor with pi_jt proportional to ||u_j|| H_jt. Then compute true delta and survival W only for selected j and required prefix, sample residual suffix pair, output G=(W delta DeltaR/(2eps pi_jt)) u_j. For zero delta return zero; if all H zero entire bank contrast zero. Because every nonzero term has positive pi, expectation equals sum_j u_j (J_j+−J_j-)/(2eps), exactly same expected bank finite difference, not exact unsmoothed gradient. For R∈[0,1], norm G <= S(A)/(2eps), S=sum_jt||u_j||H_jt, since W<=1 and delta<=H. Fixed finite horizon, hidden bounded, fixed directions, S=O(eps) as eps→0; second moment bounded in small eps, unlike naive paired FD which can have variance O(1/eps) (Bernoulli example p±=.5±eps, R=token, maximal coupling mismatch 2eps, variance 1/(2eps)−1). This is only upper bound, not uniform dominance; rare-event sampling and conditional MC/MVD are established. In this algorithm one forced branch contrast may require two actual rewards even when ordinary paired FD has zero query (if outputs equal), so variance-cost tradeoff is essential. The envelope needs base distributions/block masses and a scan; direction generation selected only, compute full selected sparse-block normalization at each prefix; unselected directions do not need perturbed projections. The oracle hazard proposal proportional to ||u|| W delta requires all true hazards and is not free. Uniform direction/time and telescoping hybrid MVD are strong controls. For hybrid, P+−P−=sum_t P+prefix (p+−p−) P−suffix; weight P+prefix/P0prefix can exceed 1. Shared anchor likelihood-ratio FD: R(A)*(P_j+(A)−P_j-(A))/(2eps P0(A)) for every j; exact same mean, only one reward call; for deterministic reward, constant baseline subtractable since expected ratio difference zero; important strong baseline, can compute all sparse block direction rows at anchor. If exact expected reward available, exact integration dominates stochastic method.
HFL proposal (H): region holds local prompt/reward, evaluates above estimators, communicates preset seed plus sparse scalar to regional and cloud weighted aggregates; ordinary equal-resource flat can delegate identical computations and sparse reporting, so hierarchy-specific necessity is still not shown. SD can accelerate anchor/suffix, but AR with same estimator can too. Need jointly learn draft based on branch-weighted contexts and verify full causal learning benefit (not implemented). No mechanism score increase simply because estimator proven.

## E/O/P: actual exact conditional moment audit
microGPT fixed aa28d7c0aa66704ce56b263dfd3270579f3d5f6d, weights df323b7e714ceeb10ca070f9945a82c51b2c233c, checkpoint 7f0b4968c56290ebad706ab740de942c369f42a5, input 736d88e18b70b292750a30e5cca6a92b0b333bd2; 2 same-size 7264-parameter checkpoints; 4 logical clients by first character mod4, one fixed name per seed/checkpoint/K/client; no training, no physical HFL, no sampled SD, no GPU/time/CI. Same 2-layer d16 V27 forward as FAR/R07/HCA R09/FEIA. 2-token horizon, 27 first tokens and all 729 full paths, real model logits at root and all first-token prefixes, 2 checkpoints x 3 seeds x 2 bank sizes (K4/8) x 3 eps (.001/.005/.04) =36 rows, 4 contexts per row=144 prefix-config instances (repeated across eps), no independent training seeds. Gold fraction correct proxy reward. Exact expected per-fixed-bank squared Euclidean estimation error (trace variance) from conditional first-divergence law; +/- suffix independent after split. Compare full-bank sequential maximal pairs, one (j,t) uniform first-divergence, envelope SDFD, true hazard oracle (not optimal reward-aware), telescoping hybrid with same envelope, full-bank shared-anchor LR, and LR oracle constant baseline per direction. Same exact finite-difference bank mean for all. Orthogonal support simplifies trace variance, mean norm from exact direct expectation. Cost proxy: sparse head projection row counts for selected j,t and 2 base anchor rows, reuse base suffix if prefix equals anchor; direction projection for t0 includes root+two branch rows, t1 root+anchor prefix; includes no draft, RNG, scans/softmax, trunk, network or actual time. For LR 4V head rows (2 base and 2 full sparse-bank rows), 1 reward; forced split 2 rewards if positive hazard; full bank pair expected 2 sum P(split) reward calls if equal outputs can skip deterministic reward, not actual reward queries. Oracle hazard allocation cost estimate excludes all-hazards proposal formation, so not actual oracle cost. Full-bank pair projection cost not estimated, because cache-sharing across directions depends on coupling. Exact moment audit is NOT real learning performance or measured end-to-end speed. 
[
  {
    "K": 4,
    "eps": 0.001,
    "n": 24,
    "meanSq": 8.472998844614738,
    "variances": {
      "paired": 19072.95898335381,
      "envelope": 183.33133947585418,
      "uniform": 228.81422917463615,
      "oracleHazard": 160.8198488842182,
      "hybrid": 183.56262357990525,
      "lr": 162.67724453165877,
      "lrOracle": 110.45958927436408
    },
    "cost": {
      "envelope": 94.03715595684635,
      "uniform": 95.35509486307735,
      "oracleHazard": 95.19419608082075,
      "hybrid": 94.03715595684635,
      "lr": 108,
      "pairedReward": 0.017214950593782596
    },
    "active": {
      "envelope": 1,
      "paired": 0.008607475296891298
    }
  },
  {
    "K": 4,
    "eps": 0.005,
    "n": 24,
    "meanSq": 8.472960080939417,
    "variances": {
      "paired": 3799.368039159656,
      "envelope": 182.7651714377154,
      "uniform": 227.64814661018497,
      "oracleHazard": 160.00011687987845,
      "hybrid": 183.91977639359803,
      "lr": 162.68426606663687,
      "lrOracle": 110.46400736466829
    },
    "cost": {
      "envelope": 93.97300665786996,
      "uniform": 95.35510902423117,
      "oracleHazard": 95.25985069893227,
      "hybrid": 93.97300665786996,
      "lr": 108,
      "pairedReward": 0.08586298956536544
    },
    "active": {
      "envelope": 1,
      "paired": 0.04293149478268272
    }
  },
  {
    "K": 4,
    "eps": 0.04,
    "n": 24,
    "meanSq": 8.470319905403962,
    "variances": {
      "paired": 458.59359603033244,
      "envelope": 177.644125235949,
      "uniform": 217.9947784599834,
      "oracleHazard": 153.17993913070035,
      "hybrid": 186.9137285900671,
      "lr": 163.15380896833378,
      "lrOracle": 110.75868884906048
    },
    "cost": {
      "envelope": 93.62982967548639,
      "uniform": 95.35604296389391,
      "oracleHazard": 95.8596028268875,
      "hybrid": 93.62982967548639,
      "lr": 108,
      "pairedReward": 0.6727021732833895
    },
    "active": {
      "envelope": 1,
      "paired": 0.3363510866416948
    }
  },
  {
    "K": 8,
    "eps": 0.001,
    "n": 24,
    "meanSq": 6.268000691757617,
    "variances": {
      "paired": 9937.445369021361,
      "envelope": 128.10103449467593,
      "uniform": 215.55568560780003,
      "oracleHazard": 100.10894234978338,
      "hybrid": 128.14379556440932,
      "lr": 84.7574023624494,
      "lrOracle": 52.24107514988459
    },
    "cost": {
      "envelope": 85.88766546360232,
      "uniform": 87.11881629846742,
      "oracleHazard": 87.90689849821622,
      "hybrid": 85.88766546360232,
      "lr": 108,
      "pairedReward": 0.020070117341262853
    },
    "active": {
      "envelope": 1.0000000000000002,
      "paired": 0.010035058670631427
    }
  },
  {
    "K": 8,
    "eps": 0.005,
    "n": 24,
    "meanSq": 6.267943720309952,
    "variances": {
      "paired": 1980.7532671281526,
      "envelope": 128.0395628525647,
      "uniform": 215.09997335477942,
      "oracleHazard": 99.8432687473016,
      "hybrid": 128.25154651919073,
      "lr": 84.76515664944347,
      "lrOracle": 52.24486943732264
    },
    "cost": {
      "envelope": 85.82877434197776,
      "uniform": 87.11882036088672,
      "oracleHazard": 87.9455132994587,
      "hybrid": 85.82877434197776,
      "lr": 108,
      "pairedReward": 0.10020353915735358
    },
    "active": {
      "envelope": 1,
      "paired": 0.05010176957867679
    }
  },
  {
    "K": 8,
    "eps": 0.04,
    "n": 24,
    "meanSq": 6.2637933253720535,
    "variances": {
      "paired": 240.55054668849115,
      "envelope": 128.22916562606025,
      "uniform": 211.40823861094728,
      "oracleHazard": 97.81071592327112,
      "hybrid": 129.81696630279257,
      "lr": 85.27969937335281,
      "lrOracle": 52.495220457467575
    },
    "cost": {
      "envelope": 85.39313940798735,
      "uniform": 87.11909083110969,
      "oracleHazard": 88.29548077111633,
      "hybrid": 85.39313940798735,
      "lr": 108,
      "pairedReward": 0.7926262301794943
    },
    "active": {
      "envelope": 1,
      "paired": 0.39631311508974715
    }
  }
]
Max absolute mean error 9.219014440731144e-14, maximum m>p0 violation 0, envelope violation 0, moment bound violation 0. Reproduce node experiments/team-far/R08/probe.js weights.json checkpoint.json input.txt. Source may have seen names in pretraining. All conditional moments refer to ideal full softmax FP64, not production sampling or greedy.

## Literature audit (actual web search 2026-11-24)
- Measure-Valued Differentiation for Markov Chains (2008) https://doi.org/10.1007/s10957-007-9297-7, finite products of kernels and unbiased estimators, direct prior.
- Perturbation analysis via coupling (1999) https://www.sciencedirect.com/science/article/abs/pii/S147466701756810X, including maximal coupling and rare perturbation sensitivity, direct near-neighbor.
- Monte Carlo Gradient Estimation in Machine Learning (JMLR 2020) https://jmlr.org/papers/v21/19-346.html, weak derivatives and signed measures.
- An Empirical Analysis of Measure-Valued Derivatives for Policy Gradients (2021) https://arxiv.org/abs/2107.09359, RL MVD.
- Efficient Federated RLHF via Zeroth-Order Policy Optimization (2026) https://arxiv.org/abs/2604.17747, federated ZO policy learning.
- HiSo ICLR 2026 https://proceedings.iclr.cc/paper_files/paper/2026/hash/3e8b1835833ef809059efa74b9df6805-Abstract-Conference.html, Hessian-informed federated ZO.
- ACL 2026 global adaptive momentum/local personalized perturbation https://aclanthology.org/2026.acl-long.1851/.
- EfficientRollout https://arxiv.org/abs/2606.18967, SpecRoll https://arxiv.org/abs/2608.04962, both 2026.
- Accelerating RL Post-Training Rollouts via System-Integrated SD https://arxiv.org/abs/2604.26779.
Only search abstracts/indices/visible excerpts, no exhaustive full-text or proof-level novelty, no formal plagiarism test. Softmax symmetric survival domination + block-mass envelope and one joint direction/time selector is concrete new combination for this project, not first weak derivative or coupling method. A/R04, A/R06 and R04/R07 related but different target (new estimator of finite difference vs exact per-sample replay). 
## Judgment
R08 is a useful mathematical asset, not a validated stronger HFL-SD candidate. Retain R07/R06 best provisional rubric 7.85/10 (8.5/8/8.5/7/6.5); do not raise G1 based on this known-family MVD estimator/extra proxy. G1 formal not passed, G2 still unproven because same estimator can be run by strong AR and delegated flat; G3 partial theory only (weighted branches not normal target samples; exact anchor/service proof still requires production implementation), G4 fail, G5/G6 unverified, G7 partial. No 80% evidence. Need real 1–2x40GB GPU with two model scales and actual expensive reward benchmarks, compare shared AR+LR with optimal baselines, MVD/hybrid, paired max/Gumbel, Infer-ZO, ordinary HFL+production self-SD and same-info flat; measure actual time to same quality, all cost, 3 heterogeneity/latency settings, 3 independent training seeds and CI, reward-call costs, no claim statistical efficiency is end-to-end speed. If strongest LR/MVD baseline matches or ordinary AR retains all benefit, do not advance. This is a bounded-variance result for the given estimator, not an 8.35-grade journal approval.
