# team-far/R06 — Orthogonal Direction Bank Speculation (ODB-SD)
Date 2026-11-24. Start/prewrite 7be9fe35a89b713ee2c5d8c1604ecc65fff707dd, README blob 264032a9d93baf011e262bd8ac9faa2148d7c663, 3199 lines. Entire README fetched, compared with previous version, targeted A/R04, FVC, FOAS, FSDC, R04-R05 and related gradient/caching results. No independent proof audit of all teams. Requested target 8.35/10; do not inflate grades.

## T: orthogonal noise-sharing theorem
At fixed parameter w, lawful prompt, and fixed orthogonal directions u_1,...,u_K (e.g. disjoint output row supports), define D_j=(R(Y_j^+)-R(Y_j^-))/(2 eps), with paired reference distributions generated from w±eps u_j, common randomness within each pair as in R05. Any coupling across direction pairs preserving their full *within-pair* marginal law leaves E[D_j] and Var(D_j) unchanged. For G=sum_j a_j D_j u_j, fixed scalar coefficients a_j, and any fixed vector g, E||G-g||² = ||E G-g||² + sum_j a_j² ||u_j||² Var(D_j), since cross inner products vanish exactly. Thus shared Gumbels across all directions incur zero additional Euclidean MSE relative to independent pairs, conditional on direction bank and prompt. Unconditional result by tower law if same distribution over banks/prompts. Covariance matrices may differ, not equality in PSD order, convergence path or Adam; direction supports must be orthogonal and weights fixed before noise. If directions identical and shared D, variance of average not reduced with bank size; if shared random prompts across banks vs independent prompts, comparison must condition on common prompt and not substitute a different data budget. Across clients with overlapping directions, keep independent noise or separately prove orthogonality. Pair reward environment needs shared exogenous seeds only if authorized; same output not sufficient for zero contrast if reward depends on parameter or private stochastic state. Objective is finite difference proxy, not exact true gradient of nonsmooth sampled objective. With random signs on each disjoint block, E[u_j u_j^T] equals coordinate projection P_j, sum P_j=I, so first-order central-difference expansion gives gradient approximation if expected reward J is sufficiently smooth; central FD bias magnitude per direction bounded M eps² ||u_j||³/6 for third derivative norm M, multiply by ||u_j|| and sum to bound vector bias. No claim Gaussian smoothing or unbiased true gradient for Rademacher. Standard orthogonal ZO and CRN already known; new integration is selecting sharing topology by exact MSE-preserving orthogonality and compiling verification support.

## T/H: support-aware verifier
Choose partition of output rows into K disjoint blocks, random signs per coordinate of each block, fixed backbone/untied W. Each direction modifies only its block. Given full base logits plus Gumbel a, compute outside-block winner w_j and score M_j. Tokens outside block unchanged, tokens inside with a_v+b_j<M_j cannot beat w_j in either sign. Evaluate only candidate block rows, reuse increment for both signs. For nonempty proper blocks, all outside maxima can be obtained from block top-1/second block max in O(V+K), not K full scans (experimental code scans outside per block and counts only projection rows). Each block has its own exact certificate; total distinct perturbation rows over all directions at shared prefix <=V, compared K*V dense directions or V if sparse full evaluation. Correct greedy using g=0 and shared Gumbel sample for full-softmax fixed T=1; same tie-break. Greedy proof identical; top-p/k changes support, needs full legal fallback; production FP16 rounding/EOS not implemented. At shared prefix, common shallow head draws causal candidate chain using same per-time Gumbels. Prefetch base outputs along chain (the actual experiment evaluates sequentially, a production SD kernel must batch), then route each direction/sign based on exact target token. Direction-specific certificates only evaluated at actual visited prefix. Prefix-keyed cache reuses base outputs, distinct histories not merged. Same per-time noise is never reused across different logical positions, so reference path marginal exact by induction; no stochastic independence of pair bank intended. If all plus/minus outputs identical and reward solely deterministic function of full output or same exogenous seed, can skip both reward queries for zero contrast; experiment counts such pairs but does not skip actual expensive oracle (proxy only). Independent unrelated users, GRPO groups or clinical/external tool side effects not covered.
A serious alternative: sparse shared-noise orthogonal paired AR with identical prefix cache and certificate performs the same work on visited nodes, minus speculative extra base/draft nodes. SD can only win via hardware parallelism/critical-path latency, not raw unique base projection counts under this implementation. Shared independent bank AR baseline also could share deterministic base features where prefixes coincide, unlike separated-pair diagnostic in this experiment. Equal-resource delegated flat can emulate orthogonal bank. HFL-specific advantage not established; do not conflate orthogonal directions with new hierarchy mathematics.

## E/P: actual experiment
External microGPT aa28d7c0aa66704ce56b263dfd3270579f3d5f6d; source weights df323b7e714ceeb10ca070f9945a82c51b2c233c, checkpoint 7f0b4968c56290ebad706ab740de942c369f42a5, input 736d88e18b70b292750a30e5cca6a92b0b333bd2, same two checkpoints 2-layer d16 V27 7264 weights, may contain upstream training names. Forward inherited from FAR/R05 and HCA/R09/FEIA. Four logical clients first-letter mod4, 2 edges, first80% train, last16 test per client. 2 cloud rounds x 2 regional sync x 2 edges x2 clients=16 updates/config, one sampled name/prompt per update, random row partition K=2/4/8, eps=.005/.04, seeds11/29/73, 2 checkpoints =36 configs, 576 updates. Four generated tokens per sign, shared Gumbel bank, cap2. Gold next-character fraction proxy reward. Fixed direction bank and prompt for all comparators, independent-pair diagnostic uses separate independent bank noises per direction and no interpair base sharing; shared AR and SD use same noise, actual full dense AR reference verifies all sequence tokens; greedy audited at computed prefixes, no EOS/top-p/finite precision contract. All updates use shared SD outputs; no separate convergence comparison. Orthogonal coefficient and SD/AR exactness checks passed. Base/draft head projections V rows each, actual shortlisted direction row dot products counted, cache includes base at speculative prefixes; full reference computation not in candidate proxy. No actual GPU, physical network, real task, full timing, CI or independent upstream holdout; 2 checkpoints not two model scales. No actual expensive reward evaluation skip. 
[
  {
    "K": 2,
    "eps": 0.005,
    "updates": 96,
    "init": 2.5139839887444695,
    "final": 2.5139563174879744,
    "counts": {
      "sharedSD": {
        "base": 419,
        "draft": 419,
        "direction": 822,
        "uncertain": 396,
        "calls": 783,
        "zeroPairs": 184,
        "greedyAudits": 783,
        "formal": 1536,
        "headRows": 23448
      },
      "sharedAR": {
        "base": 396,
        "draft": 0,
        "direction": 822,
        "uncertain": 396,
        "calls": 783,
        "zeroPairs": 184,
        "greedyAudits": 783,
        "formal": 1536,
        "headRows": 11514
      },
      "pairIndependentAR": {
        "base": 789,
        "draft": 0,
        "direction": 795,
        "uncertain": 418,
        "calls": 789,
        "zeroPairs": 181,
        "greedyAudits": 789,
        "formal": 1536,
        "headRows": 22098
      }
    },
    "sharedSDvsIndependent": 1.0610915014933477,
    "sharedARvsIndependent": 0.5210426282921531,
    "SDvsSharedAR": 2.0364773319437206,
    "sharedSecondMoment": 280000,
    "indSecondMoment": 550000
  },
  {
    "K": 2,
    "eps": 0.04,
    "updates": 96,
    "init": 2.5139839887444695,
    "final": 2.5140846114453392,
    "counts": {
      "sharedSD": {
        "base": 494,
        "draft": 494,
        "direction": 1218,
        "uncertain": 527,
        "calls": 861,
        "zeroPairs": 134,
        "greedyAudits": 861,
        "formal": 1536,
        "headRows": 27894
      },
      "sharedAR": {
        "base": 472,
        "draft": 0,
        "direction": 1218,
        "uncertain": 527,
        "calls": 861,
        "zeroPairs": 134,
        "greedyAudits": 861,
        "formal": 1536,
        "headRows": 13962
      },
      "pairIndependentAR": {
        "base": 873,
        "draft": 0,
        "direction": 1132,
        "uncertain": 537,
        "calls": 873,
        "zeroPairs": 120,
        "greedyAudits": 873,
        "formal": 1536,
        "headRows": 24703
      }
    },
    "sharedSDvsIndependent": 1.1291745941788447,
    "sharedARvsIndependent": 0.5651945107881634,
    "SDvsSharedAR": 1.9978513107004727,
    "sharedSecondMoment": 34218.75,
    "indSecondMoment": 54531.25
  },
  {
    "K": 4,
    "eps": 0.005,
    "updates": 96,
    "init": 2.5139839887444695,
    "final": 2.514185332522651,
    "counts": {
      "sharedSD": {
        "base": 417,
        "draft": 417,
        "direction": 531,
        "uncertain": 399,
        "calls": 1547,
        "zeroPairs": 376,
        "greedyAudits": 1547,
        "formal": 3072,
        "headRows": 23049
      },
      "sharedAR": {
        "base": 393,
        "draft": 0,
        "direction": 531,
        "uncertain": 399,
        "calls": 1547,
        "zeroPairs": 376,
        "greedyAudits": 1547,
        "formal": 3072,
        "headRows": 11142
      },
      "pairIndependentAR": {
        "base": 1547,
        "draft": 0,
        "direction": 529,
        "uncertain": 403,
        "calls": 1547,
        "zeroPairs": 374,
        "greedyAudits": 1547,
        "formal": 3072,
        "headRows": 42298
      }
    },
    "sharedSDvsIndependent": 0.5449193815310417,
    "sharedARvsIndependent": 0.26341671000992956,
    "SDvsSharedAR": 2.0686591276252018,
    "sharedSecondMoment": 140000,
    "indSecondMoment": 490000
  },
  {
    "K": 4,
    "eps": 0.04,
    "updates": 96,
    "init": 2.5139839887444695,
    "final": 2.5140990813928945,
    "counts": {
      "sharedSD": {
        "base": 483,
        "draft": 483,
        "direction": 790,
        "uncertain": 546,
        "calls": 1648,
        "zeroPairs": 318,
        "greedyAudits": 1648,
        "formal": 3072,
        "headRows": 26872
      },
      "sharedAR": {
        "base": 468,
        "draft": 0,
        "direction": 790,
        "uncertain": 546,
        "calls": 1648,
        "zeroPairs": 318,
        "greedyAudits": 1648,
        "formal": 3072,
        "headRows": 13426
      },
      "pairIndependentAR": {
        "base": 1657,
        "draft": 0,
        "direction": 772,
        "uncertain": 544,
        "calls": 1657,
        "zeroPairs": 316,
        "greedyAudits": 1657,
        "formal": 3072,
        "headRows": 45511
      }
    },
    "sharedSDvsIndependent": 0.5904506602799323,
    "sharedARvsIndependent": 0.295005603041023,
    "SDvsSharedAR": 2.001489646953672,
    "sharedSecondMoment": 29375,
    "indSecondMoment": 30156.25
  },
  {
    "K": 8,
    "eps": 0.005,
    "updates": 96,
    "init": 2.5139839887444695,
    "final": 2.5140991341494017,
    "counts": {
      "sharedSD": {
        "base": 431,
        "draft": 431,
        "direction": 457,
        "uncertain": 412,
        "calls": 3102,
        "zeroPairs": 756,
        "greedyAudits": 3102,
        "formal": 6144,
        "headRows": 23731
      },
      "sharedAR": {
        "base": 403,
        "draft": 0,
        "direction": 457,
        "uncertain": 412,
        "calls": 3102,
        "zeroPairs": 756,
        "greedyAudits": 3102,
        "formal": 6144,
        "headRows": 11338
      },
      "pairIndependentAR": {
        "base": 3085,
        "draft": 0,
        "direction": 470,
        "uncertain": 418,
        "calls": 3085,
        "zeroPairs": 755,
        "greedyAudits": 3085,
        "formal": 6144,
        "headRows": 83765
      }
    },
    "sharedSDvsIndependent": 0.2833044827792037,
    "sharedARvsIndependent": 0.13535486181579418,
    "SDvsSharedAR": 2.093049920620921,
    "sharedSecondMoment": 210000,
    "indSecondMoment": 100000
  },
  {
    "K": 8,
    "eps": 0.04,
    "updates": 96,
    "init": 2.5139839887444695,
    "final": 2.5139458978441755,
    "counts": {
      "sharedSD": {
        "base": 530,
        "draft": 530,
        "direction": 681,
        "uncertain": 592,
        "calls": 3250,
        "zeroPairs": 675,
        "greedyAudits": 3250,
        "formal": 6144,
        "headRows": 29301
      },
      "sharedAR": {
        "base": 510,
        "draft": 0,
        "direction": 681,
        "uncertain": 592,
        "calls": 3250,
        "zeroPairs": 675,
        "greedyAudits": 3250,
        "formal": 6144,
        "headRows": 14451
      },
      "pairIndependentAR": {
        "base": 3183,
        "draft": 0,
        "direction": 637,
        "uncertain": 558,
        "calls": 3183,
        "zeroPairs": 681,
        "greedyAudits": 3183,
        "formal": 6144,
        "headRows": 86578
      }
    },
    "sharedSDvsIndependent": 0.33843470627642125,
    "sharedARvsIndependent": 0.16691307260504978,
    "SDvsSharedAR": 2.0276105459829767,
    "sharedSecondMoment": 15625,
    "indSecondMoment": 10468.75
  }
]
maxCoeff 0, maxOrthogonalityError 0. Run node experiments/team-far/R06/probe.js weights.json checkpoint.json input.txt. Shared AR is decisive strongest proxy among tested methods. Head proxy excludes trunk, scan, RNG, sorting/gather, communication and time. Report separated-pair baseline only as diagnostic, not state of art. Shared vs independent empirical coefficient second moments not MSE or learning quality and no confidence interval.

## Literature (web original abstract/index pages 2026-11-24)
Direct near-neighbor Efficient Federated RLHF via Zeroth-Order Policy Optimization https://arxiv.org/abs/2604.17747 (binary perturbations federated RLHF); Federated Zeroth-Order Optimization with Direction Aggregation and Variance Reduction https://arxiv.org/abs/2610.11821 (direction aggregation); Inference-Native ZO/Infer-ZO https://arxiv.org/abs/2605.28760 (shared inference execution); FZOO ICLR 2026 https://iclr.cc/media/iclr-2026/Slides/10009867_eHFmpyS.pdf (batched perturbed evaluations); Revisiting ZO minimum-variance aligned perturbations ICLR2025 https://proceedings.iclr.cc/paper_files/paper/2025/file/2345275663a15ee92a06bc957be54a2c-Paper-Conference.pdf; Wanda sparse perturbations Nov2026 https://arxiv.org/abs/2611.16317; SubZero ICCV2025 https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Zeroth-Order_Fine-Tuning_of_LLMs_in_Random_Subspaces_ICCV_2025_paper.pdf; M-LoRA IJCAI2026 https://ijcai.org/proceedings/2026/502; SpecInfer https://arxiv.org/abs/2305.09781; previous R05 DISD/Gumbel. Non-exhaustive, not all full texts audited or formal plagiarism check. New integration and conditional MSE identity for bank-sharing not located verbatim, but orthogonality, CRN, sparse ZO, trie verification and FL ZO are known. No claim new basic theorem.

## Decision
Preliminary design rubric problem 8.5, mechanism 8.0, theory 8.5, actual value 7.0, literature distance 6.5; weights 25/25/20/15/15 => 7.85/10 (2.125+2+1.7+1.05+.975). Mechanism 8 is provisional subjective, not G1 pass or external review. Target 8.35 not achieved. G1 formal not met, G2 critical unproven (shared AR and same-info flat), G3 partial actual exact paired path and theorem, G4 fail real two-scale, G5/G6 unverified, G7 partial. No 80% success basis. Retain orthogonal-bank theorem and negative shared AR comparison. Need true GPU trace of batched prefetch vs shared AR+same certificate, strong federated ZO RLHF, Infer-ZO, FZOO and sparse ZO with equivalent bank/data/communication and head-update quality. Stop if no actual incremental wall-time advantage or no formal hierarchy value; do not count cheap proxy reward savings as real. At 1-2x40GB test two real untied output-head models, three nonIID/latency setups, three independent seeds, matched quality total wall-time -15% or useful work +20%, all costs and CI. No further synthetic repetitions to inflate score.
