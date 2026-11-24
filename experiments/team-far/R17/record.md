# team-far/R17 — linear sparse prescribed-diagonal transport (LSPT)
2026-11-24. Start/prewrite 3d31cd221f83982ec6e42edf141cd6e7a4dfbd00; README 960d4cd4f2dbe1ee0a61607f74352e7dc2b069c1, 3293 lines, 446520 chars. Full README obtained and R16 record reopened; headings screened for shared representation, architecture and gradient overlap. Specifically builds on R16 fixed acceptance feasible diagonal and centered reward estimator, not a new neural shared backbone. T=conditional proof, E=actual FP64 JS execution, O=oracle, P=proxy, H=unverified. No claim literature exhaustive; prior A/R06/FRCS and HMC/SpecTr near neighbors (residual transport), R16 dense max-flow is reference. The promising bottleneck addressed is V^2 edge construction, not actual HFL end-to-end.
## T: linear sparse construction for R16 feasible diagonal
Given p,q, feasible prescribed diagonal d with sum D, residual row masses a=q-d and column masses b=p-d, T=1-D, a_i+b_i<=T. A basic greedy ordered interval intersection (northwest corner) allocates residual rows to residual columns using at most 2V-1 positive entries; collect unwanted diagonal t_i and keep offdiagonal list. Let H=sum t_i, k argmax t_i. H=0 done. Case A max t_i<=H/2: allocate diagonal mass by circular rotation of one of the two identical partitions by H/2. Every interval length <=H/2, so its shifted copy has disjoint interior, no diagonal overlap. Sweep two ordered circular partitions, O(V), at most 2V new pieces. Case B t_k>H/2: for every i!=k remove t_i from (i,i) and t_i from (k,k), replace by (i,k),(k,i), each weight t_i. Leaves residual unwanted diagonal r=2t_k-H only at k. Remaining offdiagonal total mass not incident on k equals T-a_k-b_k+r>=r by feasibility. For each preexisting offdiagonal (i,j), i,j!=k, transfer z=min(r,weight) from (i,j) and (k,k) to (i,k),(k,j), decrement r. New edges incident on k never need to be scanned again. Each swap preserves row/column marginal, introduces no diagonal. A single scan suffices, at most one partially consumed edge. All cases O(V) arithmetic and O(V) memory; a conservative bound 5V emitted positive edge pieces including prescribed diagonal (duplicates allowed), not necessarily minimum support. Proof: initial <=2V-1 entries, remove h nonzero unwanted diagonals; case A add <=2h, then <=V intended diagonal => <=4V-1. Case B add 2(h-1), consume <=2V-1-h original offdiag edges; each fully consumed adds net <=1, at most one partial adds 2, then <=V intended diagonal => <=5V-3. Zero residual handled; exact real-arithmetic construction, floating precision checked separately. Can group row entries into CSR in O(V) counting with row offsets, sample a row by cumulative mass/alias, not implemented as actual GPU sampler. R16 allocation bisection O(V log(1/eps)) still needed, no reduction in model logits or full reward estimates. This proof gives a practical constructive upper bound, not a first discovery of NW corner, circular coupling or transportation swaps.
## T: oracle equivalence and nonoracle limitation
R16 centered estimator G=sum p_i m_i s_i+sum I_i p_i/pi_i (R_i-m_i)s_i has exact conditional mean grad J for pre-sample m and independent legal environment. If m=mu and independent noise, trace variance depends only on pi_i=p_i+q_i-d_i, not offdiagonal coupling; thus sparse construction retains exactly the dense allocation optimum without maxflow. With misspecification, cross terms involving Gamma_ij+Gamma_ji and residual means change, so no general dominance over dense arbitrary construction or independent p/p. In experiment all calibration conditions reuse oracle allocation to isolate effect of transport (different from R16 allocating using misspecified m); not a deployable prediction test. Same p,q and fixed D preserves expected distinct tool count 2-D. Formal Y marginal p by coupling and conditional sampling, exact in real arithmetic; sequence requires causal kernel at actual prefixes, independent random sources, no reward-dependent selection, correct EOS/bonus/temperature/top-p and finite precision. Greedy use same target argmax; no new greedy path or production sequence run. Fixed common-snapshot weighted regional aggregation possible but identical-info flat can reproduce; ordinary AR with same candidate and sandbox can implement same sampler and estimator. No new network architecture or proven dual causal necessity.
## E/O: execution
Source {"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}],"prior":"experiments/team-far/R16/probe.js"}. Actual reloaded same two 7264-param external microGPT checkpoints, 128 contexts, 3 synthetic Bernoulli reward profiles, 3 match budgets, 1152 configurations; actual R16 allocation with exact reward mean then both original dense Dinic flow and sparse algorithm, 3 residual-mean conditions m=.5+cal*(mu-.5), exact logit-score moments. Same initial forward source as R16; prior training overlap possible, no actual tool or training. Max marginal error 8.326672684688674e-16; max gradient mean error 1.3877787807814457e-16; max oracle variance difference 0. Groups:
[
  {
    "type": "sparse",
    "n": 384,
    "edges": 87.51041666666667,
    "c1": {
      "meanSparse": 0.03095091983805291,
      "meanDense": 0.03095091983805291,
      "maxAbsDifference": 0,
      "win": 0
    },
    "c0.5": {
      "meanSparse": 0.05551844852746979,
      "meanDense": 0.05552790168538349,
      "maxAbsDifference": 0.005158857067106427,
      "win": 0.3619791666666667
    },
    "c0": {
      "meanSparse": 0.12922103459572068,
      "meanDense": 0.12925884722737546,
      "maxAbsDifference": 0.020635428268425682,
      "win": 0.3619791666666667
    }
  },
  {
    "type": "dense",
    "n": 384,
    "edges": 79.8515625,
    "c1": {
      "meanSparse": 0.10325773372564662,
      "meanDense": 0.10325773372564662,
      "maxAbsDifference": 0,
      "win": 0
    },
    "c0.5": {
      "meanSparse": 0.11164908593244487,
      "meanDense": 0.11144649456983964,
      "maxAbsDifference": 0.0019107645591552452,
      "win": 0.2109375
    },
    "c0": {
      "meanSparse": 0.13682314255284,
      "meanDense": 0.13601277710241877,
      "maxAbsDifference": 0.007643058236620981,
      "win": 0.2109375
    }
  },
  {
    "type": "constant",
    "n": 384,
    "edges": 81.66145833333333,
    "c1": {
      "meanSparse": 0.13919774110077074,
      "meanDense": 0.13919774110077074,
      "maxAbsDifference": 0,
      "win": 0
    },
    "c0.5": {
      "meanSparse": 0.13919774110077074,
      "meanDense": 0.13919774110077074,
      "maxAbsDifference": 0,
      "win": 0
    },
    "c0": {
      "meanSparse": 0.13919774110077074,
      "meanDense": 0.13919774110077074,
      "maxAbsDifference": 0,
      "win": 0
    }
  }
]
Additional actual synthetic p,q stress tests 6 vocab sizes 2/3/27/1024/32768/128000, 4 distribution types (random nearly-uniform, Zipf, dominant token, disjoint sparse supports), 3 seeds 11/29/73, 3 feasible diagonals from mix of maximal and independent; 216 cases. Stress test validates construction given feasible diagonal, not R16 allocation at large scale or pretrained large model. Max element error 3.557079262046694e-12, max L1 3.5589695792282136e-12, max edge pieces/V 4.592592592592593; actual edge counts/repair branches in JSON (note output kind is repair branch because audit spreads over distribution kind; distribution label can be reconstructed from loop ordering; correction in next section if needed). No actual wall time or GPU memory measurements. No model generation, real physical HFL, CI or two pretrained scales. Reproduce node experiments/team-far/R17/probe.js weights.json checkpoint.json input.txt.
## P: cost
At V=128000, dense FP32 Gamma requires 65.536GB just values; R16 flow graph would be even larger. Conservative <=640000 sparse edge records, if two uint32 indices+one FP32 weight each, <=7.68MB for entries, plus CSR/row offsets, p/q/di, allocator and temporary arrays. Actual JS object overhead not that compact. This is representational arithmetic, not measured GPU memory or time. Classical maximal coupling and independent sampling use O(V) without materializing joint matrix; our algorithm only removes expensive prescribed-diagonal construction, does not make entire method faster than these baselines. Full R16 variance improvements remain small and require oracle. No actual G5. 
## literature web search 2026-11-24 (original abstracts/venue pages, not full audit)
SpecTr NeurIPS2023 https://arxiv.org/abs/2310.15141 (nearly linear OT approximation); SpecHub EMNLP2024 https://aclanthology.org/2024.emnlp-main.1148/ (sparse joint distribution and linear overhead, close complexity claim); Global Resolution ICLR2026 https://arxiv.org/abs/2511.15898 (convex multi-draft acceptance); List-Level Distribution Coupling NeurIPS2025 https://arxiv.org/abs/2506.05632; SpecTr-GBV EMNLP2026 https://aclanthology.org/2026.emnlp-main.1340/; Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds https://arxiv.org/abs/2610.10411 (round-coupled objective); Expected Policy Gradients https://www.jmlr.org/papers/v21/18-012.html; Action-dependent Control Variates via Stein Identity https://www.microsoft.com/en-us/research/publication/action-dependent-control-variates-policy-optimization-via-stein-identity/; classical NW corner https://gistbok-ltb.ucgis.org/current/concept/AM-05-042. No exact originality claim, sparse couplings already exist. This is a specialized prescribed-diagonal construction, not a new universal OT algorithm or trained architecture. No text-similarity check.
## Verdict
Subjective weighted 7.575/10 (8.5/7.5/8.5/7/5.5; 25/25/20/15/15); mechanism/distance below8; G1/G2/G4 fail; G3/G7 partial; G5/G6 unverified, no 80% evidence. Reject standalone. Retain useful exact linear construction; do not confuse sparse data structure with new learning primitive. Next stage require new physically grounded joint neural computation or real task evidence, not further optimizing probability transport in absence of G2. Realistic pilot if resources available: 1–2x40GB two pretrained scales 1.5B/3B, actual self-SD and resettable tool tasks, compare ordinary HFL+production SD, AR+same estimator, maximal and independent samplers, SpecHub/related, same-information flat; actual model logits, allocations, tool, comms, training, rollback, quality and CI, 3 regimes/seeds. Good result same quality/budget strongest baseline -15% real total time or +20% effective work. No GPU performed.