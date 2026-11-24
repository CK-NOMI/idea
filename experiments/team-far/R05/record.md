# team-far/R05 — CLV-ZO: Certified Lazy Verifier for Hierarchical Federated Zeroth-Order Rollouts
2026-11-24. Starting/prewrite 899027a0ca034513168691d5cc862ad013dfc94c, README blob 925f421919cedb64e34e7a5e13c150779d0e884a, 3189 lines. Fetched complete README, verified R04 appended without altering prior record; reviewed R04 and nearest A/R02/R03/R04/R06/R08/R09/R11, FCMС/FNSC, FERV/FVP, HMC, FSDC, FCRR, FVT, FLPV, FSGPA, FAR R01-04. No independent replay of all teams. User requests core mechanism 8 and higher overall; actual new computation, not grade inflation.

## H/T: research problem and mechanism
Goal: amortize actual head projection work when federated clients evaluate paired stochastic reward probes for a forward-only black-box objective. Fixed seed-derived dense Rademacher U for perturbation of *untied output head only* W, backbone frozen, common feature h, formal p_s(v|history)=softmax((W h+s eps U h)/temperature), s in {-1,+1}. Same shared q from shallow hidden (no standalone draft LLM). Standard 2-point finite difference reward contrast estimates directional finite difference, not generally exact first-order gradient. Use shared independent Gumbel vector g_t per logical time and pair, never across independent pairs. When both share a prefix, propose k causal tokens using q and same g_t, compute base z=W h at each proposed prefix using batched verifier concept, then certify both target argmax cheaply; commit both matching proposal, otherwise commit both certified outputs and discard remaining speculative path. If they diverge, distinct legal histories evaluated separately; shared randomness per time across plus/minus is intentional. No reusing future candidate evidence in commit choice, noise indexed by logical time so discarded guesses do not alter reference streams. No bonus in this variant. Correctness relative to shared-Gumbel paired AR target, not maximal coupling R04; these are different modes. Cannot claim R04 pairwise frontier achieved by this lazy algorithm. Can switch only based on pre-sampling predictable conditions, not on outcome, and must specify joint law. For service single model same rule with one sign; pair not applicable to independent GRPO sibling requirement.

### Certified lazy exact verifier (T)
Set a_v=(W h)_v/T+g_v, d_v=eps (U h)_v/T. Known uniform b>=max|d_v|, for Rademacher direction b=eps ||h||_1/T. For base winner w=argmax a, compute d_w. Lower threshold L=min_{s in signs}(a_w+s d_w), then candidate set C={v:a_v+b>=L}. For v outside C, each signed score a_v+s d_v<=a_v+b<L<=a_w+s d_w, impossible to win. Evaluate perturbation dot product only for C, reuse for both signs. C includes w. Exact greedy (g=0) or sample (iid Gumbels), with same tie convention and conservative floating-point bound/guard; full production rounding not certified, actual FP64 audited. No perturbed softmax normalization needed for full-support categorical sampling. Top-k/top-p on each perturbed target can change support; full logits fallback before sampling, no savings claim for those settings. T>0. If output tied to input embedding or backbone perturbed, common h invalid; use separate exact reference or restrict update as done here. Arbitrary U has row-specific bounds eps||U_v||*||h||; direction generation/store cost not zero.
Basic shortlist C0={a_v>=max a-2b} contains C, since L>=a_w-b. For fixed h and U before g, set p=softmax(z/T), c=exp(2b). Independent exponential race E_i~Exp(p_i), argmax a corresponds to min E_i, C0 means E_i<=c min_j E_j. Probability i in C0 = P(E_i<=c min_{j!=i}E_j) = c p_i/[1+(c-1)p_i], including p_i=1 endpoint. Thus E|C|<=E|C0|=sum_i c p_i/(1+(c-1)p_i)<=min(V,e^(2b)). Exact deterministic shortlist + expected row bound; classical Gumbel and branch-and-bound mathematics, not claim new foundations. At d4096 Rademacher bound can be loose; if b large shortlist saturates; actual GPU gather irregular and full base logits/RNG/scan remain, no speed guarantee. For full dense shared-base head at a single prefix, cost Vd (base) + Vd (direction), lazy Vd+|C|d plus O(V) scan and gathers. At most half of this narrowly defined head MAC saved, not total compute, strongest baseline rank-one/low-rank direction computes all d_v cheaply; sparse MeZO/Infer-ZO are direct challenges.
HFL aggregation: seed/scalar of FD per local step, regional average then cloud; local multiple steps can use different snapshots. q=W h_shallow updates with W, actual formal target and draft both change; paired verifier preserves prescribed joint sampling law and hence finite-difference update relative to dense common-noise AR, and can change exact amount of dense projection performed. Same-info flat can simulate regional operations; ordinary paired AR + same lazy verifier also saves head compute, so G2 not yet proven. Potential true benefit only if SD batching reduces critical path beyond paired AR with lazy verifier and layered seed/bound training adds measurable advantage. Do not claim HFL uniqueness or 80% evidence.

## E: actual multi-token sampling and local updates
Source microGPT fixed aa28d7c0aa66704ce56b263dfd3270579f3d5f6d, blobs weights df323b7e714ceeb10ca070f9945a82c51b2c233c, checkpoint 7f0b4968c56290ebad706ab740de942c369f42a5, input 736d88e18b70b292750a30e5cca6a92b0b333bd2. Same 2-layer d16 V27, 7264 weights, two same-size checkpoints, source training may have seen data. Forward inherited from team-hca/R09 via FAR/R04. Four clients by first letter mod4, two logical regions; per-client first80% training, last32 evaluation, character strings 6–14. External weights frozen backbone, W untied for this experiment. 2 cloud rounds, 2 regional syncs, 2 clients per region, 16 local updates/config, 4 paired rollouts/update, each 4 tokens after BOS+2 chars, no EOS (BOS allowed as ordinary token), cap2, temperature1 full vocab, random seed11/29/73. Rademacher direction per client update, eps .005/.02/.08, reward fraction of correct next4 chars against observed reference (proxy, not expensive black-box task), eta .001 ascent. Shared noise full dense AR for each pair compared all output tokens with lazy speculative generation. Every verified prefix including wasted proposals audited against dense Gumbel argmax, plus dense greedy audit on same prefix. All checks passed. No physical batching/communications/GPU; full reference evaluations extra audit not counted as algorithm work. 18 configurations, 288 actual local updates, 1152 paired rollouts, 9216 formal tokens. Output head MAC ratios count actual shortlisted rows but are proxy operation counts, not measured time. Base direction dense shared AR count merges exact shared history, separate after divergence. Draft rows count all speculative shallow proposals separately, not included in verifier head ratios. Real draft may use a separate small head, CPU code computes full features for draft and verifier audit; no claimed performance. Evaluation teacher-forced NLL is diagnostic only, no task-quality or convergence claim. No independent pretrained holdout, CI, true HFL network.
Results (6 configs mean per eps, counts summed):
[
  {
    "eps": 0.005,
    "initial": 2.5330156764817,
    "final": 2.533495491974995,
    "rowsRatio": 0.03992937962138792,
    "greedyRatio": 0.04107261132276599,
    "relativeHeadMAC": 0.519964689810694,
    "vsARDenseHeadMAC": 0.5583763068967882,
    "meanB": 0.05675261924861815,
    "meanBound": 1.1203758849271148,
    "meanK": 1.078093249777474,
    "tokenAgreement": 0.96484375,
    "acceptBoth": 0.8487074029535556,
    "updates": 96,
    "pairs": 384,
    "tokens": 3072,
    "calls": 1716,
    "blocks": 840,
    "arCalls": 1598,
    "fullRows": 46332,
    "lazyRows": 1850,
    "greedyRows": 1903,
    "draftRows": 42984
  },
  {
    "eps": 0.02,
    "initial": 2.5330156764817,
    "final": 2.5332968129844526,
    "rowsRatio": 0.04741368053149836,
    "greedyRatio": 0.05782049922572488,
    "relativeHeadMAC": 0.5237068402657492,
    "vsARDenseHeadMAC": 0.5639543033947266,
    "meanB": 0.22763654297490699,
    "meanBound": 1.5807040343167051,
    "meanK": 1.2801693743504556,
    "tokenAgreement": 0.8736979166666666,
    "acceptBoth": 0.8137995897672318,
    "updates": 96,
    "pairs": 384,
    "tokens": 3072,
    "calls": 1850,
    "blocks": 774,
    "arCalls": 1718,
    "fullRows": 49950,
    "lazyRows": 2369,
    "greedyRows": 2889,
    "draftRows": 40122
  },
  {
    "eps": 0.08,
    "initial": 2.5330156764817,
    "final": 2.5331079519762265,
    "rowsRatio": 0.08641766729273727,
    "greedyRatio": 0.13572164421309357,
    "relativeHeadMAC": 0.5432088336463686,
    "vsARDenseHeadMAC": 0.5963257732290277,
    "meanB": 0.9127428217341381,
    "meanBound": 6.474411373683128,
    "meanK": 2.3332770169039065,
    "tokenAgreement": 0.623046875,
    "acceptBoth": 0.6538186610295175,
    "updates": 96,
    "pairs": 384,
    "tokens": 3072,
    "calls": 2310,
    "blocks": 594,
    "arCalls": 2104,
    "fullRows": 62370,
    "lazyRows": 5387,
    "greedyRows": 8462,
    "draftRows": 31698
  }
]
Run node experiments/team-far/R05/probe.js weights.json checkpoint.json input.txt. Actual result raw rows available. No actual GPU profile. With 27 vocab and dense Rademacher small eps, shortlist count can be small; extrapolation to d4096 and >100k vocab not automatic. Finite-difference variance scaling may require larger eps, increasing b. Joint Gumbel differs from R04 maximal joint target; variance comparison required on real task. All outputs reproduced relative to dense shared-Gumbel AR, not relative to independent RNG or maximal SD.

## Literature and scope (2026-11-24)
Original publications/abstracts searched: FedKSeed ICML24 https://proceedings.mlr.press/v235/qin24a.html; Sparse MeZO NeurIPS25 https://proceedings.neurips.cc/paper_files/paper/2025/hash/1e5c2efbddc02c1d971e2f19ccdb07d0-Abstract-Conference.html; Learned ZO ICML26 https://proceedings.mlr.press/v306/zhang26da.html; Infer-ZO May 2026 https://arxiv.org/abs/2605.28760 (shared base runtime, low-rank perturbations, very strong baseline); Ferret ICML25 https://proceedings.mlr.press/v267/shu25a.html; Coupling without Communication and Drafter-Invariant SD ISIT25 https://arxiv.org/abs/2408.07978 (shared-Gumbel decoding known); List-level coupling NeurIPS25 https://arxiv.org/abs/2506.05632; A* Sampling NeurIPS14 https://papers.neurips.cc/paper_files/paper/2014/hash/937debc749f041eb5700df7211ac795c-Abstract.html (Gumbel bound-based exact sampling predates); SpecRoll 2026 https://arxiv.org/abs/2608.04962; Global Resolution ICLR26 https://proceedings.iclr.cc/paper_files/paper/2026/hash/83995c2acef585ae0f0d647154cfdd85-Abstract-Conference.html. No exhaustive literature or theorem search, no formal plagiarism check. Specific perturbation shortlist expectation bound plus paired federated ZO/SD verifier not located in searched abstracts, but known bounded argmax/ZO/inference kernels substantially overlap. R04 addresses coupling frontier, R05 changes verifier compute instead; FCMC/FNSC margin/certificate for different target/version, not same paired signed perturbation; ordinary lazy AR can copy certificate.

## Evaluation
Subjective mechanism design target 8; actual preliminary rubric problem 8, mechanism 7.5, theory 8, practical value 7, literature distance 6.5, weights .25/.25/.20/.15/.15 = 7.5/10 (2+1.875+1.6+1.05+.975=7.5). No rating inflation to 8. G1 fail formal threshold; G2 open/critical due to paired lazy AR and same-information flat; G3 partial stronger exact paired greedy/stochastic target and actual path audits at FP64 without EOS/top-p production; G4 fail two checkpoints same tiny scale/proxy; G5/G6 unverified; G7 partial code/results/source. Retain candidate for next decisive pre-screen, not approved. 1–2x40GB two real untied head models (or separate untied final adapter), online public real tasks and 3 non-IID/latency setups, 3 training seeds. Compare dense shared-base paired AR, lazy paired AR (mandatory!), production self-SD with CRN, Infer-ZO low-rank, SparseMeZO/FedKSeed, same-info delegated flat. Measure actual head fraction and kernel overhead, perturbation distribution/epsilon vs reward noise and final held-out quality, source RNG/FP16 conservative interval, top-p fallback, full wall time with all costs. Need at least 15% total time or 20% effective work, independent causality on both HFL and SD. If lazy AR matches, no layer gain or actual head time fraction too low, reject. No 80% confidence yet.
