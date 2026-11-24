# team-far/R15 — UCR: union-inclusion counterfactual reward gradient
2026-11-24. Start/prewrite f79cf2c34f577cc87a8e88387cde462766b2db49; README blob dc6df9209d79391929aa6daaf39f2b1f6b0ddf5a, 3277 lines, 443858 chars. Entire README fetched; reviewed G1–G7, R14, A/R04/R12/R13, HC/R09 FQAC, FSP-DPO, FVT/FVR, FCTG, FAR R02/R03. Not a line-by-line audit or independent reproduction of all teams. FQAC already proves action-class coupling and independent environment randomness, FSP-DPO considers pair support; this round instead uses both distinct candidate and committed action rewards to estimate an additive one-state policy gradient, not an independent preference pair, not predicting reward or reusing old trajectories. T=conditional proof; E=actual external model probability evaluation and finite enumeration; O=oracle moment audit; P=proxy; H=untested system hypothesis.
## Mechanism and information boundary
Agentic RL has expensive private read-only/resettable simulation tools, independent randomness, identical initial state snapshot per query, no irreversible effects. For fixed state s and policy version, formal p(a), internal shallow q(a); single action X~q,Y~p coupled by ordinary maximal coupling Gamma, diagonal min(q,p), residual outer product normalized by TV. Commit only Y; do not condition validation on environment outcome/time/reward. If X=Y, use one tool result, otherwise retain both independent simulator results (discard draft state, never commit it). Goal J=sum_a p_a mu_a for terminal reward mean mu_a. For nonterminal action require independent target-policy continuation to estimate Q^p(s,a), which is not free, and state distribution under current formal policy, no general GRPO/PPO unbiasedness claimed. All policy probabilities known at authorized edge, local gradients aggregated with fixed weights; no data exchange claimed private or DP. Candidate generation internal head, not standalone draft model. Exact formal action and its outcome kernel under independent random seeds follows marginal of Gamma; for full token SD ordinary processed p/q, causal masks, bonus, EOS, independent random and exact precision contract needed, not implemented. For greedy use target argmax, this stochastic estimator is not greedy proof.
## T
Set I_a=1{X=a or Y=a}. pi_a=p_a+q_a-Gamma_aa; under maximal coupling pi_a=max(p_a,q_a). For each included a sample independent R_a from environment conditional on s,a, independent of inclusion. G=sum_a I_a (p_a/pi_a) R_a score_a with score=grad log p_a. Then E G=sum p_a mu_a score_a=grad J, no assumptions about action reward values except existence. Zero p/pi handled as zero. Use baseline b(s) independent of sampled action optionally since sum p score=0. Cost expected distinct tool queries C=2-tr Gamma. Under SD C=1+TV(p,q). For separate independent actions C=2-p dot q; AR single C=1; independent two p actions C=2-||p||². Estimator variance trace V=sum_a p_a² E[R_a²]||score_a||²/pi_a + sum_{a!=b}(Gamma_ab+Gamma_ba)(p_a p_b/(pi_a pi_b))mu_a mu_b <score_a,score_b> - ||grad J||² (ordered pair summation). Unbiasedness conditional on fixed state and policy extends to weighted HFL common snapshot, not multi-local-step equivalence. For direct action candidates, exact SD maximizes agreement/minimizes distinct calls; it also MINIMIZES each inclusion probability among all couplings with same marginals, so maximum acceptance may sacrifice counterfactual learning coverage. No monotonic variance ordering: cross terms depend on joint inclusion and score signs, reward. Counterexample p=q=(.5,.5), deterministic R=(1,0), logit score e_a-p: SD X=Y, gradient estimator variance trace=.125, C=1; forced antithetic Y=1-X, both rewards observed, exact gradient variance0, C=2. For independent Bern(.5) reward on both arms, SD variance=.25, antithetic variance=.0625 (C=2); no general fixed-time guarantee. Higher p/q similarity can eliminate rejected tool outcomes altogether; ordinary AR with extra q action and same coupling can reproduce estimator if same p/q, compute and sandbox access; independent or optimized action sampling, classical Horvitz-Thompson and multiple importance sampling are strong alternatives. If the choice of reuse depends on R (e.g. Bern(.5) only accept a good tool response, otherwise fresh sample), mean becomes .75; inclusion cannot be assumed ignorable. No real task implies simulator cloning cheap. Stronger estimator could use p-only formal sample, independent p/q deduplicated estimator, all-action expected PG, optimal variance-cost sampling; no claim fundamental novelty of HT. 
## E/O: actual microGPT probability audit
Source {"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}],"forwardSource":"experiments/team-far/R13/probe.js"}. Two same-scale 7264-param checkpoints; model code reused from R13. Source training history unverified; names may overlap pretraining. Four logical groups by first letter, last32 names per group, first3 letters+BOS, 2 checkpoints x4 groups x32=256 fixed contexts; one model forward per context, shallow/deep p,q. Each 27-action probability distribution fully enumerated under three artificial independent Bernoulli reward mean profiles: sparse .85 at true next char else .05; dense .1+.8(1-|a-y|/26), constant .5. 768 context/reward cases, 4 joint sampling schemes; exact finite-sum first/second moments for logit-space score, NOT actual reward execution/rollout or parameter update. Full gradient g=p*(mu-p dot mu). Standard AR, SD maximal, independent p/q, independent p/p; deduplicate same action in all schemes. Max bias 8.881784197001252e-16, max inclusion error 4.440892098500626e-16. Cost is expected count of distinct simulator queries, CV product only fixed-independent-context variance vs number-of-tools proxy (not wall time or convergence). Group means; winAr = fraction with strict per-context CV lower than AR. No actual GPU, network, training, physical tools or CI. Reproduction: node experiments/team-far/R15/probe.js weights.json checkpoint.json input.txt. 
[
  {
    "type": "sparse",
    "n": 256,
    "ar": {
      "variance": 0.10524588169434791,
      "cost": 1,
      "cv": 0.10524588169434784,
      "accept": 1,
      "winAr": 0
    },
    "sd": {
      "variance": 0.09645116933924108,
      "cost": 1.0958774295780114,
      "cv": 0.10518170224453911,
      "accept": 0.9041225704219875,
      "winAr": 0.3828125
    },
    "pq": {
      "variance": 0.05126670328262388,
      "cost": 1.86034948847252,
      "cv": 0.095225267171576,
      "accept": 0.1396505115274799,
      "winAr": 0.84375
    },
    "pp": {
      "variance": 0.05133905393708971,
      "cost": 1.8529683811648265,
      "cv": 0.09501076733864096,
      "accept": 0.14703161883517565,
      "winAr": 1
    }
  },
  {
    "type": "dense",
    "n": 256,
    "ar": {
      "variance": 0.5496709640691907,
      "cost": 1,
      "cv": 0.5496709640691907,
      "accept": 1,
      "winAr": 0
    },
    "sd": {
      "variance": 0.5007496152358488,
      "cost": 1.0958774295780114,
      "cv": 0.5479295993464703,
      "accept": 0.9041225704219875,
      "winAr": 0.546875
    },
    "pq": {
      "variance": 0.2653680952214243,
      "cost": 1.86034948847252,
      "cv": 0.49522207340328106,
      "accept": 0.1396505115274799,
      "winAr": 0.984375
    },
    "pp": {
      "variance": 0.26315079810236786,
      "cost": 1.8529683811648265,
      "cv": 0.4892507702673999,
      "accept": 0.14703161883517565,
      "winAr": 1
    }
  },
  {
    "type": "constant",
    "n": 256,
    "ar": {
      "variance": 0.42648419058241244,
      "cost": 1,
      "cv": 0.42648419058241244,
      "accept": 1,
      "winAr": 0
    },
    "sd": {
      "variance": 0.3904869933726303,
      "cost": 1.0958774295780114,
      "cv": 0.42729370247043447,
      "accept": 0.9041225704219875,
      "winAr": 0.4921875
    },
    "pq": {
      "variance": 0.21342304198603912,
      "cost": 1.86034948847252,
      "cv": 0.3981645586936121,
      "accept": 0.1396505115274799,
      "winAr": 0.98046875
    },
    "pp": {
      "variance": 0.21166318818477706,
      "cost": 1.8529683811648265,
      "cv": 0.3934073746443092,
      "accept": 0.14703161883517565,
      "winAr": 1
    }
  }
]
## Literature verified 2026-11-24
BunshinRL https://arxiv.org/abs/2611.10453 (2026 Nov, shared environment and speculative tools); Parallelizing Tool Execution and LLM Generation https://arxiv.org/abs/2603.18897; Atomix https://arxiv.org/abs/2602.14849 (transactional tools); System-Integrated SD for RL https://arxiv.org/abs/2604.26779 (author reports 8B rollout 1.8x; 235B 2.5x simulated); EfficientRollout https://arxiv.org/abs/2606.18967 (author reports up to 19.6% rollout and 12.7% end-to-end reduction vs accelerated AR); Cost-Aware Speculative Execution https://arxiv.org/abs/2606.07846; Expected Policy Gradients https://arxiv.org/abs/1801.03326; Transfer of Samples in Policy Search via Multiple Importance Sampling https://proceedings.mlr.press/v97/tirinzoni19a.html; Counterfactual Credit Assignment https://proceedings.mlr.press/v139/mesnard21a.html; On All-Action Policy Gradients https://openreview.net/forum?id=tCulsoUa6Oq. Reviewed original abstracts/search excerpts, not exhaustive. No text plagiarism report. Direct neighbor HC/R09 already addresses isolation and class coupling; HFL-specific advantage not established, ordinary AR+p/q with same authorized simulator and same-info flat can replicate. This is an incremental learning estimator, not yet a new neural architecture.
## Cost and next
Two tools in parallel require independent isolated workspaces, memory, CPU/GPU budget; full p/q probability, action completion, tool costs, discarded state, later counterfactual rollout, reward, training, parameter sync, privacy, network and contention must be counted. With frozen feature d and vocab V, output score computation O(Vd), extra actions can reuse same logits but require own environment, no free continuation. If p,q on separate workers, IDs, versions, propensities, scores and results need authorization; no quantified network evidence. Real pilot only with available resettable code-test or deterministic replay tasks with measured cost, 1–2x40GB, 1.5B/3B two pretrained scales, 8 clients 2 edges, 3 nonIID/delay regimes 3 seeds, causal 2x2 plus independent p/q, p/p, all-action/MIS, ordinary HFL+production SD, same-information flat. Measure actual total time and quality, CI; stop if independent sampling or ordinary SD+counterfactual reuse covers all benefit. No real simulator or GPU run.
## Verdict
Provisional subjective weighted score 7.225/10 (problem8, mechanism7, theory8, value7, literature distance5.5; 25/25/20/15/15). G1 below8, G2 FAIL (AR/flat alternatives), G3 partial single-action estimator and kernel, G4 FAIL, G5/G6 unverified (no 80% evidence), G7 partial code/raw data. Reject standalone, retain inclusion/cost/variance conflict and exact estimator for fair baseline. Good result quality-budget-matched strongest baseline real total time -15% or effective work +20%, all G. Historical R04 score in README 7.3 is higher than earlier reports' R12 7.10; prior score arithmetic should be recomputed consistently, no candidate passed G1–G7.