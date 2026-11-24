# team-far/R03 — Marginalized verification acquisition (MVA), conditional dominance audit of VILA
Date 2026-11-24. Start main e5782a4ccbe673685e541ba6f76bfa0ab13ed3ce, README blob ff3f4886c5bafa1924201b3e2971ae245fc9c288, complete 3173-line README obtained, screened all headings and reviewed A/R05–06, FCIP, FCAS, FCAB, FSG, FAR/R02, G1–G7 and flat simulation obstruction; no exhaustive line-by-line independent replay. Prewrite e5782a4ccbe673685e541ba6f76bfa0ab13ed3ce, same content. Goal user requested preliminary innovation >7; not achieved by changing grade.

## T: exact conditional marginalization
B is full lawful prompt, committed Y, model snapshot and all needed score information; S=grad log P_theta(Y) is B-measurable. Z is verifier trace, with R independent of Z conditional on B (reward uses committed Y/prompt and independent private environment, or deterministic given B; not arbitrary shared randomness). At frozen parameters, acquisition indicator I~Bern(pi(B,Z)) uses independent coin before observing R, 0<pi<1. m is a fixed detached reward prediction. Previous VILA estimator G=[m+I(R-m)/pi]S. Define pbar=E[pi|B], mbar=E[(1-pi)m|B]/(1-pbar). Then E[G|B,R,I]=[mbar+I(R-mbar)/pbar]S, since Z|B,R,I=0 reweights by 1-pi, and Z|B,R,I=1 by pi. Thus one can sample I'~Bern(pbar) without sampling verifier trace Z, maintain the conditional law of (B,R,I), and use marginalized predictor mbar. Exact same expected query cost if label cost is B-measurable (not arbitrary Z-dependent). Total covariance is no larger in Loewner order by conditional expectation; conditional added query variance equals (1/pbar-1)(R-mbar)^2 ||S||². Equality if original G is measurable with respect to (B,R,I); if pi=1 identically use full label. Not a new general Rao–Blackwell theorem; specific closed form removes the hypothesized SD acquisition necessity. No guarantee cheaper to compute pbar/mbar or that full B cheaply available. If rewards depend on rejected branches, tool state or common randomness after conditioning, the premise fails; changing the reward contract must be explicit. If the entire formal reward is known, query allocation is unnecessary.
For single-token standard maximal coupling J=1{draft accepted}, Pr(J=1|Y=y)=min(p_y,q_y)/p_y. Thus for features (B,J), pbar=a*pi_A+(1-a)*pi_R and mbar=[a(1-pi_A)m_A+(1-a)(1-pi_R)m_R]/(1-pbar), a=min(1,q_y/p_y); no exponential path enumeration. Standard positive residual and same-version exact greedy/stochastic conditional induction remain required; no new decoder and no production multi-token/top-p/finite precision proof. For arbitrary full trace marginalization may be costly. This single-token case is precisely auditable; generic proof applies when conditional trace law available.
Proposed MVA: retain p and q on verified row, compile the two acquisition branches into a single query probability and predictor; avoid transmitting realized accept bit for acquisition, preserve formal target and improve or match variance at identical expected reward query count. Strong AR+same p/q scalar information can do exactly the same thing; same-info flat can perform all regional aggregation. Draft p/q not trained by this experiment; no proven G2. Cannot claim a novel HFL+SD fusion.

## E/P: actual external model experiment
Source microGPT fixed commit aa28d7c0aa66704ce56b263dfd3270579f3d5f6d; weights.json df323b7e714ceeb10ca070f9945a82c51b2c233c; checkpoint.json 7f0b4968c56290ebad706ab740de942c369f42a5; input.txt 736d88e18b70b292750a30e5cca6a92b0b333bd2. Forward adapted from team-hca/R09/probe.js (inherits team-hieracert/R11); not new pretrained model. 2 layers d16 vocab27, 7264 weights, two same-size checkpoints, no independent verification of training history or data contamination. Four logical clients by first-letter ranges, two regions (pairs). 3 deterministic hash splits seeds11/29/73, 64 calibration and 32 held-out names/client, four teacher-forced positions/name, 1024 calibration +512 test prefix instances/config, exhaustive 27 formal next tokens each. Gold next character yields binary proxy reward; deliberately NOT expensive real private reward, no full RL trajectories or real optimization. Conditional J integrated exactly, no Monte Carlo. Score norm for output-head gradient ||h||²||e_y-p||². p-only, p+q, p+q+J binned predictors (6 probability bins, pseudocount smoothing); calibration fit using full expected weights and known gold label; test budget multiplier calibrated with unlabeled p/q and features only, not test reward. Global/region/client scopes, budgets .1/.25/.5, same expected query budget and score. Conditional J policies use theoretical Pr(J|Y), not realized production traces. 2 checkpoints ×3 splits ×3 budgets ×3 scopes=54 rows, repeated contexts, not 54 independent training runs. Exact FP64 conditional means and variance; no measured time/CI/GPU/HFL parameter training. Source code node experiments/team-far/R03/probe.js weights.json checkpoint.json input.txt > results.json. Calibration and eval names may overlap source training. Rerun with same source. Results:
[
  {
    "budget": 0.1,
    "scope": "global",
    "p": 11.791000175108374,
    "pq": 11.745183989793105,
    "sd": 11.774028130020298,
    "rb": 11.763656916269753,
    "rbVsSD": 0.9991191448129717,
    "rbVsPQ": 1.0015728086075706,
    "sdVsPQ": 1.0024558270225703
  },
  {
    "budget": 0.1,
    "scope": "region",
    "p": 11.806109218458994,
    "pq": 11.798646866770534,
    "sd": 11.843797062110292,
    "rb": 11.828648958091755,
    "rbVsSD": 0.9987210094922178,
    "rbVsPQ": 1.002542841705494,
    "sdVsPQ": 1.0038267265602225
  },
  {
    "budget": 0.1,
    "scope": "client",
    "p": 11.87363390742325,
    "pq": 11.925303425036704,
    "sd": 11.988443114905834,
    "rb": 11.964131483228718,
    "rbVsSD": 0.99797207765478,
    "rbVsPQ": 1.0032559388057578,
    "sdVsPQ": 1.0052945981849462
  },
  {
    "budget": 0.25,
    "scope": "global",
    "p": 3.86815862834346,
    "pq": 3.8508564972031416,
    "sd": 3.862073850847142,
    "rb": 3.858115275620848,
    "rbVsSD": 0.9989750130683219,
    "rbVsPQ": 1.0018849776466556,
    "sdVsPQ": 1.0029129503143386
  },
  {
    "budget": 0.25,
    "scope": "region",
    "p": 3.87468956201881,
    "pq": 3.8717938095340823,
    "sd": 3.8890117956609846,
    "rb": 3.8832883233498614,
    "rbVsSD": 0.9985282964897384,
    "rbVsPQ": 1.0029687825285207,
    "sdVsPQ": 1.0044470307495466
  },
  {
    "budget": 0.25,
    "scope": "client",
    "p": 3.900860567631611,
    "pq": 3.9204196565419607,
    "sd": 3.9439369878876955,
    "rb": 3.934951626948489,
    "rbVsSD": 0.9977217280684754,
    "rbVsPQ": 1.0037067384820089,
    "sdVsPQ": 1.0059986770310396
  },
  {
    "budget": 0.5,
    "scope": "global",
    "p": 1.2272114460884884,
    "pq": 1.2194139996731643,
    "sd": 1.224755757789407,
    "rb": 1.222935803972544,
    "rbVsSD": 0.9985140271394617,
    "rbVsPQ": 1.0028881120770499,
    "sdVsPQ": 1.0043805943819526
  },
  {
    "budget": 0.5,
    "scope": "region",
    "p": 1.2308830098720651,
    "pq": 1.2295094571219292,
    "sd": 1.2374167068445356,
    "rb": 1.2348368090854587,
    "rbVsSD": 0.9979150938040462,
    "rbVsPQ": 1.0043329084885608,
    "sdVsPQ": 1.0064312231815735
  },
  {
    "budget": 0.5,
    "scope": "client",
    "p": 1.2432694543677354,
    "pq": 1.2521250670437225,
    "sd": 1.26243494554834,
    "rb": 1.2585625090284873,
    "rbVsSD": 0.9969325654890118,
    "rbVsPQ": 1.0051412132495388,
    "sdVsPQ": 1.0082339047240378
  }
]
Max conditional mean discrepancy 7.105427357601002e-15; max positive variance inequality violation 5.684341886080802e-14; max budget error 1.1102230246251565e-16. These are conditional additional gradient variance not total gradient variance, query latency or convergence. Regional budget .25: p-only 3.87468956201881, p+q 3.8717938095340823, realized-J 3.8890117956609846, marginalized 3.8832883233498614. Marginalized vs J ratio 0.9985282964897384, vs ordinary p+q 1.0029687825285207; don't interpret a small change as hardware improvement. Global scope lacks region conditioning, not identical-information flat, which can retain exactly same region labels. The two-stage calibration also uses fully labeled proxies for all outcomes, with uncounted calibration cost.

## Literature (original abstract/publication pages checked 2026-11-24)
- Active Statistical Inference ICML24 https://proceedings.mlr.press/v235/zrnic24a.html : prediction-corrected label acquisition already exists.
- Robust Sampling for Active Statistical Inference NeurIPS25 https://proceedings.nips.cc/paper_files/paper/2025/hash/6389470564214983604d1ac81631c2c5-Abstract-Conference.html : robust acquisition with imperfect uncertainty.
- Pro-Active Inference Oct 2026 https://arxiv.org/abs/2610.34681 : adaptive prediction-powered label selection.
- PILAF ICML25 https://proceedings.mlr.press/v267/feng25g.html : policy-aligned preference acquisition.
- Efficient Heterogeneity-Aware Federated Active Data Selection ICML25 https://proceedings.mlr.press/v267/tang25i.html : federated label minimization.
- Objective-Conditioned Active Preference Learning Nov 2026 https://arxiv.org/abs/2611.05506 : policy gradient coverage for queries.
- OnlineSPEC ICML26 https://proceedings.mlr.press/v306/qian26k.html and Aurora ICML26 https://proceedings.mlr.press/v306/wang26j.html : verification feedback and joint training-serving.
- Leviathan et al. ICML23 https://proceedings.mlr.press/v202/leviathan23a.html : exact SD.
No formal plagiarism test or exhaustive theorem search. Prior FCIP explicitly treats late binding of private learning values, FCAB conditional control variates, A/R05 proposal importance sampling. Here exact conditional collapse of acquisition and predictor with query coin, not identical previous formulas. Nonetheless classical RB + active statistical inference and strong flat substitution rule out >7 justified innovation on current evidence.
## Gates
Preliminary internal score problem 7.5, mechanism 5.5, theorem 7.5, value 6, literature distance 4.5, weighted .25/.25/.20/.15/.15 = 6.325/10. Not ≥7 or G1≥8. G2 CRITICAL: AR+same information can match and no formal training feedback to q shown. G3 partial theorem/one-step numerical, not complete production SD. G4 fail same-scale toy external model and proxy reward, not real HFL. G5/G6 unverified no true costs or 80% confidence; G7 partial code/source/results. T/E/P/H separated. 
Decision reject original VILA claim that verification randomness itself supplies irreducible learning information; retain MVA as audit/baseline, not candidate for standalone journal. Only reopen if real expensive reward is conditioned on additional lawful state/verification-dependent cost, or true measured p/q computation savings in deployment that cannot be matched by ordinary HFL+SD or equal-resource AR; quantify 15% end-to-end, two real pretrained scales, 3 heterogeneity setups, 3 independent seeds, CI. Do not repeat query-probability tuning as originality. No GPU run. 
