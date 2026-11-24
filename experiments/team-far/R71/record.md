# team-far/R71 — Rao-Blackwellized population draft gradients and held-out calibration failure
2026-11-25. Start main ab63e4c6048d41b70d113ef9c227b100f198de13, README b653ca735ae04d880e6e79a0ca470bfc555903e8 (3815 lines, 552775 JS chars); entire retrieved and structural keyword screen, R70 record and R69 probability table inspected, no exhaustive replay. Before write main ab63e4c6048d41b70d113ef9c227b100f198de13, README b653ca735ae04d880e6e79a0ca470bfc555903e8.
## T: conditional gradient and hierarchy
Fixed p_i and independent q_i(phi), positive support, away from tie, let r_i(v)=log(p_i(v)/q_i(v)), S_-i=sum_{j!=i}r_j(X_j), f_i(v)=grad log q_i(v). Group TV gradient is -sum_i E[1{S_-i+r_i(X_i)>0} f_i(X_i)]. Conditional integrate lane i: g_i(S)=-sum_v q_i(v)1{r_i(v)>-S} f_i(v). Unbiased with fixed current parameters and independent sampled other lanes; covariance PSD dominated by the corresponding single-lane score estimator via total covariance, not automatically the sum of all overlapping RB estimators compared to full group score. Random uniform lane multiplied by N gives unbiased full gradient; homogeneous case same selected lane repeated with N scaling, training uses normalized gradient /N. Softmax logit derivative q_i(v)(G_i-1{r_i(v)>-S}), G_i=sum_v q_i(v)1{...}; compute prefix sum of q-weighted scores sorted by r_i for repeated thresholds, O(K log K) preprocessing O(log K) scalar threshold lookup plus gradient vector/feature costs, not free for huge model weights. Edge can sum sampled local likelihood ratios L and broadcast L, clients subtract own r_i; for each lane estimate, no transfer of full logits. M draws requires M scalars and broadcasts per region per update; synchronized snapshots/independent proposals, actual communication/secure aggregation not measured; flat with same delegation reproduces. Classical Rao-Blackwell not claimed new. Exact product TV is #P-complete for general independent Bernoulli product inputs; no contradiction with unbiased MC gradient/online sampler.
## E: calibration only
48 historical K27,N32 model p/q rows from R69 (two same-size microGPT checkpoints, no new forward); 2 alternating folds of 4 train/4 heldout per wi and seed11/29/73 (12 folds, overlapping sources, no independent model seeds). Fixed p; same 3 scalar trainable logit features: centered log q/2, vowel flag (letter index), top-q token flag. Adam 100 updates, step .025, clamp +/-1.5; sample 24 independent other-lane contexts per training row/step for RB and single-lane Score; KL, TV, Hellinger deterministic. Equal update counts not matched compute. Ridge lambda=0,.1,.5 applied to normalized gradient (scale differs by objective; grid exploratory, not selection by heldout). 540 actual fits/54000 optimizer steps; 4096 MC overlap evaluations per row, common initial RNG seed; finite MC error and limited generalization. No FL updates or actual GPU. Variance audit 144 rows x1024 paired estimates; compare single-lane score vs conditional integration (not full-group score). Mean trace ratios N8 .028439, N32 .019589, N64 .016381; these are parameter-feature gradient variance ratios not time savings. Unregularized all methods worse on heldout than original; RB N32 .526756 vs initial .537454 and Hellinger .516416; RB N64 .384828 vs initial .400858. Ridge results in raw summary; no claimed universal benefit. No model quality changes as p fixed; no measured true speed. Reproduce node train.js ../R69/results.json.
## Literature verified public abstracts/pages
LayerSkip https://aclanthology.org/2024.acl-long.681/; Online Speculative Decoding https://proceedings.mlr.press/v235/liu24y.html; DistillSpec https://arxiv.org/abs/2310.08461; Variational SD https://arxiv.org/abs/2602.05774; Acceptance-Aware Draft Training https://arxiv.org/abs/2609.24150 (WTV Eq14 is sum of products of per-position overlaps, not overlap of joint product distributions); Rao-Blackwellized Stochastic Gradients https://proceedings.mlr.press/v97/liu19c.html; product TV #P-completeness https://arxiv.org/abs/2405.08255; HierFAVG https://arxiv.org/abs/1905.06641. No exhaustive full-text novelty or plagiarism check.
## Verdict
Retain conditional group gradient, reject assertion that lower variance implies improved held-out draft calibration. G1/G2/G4 not passed, G3/G7 partial, G5/G6 unverified. Historic full 8.14 unchanged, no justified 80% confidence. P0 collect larger true multi-context heterogeneous p/q data, measure actual regional fixed transaction cost before full system; compare same-capacity frozen-target LayerSkip training with KL/TV/Hellinger/WTV, full-score and optimized flat, measure compute/time/bytes, nonIID and multiple seeds. Need two pretrained scales 1–2x40GB, real wall-clock -15% or effective work +20% with CI. No actual physical HFL or full multi-step SD this round.
## Actual summary
```json
[
  {
    "N": 8,
    "method": "KL",
    "lambda": 0,
    "initial": 0.7499065274811746,
    "trainA": 0.7859868373091176,
    "testA": 0.7350269039951192,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "KL",
    "lambda": 0.1,
    "initial": 0.7499065274811746,
    "trainA": 0.7839031160533455,
    "testA": 0.7508659794574912,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "KL",
    "lambda": 0.5,
    "initial": 0.7499065274811746,
    "trainA": 0.7708888191281723,
    "testA": 0.755177840900071,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "TV",
    "lambda": 0,
    "initial": 0.7499065274811746,
    "trainA": 0.7812578674645527,
    "testA": 0.7334373578132793,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "TV",
    "lambda": 0.1,
    "initial": 0.7499065274811746,
    "trainA": 0.7846961189847802,
    "testA": 0.744293288393997,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "TV",
    "lambda": 0.5,
    "initial": 0.7499065274811746,
    "trainA": 0.7779714117469932,
    "testA": 0.7557837599517199,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "Hellinger",
    "lambda": 0,
    "initial": 0.7499065274811746,
    "trainA": 0.7865911996260794,
    "testA": 0.7357142142856393,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "Hellinger",
    "lambda": 0.1,
    "initial": 0.7499065274811746,
    "trainA": 0.7732364412220024,
    "testA": 0.7553608345479073,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "Hellinger",
    "lambda": 0.5,
    "initial": 0.7499065274811746,
    "trainA": 0.7588591954101824,
    "testA": 0.7527952952667568,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "Score",
    "lambda": 0,
    "initial": 0.7499065274811746,
    "trainA": 0.7835273220618779,
    "testA": 0.7446122641520062,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "Score",
    "lambda": 0.1,
    "initial": 0.7499065274811746,
    "trainA": 0.7797285278106499,
    "testA": 0.7518445804561288,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "Score",
    "lambda": 0.5,
    "initial": 0.7499065274811746,
    "trainA": 0.7669115919979475,
    "testA": 0.7515971611047664,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "RB",
    "lambda": 0,
    "initial": 0.7499065274811746,
    "trainA": 0.7881938825602618,
    "testA": 0.7400644192940997,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "RB",
    "lambda": 0.1,
    "initial": 0.7499065274811746,
    "trainA": 0.7815274868336054,
    "testA": 0.7547255777016999,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 8,
    "method": "RB",
    "lambda": 0.5,
    "initial": 0.7499065274811746,
    "trainA": 0.7653678593344532,
    "testA": 0.7564588602828638,
    "varRatio": 0.02843905275779493
  },
  {
    "N": 32,
    "method": "KL",
    "lambda": 0,
    "initial": 0.5374539943966293,
    "trainA": 0.5936896708168656,
    "testA": 0.5155426832276125,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "KL",
    "lambda": 0.1,
    "initial": 0.5374539943966293,
    "trainA": 0.5921771093877547,
    "testA": 0.5402346950376489,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "KL",
    "lambda": 0.5,
    "initial": 0.5374539943966293,
    "trainA": 0.5717937133918035,
    "testA": 0.5480917614526472,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "TV",
    "lambda": 0,
    "initial": 0.5374539943966293,
    "trainA": 0.5861814389880732,
    "testA": 0.5107256231767741,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "TV",
    "lambda": 0.1,
    "initial": 0.5374539943966293,
    "trainA": 0.5929642742353559,
    "testA": 0.5282824628419959,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "TV",
    "lambda": 0.5,
    "initial": 0.5374539943966293,
    "trainA": 0.5836075944046258,
    "testA": 0.5476713096904146,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "Hellinger",
    "lambda": 0,
    "initial": 0.5374539943966293,
    "trainA": 0.5947117513088104,
    "testA": 0.5164157495967432,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "Hellinger",
    "lambda": 0.1,
    "initial": 0.5374539943966293,
    "trainA": 0.5757429762707148,
    "testA": 0.5487503289512403,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "Hellinger",
    "lambda": 0.5,
    "initial": 0.5374539943966293,
    "trainA": 0.5513395419352061,
    "testA": 0.543182275007636,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "Score",
    "lambda": 0,
    "initial": 0.5374539943966293,
    "trainA": 0.581679065152451,
    "testA": 0.5163784766501996,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "Score",
    "lambda": 0.1,
    "initial": 0.5374539943966293,
    "trainA": 0.567277120533875,
    "testA": 0.5501626356716173,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "Score",
    "lambda": 0.5,
    "initial": 0.5374539943966293,
    "trainA": 0.554701862224036,
    "testA": 0.5494826263839966,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "RB",
    "lambda": 0,
    "initial": 0.5374539943966293,
    "trainA": 0.599545811810728,
    "testA": 0.5267564970463239,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "RB",
    "lambda": 0.1,
    "initial": 0.5374539943966293,
    "trainA": 0.5748796986642718,
    "testA": 0.5502552291482689,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 32,
    "method": "RB",
    "lambda": 0.5,
    "initial": 0.5374539943966293,
    "trainA": 0.5527739079923725,
    "testA": 0.5461861339235877,
    "varRatio": 0.019589107142170967
  },
  {
    "N": 64,
    "method": "KL",
    "lambda": 0,
    "initial": 0.40085837137445957,
    "trainA": 0.4623217829492816,
    "testA": 0.377419013270472,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "KL",
    "lambda": 0.1,
    "initial": 0.40085837137445957,
    "trainA": 0.4624596963738868,
    "testA": 0.406330057965357,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "KL",
    "lambda": 0.5,
    "initial": 0.40085837137445957,
    "trainA": 0.44057347164131455,
    "testA": 0.41510720358087744,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "TV",
    "lambda": 0,
    "initial": 0.40085837137445957,
    "trainA": 0.4534749898147871,
    "testA": 0.3724939348552671,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "TV",
    "lambda": 0.1,
    "initial": 0.40085837137445957,
    "trainA": 0.4646821417860167,
    "testA": 0.39254155636315113,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "TV",
    "lambda": 0.5,
    "initial": 0.40085837137445957,
    "trainA": 0.4546637159273314,
    "testA": 0.4141769465818718,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "Hellinger",
    "lambda": 0,
    "initial": 0.40085837137445957,
    "trainA": 0.4636977457268207,
    "testA": 0.37817076545538847,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "Hellinger",
    "lambda": 0.1,
    "initial": 0.40085837137445957,
    "trainA": 0.4448944685617742,
    "testA": 0.41583520414401903,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "Hellinger",
    "lambda": 0.5,
    "initial": 0.40085837137445957,
    "trainA": 0.4171448208243483,
    "testA": 0.40790508714658386,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "Score",
    "lambda": 0,
    "initial": 0.40085837137445957,
    "trainA": 0.4377127172728404,
    "testA": 0.3540335516414917,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "Score",
    "lambda": 0.1,
    "initial": 0.40085837137445957,
    "trainA": 0.41961629758314767,
    "testA": 0.4005447857091369,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "Score",
    "lambda": 0.5,
    "initial": 0.40085837137445957,
    "trainA": 0.4129140621334993,
    "testA": 0.4080599559233414,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "RB",
    "lambda": 0,
    "initial": 0.40085837137445957,
    "trainA": 0.47115517892319164,
    "testA": 0.38482805408743676,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "RB",
    "lambda": 0.1,
    "initial": 0.40085837137445957,
    "trainA": 0.44051873919054163,
    "testA": 0.4209845839668065,
    "varRatio": 0.016380535828140846
  },
  {
    "N": 64,
    "method": "RB",
    "lambda": 0.5,
    "initial": 0.40085837137445957,
    "trainA": 0.41196759110165604,
    "testA": 0.408560135952345,
    "varRatio": 0.016380535828140846
  }
]
```
