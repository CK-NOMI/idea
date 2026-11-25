# team-far/R99 — Reusing shallow FFN activations in the formal deep FFN (SCP-SD backbone candidate)
2026-11-25. Start 92af7529345d0b6f390773127837dbba9b36a0e8, README caef4acee1bead24f2b57bcc3fc9518a930c1e1e (4056 lines,597105 chars), prewrite 92af7529345d0b6f390773127837dbba9b36a0e8, blob caef4acee1bead24f2b57bcc3fc9518a930c1e1e. Full README fetched/structurally screened, focused A/R08, R56/57/64/85/87/97/98, searched SwiGLU/shared gates; reopened R44 code and two public microGPT checkpoints/input. Not exhaustive replay.
## T/H
Shallow block already computes z_s=ReLU(U_s h_s), x its residual output. Frozen deep attention produces y0; original second FFN uses z_d=ReLU(U_d norm(y0)). Choose train-only fixed subsets S,D of shallow and deep channels. Train global deep down matrix M and regional shallow down B_e. New p=softmax(W(y0+M z_d[D]+B_e z_s[S])), q=softmax(W(x+B_e z_s[S])). One B_e z_s[S] per identical legal prefix/version, use same result for q/p; no independent drafter. The formal target changes; ordinary AR can use it too. For loss L(p,q), dL/dB_e=(adj_p+adj_q) z_s[S]^T, exact sum before outer product, fixed feature and snapshot; full-network VJP not tested. Sparse deep FFN can omit unselected up/down rows/cols by design, not exact old teacher. With d hidden, m32 retained-column comparison, k shared shallow channels, active deep cost 2d(32-k), shared projection d k, vs 2d*32 deep-only, saves k/(2*32) of these MACs if z_s exists and cached, k8/16 12.5/25%. Both formal-only and joint same architecture work; ordinary sparse FFN + own q head is strong comparison. Relative full original FFN or entire model savings different; include shallow, attention, vocab, rejected rows, cache, actual hardware. No G2 inherent from shared linear algebra; flat with same regional delegation can reproduce.
## E/P
Two same-size public 7264-parameter microGPT checkpoints, 3 hash splits, 2 artificial client layouts, frozen real first block and second attention, cached full FFN activations for fitting/audit, ReLU not SwiGLU. Train-only feature ranks, 32 train/8 test names per client, 4 clients 2 logical edges, 40 synchronous steps, 8 samples/client/step, CE half hard/half teacher, beta .3 single-token overlap, exact heldout two-token EAL, no generated trajectories or actual sparse kernel. Global deep down coefficients averaged across all clients; shared shallow coefficients by edge; shuffled same capacity, global one coefficient set. deep 32 retained columns no q correction, separate q-only correction with 32-k formal columns, formal p-only shallow correction (no overlap gradient), joint p/q share same shallow correction and overlap gradient, global/shuffled same joint objective. Not matched total model storage vs global, deep-only and separate effective formal capacities differ; 32 columns per local path max; no full model training, physical HFL, GPU, wall time, CI. Formal target changed, not lossless teacher. All features dense offline, sparse counts only theoretical. Independent weights repeated seeds not independently pretrained.
Actual fits 264, client gradient evaluations 23040, label occurrences 184320, real feature calls 12288, 576 heldout client rows, 96 finite-difference probes max abs 3.4254651198883934e-11, shared correction recomputation difference 0. Public source https://github.com/sanchayan7432/microGPT/tree/aa28d7c0aa66704ce56b263dfd3270579f3d5f6d, weights blob df323b7e714ceeb10ca070f9945a82c51b2c233c, checkpoint 7f0b4968c56290ebad706ab740de942c369f42a5, corpus 736d88e18b70b292750a30e5cca6a92b0b333bd2. Train-only ranks based on deep activation energy times down-column norm and shallow activation energy. Deep coefficients same across regional copies after global update. FD checks only joint loss, formal/separate objectives differ by design; no independent gradient replay. Teacher NLL/EAL 2.5436663569052533/1.7054574380319325. 
|k|arm|NLL|teacher KL|two-step EAL|one-step acceptance|
|---|---|---|---|---|---|
|8|deep|2.543536|0.000010|1.705463|0.900888|
|8|separate|2.543536|0.000010|1.705463|0.900888|
|8|formal|2.543536|0.000010|1.705463|0.900888|
|8|joint|2.543536|0.000010|1.705463|0.900888|
|8|shuffled|2.543536|0.000010|1.705463|0.900888|
|8|global|2.543536|0.000010|1.705463|0.900888|
|16|deep|2.543536|0.000010|1.705463|0.900888|
|16|separate|2.543536|0.000010|1.705463|0.900888|
|16|formal|2.543536|0.000010|1.705463|0.900888|
|16|joint|2.543536|0.000010|1.705463|0.900888|
|16|shuffled|2.543536|0.000010|1.705463|0.900888|
|16|global|2.543536|0.000010|1.705463|0.900888|
Evidence only small real checkpoint feature-based fit, not new pretrained scales, actual SD generation, real sparse execution or GPU latency. EAL for changed p does not mean teacher quality preserved. Compare same-quality actual strong sparse/early-exit baselines before claim. No new external literature review in this turn; previous A/R08/ReLU compilation, R56 gate, R57 writeback, R98 shared coordinates, MoD/LayerSkip are obvious near neighbors, external originality still unknown. Do not claim G1. 
## Verdict
Retain testable backbone parameterization and results, no independent立项 or score increase. Historical complete 8.14 unchanged; G1/G2/G4 fail, G3/G7 partial, G5 unverified, G6 insufficient. P0 actual 1B/7B LayerSkip checkpoints under 1–2×40GB, benchmark fixed channel pruning vs dense, shared shallow FFN bridge vs same budget separate/ordinary sparse+LayerSkip, region alignment vs equal-copy shuffle and delegated flat; measure matched task quality, actual sparse kernels, all inference/training/communication costs, three nonIID/latency/independent seeds CI. R98 pilot not executed. No 80% evidence.


## Post-audit: critical degeneracy and non-pretrained activation stress
Reopened source microgpt2.py and model_runner.py to check ordering: wte,wpe,lm_head, per-layer q/k/v/o/fc1/fc2; original ReLU. Re-ran exact training-feature extraction for all 12 checkpoint/split/layout combinations. At least one |activation|>1e-10 in only 0–1/64 shallow and 0–2/64 deep channels across each training bank; shallow/deep activations extremely sparse, many all zero. This explains near-identical E1 metrics and invalidates interpreting near-preservation as a meaningful compression success. It does not establish all prompts or source-model execution have dead FFN; our JS evaluation was not cross-checked against the original Python on same inputs, and R44 reused similar evaluator, so audit upstream feature quality before relying on older FFN-related claims. Original E1 retained unchanged. As diagnostic stress ONLY, change both pretrained ReLUs to leaky ReLU slope .1 with the same weights (different untrained model, not a pretrained checkpoint result). Re-execute same training protocol, teacher is altered model, not original. Actual 12288 feature calls, 23040 client gradient evaluations, FD max 5.486986975888897e-10; 64/64 active channels first case. No GPU, no real sparse kernel or time. 
|k|arm|NLL|teacher KL|EAL|
|---|---|---|---|---|
|8|deep|52.987802|34.919421|0.000407|
|8|separate|55.031406|32.021869|0.000090|
|8|formal|58.042674|32.979225|0.040824|
|8|joint|51.084661|28.494353|0.000123|
|8|shuffled|50.846949|31.090336|0.000041|
|8|global|48.908880|31.379061|0.000082|
|16|deep|52.987802|34.919421|0.000407|
|16|separate|35.716415|15.647111|0.000093|
|16|formal|39.256981|17.015763|0.000232|
|16|joint|41.576834|16.973806|0.000043|
|16|shuffled|31.979090|16.511054|0.004564|
|16|global|30.914592|18.331212|0.054989|
Stress test is not independent replication, not new external literature review or evidence of original model quality; its purpose is to expose nondegenerate tradeoffs. Do not increase score or claim G2. Source check is limited to source ordering, not full parity. 
