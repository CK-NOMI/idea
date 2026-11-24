# team-far/R12 — DCA-SD: deferred complement adjoints
2026-11-24; starting main 3e5d8b3cb10cbdcc624789bb504ccb5568574077, prewrite 3e5d8b3cb10cbdcc624789bb504ccb5568574077, README blob 8151a380b06dc0d1a22a28a1aa2b586104bb61ad, 3248 lines/439740 chars. Full README fetched, R11 and related TJSA/FSRA/FBE/FVP and headings reviewed, not a line-by-line audit or independent rerun of other teams. Do not repeat factor alignment or low-rank model truncation. T = scoped mathematical result, E = actually executed external-checkpoint operator audit, P = communication proxy, H = unverified system. Not a claim of novelty of linear projection itself.

## Problem / mechanism
Fixed model snapshot with public prefix P on cloud and private suffix on edge; same model as ordinary autoregression, no low-rank parameter constraint or replacement model. A shallow internal head would draft candidate paths; standard exact target verification uses full model. For a head with public K in R^{m×d}, let B contain orthonormal row-space basis vectors (columns), r=rank K, Pi=BB^T. Edge knows B and sends B^T q on every candidate query; cloud computes all public logits K B (B^T q)/sqrt(d) and usual LSE and attention output. Edge retains q_perp=(I-Pi)q and computes suffix attention with original q. Combine via R10 headwise LSE. This exactness refers to real arithmetic, proper causal mask, fixed snapshot, legal sharing of public basis; no privacy claim, no change of checkpoint. Distinguish all evaluated candidate query ancestors (n_v) and the query nodes actually needed by the formal training objective (n_t), including parent of first rejected token if its distribution is used, bonus parent, necessary prompt ancestors; a rejected candidate is not automatically a zero-gradient node. Here n_v>=n_t assumes training nodes are subset of verification; prompt ancestors outside verification need separate traffic. For training losses on committed trajectories only, defer complement transmission until graph ancestors known. If the objective uses rejected candidates (draft distillation/auxiliary objectives), those may also require complements. For exact key gradient, receive q_perp when required, compute normal g,dV,dQ,dK, aggregate shared prefix VJP as R10. Retain gradients for all relevant parameters; projected q cannot be blindly reused for unconstrained dK. Basis changes with version, need rebuild and broadcast; floating-point compression tolerance is not mathematical equality. No actual SD run or distributed learning. Ordinary AR with same projection can compute identical forward and gradients; SD-specific opportunity is not sending complement of discarded descendants, compared to ordinary pass-Q SD, not irreducible synergy.

## T: first-order identifiability
Kq_perp=0 so exact public logits, and using unchanged private scores gives exact full softmax/output. Among fixed linear encoders Eq of q from which Kq can be recovered for every q, ker E subset ker K, so rank E >= rank K. Conversely basis projection attains rank K. This is for logits (including normalization), not an arbitrary output-only lower bound, and not a general discrete/interactive protocol bound. For each public key i, c_i=a_i (g^T v_i-g^T o)/sqrt(d), dK_i=c_i q, projected-only gradient missing c_i q_perp. If c_i !=0, full gradient uniquely determines q for fixed forward quantities and g; varying q_perp within ker K with private scores and g fixed, c_i unchanged. Thus a forward-only summary cannot in general preserve unrestricted key gradients. Explicit example d=2, public K=(1,0), public value 1, private key zero, private value 0, q=(0,t), g=1, both scores zero and output .5; public dK=(0,t/(4 sqrt2)), independent t in forward but derivative varies. This holds for K as free intermediate variables, not every parameterized upstream gradient map. If key updates constrained to row(K) at snapshot, complement can be irrelevant, but this changes learning or assumes fixed row space.
Multiple committed queries: Q_perp (n_t×(d-r)), C_{ji}=c_{ji}; missing block dK_perp=C^T Q_perp. Fixed C rank kappa, map image has dimension kappa(d-r). Any continuous one-shot real summary with exact recovery for arbitrary Q_perp and fixed C must have at least this dimension (restrict to image-isomorphic subspace, invariance of domain). If edge knows basis L of col(C), can send L^T Q_perp and cloud recover C^T L (L^T Q_perp), but cloud must send L (n_t*kappa scalars) or share/compute C; extra downlink, computation and round-trip may defeat savings. This is a conditional linear-algebra bound, not general communication lower bound; C can depend on q in general, so only applies to fixed side information and independent complement variations. No batch complement compression implementation in this round. No claim of bitwise greedy or sampling correctness; standard SD uses exact processed p,q and fresh independent randomness, residual (p-q)_+ on rejection; sequential conditionals preserve target if causal dependencies/version/temperature/top-p/tie/EOS correct. Production implementation not verified.

## E: externally trained microGPT operator experiment
Actual execution V8 JS FP64; reloaded fixed repo sanchayan7432/microGPT commit aa28d7c0aa66704ce56b263dfd3270579f3d5f6d, weights/checkpoint/input blobs [{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}]. Both same architecture, 7264 parameters (2 layers, d_model16, 4 heads, head dim4, char V27, learned pos context16), not two model scales. Adapted forward from team-hca/R11 (source repo record), training provenance not independently verified and names may overlap training. Filtered unique lowercase names length 12–14: 56, groups 13/27/16/0, so only 3 nonempty groups; not balanced HFL. All used, two checkpoints, layers, heads, six public lengths 2/3/4/5/8/12, 896 tests per length, 5376 head tests. For each query at index m, public first m rows, one private row, exact projected public score with twice Gram-Schmidt tolerance 1e-10 max(1,row norm); no SVD truncation or parameter change. Constructed deterministic sinusoidal output cotangent g, not task loss. Full gradient vs naive projected gradient and complement-corrected gradient, real forward activation computation; no finite difference here, no actual SGD, generation, real federated data or GPUs. Relative gradient error sqrt(sum squared diff/sum squared reference), not mean per-example percentage. Max score error 2.842170943040401e-14, output error 3.552713678800501e-15, corrected gradient error 2.220446049250313e-16. Results:
[
  {
    "m": 2,
    "n": 896,
    "avgRankK": 2,
    "avgRankV": 2,
    "fullK": 0,
    "naiveRelativeGradientError": 0.6800926618429097,
    "avgQuerySave": 0.5
  },
  {
    "m": 3,
    "n": 896,
    "avgRankK": 3,
    "avgRankV": 3,
    "fullK": 0,
    "naiveRelativeGradientError": 0.5685709883413669,
    "avgQuerySave": 0.25
  },
  {
    "m": 4,
    "n": 896,
    "avgRankK": 4,
    "avgRankV": 4,
    "fullK": 1,
    "naiveRelativeGradientError": 2.1913390384464752e-16,
    "avgQuerySave": 0
  },
  {
    "m": 5,
    "n": 896,
    "avgRankK": 4,
    "avgRankV": 4,
    "fullK": 1,
    "naiveRelativeGradientError": 1.9932993920901606e-16,
    "avgQuerySave": 0
  },
  {
    "m": 8,
    "n": 896,
    "avgRankK": 4,
    "avgRankV": 4,
    "fullK": 1,
    "naiveRelativeGradientError": 2.0765710415067735e-16,
    "avgQuerySave": 0
  },
  {
    "m": 12,
    "n": 896,
    "avgRankK": 4,
    "avgRankV": 4,
    "fullK": 1,
    "naiveRelativeGradientError": 2.1328134763146462e-16,
    "avgQuerySave": 0
  }
]
All observed K and V full rank when m>=4. No evidence natural exact low-rank structure on long contexts in this model. This is not a universal large-model rank claim. No actual network/timing.
Reproduce: download three fixed source files above, run node experiments/team-far/R12/probe.js weights.json checkpoint.json input.txt.

## P: communication
R10 equal width per head scalar bytes, full Q retained after forward, R10 elements Hq[2d(n_v+n_t)+n_v+2n_t] (both directions, includes LSE, backward g/D, partial dQ). DCA changes only Q: projected coords r*n_v plus missing (d-r)*n_t instead of d*n_v. Saving Hq(d-r)(n_v-n_t). Basis broadcast Hkv*d*r if per GQA KV-group basis sent once per edge/version, r<d, not including complement basis (may be deterministically constructed from B; numerical conventions required). No basis needed at r=d. Assume fixed one shared basis across queries and head rank uniform. This is optimistic: assumes all needed train nodes included in verification; complement uploaded with other backward data; counts neither extra computation, basis construction, memory, precision mismatch, metadata, weight sync, RTT, independent prompt queries, nor physical time. Equal nv=nt, zero saved. r=d, zero saved. Qwen-like Hq28/Hkv4/d128, nt128, nv=256,r=32 gives query-only reduction 37.5%, full interface reduction 12.4352%, after basis 11.8431%. Such low rank at long prefix is not shown; if r=32 because m=32, direct KV replication 4*Hkv*d*m scalars is a much stronger baseline (proxy net/rep 37.219x). In proxy results netToReplication sets m=r solely as short full-row-rank example; not general for low-rank long prefix. If KV already local or computed locally, network can be lower still. Byte reduction not time reduction. Rank(C) protocol above can be another strong baseline but overhead unmeasured.

## Literature audit (original pages opened/abstracts and relevant extracted text, not exhaustive)
- https://arxiv.org/abs/2411.01783 Context Parallelism (2024/2025), exact Pass-Q/Pass-KV.
- https://arxiv.org/abs/2606.01143 PrefixFlow (2026), shared prefix KV and delayed gradient accumulation/VJP.
- https://arxiv.org/abs/2611.02583 BasisKV (2026-11), value latent and communication compression with key routing.
- https://arxiv.org/abs/2512.05916 KQ-SVD (2025), query-key interaction matrix low-rank approximation.
- https://arxiv.org/abs/2601.21686 StiefAttention (2026), learned orthonormal bases minimizing layer reconstruction.
- https://arxiv.org/abs/2610.19026 JSSA (2026-10), learned shared Q/K subspace for sparse selection, not exact full attention.
- https://proceedings.mlr.press/v202/leviathan23a.html standard exact SD (2023).
No claim first discovery of row space projection, chain rule or dimension bound, no formal text similarity check. Related HC/TJSA and HCA/FSRA considered different target/aggregation problem; R12 does not compress learned parameters. Compare same target AR+projection+complement, same target standard SD+pass-Q and pass-KV, PrefixFlow, ordinary HFL+SD, same information flat/delegated, approximate KV compression quality-controlled. Neither formal training nor SD generation executed.

## Verdict
Score using R10/R11 weights .25/.25/.20/.15/.15: problem 8, mechanism 7, theory 8.5, practical value 6.5, literature distance 4.5 => 7.10/10. Subjective provisional theoretical assessment; +.025 vs 7.075 is not meaningful progress or external validation, and mechanism remains 7. G1 FAIL; G2 CRITICAL FAIL (SD affects which descendants need complement but ordinary AR same operator and strong SD+late transfer/flat can match, learning only changes standard K geometry); G3 partial (real-arithmetic operator proof and actual FP64 checks, production SD not implemented); G4 FAIL (two same-scale tiny checkpoints, no real training, GPU or three independent seeds); G5 unverified, G6 no basis for 80%, G7 partial (scripts and raw results). Reject as standalone, retain exact first-order identifiability and counterexample. Good result definition same target/quality/budget vs strongest baseline actual total time -15% or useful work +20%, two pretrained Transformer scales, three nonIID/delay scenarios three independent training seeds, CI. 1–2×40GB plan: real Qwen2.5-1.5B and 3B, collect real per-layer/head public KV spectra incl RoPE across shared contexts, actual verification and committed ancestor counts, phase timers, communication and quality. Stop if ranks full/short prefix replication dominates. Consider new direction focused on real verification-dependent learning objective that cannot be replicated by ordinary AR, with pre-registered factorial comparison and no claim that flat hierarchy alone is mathematically superior. No GPU executed.