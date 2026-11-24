# team-far/R18 — FGQ-SD: federated Gram-quadratic output head and rank-one residual envelope
2026-11-24; start/prewrite 937dc7bec168122d9cbef140b1bda9f2f0a3b97d, README blob 0840e78020f1aa6491eca652592e4f5a962cc830, 3304 lines, 447955 chars. Full README obtained; R17 report reopened, targeted A/R07/A/R08/PRE09, FSGPA/FCRH/FSRA and other headings. Not a line-by-line proof audit or independent reproduction. New target head instead of coupling optimization; spherical softmax and Gram-based exact gradients already established, do not claim their novelty. New application is positive spectral envelope of the difference of two normalized squared amplitudes with shared W. T=conditional math; E=actual FP64 V8 pretrained micro-model features, fitted heads and one-step sampling; P=cost proxy; H=untested.
## Architecture and target
Frozen/shared row dictionary W (Vxr) plus trainable hidden output representation h or internal shallow s, potentially r<<d_model. Define G=W^T W, p_h(i)=(1-eps)(w_i^T h)^2/(h^T G h)+eps/V with eps=.01. q=p_s with same W. Nonzero quadratic normalizers assumed. This is a different formal model from the original softmax, not lossless conversion. Fit W by ridge to sqrt(original teacher probabilities), shared affine features x=[first r-1 hidden coords,1], fixed ridge .01. Region sufficient stats A_e=sum xx^T,B_e=sum x sqrt(p_teacher)^T; cloud (sum A+lambda I)^-1 sum B gives exact pooled ridge solution. Multiple client equal averaging of independently fitted W not equivalent. G does not average linearly: sum alpha G_e-G(sum alpha W_e)=sum alpha (W_e-Wbar)^T(W_e-Wbar) PSD. Inference G and all block summaries must be refreshed after W changes, or W fixed and only h adapters learn. Shard disjoint output rows then G=sum shard G; this is output/model parallelism, not evidence of HFL necessity. Sufficient stats contain private feature/teacher summaries; do not claim DP. Exact formal NLL gradient wrt h is 2(1-eps)/(Z p_y) [ (w_y^T h)^2 G h/Z -(w_y^T h)w_y ] where Z=h^TGh. Can use same G in training and verification without full logits, O(r²+r), W fixed. Existing spherical loss exact gradient method directly threatens originality, and ordinary AR can use same G. W update dense cost and refresh charged.
## T: residual envelope
Use normalized u=h/sqrt(h^TGh), v=s/sqrt(s^TGs), flip sign v if c=u^TGv<0 (prob unchanged). Assume G SPD or work on its quotient support. c in [0,1]. For c<1 define a=(u+v)/sqrt(2(1+c)), b=(u-v)/sqrt(2(1-c)); G-orthonormal. Then u u^T-v v^T=ell(ab^T+ba^T), ell=sqrt(1-c²). e=(a+b)/sqrt2, f=(a-b)/sqrt2, difference =ell(ee^T-ff^T), e^TGe=f^TGf=1. Thus for every token [p_h(i)-p_s(i)]_+ <= (1-eps)ell(w_i^Te)^2, envelope total M=(1-eps)ell<=1. If c=1 distributions identical, no correction required. Standard q draft, accept min(1,p_i/q_i), otherwise sample from rank-one proposal (w_i^T e)^2 (normalized), accept with positive residual/envelope, repeat. Conditional correction expected proposals M/TV(p,q) (if TV>0), unconditional expected number per attempted SD token TV*M/TV=M<=1. Exact real-arithmetic marginal p, conditional fresh RNG and legal causal prefixes => exact autoregressive distribution for this new target under temperature 1, fixed uniform mix. Full multi-token/bonus/EOS/top-p/other temperatures/production finite precision not implemented; top-p and greedy require correct full processed target, can fall back. Greedy branch-and-bound using subtree sum of squared scores as upper bound exact in real arithmetic with tie rule, worst full tree. No proof original softmax quality. Near c=1 numerical stability may require fallback to dense residual, implemented tiny threshold for experiments, no general precision guarantee.
## Shared tree
For disjoint token blocks B, G_B=sum_{i in B}w_iw_i^T. Parent sums exact. To draw squared amplitude for any vector e or s, descend weighted by x^TG_child x; leaf sample exact. Uniform component sampled separately for p or q. Use block size B, cost O(r² log(V/B)+Br) per draw, O((V/B)r²+Vr) memory, O(r²) normalization; optional path-only validation for drafted candidate, dense fallback for unsupported transforms. Greedy sum-bound may visit all nodes; no sublinear worst-case. With r=64,V=128000,B=256, padded binary tree <=1023 matrices (4-byte entries) ~16.76MB plus W32.768MB; ~9*4096+256*64=53248 multiply-add proxy vs dense 8192000, ignores common forward, memory access, preprocessing, kernel and batching. Actual micro experiment full single-row tree not blocked deployment; no GPU timing. For softmax hidden d=4096, comparing different widths is not quality matched. Ordinary AR+same squared tree and efficient exact spherical loss strong baselines, adaptive/hierarchical softmax, standard self-SD+same head dense residual. No evidence that HFL feedback changes SD beyond ordinary shared parameter learning or vice versa; same-info flat can sum stats. 
## E: actual execution
Source {"repo":"sanchayan7432/microGPT","ref":"aa28d7c0aa66704ce56b263dfd3270579f3d5f6d","files":[{"path":"weights.json","blob":"df323b7e714ceeb10ca070f9945a82c51b2c233c"},{"path":"checkpoint.json","blob":"7f0b4968c56290ebad706ab740de942c369f42a5"},{"path":"input.txt","blob":"736d88e18b70b292750a30e5cca6a92b0b333bd2"}],"prior":"experiments/team-far/R14/probe.js"}. Two same architecture 7264 parameter checkpoints, frozen 2-layer d16 4-head character Transformer V27. Upstream training not independently verified; possible overlap. 3 train splits seed11/29/73, 4 nonIID groups by name first letter, 64 names per group/train split, two positions/name =512 examples; heldout last16 per group, two positions =128 per model/rank/split. Fit ranks5/9/17 (includes bias), 18 actual ridge fits and local fits, 2304 evaluation contexts across configurations. Shallow/deep fixed feature outputs from R14 model; rank truncation not optimized. Fit centralized stats and independent local fits/average strong baselines. Evaluate original softmax NLL, new NLL, KL, shallow/deep TV, true residual envelope, gradient FD w.r.t feature input, exact tree greedy vs dense, 32 actual single-token SD draws per context (not multi-token); conditional residual tree sampler, pseudorandom LCG; aggregate empirical marginal TV not per-prefix distribution certification, no CI. Greedy wrong 0; max FD 1.2156069928437319e-8; max envelope violation 0; max norm/Gram error 3.170796958329447e-13; max pooled fit difference 1.701971896750365e-13; samples 73728. No backbone training or actual federated networks, no GPU/wall time, no realistic LLM two scales. 3 splits not 3 independently pretrained seeds. 
[
  {
    "r": 5,
    "n": 6,
    "gramGap": 0.032035227796680284,
    "trials": 4096,
    "rejects": 276,
    "attempts": 350.1666666666667,
    "treeVisits": 1643.3333333333333,
    "sampleMarginalTV": 0.029109740640400294,
    "nll": 2.852438798104226,
    "origNll": 2.6335226668494625,
    "avgNll": 2.8510642714011003,
    "kl": 0.18603837127405828,
    "origAccept": 0.9055417539618992,
    "accept": 0.9333190634543286,
    "residual": 0.06668093654567167,
    "envMass": 0.08477113839886384,
    "condProposals": 1.3386513055904297,
    "greedyVisits": 19.846354166666668
  },
  {
    "r": 9,
    "n": 6,
    "gramGap": 0.02459726700972197,
    "trials": 4096,
    "rejects": 242.33333333333334,
    "attempts": 324.6666666666667,
    "treeVisits": 1516.1666666666667,
    "sampleMarginalTV": 0.027497115923028625,
    "nll": 2.7565258934944357,
    "origNll": 2.6335226668494625,
    "avgNll": 2.756947434116679,
    "kl": 0.08950979721093925,
    "origAccept": 0.9055417539618992,
    "accept": 0.9407058721742029,
    "residual": 0.05929412782579679,
    "envMass": 0.08070500054328962,
    "condProposals": 1.4166410489753982,
    "greedyVisits": 18.203125
  },
  {
    "r": 17,
    "n": 6,
    "gramGap": 0.04798511519759844,
    "trials": 4096,
    "rejects": 341.3333333333333,
    "attempts": 487.3333333333333,
    "treeVisits": 2280.5,
    "sampleMarginalTV": 0.027547869421067572,
    "nll": 2.670666419242272,
    "origNll": 2.6335226668494625,
    "avgNll": 2.669174563265491,
    "kl": 0.01817385410698527,
    "origAccept": 0.9055417539618992,
    "accept": 0.9178287883151555,
    "residual": 0.08217121168484452,
    "envMass": 0.11624357520008892,
    "condProposals": 1.4488348670084648,
    "greedyVisits": 17.7734375
  }
]
Reproduce node experiments/team-far/R18/probe.js weights.json checkpoint.json input.txt. Code uses dense p/q for audit and accept probabilities, no actual output-head FLOP savings claimed. The tree was used for draft draws and residual proposals; exact acceptance from precomputed probabilities. Conditional probabilities could be computed by one row+G, but not benchmarked. No actual user-facing text generation.
## primary literature checked 2026-11-24 (original abstracts/publisher pages; no exhaustive full-text)
Vincent et al. Efficient Exact Gradient Update for Training Deep Networks with Very Large Sparse Targets NeurIPS2015 https://papers.nips.cc/paper_files/paper/2015/hash/7f5d04d189dfb634e6a85bb9d9adf21e-Abstract.html ; Exact gradient updates in time independent of output size for spherical loss family https://arxiv.org/abs/1606.08061 ; Exploration of Softmax Alternatives Belonging to Spherical Loss Family https://arxiv.org/abs/1511.05042 (reports worse language-model results); Adaptive softmax ICML2017 https://proceedings.mlr.press/v70/grave17a.html ; Self-organized Hierarchical Softmax https://arxiv.org/abs/1707.08588 ; Tang quantum-inspired recommendation length-square sampling https://arxiv.org/abs/1807.04271 ; Speculative Decoding with a Speculative Vocabulary https://arxiv.org/abs/2602.13836 (2026 output head cost); Recursive SD https://arxiv.org/abs/2402.14160 ; DICE https://arxiv.org/abs/2611.27891 (2026-11-21 author claims acceleration); A Theoretical Perspective for SD NeurIPS2024 https://papers.nips.cc/paper_files/paper/2024/file/e7349e785900b93d8b4971a3f2c1cefe-Paper-Conference.pdf . No first claim on squared head/length-square sampling; spectral envelope for same W normalized rank-two residual may be useful but search not exhaustive. Need compare nearest original proofs in detail. 
## Verdict
Provisional subjective 7.500/10 (8.5/7.5/8.5/6.5/5.5;25/25/20/15/15); G1/G2/G4 FAIL, G3/G7 partial, G5/G6 unverified, no80% evidence. Reject standalone for now; keep rank-one correction envelope and Gram-sharing architecture with explicit new-target distinction. Quality gate before large-scale work: actual 1.5B/3B teacher and same-budget distilled spherical/softmax/hierarchical heads, 3 nonIID/delay regimes 3 independent seeds, true sequence SD vs ordinary AR+same head, HFL+SD and same-info flat, real end-to-end speed/quality/CI. If quality loss or G2 persists, stop. No GPU performed.