# team-far/R10 — HPA: hierarchical prefix adjoint service, GQA communication crossover
2026-11-24. Start/prewrite 01588e7c67c4aeda46331d2f97ecd0359fe11ab0, README blob d3b0ce926d9b38c0287c624742441c831122d0a7, 3231 lines. Entire README fetched, headings screened; reviewed R09, FVT, FVR, FVC, CPG, CGE, HieraCert R01, LEA, FRX, FSRA, FCRR, FOAS. Read exact archived source_metrics.json f0c7d555b4c42ad40f869e4dcc3251c7a5ecbdc4. No independent replay of all prior teams or GPU trace. User requests continued optimization toward 8.35, not grade inflation.

## H: new system candidate
Only if a long *identical authorized public prefix* is used by multiple private suffix jobs at the same model snapshot, split KV placement by deep layers. Edge retains shallow prefix KV for endogenous draft; cloud keeps deep public prefix KV and its trainable prefix graph, edge runs private suffix and deep target with a remote attention call. At a valid tree of candidate suffix queries, batch remote public-prefix attention, combine with local causal suffix attention, verify exact target. Record actual query descriptors only as permitted; discard invalid branches, retain committed loss ancestors. At backward send query adjoints referencing same descriptor ID; cloud accumulates public prefix KV cotangents across authorized edges and performs one prefix VJP at the fixed common snapshot. Direct local parameter gradients must be added separately and properly weighted. Model update invalidates prefix graph and descriptors depending on updated weights; do not reuse old logits. Region/local model can differ only on suffix-only parameter paths or else group by version and exact public prefix state. Not general multi-step FedAvg, no claim privacy (Q/g can leak suffix info), secure protocols separate. No cloud/edge training has actually been run here.

## T: exact interface
For one attention head, scale c=1/sqrt(d), public rows P and private causal rows S. For each query q, public/local logsumexp L_b and local normalized o_b. Global L=log(exp L_P+exp L_S), o=sum_b exp(L_b-L)o_b. Given full output cotangent g, D=g dot o, a_i=exp(c q dot k_i-L). Then dV_i=a_i g, dK_i=c a_i[g dot v_i-D] q, dQ_b=c sum_{i in b} a_i[g dot v_i-D] k_i. Cloud reconstructs its dK/dV and partial dQ given q, g, L, D and own K/V; sum local and remote dQ, sum all query contributions into shared K/V (also sum across query heads mapped to same GQA KV head). All loss weights applied once. For any fixed common differentiable prefix K/V=F_theta(P), sum_i J_F^T a_i=J_F^T sum_i a_i, exact prefix VJP. Public prefix losses, direct suffix derivatives, all other adjoints must be included independently. Finite precision order differs; no bitwise guarantee. Softmax/chain rule and prefix VJP are classical, not novel theorems. For fixed model, legal causal tree and independent target-first sampling (draft independent or deterministic, no future peeking) the partitioned attention exactly recovers target logits in real arithmetic, greedy uses same tie convention and sampling same processed p per step; rejected descendants not committed. Actual production SD/top-p/EOS/numerics not implemented. If all prefix rows locally accessible, regular local computation is a strong zero-network alternative. If private suffix/other model versions affect prefix or graph, cannot merge. Backward only on accepted ancestors, not all proposal nodes. Using already stored Q avoids resending Q in backward but this requires same forward snapshot and tape retention (and its storage cost); ordinary AR+same service can do it. This is not SD-specific exact gradient identity.

## T/P: communication crossover
Let m public tokens, n_v number of suffix queries during verification including rejected nodes, n_t committed query rows in gradient graph (include all private loss ancestors, e.g. private prompt tokens, in real measurement; current proxy optimistically counts output only). Dq=Hq*d, Dkv=Hkv*d. Per remote layer, one forward Q and normalized O_P vector plus Hq logsumexp; one backward G and Hq global L and D plus partial dQ. With equal scalar width s (idealized), B_remote=s[2Dq(n_v+n_t)+Hq(n_v+2n_t)]. Compare B_rep=4 s m Dkv for copying shared prefix KV once and returning dense aggregated dKV once per cohort/edge/version. Same prefix already resident or locally computed means replication bytes can be smaller or zero; activation memory, cache misses and prefix computation then dominate. If n_v=n_t=n, ignoring scalars remote cheaper iff m>n Dq/Dkv (GQA ratio). Note GQA strongly disfavors pass-Q when K/V head count small. Backward-only descriptor with already resident Q is accounted. If local independently trains full prefix, zero network for KV/dKV but repeated prefix compute; ordinary federated gradient transport counted separately. For shared prefix among G suffixes, n grows G while m fixed and one dense KV copy reused. Therefore long context and low cohort size needed, opposite naive intuition that more shared suffixes necessarily help the remote protocol. Byte-optimal choice min(B_rep,B_remote) per layer/edge with no coupled memory budget; memory capacity leads ordinary placement/knapsack. Remote calls add RTT per deep layer per verification block plus backward, draft shallow cache must be local; batching may reduce serial round trips but only physical trace can assess net benefit. Strong ordinary AR+pass-Q, pass-KV/cache, PrefixFlow with same ownership, context parallelism and same-info delegated flat mandatory. G2 unproven, no irreducible joint operation yet.

## E/P: actual numerical audit and trace-based sensitivity
Actual V8 CPU FP64, seed11/29/73 x public rows16/64/256 x per-client query counts1/4/16 =27 synthetic GQA operator cases, 4 query heads/2 KV heads/d8, 2 edges/2 clients, causal local suffix length t+1, arbitrary deterministic random Q/K/V/adjoints. Compare dense attention and split forward/backward, regional dKV aggregation and central finite differences (one Q/K/V coordinate on first query per client). This is an operator check, not real Transformer/HFL training. Max errors {"o":2.498001805406602e-16,"q":2.7755575615628914e-17,"k":6.938893903907228e-17,"v":1.3877787807814457e-16,"sk":7.806255641895632e-18,"sv":1.734723475976807e-17,"fd":3.4491489525612096e-12}. 
Read existing real third-party Qwen2.5-7B-Instruct greedy MTP2 logs via HIERASPEC/R05 source_metrics.json, original {"repo":"shreyansh26/Speculative-Decoding","ref":"76335be2f8f1a7bf25739310ae80435b83782b23"}, 100 matching records. Not independent GPU rerun. Real trace supplies prompt/output length, no public-prefix ownership or training fields. Official Qwen config hidden3584/Hq28/Hkv4/28 layers/head128. Proxy assumes every layer remote, equal 2-byte wire elements (including scalars; production may use FP32 LSE/adjoints), public prefix m=floor(prompt_tokens*share), share .25/.5/1, cohort G1/4/8 identical length same public prefix, nv=ceil(output_tokens*over)*G, nt=output_tokens*G, over1/1.5/2, no private prompt adjoints or draft traffic, one transfer per fixed version. This is optimistic in several ways; not actual shared public prefixes, network or speed. All 27 scenarios:
[
  {
    "share": 0.25,
    "cohort": 1,
    "over": 1,
    "n": 100,
    "remoteToRep": 4.792959114130739,
    "oracleSwitchToRep": 0.9993493598278859,
    "remoteWins": 1,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 98049078.4
  },
  {
    "share": 0.25,
    "cohort": 1,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 5.9892661891363455,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 122521810.88
  },
  {
    "share": 0.25,
    "cohort": 1,
    "over": 2,
    "n": 100,
    "remoteToRep": 7.184785312832876,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 146978424.32
  },
  {
    "share": 0.25,
    "cohort": 4,
    "over": 1,
    "n": 100,
    "remoteToRep": 19.171836456522957,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 392196313.6
  },
  {
    "share": 0.25,
    "cohort": 4,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 23.957064756545382,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 490087243.52
  },
  {
    "share": 0.25,
    "cohort": 4,
    "over": 2,
    "n": 100,
    "remoteToRep": 28.739141251331503,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 587913697.28
  },
  {
    "share": 0.25,
    "cohort": 8,
    "over": 1,
    "n": 100,
    "remoteToRep": 38.343672913045914,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 784392627.2
  },
  {
    "share": 0.25,
    "cohort": 8,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 47.914129513090764,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 980174487.04
  },
  {
    "share": 0.25,
    "cohort": 8,
    "over": 2,
    "n": 100,
    "remoteToRep": 57.478282502663006,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 9.754609375,
    "meanRepBytes": 20456898.56,
    "meanRemoteBytes": 1175827394.56
  },
  {
    "share": 0.5,
    "cohort": 1,
    "over": 1,
    "n": 100,
    "remoteToRep": 2.3942647581356598,
    "oracleSwitchToRep": 0.9913779598538102,
    "remoteWins": 2,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 98049078.4
  },
  {
    "share": 0.5,
    "cohort": 1,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 2.9918654889972554,
    "oracleSwitchToRep": 0.9939364349077212,
    "remoteWins": 2,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 122521810.88
  },
  {
    "share": 0.5,
    "cohort": 1,
    "over": 2,
    "n": 100,
    "remoteToRep": 3.5890726083120956,
    "oracleSwitchToRep": 0.9959912447139216,
    "remoteWins": 1,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 146978424.32
  },
  {
    "share": 0.5,
    "cohort": 4,
    "over": 1,
    "n": 100,
    "remoteToRep": 9.577059032542639,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 392196313.6
  },
  {
    "share": 0.5,
    "cohort": 4,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 11.967461955989021,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 490087243.52
  },
  {
    "share": 0.5,
    "cohort": 4,
    "over": 2,
    "n": 100,
    "remoteToRep": 14.356290433248382,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 587913697.28
  },
  {
    "share": 0.5,
    "cohort": 8,
    "over": 1,
    "n": 100,
    "remoteToRep": 19.154118065085278,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 784392627.2
  },
  {
    "share": 0.5,
    "cohort": 8,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 23.934923911978043,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 980174487.04
  },
  {
    "share": 0.5,
    "cohort": 8,
    "over": 2,
    "n": 100,
    "remoteToRep": 28.712580866496765,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 19.527265625,
    "meanRepBytes": 40951644.16,
    "meanRemoteBytes": 1175827394.56
  },
  {
    "share": 1,
    "cohort": 1,
    "over": 1,
    "n": 100,
    "remoteToRep": 1.1965961945910197,
    "oracleSwitchToRep": 0.9325503756770848,
    "remoteWins": 39,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 98049078.4
  },
  {
    "share": 1,
    "cohort": 1,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 1.4952627301126025,
    "oracleSwitchToRep": 0.9867032444083643,
    "remoteWins": 3,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 122521810.88
  },
  {
    "share": 1,
    "cohort": 1,
    "over": 2,
    "n": 100,
    "remoteToRep": 1.793732548008286,
    "oracleSwitchToRep": 0.9884737030414579,
    "remoteWins": 3,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 146978424.32
  },
  {
    "share": 1,
    "cohort": 4,
    "over": 1,
    "n": 100,
    "remoteToRep": 4.786384778364079,
    "oracleSwitchToRep": 0.9993362557036083,
    "remoteWins": 1,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 392196313.6
  },
  {
    "share": 1,
    "cohort": 4,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 5.98105092045041,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 490087243.52
  },
  {
    "share": 1,
    "cohort": 4,
    "over": 2,
    "n": 100,
    "remoteToRep": 7.174930192033144,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 587913697.28
  },
  {
    "share": 1,
    "cohort": 8,
    "over": 1,
    "n": 100,
    "remoteToRep": 9.572769556728158,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 784392627.2
  },
  {
    "share": 1,
    "cohort": 8,
    "over": 1.5,
    "n": 100,
    "remoteToRep": 11.96210184090082,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 980174487.04
  },
  {
    "share": 1,
    "cohort": 8,
    "over": 2,
    "n": 100,
    "remoteToRep": 14.349860384066288,
    "oracleSwitchToRep": 1,
    "remoteWins": 0,
    "meanResidentKV_MiB": 39.07203125,
    "meanRepBytes": 81939988.48,
    "meanRemoteBytes": 1175827394.56
  }
]
Run node experiments/team-far/R10/probe.js source_metrics.json using exact archived input. No real GPU, full model, actual network, resource profile, convergence, CI, or 80% probability. Existing logs are real greedy inference but MTP code sequential target forwards (CGE), not production batched verification. No extrapolation to actual HFL quality.
 
## Literature (actual public web lookups)
- [PrefixFlow](https://arxiv.org/abs/2606.01143) 2026 schedule-level training prefix KV and summed backward (direct major overlap).
- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783), MLSys 2025, exact pass-Q/pass-KV and crossover model (direct major overlap).
- [Ring Attention](https://arxiv.org/abs/2310.01889), ICLR 2024; [MagiAttention](https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn), distributed forward/backward partial dQ/dKV; [FlashInfer](https://arxiv.org/abs/2501.01005) 2025 cascade attention.
- [DroidSpeak](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_liu-yuhan_prepub.pdf) 2026 cross-finetuned-model prefix cache.
- [Aurora](https://proceedings.mlr.press/v306/wang26j.html) 2026 online SD training/serving.
- [DEL](https://arxiv.org/abs/2504.05598) 2025 dynamic self-speculation.
- [Heterogeneous vocabulary lossless SD](https://proceedings.mlr.press/v267/timor25a.html) 2025 (screened alternate direction).
- [Qwen official config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json).
Search abstracts/docs, not exhaustive proof-level novelty. Strong prior overlap undermines originality. Distinction from FVT/FVR: explicit *remote public-prefix attention forward/adjoint interface* and GQA communication crossover with same-cohort amortization; not a new attention algorithm.

## Decision
Provisional scores problem 8, mechanism 6.5, theory 7.5, practical value 7, literature distance 4.5 => 6.85 (weighted 25/25/20/15/15 = 6.85). No viable 8.35 candidate. G1/G2 fail, G3 partial real-arithmetic interface only, G4 fail, G5/G6 unverified, G7 partial. Reject as standalone top journal idea, retain exact interface and byte threshold, and real-log size anchors. To improve, first obtain actual authorized workload with public prefix repeated across regions and measured network/VRAM profile; compare ordinary AR+pass-Q, pass-KV, PrefixFlow, production self-SD, same-info flat; two real pretrained scales, 3 heterogeneity/latency settings, 3 training seeds, 1–2x40GB, matched quality -15% end-to-end or +20% useful work. If no shared prefix, no real exposed bottleneck, or strong baseline matches, abandon. No score inflation or claim of asynchronous work.
