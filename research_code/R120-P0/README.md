# R120-P0 / cached greedy Oracle interface preflight
Purpose: preserve original dense formal model and diagnose cache/position/rollback semantics before expensive performance tests. Oracle uses known AR answers, not a deployable predictor. Fixed-length greedy only, EOS disabled. No sampled correctness, HFL, performance claim or training.

Invariant: cache contains committed prefix except pending last token. Forward [pending]+draft gives K+1 next-token predictions. Accept maximal matching prefix of length a, emit it plus prediction at index a, keep old_cache_length+1+a, pending becomes last emitted token. Tail uses min(K, remaining-1). Perfect tape should match AR; an injected incorrect first candidate must reject. Any divergence stops and saves failure record.

## Actual local verification
Python syntax compilation passed. Python standard-library toy backend tests passed (2 tests, 2,835 candidate trajectories spanning three prompts, lengths 1–18, K=1/2/4/8/16, clean and single-token-corrupted tape; plus invalid-input tests). Mutable prefix-sensitive toy cache exercises rollback. Local source SHA-256 b58fe5ccd5f5d8987dd8b2df88dbd138f63e9cfa8e96eb210d78a23f2779699b; Git blob 11bec6aab7a15d8f346632651e1edaa31bdad54a. Local environment torch 2.10.0+cpu, no Transformers or CUDA. HF code reviewed but not executed locally; actual Llama KV API, FP16 equivalence, generation config and model load are unverified until real preflight.

## Execution approval
Only self-test and minimal GPU interface preflight, no throughput benchmark. Existing RTX 3090; verified LayerSkip 1B, single safetensors file with official SHA-256, Transformers 4.48.3. Use two distinct predetermined calibration documents, tokenize and truncate to exactly 64 and 256 IDs respectively, save JSON array [ids64,ids256], record source document IDs and tokenizer/config hashes. Do not choose based on outputs. Run from this directory:

    python preflight.py --self-test
    python preflight.py --approve-preflight --model /actual/verified/model --prompts /actual/prompts.json --out /actual/new/results.json

Use an existing compatible environment; no changes to source. Compare manual cached AR with model.generate and K=1/4/8 clean and corrupted tape on both prompts, each 16 generated tokens. Preserve all JSON, prompt IDs and provenance, exact commit, environment, stdout/stderr and exit status. If environment/model loading fails before JSON is written, preserve stderr. If a sequence mismatches, stop, do not repair code. Record any model/config hash difference. No timing conclusions, no formal performance matrix.

Return to ChatGPT for review and next code release. R119 mixed-memory and full R120 baseline/finite-length statistics remain unimplemented in this release.
