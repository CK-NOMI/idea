# First GPU pilot (1x RTX 3090, 24 GB)
Prepared after team-anchor/R117; source baseline commit 5e6e2fb081da88416243ba3d858fa1debfc8665c. R104 official_pilot.py remains unchanged. This pilot instead uses the Hugging Face integrated LayerSkip implementation in transformers 4.48.3. Scripts have been written and re-read from GitHub, but not executed on the remote instance: SSH client unavailable and DNS resolution failed in the chat execution environment. No GPU or CPU benchmark is claimed.

## CPU-only preparation
1. Confirm persistent data disk mount using `df -hT` / provider documentation. Set `DATA_ROOT` to that verified mount, not an assumed path. Clone the research repository on the persistent disk, then run commands from the repository root.
2. Request gated access to https://huggingface.co/facebook/layerskip-llama3.2-1B. If authentication is required, authenticate using a read-only HF token locally on the server; do not paste any token into chat or GitHub. The launcher sets HF_HOME to the persistent disk; set the same HF_HOME for any CLI login. Alternatively set HF_TOKEN securely only in the server environment.
3. `export DATA_ROOT=/verified/persistent/mount`
4. `bash experiments/team-anchor/gpu_pilot/launch.sh prepare`
5. Check `$DATA_ROOT/hfl_sd_gpu_pilot/prepared.json`, `prepare.log`, model revision and prompt hash. CPU readiness here verifies dependencies, tokenizer, config, model safetensors presence and dataset access; it is not an end-to-end model execution test. Do not enable GPU until this preparation succeeds. If preparation is interrupted, rerun it. Preserve data volume when changing instance mode.

## GPU-only pilot
From repo root, after user manually switches to GPU mode:
`bash experiments/team-anchor/gpu_pilot/launch.sh pilot 11`
Fails if CUDA unavailable. Offline local model snapshot, batch size 1, 32 output tokens, 2 repeats, 16 calibration and 16 held-out test articles; compare AR and 6 early-exit arms. Test selected arm is locked by calibration. Actual output IDs, greedy equality, elapsed time with CUDA synchronization, token throughput and peak allocated memory retained. 60-minute timeout limits the process only, not provider billing; set provider timed shutdown and manually stop instance after log collection. Result directory must not already exist; preserve failed runs before retrying.

P1 uses full dense teacher features to measure quality of anchor residual carry across B=2/4/8, comparing shallow, periodic sparse, raw carry, global and regional gamma selection, and dense reference. Logical regions use AG News class partitions; no real FL, no trained regional residual predictor, no physical network, no genuine sparse kernels or free-running anchor SD. Do not report query reduction as GPU acceleration; it is only an opportunity proxy. This pilot is a screening run, not G4/G5/G6 completion; full multi-seed/multi-model, actual causal sparse implementation, quality noninferiority, stochastic correctness, matched strong baselines and physical HFL still required.

Results live under `$DATA_ROOT/hfl_sd_gpu_pilot/run_seed11/` (latency.jsonl, latency_summary.json, anchor_rows.json, anchor_summary.json, meta.json). Do not automatically publish raw data or credentials. Inspect results and choose next experiment before allocating more GPU time.
