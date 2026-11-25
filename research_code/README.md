# 科研代码区 / Research code
HFL × self-speculative decoding. Created from main 9d4b1b515bca2f009ce82bec3fd71dc23c0cb6d5, README blob cc16f517d53ec8674abe5d2c85833803ceb4790f (4223 lines); latest README retrieved, R110/R115/R116/R117/R118/R119 summaries screened. Existing experiments/ remains historical evidence; do not rewrite.

## Roles and release protocol
ChatGPT owns mathematical specifications, experiment implementation, local tests, revision, GitHub releases, and independent data audit. Codex is the execution operator: fetch exact approved commit, verify source and inputs, provision the existing GPU environment, execute only approved commands, archive stdout/stderr, exit codes and raw outputs (including failures) under experiments/team-anchor/<round>/, and report result commit. Do not edit research_code/ or change algorithms, parameters, acceptance criteria, or test selection. Report blockers, do not improvise a fix. Never commit credentials, private server paths containing secrets, or model weights.

Each round starts by checking current README and related history, with SHA recorded; each release has local test evidence, declared missing tests, exact code hash and execution scope. Use a clean detached checkout at the approved commit. CPU tests do not establish real CUDA/HF compatibility. Real-model preflight is separate from full experiment approval. Do not infer GPU performance or paper novelty from toy tests. Never overwrite historical results. After experiment, append record to root README preserving existing prefix and verify remote commit.

## First release: R120-P0 preflight only
See R120-P0/README.md. The v2 review package is not approved for formal experiments. Do not run it. No full R120 benchmark, official LayerSkip comparison, sparse suffix timing, or training is authorized yet.
