# New server recovery

Recovery completed 2026-11-26. Setup exit code: 0.
Source commit: 18d4a3ec7da6851ca632ff73cd2a3d92674a1fea.
Python: /root/rivermind-data/r120/venv/bin/python
Model: /root/rivermind-data/r120/layerskip-1b
Checkout: /root/rivermind-data/r120/repo

All six model files matched R120-P1 manifest SHA-256; CUDA available on RTX 3090. This verifies assets and imports, not a model forward or diagnostic run. No scientific experiment executed during this recovery.

restore.sh is an environment recovery script, not a change to research_code. It expects the server image to provide /opt/conda Python 3.11 and PyTorch 2.5.1+cu124. It downloads only model assets from the previously used ModelScope source. requirements-recovered.txt inventories the environment; it is not a tested portable lockfile.

Model weights are not stored in GitHub. Public download source and hashes are retained in restore.sh and setup-verification.json. Server logs: /root/rivermind-data/setup.stdout.log and setup.stderr.log.
