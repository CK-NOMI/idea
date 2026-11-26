# R124-P0 inventory

2026-11-26, existing server, read-only checks. Instructions commit 42720faaae1bc3773977d961102c29ab817f4391. Repository main at archive: 8c9605fe050389376f30c2530805522e5d239fff. No installs, weight downloads, or model inference.

- RTX 3090 24 GiB; driver 580.119.02; nvidia-smi reports CUDA 13.0. GPU command exit 0.
- Data disk 47G available; df exit 0. System python3 3.10.12, exit 0.
- Existing R120 virtualenv metadata: torch 2.5.1+cu124; transformers 4.48.3; vllm None; huggingface-hub None as returned by the exact metadata name matching command. Exit 0. None means this matching query found no distribution; package naming normalization may affect the hub result.
- Hugging Face specified revision: HTTP 000, curl exit 28 (connection timeout), revision visibility not verified.
- ModelScope model page: HTTP 200, curl exit 0; page reachability only, model revision and weights not verified.
- All exact commands, raw sanitized stdout/stderr and command exits: commands.sh and inventory.log. Prior SSH failure retained in connection-failure-20261126.md.
- R124 probe.py was inspected, not run. vLLM/model runtime compatibility remains unverified.
