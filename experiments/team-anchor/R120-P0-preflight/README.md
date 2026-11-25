# R120-P0 minimal GPU preflight

This directory contains the exact commit snapshot, inputs, raw JSON result, stdout/stderr, exit codes and input hashes for the approved correctness-only preflight. No performance evaluation or training was run.

- Code commit: 3b291f1528dd513f08c5c7c7eaf4c7ae9b4fd170
- preflight.py SHA-256: b58fe5ccd5f5d8987dd8b2df88dbd138f63e9cfa8e96eb210d78a23f2779699b
- Runtime: Python 3.11.10, PyTorch 2.5.1+cu124, Transformers 4.48.3, RTX 3090 24GB
- Prompts: token ID 1 repeated to lengths 64 and 256, with no external text data
- Result: passed; both manual AR outputs match generate; all K=1/4/8 clean and first-token-corrupted tapes match the reference; injected first-token errors are rejected immediately.

See RUN_METADATA.json for machine-readable provenance, gpu_preflight.json for the complete raw result, and the *.log / *.exit_code.txt files for exact process records. SHA256SUMS.txt covers all files in this package.
