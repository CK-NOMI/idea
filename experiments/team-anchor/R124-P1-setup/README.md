# R124-P1 setup: stopped at isolated environment creation

Instruction commit 63f13c397aac7988220c91dd2ba5b7a1b124033b. Date 2026-11-26 Asia/Shanghai. Existing authorized server.

The initial check `test ! -e /root/rivermind-data/r124-p1` returned 0. Directory and logs/tmp creation succeeded. Disk check returned 0 and reported 47G available.

`python3 -m venv /root/rivermind-data/r124-p1/download-venv` returned numeric exit 1: system Python lacks ensurepip. Full stdout/stderr and numeric exit files are archived. The stage wrapper also returned 1. Setup stopped; no dependencies installed, model or dataset downloaded, CUDA smoke or inference run. Existing R120 environment was not modified.

The new base directory now exists and contains a partial venv. P1.md expressly requires stopping if it already exists, and forbids deleting it and installing system packages. Do not retry in place without updated instructions. A conda Python was observed in prior work but using it would depart from the exact specified command; no substitution was made.

SSH attempts before start twice stalled at password prompt; remote command exit unavailable for those attempts. The successful subsequent precondition check returned 0. No credentials are archived. Model access, checksums and MTP verification remain unexecuted.
