# R124-P0 inventory: server access blocked

Date: 2026-11-26 Asia/Shanghai
Instructions: 42720faaae1bc3773977d961102c29ab817f4391, experiments/team-anchor/R123/NEXT.md.
Repository main at archive preparation: ff2959c9ae378930a10060f3bcb330fc06cc6919.
R124 probe.py seen on main: Git blob 54ad75f0215e098b94d4c131f84c9408c9643a21.

Status: INCOMPLETE. Existing server sc01-ssh.gpuhome.cc port 31416 was attempted twice using the prior session credentials. TCP/SSH reached a password prompt, but neither attempt returned authentication success, an SSH exit status, nor nvidia-smi output before the expect timeout. The expect wrapper reported local exit 0 due to an unmatched timeout handler; this is NOT a remote command exit code and does not establish success. No remote nvidia-smi exit code exists in the captured logs.

GPU, driver, disk, Python, torch, transformers, vLLM, huggingface-hub, and both model metadata endpoints: NOT VERIFIED. No other server was used. No packages installed, weights downloaded, or inference/training run. Old R120 environment not modified.

Sanitized captured attempts:
```text
Attempt 1: ssh -o ConnectTimeout=12 -p 31416 root@sc01-ssh.gpuhome.cc nvidia-smi
root@sc01-ssh.gpuhome.cc's password:
(no further output; local expect wrapper exit 0; remote exit unavailable)

Attempt 2: ssh -o ConnectTimeout=10 -p 31416 root@sc01-ssh.gpuhome.cc nvidia-smi
root@sc01-ssh.gpuhome.cc's password:
(no further output; local expect wrapper exit 0; remote exit unavailable)
```
Credentials omitted. Both attempts waited approximately 25–30 seconds. A successful connection is required to complete inventory.
