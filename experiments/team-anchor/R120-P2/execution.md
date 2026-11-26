# R120-P2 actual diagnostic execution

Date: 2026-11-26 (Asia/Shanghai)
Source commit: 18d4a3ec7da6851ca632ff73cd2a3d92674a1fea

Command (working directory /root/rivermind-data/r120/repo):
```sh
/root/rivermind-data/r120/venv/bin/python research_code/R120-P0/diagnose.py --model /root/rivermind-data/r120/layerskip-1b --out /root/rivermind-data/r120/p2-results
```

Numeric diagnostic process exit code: 1 (exit_code.txt contains bytes 31 0a).
The run stopped at diagnose.py line 137: RuntimeError: wrong AR length.
Partial diagnosis.json reports completed=false and runs=[]. No completed diagnostic result is claimed. stdout.log is genuinely empty; stderr.log is the captured traceback. No code modification or additional experiment was performed after failure.
Environment and all six model file SHA-256 values were previously verified in environment-recovery-20261126/. Model weights are not included in GitHub.
