# R120-P3 execution

Source commit: c68ff74547a721e3cf34d23d8dc53b84971bf68d
Executed 2026-11-26 (Asia/Shanghai).

Working directory: /root/rivermind-data/r120/repo
```sh
/root/rivermind-data/r120/venv/bin/python research_code/R120-P0/pilot_fp32.py --model /root/rivermind-data/r120/layerskip-1b --out /root/rivermind-data/r120/p3-results
```

Numeric exit code: 0, captured in exit_code.txt. stdout/stderr also include the successful git fetch/checkout output. An initial SSH session timed out during fetch; after verifying no pilot or fetch remained and no output directory existed, the fetch/checkout/launch was retried with detached logging. Only one pilot execution produced the archived results. Existing environment, model and repository data reused; no self-tests, research code changes or additional experiments.
