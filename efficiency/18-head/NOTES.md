# #18 cheaper output head — resume notes (artifacts live in ../18-cheap-head/)
Method: sampled softmax w/ logQ correction, flag ssm=S (default 0), ssm-alpha=0.75; eval always exact full softmax. Backup: gpt-eff-pre-18.lisp.
Done:
- gradcheck: 18-cheap-head/gradcheck.txt (ssm 1.2e-3/3.9e-4 vs full 4.4e-4/3.6e-4, fp32 FD noise).
- probes (120 steps): 18-cheap-head/probes1.tsv. v16 full 8.8k tok/s, s1024 15.3k; v32 full 5.3k, s1024 13.2k.
- e18-default-400: val1m 1.2055, full 1.3236, step losses match -> bitwise identical. PASS.
- e18-proxy-base (v16 full, 2500 steps): full val 1.1253 (prior 1.1254/1.1248/1.1236).
Running (queue18a.sh, PID 523144, out 18-cheap-head/queue18a.out, prints QUEUE18A_DONE): e18-v16-s1024 (4341 steps), then e18-v32-s1024 (3752 steps). DO NOT relaunch.
Next: compare full val vs 1.1246±noise; if win -> 1h confirm vs #8 1.0480; REPORT.md #18 section; open-tasks.md line (added 08:25 CT).
- 09:38 CT: e18-v16-s1024 had a ~900s wall stall between step 1400-1800 (box/VM pause, coincided with connection drop). Its ledger tok/s is contaminated; use clean-segment rate (steps 200->1400: ~1.25k s... compute from log) + probes. Quality at fixed steps unaffected.
- 10:45 CT: queue done. full val v16-s1024 1.0924, v32-s1024 1.0902. REPORT #18 written (Unblocker). 1h confirm waits on Matthew.
