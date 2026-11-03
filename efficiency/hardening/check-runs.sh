#!/bin/bash
# check-runs.sh -- idempotent watchdog: (re)launch supervise.sh, detached, for every registered run that is not DONE/STUCK
# and not currently running. Register a run: ln -s /abs/RUNDIR /home/box/lisp-charmodel/efficiency/hardening/runs.d/
# Nothing on this box re-executes after a VM restart (PID 1 is tini, no cron/systemd; home dotfiles are re-provisioned),
# so call this from a periodic routine (e.g. every 10-15 min) and/or at the start of any agent session.
H=/home/box/lisp-charmodel/efficiency/hardening; mkdir -p $H/runs.d
B=$(cat /proc/sys/kernel/random/boot_id); last=$(cat $H/.last_boot_id 2>/dev/null)
[ "$B" != "$last" ] && { echo "$(date '+%F %T %Z') new boot id $B (previous ${last:-none}) -- box restarted since last check" >> $H/check-runs.log; echo "$B" > $H/.last_boot_id; }
for r in $H/runs.d/*; do
  [ -e "$r" ] || continue; R=$(cd "$r" && pwd)
  [ -f "$R/DONE" ] && continue
  [ -f "$R/STUCK" ] && { echo "STUCK: $R ($(cat $R/STUCK))"; continue; }
  if ! flock -n "$R/.supervise.lock" true 2>/dev/null; then echo "running: $R"; continue; fi
  echo "$(date '+%F %T %Z') launching supervisor for $R" >> $H/check-runs.log; echo "launch: $R"
  setsid nohup $H/supervise.sh "$R" > /dev/null 2>&1 < /dev/null &
done
