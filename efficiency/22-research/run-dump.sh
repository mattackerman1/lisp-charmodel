#!/bin/bash
# idempotent: resumes from kd16.u16.progress; skips if .DONE
cd /home/box/lisp-charmodel/efficiency; source hardening/env.sh
O=22-research/teacher/kd16.u16
[ -f $O.DONE ] && { echo already done; exit 0; }
exec 9>22-research/teacher/.lock; flock -n 9 || { echo running; exit 0; }
echo "===== segment $(date '+%F %T %Z')" >> 22-research/teacher/dump.log
nice -n 10 ionice -c3 $SBCL --dynamic-space-size 4000 --script 22-research/dump-teacher.lisp out=$O >> 22-research/teacher/dump.log 2>&1
echo "rc=$? $(date '+%F %T %Z')" >> 22-research/teacher/dump.log
