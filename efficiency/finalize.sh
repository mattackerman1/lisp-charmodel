#!/bin/bash
# Waits for the queue to finish, then regenerates REPORT.md and the charts. Also refreshes them every 10 min while waiting.
cd /home/box/lisp-charmodel/efficiency
until grep -q "QUEUE DONE" queue.log; do python3 make_report.py >/dev/null 2>&1; sleep 600; done
python3 make_report.py > finalize.log 2>&1; echo "$(date '+%F %T %Z') finalized" >> queue.log
