#!/bin/bash
# Waits for the queue to finish, then makes compare.png and REPORT.md.
cd /home/box/lisp-charmodel/huge
until grep -q "QUEUE DONE" queue.log 2>/dev/null; do sleep 60; done
python3 plot_compare.py >> queue.log 2>&1; python3 make_report.py >> queue.log 2>&1
echo "$(date '+%F %T %Z') FINALIZED" >> queue.log
