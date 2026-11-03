#!/bin/bash
# Sequential benchmark queue: old vs new for 300 s training time each (same config), then a 3600 s run of the new code.
# 8 threads, nice -n 10, one job at a time.
cd /home/box/lisp-charmodel/fast-gpt
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
chk() { if pgrep -f "science-team|[Ee]xperimenter" >/dev/null; then log "WARNING science/experimenter job running"; fi; }
COMMON="data=../big/corpus.txt dropout=0.1 threads=8 seed=1"
chk; log "start bench-old load=$(cut -d' ' -f1-3 /proc/loadavg)"
nice -n 10 sbcl --script gpt-old.lisp --run $COMMON secs=300 tag=bench-old > bench-old.log 2>&1
log "end bench-old"
chk; log "start bench-new load=$(cut -d' ' -f1-3 /proc/loadavg)"
nice -n 10 sbcl --script gpt-fast.lisp --run $COMMON secs=300 tag=bench-new > bench-new.log 2>&1
log "end bench-new"
chk; log "start fast-1h load=$(cut -d' ' -f1-3 /proc/loadavg)"
nice -n 10 sbcl --script gpt-fast.lisp --run $COMMON secs=3600 fullval-every=600 tag=fast > fast.log 2>&1
log "end fast-1h"
log "QUEUE DONE"
