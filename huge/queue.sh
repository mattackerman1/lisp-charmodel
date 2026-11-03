#!/bin/bash
# Overnight queue (resumable: a job whose log already contains its sample is skipped; re-run this script after an interruption).
# One job at a time, 8 threads, nice -n 10. Lisp 3 h -> PyTorch 3 h -> old small Lisp model 1 h (reference).
cd /home/box/lisp-charmodel/huge
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
if [ -n "$(pgrep -f 'gpt-huge.lisp --run|gpt_torch.py data=corpus' | grep -v $$)" ]; then log "queue: a job is already running, exiting"; exit 0; fi
run() {  # name, log, command...
  local name=$1 lg=$2; shift 2
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return; fi
  if pgrep -f "science-team|[Ee]xperimenter" >/dev/null; then log "WARNING science/experimenter job running"; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nice -n 10 "$@" > "$lg" 2>&1
  log "end $name rc=$?"
}
BIG="data=corpus.txt d=256 layers=6 heads=8 ctx=128 batch=32 threads=8 dropout=0 seed=1 secs=10800 fullval-every=1800 fullval-targets=1000000"
run lisp-big  lisp-big.log  sbcl --dynamic-space-size 3000 --script gpt-huge.lisp --run $BIG tag=lisp-big
run torch-big torch-big.log /home/box/venvs/torch/bin/python gpt_torch.py $BIG tag=torch-big
run lisp-small lisp-small.log sbcl --dynamic-space-size 3000 --script gpt-huge.lisp --run data=corpus.txt threads=8 dropout=0.1 seed=1 secs=3600 fullval-every=1800 fullval-targets=1000000 tag=lisp-small
log "QUEUE DONE"
