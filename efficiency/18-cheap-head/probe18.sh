#!/bin/bash
# throughput probe: 120 steps, no final eval. usage: probe18.sh label V16|V32 [extra args]
cd /home/box/lisp-charmodel/efficiency; source hardening/env.sh; source 18-cheap-head/common18.sh
lab=$1; vv=$2; shift 2; args=${!vv}; cpt=CPT${vv#V}; cpt=${!cpt}
mkdir -p 18-cheap-head/probes
nice -n 10 $SBCL --dynamic-space-size 6000 --script gpt-eff.lisp --run $args $SCH steps=120 seed=1 final=0 "$@" tag=18-cheap-head/probes/$lab > 18-cheap-head/probes/$lab.log 2>&1
tok=$(grep -o '[0-9.]* tokens/s' 18-cheap-head/probes/$lab.log | cut -d' ' -f1)
awk -v l="$lab" -v v="$vv" -v a="$*" -v t="$tok" -v c="$cpt" 'BEGIN{printf "%s\t%s\t%s\ttok/s=%d\tchars/s=%.0f\n",l,v,a,t,t*c}'
