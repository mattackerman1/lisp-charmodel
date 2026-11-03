#!/bin/bash
# mark.sh RUNID keep|discard|control|info  -> sets the keep column in results.tsv
f=/home/box/lisp-charmodel/efficiency/17-method/results.tsv
awk -F'\t' -v OFS='\t' -v id="$1" -v k="$2" '$1==id{$15=k}1' $f > $f.tmp && mv $f.tmp $f
