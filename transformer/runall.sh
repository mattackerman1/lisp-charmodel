#!/bin/bash
cd /home/box/lisp-charmodel/transformer
FT=single nice -n 10 sbcl --script gpt.lisp --run secs=480 tag=T480 > T480.log 2>&1
FT=single nice -n 10 sbcl --script gpt.lisp --run secs=1800 tag=T1800 > T1800.log 2>&1
echo ALLDONE > runall.done
