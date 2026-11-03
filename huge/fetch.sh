#!/bin/bash
# Download Project Gutenberg plain-text ebooks listed in ids.txt (polite: 1 request/s, skip already present).
cd /home/box/lisp-charmodel/huge
for n in $(cat ids.txt); do
  f=raw/pg$n.txt; [ -s $f ] && continue
  curl -sfL --max-time 60 -o $f "https://www.gutenberg.org/cache/epub/$n/pg$n.txt" || { rm -f $f; echo "fail $n"; }
  sleep 1
done
echo FETCH DONE
