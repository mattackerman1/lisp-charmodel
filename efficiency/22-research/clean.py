#!/usr/bin/env python3
# clean.py LOG -> clean steps/s, tok/s, chars/s (same method as #19 full/#20/#21: 200-step logged segments of train time t=,
# segments > 1.3x the median s/step dropped as box stalls). Resume segments: only consecutive step lines within one segment.
import sys,re
segs=[];prev=None
for l in open(sys.argv[1]):
    if l.startswith('====='): prev=None; continue
    m=re.match(r'step\s+(\d+)\s+t=\s*([\d.]+)s',l)
    if m:
        s,t=int(m.group(1)),float(m.group(2))
        if prev and s>prev[0]: segs.append((s-prev[0],t-prev[1]))
        prev=(s,t)
if not segs: print('na'); sys.exit()
per=sorted(dt/n for n,dt in segs); med=per[len(per)//2]
keep=[(n,dt) for n,dt in segs if dt/n<=1.3*med]
N=sum(n for n,_ in keep); T=sum(dt for _,dt in keep); sps=N/T
print(f"{sps:.3f} steps/s = {sps*4096:.0f} tok/s = {sps*4096*3.82458:.0f} chars/s over {N} unstalled steps ({len(segs)-len(keep)} of {len(segs)} segs dropped)")
