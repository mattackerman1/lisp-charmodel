import csv, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def rd(f):
    r=list(csv.DictReader(open(f))); g=lambda x: float(str(x).replace('d0',''))
    return [g(x['seconds']) for x in r],[g(x['val_subset']) for x in r]
fig,ax=plt.subplots(1,2,figsize=(14,5.2))
tr=[('../transformer/T1800.csv','T1800 dropout 0 (810k)','k'),('D1.csv','dropout 0.1','C0'),('D2.csv','dropout 0.2','C3')]
gr=[('../fast-gru/G-combined-30min.csv','GRU-G H=144 const lr (101k)','k'),('G288.csv','H=288 cosine','C0'),('G512.csv','H=512 cosine','C3'),('GD.csv','H=512 cosine + dropout 0.25','C2'),('GD288.csv','H=288 cosine + dropout 0.25','C1')]
for a,runs,title in [(ax[0],tr,'Transformer d=128 L=4 ctx64, 1800 s'),(ax[1],gr,'GRU, 1800 s')]:
    for f,l,c in runs:
        if os.path.exists(f):
            t,v=rd(f); a.plot(t,v,color=c,label=l,lw=1)
    a.set_ylim(1.35,1.9); a.set_xlim(0,1810); a.grid(alpha=.3); a.legend(); a.set_title(title)
    a.set_xlabel('wall-clock seconds (8 threads)'); a.set_ylabel('val loss on subset (nats/char)')
fig.suptitle('Same fixed split. Transformer subset: first 32000 val targets (ctx 32-63); GRU subset: 32 streams x 1000 val chars. Full-val numbers in REPORT.md')
plt.tight_layout(); plt.savefig('compare.png',dpi=110)
