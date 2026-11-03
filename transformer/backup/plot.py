import csv, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
def rd(f):
    r=list(csv.DictReader(open(f))); g=lambda x: float(str(x).replace('d0',''))
    return [g(x['seconds']) for x in r],[g(x['val_subset']) for x in r]
fig,ax=plt.subplots(1,2,figsize=(13,5))
runs=[('../fast-gru/F-combined.csv','GRU-F 480s (100k params)','C0','-'),('../fast-gru/G-combined-30min.csv','GRU-G 1800s','C0','--'),
      ('T480.csv','Transformer 480s','C3','-'),('T1800.csv','Transformer 1800s','C3','--')]
for i,(lim,title) in enumerate([(480,'480 s budget'),(1800,'30 min budget')]):
    for f,l,c,ls in runs:
        if os.path.exists(f):
            t,v=rd(f); ax[i].plot(t,v,color=c,ls=ls,label=l,lw=1)
    ax[i].set_xlim(0,lim+10); ax[i].set_ylim(1.4,2.2); ax[i].set_xlabel('wall-clock seconds (8 threads)'); ax[i].set_ylabel('val loss (subset, nats/char)')
    ax[i].set_title(title); ax[i].grid(alpha=.3); ax[i].legend()
fig.suptitle('Lisp char models on Tiny Shakespeare, same fixed split. GRU val subset: 32x1000 streamed chars; transformer: first 32000 val targets, ctx 32-63')
plt.tight_layout(); plt.savefig('transformer-vs-gru.png',dpi=110)
