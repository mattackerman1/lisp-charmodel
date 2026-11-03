import csv, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g=lambda x: float(str(x).replace('d0',''))
fig,ax=plt.subplots(1,2,figsize=(14,5.2))
for tag,lab,c in [('lisp','Lisp (SBCL + OpenBLAS sgemm, 8 threads)','C0'),('torch','PyTorch 2.14 CPU (8 threads)','C3')]:
    if os.path.exists(tag+'.csv'):
        r=list(csv.DictReader(open(tag+'.csv'))); ax[0].plot([g(x['seconds']) for x in r],[g(x['val_subset']) for x in r],color=c,lw=1,label=lab+' val subset')
        ax[1].plot([g(x['step']) for x in r],[g(x['val_subset']) for x in r],color=c,lw=1,alpha=.6,label=lab+' val subset')
    if os.path.exists(tag+'-full.csv'):
        r=list(csv.DictReader(open(tag+'-full.csv')))
        ax[0].plot([g(x['train_seconds']) for x in r],[g(x['val_full']) for x in r],'o--',color=c,label=lab+' FULL val')
        ax[1].plot([g(x['step']) for x in r],[g(x['val_full']) for x in r],'o--',color=c,label=lab+' FULL val')
for a,xl in [(ax[0],'training seconds (eval excluded)'),(ax[1],'optimizer steps (2048 tokens each)')]:
    a.set_ylim(1.2,2.2); a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel(xl); a.set_ylabel('val loss (nats/char)')
fig.suptitle('Char transformer d128 L4 H4 ctx64 dropout 0.1, 9.58 MB Gutenberg corpus (val = last 10%). Subset = first 32k val targets.')
plt.tight_layout(); plt.savefig('transformer-compare.png',dpi=110)
