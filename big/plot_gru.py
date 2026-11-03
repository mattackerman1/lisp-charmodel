import csv, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g=lambda x: float(str(x).replace('d0',''))
fig,ax=plt.subplots(1,2,figsize=(14,5.2))
runs=[('gru-lisp','GRU Lisp','C0'),('gru-torch','GRU PyTorch','C3'),('lisp','Transformer Lisp','C2'),('torch','Transformer PyTorch','C1')]
for tag,lab,c in runs:
    if os.path.exists(tag+'-full.csv'):
        r=list(csv.DictReader(open(tag+'-full.csv')))
        ax[0].plot([g(x['train_seconds']) for x in r],[g(x['val_full']) for x in r],'o-',color=c,label=lab+' full val')
    if tag.startswith('gru') and os.path.exists(tag+'.csv'):
        r=list(csv.DictReader(open(tag+'.csv')))
        ax[1].plot([g(x['seconds']) for x in r],[g(x['val_subset']) for x in r],color=c,lw=.8,label=lab+' val subset (32x1000)')
        ax[1].plot([g(x['seconds']) for x in r],[g(x['train_batch']) for x in r],color=c,lw=.5,alpha=.35,label=lab+' train batch (dropout on)')
ax[0].set_title('Full fixed val (GRU: state carried over 32 streams; transformer: ctx 32-63)')
ax[1].set_title('GRU periodic subset val and training batch loss')
for a in ax: a.grid(alpha=.3); a.legend(fontsize=8); a.set_xlabel('training seconds (eval excluded), 8 threads'); a.set_ylabel('nats/char'); a.set_ylim(1.2,2.0)
fig.suptitle('9.58 MB Gutenberg corpus, val = last 10%. GRU H=288 dropout 0.25 vs transformer d128 L4 dropout 0.1, 3600 s each')
plt.tight_layout(); plt.savefig('gru-compare.png',dpi=110)
