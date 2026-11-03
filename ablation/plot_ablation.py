import re,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as p
runs=[('s0-small-rnn','0: RNN, 270k train'),('s1-full-rnn','1: +full data'),('s2-cos','2: +cosine (dropped)'),('s3-drop','3: +dropout 0.1 (dropped)'),('s4-gru','4: GRU'),('n-gru-seed2','4: GRU seed 2'),('s5-tie','5: GRU tied (dropped)'),('b-flat','6: flat MLP 40k'),('b-flat-big','6: flat MLP 79k'),('b-wavenet','6: WaveNet MLP 82k')]
f,ax=p.subplots(1,2,figsize=(15,5.5));fin=[]
for t,l in runs:
  s=open(t+'.log').read()
  r=re.findall(r't=\s*([\d.]+)s.*val\(subset\)=([\d.]+)',s)
  ax[0].plot([float(a) for a,b in r],[float(b) for a,b in r],label=l,ls='--' if t.startswith('b-') else '-')
  fin.append((l,float(re.search(r'val\(full fixed val[^)]*\)=([\d.]+)',s).group(1))))
ax[0].set_ylim(1.6,2.4);ax[0].set_xlabel('seconds (1 thread)');ax[0].set_ylabel('val subset loss (nats/char)');ax[0].legend(fontsize=8);ax[0].set_title('Validation (subset) during training')
ax[1].barh([a for a,b in fin],[b for a,b in fin],color=['C%d'%i for i in range(len(fin))]);ax[1].set_xlim(1.6,2.15)
for i,(a,b) in enumerate(fin): ax[1].text(b+0.003,i,f'{b:.4f}',va='center',fontsize=8)
ax[1].invert_yaxis();ax[1].set_xlabel('final val loss, full fixed val (nats/char)');ax[1].set_title('Final val loss (8 min each)')
p.tight_layout();p.savefig('ablation.png',dpi=100)
