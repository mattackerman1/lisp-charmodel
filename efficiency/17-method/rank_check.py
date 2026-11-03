# Step 4: do early (15-min) rankings agree with 1-h full-val rankings? Data copied from REPORT.md tables #1-#8.
from itertools import combinations
def kendall(a,b):
    ks=list(a); c=d=0
    for x,y in combinations(ks,2):
        s=(a[x]-a[y])*(b[x]-b[y]); c+=s>0; d+=s<0
    return (c-d)/(c+d), c, d
sets={
 "#6 size (15-min mid-run val1M vs 1h full)": ({"d256L6":1.0736,"d192L6":1.0629,"d256L8":1.0788,"d384L6":1.0982},{"d256L6":1.0729,"d192L6":1.0825,"d256L8":1.0729,"d384L6":1.0803}),
 "#7 vocab (15-min mid-run val1M vs 1h full)": ({"512":1.1030,"1024":1.0934,"2048":1.0736,"4096":1.0561,"8192":1.0434},{"512":1.0938,"1024":1.0871,"2048":1.0729,"4096":1.0656,"8192":1.0582}),
 "#8 vocab up (15-min mid-run val1M vs 1h full)": ({"8192":1.0434,"16384":1.0380,"32768":1.0459},{"8192":1.0582,"16384":1.0480,"32768":1.0564}),
 "#1-#4 decayed 15-min proxies vs 1h full (winner pairs)": ({"cos":1.1725,"wsd":1.1708,"muon":1.1317,"ctx64":1.1444,"bpe":1.0289},{"cos":1.1680,"wsd":1.1477,"muon":1.1419,"ctx64":1.1490,"bpe":1.0775}),
}
for k,(a,b) in sets.items():
    t,c,d=kendall(a,b); print(f"{k}: Kendall tau {t:+.2f} ({c} concordant / {d} discordant pairs; ties skipped)")
