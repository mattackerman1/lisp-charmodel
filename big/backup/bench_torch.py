# Short speed probe for gpt_torch.py variants (run only after the main runs). Steps/s over 150 timed steps after 30 warmup.
import sys, time, math, torch, torch.nn as nn, torch.nn.functional as F
th=int(sys.argv[1]); pd=float(sys.argv[2]); attn=sys.argv[3]; torch.set_num_threads(th); torch.manual_seed(0)
D,NL,NH,TT,BT,V=128,4,4,64,32,86
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1,s.ln2=nn.LayerNorm(D),nn.LayerNorm(D); s.qkv,s.proj,s.fc,s.fc2=nn.Linear(D,3*D),nn.Linear(D,D),nn.Linear(D,4*D),nn.Linear(4*D,D)
    def forward(s,x):
        B,T,_=x.shape; q,k,v=s.qkv(s.ln1(x)).split(D,2); q,k,v=[z.view(B,T,NH,D//NH).transpose(1,2) for z in (q,k,v)]
        if attn=='sdpa': y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=(0.0 if 'noattn' in sys.argv else pd) if s.training else 0.0)
        else:
            a=(q@k.transpose(-1,-2))/math.sqrt(D//NH); a=a.masked_fill(torch.ones(T,T,dtype=torch.bool).triu(1),float('-inf')).softmax(-1); y=F.dropout(a,pd,s.training)@v
        x=x+F.dropout(s.proj(y.transpose(1,2).reshape(B,T,D)),pd,s.training); return x+F.dropout(s.fc2(F.gelu(s.fc(s.ln2(x)),approximate='tanh')),pd,s.training)
class GPT(nn.Module):
    def __init__(s): super().__init__(); s.wte=nn.Parameter(torch.randn(V,D)*.02); s.wpe=nn.Parameter(torch.randn(TT,D)*.02); s.b=nn.ModuleList(Block() for _ in range(NL)); s.lnf=nn.LayerNorm(D)
    def forward(s,i):
        x=F.dropout(s.wte[i]+s.wpe[:i.shape[1]],pd,s.training)
        for b in s.b: x=b(x)
        return s.lnf(x)@s.wte.t()
m=GPT(); o=torch.optim.AdamW(m.parameters(),1e-3,betas=(0.9,0.99),weight_decay=0.1)
def step():
    i=torch.randint(0,V,(BT,TT+1)); l=F.cross_entropy(m(i[:,:-1]).reshape(-1,V),i[:,1:].reshape(-1)); o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); o.step()
for _ in range(30): step()
t=time.time()
for _ in range(150): step()
print(f"threads={th} dropout={pd} attn={attn}: {150/(time.time()-t):.2f} steps/s")
