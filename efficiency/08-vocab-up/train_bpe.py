# #7: same tokenizer recipe as 04-bpe/train_bpe.py (train split only, GPT-2-style chunks, min_frequency=2),
# but streams the encoding into uint16 arrays (box has only ~2.5 GB free RAM). Usage: python train_bpe.py V
import sys, json, os, numpy as np
os.environ.setdefault('RAYON_NUM_THREADS', '8')
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, Regex
txt = open('../../huge/corpus.txt', encoding='ascii').read()
nall = len(txt); nval = 9988894; vstart = nall - nval
assert vstart == 89900043, vstart
train, val = txt[:vstart], txt[vstart:]
alphabet = sorted(set(txt))
PAT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
V = int(sys.argv[1])
tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.Split(Regex(PAT), behavior='isolated')
tr = trainers.BpeTrainer(vocab_size=V, initial_alphabet=alphabet, min_frequency=2, show_progress=False, special_tokens=[])
tok.train_from_iterator((train[i:i + 1_000_000] for i in range(0, len(train), 1_000_000)), tr, length=len(train) // 1_000_000 + 1)
tok.save(f'bpe{V}.json')
vocab = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
assert len(vocab) == V and V <= 65536
with open(f'vocab{V}.hex', 'w') as f:
    for s in vocab: f.write(s.encode('ascii').hex() + '\n')
lens = np.array([len(s) for s in vocab], dtype=np.int64)
def pieces(s):  # same ~1 MB newline-cut pieces as 04-bpe
    i = 0
    while i < len(s):
        j = min(len(s), i + 1_000_000)
        if j < len(s):
            k = s.rfind('\n', i, j); j = k + 1 if k > i else j
        yield s[i:j]; i = j
def encode(s):
    out, batch = [], []
    def flush():
        for e in tok.encode_batch(batch): out.append(np.array(e.ids, dtype=np.uint16))
        batch.clear()
    for p in pieces(s):
        batch.append(p)
        if len(batch) == 8: flush()
    if batch: flush()
    a = np.concatenate(out); assert lens[a].sum() == len(s); return a
a_tr = encode(train); a_v1 = encode(val[:1_000_000]); a_v2 = encode(val[1_000_000:])
assert ''.join(vocab[i] for i in a_v1) == val[:1_000_000]
np.concatenate([a_tr, a_v1, a_v2]).tofile(f'tokens{V}.u16')
meta = dict(vocab=V, train_tokens=len(a_tr), val_tokens=len(a_v1) + len(a_v2), val1m_tokens=len(a_v1),
            train_chars=len(train), val_chars=len(val), chars_per_token_train=len(train) / len(a_tr),
            chars_per_token_val=len(val) / (len(a_v1) + len(a_v2)))
import resource; meta["tokenizer_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
json.dump(meta, open(f'meta{V}.json', 'w'), indent=1); print(meta)
