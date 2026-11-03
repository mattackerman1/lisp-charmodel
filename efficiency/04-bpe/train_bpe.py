# Train BPE tokenizers on the TRAIN split only (first 90% of huge/corpus.txt, same boundary as the Lisp runs),
# report compression on val, and (for the chosen vocab) write token streams for the Lisp trainer.
import sys, json, time, numpy as np, os
os.environ.setdefault('RAYON_NUM_THREADS', '8')
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, Regex
txt = open('../../huge/corpus.txt', encoding='ascii').read()
nall = len(txt); nval = 9988894; vstart = nall - nval            # identical to the Lisp split (val chars 9,988,894)
assert vstart == 89900043, vstart
train, val = txt[:vstart], txt[vstart:]
alphabet = sorted(set(txt))
PAT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"   # GPT-2-style chunks; merges never cross chunk boundaries
def make(V):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(PAT), behavior='isolated')
    tr = trainers.BpeTrainer(vocab_size=V, initial_alphabet=alphabet, min_frequency=2, show_progress=False, special_tokens=[])
    chunks = (train[i:i + 1_000_000] for i in range(0, len(train), 1_000_000))
    tok.train_from_iterator(chunks, tr, length=len(train) // 1_000_000 + 1)
    return tok
def encode(tok, s):
    # encode in ~1 MB pieces cut at newlines (a chunk never spans a newline-delimited piece boundary safely: cut AFTER '\n')
    out, i = [], 0
    pieces = []
    while i < len(s):
        j = min(len(s), i + 1_000_000)
        if j < len(s):
            k = s.rfind('\n', i, j); j = k + 1 if k > i else j
        pieces.append(s[i:j]); i = j
    for e in tok.encode_batch(pieces): out.extend(e.ids)
    return out
mode = sys.argv[1]
if mode == 'survey':
    res = []
    sample = val[:2_000_000]
    for V in (512, 1024, 2048, 4096):
        t0 = time.time(); tok = make(V); ids = encode(tok, sample)
        assert ''.join(tok.id_to_token(i) for i in ids) == sample
        cpt = len(sample) / len(ids); emb = V * 256
        res.append(dict(V=tok.get_vocab_size(), chars_per_token=round(cpt, 3), emb_params=emb, train_s=round(time.time() - t0, 1)))
        print(res[-1], flush=True)
    json.dump(res, open('survey.json', 'w'), indent=1)
else:
    V = int(mode); tok = make(V); tok.save(f'bpe{V}.json')
    vocab = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
    with open(f'vocab{V}.hex', 'w') as f:
        for s in vocab: f.write(s.encode('ascii').hex() + '\n')
    tr = encode(tok, train); v1 = encode(tok, val[:1_000_000]); v2 = encode(tok, val[1_000_000:])   # exact 1M-char boundary
    for part, s in ((tr, train), (v1, val[:1_000_000]), (v2, val[1_000_000:])):
        assert sum(len(vocab[i]) for i in part) == len(s)
    assert ''.join(vocab[i] for i in v1) == val[:1_000_000]
    allids = np.array(tr + v1 + v2, dtype=np.uint16); allids.tofile(f'tokens{V}.u16')
    meta = dict(vocab=len(vocab), train_tokens=len(tr), val_tokens=len(v1) + len(v2), val1m_tokens=len(v1),
                train_chars=len(train), val_chars=len(val), chars_per_token_train=len(train) / len(tr),
                chars_per_token_val=len(val) / (len(v1) + len(v2)))
    json.dump(meta, open(f'meta{V}.json', 'w'), indent=1); print(meta)
