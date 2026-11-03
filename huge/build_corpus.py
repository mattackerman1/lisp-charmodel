# Build huge/corpus.txt from Project Gutenberg plain-text ebooks (public domain in the US), ids from ids.txt (raw/pg<id>.txt).
# Same cleaning as big/build_corpus.py: keep only text between the *** START OF / *** END OF markers, curly quotes -> straight,
# dashes -> '-'/'--', accents removed via NFKD, CRLF -> LF, 4+ newlines -> 3. Vocabulary is then restricted to
# printable ASCII (32..126) + '\n' (tab -> space, any other control char dropped) => at most 96 symbols.
# Only books whose header says "Language: English" are kept; duplicates skipped. Order: the 14 big/ books first (same order),
# then the remaining books shuffled with a fixed seed (so the last-10% validation split is a mix of authors), stopping at MAXCH.
import re, unicodedata, json, random, os
MAXCH = 100_000_000
ids = []
for t in open('ids.txt').read().split():
    if int(t) not in ids: ids.append(int(t))
first, rest = ids[:14], ids[14:]; random.Random(1).shuffle(rest)
MAP = {'\u2018':"'",'\u2019':"'",'\u201c':'"','\u201d':'"','\u2014':'--','\u2013':'-','\u2026':'...','\u00a0':' ','\ufeff':'','\u00e6':'ae','\u00c6':'AE','\u0153':'oe','\u00df':'ss','\t':' '}
out, meta, tot, seen_titles = [], [], 0, set()
for n in first + rest:
    f = f'raw/pg{n}.txt'
    if not os.path.exists(f): print('missing', n); continue
    s = open(f, encoding='utf-8', errors='replace').read().replace('\r\n', '\n')
    head = s[:5000]
    lang = re.search(r'^Language: *(.*)$', head, re.M); title = re.search(r'^Title: *(.*)$', head, re.M); auth = re.search(r'^Author: *(.*)$', head, re.M)
    if not lang or 'English' not in lang.group(1): print('skip (language)', n, lang and lang.group(1)); continue
    t = title.group(1).strip() if title else '?'
    if t in seen_titles: print('skip (dup title)', n, t); continue
    m1 = re.search(r'^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$', s, re.M); m2 = re.search(r'^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK', s, re.M)
    if not (m1 and m2): print('skip (markers)', n); continue
    s = s[m1.end():m2.start()]
    s = ''.join(MAP.get(c, c) for c in s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = ''.join(c for c in s if c == '\n' or 32 <= ord(c) <= 126)
    s = re.sub(r'[ ]+\n', '\n', s); s = re.sub(r'\n{4,}', '\n\n\n', s).strip() + '\n\n\n'
    if tot + len(s) > MAXCH: print('cap reached, skip', n, t); continue
    seen_titles.add(t)
    meta.append(dict(id=n, title=t, author=auth.group(1).strip() if auth else '?', url=f'https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt', chars=len(s), start=tot))
    out.append(s); tot += len(s)
txt = ''.join(out); open('corpus.txt', 'w').write(txt)
N = len(txt); vs = int(0.9 * N)
for mm in meta: mm['split'] = 'val' if mm['start'] >= vs else ('train+val' if mm['start'] + mm['chars'] > vs else 'train')
print('books', len(meta), 'chars', N, 'vocab', len(set(txt)), 'val start', vs, 'val chars', N - vs)
json.dump(dict(books=meta, chars=N, vocab=''.join(sorted(set(txt))), val_start=vs), open('corpus_meta.json', 'w'), indent=1)
