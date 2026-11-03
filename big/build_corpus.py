# Build big/corpus.txt from Project Gutenberg plain-text ebooks (public domain in the US).
# Strips the Gutenberg header/footer (everything outside the *** START OF / *** END OF markers),
# normalizes to ASCII (curly quotes -> straight, dashes -> '-'/'--', accents removed via NFKD, other non-ASCII dropped),
# converts CRLF to LF and collapses 3+ blank lines to 2. Books concatenated in the fixed order below.
import re, unicodedata, json
BOOKS=[(1342,'Pride and Prejudice','Jane Austen'),(158,'Emma','Jane Austen'),(161,'Sense and Sensibility','Jane Austen'),
 (1400,'Great Expectations','Charles Dickens'),(98,'A Tale of Two Cities','Charles Dickens'),
 (1260,'Jane Eyre','Charlotte Bronte'),(768,'Wuthering Heights','Emily Bronte'),(345,'Dracula','Bram Stoker'),
 (600,'Notes from the Underground','Fyodor Dostoevsky (C. Garnett transl.)'),(76,'Adventures of Huckleberry Finn','Mark Twain'),
 (2701,'Moby Dick','Herman Melville'),
 (174,'The Picture of Dorian Gray','Oscar Wilde'),(84,'Frankenstein','Mary Shelley'),(11,"Alice's Adventures in Wonderland",'Lewis Carroll')]
MAP={'\u2018':"'",'\u2019':"'",'\u201c':'"','\u201d':'"','\u2014':'--','\u2013':'-','\u2026':'...','\u00a0':' ','\ufeff':'','\u00e6':'ae','\u00c6':'AE','\u0153':'oe','\u00df':'ss'}
out=[]; meta=[]
for n,t,a in BOOKS:
    s=open(f'raw/pg{n}.txt',encoding='utf-8').read().replace('\r\n','\n')
    m1=re.search(r'^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$',s,re.M); m2=re.search(r'^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK',s,re.M)
    assert m1 and m2, n
    s=s[m1.end():m2.start()]
    s=''.join(MAP.get(c,c) for c in s)
    s=unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode('ascii')
    s=re.sub(r'[ \t]+\n','\n',s); s=re.sub(r'\n{4,}','\n\n\n',s).strip()+'\n\n\n'
    meta.append(dict(id=n,title=t,author=a,url=f'https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt',chars=len(s)))
    out.append(s)
txt=''.join(out); open('corpus.txt','w').write(txt)
N=len(txt); vs=int(0.9*N)
pos=0
for m in meta: m['start']=pos; pos+=m['chars']
print('chars',N,'vocab',len(set(txt)),'val start',vs,'val chars',N-vs)
for m in meta: print(m['id'],m['title'],m['chars'],'VAL' if m['start']+m['chars']>vs else '')
json.dump(dict(books=meta,chars=N,vocab=''.join(sorted(set(txt))),val_start=vs),open('corpus_meta.json','w'),indent=1)
