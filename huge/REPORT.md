# huge/: bigger char transformer on a 100 MB Gutenberg corpus, Lisp (gpt-huge.lisp) vs PyTorch

## Corpus (`corpus.txt`, built by `build_corpus.py` from `raw/`; per-book offsets/splits in `corpus_meta.json`)
- **121 public-domain English Project Gutenberg books, 99,888,937 chars (99.9 MB), vocab 95** (printable ASCII 32-126 + newline; curly quotes/dashes mapped, accents stripped via NFKD, tab -> space, other characters dropped).
- Gutenberg header/license stripped (only text between the `*** START OF` / `*** END OF` markers). Downloaded 2026-10-17 from `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt` (`fetch.sh`, 1 request/s).
- Order: the 14 books of big/corpus.txt first, then the rest shuffled with a fixed seed; filled up to 100 MB (14 further downloaded books did not fit and were left out). Val = last 10% (chars from 89,900,043; 9,988,894 chars), which spans: Mansfield Park, The Return of Sherlock Holmes, The Confessions of St. Augustine, Madame Bovary, The call of the wild, The Innocents Abroad, On the Origin of Species By Means of Natural Selection, Anne of Green Gables, Les Misérables, Andersen's Fairy Tales, The Interesting Narrative of the Life of Olaudah Equiano, Or Gustavus Vassa, The African, Metamorphosis, The Fortunes and Misfortunes of the Famous Moll Flanders, Meditations, A Modest Proposal.
- Largest books: The Complete Works of William Shakespeare (5.4 MB), The Bible, King James Version, Complete (4.5 MB), Les Misérables (3.2 MB), War and Peace (3.2 MB), Essays of Michel de Montaigne — Complete (3.0 MB), The Count of Monte Cristo (2.6 MB). Full list:

<details><summary>all books</summary>

- 1342 Pride and Prejudice -- Jane Austen (728,459 chars, train)
- 158 Emma -- Jane Austen (883,413 chars, train)
- 161 Sense and Sensibility -- Jane Austen (671,438 chars, train)
- 1400 Great Expectations -- Charles Dickens (995,646 chars, train)
- 98 A Tale of Two Cities -- Charles Dickens (757,510 chars, train)
- 1260 Jane Eyre: An Autobiography -- Charlotte Brontë (1,024,360 chars, train)
- 768 Wuthering Heights -- Emily Brontë (646,230 chars, train)
- 345 Dracula -- Bram Stoker (845,834 chars, train)
- 600 Notes from the Underground -- Fyodor Dostoyevsky (239,946 chars, train)
- 76 Adventures of Huckleberry Finn -- Mark Twain (572,084 chars, train)
- 2701 Moby Dick; Or, The Whale -- Herman Melville (1,220,685 chars, train)
- 174 The Picture of Dorian Gray -- Oscar Wilde (429,510 chars, train)
- 84 Frankenstein; or, the modern prometheus -- Mary Wollstonecraft Shelley (419,412 chars, train)
- 11 Alice's Adventures in Wonderland -- Lewis Carroll (144,839 chars, train)
- 42324 Frankenstein; Or, The Modern Prometheus -- Mary Wollstonecraft Shelley (438,090 chars, train)
- 1837 The Prince and the Pauper -- Mark Twain (395,909 chars, train)
- 844 The Importance of Being Earnest: A Trivial Comedy for Serious People -- Oscar Wilde (117,313 chars, train)
- 105 Persuasion -- Jane Austen (464,921 chars, train)
- 6130 The Iliad -- Homer (1,097,317 chars, train)
- 1023 Bleak House -- Charles Dickens (1,941,479 chars, train)
- 209 The Turn of the Screw -- Henry James (229,105 chars, train)
- 4085 The Adventures of Roderick Random -- T. Smollett (1,074,387 chars, train)
- 205 Walden, and On The Duty Of Civil Disobedience -- Henry David Thoreau (635,298 chars, train)
- 119 A Tramp Abroad -- Mark Twain (879,766 chars, train)
- 910 White Fang -- Jack London (397,625 chars, train)
- 120 Treasure Island -- Robert Louis Stevenson (364,434 chars, train)
- 30 The Bible, King James Version, Complete -- ? (4,513,974 chars, train)
- 147 Common Sense -- Thomas Paine (126,391 chars, train)
- 2600 War and Peace -- graf Leo Tolstoy (3,209,247 chars, train)
- 100 The Complete Works of William Shakespeare -- William Shakespeare (5,361,273 chars, train)
- 46 A Christmas Carol in Prose; Being a Ghost Story of Christmas -- Charles Dickens (158,414 chars, train)
- 421 Kidnapped -- Robert Louis Stevenson (423,035 chars, train)
- 2554 Crime and Punishment -- Fyodor Dostoyevsky (1,135,767 chars, train)
- 102 The Tragedy of Pudd'nhead Wilson -- Mark Twain (294,678 chars, train)
- 159 The island of Doctor Moreau -- H. G. Wells (241,382 chars, train)
- 7370 Second Treatise of Government -- John Locke (313,638 chars, train)
- 541 The Age of Innocence -- Edith Wharton (578,329 chars, train)
- 203 Uncle Tom's Cabin -- Harriet Beecher Stowe (1,007,659 chars, train)
- 41 The Legend of Sleepy Hollow -- Washington Irving (69,758 chars, train)
- 1399 Anna Karenina -- graf Leo Tolstoy (1,965,723 chars, train)
- 2852 The Hound of the Baskervilles -- Arthur Conan Doyle (354,232 chars, train)
- 139 The Lost World -- Arthur Conan Doyle (423,975 chars, train)
- 12 Through the Looking-Glass -- Lewis Carroll (164,620 chars, train)
- 164 Twenty Thousand Leagues under the Sea -- Jules Verne (597,912 chars, train)
- 1727 The Odyssey -- Homer (678,677 chars, train)
- 23 Narrative of the Life of Frederick Douglass, an American Slave -- Frederick Douglass (224,179 chars, train)
- 4705 A Treatise of Human Nature -- David Hume (1,323,906 chars, train)
- 432 The Ambassadors -- Henry James (907,931 chars, train)
- 74 The Adventures of Tom Sawyer, Complete -- Mark Twain (393,589 chars, train)
- 244 A Study in Scarlet -- Arthur Conan Doyle (239,525 chars, train)
- 580 The Pickwick Papers -- Charles Dickens (1,746,218 chars, train)
- 113 The Secret Garden -- Frances Hodgson Burnett (429,087 chars, train)
- 1322 Leaves of Grass -- Walt Whitman (737,285 chars, train)
- 786 Hard Times -- Charles Dickens (579,939 chars, train)
- 2500 Siddhartha -- Hermann Hesse (216,198 chars, train)
- 86 A Connecticut Yankee in King Arthur's Court -- Mark Twain (643,546 chars, train)
- 6593 History of Tom Jones, a Foundling -- Henry Fielding (1,960,738 chars, train)
- 2097 The Sign of the Four -- Arthur Conan Doyle (232,230 chars, train)
- 16 Peter Pan -- J. M. Barrie (255,630 chars, train)
- 103 Around the World in Eighty Days -- Jules Verne (369,305 chars, train)
- 3207 Leviathan -- Thomas Hobbes (1,211,042 chars, train)
- 829 Gulliver's Travels into Several Remote Nations of the World -- Jonathan Swift (582,432 chars, train)
- 219 Heart of Darkness -- Joseph Conrad (210,511 chars, train)
- 4300 Ulysses -- James Joyce (1,522,055 chars, train)
- 8800 The divine comedy -- Dante Alighieri (604,530 chars, train)
- 4517 Ethan Frome -- Edith Wharton (188,540 chars, train)
- 2638 The Idiot -- Fyodor Dostoyevsky (1,350,221 chars, train)
- 6761 The Adventures of Ferdinand Count Fathom — Complete -- T. Smollett (954,930 chars, train)
- 2814 Dubliners -- James Joyce (369,884 chars, train)
- 974 The Secret Agent: A Simple Tale -- Joseph Conrad (522,288 chars, train)
- 36 The war of the worlds -- H. G. Wells (337,676 chars, train)
- 5230 The Invisible Man: A Grotesque Romance -- H. G. Wells (273,241 chars, train)
- 64317 The Great Gatsby -- F. Scott Fitzgerald (271,218 chars, train)
- 33 The Scarlet Letter -- Nathaniel Hawthorne (481,052 chars, train)
- 1251 Le Morte d'Arthur: Volume 1 -- Sir Thomas Malory (880,666 chars, train)
- 3289 The Valley of Fear -- Arthur Conan Doyle (313,829 chars, train)
- 2160 The Expedition of Humphry Clinker -- T. Smollett (849,957 chars, train)
- 1695 The Man Who Was Thursday: A Nightmare -- G. K. Chesterton (322,223 chars, train)
- 121 Northanger Abbey -- Jane Austen (434,353 chars, train)
- 1952 The Yellow Wallpaper -- Charlotte Perkins Gilman (31,562 chars, train)
- 236 The Jungle Book -- Rudyard Kipling (273,016 chars, train)
- 28054 The Brothers Karamazov -- Fyodor Dostoyevsky (1,936,216 chars, train)
- 1268 The Mysterious Island -- Jules Verne (1,112,194 chars, train)
- 4363 Beyond Good and Evil -- Friedrich Wilhelm Nietzsche (382,756 chars, train)
- 47 Anne of Avonlea -- L. M. Montgomery (488,064 chars, train)
- 245 Life on the Mississippi -- Mark Twain (808,213 chars, train)
- 1497 The Republic -- Plato (1,195,240 chars, train)
- 82 Ivanhoe: A Romance -- Walter Scott (1,105,877 chars, train)
- 4280 The Critique of Pure Reason -- Immanuel Kant (1,271,351 chars, train)
- 160 The Awakening, and Selected Short Stories -- Kate Chopin (359,655 chars, train)
- 996 Don Quixote -- Miguel de Cervantes Saavedra (2,298,628 chars, train)
- 1250 Anthem -- Ayn Rand (108,046 chars, train)
- 19942 Candide -- Voltaire (205,527 chars, train)
- 7849 The Trial -- Franz Kafka (449,748 chars, train)
- 1257 The three musketeers -- Alexandre Dumas (1,298,687 chars, train)
- 5658 Lord Jim -- Joseph Conrad (709,859 chars, train)
- 5827 The Problems of Philosophy -- Bertrand Russell (245,701 chars, train)
- 1998 Thus Spake Zarathustra: A Book for All and None -- Friedrich Wilhelm Nietzsche (639,137 chars, train)
- 1184 The Count of Monte Cristo -- Alexandre Dumas (2,629,446 chars, train)
- 18857 A Journey to the Centre of the Earth -- Jules Verne (486,878 chars, train)
- 766 David Copperfield -- Charles Dickens (1,934,591 chars, train)
- 3600 Essays of Michel de Montaigne — Complete -- Michel de Montaigne (3,013,877 chars, train)
- 1404 The Federalist Papers -- Alexander Hamilton (1,147,445 chars, train)
- 271 Black Beauty -- Anna Sewell (303,866 chars, train)
- 145 Middlemarch -- George Eliot (1,782,425 chars, train)
- 3268 The Mysteries of Udolpho -- Ann Ward Radcliffe (1,681,682 chars, train)
- 141 Mansfield Park -- Jane Austen (884,292 chars, train+val)
- 108 The Return of Sherlock Holmes -- Arthur Conan Doyle (674,097 chars, val)
- 3296 The Confessions of St. Augustine -- Saint of Hippo Augustine (602,938 chars, val)
- 2413 Madame Bovary -- Gustave Flaubert (652,079 chars, val)
- 215 The call of the wild -- Jack London (175,617 chars, val)
- 3176 The Innocents Abroad -- Mark Twain (1,106,796 chars, val)
- 1228 On the Origin of Species By Means of Natural Selection -- Charles Darwin (933,465 chars, val)
- 45 Anne of Green Gables -- L. M. Montgomery (561,071 chars, val)
- 135 Les Misérables -- Victor Hugo (3,233,733 chars, val)
- 1597 Andersen's Fairy Tales -- H. C. Andersen (299,283 chars, val)
- 15399 The Interesting Narrative of the Life of Olaudah Equiano, Or Gustavus Vassa, The African -- Olaudah Equiano (451,424 chars, val)
- 5200 Metamorphosis -- Franz Kafka (118,491 chars, val)
- 370 The Fortunes and Misfortunes of the Famous Moll Flanders -- Daniel Defoe (705,032 chars, val)
- 2680 Meditations -- Emperor of Rome Marcus Aurelius (397,788 chars, val)
- 1080 A Modest Proposal -- Jonathan Swift (19,496 chars, val)
</details>

## Setup (identical in both implementations)
- Model: pre-LN GPT, **d=256, 6 layers, 8 heads (head dim 32), ctx 128**, GELU (tanh), LM head tied to wte, learned positions. **4,796,160 params** (vocab 95).
- **Dropout 0** (100 MB of data, ~2-3 passes in 3 h; see overfitting check below). Init as before (N(0,0.02); proj/fc2 N(0,0.02/sqrt(2L))).
- AdamW lr 1e-3, betas 0.9/0.99, eps 1e-8, wd 0.1 on matrices+embeddings, grad-norm clip 1.0, 2% warmup then cosine to lr/10 over **10800 s of training time** (eval excluded). Batch 32 x 128 = 4096 tokens. float32, 8 threads, `nice -n 10`, run back to back by `queue.sh` (`queue.log`).
- Eval: windows of 128, stride 64, score the 2nd half (64-127 chars of context). Subset val (first 32k val targets) every 200 steps; "val (1M)" = first 1,000,000 val targets every 1800 s; **FINAL = the whole 10% val set (9.99M chars)**.
- Code: `gpt-huge.lisp` = fast-gpt/gpt-fast.lisp + octet corpus loading + `fullval-targets=`; `gpt_torch.py` = big/gpt_torch.py + uint8 data + `fullval-targets=`. Runtime options only, no maths changes.
- Gradient check of the scaled code (`gradcheck.log`, `equiv.log`, scripts in `check/`): analytic gradients of gpt-huge.lisp are **bitwise identical** to the reference gpt-big.lisp on a scaled-shape tiny config (V=11 d=16 L=6 H=8 T=16; 20,144 params, dropout 0/0.2) and on the full 4.8M-param config (8 workers, dropout 0/0.05): 0 differing gradients. The reference in double precision matches finite differences on the scaled-shape tiny config with global rel. error 1.5e-9 / 2.2e-9 / 3.1e-9 (dropout 0 / 0.2 / 0.5).

## Results
| | Lisp gpt-huge.lisp (4.8M) | PyTorch 2.14 CPU (4.8M) | Lisp small model (0.81M, d128 L4 H4 ctx64, dropout 0.1, 1 h) |
|---|---|---|---|
| training time | 10800 s | 10800 s | 3600 s |
| steps | 53,964 | 32,824 | 131,482 |
| tokens/s (training time) | **20,466** | **12,449** | **74,798** |
| eval time (excluded) | 715 s | 490 s | 270 s |
| **final val, full 10% (9.99M chars)** | **1.1227** | **1.1397** | **1.3033** |
| tokens seen (epochs of 89.9M train chars) | 221M (2.46) | 134M (1.50) | 269M (3.00) |
| val (first 1M targets) at 0.5/1/1.5/2/2.5 h | 1.1959 / 1.1343 / 1.1059 / 1.0671 / 1.0367 | 1.2147 / 1.1508 / 1.1146 / 1.0848 / 1.0570 | 1.3142 |

- Lisp ran 1.64x the tokens/s of PyTorch on this model (20,466 vs 12,449).
- Final full val: Lisp 1.1227 vs PyTorch 1.1397 (difference +0.0170 nats/char in Lisp's favour if positive); per optimizer step the two implementations compute the same function, so the gap comes from the number of steps in the time budget (see right panel of compare.png).
- The old small model (0.81M params, 1 h) reaches 1.3033 on the same val set; the 4.8M model improves on it by 0.1806 nats/char.

## Overfitting check
- Lisp 4.8M: periodic val (1M) falls at every checkpoint; mean (val subset - train batch) = -0.122 over the last 25 checkpoints vs -0.163 early in training.
- PyTorch 4.8M: periodic val (1M) falls at every checkpoint; mean (val subset - train batch) = -0.136 over the last 25 checkpoints vs -0.157 early in training.
- Lisp small: periodic val (1M) falls at every checkpoint; mean (val subset - train batch) = -0.223 over the last 25 checkpoints vs -0.247 early in training.
- Negative or near-zero val-minus-train gaps mean no measurable overfitting (the val books differ in style from the train mix, so absolute train/val levels are not directly comparable; the trend is what matters).

## Samples (300 chars, T=0.8, starting from a newline)
Lisp 4.8M:
```
cast advisable words to his style through returning to the seas, or the
secrets of the company were at the appointed door, the mere and mysterious
reputation of Lady Bertram was as if one could come to the best of
her going near to him and the general manners of the day, employed
the suggestion as i
```
PyTorch 4.8M:
```
Her nose was house; and her face continued in return to her place, and
to which she felt the direction of her companion.

Mitya was soon that his father had been overwhelmed by Miss Havisham,
and succeeded in a scandal with her almost a little brother than in a
low-coloured tuzwiser coat and silent
```
Lisp small (0.81M):
```
for him a man who could say that he would like to endure the whole
cardinal sense of the mountain, with his armours, and thus hither with
slipping for a conduct of the great side. It was a covered in his
chair, and eager state of all the children of New Arthur; and Mercedes
they had repeated to him
```

## Files (all in /home/box/lisp-charmodel/huge/)
corpus.txt, corpus_meta.json, build_corpus.py, fetch.sh, ids.txt, raw/ | gpt-huge.lisp, gpt_torch.py | queue.sh, queue.log | lisp-big.{log,csv}, lisp-big-full.csv, torch-big.{log,csv}, torch-big-full.csv, torch-big.pt, lisp-small.{log,csv}, lisp-small-full.csv | gradcheck.log, equiv.log, check/ | compare.png (plot_compare.py) | REPORT.md (make_report.py, finalize.sh)
