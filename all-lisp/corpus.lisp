;;;; corpus.lisp -- pure-Lisp port of huge/build_corpus.py (Gutenberg raw/pg<id>.txt -> corpus.txt, ASCII).
;;;; Same steps: ids.txt dedup, first 14 kept in order, rest shuffled with Python's random.Random(1).shuffle
;;;; (MT19937 init_by_array([1]) + _randbelow via getrandbits, reimplemented here), English-only, dup titles skipped,
;;;; text between *** START/END OF THE PROJECT GUTENBERG EBOOK lines, MAP table, NFKD (sb-unicode) + drop non-ASCII,
;;;; keep \n and 32..126, "[ ]+\n"->"\n", "\n{4,}"->"\n\n\n", strip, + "\n\n\n", 100,000,000-char cap.
;;;; Usage: sbcl --dynamic-space-size 4000 --script corpus.lisp HUGEDIR OUT.txt [BOOKS.tsv]

(defpackage :corpus (:use :cl)) (in-package :corpus)

;;; Python random: MT19937
(defvar *mt* (make-array 624 :element-type '(unsigned-byte 32))) (defvar *mti* 625)
(defun init-genrand (s) (setf (aref *mt* 0) s)
  (loop for i from 1 below 624 do (setf (aref *mt* i) (ldb (byte 32 0) (+ (* 1812433253 (logxor (aref *mt* (1- i)) (ash (aref *mt* (1- i)) -30))) i))))
  (setf *mti* 624))
(defun init-by-array (key)
  (init-genrand 19650218)
  (let ((i 1) (j 0) (n 624) (kl (length key)))
    (loop repeat (max n kl) do
      (setf (aref *mt* i) (ldb (byte 32 0) (+ (logxor (aref *mt* i) (* (logxor (aref *mt* (1- i)) (ash (aref *mt* (1- i)) -30)) 1664525)) (aref key j) j)))
      (incf i) (incf j) (when (>= i n) (setf (aref *mt* 0) (aref *mt* (1- n)) i 1)) (when (>= j kl) (setf j 0)))
    (loop repeat (1- n) do
      (setf (aref *mt* i) (ldb (byte 32 0) (- (logxor (aref *mt* i) (* (logxor (aref *mt* (1- i)) (ash (aref *mt* (1- i)) -30)) 1566083941)) i)))
      (incf i) (when (>= i n) (setf (aref *mt* 0) (aref *mt* (1- n)) i 1)))
    (setf (aref *mt* 0) #x80000000)))
(defun genrand ()
  (when (>= *mti* 624)
    (dotimes (k 624)
      (let ((y (logior (logand (aref *mt* k) #x80000000) (logand (aref *mt* (mod (1+ k) 624)) #x7fffffff))))
        (setf (aref *mt* k) (logxor (aref *mt* (mod (+ k 397) 624)) (ash y -1) (if (oddp y) #x9908b0df 0)))))
    (setf *mti* 0))
  (let ((y (aref *mt* *mti*))) (incf *mti*)
    (setf y (logxor y (ash y -11))) (setf y (logxor y (logand (ash y 7) #x9d2c5680)))
    (setf y (logxor y (logand (ash y 15) #xefc60000))) (logxor y (ash y -18))))
(defun randbelow (n) (let ((k (integer-length n))) (loop (let ((r (ash (genrand) (- k 32)))) (when (< r n) (return r))))))
(defun py-shuffle (v) (loop for i from (1- (length v)) downto 1 do (rotatef (aref v i) (aref v (randbelow (1+ i))))) v)

;;; helpers
(defun line-start-p (s i) (or (= i 0) (char= (char s (1- i)) #\Newline)))
(defun line-end (s i) (or (position #\Newline s :start i) (length s)))
(defun header-field (head name)
  "re.search(r'^NAME: *(.*)$', head, re.M) -> group 1 or nil"
  (let ((pat (concatenate 'string name ":")))
    (loop for p = (search pat head) then (search pat head :start2 (1+ p)) while p
          when (line-start-p head p)
            do (let ((i (+ p (length pat)))) (loop while (and (< i (length head)) (char= (char head i) #\Space)) do (incf i))
                 (return (subseq head i (line-end head i)))))))
(defun marker (s word)
  "^\*\*\* ?WORD OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$ -> (line-start . line-end)"
  (loop for p = (search "***" s) then (search "***" s :start2 (1+ p)) while p
        when (line-start-p s p)
          do (let ((i (+ p 3))) (when (and (< i (length s)) (char= (char s i) #\Space)) (incf i))
               (dolist (mid '("THE" "THIS"))
                 (let ((m (concatenate 'string word " OF " mid " PROJECT GUTENBERG EBOOK")))
                   (when (and (<= (+ i (length m)) (length s)) (string= m s :start2 i :end2 (+ i (length m))))
                     (return-from marker (cons p (line-end s (+ i (length m)))))))))))
(defun py-strip (s) (string-trim '(#\Space #\Newline #\Tab #\Return #\Page #.(code-char 11)) s))
(defparameter *map* '((#x2018 . "'") (#x2019 . "'") (#x201c . "\"") (#x201d . "\"") (#x2014 . "--") (#x2013 . "-") (#x2026 . "...")
                      (#xa0 . " ") (#xfeff . "") (#xe6 . "ae") (#xc6 . "AE") (#x153 . "oe") (#xdf . "ss") (9 . " ")))
(defun clean (s)
  (let* ((m (with-output-to-string (o) (loop for c across s do (let ((r (assoc (char-code c) *map*))) (if r (write-string (cdr r) o) (write-char c o))))))
         (n (sb-unicode:normalize-string m :nfkd))
         (a (with-output-to-string (o) (loop for c across n do (let ((k (char-code c))) (when (or (= k 10) (<= 32 k 126)) (write-char c o))))))
         (b (with-output-to-string (o)   ; [ ]+\n -> \n
              (let ((i 0) (len (length a)))
                (loop while (< i len) do
                  (if (char= (char a i) #\Space)
                      (let ((j (or (position #\Space a :start i :test-not #'char=) len)))
                        (unless (and (< j len) (char= (char a j) #\Newline)) (write-string a o :start i :end j)) (setf i j))
                      (progn (write-char (char a i) o) (incf i)))))))
         (c (with-output-to-string (o)   ; \n{4,} -> \n\n\n
              (let ((i 0) (len (length b)))
                (loop while (< i len) do
                  (if (char= (char b i) #\Newline)
                      (let ((j (or (position #\Newline b :start i :test-not #'char=) len)))
                        (write-string (if (>= (- j i) 4) (format nil "~%~%~%") (subseq b i j)) o) (setf i j))
                      (progn (write-char (char b i) o) (incf i))))))))
    (concatenate 'string (py-strip c) (format nil "~%~%~%"))))

(defun read-utf8 (path)
  (with-open-file (s path :element-type '(unsigned-byte 8))
    (let ((b (make-array (file-length s) :element-type '(unsigned-byte 8)))) (read-sequence b s)
      (sb-ext:octets-to-string b :external-format '(:utf-8 :replacement #\ufffd)))))
(defun crlf->lf (s) (with-output-to-string (o) (loop for i from 0 below (length s) do (let ((c (char s i))) (unless (and (char= c #\Return) (< (1+ i) (length s)) (char= (char s (1+ i)) #\Newline)) (write-char c o))))))

(let* ((args (cdr sb-ext:*posix-argv*)) (dir (first args)) (outp (second args)) (tsv (third args))
       (ids (let ((l '())) (with-open-file (s (format nil "~a/ids.txt" dir)) (loop for x = (read s nil) while x do (unless (member x l) (setf l (append l (list x)))))) l))
       (first14 (subseq ids 0 14)) (rest (coerce (subseq ids 14) 'vector)) (tot 0) (seen (make-hash-table :test 'equal)) (books '()))
  (init-by-array (vector 1)) (py-shuffle rest)
  (with-open-file (out outp :direction :output :if-exists :supersede :external-format :latin-1)
    (dolist (n (append first14 (coerce rest 'list)))
      (let ((f (format nil "~a/raw/pg~d.txt" dir n)))
        (if (not (probe-file f)) (format t "missing ~d~%" n)
            (let* ((s (crlf->lf (read-utf8 f))) (head (subseq s 0 (min 5000 (length s))))
                   (lang (header-field head "Language")) (title (header-field head "Title")) (auth (header-field head "Author"))
                   (tt (if title (py-strip title) "?")))
              (cond ((not (and lang (search "English" lang))) (format t "skip (language) ~d ~a~%" n lang))
                    ((gethash tt seen) (format t "skip (dup title) ~d ~a~%" n tt))
                    (t (let ((m1 (marker s "START")) (m2 (marker s "END")))
                         (if (not (and m1 m2)) (format t "skip (markers) ~d~%" n)
                             (let ((c (clean (subseq s (cdr m1) (car m2)))))
                               (if (> (+ tot (length c)) 100000000) (format t "cap reached, skip ~d ~a~%" n tt)
                                   (progn (setf (gethash tt seen) t) (push (list n tt (if auth (py-strip auth) "?") (length c) tot) books)
                                          (write-string c out) (incf tot (length c))))))))))))))
  (format t "books ~d chars ~d val start ~d~%" (length books) tot (floor (* 9 tot) 10))
  (when tsv (with-open-file (o tsv :direction :output :if-exists :supersede)
              (format o "id	title	author	chars	start~%")
              (dolist (b (reverse books)) (format o "~{~a~^	~}~%" b)))))
