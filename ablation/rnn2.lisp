;;;; rnn2.lisp -- configurable char RNN / GRU for the ablation study. Hand-written BPTT, pure Lisp.
;;;; Kernels: /home/box/lisp-nn/fast/fnn.lisp.  Options are key=value args:
;;;;   data=small|full  cell=rnn|gru  h=256  tie=0|1  drop=0.0  sched=const|cos  secs=480  mode=train|gradcheck
;;;; Cell, written with input gates xg (m x G) and hidden gates hg (m x G); G = H (rnn) or 3H (gru):
;;;;   rnn: h' = tanh(xg + hg)
;;;;   gru: z = sig(xz+hz), r = sig(xr+hr), n = tanh(xn + r*hn), h' = (1-z) n + z h
;;;; Input:  xg = Ein[x] + bx              (untied)   or  Emb[x] Wx + bx  (tied, Emb is V x H)
;;;; Output: logits = drop(h') Why + by    (untied)   or  drop(h') Emb^T + by (tied)
(load (compile-file "/home/box/lisp-nn/fast/fnn.lisp" :output-file "/tmp/fnn-abl.fasl" :verbose nil :print nil))
(in-package :fnn)
(declaim (optimize (speed 3) (safety 0)))

(defun opt (key default)
  (let ((a (find-if (lambda (s) (and (> (length s) (length key)) (string= (concatenate 'string key "=") s :end2 (1+ (length key)))))
                    sb-ext:*posix-argv*)))
    (if a (subseq a (1+ (length key))) default)))

(defstruct (rnet (:conc-name net-) (:constructor make-rnet)) cell (v 0 :type fixnum) (h 0 :type fixnum) (g 0 :type fixnum) tie (drop 0d0 :type double-float)
  ein emb wx bx wh bh why by             ; parameters (unused ones are NIL)
  grads adam-m adam-v (adam-t 0 :type fixnum))
(defun net-plist (n)      ; (name param) for existing params
  (loop for (name p) on (list "Ein" (net-ein n) "Emb" (net-emb n) "Wx" (net-wx n) "bx" (net-bx n)
                              "Wh" (net-wh n) "bh" (net-bh n) "Why" (net-why n) "by" (net-by n)) by #'cddr
        when p collect (list name p)))
(defun grad-of (n p) (cdr (assoc p (net-grads n))))

(defun make-net* (cell v h tie drop)
  (let* ((g (if (eq cell :gru) (* 3 h) h)) (s (/ 1d0 (sqrt (float h 1d0))))
         (n (make-rnet :cell cell :v v :h h :g g :tie tie :drop drop)))
    (flet ((rnd (k sc) (let ((a (make-dvec k))) (dotimes (i k a) (setf (aref a i) (* sc (randn)))))))
      (if tie
          (setf (net-emb n) (rnd (* v h) 0.1d0) (net-wx n) (rnd (* h g) s))
          (setf (net-ein n) (rnd (* v g) 0.1d0) (net-why n) (rnd (* h v) s)))
      (setf (net-bx n) (make-dvec g) (net-wh n) (rnd (* h g) s) (net-bh n) (make-dvec g) (net-by n) (make-dvec v)))
    (flet ((zeros-like () (mapcar (lambda (np) (cons (second np) (make-dvec (length (second np))))) (net-plist n))))
      (setf (net-grads n) (zeros-like) (net-adam-m n) (zeros-like) (net-adam-v n) (zeros-like)))
    n))
(defun nparams (n) (reduce #'+ (mapcar (lambda (np) (length (second np))) (net-plist n))))

;;; Per-time-step activations for a given batch size and truncation length
(defstruct buf (b 0 :type fixnum) (tt 0 :type fixnum) hs gates hnh xe hd masks ps
  hg dh dnext dprev dxg dhg dxe)
(defun make-buf* (n b tt)
  (let ((h (net-h n)) (g (net-g n)) (v (net-v n)))
    (flet ((vec (k sz) (coerce (loop repeat k collect (make-dvec sz)) 'simple-vector)))
      (make-buf :b b :tt tt :hs (vec (1+ tt) (* b h)) :gates (vec tt (* b g)) :hnh (vec tt (* b h))
                :xe (vec tt (* b h)) :hd (vec tt (* b h)) :masks (vec tt (* b h)) :ps (vec tt (* b v))
                :hg (make-dvec (* b g)) :dh (make-dvec (* b h)) :dnext (make-dvec (* b h)) :dprev (make-dvec (* b h))
                :dxg (make-dvec (* b g)) :dhg (make-dvec (* b g)) :dxe (make-dvec (* b h))))))

(declaim (inline sig))
(defun sig (x) (declare (double-float x))
  (if (>= x 0d0) (/ 1d0 (+ 1d0 (exp-neg (- x)))) (let ((e (exp-neg x))) (/ e (+ 1d0 e)))))

(defvar *freeze-masks* nil)   ; gradient check: reuse dropout masks between forward passes

(defun step-forward (n bf xs s m train)
  "Time step S for M rows with input ids XS."
  (declare (simple-vector xs) (fixnum s m))
  (let* ((h (net-h n)) (g (net-g n)) (v (net-v n)) (gru (eq (net-cell n) :gru))
         (hp (svref (buf-hs bf) s)) (hn (svref (buf-hs bf) (1+ s))) (gt (svref (buf-gates bf) s))
         (hnh (svref (buf-hnh bf) s)) (xe (svref (buf-xe bf) s)) (hd (svref (buf-hd bf) s))
         (mask (svref (buf-masks bf) s)) (P (svref (buf-ps bf) s)) (hg (buf-hg bf))
         (bx (net-bx n)) (bh (net-bh n)) (by (net-by n)) (pdrop (net-drop n)))
    (declare (dim h g v) (dvec hp hn gt hnh xe hd mask P hg bx bh by) (double-float pdrop))
    ;; input gates -> gt
    (dotimes (i m) (replace gt bx :start1 (* i g)))
    (if (net-tie n)
        (let ((emb (net-emb n)))
          (declare (dvec emb))
          (dotimes (i m) (replace xe emb :start1 (* i h) :start2 (* (the dim (svref xs i)) h) :end2 (* (1+ (the dim (svref xs i))) h)))
          (mm-add xe (net-wx n) gt m h g))
        (let ((ein (net-ein n)))
          (declare (dvec ein))
          (dotimes (i m) (axpy 1d0 ein (* (the dim (svref xs i)) g) gt (* i g) g))))
    ;; hidden gates
    (dotimes (i m) (replace hg bh :start1 (* i g)))
    (mm-add hp (net-wh n) hg m h g)
    ;; cell
    (if gru
        (dotimes (i m)
          (let ((o (* i g)) (oh (* i h)))
            (declare (idx o oh))
            (dotimes (j h)
              (let* ((z (sig (+ (aref gt (+ o j)) (aref hg (+ o j)))))
                     (r (sig (+ (aref gt (+ o h j)) (aref hg (+ o h j)))))
                     (hh (aref hg (+ o h h j)))
                     (nn (fast-tanh (+ (aref gt (+ o h h j)) (* r hh)))))
                (setf (aref gt (+ o j)) z (aref gt (+ o h j)) r (aref gt (+ o h h j)) nn
                      (aref hnh (+ oh j)) hh
                      (aref hn (+ oh j)) (+ (* (- 1d0 z) nn) (* z (aref hp (+ oh j)))))))))
        (dotimes (q (* m h)) (setf (aref hn q) (fast-tanh (+ (aref gt q) (aref hg q))))))
    ;; dropout (non-recurrent: only on the path to the output layer)
    (if (and train (> pdrop 0d0))
        (let ((keep (/ 1d0 (- 1d0 pdrop))))
          (dotimes (q (* m h))
            (unless *freeze-masks* (setf (aref mask q) (if (< (random 1d0 *rs*) pdrop) 0d0 keep)))
            (setf (aref hd q) (* (aref hn q) (aref mask q)))))
        (replace hd hn :end1 (* m h)))
    ;; output
    (if (net-tie n)
        (progn (mmt hd (net-emb n) P m h v) (dotimes (i m) (axpy 1d0 by 0 P (* i v) v)))
        (progn (dotimes (i m) (replace P by :start1 (* i v))) (mm-add hd (net-why n) P m h v)))
    (dotimes (i m)
      (let* ((o (* i v)) (mx (aref P o)) (sm 0d0))
        (declare (idx o) (double-float mx sm))
        (loop for j from o below (+ o v) do (setf mx (max mx (aref P j))))
        (loop for j from o below (+ o v) do (setf (aref P j) (exp-neg (- (aref P j) mx))) (incf sm (aref P j)))
        (loop for j from o below (+ o v) do (setf (aref P j) (/ (aref P j) sm)))))))

(defun colsum-add (D G m n) (declare (dvec D G) (dim m n)) (dotimes (i m) (axpy 1d0 D (* i n) G 0 n)))

(defun chunk (n bf X Y backprop &optional (train backprop))
  "Forward T steps (X,Y: vectors of id vectors), mean loss; BPTT gradients if BACKPROP."
  (let* ((b (buf-b bf)) (tt (buf-tt bf)) (h (net-h n)) (g (net-g n)) (v (net-v n))
         (loss 0d0) (scale (/ 1d0 (* b tt))) (gru (eq (net-cell n) :gru)))
    (declare (dim b tt h g v) (double-float loss scale))
    (dotimes (s tt)
      (step-forward n bf (svref X s) s b train)
      (let ((P (svref (buf-ps bf) s)) (ys (svref Y s)))
        (declare (dvec P) (simple-vector ys))
        (dotimes (i b) (decf loss (log (max 1d-300 (aref P (+ (* i v) (the dim (svref ys i))))))))))
    (when backprop
      (let ((dh (buf-dh bf)) (dnext (buf-dnext bf)) (dprev (buf-dprev bf)) (dxg (buf-dxg bf)) (dhg (buf-dhg bf))
            (dxe (buf-dxe bf)) (gby (grad-of n (net-by n))) (gbx (grad-of n (net-bx n)))
            (gwh (grad-of n (net-wh n))) (gbh (grad-of n (net-bh n))))
        (declare (dvec dh dnext dprev dxg dhg dxe gby gbx gwh gbh))
        (fill dnext 0d0)
        (loop for s of-type fixnum from (1- tt) downto 0
              do (let ((P (svref (buf-ps bf) s)) (ys (svref Y s)) (xs (svref X s))
                       (hn (svref (buf-hs bf) (1+ s))) (hp (svref (buf-hs bf) s)) (gt (svref (buf-gates bf) s))
                       (hnh (svref (buf-hnh bf) s)) (hd (svref (buf-hd bf) s)) (mask (svref (buf-masks bf) s))
                       (xe (svref (buf-xe bf) s)))
                   (declare (dvec P hn hp gt hnh hd mask xe) (simple-vector ys xs))
                   (dotimes (q (* b v)) (setf (aref P q) (* scale (aref P q))))          ; dlogits
                   (dotimes (i b) (decf (aref P (+ (* i v) (the dim (svref ys i)))) scale))
                   (colsum-add P gby b v)
                   (if (net-tie n)
                       (progn (mtm-add P hd (grad-of n (net-emb n)) b v h)             ; gEmb += dlogits^T hd
                              (fill dh 0d0) (mm-add P (net-emb n) dh b v h))           ; dh = dlogits Emb
                       (progn (mtm-add hd P (grad-of n (net-why n)) b h v)
                              (mmt P (net-why n) dh b v h)))
                   (when (> (net-drop n) 0d0) (dotimes (q (* b h)) (setf (aref dh q) (* (aref dh q) (aref mask q)))))
                   (dotimes (q (* b h)) (incf (aref dh q) (aref dnext q)))
                   ;; cell backward -> dxg (input gates), dhg (hidden gates), dprev (direct path to h_{t-1})
                   (if gru
                       (dotimes (i b)
                         (let ((o (* i g)) (oh (* i h)))
                           (declare (idx o oh))
                           (dotimes (j h)
                             (let* ((z (aref gt (+ o j))) (r (aref gt (+ o h j))) (nn (aref gt (+ o h h j)))
                                    (hh (aref hnh (+ oh j))) (d (aref dh (+ oh j)))
                                    (dn (* d (- 1d0 z))) (dz (* d (- (aref hp (+ oh j)) nn)))
                                    (dan (* dn (- 1d0 (* nn nn)))) (dr (* dan hh))
                                    (daz (* dz z (- 1d0 z))) (dar (* dr r (- 1d0 r))))
                               (setf (aref dprev (+ oh j)) (* d z)
                                     (aref dxg (+ o j)) daz (aref dxg (+ o h j)) dar (aref dxg (+ o h h j)) dan
                                     (aref dhg (+ o j)) daz (aref dhg (+ o h j)) dar (aref dhg (+ o h h j)) (* dan r))))))
                       (dotimes (q (* b h))
                         (let* ((a (aref hn q)) (da (* (aref dh q) (- 1d0 (* a a)))))
                           (setf (aref dxg q) da (aref dhg q) da))))
                   (mtm-add hp dhg gwh b h g) (colsum-add dhg gbh b g)               ; hidden weights
                   (mmt dhg (net-wh n) dnext b g h)                                  ; dnext = dhg Wh^T
                   (when gru (dotimes (q (* b h)) (incf (aref dnext q) (aref dprev q))))
                   (colsum-add dxg gbx b g)                                          ; input side
                   (if (net-tie n)
                       (let ((gemb (grad-of n (net-emb n))))
                         (declare (dvec gemb))
                         (mtm-add xe dxg (grad-of n (net-wx n)) b h g)
                         (mmt dxg (net-wx n) dxe b g h)
                         (dotimes (i b) (axpy 1d0 dxe (* i h) gemb (* (the dim (svref xs i)) h) h)))
                       (let ((gein (grad-of n (net-ein n))))
                         (declare (dvec gein))
                         (dotimes (i b) (axpy 1d0 dxg (* i g) gein (* (the dim (svref xs i)) g) g))))))))
    (* loss scale)))

(defun carry (bf) (replace (svref (buf-hs bf) 0) (svref (buf-hs bf) (buf-tt bf))))
(defun reset (bf) (fill (the dvec (svref (buf-hs bf) 0)) 0d0))

(defun adam-step (n lr &key (b1 0.9d0) (b2 0.999d0) (eps 1d-8) (clip 1d0))
  (declare (double-float lr b1 b2 eps clip))
  (let* ((norm (sqrt (loop for (nil . g) in (net-grads n) sum (let ((g g)) (declare (dvec g)) (dot g 0 g 0 (length g))) of-type double-float)))
         (cs (if (> norm clip) (/ clip norm) 1d0)) (tn (incf (net-adam-t n)))
         (c1 (- 1d0 (expt b1 tn))) (c2 (- 1d0 (expt b2 tn))))
    (declare (double-float norm cs c1 c2))
    (loop for (p . g) in (net-grads n) for (nil . m) in (net-adam-m n) for (nil . v) in (net-adam-v n)
          do (let ((p p) (g g) (m m) (v v)) (declare (dvec p g m v))
               (dotimes (i (length p))
                 (let ((gi (* cs (aref g i))))
                   (setf (aref m i) (+ (* b1 (aref m i)) (* (- 1d0 b1) gi))
                         (aref v i) (+ (* b2 (aref v i)) (* (- 1d0 b2) gi gi)))
                   (decf (aref p i) (/ (* lr (/ (aref m i) c1)) (+ (sqrt (/ (aref v i) c2)) eps)))
                   (setf (aref g i) 0d0)))))))

;;; ---------- gradient check ----------
(defun gradcheck (cell tie drop)
  (let* ((v 5) (b 3) (tt 4) (n (make-net* cell v 6 tie drop)) (bf (make-buf* n b tt))
         (X (coerce (loop repeat tt collect (coerce (loop repeat b collect (random v *rs*)) 'simple-vector)) 'simple-vector))
         (Y (coerce (loop repeat tt collect (coerce (loop repeat b collect (random v *rs*)) 'simple-vector)) 'simple-vector))
         (h0 (let ((a (make-dvec (* b 6)))) (dotimes (i (length a) a) (setf (aref a i) (* 0.5d0 (randn))))))
         (eps 1d-5) (worst 0d0) (cnt 0) (worst-abs 0d0))
    ;; make biases nonzero so their gradients are exercised
    (dolist (np (net-plist n)) (let ((p (second np))) (declare (dvec p)) (dotimes (i (length p)) (incf (aref p i) (* 0.1d0 (randn))))))
    (flet ((loss (bp) (replace (svref (buf-hs bf) 0) h0) (chunk n bf X Y bp t)))
      (let ((*freeze-masks* nil)) (loss t))           ; draws dropout masks once
      (let ((*freeze-masks* t))
        (dolist (np (net-plist n))
          (let* ((p (second np)) (gr (grad-of n p)) (w 0d0))
            (declare (dvec p gr))
            (dotimes (i (length p))
              (let* ((old (aref p i)) (lp (progn (setf (aref p i) (+ old eps)) (loss nil)))
                     (lm (progn (setf (aref p i) (- old eps)) (loss nil))) (num (/ (- lp lm) (* 2 eps))))
                (setf (aref p i) old) (incf cnt)
                (setf worst-abs (max worst-abs (abs (- num (aref gr i)))))
                (setf w (max w (/ (abs (- num (aref gr i))) (max 1d-12 (+ (abs num) (abs (aref gr i)))))))))
            (setf worst (max worst w))
            (format t "    ~4a ~,2e~%" (first np) w)))))
    (format t "gradcheck cell=~a tie=~a drop=~a: ~d params, max relative error ~,3e, max abs error ~,3e~%" cell tie drop cnt worst worst-abs)))

;;; ---------- training ----------
(defun main ()
  (setf *rs* (sb-ext:seed-random-state (parse-integer (opt "seed" "1"))))   ; seed=1 matches earlier runs
  (let* ((full (with-open-file (s "../input.txt" :external-format :utf-8)
                 (let* ((b (make-string (file-length s))) (k (read-sequence b s))) (subseq b 0 k))))
         (chars (coerce (sort (remove-duplicates (coerce full 'list)) #'char<) 'simple-vector))
         (v (length chars)) (c2i (let ((ht (make-hash-table))) (loop for c across chars for i from 0 do (setf (gethash c ht) i)) ht))
         (data (map 'simple-vector (lambda (c) (gethash c c2i)) full))
         (nall (length data)) (vstart (floor (* 0.9 nall)))           ; fixed val set: last 10% of full text
         (small (string= (opt "data" "full") "small"))
         (tend (if small 270000 vstart))                               ; small = old 270k training chars
         (cell (if (string= (opt "cell" "rnn") "gru") :gru :rnn)) (h (parse-integer (opt "h" "256")))
         (tie (string= (opt "tie" "0") "1")) (drop (float (read-from-string (opt "drop" "0")) 1d0))
         (cos (string= (opt "sched" "const") "cos")) (budget (float (parse-integer (opt "secs" "480")) 1d0))
         (n (make-net* cell v h tie drop)) (b 32) (tt 32) (bf (make-buf* n b tt))
         (slen (floor tend b)) (vlen (floor (- nall vstart) 32)) (lr0 2d-3)
         (pos 0) (step 0) (run 0d0) (nrun 0) (t0 (get-internal-real-time)) (hist nil))
    (declare (fixnum pos step nrun) (double-float run))
    (labels ((secs () (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0))
             (ids (base len k) (coerce (loop for i below 32 collect (svref data (+ base (* i len) k))) 'simple-vector))
             (stream-loss (base len)          ; 32 contiguous streams of LEN chars from BASE, carried state
               (let ((eb (make-buf* n 32 1)) (s 0d0))
                 (reset eb)
                 (loop for k below (1- len)
                       do (incf s (chunk n eb (vector (ids base len k)) (vector (ids base len (1+ k))) nil)) (carry eb))
                 (/ s (1- len))))
             (lr-at (frac) (if (not cos) lr0
                               (if (< frac 0.02d0) (* lr0 (max 0.01d0 (/ frac 0.02d0)))
                                   (+ (/ lr0 10) (* 0.9d0 lr0 0.5d0 (+ 1d0 (cos (* pi (/ (- frac 0.02d0) 0.98d0))))))))))
      (format t "config: data=~a cell=~a H=~d tie=~a drop=~,2f sched=~a budget=~ds | vocab=~d params=~d~%"
              (if small "small(270k train)" "full") cell h tie drop (if cos "warmup2%+cosine->lr/10" "const 2e-3") (round budget) v (nparams n))
      (format t "train chars ~d, val chars ~d (fixed: last 10% of full text, 32 streams x ~d); periodic val = first 32x1000 of val~%"
              tend (- nall vstart) vlen)
      (finish-output)
      (reset bf)
      (loop for el = (secs) while (< el budget)
            do (when (> (+ pos tt 1) slen) (setf pos 0) (reset bf))
               (let ((X (coerce (loop for s below tt collect (ids 0 slen (+ pos s))) 'simple-vector))
                     (Y (coerce (loop for s below tt collect (ids 0 slen (+ pos s 1))) 'simple-vector)))
                 (incf run (chunk n bf X Y t)) (incf nrun)
                 (adam-step n (lr-at (/ el budget))) (carry bf) (incf pos tt) (incf step))
               (when (zerop (mod step 200))
                 (let ((vl (stream-loss vstart 1000)))
                   (format t "step ~5d  t=~6,1fs  lr=~,5f  train(batch avg)=~,4f  val(subset)=~,4f~%" step (secs) (lr-at (/ (secs) budget)) (/ run nrun) vl)
                   (push (list step (secs) (/ run nrun) vl) hist)
                   (finish-output) (setf run 0d0 nrun 0))))
      (let ((s (secs)))
        (format t "trained ~d steps in ~,1fs (~,2f steps/s)~%" step s (/ step s))
        (format t "FINAL: train(32x3000 subset)=~,4f  val(full fixed val, ~d chars scored)=~,4f~%"
                (stream-loss 0 3000) (* 32 (1- vlen)) (stream-loss vstart vlen)))
      (with-open-file (f (format nil "~a.csv" (opt "tag" "run")) :direction :output :if-exists :supersede)
        (format f "step,seconds,train_batch,val_subset~%")
        (dolist (r (reverse hist)) (format f "~{~a~^,~}~%" r)))
      (let ((sb (make-buf* n 1 1)) (c (gethash #\Newline c2i)))
        (reset sb)
        (format t "~%=== sample (T=0.8) ===~%~a~%"
                (with-output-to-string (o)
                  (dotimes (k 300)
                    (step-forward n sb (vector c) 0 1 nil)
                    (let* ((P (svref (buf-ps sb) 0)) (w (map 'vector (lambda (p) (expt (max p 1d-300) 1.25d0)) (subseq P 0 v)))
                           (u (* (random 1d0 *rs*) (reduce #'+ w))) (acc 0d0))
                      (setf c (1- v))
                      (dotimes (i v) (incf acc (aref w i)) (when (> acc u) (setf c i) (return)))
                      (write-char (svref chars c) o) (carry sb)))))))))

(if (string= (opt "mode" "train") "gradcheck")
    (progn (gradcheck :rnn nil 0d0) (gradcheck :rnn nil 0.2d0) (gradcheck :gru nil 0d0) (gradcheck :gru nil 0.2d0)
           (gradcheck :rnn t 0d0) (gradcheck :gru t 0.2d0))
    (main))
