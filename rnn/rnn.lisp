;;;; rnn.lisp -- vanilla character RNN with hand-written truncated BPTT, Adam, grad clipping.
;;;; Uses the batched kernels from /home/box/lisp-nn/fast/fnn.lisp (pure Lisp, scalar).
;;;;   h_t = tanh(E[x_t] + h_{t-1} Whh + bh),   p_t = softmax(h_t Why + by)
;;;; Training: the train text is cut into B contiguous streams; each step processes the next T
;;;; characters of every stream, carrying h across steps (detached, i.e. truncated BPTT).
;;;; Usage: sbcl --script rnn.lisp gradcheck | sbcl --script rnn.lisp train SECONDS [H]
(load (compile-file "/home/box/lisp-nn/fast/fnn.lisp" :output-file "/tmp/fnn-rnn.fasl" :verbose nil :print nil))
(in-package :fnn)
(declaim (optimize (speed 3) (safety 0)))

(defstruct (rnn (:conc-name r-))
  v h b tt                         ; vocab, hidden, batch streams, truncation length
  params grads                     ; lists of dvecs: E Whh bh Why by
  hs ps dh dnext dpre adam-m adam-v (adam-t 0))
(defmacro with-params ((r) &body body)
  `(destructuring-bind (E Whh bh Why by) (r-params ,r)
     (destructuring-bind (gE gWhh gbh gWhy gby) (r-grads ,r)
       (declare (dvec E Whh bh Why by gE gWhh gbh gWhy gby) (ignorable E Whh bh Why by gE gWhh gbh gWhy gby))
       ,@body)))

(defun make-rnn* (v h b tt)
  (flet ((rnd (n sc) (let ((a (make-dvec n))) (dotimes (i n a) (setf (aref a i) (* sc (randn)))))))
    (let ((params (list (rnd (* v h) 0.1d0)                      ; E (embedding = input weights)
                        (rnd (* h h) (/ 1d0 (sqrt (float h 1d0)))) ; Whh
                        (make-dvec h)                             ; bh
                        (rnd (* h v) (/ 1d0 (sqrt (float h 1d0)))) ; Why
                        (make-dvec v))))                          ; by
      (make-rnn :v v :h h :b b :tt tt :params params
                :grads (mapcar (lambda (p) (make-dvec (length p))) params)
                :adam-m (mapcar (lambda (p) (make-dvec (length p))) params)
                :adam-v (mapcar (lambda (p) (make-dvec (length p))) params)
                :hs (coerce (loop repeat (1+ tt) collect (make-dvec (* b h))) 'simple-vector) ; hs[0] = carried state
                :ps (coerce (loop repeat tt collect (make-dvec (* b v))) 'simple-vector)
                :dh (make-dvec (* b h)) :dnext (make-dvec (* b h)) :dpre (make-dvec (* b h))))))

(defun n-params (r) (reduce #'+ (mapcar #'length (r-params r))))

(defun step-forward (r xs t* m)
  "One time step for M rows: hs[t+1] = tanh(E[xs] + hs[t] Whh + bh); ps[t] = softmax(hs[t+1] Why + by).
   XS: simple-vector of M input ids."
  (declare (simple-vector xs) (fixnum t* m))
  (with-params (r)
    (let* ((h (r-h r)) (v (r-v r)) (hp (svref (r-hs r) t*)) (hn (svref (r-hs r) (1+ t*))) (P (svref (r-ps r) t*)))
      (declare (dim h v) (dvec hp hn P))
      (dotimes (i m)                                      ; pre = E[x] + bh
        (let ((x (svref xs i)))
          (declare (dim x))
          (dotimes (j h) (setf (aref hn (+ (* i h) j)) (+ (aref E (+ (* x h) j)) (aref bh j))))))
      (mm-add hp Whh hn m h h)                            ; += h_{t-1} Whh
      (dotimes (q (* m h)) (setf (aref hn q) (fast-tanh (aref hn q))))
      (dotimes (i m) (replace P by :start1 (* i v)))      ; logits = h Why + by
      (mm-add hn Why P m h v)
      (dotimes (i m)                                      ; row softmax
        (let* ((o (* i v)) (mx (aref P o)) (s 0d0))
          (declare (idx o) (double-float mx s))
          (loop for j from o below (+ o v) do (setf mx (max mx (aref P j))))
          (loop for j from o below (+ o v) do (setf (aref P j) (exp-neg (- (aref P j) mx))) (incf s (aref P j)))
          (loop for j from o below (+ o v) do (setf (aref P j) (/ (aref P j) s))))))))

(defun chunk (r X Y &optional (backprop t))
  "Forward over T steps (X, Y: vectors of T simple-vectors of B ids); returns mean loss.
   If BACKPROP, accumulates gradients via BPTT. Leaves final hidden state in hs[T]."
  (let* ((b (r-b r)) (tt (r-tt r)) (h (r-h r)) (v (r-v r)) (loss 0d0) (scale (/ 1d0 (* b tt))))
    (declare (dim b tt h v) (double-float loss scale))
    (dotimes (s tt)
      (step-forward r (svref X s) s b)
      (let ((P (svref (r-ps r) s)) (ys (svref Y s)))
        (declare (dvec P) (simple-vector ys))
        (dotimes (i b) (decf loss (log (max 1d-300 (aref P (+ (* i v) (the dim (svref ys i))))))))))
    (when backprop
      (with-params (r)
        (let ((dh (r-dh r)) (dnext (r-dnext r)) (dpre (r-dpre r)))
          (declare (dvec dh dnext dpre))
          (fill dnext 0d0)
          (loop for s of-type fixnum from (1- tt) downto 0
                do (let ((P (svref (r-ps r) s)) (ys (svref Y s)) (xs (svref X s))
                         (hn (svref (r-hs r) (1+ s))) (hp (svref (r-hs r) s)))
                     (declare (dvec P hn hp) (simple-vector ys xs))
                     ;; dlogits = (P - onehot) * scale   (in place in P)
                     (dotimes (q (* b v)) (setf (aref P q) (* scale (aref P q))))
                     (dotimes (i b) (decf (aref P (+ (* i v) (the dim (svref ys i)))) scale))
                     (mtm-add hn P gWhy b h v)                       ; gWhy += h^T dlogits
                     (dotimes (i b) (axpy 1d0 P (* i v) gby 0 v))
                     (mmt P Why dh b v h)                            ; dh = dlogits Why^T
                     (dotimes (q (* b h))                            ; + dnext, through tanh
                       (let ((a (aref hn q)))
                         (setf (aref dpre q) (* (+ (aref dh q) (aref dnext q)) (- 1d0 (* a a))))))
                     (mtm-add hp dpre gWhh b h h)                    ; gWhh += h_{t-1}^T dpre
                     (dotimes (i b)
                       (axpy 1d0 dpre (* i h) gbh 0 h)
                       (axpy 1d0 dpre (* i h) gE (* (the dim (svref xs i)) h) h))
                     (mmt dpre Whh dnext b h h))))))                 ; dnext = dpre Whh^T
    (* loss scale)))

(defun carry-state (r)  (replace (svref (r-hs r) 0) (svref (r-hs r) (r-tt r))))
(defun reset-state (r) (fill (the dvec (svref (r-hs r) 0)) 0d0))

(defun adam-step (r lr &key (b1 0.9d0) (b2 0.999d0) (eps 1d-8) (clip 1d0))
  "Clip global grad norm to CLIP, then Adam update; clears grads."
  (declare (double-float lr b1 b2 eps clip))
  (let* ((norm (sqrt (loop for g in (r-grads r) sum (let ((g g)) (declare (dvec g)) (dot g 0 g 0 (length g))) of-type double-float)))
         (cs (if (> norm clip) (/ clip norm) 1d0))
         (tn (incf (r-adam-t r)))
         (c1 (- 1d0 (expt b1 tn))) (c2 (- 1d0 (expt b2 tn))))
    (declare (double-float norm cs c1 c2))
    (loop for p in (r-params r) for g in (r-grads r) for m in (r-adam-m r) for v in (r-adam-v r)
          do (let ((p p) (g g) (m m) (v v)) (declare (dvec p g m v))
               (dotimes (i (length p))
                 (let ((gi (* cs (aref g i))))
                   (setf (aref m i) (+ (* b1 (aref m i)) (* (- 1d0 b1) gi))
                         (aref v i) (+ (* b2 (aref v i)) (* (- 1d0 b2) gi gi)))
                   (decf (aref p i) (/ (* lr (/ (aref m i) c1)) (+ (sqrt (/ (aref v i) c2)) eps)))
                   (setf (aref g i) 0d0)))))
    norm))

;;; ---------- Gradient check on a tiny RNN ----------
(defun gradcheck ()
  (let* ((r (make-rnn* 5 6 3 4))
         (X (coerce (loop repeat 4 collect (coerce (loop repeat 3 collect (random 5 *rs*)) 'simple-vector)) 'simple-vector))
         (Y (coerce (loop repeat 4 collect (coerce (loop repeat 3 collect (random 5 *rs*)) 'simple-vector)) 'simple-vector))
         (h0 (let ((v (make-dvec 18))) (dotimes (i 18 v) (setf (aref v i) (* 0.5d0 (randn))))))
         (eps 1d-5) (worst 0d0) (n 0))
    (flet ((loss (bp) (replace (svref (r-hs r) 0) h0) (chunk r X Y bp)))
      (loss t)
      (loop for p in (r-params r) for g in (r-grads r) for name in '("E" "Whh" "bh" "Why" "by")
            do (let ((p p) (g g) (w 0d0)) (declare (dvec p g))
                 (dotimes (i (length p))
                   (let* ((old (aref p i)) (lp (progn (setf (aref p i) (+ old eps)) (loss nil)))
                          (lm (progn (setf (aref p i) (- old eps)) (loss nil)))
                          (num (/ (- lp lm) (* 2 eps))))
                     (setf (aref p i) old) (incf n)
                     (setf w (max w (/ (abs (- num (aref g i))) (max 1d-12 (+ (abs num) (abs (aref g i)))))))))
                 (format t "  ~4a max rel err ~,3e~%" name w) (setf worst (max worst w))))
      (format t "RNN gradient check (V=5 H=6 B=3 T=4, ~d params, nonzero h0): max relative error ~,3e~%" n worst))))

;;; ---------- Data & training ----------
(defun main-train (seconds hsize)
  (let* ((text (with-open-file (s "../input.txt" :external-format :utf-8)
                 (let* ((b (make-string 300000)) (n (read-sequence b s))) (subseq b 0 n))))
         (chars (coerce (sort (remove-duplicates (coerce text 'list)) #'char<) 'simple-vector))
         (v (length chars)) (data (map 'simple-vector (lambda (c) (position c chars)) text))
         (n (length data)) (split (floor (* 0.9 n)))
         (b 32) (tt 32) (r (make-rnn* v hsize b tt))
         (slen (floor split b))                      ; train stream length
         (vb 32) (vlen (floor (- n split) vb))       ; val: 32 contiguous streams
         (rv (make-rnn* v hsize vb 1))
         (pos 0) (t0 (get-internal-real-time)) (step 0) (run 0d0) (nrun 0))
    (declare (fixnum pos step nrun) (double-float run))
    (labels ((secs () (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0))
             (ids (start stride count off)
               (coerce (loop for k below count collect (svref data (+ start (* k stride) off))) 'simple-vector))
             (stream-loss (base len nstreams)
               ;; run a 1-step RNN that shares r's weights over NSTREAMS contiguous streams, carrying h;
               ;; scores every char except each stream's first.
               (setf (r-params rv) (r-params r))
               (let ((rr (make-rnn* v hsize nstreams 1)) (s 0d0))
                 (setf (r-params rr) (r-params r)) (reset-state rr)
                 (loop for k from 0 below (1- len)
                       do (incf s (* nstreams (chunk rr (vector (ids base len nstreams k)) (vector (ids base len nstreams (1+ k))) nil)))
                          (carry-state rr))
                 (/ s (* nstreams (1- len))))))
      (format t "Lisp vanilla RNN: vocab=~d hidden=~d params=~d, B=~d streams, T=~d (truncated BPTT, carried state), Adam lr 2e-3, clip 1.0, budget ~ds~%"
              v hsize (n-params r) b tt seconds)
      (format t "train chars ~d (stream length ~d), val chars ~d (~d streams x ~d, ~d scored)~%" split slen (- n split) vb vlen (* vb (1- vlen)))
      (finish-output)
      (reset-state r)
      (loop while (< (secs) seconds)
            do (when (> (+ pos tt 1) slen) (setf pos 0) (reset-state r))   ; new epoch: restart streams
               (let ((X (coerce (loop for s below tt collect (ids 0 slen b (+ pos s))) 'simple-vector))
                     (Y (coerce (loop for s below tt collect (ids 0 slen b (+ pos s 1))) 'simple-vector)))
                 (incf run (chunk r X Y t)) (incf nrun)
                 (adam-step r 2d-3)
                 (carry-state r) (incf pos tt) (incf step))
               (when (zerop (mod step 100))
                 (format t "step ~5d  t=~6,1fs  train(batch avg)=~,4f  val=~,4f~%" step (secs) (/ run nrun) (stream-loss split vlen vb))
                 (finish-output) (setf run 0d0 nrun 0)))
      (let ((s (secs)))
        (format t "trained ~d steps in ~,1fs (~,1f steps/s, ~,0f chars/s)~%" step s (/ step s) (/ (* step b tt) s))
        (format t "final: train=~,4f (32 streams over full train split) val=~,4f~%" (stream-loss 0 slen b) (stream-loss split vlen vb)))
      ;; sample 300 chars at T=0.8 from a newline, single stream
      (let ((rs (make-rnn* v hsize 1 1)) (c (position #\Newline chars)))
        (setf (r-params rs) (r-params r)) (reset-state rs)
        (format t "~%=== Lisp RNN sample (T=0.8) ===~%~a~%"
                (with-output-to-string (o)
                  (dotimes (k 300)
                    (step-forward rs (vector c) 0 1)
                    (let* ((P (svref (r-ps rs) 0)) (w (map 'vector (lambda (p) (expt (max p 1d-300) 1.25d0)) (subseq P 0 v)))
                           (u (* (random 1d0 *rs*) (reduce #'+ w))) (acc 0d0))
                      (setf c (1- v))
                      (dotimes (i v) (incf acc (aref w i)) (when (> acc u) (setf c i) (return)))
                      (write-char (svref chars c) o) (carry-state rs)))))))))

(let ((args sb-ext:*posix-argv*))
  (if (equal (second args) "gradcheck") (gradcheck)
      (main-train (parse-integer (or (third args) "480")) (parse-integer (or (fourth args) "256")))))
