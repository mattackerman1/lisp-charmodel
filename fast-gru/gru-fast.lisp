;;;; gru-fast.lisp -- fast GRU char model (same model/data/protocol as ../ablation/rnn2.lisp cell=gru h=144).
;;;; Compile-time switches (environment):
;;;;   FT=double|single    element type (float64 / float32)
;;;;   MM=lisp|blas        matmul backend: pure-Lisp kernels, or OpenBLAS cblas_?gemm via sb-alien
;;;; Runtime options (key=value): threads=1..8 (streams split across sb-threads, grads summed)
;;;;   secs=480 h=144 seed=1 tag=name mode=train|gradcheck blas-threads=1
;;;; Restructured for batching: all T steps' hidden states are contiguous, so the output layer
;;;; (forward, dWhy, dh) and dWh are single big GEMMs per chunk; only h.Wh and dnext.Wh^T stay per step.
(defpackage :gf (:use :cl)) (in-package :gf)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *ft* (if (equal (sb-ext:posix-getenv "FT") "single") 'single-float 'double-float))
  (when (equal (sb-ext:posix-getenv "MM") "blas") (pushnew :gf-blas *features*)))
(declaim (optimize (speed 3) (safety 0) (debug 0)))
(deftype fl () *ft*)
(deftype fvec () `(simple-array ,*ft* (*)))
(deftype idx () '(unsigned-byte 40))
(deftype dim () '(unsigned-byte 20))
(defmacro f (x) (coerce x *ft*))                 ; literal constant in the working precision
(declaim (inline mkv)) (defun mkv (n) (make-array n :element-type *ft* :initial-element (f 0)))

(defun opt (key default)
  (let ((a (find-if (lambda (s) (and (> (length s) (length key)) (string= (concatenate 'string key "=") s :end2 (1+ (length key)))))
                    sb-ext:*posix-argv*)))
    (if a (subseq a (1+ (length key))) default)))
(defvar *rs* (sb-ext:seed-random-state 1))
(defun randn () (* (sqrt (* -2d0 (log (- 1d0 (random 1d0 *rs*))))) (cos (* 2 pi (random 1d0 *rs*)))))

;;; ---------- exp / tanh / sigmoid (computed in double, returned in working precision) ----------
(defmacro horner (x &rest cs) (if (cdr cs) `(+ ,(car cs) (* ,x (horner ,x ,@(cdr cs)))) (car cs)))
(declaim (inline exp-neg tanh* sig*))
(defun exp-neg (y)
  (declare (double-float y))
  (if (< y -700d0) 0d0
      (let* ((n (floor (+ (* (the (double-float -700d0 0d0) y) 1.4426950408889634d0) 0.5d0)))
             (r (- (- y (* n 0.6931471803691238d0)) (* n 1.9082149292705877d-10)))
             (p (horner r 1d0 1d0 0.5d0 1.6666666666666666d-1 4.1666666666666664d-2 8.333333333333333d-3
                        1.388888888888889d-3 1.984126984126984d-4 2.48015873015873d-5 2.7557319223985893d-6
                        2.755731922398589d-7 2.505210838544172d-8 2.08767569878681d-9)))
        (declare (type (integer -1010 1) n))
        (* p (sb-kernel:make-double-float (ash (+ n 1023) 20) 0)))))
(defun tanh* (x)
  (declare (type fl x))
  (let ((x (coerce x 'double-float)))
    (coerce (if (> (abs x) 19d0) (if (> x 0d0) 1d0 -1d0)
                (let* ((e (exp-neg (* -2d0 (abs x)))) (th (/ (- 1d0 e) (+ 1d0 e)))) (if (< x 0d0) (- th) th)))
            'fl)))
(defun sig* (x)
  (declare (type fl x))
  (let ((x (coerce x 'double-float)))
    (coerce (if (>= x 0d0) (/ 1d0 (+ 1d0 (exp-neg (- x)))) (let ((e (exp-neg x))) (/ e (+ 1d0 e)))) 'fl)))

;;; ---------- GEMM: row-major, C (+)= op(A) op(B); three shapes used ----------
;;; nn: C[m,n] (+)= A[m,k] B[k,n]     nt: C[m,k] = D[m,n] W[k,n]^T     tn: G[k,n] += A[m,k]^T D[m,n]
#+gf-blas
(progn
  (sb-alien:load-shared-object "/usr/lib/x86_64-linux-gnu/libopenblas.so.0")
  (macrolet ((def (name cname ty)
               `(sb-alien:define-alien-routine (,cname ,name) sb-alien:void
                  (order sb-alien:int) (ta sb-alien:int) (tb sb-alien:int) (m sb-alien:int) (n sb-alien:int) (k sb-alien:int)
                  (alpha ,ty) (a sb-sys:system-area-pointer) (lda sb-alien:int) (b sb-sys:system-area-pointer) (ldb sb-alien:int)
                  (beta ,ty) (c sb-sys:system-area-pointer) (ldc sb-alien:int))))
    (def %dgemm "cblas_dgemm" sb-alien:double)
    (def %sgemm "cblas_sgemm" sb-alien:single-float))
  (sb-alien:define-alien-routine ("openblas_set_num_threads" %set-threads) sb-alien:void (n sb-alien:int))
  (defconstant +esz+ (if (eq *ft* 'single-float) 4 8))
  (defmacro gemm* (ta tb m n k a aoff lda b boff ldb beta c coff ldc)
    `(sb-sys:with-pinned-objects (,a ,b ,c)
       (,(if (eq *ft* 'single-float) '%sgemm '%dgemm)
        101 ,ta ,tb ,m ,n ,k (f 1)
        (sb-sys:sap+ (sb-sys:vector-sap ,a) (* +esz+ ,aoff)) ,lda
        (sb-sys:sap+ (sb-sys:vector-sap ,b) (* +esz+ ,boff)) ,ldb
        ,beta (sb-sys:sap+ (sb-sys:vector-sap ,c) (* +esz+ ,coff)) ,ldc)))
  (defun gemm-nn (A ao B bo C co m k n accumulate)
    (declare (fvec A B C) (idx ao bo co) (dim m k n))
    (gemm* 111 111 m n k A ao k B bo n (if accumulate (f 1) (f 0)) C co n))
  (defun gemm-nt (D do W wo X xo m n k)
    (declare (fvec D W X) (idx do wo xo) (dim m n k))
    (gemm* 111 112 m k n D do n W wo n (f 0) X xo k))
  (defun gemm-tn (A ao D do G go m k n)
    (declare (fvec A D G) (idx ao do go) (dim m k n))
    (gemm* 112 111 k n m A ao k D do n (f 1) G go n)))

#-gf-blas
(progn
  (declaim (inline axpy* dot*))
  (defun axpy* (a x xo y yo n)            ; y[yo..] += a x[xo..]
    (declare (type fl a) (fvec x y) (idx xo yo) (dim n))
    (let ((n4 (- n (mod n 4))))
      (declare (dim n4))
      (loop for j of-type dim from 0 below n4 by 4
            do (incf (aref y (+ yo j)) (* a (aref x (+ xo j)))) (incf (aref y (+ yo j 1)) (* a (aref x (+ xo j 1))))
               (incf (aref y (+ yo j 2)) (* a (aref x (+ xo j 2)))) (incf (aref y (+ yo j 3)) (* a (aref x (+ xo j 3)))))
      (loop for j of-type dim from n4 below n do (incf (aref y (+ yo j)) (* a (aref x (+ xo j)))))))
  (defun dot* (x xo y yo n)
    (declare (fvec x y) (idx xo yo) (dim n))
    (let ((s0 (f 0)) (s1 (f 0)) (s2 (f 0)) (s3 (f 0)) (n4 (- n (mod n 4))))
      (declare (type fl s0 s1 s2 s3) (dim n4))
      (loop for j of-type dim from 0 below n4 by 4
            do (incf s0 (* (aref x (+ xo j)) (aref y (+ yo j)))) (incf s1 (* (aref x (+ xo j 1)) (aref y (+ yo j 1))))
               (incf s2 (* (aref x (+ xo j 2)) (aref y (+ yo j 2)))) (incf s3 (* (aref x (+ xo j 3)) (aref y (+ yo j 3)))))
      (loop for j of-type dim from n4 below n do (incf s0 (* (aref x (+ xo j)) (aref y (+ yo j)))))
      (+ s0 s1 s2 s3)))
  (defun gemm-nn (A ao B bo C co m k n accumulate)
    (declare (fvec A B C) (idx ao bo co) (dim m k n))
    (unless accumulate (fill C (f 0) :start co :end (+ co (* m n))))
    (loop for i0 of-type dim from 0 below m by 32          ; 32-row blocks keep C rows in cache
          do (let ((i1 (min m (+ i0 32))))
               (dotimes (kk k)
                 (loop for i of-type dim from i0 below i1
                       do (axpy* (aref A (+ ao (* i k) kk)) B (+ bo (* kk n)) C (+ co (* i n)) n))))))
  (defun gemm-nt (D do W wo X xo m n k)
    (declare (fvec D W X) (idx do wo xo) (dim m n k))
    (dotimes (i m) (dotimes (kk k) (setf (aref X (+ xo (* i k) kk)) (dot* D (+ do (* i n)) W (+ wo (* kk n)) n)))))
  (defun gemm-tn (A ao D do G go m k n)
    (declare (fvec A D G) (idx ao do go) (dim m k n))
    (dotimes (kk k)
      (dotimes (i m) (axpy* (aref A (+ ao (* i k) kk)) D (+ do (* i n)) G (+ go (* kk n)) n)))))

;;; ---------- model ----------
(defstruct (net (:conc-name n-)) (v 0 :type fixnum) (h 0 :type fixnum) (g 0 :type fixnum) params (adam-t 0 :type fixnum) m v2)
;; params: plist-like list of fvecs in order: Ein bx Wh bh Why by
(defun make-net* (v h)
  (let* ((g (* 3 h)) (s (/ 1d0 (sqrt (float h 1d0)))))
    (flet ((rnd (k sc) (let ((a (mkv k))) (dotimes (i k a) (setf (aref a i) (coerce (* sc (randn)) 'fl))))))
      ;; same draw order as rnn2.lisp (Ein, Wh, Why)
      (let* ((ein (rnd (* v g) 0.1d0)) (why (rnd (* h v) s)) (bx (mkv g)) (wh (rnd (* h g) s)) (bh (mkv g)) (by (mkv v))
             (ps (list ein bx wh bh why by)))
        (make-net :v v :h h :g g :params ps
                  :m (mapcar (lambda (p) (mkv (length p))) ps) :v2 (mapcar (lambda (p) (mkv (length p))) ps))))))
(defun nparams (n) (reduce #'+ (mapcar #'length (n-params n))))

;;; A worker owns B rows (streams), T-step activation buffers and its own gradient copies.
(defstruct (wk (:conc-name w-)) (b 0 :type fixnum) (tt 0 :type fixnum) hall gates hnh hg P dhall dhg dxg dnext dprev grads
  xs ys (loss 0d0 :type double-float) (count 0 :type fixnum))
(defun make-wk* (n b tt)
  (let ((h (n-h n)) (g (n-g n)) (v (n-v n)))
    (make-wk :b b :tt tt :hall (mkv (* (1+ tt) b h)) :gates (mkv (* tt b g)) :hnh (mkv (* tt b h)) :hg (mkv (* b g))
             :P (mkv (* tt b v)) :dhall (mkv (* tt b h)) :dhg (mkv (* tt b g)) :dxg (mkv (* b g))
             :dnext (mkv (* b h)) :dprev (mkv (* b h))
             :grads (mapcar (lambda (p) (mkv (length p))) (n-params n))
             :xs (make-array (* tt b) :element-type 'fixnum) :ys (make-array (* tt b) :element-type 'fixnum))))

(defun run-chunk (n w nt backprop scale)
  "Forward NT (<= tt) steps on w's ids (xs/ys, index t*b+i); accumulates sum of -log p into w-loss.
   If BACKPROP, adds gradients of SCALE * sum(-log p) to w's grads."
  (declare (dim nt) (type fl scale))
  (destructuring-bind (ein bx wh bh why by) (n-params n)
    (declare (fvec ein bx wh bh why by))
    (let* ((h (n-h n)) (g (n-g n)) (v (n-v n)) (b (w-b w)) (bh* (* b h)) (bg (* b g))
           (hall (w-hall w)) (gates (w-gates w)) (hnh (w-hnh w)) (hg (w-hg w)) (P (w-P w))
           (xs (w-xs w)) (ys (w-ys w)) (loss 0d0))
      (declare (dim h g v b) (idx bh* bg) (fvec hall gates hnh hg P) ((simple-array fixnum (*)) xs ys) (double-float loss))
      ;; ---- forward, step by step through the recurrence ----
      (dotimes (s nt)
        (let ((go (* s bg)) (ho (* s bh*)) (hno (* (1+ s) bh*)))
          (declare (idx go ho hno))
          (dotimes (i b)
            (replace gates ein :start1 (+ go (* i g)) :start2 (* (aref xs (+ (* s b) i)) g) :end2 (* (1+ (aref xs (+ (* s b) i))) g))
            (let ((o (+ go (* i g)))) (declare (idx o)) (dotimes (j g) (incf (aref gates (+ o j)) (aref bx j))))
            (replace hg bh :start1 (* i g)))
          (gemm-nn hall ho wh 0 hg 0 b h g t)                     ; hg += h_{t-1} Wh
          (dotimes (i b)
            (let ((o (+ go (* i g))) (oq (* i g)) (oh (* i h)))
              (declare (idx o oq oh))
              (dotimes (j h)
                (let* ((z (sig* (+ (aref gates (+ o j)) (aref hg (+ oq j)))))
                       (r (sig* (+ (aref gates (+ o h j)) (aref hg (+ oq h j)))))
                       (hh (aref hg (+ oq h h j)))
                       (nn (tanh* (+ (aref gates (+ o h h j)) (* r hh)))))
                  (setf (aref gates (+ o j)) z (aref gates (+ o h j)) r (aref gates (+ o h h j)) nn
                        (aref hnh (+ ho oh j)) hh
                        (aref hall (+ hno oh j)) (+ (* (- (f 1) z) nn) (* z (aref hall (+ ho oh j)))))))))))
      ;; ---- output layer for all steps at once ----
      (let ((rows (* nt b)))
        (declare (idx rows))
        (dotimes (q rows) (replace P by :start1 (* q v)))
        (gemm-nn hall bh* why 0 P 0 rows h v t)
        (dotimes (q rows)
          (let* ((o (* q v)) (mx (aref P o)) (sm 0d0))
            (declare (idx o) (type fl mx) (double-float sm))
            (loop for j from o below (+ o v) do (setf mx (max mx (aref P j))))
            (loop for j from o below (+ o v)
                  do (let ((e (exp-neg (coerce (- (aref P j) mx) 'double-float)))) (setf (aref P j) (coerce e 'fl)) (incf sm e)))
            (let ((inv (/ 1d0 sm)))
              (decf loss (log (max 1d-300 (* inv (coerce (aref P (+ o (aref ys q))) 'double-float)))))
              (loop for j from o below (+ o v) do (setf (aref P j) (coerce (* inv (aref P j)) 'fl)))))))
      (incf (w-loss w) loss) (incf (w-count w) (* nt b))
      (when backprop
        (destructuring-bind (gein gbx gwh gbh gwhy gby) (w-grads w)
          (declare (fvec gein gbx gwh gbh gwhy gby))
          (let ((rows (* nt b)) (dhall (w-dhall w)) (dhg (w-dhg w)) (dxg (w-dxg w)) (dnext (w-dnext w)) (dprev (w-dprev w)))
            (declare (idx rows) (fvec dhall dhg dxg dnext dprev))
            (dotimes (q (* rows v)) (setf (aref P q) (* scale (aref P q))))           ; dlogits
            (dotimes (q rows) (decf (aref P (+ (* q v) (aref ys q))) scale))
            (dotimes (q rows) (let ((o (* q v))) (dotimes (j v) (incf (aref gby j) (aref P (+ o j))))))
            (gemm-tn hall bh* P 0 gwhy 0 rows h v)                                     ; dWhy += H^T dlogits
            (gemm-nt P 0 why 0 dhall 0 rows v h)                                       ; dH = dlogits Why^T
            (fill dnext (f 0))
            (loop for s of-type fixnum from (1- nt) downto 0
                  do (let ((go (* s bg)) (ho (* s bh*)))
                       (declare (idx go ho))
                       (dotimes (i b)
                         (let ((o (+ go (* i g))) (oq (* i g)) (oh (* i h)))
                           (declare (idx o oq oh))
                           (dotimes (j h)
                             (let* ((z (aref gates (+ o j))) (r (aref gates (+ o h j))) (nn (aref gates (+ o h h j)))
                                    (hh (aref hnh (+ ho oh j))) (d (+ (aref dhall (+ ho oh j)) (aref dnext (+ oh j))))
                                    (dn (* d (- (f 1) z))) (dz (* d (- (aref hall (+ ho oh j)) nn)))
                                    (dan (* dn (- (f 1) (* nn nn)))) (dr (* dan hh))
                                    (daz (* dz z (- (f 1) z))) (dar (* dr r (- (f 1) r))))
                               (setf (aref dprev (+ oh j)) (* d z)
                                     (aref dxg (+ oq j)) daz (aref dxg (+ oq h j)) dar (aref dxg (+ oq h h j)) dan
                                     (aref dhg (+ o j)) daz (aref dhg (+ o h j)) dar (aref dhg (+ o h h j)) (* dan r))))
                           ;; input side: gbx, scatter into Ein row
                           (let ((eo (* (aref xs (+ (* s b) i)) g)))
                             (declare (idx eo))
                             (dotimes (j g) (incf (aref gbx j) (aref dxg (+ oq j))) (incf (aref gein (+ eo j)) (aref dxg (+ oq j)))))))
                       (gemm-nt dhg go wh 0 dnext 0 b g h)                             ; dnext = dhg_t Wh^T
                       (dotimes (q bh*) (incf (aref dnext q) (aref dprev q)))))
            (gemm-tn hall 0 dhg 0 gwh 0 rows h g)                                      ; dWh += H_{t-1}^T dHg (all t)
            (dotimes (q rows) (let ((o (* q g))) (dotimes (j g) (incf (aref gbh j) (aref dhg (+ o j))))))))))))

(defun carry (w nt)   ; hidden state after step NT becomes the next chunk's h0
  (let ((bh (length (the fvec (w-dnext w)))))
    (replace (the fvec (w-hall w)) (the fvec (w-hall w)) :start1 0 :start2 (* nt bh) :end2 (* (1+ nt) bh))))
(defun reset (w) (fill (the fvec (w-hall w)) (f 0) :end (length (the fvec (w-dnext w)))))

(defun parallel (workers fn)
  "Run (FN worker) for each worker; one sb-thread per extra worker."
  (if (null (cdr workers)) (funcall fn (car workers))
      (let ((ths (mapcar (lambda (w) (sb-thread:make-thread (lambda () (funcall fn w)))) (cdr workers))))
        (funcall fn (car workers))
        (mapc #'sb-thread:join-thread ths))))

(defun sum-grads (workers)
  "Add all workers' grads into the first worker's; zero the others."
  (let ((g0 (w-grads (car workers))))
    (dolist (w (cdr workers))
      (loop for a in g0 for b in (w-grads w)
            do (let ((a a) (b b)) (declare (fvec a b)) (dotimes (i (length a)) (incf (aref a i) (aref b i)) (setf (aref b i) (f 0))))))
    g0))

(defun adam-step (n grads lr)
  (declare (double-float lr))
  (let* ((norm (sqrt (loop for gr in grads sum (let ((gr gr)) (declare (fvec gr)) (loop for x across gr sum (* (coerce x 'double-float) x) of-type double-float)) of-type double-float)))
         (cs (coerce (if (> norm 1d0) (/ 1d0 norm) 1d0) 'fl)) (tn (incf (n-adam-t n)))
         (c1 (- 1d0 (expt 0.9d0 tn))) (c2 (- 1d0 (expt 0.999d0 tn)))
         (lrc (coerce (/ lr c1) 'fl)) (ic2 (coerce (/ 1d0 c2) 'fl)))
    (loop for p in (n-params n) for g in grads for m in (n-m n) for v in (n-v2 n)
          do (let ((p p) (g g) (m m) (v v)) (declare (fvec p g m v))
               (dotimes (i (length p))
                 (let ((gi (* cs (aref g i))))
                   (setf (aref m i) (+ (* (f 0.9) (aref m i)) (* (f 0.1) gi))
                         (aref v i) (+ (* (f 0.999) (aref v i)) (* (f 0.001) gi gi)))
                   (decf (aref p i) (/ (* lrc (aref m i)) (+ (sqrt (* (aref v i) ic2)) (f 1e-8))))
                   (setf (aref g i) (f 0))))))))

;;; ---------- gradient check (run with FT=double for a tight check) ----------
(defun gradcheck ()
  (let* ((v 5) (h 4) (tt 3) (k 2) (b 2) (n (make-net* v h)) (ws (loop repeat k collect (make-wk* n b tt)))
         (scale (coerce (/ 1d0 (* k b tt)) 'fl)) (single (eq *ft* 'single-float))
         (eps (if single 1d-2 1d-5)) (h0s (loop repeat k collect (let ((a (mkv (* b h)))) (dotimes (i (length a) a) (setf (aref a i) (coerce (* 0.5 (randn)) 'fl)))))))
    (dolist (p (n-params n)) (let ((p p)) (declare (fvec p)) (dotimes (i (length p)) (incf (aref p i) (coerce (* 0.1 (randn)) 'fl)))))
    (dolist (w ws) (dotimes (q (* tt b)) (setf (aref (w-xs w) q) (random v *rs*) (aref (w-ys w) q) (random v *rs*))))
    (flet ((loss (bp) (loop for w in ws for h0 in h0s
                            do (replace (the fvec (w-hall w)) (the fvec h0)) (setf (w-loss w) 0d0)
                               (run-chunk n w tt bp scale))
                 (* (coerce scale 'double-float) (loop for w in ws sum (w-loss w)))))
      (loss t)
      (let ((grads (sum-grads ws)) (worst 0d0) (wa 0d0) (cnt 0) (sd2 0d0) (sn2 0d0))
        (loop for p in (n-params n) for g in grads for name in '("Ein" "bx" "Wh" "bh" "Why" "by")
              do (let ((p p) (g g) (w 0d0)) (declare (fvec p g))
                   (dotimes (i (length p))
                     (let* ((old (aref p i))
                            (lp (progn (setf (aref p i) (coerce (+ old eps) 'fl)) (loss nil)))
                            (lm (progn (setf (aref p i) (coerce (- old eps) 'fl)) (loss nil)))
                            (num (/ (- lp lm) (- (coerce (coerce (+ old eps) 'fl) 'double-float) (coerce (coerce (- old eps) 'fl) 'double-float))))
                            (an (coerce (aref g i) 'double-float)))
                       (setf (aref p i) old) (incf cnt) (incf sd2 (expt (- num an) 2)) (incf sn2 (expt num 2))
                       (setf wa (max wa (abs (- num an))) w (max w (/ (abs (- num an)) (max 1d-12 (+ (abs num) (abs an))))))))
                   (setf worst (max worst w)) (format t "    ~4a max rel err ~,3e~%" name w)))
        (format t "gradcheck FT=~a MM=~a, 2 workers x 2 streams (tests grad summing), ~d params, eps=~a: max rel err ~,3e, max abs err ~,3e, global rel err |g-num|/|num| = ~,3e~%"
                *ft* #+gf-blas "blas" #-gf-blas "lisp" cnt eps worst wa (sqrt (/ sd2 sn2)))))))

;;; ---------- training ----------
(defun main ()
  (setf *rs* (sb-ext:seed-random-state (parse-integer (opt "seed" "1"))))
  #+gf-blas (%set-threads (parse-integer (opt "blas-threads" "1")))
  (let* ((full (with-open-file (s "../input.txt" :external-format :utf-8)
                 (let* ((b (make-string (file-length s))) (k (read-sequence b s))) (subseq b 0 k))))
         (chars (coerce (sort (remove-duplicates (coerce full 'list)) #'char<) 'simple-vector))
         (v (length chars)) (c2i (let ((ht (make-hash-table))) (loop for c across chars for i from 0 do (setf (gethash c ht) i)) ht))
         (data (map '(simple-array fixnum (*)) (lambda (c) (gethash c c2i)) full))
         (nall (length data)) (vstart (floor (* 0.9 nall))) (tend vstart)
         (h (parse-integer (opt "h" "144"))) (k (parse-integer (opt "threads" "1")))
         (budget (float (parse-integer (opt "secs" "480")) 1d0))
         (n (make-net* v h)) (bt 32) (tt 32) (b (/ bt k))
         (ws (loop repeat k collect (make-wk* n b tt))) (evs (loop repeat k collect (make-wk* n b tt)))
         (slen (floor tend bt)) (vlen (floor (- nall vstart) bt)) (scale (coerce (/ 1d0 (* bt tt)) 'fl))
         (pos 0) (step 0) (t0 (get-internal-real-time)) (hist nil) (eval-secs 0d0))
    (declare (fixnum pos step) (double-float eval-secs))
    (assert (= (* b k) bt))
    (labels ((secs () (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0))
             (fill-ids (wlist base len p nt)
               (loop for w in wlist for kk from 0
                     do (let ((xs (w-xs w)) (ys (w-ys w)))
                          (dotimes (s nt)
                            (dotimes (i b)
                              (let ((src (+ base (* (+ (* kk b) i) len) p s)))
                                (setf (aref xs (+ (* s b) i)) (aref data src) (aref ys (+ (* s b) i)) (aref data (1+ src)))))))))
             (stream-loss (base len)
               (let ((e0 (secs)))
                 (dolist (w evs) (reset w) (setf (w-loss w) 0d0 (w-count w) 0))
                 (loop for p from 0 below (1- len) by tt
                       do (let ((nt (min tt (- len 1 p))))
                            (fill-ids evs base len p nt)
                            (parallel evs (lambda (w) (run-chunk n w nt nil scale) (carry w nt)))))
                 (prog1 (/ (loop for w in evs sum (w-loss w)) (loop for w in evs sum (w-count w)))
                   (incf eval-secs (- (secs) e0))))))
      (format t "config: FT=~a MM=~a threads=~d (x ~d streams each)~a | GRU H=~d vocab=~d params=~d | B=32 T=32 Adam 2e-3 clip 1 | budget ~ds~%"
              *ft* #+gf-blas "blas" #-gf-blas "lisp" k b #+gf-blas (format nil " blas-threads=~a" (opt "blas-threads" "1")) #-gf-blas ""
              h v (nparams n) (round budget))
      (format t "train chars ~d, val chars ~d (fixed last 10%); periodic val = first 32x1000 of val every 200 steps~%" tend (- nall vstart))
      (finish-output)
      (dolist (w ws) (reset w))
      (loop for el = (secs) while (< el budget)
            do (when (> (+ pos tt 1) slen) (setf pos 0) (dolist (w ws) (reset w)))
               (fill-ids ws 0 slen pos tt)
               (dolist (w ws) (setf (w-loss w) 0d0 (w-count w) 0))
               (parallel ws (lambda (w) (run-chunk n w tt t scale)))
               (adam-step n (sum-grads ws) 2d-3)
               (dolist (w ws) (carry w tt))
               (incf pos tt) (incf step)
               (when (zerop (mod step 200))
                 (let ((tl (/ (loop for w in ws sum (w-loss w)) (loop for w in ws sum (w-count w)))) (vl (stream-loss vstart 1000)))
                   (format t "step ~5d  t=~6,1fs  train(last batch)=~,4f  val(subset)=~,4f~%" step (secs) tl vl)
                   (push (list step (secs) tl vl) hist) (finish-output))))
      (let ((s (secs)))
        (format t "trained ~d steps in ~,1fs: ~,2f steps/s, ~,0f chars/s (incl. periodic val ~,1fs)~%" step s (/ step s) (/ (* step bt tt) s) eval-secs)
        (format t "FINAL: train(32x3000 subset)=~,4f  val(full fixed val)=~,4f~%" (stream-loss 0 3000) (stream-loss vstart vlen)))
      (with-open-file (fo (format nil "~a.csv" (opt "tag" "run")) :direction :output :if-exists :supersede)
        (format fo "step,seconds,train_batch,val_subset~%") (dolist (r (reverse hist)) (format fo "~{~a~^,~}~%" r)))
      (let ((w (make-wk* n 1 1)) (c (gethash #\Newline c2i)))
        (reset w)
        (format t "~%=== sample (T=0.8) ===~%~a~%"
                (with-output-to-string (o)
                  (dotimes (q 300)
                    (setf (aref (w-xs w) 0) c (aref (w-ys w) 0) 0)
                    (run-chunk n w 1 nil (f 1)) (carry w 1)
                    (let* ((P (w-P w)) (wt (map 'vector (lambda (p) (expt (max (coerce p 'double-float) 1d-300) 1.25d0)) (subseq P 0 v)))
                           (u (* (random 1d0 *rs*) (reduce #'+ wt))) (acc 0d0))
                      (setf c (1- v))
                      (dotimes (i v) (incf acc (aref wt i)) (when (> acc u) (setf c i) (return)))
                      (write-char (svref chars c) o)))))))))

(if (string= (opt "mode" "train") "gradcheck") (gradcheck) (main))
