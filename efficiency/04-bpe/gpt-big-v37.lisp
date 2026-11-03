;;;; gpt-big.lisp -- gpt-drop.lisp + data= option, budget counts training time only (eval excluded), periodic full val.
;;;; gpt-drop.lisp -- gpt.lisp + inverted dropout (embeddings, attention probs, attn/MLP residual branch outputs).
;;;; gpt.lisp -- small decoder-only character transformer in Common Lisp with hand-written backprop.
;;;; Pre-LN GPT block: x += Attn(LN1(x)); x += MLP(LN2(x)); final LN; LM head tied to token embedding.
;;;; Token + learned position embeddings, causal multi-head attention, GELU (tanh approx) MLP,
;;;; AdamW (decoupled weight decay on weight matrices/embeddings), linear warmup + cosine to lr/10 vs wall-clock.
;;;; Matmuls: OpenBLAS cblas_sgemm / cblas_dgemm via sb-alien. Batch sequences split across sb-threads.
;;;; Compile-time: FT=single|double (default single).  Runtime key=value options: see MAIN.
(defpackage :gpt (:use :cl)) (in-package :gpt)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *ft* (if (equal (sb-ext:posix-getenv "FT") "double") 'double-float 'single-float)))
(declaim (optimize (speed 3) (safety 0) (debug 0)))
(deftype fl () *ft*)
(deftype fvec () `(simple-array ,*ft* (*)))
(deftype idx () '(unsigned-byte 40))
(deftype dim () '(unsigned-byte 24))
(defmacro f (x) (coerce x *ft*))
(declaim (inline mkv)) (defun mkv (n) (make-array n :element-type *ft* :initial-element (f 0)))
(defun opt (key default)
  (let ((a (find-if (lambda (s) (and (> (length s) (length key)) (string= (concatenate 'string key "=") s :end2 (1+ (length key)))))
                    sb-ext:*posix-argv*)))
    (if a (subseq a (1+ (length key))) default)))
(defvar *rs* (sb-ext:seed-random-state 1))
(defvar *pdrop* 0d0)
(defun randn () (* (sqrt (* -2d0 (log (- 1d0 (random 1d0 *rs*))))) (cos (* 2 pi (random 1d0 *rs*)))))

;;; ---------- math helpers (double internally) ----------
(defmacro horner (x &rest cs) (if (cdr cs) `(+ ,(car cs) (* ,x (horner ,x ,@(cdr cs)))) (car cs)))
(declaim (inline exp-neg tanh-d))
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
(defun tanh-d (x)
  (declare (double-float x))
  (if (> (abs x) 19d0) (if (> x 0d0) 1d0 -1d0)
      (let* ((e (exp-neg (* -2d0 (abs x)))) (th (/ (- 1d0 e) (+ 1d0 e)))) (if (< x 0d0) (- th) th))))

;;; ---------- BLAS ----------
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
(defun gemm (ta tb m n k alpha A ao lda B bo ldb beta C co ldc)
  "Row-major C[m,n] = alpha op(A) op(B) + beta C; TA/TB true = transpose. Offsets in elements."
  (declare (fvec A B C) (idx ao bo co) (dim m n k lda ldb ldc) (type fl alpha beta))
  (sb-sys:with-pinned-objects (A B C)
    (#.(if (eq *ft* 'single-float) '%sgemm '%dgemm)
     101 (if ta 112 111) (if tb 112 111) m n k alpha
     (sb-sys:sap+ (sb-sys:vector-sap A) (* +esz+ ao)) lda
     (sb-sys:sap+ (sb-sys:vector-sap B) (* +esz+ bo)) ldb
     beta (sb-sys:sap+ (sb-sys:vector-sap C) (* +esz+ co)) ldc)))

;;; ---------- parameters ----------
;;; Per-layer parameter vector indices
(defconstant +ln1g+ 0) (defconstant +ln1b+ 1) (defconstant +wqkv+ 2) (defconstant +bqkv+ 3)
(defconstant +wproj+ 4) (defconstant +bproj+ 5) (defconstant +ln2g+ 6) (defconstant +ln2b+ 7)
(defconstant +wfc+ 8) (defconstant +bfc+ 9) (defconstant +wfc2+ 10) (defconstant +bfc2+ 11)

(defstruct (cfg (:conc-name c-)) (v 0 :type fixnum) (d 0 :type fixnum) (nl 0 :type fixnum) (nh 0 :type fixnum) (tt 0 :type fixnum))
(defstruct (model (:conc-name m-)) cfg wte wpe layers lnfg lnfb)   ; layers: list of simple-vectors of 12 fvecs

(defun model-vecs (m)
  "Flat list of (fvec . decay?) in a fixed order."
  (append (list (cons (m-wte m) t) (cons (m-wpe m) t))
          (loop for ly in (m-layers m)
                append (loop for i below 12 collect (cons (svref ly i) (member i (list +wqkv+ +wproj+ +wfc+ +wfc2+)))))
          (list (cons (m-lnfg m) nil) (cons (m-lnfb m) nil))))

(defun make-model* (c &optional (zero nil))
  (let* ((v (c-v c)) (d (c-d c)) (nl (c-nl c)) (tt (c-tt c)) (rstd (/ 0.02d0 (sqrt (* 2d0 nl)))))
    (flet ((rnd (n sd) (let ((a (mkv n))) (unless zero (dotimes (i n) (setf (aref a i) (coerce (* sd (randn)) 'fl)))) a))
           (ones (n) (let ((a (mkv n))) (unless zero (fill a (f 1))) a)))
      (make-model :cfg c :wte (rnd (* v d) 0.02d0) :wpe (rnd (* tt d) 0.02d0)
                  :layers (loop repeat nl collect
                                (vector (ones d) (mkv d) (rnd (* d 3 d) 0.02d0) (mkv (* 3 d))
                                        (rnd (* d d) rstd) (mkv d) (ones d) (mkv d)
                                        (rnd (* d 4 d) 0.02d0) (mkv (* 4 d)) (rnd (* 4 d d) rstd) (mkv d)))
                  :lnfg (ones d) :lnfb (mkv d)))))
(defun nparams (m) (reduce #'+ (mapcar (lambda (p) (length (the fvec (car p)))) (model-vecs m))))

;;; ---------- worker: activations for b sequences of length <= tt ----------
(defstruct (wk (:conc-name w-)) (b 0 :type fixnum) grads
  xs                ; residual stream inputs per layer: vector of nl+1 (R x d)
  ln1h ln1r xn1 qkv att y xmid ln2h ln2r xn2 hfc gel    ; per layer (vectors of fvecs)
  lnfh lnfr xf P dx tmp dqkv dy dg dA ids tgt (loss 0d0 :type double-float) (count 0 :type fixnum)
  (drop nil) (newmask t) rng me m1 m2 ma attd dm)

(defun make-wk* (c b)
  (let* ((d (c-d c)) (tt (c-tt c)) (nl (c-nl c)) (r (* b tt)) (v (c-v c)) (nh (c-nh c)))
    (flet ((per (n) (coerce (loop repeat nl collect (mkv n)) 'simple-vector)))
      (make-wk :b b :grads (make-model* c t)
               :xs (coerce (loop repeat (1+ nl) collect (mkv (* r d))) 'simple-vector)
               :ln1h (per (* r d)) :ln1r (per r) :xn1 (per (* r d)) :qkv (per (* r 3 d)) :att (per (* b nh tt tt))
               :y (per (* r d)) :xmid (per (* r d)) :ln2h (per (* r d)) :ln2r (per r) :xn2 (per (* r d))
               :hfc (per (* r 4 d)) :gel (per (* r 4 d))
               :lnfh (mkv (* r d)) :lnfr (mkv r) :xf (mkv (* r d)) :P (mkv (* r v))
               :dx (mkv (* r d)) :tmp (mkv (* r d)) :dqkv (mkv (* r 3 d)) :dy (mkv (* r d)) :dg (mkv (* r 4 d)) :dA (mkv (* tt tt))
               :ids (make-array r :element-type 'fixnum) :tgt (make-array r :element-type 'fixnum)
               :rng (make-array 1 :element-type '(unsigned-byte 64) :initial-element (+ 1 (random (expt 2 62) *rs*)))
               :me (mkv (* r d)) :m1 (per (* r d)) :m2 (per (* r d)) :ma (per (* b nh tt tt)) :attd (per (* b nh tt tt)) :dm (mkv (* r d))))))

(defun fill-mask (st mask n)
  "Inverted-dropout mask: 0 with prob *pdrop*, else 1/(1-p). xorshift64 RNG per worker."
  (declare (type (simple-array (unsigned-byte 64) (1)) st) (fvec mask) (idx n))
  (let ((x (aref st 0)) (thr (round (* *pdrop* 16777216))) (ks (coerce (/ 1d0 (- 1d0 *pdrop*)) 'fl)))
    (declare (type (unsigned-byte 64) x) (fixnum thr) (type fl ks))
    (dotimes (i n)
      (setf x (logxor x (logand (ash x 13) #xFFFFFFFFFFFFFFFF)))
      (setf x (logxor x (ash x -7)))
      (setf x (logxor x (logand (ash x 17) #xFFFFFFFFFFFFFFFF)))
      (setf (aref mask i) (if (< (ash x -40) thr) (f 0) ks)))
    (setf (aref st 0) x)))
(defun vmul (a m n) (declare (fvec a m) (idx n)) (dotimes (q n) (setf (aref a q) (* (aref a q) (aref m q)))))
(defun vmul-into (o a m n) (declare (fvec o a m) (idx n)) (dotimes (q n) (setf (aref o q) (* (aref a q) (aref m q)))))

;;; ---------- building blocks (R rows, row-major) ----------
(defun layernorm (X R d g b H rs Y)
  "Y = g * (X-mean)/sqrt(var+1e-5) + b per row; stores xhat in H and 1/std in RS."
  (declare (fvec X g b H rs Y) (dim R d))
  (dotimes (r R)
    (let ((o (* r d)) (mu 0d0) (var 0d0))
      (declare (idx o) (double-float mu var))
      (dotimes (j d) (incf mu (aref X (+ o j)))) (setf mu (/ mu d))
      (dotimes (j d) (let ((z (- (aref X (+ o j)) mu))) (incf var (* z z)))) (setf var (/ var d))
      (let ((s (/ 1d0 (sqrt (+ var 1d-5)))))
        (setf (aref rs r) (coerce s 'fl))
        (dotimes (j d)
          (let ((xh (coerce (* (- (aref X (+ o j)) mu) s) 'fl)))
            (setf (aref H (+ o j)) xh (aref Y (+ o j)) (+ (* (aref g j) xh) (aref b j)))))))))

(defun ln-back (dY H rs g dg db R d dX accumulate)
  "Backprop through LAYERNORM; dX (+)= dL/dX."
  (declare (fvec dY H rs g dg db dX) (dim R d))
  (dotimes (r R)
    (let ((o (* r d)) (m1 0d0) (m2 0d0))
      (declare (idx o) (double-float m1 m2))
      (dotimes (j d)
        (let* ((dy (aref dY (+ o j))) (dxh (* dy (aref g j))))
          (incf (aref dg j) (* dy (aref H (+ o j)))) (incf (aref db j) dy)
          (incf m1 dxh) (incf m2 (* dxh (aref H (+ o j))))))
      (setf m1 (/ m1 d) m2 (/ m2 d))
      (let ((s (aref rs r)))
        (dotimes (j d)
          (let ((v (* s (coerce (- (* (aref dY (+ o j)) (aref g j)) m1 (* (aref H (+ o j)) m2)) 'fl))))
            (if accumulate (incf (aref dX (+ o j)) v) (setf (aref dX (+ o j)) v))))))))

(defun linear (X R din W bias Y dout)
  "Y[R,dout] = X W + bias."
  (declare (fvec X W bias Y) (dim R din dout))
  (dotimes (r R) (replace Y bias :start1 (* r dout)))
  (gemm nil nil R dout din (f 1) X 0 din W 0 dout (f 1) Y 0 dout))

(defun linear-back (X R din W dY dout gW gb dX)
  (declare (fvec X W dY gW gb) (dim R din dout))
  (gemm t nil din dout R (f 1) X 0 din dY 0 dout (f 1) gW 0 dout)
  (dotimes (r R) (let ((o (* r dout))) (declare (idx o)) (dotimes (j dout) (incf (aref gb j) (aref dY (+ o j))))))
  (when dX (gemm nil t R din dout (f 1) dY 0 dout W 0 dout (f 0) dX 0 din)))

(defconstant +gc+ 0.7978845608028654d0)   ; sqrt(2/pi)
(defun gelu (Hh G n)
  (declare (fvec Hh G) (idx n))
  (dotimes (q n)
    (let ((x (coerce (aref Hh q) 'double-float)))
      (setf (aref G q) (coerce (* 0.5d0 x (+ 1d0 (tanh-d (* +gc+ (+ x (* 0.044715d0 x x x)))))) 'fl)))))
(defun gelu-back (Hh dG n)       ; dG *= gelu'(h)
  (declare (fvec Hh dG) (idx n))
  (dotimes (q n)
    (let* ((x (coerce (aref Hh q) 'double-float)) (t1 (tanh-d (* +gc+ (+ x (* 0.044715d0 x x x)))))
           (dd (+ (* 0.5d0 (+ 1d0 t1)) (* 0.5d0 x (- 1d0 (* t1 t1)) +gc+ (+ 1d0 (* 3d0 0.044715d0 x x))))))
      (setf (aref dG q) (* (aref dG q) (coerce dd 'fl))))))

(defun attention (c qkv att y b nt &optional ma attd)
  "Causal multi-head self-attention for b sequences of length nt. If MA: dropout on probs (ATTD = ATT*MA)."
  (declare (fvec qkv att y) (dim b nt))
  (let* ((d (c-d c)) (nh (c-nh c)) (hd (floor d nh)) (d3 (* 3 d)) (scale (coerce (/ 1d0 (sqrt (float hd 1d0))) 'fl)))
    (declare (dim d nh hd d3))
    (dotimes (s b)
      (dotimes (h nh)
        (let* ((qo (+ (* s nt d3) (* h hd))) (ko (+ qo d)) (vo (+ qo d d))
               (ao (* (+ (* s nh) h) nt nt)) (yo (+ (* s nt d) (* h hd))))
          (declare (idx qo ko vo ao yo))
          (gemm nil t nt nt hd scale qkv qo d3 qkv ko d3 (f 0) att ao nt)      ; S = Q K^T / sqrt(hd)
          (dotimes (i nt)                                                      ; causal softmax by row
            (let* ((o (+ ao (* i nt))) (mx (aref att o)) (sm 0d0))
              (declare (idx o) (type fl mx) (double-float sm))
              (loop for j from 0 to i do (setf mx (max mx (aref att (+ o j)))))
              (loop for j from 0 to i do (let ((e (exp-neg (coerce (- (aref att (+ o j)) mx) 'double-float))))
                                           (setf (aref att (+ o j)) (coerce e 'fl)) (incf sm e)))
              (let ((inv (/ 1d0 sm)))
                (loop for j from 0 to i do (setf (aref att (+ o j)) (coerce (* inv (aref att (+ o j))) 'fl)))
                (loop for j from (1+ i) below nt do (setf (aref att (+ o j)) (f 0))))))
          (if ma
              (let ((ma ma) (attd attd)) (declare (fvec ma attd))
                (loop for q from ao below (+ ao (* nt nt)) do (setf (aref attd q) (* (aref att q) (aref ma q))))
                (gemm nil nil nt hd nt (f 1) attd ao nt qkv vo d3 (f 0) y yo d))
              (gemm nil nil nt hd nt (f 1) att ao nt qkv vo d3 (f 0) y yo d)))))))   ; Y = A V

(defun attention-back (c qkv att dy dqkv dA b nt &optional ma attd)
  (declare (fvec qkv att dy dqkv dA) (dim b nt))
  (let* ((d (c-d c)) (nh (c-nh c)) (hd (floor d nh)) (d3 (* 3 d)) (scale (coerce (/ 1d0 (sqrt (float hd 1d0))) 'fl)))
    (declare (dim d nh hd d3))
    (dotimes (s b)
      (dotimes (h nh)
        (let* ((qo (+ (* s nt d3) (* h hd))) (ko (+ qo d)) (vo (+ qo d d))
               (ao (* (+ (* s nh) h) nt nt)) (yo (+ (* s nt d) (* h hd))))
          (declare (idx qo ko vo ao yo))
          (gemm nil t nt nt hd (f 1) dy yo d qkv vo d3 (f 0) dA 0 nt)          ; dA = dY V^T
          (when ma (let ((ma ma)) (declare (fvec ma)) (dotimes (q (* nt nt)) (setf (aref dA q) (* (aref dA q) (aref ma (+ ao q)))))))
          (gemm t nil nt hd nt (f 1) (if ma attd att) ao nt dy yo d (f 0) dqkv vo d3)        ; dV = A^T dY
          (dotimes (i nt)                                                      ; softmax backward
            (let ((o (* i nt)) (sd 0d0))
              (declare (idx o) (double-float sd))
              (loop for j from 0 to i do (incf sd (* (aref att (+ ao o j)) (aref dA (+ o j)))))
              (let ((sdf (coerce sd 'fl)))
                (loop for j from 0 to i do (setf (aref dA (+ o j)) (* (aref att (+ ao o j)) (- (aref dA (+ o j)) sdf))))
                (loop for j from (1+ i) below nt do (setf (aref dA (+ o j)) (f 0))))))
          (gemm nil nil nt hd nt scale dA 0 nt qkv ko d3 (f 0) dqkv qo d3)      ; dQ = dS K / sqrt(hd)
          (gemm t nil nt hd nt scale dA 0 nt qkv qo d3 (f 0) dqkv ko d3))))))  ; dK = dS^T Q / sqrt(hd)

;;; ---------- full model forward / backward ----------
(defun forward-backward (m w nt sf backprop scale)
  "W's ids/tgt hold b sequences of length NT (row s*nt+i). Loss counted on positions i >= SF.
   Adds sum(-log p) to w-loss; if BACKPROP adds gradients of SCALE*sum to w's grads."
  (declare (dim nt sf) (type fl scale))
  (let* ((c (m-cfg m)) (d (c-d c)) (v (c-v c)) (nl (c-nl c)) (b (w-b w)) (R (* b nt)) (d3 (* 3 d)) (d4 (* 4 d))
         (wte (m-wte m)) (wpe (m-wpe m)) (ids (w-ids w)) (tgt (w-tgt w)) (P (w-P w)) (xf (w-xf w)) (loss 0d0))
    (declare (dim d v nl b R d3 d4) (fvec wte wpe P xf) ((simple-array fixnum (*)) ids tgt) (double-float loss))
    ;; embeddings
    (let ((x0 (svref (w-xs w) 0)))
      (declare (fvec x0))
      (dotimes (r R)
        (let ((o (* r d)) (eo (* (aref ids r) d)) (po (* (mod r nt) d)))
          (declare (idx o eo po))
          (dotimes (j d) (setf (aref x0 (+ o j)) (+ (aref wte (+ eo j)) (aref wpe (+ po j)))))))
      (when (w-drop w)
        (when (w-newmask w)
          (fill-mask (w-rng w) (w-me w) (* R d))
          (dotimes (l nl) (fill-mask (w-rng w) (svref (w-m1 w) l) (* R d)) (fill-mask (w-rng w) (svref (w-m2 w) l) (* R d))
                          (fill-mask (w-rng w) (svref (w-ma w) l) (* b (c-nh c) nt nt))))
        (vmul x0 (w-me w) (* R d))))
    ;; blocks
    (loop for ly in (m-layers m) for l from 0
          do (let ((X (svref (w-xs w) l)) (Xo (svref (w-xs w) (1+ l))) (xmid (svref (w-xmid w) l)))
               (declare (fvec X Xo xmid))
               (layernorm X R d (svref ly +ln1g+) (svref ly +ln1b+) (svref (w-ln1h w) l) (svref (w-ln1r w) l) (svref (w-xn1 w) l))
               (linear (svref (w-xn1 w) l) R d (svref ly +wqkv+) (svref ly +bqkv+) (svref (w-qkv w) l) d3)
               (if (w-drop w)
                   (attention c (svref (w-qkv w) l) (svref (w-att w) l) (svref (w-y w) l) b nt (svref (w-ma w) l) (svref (w-attd w) l))
                   (attention c (svref (w-qkv w) l) (svref (w-att w) l) (svref (w-y w) l) b nt))
               (linear (svref (w-y w) l) R d (svref ly +wproj+) (svref ly +bproj+) xmid d)
               (when (w-drop w) (vmul xmid (svref (w-m1 w) l) (* R d)))
               (dotimes (q (* R d)) (incf (aref xmid q) (aref X q)))                         ; residual
               (layernorm xmid R d (svref ly +ln2g+) (svref ly +ln2b+) (svref (w-ln2h w) l) (svref (w-ln2r w) l) (svref (w-xn2 w) l))
               (linear (svref (w-xn2 w) l) R d (svref ly +wfc+) (svref ly +bfc+) (svref (w-hfc w) l) d4)
               (gelu (svref (w-hfc w) l) (svref (w-gel w) l) (* R d4))
               (linear (svref (w-gel w) l) R d4 (svref ly +wfc2+) (svref ly +bfc2+) Xo d)
               (when (w-drop w) (vmul Xo (svref (w-m2 w) l) (* R d)))
               (dotimes (q (* R d)) (incf (aref Xo q) (aref xmid q)))))                      ; residual
    ;; final LN + tied head + softmax
    (layernorm (svref (w-xs w) nl) R d (m-lnfg m) (m-lnfb m) (w-lnfh w) (w-lnfr w) xf)
    (gemm nil t R v d (f 1) xf 0 d wte 0 d (f 0) P 0 v)
    (dotimes (r R)
      (let* ((o (* r v)) (mx (aref P o)) (sm 0d0) (scored (>= (mod r nt) sf)))
        (declare (idx o) (type fl mx) (double-float sm))
        (loop for j from o below (+ o v) do (setf mx (max mx (aref P j))))
        (loop for j from o below (+ o v) do (let ((e (exp-neg (coerce (- (aref P j) mx) 'double-float)))) (setf (aref P j) (coerce e 'fl)) (incf sm e)))
        (let ((inv (/ 1d0 sm)))
          (when scored (decf loss (log (max 1d-300 (* inv (aref P (+ o (aref tgt r))))))) (incf (w-count w)))
          (loop for j from o below (+ o v) do (setf (aref P j) (if scored (coerce (* inv (aref P j)) 'fl) (f 0)))))))
    (incf (w-loss w) loss)
    (when backprop
      (let* ((g (w-grads w)) (gwte (m-wte g)) (dx (w-dx w)) (tmp (w-tmp w)) (dy (w-dy w)) (dg (w-dg w)) (dqkv (w-dqkv w)))
        (declare (fvec gwte dx tmp dy dg dqkv))
        (dotimes (r R) (let ((o (* r v)))                                                    ; dlogits
                         (when (>= (mod r nt) sf) (decf (aref P (+ o (aref tgt r))) (f 1)))
                         (dotimes (j v) (setf (aref P (+ o j)) (* scale (aref P (+ o j)))))))
        (gemm t nil v d R (f 1) P 0 v xf 0 d (f 1) gwte 0 d)                               ; head grad (tied)
        (gemm nil nil R d v (f 1) P 0 v wte 0 d (f 0) tmp 0 d)
        (ln-back tmp (w-lnfh w) (w-lnfr w) (m-lnfg m) (m-lnfg g) (m-lnfb g) R d dx nil)
        (loop for l from (1- nl) downto 0
              do (let ((ly (nth l (m-layers m))) (gl (nth l (m-layers g))))
                   ;; MLP branch
                   (linear-back (svref (w-gel w) l) R d4 (svref ly +wfc2+) (if (w-drop w) (progn (vmul-into (w-dm w) dx (svref (w-m2 w) l) (* R d)) (w-dm w)) dx) d (svref gl +wfc2+) (svref gl +bfc2+) dg)
                   (gelu-back (svref (w-hfc w) l) dg (* R d4))
                   (linear-back (svref (w-xn2 w) l) R d (svref ly +wfc+) dg d4 (svref gl +wfc+) (svref gl +bfc+) tmp)
                   (ln-back tmp (svref (w-ln2h w) l) (svref (w-ln2r w) l) (svref ly +ln2g+) (svref gl +ln2g+) (svref gl +ln2b+) R d dx t)
                   ;; attention branch
                   (linear-back (svref (w-y w) l) R d (svref ly +wproj+) (if (w-drop w) (progn (vmul-into (w-dm w) dx (svref (w-m1 w) l) (* R d)) (w-dm w)) dx) d (svref gl +wproj+) (svref gl +bproj+) dy)
                   (if (w-drop w)
                       (attention-back c (svref (w-qkv w) l) (svref (w-att w) l) dy dqkv (w-dA w) b nt (svref (w-ma w) l) (svref (w-attd w) l))
                       (attention-back c (svref (w-qkv w) l) (svref (w-att w) l) dy dqkv (w-dA w) b nt))
                   (linear-back (svref (w-xn1 w) l) R d (svref ly +wqkv+) dqkv d3 (svref gl +wqkv+) (svref gl +bqkv+) tmp)
                   (ln-back tmp (svref (w-ln1h w) l) (svref (w-ln1r w) l) (svref ly +ln1g+) (svref gl +ln1g+) (svref gl +ln1b+) R d dx t)))
        (let ((gwpe (m-wpe g)))                                                              ; embeddings
          (declare (fvec gwpe))
          (when (w-drop w) (vmul dx (w-me w) (* R d)))
          (dotimes (r R)
            (let ((o (* r d)) (eo (* (aref ids r) d)) (po (* (mod r nt) d)))
              (declare (idx o eo po))
              (dotimes (j d) (incf (aref gwte (+ eo j)) (aref dx (+ o j))) (incf (aref gwpe (+ po j)) (aref dx (+ o j)))))))))))

(defun parallel (workers fn)
  (if (null (cdr workers)) (funcall fn (car workers))
      (let ((ths (mapcar (lambda (w) (sb-thread:make-thread (lambda () (funcall fn w)))) (cdr workers))))
        (funcall fn (car workers)) (mapc #'sb-thread:join-thread ths))))

(defun sum-grads (workers)
  "Sum all workers' grads into the first worker's (others zeroed). Returns list of fvecs (model-vecs order)."
  (let ((g0 (mapcar #'car (model-vecs (w-grads (car workers))))))
    (dolist (w (cdr workers))
      (loop for a in g0 for (bv) in (model-vecs (w-grads w))
            do (let ((a a) (bv bv)) (declare (fvec a bv)) (dotimes (i (length a)) (incf (aref a i) (aref bv i)) (setf (aref bv i) (f 0))))))
    g0))

(defstruct (adam (:conc-name a-)) ms vs (tn 0 :type fixnum))
(defun make-adam* (m) (make-adam :ms (mapcar (lambda (p) (mkv (length (the fvec (car p))))) (model-vecs m))
                                 :vs (mapcar (lambda (p) (mkv (length (the fvec (car p))))) (model-vecs m))))
(defun adamw (m opt grads lr wd)
  "AdamW (betas 0.9/0.99, eps 1e-8), global grad-norm clip 1.0, decoupled weight decay WD on matrices."
  (declare (double-float lr wd))
  (let* ((norm (sqrt (loop for gr in grads sum (let ((gr gr)) (declare (fvec gr)) (loop for x across gr sum (* (coerce x 'double-float) x) of-type double-float)) of-type double-float)))
         (cs (coerce (if (> norm 1d0) (/ 1d0 norm) 1d0) 'fl)) (tn (incf (a-tn opt)))
         (c1 (- 1d0 (expt 0.9d0 tn))) (c2 (- 1d0 (expt 0.99d0 tn)))
         (lrc (coerce (/ lr c1) 'fl)) (ic2 (coerce (/ 1d0 c2) 'fl)))
    (loop for (p . decay) in (model-vecs m) for g in grads for mm in (a-ms opt) for vv in (a-vs opt)
          do (let ((p p) (g g) (mm mm) (vv vv) (shrink (coerce (if decay (- 1d0 (* lr wd)) 1d0) 'fl)))
               (declare (fvec p g mm vv))
               (dotimes (i (length p))
                 (let ((gi (* cs (aref g i))))
                   (setf (aref mm i) (+ (* (f 0.9) (aref mm i)) (* (f 0.1) gi))
                         (aref vv i) (+ (* (f 0.99) (aref vv i)) (* (f 0.01) gi gi)))
                   (setf (aref p i) (- (* shrink (aref p i)) (/ (* lrc (aref mm i)) (+ (sqrt (* (aref vv i) ic2)) (f 1e-8)))))
                   (setf (aref g i) (f 0))))))
    norm))

;;; ---------- gradient check (use FT=double) ----------
(defun gradcheck (&optional (p 0d0))
  (setf *pdrop* p)
  (let* ((c (make-cfg :v 37 :d 8 :nl 2 :nh 2 :tt 4)) (m (make-model* c)) (k 2) (b 2) (nt 4)
         (ws (loop repeat k collect (make-wk* c b))) (scale (coerce (/ 1d0 (* k b nt)) 'fl))
         (eps (if (eq *ft* 'single-float) 1d-2 1d-5)) (worst 0d0) (wa 0d0) (cnt 0) (sd2 0d0) (sn2 0d0))
    ;; perturb all params (so LN gains/biases are not trivially 1/0)
    (dolist (pv (model-vecs m)) (let ((p (car pv))) (declare (fvec p)) (dotimes (i (length p)) (incf (aref p i) (coerce (* 0.3 (randn)) 'fl)))))
    (dolist (w ws) (dotimes (q (* b nt)) (setf (aref (w-ids w) q) (random 37 *rs*) (aref (w-tgt w) q) (random 37 *rs*))))
    (dolist (w ws) (setf (w-drop w) (> p 0d0) (w-newmask w) t))
    (flet ((loss (bp) (dolist (w ws) (setf (w-loss w) 0d0 (w-count w) 0) (forward-backward m w nt 0 bp scale) (setf (w-newmask w) nil))
                 (* (coerce scale 'double-float) (loop for w in ws sum (w-loss w)))))
      (loss t)
      (loop for (p) in (model-vecs m) for g in (sum-grads ws) for name from 0
            do (let ((p p) (g g)) (declare (fvec p g))
                 (dotimes (i (length p))
                   (let* ((old (aref p i)) (pp (coerce (+ old eps) 'fl)) (pm (coerce (- old eps) 'fl))
                          (lp (progn (setf (aref p i) pp) (loss nil))) (lm (progn (setf (aref p i) pm) (loss nil)))
                          (num (/ (- lp lm) (- (coerce pp 'double-float) (coerce pm 'double-float)))) (an (coerce (aref g i) 'double-float)))
                     (setf (aref p i) old) (incf cnt) (incf sd2 (expt (- num an) 2)) (incf sn2 (expt num 2))
                     (setf wa (max wa (abs (- num an))) worst (if (> (+ (abs num) (abs an)) 1d-7) (max worst (/ (abs (- num an)) (+ (abs num) (abs an)))) worst)))))))
    (format t "gradcheck FT=~a dropout=~a (masks fixed): tiny GPT (V=37 d=8 L=2 heads=2 T=4), 2 workers x 2 seqs, ~d params, eps=~a~%  max rel err (entries with |g|+|num|>1e-7) ~,3e, max abs err ~,3e, global |g-num|/|num| = ~,3e~%"
            *ft* p cnt eps worst wa (sqrt (/ sd2 sn2)))))

;;; ---------- training ----------
(defun main ()
  (setf *rs* (sb-ext:seed-random-state (parse-integer (opt "seed" "1"))))
  (%set-threads 1)
  (let* ((full (with-open-file (s (opt "data" "../input.txt") :external-format :utf-8)
                 (let* ((b (make-string (file-length s))) (k (read-sequence b s))) (subseq b 0 k))))
         (chars (coerce (sort (remove-duplicates (coerce full 'list)) #'char<) 'simple-vector))
         (v (length chars)) (c2i (let ((ht (make-hash-table))) (loop for ch across chars for i from 0 do (setf (gethash ch ht) i)) ht))
         (data (map '(simple-array fixnum (*)) (lambda (ch) (gethash ch c2i)) full))
         (nall (length data)) (vstart (floor (* 0.9 nall)))             ; same fixed split as the ablation
         (tt (parse-integer (opt "ctx" "64"))) (c (make-cfg :v v :d (parse-integer (opt "d" "128")) :nl (parse-integer (opt "layers" "4"))
                                                       :nh (parse-integer (opt "heads" "4")) :tt tt))
         (k (parse-integer (opt "threads" "8"))) (bt (parse-integer (opt "batch" "32"))) (b (floor bt k))
         (lr0 (float (read-from-string (opt "lr" "1e-3")) 1d0)) (wd (float (read-from-string (opt "wd" "0.1")) 1d0))
         (budget (float (parse-integer (opt "secs" "480")) 1d0)) (warm 0.02d0)
         (pdrop (setf *pdrop* (float (read-from-string (opt "dropout" "0")) 1d0)))
         (m (make-model* c)) (opt (make-adam* m)) (ws (loop repeat k collect (make-wk* c b)))
         (scale (coerce (/ 1d0 (* bt tt)) 'fl)) (half (floor tt 2))
         (step 0) (t0 (get-internal-real-time)) (hist nil) (fhist nil) (eval-secs 0d0)
         (fevery (float (parse-integer (opt "fullval-every" "0")) 1d0)) (fnext (float (parse-integer (opt "fullval-every" "0")) 1d0)))
    (declare (fixnum step) (double-float eval-secs))
    (assert (= (* b k) bt))
    (labels ((secs () (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0))
             (tsecs () (- (secs) eval-secs))     ; training time = wall time minus evaluation time
             (write-full () (with-open-file (fo (format nil "~a-full.csv" (opt "tag" "gpt")) :direction :output :if-exists :supersede)
                              (format fo "step,train_seconds,val_full~%") (dolist (r (reverse fhist)) (format fo "~{~a~^,~}~%" r))))
             (lr-at (fr) (if (< fr warm) (* lr0 (max 0.01d0 (/ fr warm)))
                             (+ (* 0.1d0 lr0) (* 0.9d0 lr0 0.5d0 (+ 1d0 (cos (* pi (min 1d0 (/ (- fr warm) (- 1d0 warm))))))))))
             (load-window (w s0 start)       ; sequence slot s0 of worker w <- window starting at START
               (dotimes (i tt) (setf (aref (w-ids w) (+ (* s0 tt) i)) (aref data (+ start i))
                                     (aref (w-tgt w) (+ (* s0 tt) i)) (aref data (+ start i 1)))))
             (val-loss (ntargets)
               ;; Windows of length tt with stride tt/2; score only the second half of each window
               ;; (so every scored char has between tt/2 and tt-1 chars of context). First scored target = vstart.
               (dolist (w ws) (setf (w-drop w) nil))
               (let* ((e0 (secs)) (starts (loop for s from (- vstart 1 half) by half
                                                while (and (< (+ s half 1) (+ vstart ntargets)) (<= (+ s tt 1) nall))
                                                collect s)))
                 (dolist (w ws) (setf (w-loss w) 0d0 (w-count w) 0))
                 (loop while starts
                       do (let ((batch (loop repeat bt while starts collect (pop starts))))
                            ;; pad incomplete batches by repeating the last window, then discount
                            (let* ((nreal (length batch)) (padded (append batch (make-list (- bt nreal) :initial-element (car (last batch))))))
                              (loop for w in ws for kk from 0
                                    do (dotimes (s0 b) (load-window w s0 (nth (+ (* kk b) s0) padded))))
                              (let ((before (mapcar (lambda (w) (cons (w-loss w) (w-count w))) ws)))
                                (parallel ws (lambda (w) (forward-backward m w tt half nil scale)))
                                ;; drop padded sequences' contribution: recompute is costly, so pads are excluded by
                                ;; re-running only the real ones when the batch is partial (last batch only)
                                (when (< nreal bt)
                                  (loop for w in ws for (l0 . c0) in before do (setf (w-loss w) l0 (w-count w) c0))
                                  (let ((w1 (make-wk* c 1)))
                                    (dolist (s batch)
                                      (load-window w1 0 s)
                                      (forward-backward m w1 tt half nil scale)
                                      (incf (w-loss (car ws)) (w-loss w1)) (incf (w-count (car ws)) (w-count w1))
                                      (setf (w-loss w1) 0d0 (w-count w1) 0))))))))
                 (incf eval-secs (- (secs) e0))
                 (/ (loop for w in ws sum (w-loss w)) (loop for w in ws sum (w-count w))))))
      (format t "dropout=~a (emb, attn probs, attn/MLP residual outputs; off at eval)~%" pdrop)
      (format t "config: GPT d=~d layers=~d heads=~d ctx=~d vocab=~d params=~d | batch ~d seqs (~d tokens), threads=~d (x ~d seqs), FT=~a~%"
              (c-d c) (c-nl c) (c-nh c) tt v (nparams m) bt (* bt tt) k b *ft*)
      (format t "AdamW lr=~a betas 0.9/0.99 wd=~a clip 1.0, warmup 2% then cosine to lr/10 over ~ds of training time (eval excluded) | data ~a | train chars ~d, val chars ~d (last 10%)~%"
              lr0 wd (round budget) (opt "data" "../input.txt") vstart (- nall vstart))
      (format t "val protocol: windows of ~d, stride ~d, score 2nd half (context ~d..~d); periodic val = first 32000 val targets every 200 steps~%"
              tt half half (1- tt))
      (finish-output)
      (loop for el = (tsecs) while (< el budget)
            do (dolist (w ws)
                 (dotimes (s0 b) (load-window w s0 (random (- vstart tt 1) *rs*)))
                 (setf (w-loss w) 0d0 (w-count w) 0 (w-drop w) (> pdrop 0d0) (w-newmask w) t))
               (parallel ws (lambda (w) (forward-backward m w tt 0 t scale)))
               (adamw m opt (sum-grads ws) (lr-at (/ el budget)) wd)
               (incf step)
               (when (zerop (mod step 200))
                 (let ((tl (/ (loop for w in ws sum (w-loss w)) (loop for w in ws sum (w-count w)))) (vl (val-loss 32000)))
                   (format t "step ~6d  t=~6,1fs  wall=~6,1fs  lr=~,6f  train(last batch)=~,4f  val(subset)=~,4f~%" step (tsecs) (secs) (lr-at (/ (tsecs) budget)) tl vl)
                   (push (list step (tsecs) tl vl) hist) (finish-output)
                   (with-open-file (fo (format nil "~a.csv" (opt "tag" "gpt")) :direction :output :if-exists :supersede)
                     (format fo "step,seconds,train_batch,val_subset~%") (dolist (r (reverse hist)) (format fo "~{~a~^,~}~%" r)))))
               (when (and (> fevery 0d0) (>= (tsecs) fnext) (< (tsecs) budget))
                 (incf fnext fevery)
                 (let* ((ts (tsecs)) (vf (val-loss nall)))
                   (format t "FULLVAL step ~d  t=~,1fs  val(full)=~,4f~%" step ts vf) (finish-output)
                   (push (list step ts vf) fhist) (write-full))))
      (let ((s (tsecs)))
        (format t "trained ~d steps in ~,1fs training time (wall ~,1fs, eval ~,1fs): ~,2f steps/s, ~,0f tokens/s (training time)~%" step s (secs) eval-secs (/ step s) (/ (* step bt tt) s))
        (finish-output)
        (let ((vl (val-loss nall))) (format t "FINAL: val(full fixed val, ~d targets scored)=~,4f~%" (loop for w in ws sum (w-count w)) vl)
          (push (list step s vl) fhist) (write-full)) (finish-output))
      ;; sample 300 chars at T=0.8 from a newline, context window up to tt
      (let ((w (make-wk* c 1)) (ctx (list (gethash #\Newline c2i))))
        (format t "~%=== sample (T=0.8) ===~%~a~%"
                (with-output-to-string (o)
                  (dotimes (q 300)
                    (let* ((cl (last ctx tt)) (nt (length cl)))
                      (loop for ch in cl for i from 0 do (setf (aref (w-ids w) i) ch (aref (w-tgt w) i) 0))
                      (forward-backward m w nt 0 nil (f 1))
                      (let* ((P (w-P w)) (o0 (* (1- nt) v))
                             (wt (loop for j below v collect (expt (max (coerce (aref P (+ o0 j)) 'double-float) 1d-300) 1.25d0)))
                             (u (* (random 1d0 *rs*) (reduce #'+ wt))) (acc 0d0) (ch (1- v)))
                        (loop for x in wt for j from 0 do (incf acc x) (when (> acc u) (setf ch j) (return)))
                        (write-char (svref chars ch) o) (setf ctx (append ctx (list ch))))))))))))

(let ((mode (opt "mode" "train")))
  (when (member "--run" sb-ext:*posix-argv* :test #'string=)
    (if (string= mode "gradcheck") (progn (gradcheck 0d0) (gradcheck 0.2d0) (gradcheck 0.5d0)) (main))))
