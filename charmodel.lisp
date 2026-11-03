;;;; charmodel.lisp -- tiny character-level language models in plain Common Lisp.
;;;; 1) count-based bigram baseline  2) Bengio-style MLP with hand-written backprop.
;;;; Usage: sbcl --script charmodel.lisp [max-seconds] [ctx] [hidden]   (defaults 420 8 200)
(declaim (optimize (speed 3) (safety 0) (debug 0)))
(deftype dvec () '(simple-array double-float (*)))
(setf *read-default-float-format* (quote double-float)) ; print doubles without d0 suffix
(defparameter *rs* (sb-ext:seed-random-state 42))

;;; ---------- Data ----------
(defun read-text (path limit)
  (with-open-file (s path :external-format :utf-8)
    (let* ((buf (make-string limit)) (n (read-sequence buf s)))
      (subseq buf 0 n))))

(defparameter *text* (read-text "input.txt" 300000))         ; first ~300 KB
(defparameter *chars* (sort (remove-duplicates (coerce *text* 'list)) #'char<))
(defparameter *V* (length *chars*))
(defparameter *c2i* (let ((h (make-hash-table))) (loop for c in *chars* for i from 0 do (setf (gethash c h) i)) h))
(defparameter *data* (map '(simple-array fixnum (*)) (lambda (c) (gethash c *c2i*)) *text*))
(defparameter *split* (floor (* 0.9 (length *data*))))        ; 90% train / 10% val

(defmacro logf (fmt &rest args)
  `(progn (format t ,fmt ,@args) (finish-output)))

;;; ---------- Sampling helper ----------
(defun sample-from (probs)
  (declare (type dvec probs))
  (let ((r (random 1d0 *rs*)) (acc 0d0))
    (declare (double-float r acc))
    (dotimes (i (length probs) (1- (length probs)))
      (incf acc (aref probs i))
      (when (> acc r) (return i)))))

;;; ---------- Bigram baseline ----------
(defun bigram-model ()
  "Counts with add-1 smoothing, from training split only; returns VxV probs."
  (let* ((v *V*) (p (make-array (* v v) :element-type 'double-float :initial-element 1d0)))
    (loop for i from 1 below *split*
          do (incf (aref p (+ (* v (aref *data* (1- i))) (aref *data* i)))))
    (dotimes (a v p)
      (let ((s (loop for b below v sum (aref p (+ (* a v) b)) of-type double-float)))
        (dotimes (b v) (setf (aref p (+ (* a v) b)) (/ (aref p (+ (* a v) b)) s)))))))

(defun bigram-loss (p from to)
  (/ (loop for i from (max 1 from) below to
           sum (- (log (aref p (+ (* *V* (aref *data* (1- i))) (aref *data* i))))) of-type double-float)
     (- to (max 1 from))))

(defun bigram-sample (p n temp)
  (let ((row (make-array *V* :element-type 'double-float)) (c (gethash #\Newline *c2i*)))
    (with-output-to-string (o)
      (dotimes (k n)
        (dotimes (b *V*) (setf (aref row b) (expt (aref p (+ (* *V* c) b)) (/ 1d0 temp))))
        (let ((s (reduce #'+ row))) (dotimes (b *V*) (setf (aref row b) (/ (aref row b) s))))
        (setf c (sample-from row))
        (write-char (nth c *chars*) o)))))

;;; ---------- MLP: emb(ctx chars) -> concat -> tanh hidden -> softmax ----------
(defun argn (i default)   ; i-th command-line integer argument
  (let ((a (nth i sb-ext:*posix-argv*))) (if a (parse-integer a) default)))
;; Read from argv before later forms are compiled, so they stay compile-time constants.
(defconstant +ctx+ (argn 2 8))     ; previous characters used as context
(defconstant +emb+ 16)             ; embedding dimension
(defconstant +hid+ (argn 3 200))   ; hidden units
(defparameter *tag* (format nil "ctx~d-h~d" +ctx+ +hid+))
(defconstant +in+ (* +ctx+ +emb+))

(defun randn () ; Box-Muller
  (* (sqrt (* -2d0 (log (- 1d0 (random 1d0 *rs*))))) (cos (* 2 pi (random 1d0 *rs*)))))
(defun init-vec (n scale)
  (let ((a (make-array n :element-type 'double-float)))
    (dotimes (i n a) (setf (aref a i) (* scale (randn))))))
(defun zeros (n) (make-array n :element-type 'double-float :initial-element 0d0))

;; Parameters (row-major flat arrays)
(defparameter *C*  (init-vec (* *V* +emb+) 1d0))                  ; V x E
(defparameter *W1* (init-vec (* +in+ +hid+) (/ 1d0 (sqrt +in+)))) ; IN x H
(defparameter *b1* (zeros +hid+))
(defparameter *W2* (init-vec (* +hid+ *V*) (/ 0.1d0 (sqrt +hid+)))) ; H x V
(defparameter *b2* (zeros *V*))
(defparameter *nparams* (+ (* *V* +emb+) (* +in+ +hid+) +hid+ (* +hid+ *V*) *V*))
;; Gradients
(defparameter *gC* (zeros (* *V* +emb+))) (defparameter *gW1* (zeros (* +in+ +hid+)))
(defparameter *gb1* (zeros +hid+)) (defparameter *gW2* (zeros (* +hid+ *V*))) (defparameter *gb2* (zeros *V*))
;; Scratch buffers
(defparameter *x* (zeros +in+)) (defparameter *h* (zeros +hid+))
(defparameter *pr* (zeros *V*)) (defparameter *dh* (zeros +hid+)) (defparameter *dx* (zeros +in+))

(defun ctx-id (pos k)
  "Char id at position pos-ctx+k; positions before 0 use newline as padding."
  (declare (fixnum pos k))
  (let ((j (+ (- pos +ctx+) k)))
    (if (< j 0) (the fixnum (gethash #\Newline *c2i*)) (aref (the (simple-array fixnum (*)) *data*) j))))

(defun forward (ids temp)
  "Fill *x*, *h*, *pr* (softmax probs) for context ids (list/vector of length ctx)."
  (declare (simple-vector ids) (double-float temp))
  (let ((C *C*) (W1 *W1*) (b1 *b1*) (W2 *W2*) (b2 *b2*) (x *x*) (h *h*) (pr *pr*) (v *V*))
    (declare (type dvec C W1 b1 W2 b2 x h pr) (fixnum v))
    (dotimes (k +ctx+)                       ; embedding lookup + concat
      (let ((id (svref ids k)))
        (declare (fixnum id))
        (dotimes (e +emb+) (setf (aref x (+ (* k +emb+) e)) (aref C (+ (* id +emb+) e))))))
    (replace h b1)                           ; h = tanh(x W1 + b1)
    (dotimes (i +in+)
      (let ((xi (aref x i)) (row (* i +hid+)))
        (declare (double-float xi) (fixnum row))
        (dotimes (j +hid+) (incf (aref h j) (* xi (aref W1 (+ row j)))))))
    (dotimes (j +hid+) (setf (aref h j) (tanh (aref h j))))
    (replace pr b2)                          ; logits = h W2 + b2
    (dotimes (j +hid+)
      (let ((hj (aref h j)) (row (* j v)))
        (declare (double-float hj) (fixnum row))
        (dotimes (o v) (incf (aref pr o) (* hj (aref W2 (+ row o)))))))
    (let ((mx (reduce #'max pr)) (s 0d0))    ; stable softmax with temperature
      (declare (double-float mx s))
      (dotimes (o v) (let ((e (exp (/ (- (aref pr o) mx) temp)))) (setf (aref pr o) e) (incf s e)))
      (dotimes (o v) (setf (aref pr o) (/ (aref pr o) s))))
    pr))

(defun backward (ids target)
  "Accumulate gradients of -log p(target) into g* buffers (uses forward's scratch)."
  (declare (simple-vector ids) (fixnum target))
  (let ((W1 *W1*) (W2 *W2*) (x *x*) (h *h*) (pr *pr*) (dh *dh*) (dx *dx*)
        (gC *gC*) (gW1 *gW1*) (gb1 *gb1*) (gW2 *gW2*) (gb2 *gb2*) (v *V*))
    (declare (type dvec W1 W2 x h pr dh dx gC gW1 gb1 gW2 gb2) (fixnum v))
    (decf (aref pr target) 1d0)              ; dlogits = p - onehot
    (dotimes (o v) (incf (aref gb2 o) (aref pr o)))
    (dotimes (j +hid+)                       ; dW2, dh
      (let ((hj (aref h j)) (row (* j v)) (acc 0d0))
        (declare (double-float hj acc) (fixnum row))
        (dotimes (o v)
          (incf (aref gW2 (+ row o)) (* hj (aref pr o)))
          (incf acc (* (aref W2 (+ row o)) (aref pr o))))
        (setf (aref dh j) (* acc (- 1d0 (* hj hj))))))   ; through tanh
    (dotimes (j +hid+) (incf (aref gb1 j) (aref dh j)))
    (dotimes (i +in+)                        ; dW1, dx
      (let ((xi (aref x i)) (row (* i +hid+)) (acc 0d0))
        (declare (double-float xi acc) (fixnum row))
        (dotimes (j +hid+)
          (incf (aref gW1 (+ row j)) (* xi (aref dh j)))
          (incf acc (* (aref W1 (+ row j)) (aref dh j))))
        (setf (aref dx i) acc)))
    (dotimes (k +ctx+)                       ; scatter into embedding rows
      (let ((id (svref ids k)))
        (declare (fixnum id))
        (dotimes (e +emb+) (incf (aref gC (+ (* id +emb+) e)) (aref dx (+ (* k +emb+) e))))))))

(defun ids-at (pos ids)
  (dotimes (k +ctx+ ids) (setf (svref ids k) (ctx-id pos k))))

(defun eval-loss (from to &optional (stride 1))
  "Mean NLL (nats/char) over positions from..to-1 (every STRIDE-th)."
  (let ((ids (make-array +ctx+)) (s 0d0) (n 0))
    (declare (double-float s) (fixnum n))
    (loop for pos of-type fixnum from from below to by stride
          do (forward (ids-at pos ids) 1d0)
             (incf s (- (log (max 1d-300 (aref (the dvec *pr*) (aref *data* pos))))))
             (incf n))
    (/ s n)))

(defun sgd-step (lr bs)
  (declare (double-float lr) (fixnum bs))
  (let ((sc (/ lr bs)))
    (loop for (p g) in (list (list *C* *gC*) (list *W1* *gW1*) (list *b1* *gb1*)
                             (list *W2* *gW2*) (list *b2* *gb2*))
          do (let ((p p) (g g))
               (declare (type dvec p g))
               (dotimes (i (length p)) (decf (aref p i) (* sc (aref g i))) (setf (aref g i) 0d0))))))

(defun train (max-seconds)
  (let* ((bs 32) (ids (make-array +ctx+)) (start (get-internal-real-time)) (lr0 0.1d0)
         (run-loss 0d0) (run-n 0) (step 0) (hist '()))
    (declare (fixnum step run-n) (double-float run-loss))
    (flet ((elapsed () (/ (- (get-internal-real-time) start) internal-time-units-per-second 1d0)))
      (loop
        (let* ((frac (min 1d0 (/ (elapsed) max-seconds)))
               (lr (if (< frac 0.75d0) lr0 (* 0.1d0 lr0))))   ; step decay late in training
          (dotimes (b bs)
            (let ((pos (+ 1 (random (1- *split*) *rs*))))
              (forward (ids-at pos ids) 1d0)
              (incf run-loss (- (log (max 1d-300 (aref (the dvec *pr*) (aref *data* pos))))))
              (incf run-n)
              (backward ids (aref *data* pos))))
          (sgd-step lr bs)
          (incf step)
          (when (or (zerop (mod step 2000)) (>= frac 1d0))
            (let ((tr (eval-loss 0 *split* 9)) (va (eval-loss *split* (length *data*))))
              (logf "step ~6d  t=~6,1fs  lr=~,3f  batch-avg=~,4f  train=~,4f  val=~,4f~%"
                    step (elapsed) lr (/ run-loss run-n) tr va)
              (push (list step (elapsed) tr va) hist)
              (setf run-loss 0d0 run-n 0)))
          (when (>= frac 1d0) (return (values (nreverse hist) step (elapsed)))))))))

(defun mlp-sample (n temp)
  (let ((ids (make-array +ctx+ :initial-element (gethash #\Newline *c2i*))))
    (with-output-to-string (o)
      (dotimes (k n)
        (let ((c (sample-from (forward ids temp))))
          (write-char (nth c *chars*) o)
          (setf ids (concatenate 'simple-vector (subseq ids 1) (vector c))))))))

;;; ---------- Main ----------
(let ((max-sec (argn 1 420)))
  (logf "chars=~d vocab=~d train=~d val=~d~%" (length *data*) *V* *split* (- (length *data*) *split*))
  (logf "uniform loss ln(V) = ~,4f~%" (log (float *V* 1d0)))
  (let ((bp (bigram-model)))
    (logf "bigram (add-1): train=~,4f val=~,4f~%" (bigram-loss bp 0 *split*) (bigram-loss bp *split* (length *data*)))
    (logf "MLP: ctx=~d emb=~d hidden=~d params=~d, SGD batch 32, budget ~ds~%" +ctx+ +emb+ +hid+ *nparams* max-sec)
    (multiple-value-bind (hist steps secs) (train max-sec)
      (logf "trained ~d steps in ~,1fs; final train=~,4f val=~,4f (full train eval)~%"
            steps secs (eval-loss 0 *split*) (eval-loss *split* (length *data*)))
      (with-open-file (f (format nil "loss-~a.csv" *tag*) :direction :output :if-exists :supersede)
        (format f "step,seconds,train,val~%")
        (dolist (r hist) (format f "~{~a~^,~}~%" r))))
    (let ((ms (mlp-sample 300 0.8d0)) (bs (bigram-sample bp 300 0.8d0)))
      (with-open-file (f (format nil "samples-~a.txt" *tag*) :direction :output :if-exists :supersede)
        (format f "=== MLP sample (T=0.8) ===~%~a~%~%=== Bigram sample (T=0.8) ===~%~a~%" ms bs))
      (logf "~%=== MLP sample (T=0.8) ===~%~a~%~%=== Bigram sample (T=0.8) ===~%~a~%" ms bs))))
