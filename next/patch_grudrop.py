p='gru-drop.lisp'; s=open(p).read()
def rep(a,b,cnt=1):
    global s
    assert s.count(a)==cnt, (a[:70], s.count(a))
    s=s.replace(a,b)
rep("(defun randn ()","(defvar *pdrop* 0d0)\n(defun randn ()")
rep("""  xs ys (loss 0d0 :type double-float) (count 0 :type fixnum))""","""  xs ys (loss 0d0 :type double-float) (count 0 :type fixnum)
  (drop nil) (newmask t) rng mx mh hd)   ; non-recurrent dropout: masks on input projection Ein[x] and on h before the output layer""")
rep("""             :xs (make-array (* tt b) :element-type 'fixnum) :ys (make-array (* tt b) :element-type 'fixnum))))""",
"""             :xs (make-array (* tt b) :element-type 'fixnum) :ys (make-array (* tt b) :element-type 'fixnum)
             :rng (make-array 1 :element-type '(unsigned-byte 64) :initial-element (+ 1 (random (expt 2 62) *rs*)))
             :mx (mkv (* tt b g)) :mh (mkv (* tt b h)) :hd (mkv (* tt b h)))))

(defun fill-mask (st mask n)
  "Inverted-dropout mask: 0 with prob *pdrop*, else 1/(1-p). xorshift64 RNG per worker."
  (declare (type (simple-array (unsigned-byte 64) (1)) st) (fvec mask) (fixnum n))
  (let ((x (aref st 0)) (thr (round (* *pdrop* 16777216))) (ks (coerce (/ 1d0 (- 1d0 *pdrop*)) 'fl)))
    (declare (type (unsigned-byte 64) x) (fixnum thr) (type fl ks))
    (dotimes (i n)
      (setf x (logxor x (logand (ash x 13) #xFFFFFFFFFFFFFFFF)))
      (setf x (logxor x (ash x -7)))
      (setf x (logxor x (logand (ash x 17) #xFFFFFFFFFFFFFFFF)))
      (setf (aref mask i) (if (< (ash x -40) thr) (f 0) ks)))
    (setf (aref st 0) x)))""")
# forward input
rep("""      ;; ---- forward, step by step through the recurrence ----
      (dotimes (s nt)""","""      (when (and (w-drop w) (w-newmask w))
        (fill-mask (w-rng w) (w-mx w) (* nt bg)) (fill-mask (w-rng w) (w-mh w) (* nt bh*)))
      ;; ---- forward, step by step through the recurrence ----
      (dotimes (s nt)""")
rep("""            (replace gates ein :start1 (+ go (* i g)) :start2 (* (aref xs (+ (* s b) i)) g) :end2 (* (1+ (aref xs (+ (* s b) i))) g))
""","""            (replace gates ein :start1 (+ go (* i g)) :start2 (* (aref xs (+ (* s b) i)) g) :end2 (* (1+ (aref xs (+ (* s b) i))) g))
            (when (w-drop w) (let ((mx (w-mx w)) (o (+ go (* i g)))) (declare (fvec mx) (idx o)) (dotimes (j g) (setf (aref gates (+ o j)) (* (aref gates (+ o j)) (aref mx (+ o j)))))))
""")
rep("""        (gemm-nn hall bh* why 0 P 0 rows h v t)""","""        (if (w-drop w)
            (let ((hd (w-hd w)) (mh (w-mh w))) (declare (fvec hd mh))
              (dotimes (q (* rows h)) (setf (aref hd q) (* (aref hall (+ bh* q)) (aref mh q))))
              (gemm-nn hd 0 why 0 P 0 rows h v t))
            (gemm-nn hall bh* why 0 P 0 rows h v t))""")
rep("""            (gemm-tn hall bh* P 0 gwhy 0 rows h v)                                     ; dWhy += H^T dlogits
            (gemm-nt P 0 why 0 dhall 0 rows v h)                                       ; dH = dlogits Why^T""",
"""            (if (w-drop w) (gemm-tn (w-hd w) 0 P 0 gwhy 0 rows h v) (gemm-tn hall bh* P 0 gwhy 0 rows h v))   ; dWhy += H^T dlogits
            (gemm-nt P 0 why 0 dhall 0 rows v h)                                       ; dH = dlogits Why^T
            (when (w-drop w) (let ((mh (w-mh w))) (declare (fvec mh)) (dotimes (q (* rows h)) (setf (aref dhall q) (* (aref dhall q) (aref mh q))))))""")
rep("""                             (dotimes (j g) (incf (aref gbx j) (aref dxg (+ oq j))) (incf (aref gein (+ eo j)) (aref dxg (+ oq j)))))))""",
"""                             (if (w-drop w)
                                 (let ((mx (w-mx w))) (declare (fvec mx))
                                   (dotimes (j g) (incf (aref gbx j) (aref dxg (+ oq j))) (incf (aref gein (+ eo j)) (* (aref dxg (+ oq j)) (aref mx (+ o j))))))
                                 (dotimes (j g) (incf (aref gbx j) (aref dxg (+ oq j))) (incf (aref gein (+ eo j)) (aref dxg (+ oq j))))))))""")
open(p,'w').write(s)
