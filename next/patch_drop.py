p='gpt-drop.lisp'; s=open(p).read()
def rep(a,b,cnt=1):
    global s
    assert s.count(a)==cnt, (a[:60], s.count(a))
    s=s.replace(a,b)
rep(';;;; gpt.lisp -- small',';;;; gpt-drop.lisp -- gpt.lisp + inverted dropout (embeddings, attention probs, attn/MLP residual branch outputs).\n;;;; gpt.lisp -- small')
rep("(defvar *rs* (sb-ext:seed-random-state 1))","(defvar *rs* (sb-ext:seed-random-state 1))\n(defvar *pdrop* 0d0)")
rep("lnfh lnfr xf P dx tmp dqkv dy dg dA ids tgt (loss 0d0 :type double-float) (count 0 :type fixnum))",
    "lnfh lnfr xf P dx tmp dqkv dy dg dA ids tgt (loss 0d0 :type double-float) (count 0 :type fixnum)\n  (drop nil) (newmask t) rng me m1 m2 ma attd dm)")
rep(":ids (make-array r :element-type 'fixnum) :tgt (make-array r :element-type 'fixnum)))))",
    """:ids (make-array r :element-type 'fixnum) :tgt (make-array r :element-type 'fixnum)
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
(defun vmul-into (o a m n) (declare (fvec o a m) (idx n)) (dotimes (q n) (setf (aref o q) (* (aref a q) (aref m q)))))""")
rep('''(defun attention (c qkv att y b nt)
  "Causal multi-head self-attention for b sequences of length nt."''','''(defun attention (c qkv att y b nt &optional ma attd)
  "Causal multi-head self-attention for b sequences of length nt. If MA: dropout on probs (ATTD = ATT*MA)."''')
rep('''                (loop for j from (1+ i) below nt do (setf (aref att (+ o j)) (f 0))))))
          (gemm nil nil nt hd nt (f 1) att ao nt qkv vo d3 (f 0) y yo d))))))   ; Y = A V''',
'''                (loop for j from (1+ i) below nt do (setf (aref att (+ o j)) (f 0))))))
          (if ma
              (let ((ma ma) (attd attd)) (declare (fvec ma attd))
                (loop for q from ao below (+ ao (* nt nt)) do (setf (aref attd q) (* (aref att q) (aref ma q))))
                (gemm nil nil nt hd nt (f 1) attd ao nt qkv vo d3 (f 0) y yo d))
              (gemm nil nil nt hd nt (f 1) att ao nt qkv vo d3 (f 0) y yo d)))))))   ; Y = A V''')
rep('''(defun attention-back (c qkv att dy dqkv dA b nt)''','''(defun attention-back (c qkv att dy dqkv dA b nt &optional ma attd)''')
rep('''          (gemm t nil nt hd nt (f 1) att ao nt dy yo d (f 0) dqkv vo d3)        ; dV = A^T dY''',
'''          (when ma (let ((ma ma)) (declare (fvec ma)) (dotimes (q (* nt nt)) (setf (aref dA q) (* (aref dA q) (aref ma (+ ao q)))))))
          (gemm t nil nt hd nt (f 1) (if ma attd att) ao nt dy yo d (f 0) dqkv vo d3)        ; dV = A^T dY''')
rep('''          (dotimes (j d) (setf (aref x0 (+ o j)) (+ (aref wte (+ eo j)) (aref wpe (+ po j))))))))''',
'''          (dotimes (j d) (setf (aref x0 (+ o j)) (+ (aref wte (+ eo j)) (aref wpe (+ po j)))))))
      (when (w-drop w)
        (when (w-newmask w)
          (fill-mask (w-rng w) (w-me w) (* R d))
          (dotimes (l nl) (fill-mask (w-rng w) (svref (w-m1 w) l) (* R d)) (fill-mask (w-rng w) (svref (w-m2 w) l) (* R d))
                          (fill-mask (w-rng w) (svref (w-ma w) l) (* b (c-nh c) nt nt))))
        (vmul x0 (w-me w) (* R d))))''')
rep('''(attention c (svref (w-qkv w) l) (svref (w-att w) l) (svref (w-y w) l) b nt)''',
'''(if (w-drop w)
                   (attention c (svref (w-qkv w) l) (svref (w-att w) l) (svref (w-y w) l) b nt (svref (w-ma w) l) (svref (w-attd w) l))
                   (attention c (svref (w-qkv w) l) (svref (w-att w) l) (svref (w-y w) l) b nt))''')
rep('''               (dotimes (q (* R d)) (incf (aref xmid q) (aref X q)))                         ; residual''',
'''               (when (w-drop w) (vmul xmid (svref (w-m1 w) l) (* R d)))
               (dotimes (q (* R d)) (incf (aref xmid q) (aref X q)))                         ; residual''')
rep('''               (dotimes (q (* R d)) (incf (aref Xo q) (aref xmid q)))))                      ; residual''',
'''               (when (w-drop w) (vmul Xo (svref (w-m2 w) l) (* R d)))
               (dotimes (q (* R d)) (incf (aref Xo q) (aref xmid q)))))                      ; residual''')
rep('''(linear-back (svref (w-gel w) l) R d4 (svref ly +wfc2+) dx d''','''(linear-back (svref (w-gel w) l) R d4 (svref ly +wfc2+) (if (w-drop w) (progn (vmul-into (w-dm w) dx (svref (w-m2 w) l) (* R d)) (w-dm w)) dx) d''')
rep('''(linear-back (svref (w-y w) l) R d (svref ly +wproj+) dx d''','''(linear-back (svref (w-y w) l) R d (svref ly +wproj+) (if (w-drop w) (progn (vmul-into (w-dm w) dx (svref (w-m1 w) l) (* R d)) (w-dm w)) dx) d''')
rep('''(attention-back c (svref (w-qkv w) l) (svref (w-att w) l) dy dqkv (w-dA w) b nt)''',
'''(if (w-drop w)
                       (attention-back c (svref (w-qkv w) l) (svref (w-att w) l) dy dqkv (w-dA w) b nt (svref (w-ma w) l) (svref (w-attd w) l))
                       (attention-back c (svref (w-qkv w) l) (svref (w-att w) l) dy dqkv (w-dA w) b nt))''')
rep('''          (declare (fvec gwpe))
          (dotimes (r R)''','''          (declare (fvec gwpe))
          (when (w-drop w) (vmul dx (w-me w) (* R d)))
          (dotimes (r R)''')
rep("(defun gradcheck ()","(defun gradcheck (&optional (p 0d0))\n  (setf *pdrop* p)")
rep("(flet ((loss (bp) (dolist (w ws) (setf (w-loss w) 0d0 (w-count w) 0) (forward-backward m w nt 0 bp scale))",
    "(dolist (w ws) (setf (w-drop w) (> p 0d0) (w-newmask w) t))\n    (flet ((loss (bp) (dolist (w ws) (setf (w-loss w) 0d0 (w-count w) 0) (forward-backward m w nt 0 bp scale) (setf (w-newmask w) nil))")
rep('''(format t "gradcheck FT=~a: tiny GPT''','''(format t "gradcheck FT=~a dropout=~a (masks fixed): tiny GPT''')
rep('''            *ft* cnt eps worst''','''            *ft* p cnt eps worst''')
rep("(budget (float (parse-integer (opt \"secs\" \"480\")) 1d0)) (warm 0.02d0)",
    "(budget (float (parse-integer (opt \"secs\" \"480\")) 1d0)) (warm 0.02d0)\n         (pdrop (setf *pdrop* (float (read-from-string (opt \"dropout\" \"0\")) 1d0)))")
rep('''(let* ((e0 (secs)) (starts''','''(dolist (w ws) (setf (w-drop w) nil))
               (let* ((e0 (secs)) (starts''')
rep('''                 (setf (w-loss w) 0d0 (w-count w) 0))''','''                 (setf (w-loss w) 0d0 (w-count w) 0 (w-drop w) (> pdrop 0d0) (w-newmask w) t))''')
rep('''(format t "config: GPT d=~d''','''(format t "dropout=~a (emb, attn probs, attn/MLP residual outputs; off at eval)~%" pdrop)
      (format t "config: GPT d=~d''')
rep('''(if (string= mode "gradcheck") (gradcheck) (main))''','''(if (string= mode "gradcheck") (progn (gradcheck 0d0) (gradcheck 0.2d0) (gradcheck 0.5d0)) (main))''')
open(p,'w').write(s)
