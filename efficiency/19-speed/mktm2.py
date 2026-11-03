import sys,re
s=open(sys.argv[1]).read()
s=s.replace("(defun forward-backward (m w nt sf backprop scale)",
"""(defvar *tk* (make-array 16 :element-type 'double-float :initial-element 0d0))
(defun now () (multiple-value-bind (s ns) (sb-unix:clock-gettime 1) (+ (coerce s 'double-float) (* 1d-9 ns))))
(defmacro tmk (i &body body) `(if (eq sb-thread:*current-thread* (sb-thread:main-thread)) (let ((t0 (now))) (multiple-value-prog1 (progn ,@body) (incf (aref *tk* ,i) (- (now) t0)))) (progn ,@body)))
(defun forward-backward (m w nt sf backprop scale)""",1)
# trunk forward: wrap layers loop
a="    (dotimes (l nl)\n      (let ((ly (svref layers l)) (X (svref (w-xs w) l))"
assert a in s
s=s.replace(a,"    (tmk 0 (dotimes (l nl)\n      (let ((ly (svref layers l)) (X (svref (w-xs w) l))",1)
b="        (vadd Xo xmid (* R d))))\n    (layernorm (svref (w-xs w) nl)"
assert b in s
s=s.replace(b,"        (vadd Xo xmid (* R d)))))\n    (layernorm (svref (w-xs w) nl)",1)
# trunk-back
c="        (ln-back tmp (w-lnfh w) (w-lnfr w) (m-lnfg m) (m-lnfg g) (m-lnfb g) R d dx nil)"
assert c in s
s=s.replace("(when backprop (trunk-back m w nt)))))","(when backprop (tmk 1 (trunk-back m w nt))))))",1)
# ssm sections
s=s.replace("      (flet ((add (j)","      (tmk 2 (flet ((add (j)",1)
s=s.replace("          (replace wc wte :start1 (* a d) :start2 (* j d) :end2 (* (1+ j) d))))\n","          (replace wc wte :start1 (* a d) :start2 (* j d) :end2 (* (1+ j) d)))))\n",1)
s=s.replace("      (gemm nil t R nc d (f 1) xf 0 d wc 0 d (f 0) P 0 nc)\n      (dotimes (r R)","      (tmk 3 (gemm nil t R nc d (f 1) xf 0 d wc 0 d (f 0) P 0 nc))\n      (tmk 4 (dotimes (r R)",1)
s=s.replace("              (fill P (f 0) :start o :end (+ o nc)))))\n      (incf (w-loss w) loss)","              (fill P (f 0) :start o :end (+ o nc))))))\n      (incf (w-loss w) loss)",1)
s=s.replace("      (when backprop\n        (dotimes (r R) (let ((o (* r nc)))","      (when backprop\n        (tmk 5 (dotimes (r R) (let ((o (* r nc)))",1)
s=s.replace("(loop for j from o below (+ o nc) do (setf (aref P j) (* scale (aref P j))))))\n        (let ((gwte","(loop for j from o below (+ o nc) do (setf (aref P j) (* scale (aref P j)))))))\n        (let ((gwte",1)
s=s.replace("          (gemm t nil nc d R (f 1) P 0 nc xf 0 d (f 0) gwc 0 d)\n          (gemm nil nil R d nc (f 1) P 0 nc wc 0 d (f 0) tmp 0 d)\n          (dotimes (a nc) (let ((go (* (aref cand a) d)) (ao (* a d))) (dotimes (q d) (incf (aref gwte (+ go q)) (aref gwc (+ ao q))))))))",
"          (tmk 6 (gemm t nil nc d R (f 1) P 0 nc xf 0 d (f 0) gwc 0 d))\n          (tmk 7 (gemm nil nil R d nc (f 1) P 0 nc wc 0 d (f 0) tmp 0 d))\n          (tmk 8 (dotimes (a nc) (let ((go (* (aref cand a) d)) (ao (* a d))) (dotimes (q d) (incf (aref gwte (+ go q)) (aref gwc (+ ao q)))))))))",1)
e="               (incf step)\n"
s=s.replace(e,e+"               (when (zerop (mod step 100)) (format t \"TK step ~d (ms/step, main thread) trunkfwd ~,1f trunkback ~,1f cand ~,1f gemm-logits ~,1f softmax ~,1f scale ~,1f gemm-gwc ~,1f gemm-dx ~,1f scatter ~,1f~%\" step ~{(* 10 (aref *tk* ~d))~^ ~}) (fill *tk* 0d0))\n".replace("~{(* 10 (aref *tk* ~d))~^ ~}"," ".join("(* 10 (aref *tk* %d))"%i for i in range(9))),1)
open(sys.argv[2],'w').write(s)
