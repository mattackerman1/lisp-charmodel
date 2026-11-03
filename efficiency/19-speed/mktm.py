import sys
s=open(sys.argv[1]).read()
s=s.replace("(defun opt-step (pool opt lr wd &optional (lr0 lr))",
"(defvar *tm* (make-array 8 :element-type 'double-float :initial-element 0d0))\n(defun now () (multiple-value-bind (s ns) (sb-unix:clock-gettime 1) (+ (coerce s 'double-float) (* 1d-9 ns))))\n(defun opt-step (pool opt lr wd &optional (lr0 lr))",1)
a="    (run-pool pool (lambda (th) (when chunk-wte (multiple-value-bind (lo hi) (wte-chunk th nth* (length (the fvec (svref (a-ps opt) 0))))"
assert a in s
s=s.replace(a,"    (setf (aref *tm* 3) (now))\n"+a,1)
b="    (when chunk-wte (let ((s 0d0)) (declare (double-float s)) (dotimes (th nth*) (incf s (aref *wte-parts* th))) (setf (aref parts 0) s)))"
assert b in s
s=s.replace(b,b+"\n    (incf (aref *tm* 1) (- (now) (aref *tm* 3))) (setf (aref *tm* 3) (now))",1)
c="      norm)))\n\n;;; reference (serial)"
assert c in s
s=s.replace(c,"      (incf (aref *tm* 2) (- (now) (aref *tm* 3)))\n"+c,1)
d="(run-pool pool (lambda (i) (forward-backward m (nth i ws) ntc 0 t scale)))"
assert d in s
s=s.replace(d,"(let ((t0 (now))) "+d+" (incf (aref *tm* 0) (- (now) t0)))",1)
e="               (incf step)\n"
s=s.replace(e,e+"               (when (zerop (mod step 100)) (format t \"TM step ~d fb ~,1f reduce ~,1f update ~,1f ms/step~%\" step (* 10 (aref *tm* 0)) (* 10 (aref *tm* 1)) (* 10 (aref *tm* 2))) (fill *tm* 0d0))\n",1)
open(sys.argv[2],'w').write(s)
