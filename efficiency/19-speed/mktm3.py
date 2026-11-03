import sys
s=open(sys.argv[1]).read()
def wrap(s, head, idx, start, end):
    out=[];i=start
    seg=s[start:end]; res=''; p=0
    while True:
        j=seg.find(head,p)
        if j<0: res+=seg[p:]; break
        # find matching paren
        d=0;k=j
        while True:
            c=seg[k]
            if c=='(' : d+=1
            elif c==')':
                d-=1
                if d==0: break
            k+=1
        res+=seg[p:j]+"(tmk %d %s)"%(idx,seg[j:k+1]); p=k+1
    return s[:start]+res+s[end:]
s=s.replace("(defun forward-backward (m w nt sf backprop scale)",
"""(defvar *tk* (make-array 16 :element-type 'double-float :initial-element 0d0))
(defun now () (multiple-value-bind (s ns) (sb-unix:clock-gettime 1) (+ (coerce s 'double-float) (* 1d-9 ns))))
(defmacro tmk (i &body body) `(if (eq sb-thread:*current-thread* (sb-thread:main-thread)) (let ((t0 (now))) (multiple-value-prog1 (progn ,@body) (incf (aref *tk* ,i) (- (now) t0)))) (progn ,@body)))
(defun forward-backward (m w nt sf backprop scale)""",1)
a=s.index("(defun forward-backward"); b=s.index("(defun ssm-head-and-rest")
body=s[a:b]
# within forward-backward + trunk-back
for head,idx in [("(layernorm ",0),("(linear ",1),("(attention ",2),("(gelu ",3),("(vadd ",4),("(vmul ",4),("(ln-back ",5),("(linear-back ",6),("(gelu-back ",7),("(attention-back ",8)]:
    body=wrap(body,head,idx,0,len(body))
s=s[:a]+body+s[b:]
e="               (incf step)\n"
names="ln linear attn gelu vadd/vmul ln-back linear-back gelu-back attn-back"
s=s.replace(e,e+"               (when (zerop (mod step 100)) (format t \"TK step ~d (ms/step main thread) \"~%step) "+"".join("(format t \"%s ~,1f \" (* 10 (aref *tk* %d))) "%(n,i) for i,n in enumerate(names.split()))+"(terpri) (fill *tk* 0d0))\n",1)
open(sys.argv[2],'w').write(s)
