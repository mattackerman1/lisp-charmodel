;;;; #22 step 0: dump Random-Sampling KD targets from the #21 teacher (e21-v16-full-100k final weights, full val 1.0060).
;;;; For every train input position j in [0, vstart-2] (teacher predicts data[j+1] from data[..j]) draw K=16 ids with
;;;; replacement from the teacher's full softmax at T=1 and store them as u16 at out[j*K .. j*K+15] (counts are implicit
;;;; as repeats; the true token is data[j+1] itself). Positions j=vstart-1.. are left 0 and are never read in training.
;;;; Context: windows of 64 with stride 32, rows 32..63 scored (context 32..63, as the val protocol); the first window
;;;; scores rows 0..63 (j<32 has only j+1 tokens of context), a final window covers the tail.
;;;; Resumable: output written in order per pool call; <out>.progress (written after fsync) holds the next call index,
;;;; the running teacher train NLL and the accumulated wall seconds. RNG seeded per (call, worker) => resume is exact.
;;;; usage: sbcl --dynamic-space-size 4000 --script 22-research/dump-teacher.lisp out=22-research/teacher/kd16.u16
(load "/home/box/lisp-charmodel/efficiency/gpt-eff.lisp")
(in-package :gpt)
(defun dump-main ()
  (setf *ew32* t)   ; teacher was trained with ew=f32
  (let* ((K 16) (vstart 23505859) (tt 64) (half 32) (nb 8) (nth 8) (v 16384)
         (wf (opt "teacher" "21-long/run/e21-v16-full-100k-final-100604.f32"))
         (out (opt "out" "22-research/teacher/kd16.u16")) (progf (concatenate 'string out ".progress"))
         (data (with-open-file (s "08-vocab-up/tokens16384.u16" :element-type '(unsigned-byte 16))
                 (let ((d (make-array (file-length s) :element-type '(unsigned-byte 16)))) (read-sequence d s) d)))
         (clen (with-open-file (s "08-vocab-up/vocab16384.hex")
                 (coerce (loop for l = (read-line s nil) while l collect (floor (length l) 2)) '(simple-array fixnum (*)))))
         (c (make-cfg :v v :d 256 :nl 6 :nh 8 :tt tt)) (m (make-model* c))
         (jmax (- vstart 2)) (nbulk (floor (- jmax 63) 32))          ; bulk windows s=32k, k=1..nbulk
         (per-call (* nb nth)) (ncalls (ceiling nbulk per-call))
         (ws (loop repeat nth collect (make-wk* c nb))) (pool (make-pool* nth))
         (stage (make-array (* per-call half K) :element-type '(unsigned-byte 16)))
         (wl (make-array nth :element-type 'double-float :initial-element 0d0))
         (wc (make-array nth :element-type 'fixnum :initial-element 0))
         (prog (if (probe-file progf) (with-open-file (s progf) (read s)) (list :call 0 :loss 0d0 :ntok 0 :nchar 0 :secs 0d0)))
         (c0 (getf prog :call)) (loss (getf prog :loss)) (ntok (getf prog :ntok)) (nchar (getf prog :nchar))
         (secs0 (getf prog :secs)) (t0 (get-internal-real-time)) (r0 (get-internal-run-time)) (scale 1f0))
    (declare ((simple-array (unsigned-byte 16) (*)) data stage) ((simple-array fixnum (*)) clen))
    (labels ((now () (+ secs0 (/ (- (get-internal-real-time) t0) internal-time-units-per-second 1d0)))
             (log* (fmt &rest args) (format t "~a ~?~%" (now) fmt args) (finish-output))
             (sample-rows (w nrows sf s-of-seq dst dst-j0 rs wi)
               ;; rows r with (mod r tt) >= sf of the NROWS*tt rows; window start of seq q = (funcall s-of-seq q)
               (let ((P (w-P w)) (us (make-array K :element-type 'double-float)))
                 (declare (fvec P) ((simple-array double-float (*)) us) ((simple-array (unsigned-byte 16) (*)) dst))
                 (dotimes (q nrows)
                   (let ((s (funcall s-of-seq q)))
                     (loop for i from sf below tt
                           do (let* ((r (+ (* q tt) i)) (o (* r v)) (j (+ s i)) (dk (* (- j dst-j0) K)) (tg (aref data (1+ j))))
                                (declare (fixnum r o j dk))
                                (incf (aref wl wi) (- (log (max 1d-30 (coerce (aref P (+ o tg)) 'double-float)))))
                                (incf (aref wc wi) (aref clen tg))
                                (dotimes (a K) (setf (aref us a) (random 1d0 rs)))
                                (sort us #'<)
                                (let ((acc 0d0) (a 0) (last 0))
                                  (declare (double-float acc) (fixnum a last))
                                  (loop for jj of-type fixnum below v while (< a K)
                                        do (let ((p (aref P (+ o jj))))
                                             (when (> p 0f0) (setf last jj) (incf acc p)
                                               (loop while (and (< a K) (< (aref us a) acc)) do (setf (aref dst (+ dk a)) jj) (incf a)))))
                                  (loop while (< a K) do (setf (aref dst (+ dk a)) last) (incf a)))))))))
             (fw (w nseq sf starts)   ; load NSEQ windows into W and run the teacher forward
               (setf (w-b w) nseq)
               (loop for q below nseq for s in starts
                     do (dotimes (i tt) (setf (aref (w-ids w) (+ (* q tt) i)) (aref data (+ s i)) (aref (w-tgt w) (+ (* q tt) i)) (aref data (+ s i 1)))))
               (setf (w-drop w) nil (w-loss w) 0d0 (w-count w) 0)
               (forward-backward m w tt sf nil scale)))
      (with-open-file (fi wf :element-type '(unsigned-byte 32))
        (assert (= (file-length fi) (loop for pv in (model-vecs m) sum (length (car pv)))))
        (dolist (pv (model-vecs m))
          (let ((vec (car pv))) (dotimes (i (length vec)) (setf (aref vec i) (sb-kernel:make-single-float (let ((u (read-byte fi))) (if (>= u #x80000000) (- u #x100000000) u))))))))
      (log* "teacher ~a loaded; train positions j=0..~d, K=~d, ~d bulk windows in ~d calls; resuming at call ~d" wf jmax K nbulk ncalls c0)
      (with-open-file (fo out :direction :io :element-type '(unsigned-byte 16) :if-exists :overwrite :if-does-not-exist :create)
        (when (zerop c0)   ; first window: rows 0..63 of s=0
          (let ((w (car ws)) (dst (make-array (* tt K) :element-type '(unsigned-byte 16))))
            (fw w 1 0 (list 0)) (sample-rows w 1 0 (lambda (q) q 0) dst 0 (sb-ext:seed-random-state 7) 0)
            (file-position fo 0) (write-sequence dst fo)))
        (loop for call from c0 below ncalls
              do (let* ((k0 (1+ (* call per-call))) (jbase (+ (* 32 k0) half)))
                   (fill wl 0d0) (fill wc 0)
                   (run-pool pool (lambda (i)
                                    (let* ((kw (+ k0 (* i nb))) (n (max 0 (min nb (- (1+ nbulk) kw)))))
                                      (when (plusp n)
                                        (let ((w (nth i ws)))
                                          (fw w n half (loop for q below n collect (* 32 (+ kw q))))
                                          (sample-rows w n half (lambda (q) (* 32 (+ kw q))) stage jbase
                                                       (sb-ext:seed-random-state (+ 1000 (* call nth) i)) i))))))
                   (let* ((kl (min nbulk (+ k0 per-call -1))) (jend (+ (* 32 kl) tt)))
                     (file-position fo (* jbase K)) (write-sequence stage fo :end (* (- jend jbase) K))
                     (incf loss (reduce #'+ wl)) (incf nchar (reduce #'+ wc)) (incf ntok (- jend jbase))
                   (when (or (zerop (mod (1+ call) 100)) (= (1+ call) ncalls))
                     (finish-output fo) (fsync-stream fo)
                     (with-open-file (p (concatenate 'string progf ".tmp") :direction :output :if-exists :supersede)
                       (with-standard-io-syntax (prin1 (list :call (1+ call) :loss loss :ntok ntok :nchar nchar :secs (now)) p)) (fsync-stream p))
                     (atomic-rename (concatenate 'string progf ".tmp") progf)
                     (log* "call ~d/~d  j<~d  teacher train NLL ~,4f nats/char (bulk rows so far)  cpu ~,0fs this segment" (1+ call) ncalls jend
                           (/ loss (max 1 nchar)) (/ (- (get-internal-run-time) r0) internal-time-units-per-second 1d0))))))
        ;; tail window
        (let* ((kl nbulk) (covered (+ (* 32 kl) 63)) (st (- jmax 63)) (sf (- (1+ covered) st)))
          (when (<= covered (1- jmax))
            (let ((w (car ws)) (dst (make-array (* tt K) :element-type '(unsigned-byte 16))))
              (fw w 1 sf (list st)) (sample-rows w 1 sf (lambda (q) q st) dst (+ st sf) (sb-ext:seed-random-state 9) 0)
              (file-position fo (* (+ st sf) K)) (write-sequence dst fo :end (* (- tt sf) K))))
          (file-position fo (* vstart K)) ; pad file to vstart*K entries
          (let ((pad (make-array (- (* vstart K) (* (1+ jmax) K)) :element-type '(unsigned-byte 16) :initial-element 0)))
            (file-position fo (* (1+ jmax) K)) (write-sequence pad fo))
          (finish-output fo) (fsync-stream fo)))
      (log* "DUMP-DONE wall ~,1fs total (all segments), cpu ~,0fs this segment, ~d bulk positions, teacher train NLL ~,4f nats/char; output ~a"
            (now) (/ (- (get-internal-run-time) r0) internal-time-units-per-second 1d0) ntok (/ loss (max 1 nchar)) out)
      (with-open-file (d (concatenate 'string out ".DONE") :direction :output :if-exists :supersede) (format d "~,1f~%" (now)))
      (stop-pool pool))))
(dump-main)
