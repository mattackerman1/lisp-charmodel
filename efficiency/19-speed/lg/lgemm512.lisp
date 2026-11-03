;;;; lgemm512.lisp -- pure-Lisp SGEMM with an AVX-512 micro-kernel (gemm=lisp512 in gpt-eff.lisp).
;;;; Same API/semantics as lgemm.lisp (package :lgemm, lgemm:sgemm, row-major, op(A)/op(B), offsets, beta=0 never reads C).
;;;; Goto blocking KC x NC panel of op(B) in 32-wide strips, MC x KC block of op(A) in 6-high strips (plain, no
;;;; pre-broadcast: the kernel uses EVEX embedded broadcast m32{1to16}). 6x32 tile = 12 zmm accumulators, computed by
;;;; the custom VOP sb-vm::%zkern in zmm-vops.lisp; alpha/beta merge into C with sb-simd AVX2.
(eval-when (:compile-toplevel :load-toplevel :execute) (require :sb-simd))
(load (merge-pathnames "zmm-vops.lisp" (or *compile-file-truename* *load-truename*)))
(defpackage :lgemm (:use :cl) (:export #:sgemm)) (in-package :lgemm)
(declaim (optimize (speed 3) (safety 0) (debug 0)))
(deftype fvec () '(simple-array single-float (*)))
(deftype ix () '(unsigned-byte 31))
(defconstant +mr+ 6) (defconstant +nr+ 32)
(defun envi (name def) (let ((v (sb-ext:posix-getenv name))) (if (and v (plusp (length v))) (parse-integer v) def)))
;; Tuning knobs (env, read at load): LGEMM_KC/MC/NC block sizes; LGEMM_PACK=1 AVX-512 gather packing; LGEMM_EDGE=1 masked edge kernel.
(defparameter *kc* (envi "LGEMM_KC" 256)) (defparameter *mc* (* +mr+ (ceiling (envi "LGEMM_MC" 144) +mr+))) (defparameter *nc* (envi "LGEMM_NC" 1024))
(defparameter *simd-pack* (= 1 (envi "LGEMM_PACK" 1))) (defparameter *masked-edge* (= 1 (envi "LGEMM_EDGE" 1)))
(defparameter *pack2* (= 1 (envi "LGEMM_PACK2" 1)))
(declaim (type ix *kc* *mc* *nc*))

(defvar *bufs* (make-hash-table :test 'eq :synchronized t :weakness :key))
(declaim (inline bufs))
(defun bufs ()
  (let ((th sb-thread:*current-thread*))
    (or (gethash th *bufs*)
        (setf (gethash th *bufs*)
              (list (make-array (+ (* (ceiling *mc* +mr+) +mr+ *kc*) 64) :element-type 'single-float :initial-element 0f0)
                    (make-array (+ (* *kc* *nc*) 64) :element-type 'single-float :initial-element 0f0)
                    (make-array (* +mr+ +nr+) :element-type 'single-float :initial-element 0f0)
                    (make-array 16 :element-type '(unsigned-byte 32) :initial-element 0)
                    (make-array 2 :element-type 'single-float :initial-element 0f0))))))

(defmacro rep4 (v &body body) `(progn ,@(loop for i below 4 collect `(symbol-macrolet ((,v ,i)) ,@body))))
(declaim (inline sap@))
(defun sap@ (v off) (declare (fvec v) (ix off)) (sb-sys:sap+ (sb-sys:vector-sap v) (* 4 off)))
(defun gpack (src so ix stride n sstep dst do dstep lanes valid)
  "AVX-512 gather pack: for p<n, dst[do + p*dstep + l] = src[so + p*sstep + l*stride] for l<valid, 0 for valid<=l<lanes."
  (declare (fvec src dst) (type (simple-array (unsigned-byte 32) (16)) ix) (ix so stride n sstep do dstep lanes valid))
  (dotimes (l 16) (setf (aref ix l) (if (< l valid) (* l stride) 0)))
  (sb-sys:with-pinned-objects (src dst ix)
    (sb-vm::%zgather (sap@ src so) (sap@ dst do) (sb-sys:vector-sap ix) n (* 4 dstep) (1- (ash 1 valid)) (1- (ash 1 lanes)) (* 4 sstep))))
(defun pack-a (ta A ao lda i0 mb p0 kb Ap ix)
  "Ap[(s*kb + p)*MR + r] = op(A)[i0+s*MR+r, p0+p], zero padded."
  (declare (fvec A Ap) (ix ao lda i0 mb p0 kb))
  (when *simd-pack*
    (dotimes (s (ceiling mb +mr+))
      (let ((rw (min +mr+ (- mb (* s +mr+)))) (is (+ i0 (* s +mr+))))
        (if (and *pack2* (= rw +mr+) (>= kb 2))
            ;; #19: one 12-lane gather moves TWO k-steps (2 x 6 rows) into the contiguous Ap order (p-major, row-minor)
            (let ((h (ash kb -1)))
              (declare (ix h))
              (dotimes (l 12) (setf (aref ix l) (if ta (+ (* (floor l +mr+) lda) (mod l +mr+)) (+ (floor l +mr+) (* (mod l +mr+) lda)))))
              (dotimes (l 4) (setf (aref ix (+ 12 l)) 0))
              (sb-sys:with-pinned-objects (A Ap ix)
                (sb-vm::%zgather (sap@ A (if ta (+ ao (* p0 lda) is) (+ ao (* is lda) p0))) (sap@ Ap (* s kb +mr+)) (sb-sys:vector-sap ix)
                                 h (* 4 2 +mr+) (1- (ash 1 12)) (1- (ash 1 12)) (* 4 2 (if ta lda 1))))
              (when (oddp kb)   ; last k-step (single, 6 lanes)
                (if ta
                    (gpack A (+ ao (* (+ p0 kb -1) lda) is) ix 1 1 0 Ap (+ (* s kb +mr+) (* (1- kb) +mr+)) 0 +mr+ +mr+ rw)
                    (gpack A (+ ao (* is lda) p0 (1- kb)) ix lda 1 0 Ap (+ (* s kb +mr+) (* (1- kb) +mr+)) 0 +mr+ +mr+ rw))))
            (if ta
                (gpack A (+ ao (* p0 lda) is) ix 1 kb lda Ap (* s kb +mr+) +mr+ +mr+ rw)
                (gpack A (+ ao (* is lda) p0) ix lda kb 1 Ap (* s kb +mr+) +mr+ +mr+ rw)))))
    (return-from pack-a))
  (dotimes (s (ceiling mb +mr+))
    (let ((base (* s kb +mr+)) (rw (min +mr+ (- mb (* s +mr+)))) (is (+ i0 (* s +mr+))))
      (declare (ix base rw is))
      (if ta
          (dotimes (p kb)
            (let ((ro (+ ao (* (+ p0 p) lda) is)) (o (+ base (* p +mr+))))
              (dotimes (r +mr+) (setf (aref Ap (+ o r)) (if (< r rw) (aref A (+ ro r)) 0f0)))))
          (dotimes (r +mr+)
            (if (< r rw)
                (let ((ro (+ ao (* (+ is r) lda) p0))) (dotimes (p kb) (setf (aref Ap (+ base (* p +mr+) r)) (aref A (+ ro p)))))
                (dotimes (p kb) (setf (aref Ap (+ base (* p +mr+) r)) 0f0))))))))

(defun pack-b (tb B bo ldb p0 kb j0 nb Bp ix)
  "Bp[(s*kb + p)*NR + c] = op(B)[p0+p, j0+s*NR+c], zero padded."
  (declare (fvec B Bp) (ix bo ldb p0 kb j0 nb))
  (when (and *simd-pack* tb)
    (dotimes (s (ceiling nb +nr+))
      (let ((jw (min +nr+ (- nb (* s +nr+)))) (js (+ j0 (* s +nr+))))
        (dotimes (h 2)
          (gpack B (+ bo (* (+ js (* 16 h)) ldb) p0) ix ldb kb 1 Bp (+ (* s kb +nr+) (* 16 h)) +nr+ 16 (max 0 (min 16 (- jw (* 16 h))))))))
    (return-from pack-b))
  (dotimes (s (ceiling nb +nr+))
    (let ((base (* s kb +nr+)) (jw (min +nr+ (- nb (* s +nr+)))) (js (+ j0 (* s +nr+))))
      (declare (ix base jw js))
      (if tb
          (dotimes (c +nr+)
            (if (< c jw)
                (let ((ro (+ bo (* (+ js c) ldb) p0))) (dotimes (p kb) (setf (aref Bp (+ base (* p +nr+) c)) (aref B (+ ro p)))))
                (dotimes (p kb) (setf (aref Bp (+ base (* p +nr+) c)) 0f0))))
          (dotimes (p kb)
            (let ((ro (+ bo (* (+ p0 p) ldb) js)) (o (+ base (* p +nr+))))
              (declare (ix ro o))
              (if (= jw +nr+)
                  (rep4 q (sb-simd-avx::%f32.8-store (sb-simd-avx::%f32.8-load B ro (* 8 q)) Bp o (* 8 q)))
                  (if *simd-pack*
                      (dotimes (h 2) (gpack B (+ ro (* 16 h)) ix 1 1 0 Bp (+ o (* 16 h)) 0 16 (max 0 (min 16 (- jw (* 16 h))))))
                    (progn (sb-simd-avx:vzeroupper)
                         (dotimes (c +nr+) (setf (aref Bp (+ o c)) (if (< c jw) (aref B (+ ro c)) 0f0)))))))))))
  (sb-simd-avx:vzeroupper))


(defun kernel (kb Ap ao Bp bo alpha beta C co ldc mr nr tmp sc)
  (declare (ix kb ao bo co ldc mr nr) (fvec Ap Bp C tmp sc) (single-float alpha beta))
  (cond
   ((and *masked-edge* (not (and (= mr +mr+) (= nr +nr+) (= alpha 1f0) (or (= beta 0f0) (= beta 1f0)))))
    ;; any tile/alpha/beta written straight into C with k-masked zmm loads/stores (no scratch tile)
    (setf (aref sc 0) alpha (aref sc 1) beta)
    (sb-sys:with-pinned-objects (Ap Bp C sc)
      (sb-vm::%zkern2 (sap@ Ap ao) (sap@ Bp bo) (sap@ C co) kb (* 4 ldc) (if (= beta 0f0) 0 1) mr (1- (ash 1 nr)) (sb-sys:vector-sap sc))))
   (t (if (and (= mr +mr+) (= nr +nr+) (= alpha 1f0) (or (= beta 0f0) (= beta 1f0)))
      ;; fast path (all gpt-eff calls): the VOP writes the tile straight into C, adding C when beta=1
      (sb-sys:with-pinned-objects (Ap Bp C)
        (sb-vm::%zkern (sap@ Ap ao) (sap@ Bp bo) (sap@ C co) kb (* 4 ldc) (if (= beta 0f0) 0 1)))
      (progn
        (sb-sys:with-pinned-objects (Ap Bp tmp)
          (sb-vm::%zkern (sap@ Ap ao) (sap@ Bp bo) (sb-sys:vector-sap tmp) kb (* 4 +nr+) 0))
        (dotimes (r mr)
          (dotimes (cc nr)
            (let ((o (+ co (* r ldc) cc)) (v (* alpha (aref tmp (+ (* r +nr+) cc)))))
              (setf (aref C o) (if (= beta 0f0) v (+ v (* beta (aref C o)))))))))))))

(defun sgemm (ta tb m n k alpha A ao lda B bo ldb beta C co ldc)
  (declare (fvec A B C) (ix m n k ao lda bo ldb co ldc) (single-float alpha beta))
  (when (or (zerop m) (zerop n)) (return-from sgemm))
  (when (zerop k)
    (dotimes (i m) (dotimes (j n) (let ((o (+ co (* i ldc) j))) (setf (aref C o) (if (= beta 0f0) 0f0 (* beta (aref C o)))))))
    (return-from sgemm))
  (destructuring-bind (Ap Bp tmp ix sc) (bufs)
    (declare (fvec Ap Bp tmp sc))
    (loop for j0 of-type ix from 0 below n by *nc* do
      (let ((nb (min *nc* (- n j0))))
        (loop for p0 of-type ix from 0 below k by *kc* do
          (let ((kb (min *kc* (- k p0))) (bet (if (= p0 0) beta 1f0)))
            (pack-b tb B bo ldb p0 kb j0 nb Bp ix)
            (loop for i0 of-type ix from 0 below m by *mc* do
              (let ((mb (min *mc* (- m i0))))
                (pack-a ta A ao lda i0 mb p0 kb Ap ix)
                (loop for js of-type ix from 0 below nb by +nr+ do
                  (loop for is of-type ix from 0 below mb by +mr+ do
                    (kernel kb Ap (* (floor is +mr+) kb +mr+) Bp (* (floor js +nr+) kb +nr+) alpha bet
                            C (+ co (* (+ i0 is) ldc) j0 js) ldc (min +mr+ (- mb is)) (min +nr+ (- nb js)) tmp sc)))))))))))
