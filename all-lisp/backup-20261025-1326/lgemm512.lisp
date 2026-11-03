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
(defconstant +mr+ 6) (defconstant +nr+ 32) (defconstant +kc+ 256) (defconstant +mc+ 144) (defconstant +nc+ 1024)

(defvar *bufs* (make-hash-table :test 'eq :synchronized t :weakness :key))
(declaim (inline bufs))
(defun bufs ()
  (let ((th sb-thread:*current-thread*))
    (or (gethash th *bufs*)
        (setf (gethash th *bufs*)
              (list (make-array (+ (* (ceiling +mc+ +mr+) +mr+ +kc+) 64) :element-type 'single-float :initial-element 0f0)
                    (make-array (+ (* +kc+ +nc+) 64) :element-type 'single-float :initial-element 0f0)
                    (make-array (* +mr+ +nr+) :element-type 'single-float :initial-element 0f0))))))

(defmacro rep4 (v &body body) `(progn ,@(loop for i below 4 collect `(symbol-macrolet ((,v ,i)) ,@body))))
(defun pack-a (ta A ao lda i0 mb p0 kb Ap)
  "Ap[(s*kb + p)*MR + r] = op(A)[i0+s*MR+r, p0+p], zero padded."
  (declare (fvec A Ap) (ix ao lda i0 mb p0 kb))
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

(defun pack-b (tb B bo ldb p0 kb j0 nb Bp)
  "Bp[(s*kb + p)*NR + c] = op(B)[p0+p, j0+s*NR+c], zero padded."
  (declare (fvec B Bp) (ix bo ldb p0 kb j0 nb))
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
                  (progn (sb-simd-avx:vzeroupper)
                         (dotimes (c +nr+) (setf (aref Bp (+ o c)) (if (< c jw) (aref B (+ ro c)) 0f0))))))))))
  (sb-simd-avx:vzeroupper))

(declaim (inline sap@))
(defun sap@ (v off) (declare (fvec v) (ix off)) (sb-sys:sap+ (sb-sys:vector-sap v) (* 4 off)))

(defun kernel (kb Ap ao Bp bo alpha beta C co ldc mr nr tmp)
  (declare (ix kb ao bo co ldc mr nr) (fvec Ap Bp C tmp) (single-float alpha beta))
  (if (and (= mr +mr+) (= nr +nr+) (= alpha 1f0) (or (= beta 0f0) (= beta 1f0)))
      ;; fast path (all gpt-eff calls): the VOP writes the tile straight into C, adding C when beta=1
      (sb-sys:with-pinned-objects (Ap Bp C)
        (sb-vm::%zkern (sap@ Ap ao) (sap@ Bp bo) (sap@ C co) kb (* 4 ldc) (if (= beta 0f0) 0 1)))
      (progn
        (sb-sys:with-pinned-objects (Ap Bp tmp)
          (sb-vm::%zkern (sap@ Ap ao) (sap@ Bp bo) (sb-sys:vector-sap tmp) kb (* 4 +nr+) 0))
        (dotimes (r mr)
          (dotimes (cc nr)
            (let ((o (+ co (* r ldc) cc)) (v (* alpha (aref tmp (+ (* r +nr+) cc)))))
              (setf (aref C o) (if (= beta 0f0) v (+ v (* beta (aref C o)))))))))))

(defun sgemm (ta tb m n k alpha A ao lda B bo ldb beta C co ldc)
  (declare (fvec A B C) (ix m n k ao lda bo ldb co ldc) (single-float alpha beta))
  (when (or (zerop m) (zerop n)) (return-from sgemm))
  (when (zerop k)
    (dotimes (i m) (dotimes (j n) (let ((o (+ co (* i ldc) j))) (setf (aref C o) (if (= beta 0f0) 0f0 (* beta (aref C o)))))))
    (return-from sgemm))
  (destructuring-bind (Ap Bp tmp) (bufs)
    (declare (fvec Ap Bp tmp))
    (loop for j0 of-type ix from 0 below n by +nc+ do
      (let ((nb (min +nc+ (- n j0))))
        (loop for p0 of-type ix from 0 below k by +kc+ do
          (let ((kb (min +kc+ (- k p0))) (bet (if (= p0 0) beta 1f0)))
            (pack-b tb B bo ldb p0 kb j0 nb Bp)
            (loop for i0 of-type ix from 0 below m by +mc+ do
              (let ((mb (min +mc+ (- m i0))))
                (pack-a ta A ao lda i0 mb p0 kb Ap)
                (loop for js of-type ix from 0 below nb by +nr+ do
                  (loop for is of-type ix from 0 below mb by +mr+ do
                    (kernel kb Ap (* (floor is +mr+) kb +mr+) Bp (* (floor js +nr+) kb +nr+) alpha bet
                            C (+ co (* (+ i0 is) ldc) j0 js) ldc (min +mr+ (- mb is)) (min +nr+ (- nb js)) tmp)))))))))))
