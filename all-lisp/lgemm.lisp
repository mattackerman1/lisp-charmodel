;;;; lgemm.lisp -- pure-Lisp single-precision GEMM (replaces OpenBLAS cblas_sgemm in gpt-eff.lisp with gemm=lisp).
;;;; Row-major C[m,n] = alpha*op(A)*op(B) + beta*C, op = optional transpose, element offsets, any lda/ldb/ldc.
;;;; Goto-style blocking: KC x NC panel of op(B) packed into 16-wide strips, MC x KC block of op(A) packed into 4-high
;;;; strips, 4x16 register micro-kernel with sb-simd AVX2 FMA (8 ymm accumulators). beta=0 never reads C (BLAS rule).
;;;; Threading: one call runs on the calling thread. gpt-eff.lisp already runs 8 data-parallel worker threads
;;;; (threads=8) that each call GEMM, exactly like OpenBLAS with openblas_set_num_threads(1); pack buffers are per thread.
(eval-when (:compile-toplevel :load-toplevel :execute) (require :sb-simd))
(defpackage :lgemm (:use :cl) (:export #:sgemm)) (in-package :lgemm)
(declaim (optimize (speed 3) (safety 0) (debug 0)))
(deftype fvec () '(simple-array single-float (*)))
(deftype ix () '(unsigned-byte 31))
(defconstant +mr+ 4) (defconstant +nr+ 16) (defconstant +kc+ 256) (defconstant +mc+ 128) (defconstant +nc+ 1024)

(defvar *bufs* (make-hash-table :test 'eq :synchronized t :weakness :key))
(declaim (inline bufs))
(defun bufs ()
  (let ((th sb-thread:*current-thread*))
    (or (gethash th *bufs*)
        (setf (gethash th *bufs*)
              (cons (make-array (* (ceiling +mc+ +mr+) +mr+ +kc+ 8) :element-type 'single-float :initial-element 0f0)
                    (make-array (+ (* +kc+ +nc+) 64) :element-type 'single-float :initial-element 0f0))))))

(defmacro put8 (v o x) `(let ((xv ,x) (oo ,o)) (dotimes (q 8) (setf (aref ,v (+ oo q)) xv))))   ; scalar only: no AVX in pack-a
(defun pack-a (ta A ao lda i0 mb p0 kb Ap)
  "Ap[((s*kb + p)*MR + r)*8 + 0..7] = op(A)[i0+s*MR+r, p0+p] (pre-broadcast x8: sb-simd has no VEX scalar load,
and legacy MOVSS between AVX ops costs SSE/AVX transitions), zero padded. Scalar code only."
  (declare (fvec A Ap) (ix ao lda i0 mb p0 kb))
  (sb-simd-avx:vzeroupper)
  (dotimes (s (ceiling mb +mr+))
    (let ((base (* s kb +mr+)) (rw (min +mr+ (- mb (* s +mr+)))) (is (+ i0 (* s +mr+))))
      (declare (ix base rw is))
      (if ta
          (dotimes (p kb)   ; op(A)[i,p] = A[p*lda + i]: rows of the strip are contiguous for fixed p
            (let ((ro (+ ao (* (+ p0 p) lda) is)) (o (+ base (* p +mr+))))
              (dotimes (r +mr+) (put8 Ap (* 8 (+ o r)) (if (< r rw) (aref A (+ ro r)) 0f0)))))
          (dotimes (r +mr+)
            (if (< r rw)
                (let ((ro (+ ao (* (+ is r) lda) p0))) (dotimes (p kb) (put8 Ap (* 8 (+ base (* p +mr+) r)) (aref A (+ ro p)))))
                (dotimes (p kb) (put8 Ap (* 8 (+ base (* p +mr+) r)) 0f0))))))))

(defun pack-b (tb B bo ldb p0 kb j0 nb Bp)
  "Bp[(s*kb + p)*NR + c] = op(B)[p0+p, j0+s*NR+c], zero padded."
  (declare (fvec B Bp) (ix bo ldb p0 kb j0 nb))
  (let ((ns (ceiling nb +nr+)))
    (dotimes (s ns)
      (let ((base (* s kb +nr+)) (jw (min +nr+ (- nb (* s +nr+)))) (js (+ j0 (* s +nr+))))
        (declare (ix base jw js))
        (if tb
            (dotimes (c +nr+)
              (if (< c jw)
                  (let ((ro (+ bo (* (+ js c) ldb) p0))) (dotimes (p kb) (setf (aref Bp (+ base (* p +nr+) c)) (aref B (+ ro p)))))
                  (dotimes (p kb) (setf (aref Bp (+ base (* p +nr+) c)) 0f0))))
            (dotimes (p kb)
              (let ((ro (+ bo (* (+ p0 p) ldb) js)) (o (+ base (* p +nr+))))
                (if (= jw +nr+)
                    (setf (sb-simd-avx:f32.8-aref Bp o) (sb-simd-avx:f32.8-aref B ro)
                          (sb-simd-avx:f32.8-aref Bp (+ o 8)) (sb-simd-avx:f32.8-aref B (+ ro 8)))
                    (progn (sb-simd-avx:vzeroupper) (dotimes (c +nr+) (setf (aref Bp (+ o c)) (if (< c jw) (aref B (+ ro c)) 0f0))))))))))
    (sb-simd-avx:vzeroupper)))

(defmacro def-kernel ()
  "MR x 16 micro-kernel: MR*2 ymm accumulators; unchecked VEX loads/stores (sb-simd-avx::%f32.8-load/-store, element
offsets) so the hot loop has no bounds checks or spills."
  (let* ((acc (loop for r below +mr+ collect (list (intern (format nil "C~d0" r)) (intern (format nil "C~d1" r))))))
    `(defun kernel (kb Ap ao Bp bo alpha beta C co ldc mr nr tmp)
       (declare (type (unsigned-byte 31) kb ao bo co ldc mr nr) (fvec Ap Bp C tmp) (single-float alpha beta))
       (let ,(loop for (x y) in acc append `((,x (sb-simd-avx:f32.8 0f0)) (,y (sb-simd-avx:f32.8 0f0))))
         (declare (type sb-simd-avx:f32.8 ,@(apply #'append acc)))
         (let ((pa ao) (pb bo))
           (declare (type (unsigned-byte 31) pa pb))
           (loop repeat kb do
             (let ((b0 (sb-simd-avx::%f32.8-load Bp pb 0)) (b1 (sb-simd-avx::%f32.8-load Bp pb 8)))
               ,@(loop for (x y) in acc for r from 0
                       collect `(let ((a (sb-simd-avx::%f32.8-load Ap pa ,(* 8 r))))
                                  (setf ,x (sb-simd-fma:f32.8-fmadd a b0 ,x) ,y (sb-simd-fma:f32.8-fmadd a b1 ,y)))))
             (setf pa (+ pa ,(* 8 +mr+)) pb (+ pb +nr+))))
         (let ((va (sb-simd-avx:f32.8 alpha)) (vb (sb-simd-avx:f32.8 beta)))
           (if (and (= mr +mr+) (= nr +nr+))
               (if (= beta 0f0)
                   (progn ,@(loop for (x y) in acc for r from 0
                                  collect `(let ((o (+ co (* ,r ldc))))
                                             (sb-simd-avx::%f32.8-store (sb-simd-avx:f32.8* va ,x) C o 0)
                                             (sb-simd-avx::%f32.8-store (sb-simd-avx:f32.8* va ,y) C o 8))))
                   (progn ,@(loop for (x y) in acc for r from 0
                                  collect `(let ((o (+ co (* ,r ldc))))
                                             (sb-simd-avx::%f32.8-store (sb-simd-fma:f32.8-fmadd vb (sb-simd-avx::%f32.8-load C o 0) (sb-simd-avx:f32.8* va ,x)) C o 0)
                                             (sb-simd-avx::%f32.8-store (sb-simd-fma:f32.8-fmadd vb (sb-simd-avx::%f32.8-load C o 8) (sb-simd-avx:f32.8* va ,y)) C o 8)))))
               (progn ,@(loop for (x y) in acc for r from 0
                              collect `(progn (sb-simd-avx::%f32.8-store ,x tmp ,(* r 16) 0) (sb-simd-avx::%f32.8-store ,y tmp ,(* r 16) 8)))
                      (sb-simd-avx:vzeroupper)
                      (dotimes (r mr)
                        (dotimes (cc nr)
                          (let ((o (+ co (* r ldc) cc)) (v (* alpha (aref tmp (+ (* r 16) cc)))))
                            (setf (aref C o) (if (= beta 0f0) v (+ v (* beta (aref C o))))))))))))
       (sb-simd-avx:vzeroupper))))
(def-kernel)

(defun sgemm (ta tb m n k alpha A ao lda B bo ldb beta C co ldc)
  (declare (fvec A B C) (ix m n k ao lda bo ldb co ldc) (single-float alpha beta))
  (when (or (zerop m) (zerop n)) (return-from sgemm))
  (when (zerop k)   ; C = beta*C
    (dotimes (i m) (dotimes (j n) (let ((o (+ co (* i ldc) j))) (setf (aref C o) (if (= beta 0f0) 0f0 (* beta (aref C o)))))))
    (return-from sgemm))
  (let* ((bf (bufs)) (Ap (car bf)) (Bp (cdr bf)) (tmp (make-array (* 16 +mr+) :element-type 'single-float)))
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
                    (kernel kb Ap (* (floor is +mr+) kb +mr+ 8) Bp (* (floor js +nr+) kb +nr+) alpha bet
                            C (+ co (* (+ i0 is) ldc) j0 js) ldc (min +mr+ (- mb is)) (min +nr+ (- nb js)) tmp)))))))))))
