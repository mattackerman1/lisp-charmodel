;;;; zmm-vops.lisp -- AVX-512 (EVEX, zmm) micro-kernel as a custom SBCL VOP.
;;;; SBCL 2.5.2's x86-64 assembler has no EVEX encoder and no 512-bit storage class, so the whole MRxNR register-tile
;;;; inner loop is a single VOP whose body emits hand-encoded EVEX instructions (62 P0 P1 P2 opcode modrm disp32) via
;;;; (inst byte ..). It only uses zmm16-zmm31, which SBCL's register allocator never touches (its float SCs are
;;;; xmm0-15), and ends with VZEROUPPER. Pointer/counter operands are normal SBCL GPR TNs (tn-offset = hw encoding).
;;;; Instructions used: VPXORD zmm (zero acc), VMOVUPS zmm<-m512 / m512<-zmm, VFMADD231PS zmm, zmm, m32{1to16}
;;;; (embedded broadcast straight from packed A: no pre-broadcast packing needed), PREFETCHT0 via normal SBCL inst.
(in-package :sb-vm)
(defconstant +zmr+ 6) (defconstant +znr+ 32)

(defun eb (x) (sb-assem:inst* 'byte x))
(defun %evex (seg reg vreg rm mm pp bcst rm-is-zmm)
  (let ((p0 (logior (if (logbitp 3 reg) 0 #x80) (if (and rm-is-zmm (logbitp 4 rm)) 0 #x40)
                    (if (logbitp 3 rm) 0 #x20) (if (logbitp 4 reg) 0 #x10) mm))
        (p1 (logior (ash (logxor 15 (logand vreg 15)) 3) 4 pp))          ; W0
        (p2 (logior #x40 (if bcst #x10 0) (if (logbitp 4 vreg) 0 8))))  ; L'L=10 (512b), z=0, aaa=000
    (dolist (x (list #x62 p0 p1 p2)) (eb x))))
(defun %zmem (seg op mm pp reg vreg base disp &optional bcst)
  "op zmm(reg), zmm(vreg), [gpr(base)+disp32]"
  (%evex seg reg vreg base mm pp bcst nil)
  (eb op)
  (eb (logior #x80 (ash (logand reg 7) 3) (logand base 7)))
  (when (= (logand base 7) 4) (eb #x24))
  (dotimes (i 4) (eb (ldb (byte 8 (* 8 i)) disp))))
(defun z-load (seg z base disp) (%zmem seg #x10 1 0 z 0 base disp))
(defun z-store (seg z base disp) (%zmem seg #x11 1 0 z 0 base disp))
(defun z-fma-bcst (seg acc b base disp) (%zmem seg #xB8 2 1 acc b base disp t))
(defun z-zero (seg z)
  (%evex seg z z z 1 1 nil t) (eb #xEF)
  (eb (logior #xC0 (ash (logand z 7) 3) (logand z 7))))

;;; vaddps zmm(acc), zmm(acc), [base+disp]
(defun z-add-mem (acc base disp) (%zmem nil #x58 1 0 acc acc base disp))

(sb-c:defknown %zkern (system-area-pointer system-area-pointer system-area-pointer (unsigned-byte 31) (unsigned-byte 31) (unsigned-byte 1)) (values) ())

;;; acc(r,h) = zmm(16 + 2r + h), r<6, h<2 -> zmm16..27; B halves in zmm28/29.
;;; Ap per k: 6 floats (24 B); Bp per k: 32 floats (128 B).
;;; Output: row r of the 6x32 tile goes to c + r*ldcb bytes; add=0 stores acc (C = AB), add=1 stores acc + C (C += AB).
(define-vop (zkern)
  (:translate %zkern) (:policy :fast-safe)
  (:args (a :scs (sap-reg) :to :save) (b :scs (sap-reg) :to :save) (c :scs (sap-reg) :to :save)
         (k :scs (unsigned-reg) :to :save) (ldcb :scs (unsigned-reg) :to :save) (add :scs (unsigned-reg) :to :save))
  (:arg-types system-area-pointer system-area-pointer system-area-pointer unsigned-num unsigned-num unsigned-num)
  (:temporary (:sc unsigned-reg) pa) (:temporary (:sc unsigned-reg) pb) (:temporary (:sc unsigned-reg) kk)
  (:generator 100
    (let ((seg nil) (loop-l (gen-label)) (done (gen-label)) (st (gen-label)) (fin (gen-label)))
      (declare (ignorable seg))
      (move pa a) (move pb b) (move kk k)
      (dotimes (z 12) (z-zero seg (+ 16 z)))
      (inst test kk kk) (inst jmp :z done)
      (emit-label loop-l)
      (let ((ra (tn-offset pa)) (rb (tn-offset pb)))
        (z-load seg 28 rb 0) (z-load seg 29 rb 64)
        (inst prefetch :t0 (ea 1024 pb))
        (dotimes (r 6)
          (z-fma-bcst seg (+ 16 (* 2 r)) 28 ra (* 4 r))
          (z-fma-bcst seg (+ 17 (* 2 r)) 29 ra (* 4 r))))
      (inst add pa 24) (inst add pb 128) (inst sub kk 1) (inst jmp :nz loop-l)
      (emit-label done)
      (move pa c)
      (let ((rc (tn-offset pa)))
        (inst test add add) (inst jmp :z st)
        (dotimes (r 6)
          (z-add-mem (+ 16 (* 2 r)) rc 0) (z-add-mem (+ 17 (* 2 r)) rc 64)
          (z-store seg (+ 16 (* 2 r)) rc 0) (z-store seg (+ 17 (* 2 r)) rc 64)
          (inst add pa ldcb))
        (inst jmp fin)
        (emit-label st)
        (dotimes (r 6)
          (z-store seg (+ 16 (* 2 r)) rc 0) (z-store seg (+ 17 (* 2 r)) rc 64)
          (inst add pa ldcb)))
      (emit-label fin)
      (inst vzeroupper))))
