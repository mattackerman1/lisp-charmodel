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
(defun %evex (seg reg vreg rm mm pp bcst rm-is-zmm &optional (aaa 0) z (idx nil))
  "idx: VSIB index zmm number (X/V' then extend the index, not vreg)."
  (let ((p0 (logior (if (logbitp 3 reg) 0 #x80) (if (or (and rm-is-zmm (logbitp 4 rm)) (and idx (logbitp 3 idx))) 0 #x40)
                    (if (logbitp 3 rm) 0 #x20) (if (logbitp 4 reg) 0 #x10) mm))
        (p1 (logior (ash (logxor 15 (logand vreg 15)) 3) 4 pp))          ; W0
        (p2 (logior #x40 (if z #x80 0) (if bcst #x10 0) aaa (if (logbitp 4 (or idx vreg)) 0 8))))  ; L'L=10 (512b), z=0, aaa=000
    (dolist (x (list #x62 p0 p1 p2)) (eb x))))
(defun %zmem (seg op mm pp reg vreg base disp &optional bcst (aaa 0) z)
  "op zmm(reg){k aaa}{z}, zmm(vreg), [gpr(base)+disp32]"
  (%evex seg reg vreg base mm pp bcst nil aaa z)
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

;;;; ---- v2: masked general tile kernel + gather packing ----
(defun z-load-m (z base disp k) (%zmem nil #x10 1 0 z 0 base disp nil k t))      ; vmovups zmm{k}{z}, m512
(defun z-store-m (z base disp k) (%zmem nil #x11 1 0 z 0 base disp nil k nil))   ; vmovups m512{k}, zmm
(defun z-mul-bcst (acc base disp) (%zmem nil #x59 1 0 acc acc base disp t))      ; vmulps acc, acc, m32{1to16}
(defun z-fma-bcst-v (acc b base disp) (%zmem nil #xB8 2 1 acc b base disp t))    ; vfmadd231ps acc, zmm(b), m32{1to16}
(defun kmovw (k r)                                                                ; kmovw k, r32 (3-byte VEX)
  (eb #xC4) (eb (logior #xC0 (if (logbitp 3 r) 0 #x20) 1)) (eb #x78) (eb #x92) (eb (logior #xC0 (ash k 3) (logand r 7))))
(defun z-gather (dst idx base k)                                                  ; vgatherdps zmm(dst){k}, [base + zmm(idx)*4]
  (%evex nil dst 0 base 2 1 nil nil k nil idx)
  (eb #x92) (eb (logior #x40 (ash (logand dst 7) 3) 4)) (eb (logior #x80 (ash (logand idx 7) 3) (logand base 7))) (eb 0))

(sb-c:defknown %zkern2 (system-area-pointer system-area-pointer system-area-pointer (unsigned-byte 31) (unsigned-byte 31)
                        (unsigned-byte 1) (unsigned-byte 31) (unsigned-byte 32) system-area-pointer) (values) ())
;;; Any tile: rows mr<=6, columns given by 32-bit mask (k1 = low 16 = cols 0-15, k2 = high 16). [s]=alpha, [s+4]=beta.
;;; C = alpha*AB (mode 0, C never read) or alpha*AB + beta*C (mode 1). Masked loads/stores: no scratch tile, no overrun.
(define-vop (zkern2)
  (:translate %zkern2) (:policy :fast-safe)
  (:args (a :scs (sap-reg) :to :save) (b :scs (sap-reg) :to :save) (c :scs (sap-reg) :to :save)
         (k :scs (unsigned-reg) :to :save) (ldcb :scs (unsigned-reg) :to :save) (mode :scs (unsigned-reg) :to :save)
         (mr :scs (unsigned-reg) :to :save) (mask :scs (unsigned-reg) :to :save) (s :scs (sap-reg) :to :save))
  (:arg-types system-area-pointer system-area-pointer system-area-pointer unsigned-num unsigned-num unsigned-num
              unsigned-num unsigned-num system-area-pointer)
  (:temporary (:sc unsigned-reg) pa) (:temporary (:sc unsigned-reg) pb) (:temporary (:sc unsigned-reg) kk)
  (:generator 100
    (let ((loop-l (gen-label)) (done (gen-label)) (fin (gen-label)))
      (move pa a) (move pb b) (move kk k)
      (dotimes (z 12) (z-zero nil (+ 16 z)))
      (inst test kk kk) (inst jmp :z done)
      (emit-label loop-l)
      (let ((ra (tn-offset pa)) (rb (tn-offset pb)))
        (z-load nil 28 rb 0) (z-load nil 29 rb 64)
        (inst prefetch :t0 (ea 1024 pb))
        (dotimes (r 6)
          (z-fma-bcst nil (+ 16 (* 2 r)) 28 ra (* 4 r))
          (z-fma-bcst nil (+ 17 (* 2 r)) 29 ra (* 4 r))))
      (inst add pa 24) (inst add pb 128) (inst sub kk 1) (inst jmp :nz loop-l)
      (emit-label done)
      (move kk mask) (kmovw 1 (tn-offset kk)) (inst shr kk 16) (kmovw 2 (tn-offset kk))
      (move pa c)
      (let ((rc (tn-offset pa)) (rs (tn-offset s)))
        (dotimes (r 6)
          (when (> r 0) (inst cmp mr r) (inst jmp :be fin))
          (let ((x (+ 16 (* 2 r))) (y (+ 17 (* 2 r))) (nob (gen-label)))
            (z-mul-bcst x rs 0) (z-mul-bcst y rs 0)
            (inst test mode mode) (inst jmp :z nob)
            (z-load-m 30 rc 0 1) (z-fma-bcst-v x 30 rs 4)
            (z-load-m 31 rc 64 2) (z-fma-bcst-v y 31 rs 4)
            (emit-label nob)
            (z-store-m x rc 0 1) (z-store-m y rc 64 2))
          (inst add pa ldcb)))
      (emit-label fin)
      (inst vzeroupper))))

(sb-c:defknown %zgather (system-area-pointer system-area-pointer system-area-pointer (unsigned-byte 31) (unsigned-byte 31)
                         (unsigned-byte 16) (unsigned-byte 16) (unsigned-byte 31)) (values) ())
;;; for p < n: dst[p*dstep bytes .. ] {k2 lanes} = src[(p + idx[l]) floats] for lanes l in gmask (others 0).
;;; idx = 16 int32 element offsets at sap ix. src advances sstep bytes per p.
(define-vop (zgather)
  (:translate %zgather) (:policy :fast-safe)
  (:args (src :scs (sap-reg) :to :save) (dst :scs (sap-reg) :to :save) (ix :scs (sap-reg) :to :save)
         (n :scs (unsigned-reg) :to :save) (dstep :scs (unsigned-reg) :to :save)
         (gmask :scs (unsigned-reg) :to :save) (smask :scs (unsigned-reg) :to :save) (sstep :scs (unsigned-reg) :to :save))
  (:arg-types system-area-pointer system-area-pointer system-area-pointer unsigned-num unsigned-num unsigned-num unsigned-num unsigned-num)
  (:temporary (:sc unsigned-reg) ps) (:temporary (:sc unsigned-reg) pd) (:temporary (:sc unsigned-reg) kk)
  (:generator 100
    (let ((loop-l (gen-label)) (done (gen-label)))
      (move ps src) (move pd dst) (move kk n)
      (z-load nil 31 (tn-offset ix) 0)
      (kmovw 2 (tn-offset smask))
      (inst test kk kk) (inst jmp :z done)
      (emit-label loop-l)
      (kmovw 1 (tn-offset gmask))
      (z-zero nil 30)
      (z-gather 30 31 (tn-offset ps) 1)
      (z-store-m 30 (tn-offset pd) 0 2)
      (inst add ps sstep) (inst add pd dstep) (inst sub kk 1) (inst jmp :nz loop-l)
      (emit-label done)
      (inst vzeroupper))))
