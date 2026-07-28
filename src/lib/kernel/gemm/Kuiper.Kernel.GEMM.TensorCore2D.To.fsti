module Kuiper.Kernel.GEMM.TensorCore2D.To

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
module MS = Kuiper.Spec.GEMM

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
// ^ Only opened here for `constraints`? If so would be nice
// to factor out.

inline_for_extraction noextract
val mk_kernel
  (#et_ab #et_cd #et_acc: Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb: et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA { is_global gA })
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB { is_global gB })
  (#eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n) { is_global gC })
  (#_ : squash (SZ.fits (m * n)))
  (#eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n) { is_global gD })
  (#eD : chest2 et_cd m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#fA #fB #fC: perm)
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (nblk <= max_blocks))
  (#_ : squash (nthr <= max_threads))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** pure (eA %~ rA) **
       gB |-> Frac fB eB ** pure (eB %~ rB) **
       gC |-> Frac fC eC ** pure (eC %~ rC) ** 
       live gD)
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       gC |-> Frac fC eC ** 
       (exists* (eD' : chest2 et_cd m n).
         gD |-> eD' ** pure (eD' %~ MS.mmcomb comb_r rC rA rB)))
