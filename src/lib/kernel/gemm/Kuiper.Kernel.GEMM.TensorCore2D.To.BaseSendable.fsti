module Kuiper.Kernel.GEMM.TensorCore2D.To.BaseSendable

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy }

module SZ = Kuiper.SizeT

open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

val kpre1_sendable_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA { is_global gA })
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB { is_global gB })
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n) { is_global gC })
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n) { is_global gD })
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of (
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)

val kpost1_sendable_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA { is_global gA })
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB { is_global gB })
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n) { is_global gC })
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n) { is_global gD })
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of (
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
