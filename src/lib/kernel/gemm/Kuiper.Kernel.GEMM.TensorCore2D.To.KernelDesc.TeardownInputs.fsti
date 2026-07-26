module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownInputs

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

let teardown_inputs_pre
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB) (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n)) (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  : slprop
= (forall+ (bid : natlt nblk) (tid : natlt nthr).
    kpost1_to comb_r gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn fA fB fC rA rB rC
      nblk nthr bid tid) **
  pure (SZ.fits ((rm m n).ulen))

let teardown_inputs_post
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB) (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n)) (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  : slprop
= gA |-> Frac fA eA **
  gB |-> Frac fB eB **
  gC |-> Frac fC eC **
  (forall+ (bid : natlt nblk) (tid : natlt nthr).
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (ematrix_subtile
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn (bid / (n / bn)) (bid % (n / bn)))
        (wm * tm) (wn * tn)
        ((tid / warp_size) / (bn / (wn * tn)))
        ((tid / warp_size) % (bn / (wn * tn))))) **
  pure (SZ.fits ((rm m n).ulen))

ghost
fn gather_kernel_outputs
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    teardown_inputs_pre comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn nblk nthr
      fA fB fC rA rB rC
  ensures
    teardown_inputs_post comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn nblk nthr
      fA fB fC rA rB rC
