module Kuiper.Kernel.GEMM.TensorCore2D.To.PrepareEpilogue

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

noextract
ghost fn prepare_epilogue
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd,
     scalar et_acc, real_like et_acc |}
  (#m #n : szp)
  (gC : array2 et_cd (rm m n))
  (fC : perm)
  (eC : chest2 et_cd m n)
  (rC : chest2 real m n)
  (bm bn bk tm tn tk wm wn nthr : szp {
    constraints bm bn bk tm tn tk wm wn /\
    Kuiper.SizeT.v nthr ==
      bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (#_ : squash (Kuiper.SizeT.fits (bm * bk) /\
                Kuiper.SizeT.fits (bk * bn)))
  (#_ : squash (Kuiper.SizeT.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc) {
    Pulse.Lib.Array.length accFrags == wm * wn })
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (tid : szlt nthr)
  requires
    pure (eC %~ rC) **
    gpu **
    thread_id nthr tid **
    gC |-> Frac fC eC **
    fragarrayAcc_approximates wm wn accFrags rAcc **
    scratch_tile_live bm bn bk tm tn nthr sh tid
  ensures
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid
{
  fold epilogue_frame #et_ab #et_cd #et_acc
    #_ #_ #_ #_ #_
    #m #n gC fC eC rC
    bm bn bk tm tn tk wm wn nthr sh accFrags rAcc tid;
}
