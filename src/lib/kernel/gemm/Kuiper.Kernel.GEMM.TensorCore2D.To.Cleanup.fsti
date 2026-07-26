module Kuiper.Kernel.GEMM.TensorCore2D.To.Cleanup

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy }
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore

module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

noextract
ghost fn cleanup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab,
     scalar et_acc, real_like et_acc |}
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (nthr : szp {
    Kuiper.SizeT.v nthr ==
      bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (#_ : squash (Kuiper.SizeT.fits (bm * bk) /\
                Kuiper.SizeT.fits (bk * bn)))
  (#_ : squash (Kuiper.SizeT.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (sA : array2 et_ab (rm bm bk) { core sA == fst sh })
  (sB : array2 et_ab (rm bk bn) { core sB == fst (snd sh) })
  (tid : szlt nthr)
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  (rAcc : chest2 real (wm * tm) (wn * tn))
  requires
    scratch_tile_live bm bn bk tm tn nthr sh tid **
    fragarrayAcc_approximates wm wn accFrags rAcc **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr)
  ensures
    shared_thread_live bm bn bk tm tn nthr sh tid
