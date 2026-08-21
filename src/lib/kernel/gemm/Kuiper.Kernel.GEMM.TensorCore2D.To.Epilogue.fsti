module Kuiper.Kernel.GEMM.TensorCore2D.To.Epilogue

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueLoopStep

inline_for_extraction noextract
fn epilogue_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd,
     scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n : szp)
  (gC : array2 et_cd (rm m n))
  (#fC : perm)
  (#eC : chest2 et_cd m n)
  (#rC : chest2 real m n)
  (gD : array2 et_cd (rm m n))
  (d : epilogue_dims m n)
  (sh : c_shmems
    (shmems_desc_to et_ab et_acc d.bm d.bn d.bk d.tm d.tn d.nthr))
  (accFrags : array
    (fragment et_acc FragAcc d.tm d.tn d.tk FragLAcc))
  (rAcc : chest2 real (d.wm * d.tm) (d.wn * d.tn))
  (bid : szlt (m / d.bm * (n / d.bn)))
  (tid : szlt d.nthr)
  (#_ : squash (Pulse.Lib.Array.length accFrags == d.wm * d.wn))
  norewrite
  requires
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      d.bm d.bn d.bk d.tm d.tn d.tk d.wm d.wn d.nthr
      sh accFrags rAcc tid **
    output_lane_live gD d.bm d.bn d.tm d.tn d.wm d.wn bid tid
  ensures
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      d.bm d.bn d.bk d.tm d.tn d.tk d.wm d.wn d.nthr
      sh accFrags rAcc tid **
    output_lane_approximates
      gD d.bm d.bn d.tm d.tn d.wm d.wn bid tid
      (epilogue_warp_output comb_r rC
        d.bm d.bn d.tm d.tn d.wm d.wn bid tid rAcc)
