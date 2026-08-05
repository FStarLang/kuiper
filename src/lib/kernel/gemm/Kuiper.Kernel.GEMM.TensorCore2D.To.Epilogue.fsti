module Kuiper.Kernel.GEMM.TensorCore2D.To.Epilogue

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
module RO = Kuiper.TensorRO
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array

module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueLoopStep

inline_for_extraction noextract
fn epilogue_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd, has_vec_cpy et_cd,
     scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n : szp)
  (#lC : RO.vlayout2 m n)
  {| str : strided_row_major lC,
     strD : strided_row_major (vtlayout_of_tlayout (rm m n)) |}
  (gC : RO.roarray2 et_cd lC)
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
  (#_ : squash (chunk et_cd /?+ d.tn))
  norewrite
  preserves
    pure (aligned 16 (RO.core gC) /\ aligned 16 (T.core gD) /\
          aligned_strided_row_major (chunk et_cd) str /\
          aligned_strided_row_major (chunk et_cd) strD)
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
      (chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC d.bm d.bn
            (bid / (n / d.bn)) (bid % (n / d.bn)))
          (d.wm * d.tm) (d.wn * d.tn)
          ((tid / warp_size) / (d.bn / (d.wn * d.tn)))
          ((tid / warp_size) % (d.bn / (d.wn * d.tn))))
        rAcc)
