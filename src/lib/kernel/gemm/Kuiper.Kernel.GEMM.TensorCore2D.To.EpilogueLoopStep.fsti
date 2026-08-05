module Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueLoopStep

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
module RO = Kuiper.TensorRO
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Epilogue
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

inline_for_extraction noextract
fn epilogue_loop_step
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
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (SZ.fits (wm * wn)))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (SZ.fits (tm * tn + warp_size)))
  (#_ : squash (chunk et_cd /?+ tn))
  (sh : c_shmems
    (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (accFrags : array
    (fragment et_acc FragAcc tm tn tk FragLAcc))
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  (wid : szlt (bm / (wm * tm) * (bn / (wn * tn))))
  (lane : szlt warp_size)
  (#_ : squash (
    SZ.v wid == SZ.v tid / warp_size /\
    SZ.v lane == SZ.v tid % warp_size))
  (rCWarp : chest2 real (wm * tm) (wn * tn))
  (#_ : squash (
    rCWarp == epilogue_warp_input rC bm bn tm tn wm wn bid tid))
  (idx : ref (szle (wm * wn)))
  (done : szle (wm * wn) { SZ.v done < wm * wn })
  norewrite
  requires
    pure (aligned 16 (RO.core gC) /\ aligned 16 (T.core gD) /\
          aligned_strided_row_major (chunk et_cd) str /\
          aligned_strided_row_major (chunk et_cd) strD) **
    idx |-> done **
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      bm bn bk tm tn tk wm wn nthr
      sh accFrags rAcc tid **
    output_epilogue_state
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc)
      (SZ.v done)
  ensures
    pure (aligned 16 (RO.core gC) /\ aligned 16 (T.core gD) /\
          aligned_strided_row_major (chunk et_cd) str /\
          aligned_strided_row_major (chunk et_cd) strD) **
    (exists* (next : szle (wm * wn)).
      idx |-> next **
      pure (SZ.v next == SZ.v done + 1)) **
    epilogue_frame #et_ab #et_cd #et_acc
      #_ #_ #_ #_ #_
      #m #n gC fC eC rC
      bm bn bk tm tn tk wm wn nthr
      sh accFrags rAcc tid **
    output_epilogue_state
      gD bm bn tm tn wm wn bid wid lane
      (chest_comb comb_r rCWarp rAcc)
      (SZ.v done + 1)
