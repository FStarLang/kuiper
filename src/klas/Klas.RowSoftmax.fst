module Klas.RowSoftmax

#lang-pulse
open Kuiper

module K = Kuiper.Kernel.RowSoftmax
module Array2 = Kuiper.Array2
module SZ = Kuiper.SizeT
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l2_row_major }

inline_for_extraction noextract
fn inst_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (a : Array2.t et (l2_row_major m n) { Array2.is_global a })
  (#sa : ematrix et m n)
  (ra : ematrix real m n)
  preserves cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  ensures
    exists* (sa' : ematrix et m n).
      on gpu_loc (a |-> sa') **
      pure (sa' %~ K.row_softmax_real #(SZ.v m) #(SZ.v n) ra)
{
  K.row_softmax_gpu #et m n a #sa ra;
}

let row_softmax_rm_f32 = inst_gpu #f32
let row_softmax_rm_f64 = inst_gpu #f64
