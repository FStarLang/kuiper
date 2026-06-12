module Klas.Reduce

#lang-pulse

open Kuiper
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Seq.Common
module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let mean_scale (inv_cols : f32) (x : f32) : f32 = mul x inv_cols

inline_for_extraction noextract
let mean_fw_f32_row_post
  (rows cols : nat)
  (inv_cols : f32)
  (sx : EM.ematrix f32 rows cols)
  (sy : lseq f32 rows)
  : prop
  = forall (r : nat). r < rows ==>
      exists (sumr : f32).
        sumr %~ rsum (lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r)) /\
        sy @! r == mean_scale inv_cols sumr

inline_for_extraction noextract
type mean_fw_f32_row_ty =
  fn (rows : szp { rows <= max_blocks })
     (cols : szp { SZ.fits (cols + max_threads) })
     (inv_cols : f32)
     (x : Array2.t f32 (l2_row_major rows cols) { Array2.is_global x })
     (y : Array1.t f32 (l1_forward rows) { Array1.is_global y })
     (#sx : erased (EM.ematrix f32 rows cols))
     (#sy : erased (lseq f32 rows))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    on gpu_loc (y |-> sy)
  ensures
    on gpu_loc (x |-> sx) **
    (exists* (sy' : lseq f32 rows).
       on gpu_loc (y |-> sy') **
       pure (mean_fw_f32_row_post rows cols inv_cols (reveal sx) sy'))

val mean_fw_f32_row : mean_fw_f32_row_ty
