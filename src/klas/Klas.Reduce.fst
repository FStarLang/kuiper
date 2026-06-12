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
module HRedB = Kuiper.Kernel.HReduce.Block
module Map = Kuiper.Kernel.Map

let mean_row_aux
  (#rows #cols : nat)
  (inv_cols : f32)
  (sx : EM.ematrix f32 rows cols)
  (s_sum : lseq f32 rows)
  (r : nat { r < rows })
  : Lemma
      (requires (s_sum @! r) %~ rsum (lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r)))
      (ensures
        (exists (sumr : f32).
          sumr %~ rsum (lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r)) /\
          (lseq_map (mean_scale inv_cols) s_sum) @! r == mean_scale inv_cols sumr))
  = assert ((lseq_map (mean_scale inv_cols) s_sum) @! r == mean_scale inv_cols (s_sum @! r))

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn mean_fw_f32_row_impl
  (rows : szp { rows <= max_blocks })
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
{
  let vr : erased (EM.ematrix real rows cols) = hide (EM.to_real_matrix (reveal sx));
  EM.lemma_to_real_matrix_approximates (reveal sx);
  HRedB.reduce_batched_block #f32 id id rows cols 1024sz
    #_ #(c_l2_row_major _ _)
    #_ #(c_l1_forward _)
    x y (reveal vr);
  with sy1. assert (on gpu_loc (y |-> sy1));

  assert pure (rows <= max_blocks * max_threads);
  Map.map_gpu (mean_scale inv_cols) rows #_ #(c_l1_forward _) y;

  Classical.forall_intro
    (Classical.move_requires
       (mean_row_aux inv_cols (reveal sx) (reveal sy1)));
  ()
}
#pop-options

let mean_fw_f32_row : mean_fw_f32_row_ty = mean_fw_f32_row_impl
