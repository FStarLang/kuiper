module Kuiper.Kernel.RowSoftmax

(* Per-row softmax over a 2D tensor resident on the GPU.

   Implementation: TWO kernel launches (independent of m).
     1. [Kuiper.Kernel.HReduce.Block.reduce_batched_block exp exp]
        — one block per row, tree-reduces row sums of exp(x).
     2. [Kuiper.Kernel.RowBroadcast.row_broadcast (fun x s -> div (exp x) s)]
        — one thread per cell (i, j); writes
          [a[i, j] := exp(a_old[i, j]) / sums[i]] in place. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Shape
open Kuiper.Seq.Common
module SM = Kuiper.Spec.Softmax
module A1 = Kuiper.Array1
module Array2 = Kuiper.Array2

(* All-real specification: each row independently softmax'd. *)
let row_softmax_real
  (#m #n : nat)
  (ra : ematrix real m n)
  : ematrix real m n
  = mkM (fun i j ->
      Seq.index (SM.softmax_real (ematrix_row ra i)) j)

inline_for_extraction noextract
fn row_softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (#l : Array2.layout m n) {| ctlayout l |}
  (a : Array2.t et l { Array2.is_global a })
  (#sa : ematrix et m n)
  (ra : ematrix real m n)
  preserves
    cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  ensures
    exists* (sa' : ematrix et m n).
      on gpu_loc (a |-> sa') **
      pure (sa' %~ row_softmax_real ra)

// identical to the above but it returns the temporary sums array it creates
inline_for_extraction noextract
fn row_softmax_gpu_with_sum
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (#l : Array2.layout m n) {| ctlayout l |}
  (a : Array2.t et l { Array2.is_global a })
  (#sa : ematrix et m n)
  (ra : ematrix real m n)
  preserves
    cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  returns 
    sums: (sums: A1.t et (l1_forward m) { A1.is_global sums })
  ensures
    exists* (sa' : ematrix et m n) (esums : lseq et m).
      on gpu_loc (a |-> sa') ** 
      on gpu_loc (sums |-> esums) **
      pure (sa' %~ row_softmax_real ra) **
      pure (esums %~ Seq.init_ghost m (fun i -> rsum (lseq_map exp (ematrix_row ra i))))
