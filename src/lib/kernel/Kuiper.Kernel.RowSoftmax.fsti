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
open Kuiper.Real { exp }
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
module SM = Kuiper.Spec.Softmax

(* All-real specification: each row independently softmax'd. *)
let row_softmax_real
  (#m #n : nat)
  (ra : chest2 real m n)
  : chest2 real m n
  = mk2 (fun i j ->
      acc1 (SM.softmax_real (chest2_row ra i)) j)

inline_for_extraction noextract
fn row_softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (nth : szp { nth <= max_threads })
  (#l : layout2 m n) {| ctlayout l |}
  (a : array2 et l { is_global a })
  (#sa : chest2 et m n)
  (ra :  chest2 real m n)
  preserves
    cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  ensures
    exists* (sa' : chest2 et m n).
      on gpu_loc (a |-> sa') **
      pure (sa' %~ row_softmax_real ra)

// Identical to the above but it returns the temporary sums array it creates.
// Useful for some implementations of attention, namely those that need to return the log-sum-exp.
inline_for_extraction noextract
fn row_softmax_gpu_with_sum
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (#l : layout2 m n) {| ctlayout l |}
  (a : array2 et l { is_global a })
  (#sa : ematrix et m n)
  (ra : ematrix real m n)
  preserves
    cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  returns 
    sums: (sums: array1 et (l1_forward m) { is_global sums })
  ensures
    exists* (sa' : ematrix et m n) (esums : chest1 et m).
      on gpu_loc (a |-> sa') ** 
      on gpu_loc (sums |-> esums) **
      pure (sa' %~ row_softmax_real ra) **
      pure (forall (i:nat). i < SZ.v m ==>
              acc1 esums i %~ rsum (lseq_map exp (ematrix_row ra i)))
