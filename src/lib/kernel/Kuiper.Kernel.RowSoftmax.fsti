module Kuiper.Kernel.RowSoftmax

(* Per-row softmax over an [Array2 et m n] resident on the GPU.

   Implementation: TWO kernel launches (independent of m).
     1. [Kuiper.Kernel.HReduce.Block.reduce_batched_block exp exp]
        — one block per row, tree-reduces row sums of exp(x).
     2. [Kuiper.Kernel.RowBroadcast.row_broadcast (fun x s -> div (exp x) s)]
        — one thread per cell (i, j); writes
          [a[i, j] := exp(a_old[i, j]) / sums[i]] in place. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
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
