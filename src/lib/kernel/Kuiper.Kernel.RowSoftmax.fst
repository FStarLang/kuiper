module Kuiper.Kernel.RowSoftmax

#lang-pulse
open Kuiper
open Kuiper.Real { rexp }
open Kuiper.EMatrix
open Kuiper.Seq.Common
module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2
module SZ = Kuiper.SizeT
module KB = Kuiper.Kernel.HReduce.Block
module RB = Kuiper.Kernel.RowBroadcast
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

(* ── Approximation glue: cell-wise softmax ───────────────────────────── *)

(* If [sums @! i %~ rsum (lseq_map rexp (ematrix_row ra i))] for every row
   and [sa %~ ra], then
   [s_row_broadcast (fun x s -> div (exp x) s) sums sa %~ row_softmax_real ra]. *)
#push-options "--z3rlimit 60"
let s_row_div_exp_approx_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#m : nat) (#n : nat { n > 0 })
  (sums : lseq et m) (sa : ematrix et m n) (ra : ematrix real m n)
  : Lemma
      (requires
        sa %~ ra /\
        (forall (i : nat). i < m ==>
          v_approximates (sums @! i)
                         (rsum (lseq_map rexp (ematrix_row ra i)))))
      (ensures RB.s_row_broadcast (fun x s -> div (exp x) s) sums sa %~ row_softmax_real #m #n ra)
  =
    ()
#pop-options

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
{
  (* Allocate per-row sums on the GPU. *)
  let sums = Array1.alloc0 #et m (l1_forward m);

  (* Step 1: tree-reduce exp(row) into sums[i]. *)
  KB.reduce_batched_block #et exp rexp m n max_threads a sums ra;

  with sums_v. assert (on gpu_loc (sums |-> sums_v));

  (* Step 2: in-place fused exp(x) / sums[i] over every cell. *)
  RB.row_broadcast (fun x s -> div (exp x) s) m n sums a;

  (* Free the per-row sums temp. *)
  Array1.free sums;

  (* Glue: prove the resulting matrix approximates [row_softmax_real ra]. *)
  s_row_div_exp_approx_softmax #et #_ #_ #_ #(SZ.v m) #(SZ.v n) sums_v sa ra;
  ()
}
