module Kuiper.Kernel.Softmax

(* Softmax, in two kernel calls.
   1. Compute sum of exps (does not modify array)
   2. Exp and divide by sum. Note: we are callng exp twice for every value.  An
   alernative would be to first do an exp pass, then reduce, then divide, but I
   think the extra kernel launch would be more expensive than the redundant
   exp's.
   *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Softmax
open Kuiper.Math.OnlineSoftmax { seq_max }

module Max = Kuiper.Kernel.HReduce.Max
module SZ = Kuiper.SizeT

(* Glue for the numerically-stable path: subtracting [m %~ m_r] before
   exponentiating still refines [softmax_real].  We relate each cell to the
   already-well-typed value of [softmax_real] on the shifted chest (so we never
   write a real division ourselves, avoiding the nonzero-divisor obligation). *)
#push-options "--z3rlimit 100 --fuel 1 --ifuel 1 --split_queries always"
let softmax_shift_approx
    (#et:Type0) {| floating et, real_like et, floating_real_like et |}
    (#n:nat) (va:chest1 et n) (ra:chest1 real n { va %~ ra /\ n > 0 })
    (m : et) (m_r : real { m %~ m_r })
    (summ : et { summ %~ chest1_rsum (chest_map (fun z -> exp (z -. m_r)) ra) })
  : Lemma
    (ensures chest_map (fun x -> div (fexp (sub x m)) summ) va %~ softmax_real ra)
  = let rs = chest_map (fun (z:real) -> z -. m_r) ra in
    (* softmax is shift-invariant, so [softmax_real ra == softmax_real rs]. *)
    softmax_shift ra m_r;
    (* the denominator of [softmax_real rs] is exactly what [summ] approximates. *)
    lemma_equal_intro (chest_map exp rs) (chest_map (fun z -> exp (z -. m_r)) ra);
    ext (chest_map exp rs) (chest_map (fun z -> exp (z -. m_r)) ra);
    let aux (i : natlt n)
      : Lemma (acc1 (chest_map (fun x -> div (fexp (sub x m)) summ) va) i
               %~ acc1 (softmax_real ra) i)
      = () in
    Classical.forall_intro aux
#pop-options

(* Clamp the block thread count so it never exceeds the data length: this keeps
   every strided bucket of the max reduction non-empty. *)
inline_for_extraction noextract
let clamp_threads (nth lena : szp)
  : (r : szp { SZ.v r <= SZ.v nth /\ SZ.v r <= SZ.v lena })
  = if nth <=^ lena then nth else lena

inline_for_extraction noextract
fn softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lena : szp)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (ra  : chest1 real lena)
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : chest1 et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ softmax_real ra)
{
  (* Step 1: find the max over the array (does not modify it). Clamp the thread
     count so every strided bucket is non-empty (the max reduction has no
     identity element). *)
  let nthm : szp = clamp_threads nth lena;
  assert pure (0 < SZ.v nthm /\ SZ.v nthm <= lena /\ nthm <= max_threads /\ SZ.fits (lena + nthm));
  let m = Max.reduce_max (fun x -> x) (fun z -> z) nthm lena a ra;
  assert pure (equal (chest_map (fun (z:real) -> z) ra) ra);
  let m_r : erased real = hide (seq_max (chest1_to_seq ra));
  assert pure (m %~ reveal m_r);

  (* Step 2: sum of exp(x - m) (does not modify the array). *)
  assert pure (SZ.fits (lena + nth));
  let sum = Kuiper.Kernel.Reduce.reduce1
              (fun x -> fexp (sub x m)) (fun z -> exp (z -. reveal m_r)) lena nth a ra;

  (* Step 3: write exp(x - m) / sum into every cell. *)
  Kuiper.Kernel.Map.map_gpu (fun x -> div (fexp (sub x m)) sum) lena a;

  softmax_shift_approx va ra m (reveal m_r) sum;
  ()
}
