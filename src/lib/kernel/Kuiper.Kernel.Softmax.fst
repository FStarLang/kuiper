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
open Kuiper.Seq.Common
open Kuiper.Array1
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Softmax
open Kuiper.Math.OnlineSoftmax { seq_max }

module Max = Kuiper.Kernel.HReduce.Max
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT

(* ── Shift invariance of softmax over the reals ──────────────────────────────
   Subtracting any constant [c] from every element before exponentiating does
   not change [softmax_real].  This is what makes the numerically-stable
   "subtract the max" implementation refine the unchanged golden spec. *)

(* Glue for the numerically-stable path: subtracting [m %~ m_r] before
   exponentiating still refines [softmax_real]. *)
let softmax_shift_approx
    (#et:Type0) {| floating et, real_like et, floating_real_like et |}
    (s0:seq et) (r0:seq real { s0 %~ r0 /\ Seq.length r0 > 0 })
    (m : et) (m_r : real { m %~ m_r })
    (summ : et { summ %~ rsum (seq_map (fun z -> exp (z -. m_r)) r0) })
  : Lemma
    (ensures seq_map (fun x -> div (fexp (sub x m)) summ) s0 %~ softmax_real r0)
  = let lhs = seq_map (fun x -> div (fexp (sub x m)) summ) s0 in
    let aux (i : natlt (Seq.length s0))
      : Lemma ((lhs @! i) %~ (softmax_real (seq_map (fun z -> z -. m_r) r0) @! i))
    = let x = lhs @! i in
      let shifted_s0 = seq_map (fun z -> z -. m_r) r0 in
      let y = softmax_real shifted_s0 @! i in
      assert (x == div (fexp (sub (s0 @! i) m)) summ);
      assert (y == exp (shifted_s0 @! i) /. rsum (seq_map exp shifted_s0));
      assert (y == exp ((r0 @! i) -. m_r) /. rsum (seq_map exp shifted_s0));
      assert
        seq_map exp shifted_s0
        `Seq.equal`
        seq_map (fun x -> exp (x -. m_r)) r0;
      assert (x %~ y);
      ()
    in
    Classical.forall_intro aux;
    assert (lhs %~ softmax_real (seq_map (fun z -> z -. m_r) r0));
    softmax_shift r0 m_r;
    ()

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
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va: erased (lseq et lena))
  (ra: erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ softmax_real ra)
{
  (* Step 1: find the max over the array (does not modify it). Clamp the thread
     count so every strided bucket is non-empty (the max reduction has no
     identity element). *)
  let nthm : szp = clamp_threads nth lena;
  assert pure (0 < SZ.v nthm /\ SZ.v nthm <= lena /\ nthm <= max_threads /\ SZ.fits (lena + nthm));
  let m = Max.reduce_max (fun x -> x) (fun z -> z) nthm lena a ra;
  Seq.lemma_eq_elim (seq_map (fun (z:real) -> z) (reveal ra)) (reveal ra);
  let m_r : erased real = hide (seq_max (reveal ra));
  assert pure (m %~ reveal m_r);

  (* Step 2: sum of exp(x - m) (does not modify the array). *)
  let sum = Kuiper.Kernel.HReduce.reduce
              (fun x -> fexp (sub x m)) (fun z -> exp (z -. reveal m_r)) nth lena a ra;

  (* Step 3: write exp(x - m) / sum into every cell. *)
  Kuiper.Kernel.Map.map_gpu (fun x -> div (fexp (sub x m)) sum) lena a;

  (* Note: this could also be fused into a single kcall. The comfortable way of
     doing that would require extensible barrier contracts, so we can add one more
     step to the barrier of HReduce. *)

  softmax_shift_approx va ra m (reveal m_r) sum;
  ()
}
