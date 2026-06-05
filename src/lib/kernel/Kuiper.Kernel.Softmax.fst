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
module Vec = Pulse.Lib.Vec
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
open Kuiper.Array1

open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Softmax
module Max = Kuiper.Kernel.HReduce.Max
open Kuiper.Math.OnlineSoftmax { seq_max }

(* ── Shift invariance of softmax over the reals ──────────────────────────────
   Subtracting any constant [c] from every element before exponentiating does
   not change [softmax_real].  This is what makes the numerically-stable
   "subtract the max" implementation refine the unchanged golden spec. *)

(* [seq_fold_left (+.)] commutes with pointwise division by a constant, scaling
   both the elements and the initial accumulator.  Generalizing over [acc] is
   what makes the recursion go through (the tail fold carries [acc +. hd], not
   [0.0R]).  Mirrors [lemma_seq_fold_left_distrib_mul] in Kuiper.Math.OnlineSoftmax. *)
let rec fold_div_scale (acc : real) (k : real { k =!= 0.0R }) (s : Seq.seq real)
  : Lemma (ensures seq_fold_left (+.) (acc /. k) (seq_map (fun (e:real) -> e /. k) s)
                   == seq_fold_left (+.) acc s /. k)
          (decreases Seq.length s)
  = let f : real -> real = fun (e:real) -> e /. k in
    let s_mapped = seq_map f s in
    match view_seq s with
    | SNil ->
      assert (Seq.equal s Seq.empty);
      assert (Seq.equal s_mapped Seq.empty)
    | SCons hd tl ->
      assert (Seq.equal s_mapped (Seq.cons (hd /. k) (seq_map f tl)));
      calc (==) {
        seq_fold_left (+.) (acc /. k) s_mapped;
        == { }
        seq_fold_left (+.) ((acc /. k) +. (hd /. k)) (seq_map f tl);
        == { }
        seq_fold_left (+.) ((acc +. hd) /. k) (seq_map f tl);
        == { fold_div_scale (acc +. hd) k tl }
        seq_fold_left (+.) (acc +. hd) tl /. k;
        == { }
        seq_fold_left (+.) acc s /. k;
      }

let rsum_div_scale (s : Seq.seq real) (k : real { k =!= 0.0R })
  : Lemma (ensures rsum (seq_map (fun w -> w /. k) s) == rsum s /. k)
  = fold_div_scale 0.0R k s

(* The shifted exps are the unshifted exps divided by [rexp c]. *)
let shift_denom (r0 : Seq.seq real) (c : real)
  : Lemma (rsum (seq_map (fun z -> rexp (z -. c)) r0)
           == rsum (seq_map rexp r0) /. rexp c)
  = assert (Seq.equal (seq_map (fun z -> rexp (z -. c)) r0)
                      (seq_map (fun w -> w /. rexp c) (seq_map rexp r0)));
    rsum_div_scale (seq_map rexp r0) (rexp c)

(* The pointwise softmax value is unchanged by the shift. *)
let softmax_shift (r0 : Seq.seq real { Seq.length r0 > 0 }) (c : real)
  : Lemma
    (ensures seq_map (fun x -> rexp (x -. c)
                               /. rsum (seq_map (fun z -> rexp (z -. c)) r0)) r0
             == softmax_real r0)
  = let exps = seq_map rexp r0 in
    sum_non_zero exps 0.0R;
    shift_denom r0 c;
    let lhs = seq_map (fun x -> rexp (x -. c)
                               /. rsum (seq_map (fun z -> rexp (z -. c)) r0)) r0 in
    let rhs = softmax_real r0 in
    let aux (i : nat { i < Seq.length r0 }) : Lemma (lhs @! i == rhs @! i) =
      () in
    Classical.forall_intro aux;
    assert (Seq.equal lhs rhs)

(* Glue for the numerically-stable path: subtracting [m %~ m_r] before
   exponentiating still refines [softmax_real]. *)
let softmax_shift_approx
    (#et:Type0) {| floating et, real_like et, floating_real_like et |}
    (s0:seq et) (r0:seq real { s0 %~ r0 /\ Seq.length r0 > 0 })
    (m : et) (m_r : real { m %~ m_r })
    (summ : et { summ %~ rsum (seq_map (fun z -> rexp (z -. m_r)) r0) })
  : Lemma
    (ensures seq_map (fun x -> div (exp (sub x m)) summ) s0 %~ softmax_real r0)
  = let sexps = seq_map (fun z -> rexp (z -. m_r)) r0 in
    sum_non_zero sexps 0.0R;
    shift_denom r0 m_r;
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
              (fun x -> exp (sub x m)) (fun z -> rexp (z -. reveal m_r)) nth lena a ra;

  (* Step 3: write exp(x - m) / sum into every cell. *)
  Kuiper.Kernel.Map.map_gpu (fun x -> div (exp (sub x m)) sum) lena a;

  (* Note: this could also be fused into a single kcall. The comfortable way of
     doing that would require extensible barrier contracts, so we can add one more
     step to the barrier of HReduce. *)

  softmax_shift_approx va ra m (reveal m_r) sum;
  ()
}

inline_for_extraction noextract
fn softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lena : szp)
  (a : Vec.lvec et lena)
  (#va : erased (lseq et lena))
  (ra  : erased (lseq real lena))
  preserves
    cpu
  requires
    a |-> va **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : lseq et lena).
      a |-> va' **
      pure (va' %~ softmax_real ra)
{
  let ga = Array1.alloc0 #et lena (l1_forward lena);
  Array1.memcpy_host_to_device ga a lena;
  softmax_gpu nth ga ra;
  Array1.memcpy_device_to_host' a 0sz ga 0sz lena;
  Array1.free ga;
  ()
}
