module Kuiper.Approximates

include Kuiper.Approximates.Base
include Kuiper.Approximates.U8
include Kuiper.Approximates.U16
include Kuiper.Approximates.U32
include Kuiper.Approximates.U64

open FStar.Ghost
open FStar.Seq
open Pulse
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Seq.Common
open Kuiper.Len

(* This class provides some syntactic sugar to use the %~ operator
   on scalars, sequences, matrices, etc. *)
[@@Tactics.Typeclasses.fundeps[1]] // OK?
class can_approximate (c m : Type) = {
  approximates : c -> m -> prop;
}

instance erased_can_approximate_lhs (c m : Type)
  {| _: can_approximate c m |}
  : can_approximate (erased c) m = {
  approximates = (fun (x: erased c) (y: m) -> approximates (reveal x) y);
}

instance erased_can_approximate_rhs (c m : Type)
  {| _: can_approximate c m |}
  : can_approximate c (erased m) = {
  approximates = (fun (x: c) (y: erased m) -> approximates x (reveal y));
}

unfold let (%~) #c #m (x:c) (y:m) {| can_approximate c m |}
  : prop = approximates x y

let pts_to_approx_via #pt #rt #mt {| has_pts_to pt rt, can_approximate rt mt |}
  (p : pt) (#[full_default()] f:perm) (v : rt) (m : mt)
=
  p |-> v ** pure (v %~ m)

(* "Approximated" points-to. Inference does not really work to make this useful,
so we use pts_to_approx_via instead, but it would be really nice. *)
let ( |~> ) #pt #rt #mt {| has_pts_to pt rt, can_approximate rt mt |}
  (p : pt) (#[full_default()] f:perm) (m : mt)
  : slprop =
  exists* (v : rt). pts_to_approx_via p #f v m

instance real_like_can_approximate (#a:Type) (_ : scalar a) (_ : real_like a)
  : can_approximate a real = {
  approximates = v_approximates;
}

let seq_approximates (#a:Type) {| scalar a, real_like a |}
  (s : seq a) (r : seq real) : prop
  = Seq.length s == Seq.length r /\
    (forall i. i < len s ==> (s @! i) %~ (r @! i))

instance seq_real_like_can_approximate (#a:Type) {| scalar a, real_like a |}
  : can_approximate (seq a) (seq real) = {
  approximates = seq_approximates;
}

instance lseq_lhs (#a #b : Type)
  (d : can_approximate (seq a) b)
  (len : erased nat)
  : can_approximate (lseq a len) b = {
  approximates = (fun (s: lseq a len) (m: b) -> approximates (s <: seq a) m);
}

instance lseq_rhs (#a #b : Type)
  (d : can_approximate a (seq b))
  (len : erased nat)
  : can_approximate a (lseq b len) = {
  approximates = (fun (s: a) (m: lseq b len) -> approximates s (m <: seq b));
}

let to_real_seq (#a:Type) {| scalar a, real_like a |}
  (s : seq a) : GTot (seq real)
  = Seq.init_ghost (Seq.length s) (fun i -> to_real (s @! i))

val to_real_seq_is_approx (#a:Type) {| scalar a, real_like a |}
  (s : seq a)
  : Lemma (s %~ to_real_seq s)
          [SMTPat (seq_approximates s (to_real_seq s))]

val rsum_append
  (r1 r2 : seq real)
  : Lemma (rsum (r1 `Seq.append` r2) == rsum r1 +. rsum r2)

val seq_approximates_append (#a:Type) {| scalar a, real_like a |}
  (s1 s2 : a) (r1 r2 : seq real)
  : Lemma (requires s1 %~ rsum r1 /\ s2 %~ rsum r2)
          (ensures (s1 `add` s2) %~ rsum (r1 `Seq.append` r2))

val sum_is_approx' #a {| scalar a, real_like a |}
      (s: seq a) (s': seq real) (acc: a) (acc': real) :
    Lemma (requires s %~ s' /\ acc %~ acc')
          (ensures seq_fold_left add acc s %~ seq_fold_left (+.) acc' s')

val sum_is_approx #a {| scalar a, real_like a |} (s: seq a) (s': seq real) :
    Lemma (requires s %~ s')
          (ensures seq_fold_left add zero s %~ seq_fold_left (+.) 0.0R s')

let approx2
  (#a #b #c : Type)
  {| scalar a, real_like a,
     scalar b, real_like b,
     scalar c, real_like c |}
  (f : a -> b -> c)
  (g : real -> real -> real)
  : prop
  = forall x y r s.
      x %~ r /\ y %~ s ==> f x y %~ g r s

(* Could we use this instead of approx2? *)
// instance approx_function_can_approximate
//   (dom1 dom2 cod1 cod2 : Type)
//   {| can_approximate dom1 dom2, can_approximate cod1 cod2 |}
//   : can_approximate (dom1 -> cod1) (dom2 -> cod2) = {
//   approximates = (fun f g -> forall x y. x %~ y ==> f x %~ g y);
// }
