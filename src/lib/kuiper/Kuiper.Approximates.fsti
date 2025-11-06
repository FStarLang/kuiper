module Kuiper.Approximates

include Kuiper.Approximates.Base

include Kuiper.Approximates.U8
include Kuiper.Approximates.U16
include Kuiper.Approximates.U32
include Kuiper.Approximates.U64
include Kuiper.Approximates.F16
include Kuiper.Approximates.F32
include Kuiper.Approximates.F64

open Kuiper
open FStar.Real
open Kuiper.Scalars
open Kuiper.Seq.Common

(* This class provides some syntactic sugar to use the %~ operator
   on scalars, sequences, matrices, etc. *)
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

(* "Approximated" points-to. Inference does not really work to make this useful,
but it would be really nice. *)
let ( |~> ) #pt #rt #mt {| has_pts_to pt rt, can_approximate rt mt |}
  (p : pt) (#[full_default()] f:perm) (m : mt)
  : slprop =
  exists* (v : rt). p |-> v ** pure (v %~ m)

instance real_like_can_approximate (#a:Type) (_ : scalar a) (_ : real_like a)
  : can_approximate a Real.real = {
  approximates = v_approximates;
}

let seq_approximates (#a:Type) {| scalar a, real_like a |}
  (s : seq a) (r : seq Real.real) : prop
  = Seq.length s == Seq.length r /\
    (forall i. i < len s ==> (s @! i) %~ (r @! i))

instance seq_real_like_can_approximate (#a:Type) {| scalar a, real_like a |}
  : can_approximate (seq a) (seq Real.real) = {
  approximates = seq_approximates;
}

let to_real_seq (#a:Type) {| scalar a, real_like a |}
  (s : seq a) : GTot (seq Real.real)
  = Seq.init_ghost (Seq.length s) (fun i -> to_real (s @! i))

val to_real_seq_is_approx (#a:Type) {| scalar a, real_like a |}
  (s : seq a)
  : Lemma (s %~ to_real_seq s)
          [SMTPat (seq_approximates s (to_real_seq s))]

let real_seq_sum (r : seq Real.real) : Real.real
  = seq_fold_left (+.) 0.0R r

val real_seq_sum_append
  (r1 r2 : seq Real.real)
  : Lemma (real_seq_sum (r1 `Seq.append` r2) == real_seq_sum r1 +. real_seq_sum r2)

val seq_approximates_append (#a:Type) {| scalar a, real_like a |}
  (s1 s2 : a) (r1 r2 : seq Real.real)
  : Lemma (requires s1 %~ real_seq_sum r1 /\ s2 %~ real_seq_sum r2)
          (ensures (s1 `add` s2) %~ real_seq_sum (r1 `Seq.append` r2))
