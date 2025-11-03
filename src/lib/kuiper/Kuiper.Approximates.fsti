module Kuiper.Approximates

include Kuiper.Approximates.Class

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

let seq_approximates (#a:Type) {| scalar a, real_like a |}
  (s : seq a) (r : seq Real.real) : prop
  = Seq.length s == Seq.length r /\
    (forall i. i < len s ==> (s @! i) `approximates` (r @! i))

let to_real_seq (#a:Type) {| scalar a, real_like a |}
  (s : seq a) : GTot (seq Real.real)
  = Seq.init_ghost (Seq.length s) (fun i -> to_real (s @! i))

val to_real_seq_is_approx (#a:Type) {| scalar a, real_like a |}
  (s : seq a)
  : Lemma (seq_approximates s (to_real_seq s))
          [SMTPat (seq_approximates s (to_real_seq s))]

let real_seq_sum (r : seq Real.real) : Real.real
  = seq_fold_left (+.) 0.0R r

val real_seq_sum_append
  (r1 r2 : seq Real.real)
  : Lemma (real_seq_sum (r1 `Seq.append` r2) == real_seq_sum r1 +. real_seq_sum r2)

val seq_approximates_append (#a:Type) {| scalar a, real_like a |}
  (s1 s2 : a) (r1 r2 : seq Real.real)
  : Lemma (requires s1 `approximates` real_seq_sum r1 /\ s2 `approximates` real_seq_sum r2)
          (ensures (s1 `add` s2) `approximates` real_seq_sum (r1 `Seq.append` r2))
