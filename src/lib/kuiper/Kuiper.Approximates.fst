module Kuiper.Approximates

open Kuiper
open FStar.Real
open Kuiper.Scalars

let real_seq_sum_append
  (r1 r2 : seq Real.real)
  : Lemma (ensures real_seq_sum (r1 `Seq.append` r2) == real_seq_sum r1 +. real_seq_sum r2)
          (decreases Seq.length r1)
  = lemma_seq_fold_left_sum 0.0R (+.) r1 r2

let seq_approximates_append (#a:Type) {| scalar a, real_like a |}
  (s1 s2 : a) (r1 r2 : seq Real.real)
  : Lemma (requires s1 `approximates` real_seq_sum r1 /\ s2 `approximates` real_seq_sum r2)
          (ensures (s1 `add` s2) `approximates` real_seq_sum (r1 `Seq.append` r2))
  = a_add s1 s2 (real_seq_sum r1) (real_seq_sum r2);
    real_seq_sum_append r1 r2
