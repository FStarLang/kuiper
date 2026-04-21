module Kuiper.Approximates

open FStar.Real
open Kuiper.Scalars

let to_real_seq_is_approx (#a:Type) {| scalar a, d : real_like a |}
  (s : seq a)
  : Lemma (seq_approximates s (to_real_seq s))
          [SMTPat (seq_approximates s (to_real_seq s))]
  = let aux (x : a) : Lemma (x `approximates` to_real x)
      = d.to_real_ok x
    in
    Classical.forall_intro aux

let real_seq_sum_append
  (r1 r2 : seq real)
  : Lemma (ensures real_seq_sum (r1 `Seq.append` r2) == real_seq_sum r1 +. real_seq_sum r2)
          (decreases Seq.length r1)
  = lemma_seq_fold_left_sum 0.0R (+.) r1 r2

let seq_approximates_append (#a:Type) {| scalar a, real_like a |}
  (s1 s2 : a) (r1 r2 : seq real)
  : Lemma (requires s1 `approximates` real_seq_sum r1 /\ s2 `approximates` real_seq_sum r2)
          (ensures (s1 `add` s2) `approximates` real_seq_sum (r1 `Seq.append` r2))
  = a_add s1 s2 (real_seq_sum r1) (real_seq_sum r2);
    real_seq_sum_append r1 r2

let rec sum_is_approx' #a {| scalar a, real_like a |}
      (s: seq a) (s': seq real) (acc: a) (acc': real) :
    Lemma (requires s %~ s' /\ acc %~ acc')
          (ensures seq_fold_left add acc s %~ seq_fold_left (+.) acc' s')
          (decreases Seq.length s) =
  match view_seq s, view_seq s' with
  | SNil, SNil -> ()
  | SCons hd tl, SCons hd' tl' ->
    a_add acc hd acc' hd';
    sum_is_approx' #a tl tl' (add acc hd) (acc' +. hd')

let sum_is_approx #a {| scalar a, real_like a |} (s: seq a) (s': seq real) :
    Lemma (requires s %~ s')
          (ensures seq_fold_left add zero s %~ seq_fold_left (+.) 0.0R s') =
  sum_is_approx' s s' zero 0.0R
