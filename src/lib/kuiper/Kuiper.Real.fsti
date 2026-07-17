module Kuiper.Real

open FStar.Real
include FStar.Real
include FStar.Math.Exp
open Kuiper.Seq.Common
open Kuiper.Common

let rsum (s : Seq.seq real) : real = seq_fold_left (+.) 0.0R s

val rsum_append (s1 s2 : Seq.seq real)
  : Lemma (ensures rsum (s1 @+ s2) == rsum s1 +. rsum s2)
          [SMTPat (rsum (s1 @+ s2))]

let rmax (x y: real) : real =
  if t2b (x >. y) then x else y

val lem_rmax_comm (x: real) (y: real)
  : Lemma (ensures rmax x y == rmax y x)

val lem_rmax_assoc (x: real) (y: real) (z: real)
  : Lemma (ensures rmax x (rmax y z) == rmax (rmax x y) z)

val sum_non_zero
    (s : Seq.seq real { forall (i:natlt (Seq.length s)). Seq.index s i >. 0.0R })
    (acc : real)
  : Lemma (requires Seq.length s > 0)
          (ensures seq_fold_left (+.) acc s >. acc)
          [SMTPat (seq_fold_left (+.) acc s)]
