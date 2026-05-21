module Kuiper.Real

open FStar.Real
include FStar.Real
open Kuiper.Seq.Common
open Kuiper.Common

(* The exp function is assumed. F*'s real formalization
does not expose one. *)
val rexp (x:real) : real
val rlog (x:real{x >. 0.0R}) : real

(* Usual math laws about exponentiation and logarithms. *)

val rexp_positive (x : real)
  : Lemma (ensures rexp x >. 0.0R)
          [SMTPat (rexp x)]

val rexp_base ()
  : Lemma (rexp 0.0R == 1.0R)

val exp_add (x y : real)
  : Lemma (ensures rexp (x +. y) == rexp x *. rexp y)
          [SMTPat (rexp (x +. y))]

val log_exp (x : real)
  : Lemma (ensures rlog (rexp x) == x)
          [SMTPat (rlog (rexp x))]

val exp_log (x : real{x >. 0.0R})
  : Lemma (ensures rexp (rlog x) == x)
          [SMTPat (rexp (rlog x))]

val exp_sub (x y : real)
  : Lemma (ensures rexp (x -. y) == rexp x /. rexp y)
          [SMTPat (rexp (x -. y))]

val log_mul (x y : real{x >. 0.0R /\ y >. 0.0R})
  : Lemma (ensures rlog (x *. y) == rlog x +. rlog y)
          [SMTPat (rlog (x *. y))]

val log_div (x y : real{x >. 0.0R /\ y >. 0.0R})
  : Lemma (ensures rlog (x /. y) == rlog x -. rlog y)
          [SMTPat (rlog (x /. y))]

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
