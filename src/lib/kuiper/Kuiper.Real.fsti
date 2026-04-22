module Kuiper.Real

open FStar.Real
include FStar.Real

(* The exp function is assumed. F*'s real formalization
does not expose one. *)
val rexp (x:real) : real
val rlog (x:real{x >. 0.0R}) : real

val rexp_pos (x:real)
: Lemma (ensures rexp x >. 0.0R)
        [SMTPat (rexp x)]

(* Usual math laws about exponentiation and logarithms.  We don't prove these
because we can't--- F* reals are very opaque. Though we could reduce the
number of axioms a bit, and upstream it. *)
val log_exp (x : real)
  : Lemma (ensures rlog (rexp x) == x)
          [SMTPat (rlog (rexp x))]

val exp_log (x : real{x >. 0.0R})
  : Lemma (ensures rexp (rlog x) == x)
          [SMTPat (rexp (rlog x))]

val exp_add (x y : real)
  : Lemma (ensures rexp (x +. y) == rexp x *. rexp y)
          [SMTPat (rexp (x +. y))]

val exp_sub (x y : real)
  : Lemma (ensures rexp (x -. y) == rexp x /. rexp y)
          [SMTPat (rexp (x -. y))]

val log_mul (x y : real{x >. 0.0R /\ y >. 0.0R})
  : Lemma (ensures rlog (x *. y) == rlog x +. rlog y)
          [SMTPat (rlog (x *. y))]

val log_div (x y : real{x >. 0.0R /\ y >. 0.0R})
  : Lemma (ensures rlog (x /. y) == rlog x -. rlog y)
          [SMTPat (rlog (x /. y))]
