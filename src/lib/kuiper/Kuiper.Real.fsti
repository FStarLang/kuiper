module Kuiper.Real

open FStar.Real
include FStar.Real

(* The exp function is assumed. F*'s real formalization
does not expose one. *)
val rexp (x:real) : real
val rlog (x:real) : real

val rexp_pos (x:real)
: Lemma (ensures rexp x >. 0.0R)
        [SMTPat (rexp x)]
