module Kuiper.Real

open FStar.Real
include FStar.Real { of_int }

(* The exp function is assumed. F*'s real formalization
does not expose one. *)
val rexp (x:real) : real

val rexp_pos (x:real)
: Lemma (ensures rexp x >. 0.0R)
        [SMTPat (rexp x)]

// assume
// val rexp_approx #et {| floating et, real_like et |}
// : Lemma (forall (s:et) (r:real { s %~ r }). exp s %~ rexp r)

// assume
// val div_approx #et {| floating et, real_like et |}
// : Lemma (forall (x y:et) (rx ry:real). x%~rx /\ y%~ry /\ ry=!=0.0R ==> (x `div` y) %~ (rx /. ry))
