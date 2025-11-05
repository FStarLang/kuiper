module Kuiper.RealExpDiv
open Kuiper.Scalars
open Kuiper.Approximates
open FStar.Real
assume
val rexp (x:real) : real

assume
val rexp_pos (x:real)
: Lemma (ensures rexp x >. 0.0R)
        [SMTPat (rexp x)]

assume
val rexp_approx #et {| floating et, real_like et |}
: Lemma (forall (s:et) (r:real { s %~ r }). exp s %~ rexp r)

assume
val div_approx #et {| floating et, real_like et |}
: Lemma (forall (x y:et) (rx ry:real). x%~rx /\ y%~ry /\ ry=!=0.0R ==> (x `div` y) %~ (rx /. ry))
