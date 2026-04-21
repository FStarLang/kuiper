module Kuiper.Math.Relu

#lang-pulse
open Kuiper

let relu
  (#et:Type) {| scalar et |}
  (x : et)
  : et
  = if x `gt` zero then x else zero

(* must redefine since real comparisons are admitted. *)
let relu_real
  (x : real)
  : real
  = if t2b (x >. zero) then x else zero

let relu_lem
  (#et:Type) {| scalar et, real_like et |}
  (x : et) (r : real)
  : Lemma (requires x %~ r)
          (ensures relu x %~ relu_real r)
          [SMTPat (relu x); SMTPat (x %~ r)]
  = admit() (* intentional, for now, see below. *)


(* One attempt at proving the lemma above was to extend
Kuiper.Approximates with this notion for booleans approximating
props:

val bool_approx_prop (b : bool) (p : prop) : prop

val bool_approx_gt
  (#et:Type) {| scalar et, real_like et |}
  (x y : et) (r s : real{x %~ r /\ y %~ s})
  : Lemma (bool_approx_prop (x `gt` y) (r >. s))

val approx_if
  (#et:Type) {| scalar et, real_like et |}
  (b : bool) (p : prop{bool_approx_prop b p})
  (x y : et) (r s : real{x %~ r /\ y %~ s})
  : Lemma ((if b then x else y) %~ (if t2b p then r else s))

let relu_lem
  (#et:Type) {| scalar et, real_like et |}
  (x : et) (r : real)
  : Lemma (requires x %~ r)
          (ensures relu x %~ relu_real r)
          [SMTPat (relu x); SMTPat (x %~ r)]
  = assert (x %~ r);
    assert (zero #et %~ 0.0R);
    bool_approx_gt x zero r 0.0R;
    approx_if
      (x `gt` zero) (r >. 0.0R)
       x zero        r 0.0R;
    ()

But I think this makes it trivial to blow up the bool_approx_prop
into the universal relation, and therefore do the same to %~. *)
