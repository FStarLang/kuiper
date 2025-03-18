module SizeTMul

#lang-pulse
open Pulse.Nolib
open FStar.SizeT
module SZ = FStar.SizeT
open Kuiper.ForEvery

let lt x y = y < x

let natbref (p : int -> bool) = x:nat{p x}

[@@pulse_unfold]
let natlt (x:nat) = natbref (lt x)

let natlt_eq_lem (x y : nat)
  : Lemma (requires x == y) (ensures natlt x == natlt y)
  [SMTPat (natlt x); SMTPat (natlt y)]
  = ()

let f1 (s1 s2 : SZ.t)
  (_ : squash (SZ.fits (SZ.v s1 * SZ.v s2)))
  (x : natlt (SZ.v s1 * SZ.v s2))
  : natlt (SZ.v (s1 *^ s2))
  = x

let f2 (s1 s2 : SZ.t)
  (_ : squash (SZ.fits (SZ.v s1 * SZ.v s2)))
  (x : natlt (SZ.v (s1 *^ s2)))
  : natlt (SZ.v s1 * SZ.v s2)
  = x

let eq_types
  (s1 s2 : SZ.t)
  (_ : squash (SZ.fits (SZ.v s1 * SZ.v s2)))
  = assert (natlt (SZ.v s1 * SZ.v s2) == natlt (SZ.v (s1 *^ s2)))

assume val p : int -> slprop

let eq_foralls0 (s : nat)
= assert (
    forevery (natlt (1+s)) p
    ==
    forevery (natlt (s+1)) p
  )

let eq_foralls1 (s : nat)
= assert (
    forevery (natlt (1+s)) p
    ==
    forevery (natlt (s+1)) p
  )

(* It would be really nice if this worked. *)
[@@expect_failure]
let eq_foralls2 (s : nat)
= assert (
    (forall+ (x : natlt (1+s)). p x)
    ==
    (forall+ (x : natlt (s+1)).  p x)
  )

[@@expect_failure]
let eq_foralls3 (s1 s2 : SZ.t)
  (_ : squash (SZ.fits (SZ.v s1 * SZ.v s2)))
= natlt_eq_lem (SZ.v s1 * SZ.v s2) (SZ.v (s1 *^ s2));
  assert (
    (forall+ (x : natlt (SZ.v s1 * SZ.v s2)). p x)
    ==
    (forall+ (x : natlt (SZ.v (s1 *^ s2))).  p x)
  )

[@@expect_failure]
fn f3 (s1 s2 : SZ.t)
  (_ : squash (SZ.fits (SZ.v s1 * SZ.v s2)))
  requires
    forall+ (x : natlt (SZ.v s1 * SZ.v s2)).
      p x
  ensures
    forall+ (x : natlt (SZ.v (s1 *^ s2))).
      p x
{
  ()
}
