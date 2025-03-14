module PtsToWildcard

#lang-pulse
open Pulse
open Pulse.Lib.Box

(* Would be nice if this worked instead
of needing random names for x/y. But, they
should be two different binders, instead of a
single binder with name __. *)
[@@expect_failure]
fn test (x y : box int)
  requires (x |-> __) ** (y |-> __)
  ensures  emp
{
  let vx = !x;
  let vy = !y;
  // assert (pure (vx == vy));
  free x;
  free y;
  ();
}
