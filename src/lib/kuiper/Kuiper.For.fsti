module Kuiper.For

#lang-pulse

open Kuiper
open Kuiper.ForEvery
module SZ = Kuiper.SizeT

(* A sequential for loop, but whose verification is
   morally done in parallel over a forall+. *)
fn for_loop (lo hi : SZ.t)
  (pre post : between lo hi -> slprop)
  (fn f (x:SZ.t{lo <= x /\ x < hi})
       requires pre (SZ.v x)
       ensures  post (SZ.v x))
  requires pure (lo <= hi)
  requires forall+ (x : between lo hi). pre x
  ensures  forall+ (x : between lo hi). post x

(* Similar, but with an extra frame in pre/post *)
fn for_loop' (lo hi : SZ.t)
  (pre post : between lo hi -> slprop)
  (frame : slprop)
  (fn f (x:SZ.t{lo <= x /\ x < hi})
       requires frame ** pre (SZ.v x)
       ensures  frame ** post (SZ.v x))
  requires pure (lo <= hi)
  preserves frame
  requires forall+ (x : between lo hi). pre x
  ensures  forall+ (x : between lo hi). post x

