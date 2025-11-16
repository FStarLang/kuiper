module Kuiper.For

#lang-pulse

open Kuiper
open Kuiper.ForEvery
module SZ = Kuiper.SizeT

let between (lo hi : nat) : Type =
  x:nat{lo <= x /\ x < hi}

(* A sequential for loop, but whose verification is
   morally done in parallel over a forall+. *)
fn for_loop (lo hi : SZ.t)
  (pre post : between lo hi -> slprop)
  (f : (x:SZ.t{lo <= SZ.v x /\ SZ.v x < hi}) ->
          stt unit
            (requires (pre (SZ.v x)))
            (ensures (fun _ -> post (SZ.v x))))
  requires pure (lo <= hi)
  requires forall+ (x : between lo hi). pre x
  ensures  forall+ (x : between lo hi). post x

