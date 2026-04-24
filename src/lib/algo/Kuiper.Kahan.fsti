module Kuiper.Kahan

#lang-pulse

open Kuiper
open Kuiper.Approximates

let rec sum
  (#et:Type) {| scalar et |}
  (from to : nat)
  (f : (x:nat{from <= x /\ x < to}) -> GTot et)
  : GTot et (decreases to-from)
  // = if from < to then f from `add` sum (from + 1) to f else zero
  = if from < to then sum from (to-1) f `add` f (to-1) else zero

inline_for_extraction noextract
fn kahan_sum
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (len : sz)
  (frame : slprop)
  (vf : natlt len -> GTot real) (* spec function *)
  (f : fn (i:szlt len)
         preserves frame
         returns   r : et
         ensures   pure (r %~ vf i))
  preserves
    frame
  returns
    res : et
  ensures
    pure (res %~ sum 0 len vf)
