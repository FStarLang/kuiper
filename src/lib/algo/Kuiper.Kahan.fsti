module Kuiper.Kahan

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Sum { sum }

inline_for_extraction noextract
fn kahan_sum
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (len : sz)
  (frame : slprop)
  (vf : natlt len -> real) (* spec function *)
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
