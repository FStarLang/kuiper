module Kuiper.Float32.FastMath

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Float32
open Kuiper.Approximates.Base
open Kuiper.Real
module Trig = Kuiper.Real.Trigonometry

(* New trusted approximation contracts, not derived generic-trig refinements.
   Extraction selects __sinf/__cosf; no IEEE bit or error-bound claim is made. *)
noextract
fn sin (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (Trig.sin xr))

noextract
fn cos (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (Trig.cos xr))
