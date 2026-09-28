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

private let log_refinement (x : t)
  : Lemma (
      forall (xr : real{xr >. 0.0R}).
        v_approximates x xr ==>
        v_approximates (Kuiper.Floating.Base.flog x) (Kuiper.Real.log xr))
  = let related (xr : real{xr >. 0.0R /\ v_approximates x xr})
      : Lemma (
          v_approximates (Kuiper.Floating.Base.flog x) (Kuiper.Real.log xr))
      = Kuiper.Approximates.Base.log_approx x xr
    in FStar.Classical.forall_intro related

(* The refinement follows the existing logarithm model. Native extraction to
   __logf is a separate trusted lowering, not bitwise equality with flog.
   The real relation does not specify exceptional values or error bounds. *)
noextract
fn log (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real{xr >. 0.0R}).
      v_approximates x xr ==>
      v_approximates result (Kuiper.Real.log xr))
{
  log_refinement x;
  Kuiper.Floating.Base.flog x
}
