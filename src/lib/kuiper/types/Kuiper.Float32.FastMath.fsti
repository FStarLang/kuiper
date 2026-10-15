module Kuiper.Float32.FastMath

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Float32
open Kuiper.Approximates.Base
open Kuiper.Real
open Kuiper.Float.Realops

(* New trusted approximation contracts, not derived generic-trig refinements.
   Extraction selects __sinf/__cosf; no IEEE bit or error-bound claim is made. *)
noextract
fn sin (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (r_sin xr))

noextract
fn cos (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (r_cos xr))

private let log_refinement (x : t)
  : Lemma (
      forall (xr : real{xr >. 0.0R}).
        v_approximates x xr ==>
        v_approximates (Kuiper.Floating.Base.flog x) (r_log xr))
  = let related (xr : real{xr >. 0.0R /\ v_approximates x xr})
      : Lemma (
          v_approximates (Kuiper.Floating.Base.flog x) (r_log xr))
      = let _ = approx_flog #t in ()
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
      v_approximates result (r_log xr))
{
  log_refinement x;
  Kuiper.Floating.Base.flog x
}

(* Trusted device intrinsics, extracted directly without host fallbacks or
   algebraic substitutions. These real approximation contracts do not specify
   IEEE bits, exceptional values, or numerical error bounds. *)
noextract
fn exp (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (r_exp xr))

noextract
fn divide (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real) (yr : real{yr =!= 0.0R}).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (r_div xr yr))

noextract
fn fma_rn (x y z : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr zr : real).
      v_approximates x xr /\ v_approximates y yr /\ v_approximates z zr ==>
      v_approximates result (r_fma xr yr zr))

noextract
fn sub_rn (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (r_sub xr yr))
