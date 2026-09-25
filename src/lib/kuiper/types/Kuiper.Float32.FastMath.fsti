module Kuiper.Float32.FastMath

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Float32
open Kuiper.Approximates.Base
open Kuiper.Real

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
      v_approximates result (Kuiper.Real.exp xr))

noextract
fn divide (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real) (yr : real{yr =!= 0.0R}).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (xr /. yr))

noextract
fn fma_rn (x y z : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr zr : real).
      v_approximates x xr /\ v_approximates y yr /\ v_approximates z zr ==>
      v_approximates result (xr *. yr +. zr))

noextract
fn sub_rn (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (xr -. yr))

inline_for_extraction let () = ()
