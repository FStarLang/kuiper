module Kuiper.Float32.FastMath

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Float32
open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real
open Kuiper.Float.Realops
module F = Kuiper.Floating.Base

(* Checked ideal-value contracts for the native fast-math primitives.
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

noextract
fn log (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real{xr >. 0.0R}).
      v_approximates x xr ==>
      v_approximates result (r_log xr))
(* Device intrinsics, extracted directly without host fallbacks or algebraic
   substitutions. The checked real approximation clauses do not specify
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

(* Ghost-only model of the deterministic GPU __fsub_rn intrinsic:
   https://docs.nvidia.com/cuda/archive/13.0.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fsub_rnff
   Repeated identical inputs give identical results, but no particular NaN
   sign or payload is specified. This does not add a host implementation. *)
val pure_sub_rn (x y : t) : GTot (result : t{
    (forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (r_sub xr yr)) /\
    (match kind x, kind y with
     | NaN, _ | _, NaN -> NaN? (kind result)
     | Infinite, Finite -> bit_eq result x
     | Finite, Infinite -> bit_eq result (F.neg y)
     | Infinite, Infinite ->
         (bit_eq x y ==> NaN? (kind result)) /\
         (~(bit_eq x y) ==> bit_eq result x)
     | Finite, Finite ->
         (x == y ==> result == F.zero) /\
         (bit_eq x y ==> bit_eq result F.zero) /\
         (is_zero x /\ is_zero y /\ ~(bit_eq x y) ==> bit_eq result x))})

noextract
fn sub_rn (x y : t)
  preserves gpu
  returns result : t
  ensures pure (result == pure_sub_rn x y)
