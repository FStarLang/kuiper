module Kuiper.Float32

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real
open Kuiper.Float.Realops
module Sqrt = FStar.Math.Sqrt

inline_for_extraction noextract
val t : Type0

inline_for_extraction noextract
instance val is_floating : floating t

instance val is_real_like : real_like t
instance val is_floating_real_like : floating_real_like t

(* Exact comparison laws inherited from the trusted native binary32 interface. *)
val lt_ordered (x y : t)
  : Lemma
      (requires lt x y)
      (ensures ~(NaN? (kind x)) /\ ~(NaN? (kind y)))

val lt_transitive (x y z : t)
  : Lemma
      (requires lt x y /\ lt y z)
      (ensures lt x z)

inline_for_extraction noextract
val fexpm1 : t -> t
assume Fexpm1Models : fexpm1 %~ r_expm1

inline_for_extraction noextract
val flog1p : t -> t
assume Flog1pModels : flog1p %~ r_log1p

(* GPU-only operations with explicit rounding and flush-to-zero behavior.
   As for the other floating operations, the trusted real approximation
   contracts do not model rounding, FTZ, or numerical error bounds. *)
noextract
fn add_rn_ftz (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (r_add xr yr))

noextract
fn fma_rn_ftz (x y z : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr zr : real).
      v_approximates x xr /\ v_approximates y yr /\ v_approximates z zr ==>
      v_approximates result (r_fma xr yr zr))

noextract
fn mul_rn_ftz (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (r_mul xr yr))

noextract
fn exp2_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (r_exp2 xr))

noextract
fn rcp_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real{xr =!= 0.0R}).
      v_approximates x xr ==>
      v_approximates result (r_rcp xr))

noextract
fn rsqrt_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : Sqrt.rpos).
      v_approximates x xr ==>
      v_approximates result (r_rsqrt xr))

val lem_sizeof () : Lemma (Sized.size #t == 4sz)
