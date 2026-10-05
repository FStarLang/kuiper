module Kuiper.Float32

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real
module Pow = FStar.Math.Pow
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

inline_for_extraction noextract
val flog1p : t -> t

val expm1_approx
  (x : t)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (fexpm1 x) (exp r -. 1.0R))

val log1p_approx
  (x : t)
  (r : real { r >. 0.0R -. 1.0R })
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (flog1p x) (log (1.0R +. r)))

(* Trusted output approximation for softplus with beta=1 and threshold=20,
   using ordinary expf/log1pf. This does not assert agreement of the native
   and real branch decisions, an error bound, or exceptional-value laws. *)
val softplus20_approx
  (x : t)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates
        (if lt (Kuiper.Floating.Base.of_int 20L) x then x else flog1p (fexp x))
        (if Kuiper.ForEvery.t2b (r >. 20.0R) then r
         else (exp_positive r; log (1.0R +. exp r))))

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
      v_approximates result (xr +. yr))

noextract
fn fma_rn_ftz (x y z : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr zr : real).
      v_approximates x xr /\ v_approximates y yr /\ v_approximates z zr ==>
      v_approximates result (xr *. yr +. zr))

noextract
fn mul_rn_ftz (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (xr *. yr))

noextract
fn exp2_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (Pow.exp2 xr))

noextract
fn rcp_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real{xr =!= 0.0R}).
      v_approximates x xr ==>
      v_approximates result (1.0R /. xr))

noextract
fn rsqrt_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : Sqrt.rpos).
      v_approximates x xr ==>
      v_approximates result (1.0R /. Sqrt.sqrt xr))

val lem_sizeof () : Lemma (Sized.size #t == 4sz)
