module Kuiper.Float32

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real
module Pow = FStar.Math.Pow

inline_for_extraction noextract
val t : Type0

inline_for_extraction noextract
instance val is_floating : floating t

instance val is_real_like : real_like t
instance val is_floating_real_like : floating_real_like t

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

(* GPU-only operations with explicit rounding and flush-to-zero behavior.
   As for the other floating operations, the trusted real approximation
   contracts do not model rounding, FTZ, or numerical error bounds. *)
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

val lem_sizeof () : Lemma (Sized.size #t == 4sz)
