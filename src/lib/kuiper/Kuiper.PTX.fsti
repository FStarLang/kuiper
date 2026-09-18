module Kuiper.PTX

open Kuiper

(* Thread-local PTX instructions. Each operation consumes scalar register
   values and produces a scalar register value without accessing memory. *)

inline_for_extraction noextract
let exp2_real (x : real) : real =
  exp (log 2.0R *. x)

val mul_rn_ftz_f32
  (x y : f32)
  : f32

val mul_rn_ftz_f32_approx
  (x y : f32)
  (xr yr : real)
  : Lemma
      (requires v_approximates x xr /\ v_approximates y yr)
      (ensures v_approximates (mul_rn_ftz_f32 x y) (xr *. yr))

val ex2_approx_ftz_f32
  (x : f32)
  : f32

val ex2_approx_ftz_f32_approx
  (x : f32)
  (xr : real)
  : Lemma
      (requires v_approximates x xr)
      (ensures v_approximates (ex2_approx_ftz_f32 x) (exp2_real xr))
