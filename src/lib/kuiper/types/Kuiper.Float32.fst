module Kuiper.Float32

open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Canonical
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real

open Kuiper.Float32.Base

let t = Float32.Base.t

inline_for_extraction noextract
instance is_sized : sized t = { size = 4sz; default = zero }

assume CanonicalSized : canonical is_sized

inline_for_extraction noextract
instance _ : scalar t = {
  is_sized = solve;
  add; mul; zero; one; lt; lte; eq;
}

inline_for_extraction noextract
instance is_floating : floating t = {
  is_scalar = solve;
  sub; div; bit_eq;
  of_int; of_literal; of_int_zero; of_int_one;
  kind; is_zero;
  largest; infinity;
  kind_one; kind_zero; kind_largest; kind_infinity;
  zero_is_zero; one_is_nonzero;
  fexp; flog; sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
  sinh; cosh; tanh; ceil; floor; round; fabs; erf; log2;
  log10; exp2; pow; atan2; fmin; fmax; fmod; copysign;
  fma;
}

(* Approximation semantics is assumed. *)
instance is_real_like          : real_like t = magic()
instance is_floating_real_like : floating_real_like t = magic()

inline_for_extraction noextract
let fexpm1 = Kuiper.Float32.Base.fexpm1

inline_for_extraction noextract
let flog1p = Kuiper.Float32.Base.flog1p

private noextract
noeq type cuda_math_real_like = {
  expm1_refines :
    x:t -> r:real ->
    Lemma
      (requires v_approximates x r)
      (ensures v_approximates (fexpm1 x) (exp r -. 1.0R));
  log1p_refines :
    x:t -> r:real{r >. 0.0R -. 1.0R} ->
    Lemma
      (requires v_approximates x r)
      (ensures v_approximates (flog1p x) (log (1.0R +. r)));
}

private noextract
let trusted_cuda_math : cuda_math_real_like = magic()

let expm1_approx
  (x : t)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (fexpm1 x) (exp r -. 1.0R))
= trusted_cuda_math.expm1_refines x r

let log1p_approx
  (x : t)
  (r : real { r >. 0.0R -. 1.0R })
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (flog1p x) (log (1.0R +. r)))
= trusted_cuda_math.log1p_refines x r

let lem_sizeof () = ()
