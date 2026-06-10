module Kuiper.Float64

open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Approximates.Base

open Kuiper.Float64.Base

let t = Float64.Base.t

inline_for_extraction noextract
instance _ : sized t = { size = 8sz; default = zero }

inline_for_extraction noextract
instance _ : scalar t = {
  is_sized = solve;
  add; mul; zero; one; lt; lte; eq;
}

inline_for_extraction noextract
instance is_floating : floating t = {
  is_scalar = solve;
  sub; div;
  of_int; of_int_zero; of_int_one;
  kind;
  largest; infinity;
  kind_one; kind_zero; kind_largest; kind_infinity;
  fexp; flog; sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
  sinh; cosh; tanh; ceil; floor; round; fabs; erf; log2;
  log10; exp2; pow; atan2; fmin; fmax; fmod; copysign;
  fma;
}

(* Approximation semantics is assumed. *)
instance is_real_like          : real_like t = magic()
instance is_floating_real_like : floating_real_like t = magic()

let lem_sizeof () = ()
