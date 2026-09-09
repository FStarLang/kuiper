module Kuiper.Float32

open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Canonical
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Approximates.Base

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
  sub; div;
  of_int; of_literal; of_int_zero; of_int_one;
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

inline_for_extraction noextract
let neg = Float32.Base.neg

inline_for_extraction noextract
let add_rn = Float32.Base.add_rn

inline_for_extraction noextract
let div_rn = Float32.Base.div_rn

(* The explicit CUDA operations share that trusted approximation model. *)
let neg_approx = magic()
let add_rn_approx = magic()
let div_rn_approx = magic()

let lem_sizeof () = ()
