module Kuiper.Float32

friend Kuiper.Float32.Base

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Canonical
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Approximates.Base
open Kuiper.Real
open Kuiper.Float.Model
open Kuiper.Float.Realops
module Sqrt = FStar.Math.Sqrt
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry

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
  neg; sub; div; bit_eq; bits;
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

instance is_real_like : real_like t = {
  to_real = (fun x -> x.ideal);
  v_approximates = (fun x r -> x.ideal == r);
  to_real_ok = (fun x -> ());
  a0 = ();
  a1 = ();
  a_add = (fun x y r s -> ());
  a_mul = (fun x y r s -> ());
}

instance is_floating_real_like : floating_real_like t = {
  approx_of_int = ();
  approx_fmax = ();
  approx_neg = ();
  approx_sub = ();
  approx_fexp = ();
  approx_div = ();
  approx_flog = ();
  approx_sqrt = ();
  approx_rsqrt = ();
}

let lt_ordered (x y : t) = Kuiper.Float32.Base.lt_ordered x y
let lt_transitive (x y z : t) = Kuiper.Float32.Base.lt_transitive x y z

inline_for_extraction noextract
let fexpm1 = Kuiper.Float32.Base.fexpm1
let fexpm1_models x r = ()

inline_for_extraction noextract
let flog1p = Kuiper.Float32.Base.flog1p
let flog1p_models x r = ()

(* Extraction selects the GPU primitive; its ideal component is computed here. *)
noextract
fn add_rn_ftz (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (r_add xr yr))
{
  binary Kuiper.Float32.Raw.add_rn_ftz r_add x y
}

noextract
fn fma_rn_ftz (x y z : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr zr : real).
      v_approximates x xr /\ v_approximates y yr /\ v_approximates z zr ==>
      v_approximates result (r_fma xr yr zr))
{
  ternary Kuiper.Float32.Raw.fma_rn_ftz r_fma x y z
}

noextract
fn mul_rn_ftz (x y : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (r_mul xr yr))
{
  binary Kuiper.Float32.Raw.mul_rn_ftz r_mul x y
}

noextract
fn exp2_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real).
      v_approximates x xr ==>
      v_approximates result (r_exp2 xr))
{
  unary Kuiper.Float32.Raw.exp2_approx_ftz r_exp2 x
}

noextract
fn rcp_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : real{xr =!= 0.0R}).
      v_approximates x xr ==>
      v_approximates result (r_rcp xr))
{
  unary Kuiper.Float32.Raw.rcp_approx_ftz Kuiper.Float.Model.rcp x
}

noextract
fn rsqrt_approx_ftz (x : t)
  preserves gpu
  returns result : t
  ensures pure (
    forall (xr : Sqrt.rpos).
      v_approximates x xr ==>
      v_approximates result (r_rsqrt xr))
{
  unary Kuiper.Float32.Raw.rsqrt_approx_ftz Kuiper.Float.Model.rsqrt x
}

let pow_approx
  (x y : t)
  (xr : Pow.rpos)
  (yr : real)
  : Lemma
      (requires v_approximates x xr /\ v_approximates y yr)
      (ensures v_approximates (pow x y) (Pow.pow xr yr))
= ()

let sin_approx
  (x : t)
  (xr : real)
  : Lemma
      (requires v_approximates x xr)
      (ensures v_approximates (sin x) (Trig.sin xr))
= ()

let cos_approx
  (x : t)
  (xr : real)
  : Lemma
      (requires v_approximates x xr)
      (ensures v_approximates (cos x) (Trig.cos xr))
= ()

let lem_sizeof () = ()
