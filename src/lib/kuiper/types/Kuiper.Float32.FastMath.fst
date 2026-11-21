module Kuiper.Float32.FastMath

friend Kuiper.Float32.Base
friend Kuiper.Float32

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs
open Kuiper.Float32
open Kuiper.Float.Model
open Kuiper.Approximates.Base
open Kuiper.Real
open Kuiper.Float.Realops
module M = Kuiper.Float.Model
module R = Kuiper.Float32.FastMath.Raw

noextract
fn sin (x:t)
  preserves gpu
  returns result:t
  ensures pure (forall (xr:real).
    v_approximates x xr ==> v_approximates result (r_sin xr))
{
  unary R.sin r_sin x
}

noextract
fn cos (x:t)
  preserves gpu
  returns result:t
  ensures pure (forall (xr:real).
    v_approximates x xr ==> v_approximates result (r_cos xr))
{
  unary R.cos r_cos x
}

noextract
fn log (x:t)
  preserves gpu
  returns result:t
  ensures pure (forall (xr:real{xr >. 0.0R}).
    v_approximates x xr ==> v_approximates result (r_log xr))
{
  unary R.log M.log x
}

noextract
fn exp (x:t)
  preserves gpu
  returns result:t
  ensures pure (forall (xr:real).
    v_approximates x xr ==> v_approximates result (r_exp xr))
{
  unary R.exp r_exp x
}

noextract
fn divide (x y:t)
  preserves gpu
  returns result:t
  ensures pure (forall (xr:real) (yr:real{yr =!= 0.0R}).
    v_approximates x xr /\ v_approximates y yr ==>
    v_approximates result (r_div xr yr))
{
  binary R.divide M.div x y
}

noextract
fn fma_rn (x y z:t)
  preserves gpu
  returns result:t
  ensures pure (forall (xr yr zr:real).
    v_approximates x xr /\ v_approximates y yr /\ v_approximates z zr ==>
    v_approximates result (r_fma xr yr zr))
{
  ternary R.fma_rn r_fma x y z
}

let pure_sub_rn (x y:t) = binary R.sub_rn r_sub x y

noextract
fn sub_rn (x y:t)
  preserves gpu
  returns result:t
  ensures pure (result == pure_sub_rn x y)
{
  binary R.sub_rn r_sub x y
}
