module Kuiper.Example.PTX

#lang-pulse

open Kuiper
module PTX = Kuiper.PTX

let approximation_contract
  (x y : f32)
  (xr yr : real)
  : Lemma
      (requires x %~ xr /\ y %~ yr)
      (ensures
        PTX.ex2_approx_ftz_f32 (PTX.mul_rn_ftz_f32 x y)
          %~ PTX.exp2_real (xr *. yr))
=
  PTX.mul_rn_ftz_f32_approx x y xr yr;
  PTX.ex2_approx_ftz_f32_approx
    (PTX.mul_rn_ftz_f32 x y)
    (xr *. yr)

[@@CPrologue "__device__"]
fn apply
  (x y : f32)
  returns f32
{
  PTX.ex2_approx_ftz_f32 (PTX.mul_rn_ftz_f32 x y)
}
