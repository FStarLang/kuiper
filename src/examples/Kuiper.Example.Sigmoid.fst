module Kuiper.Example.Sigmoid

#lang-pulse

open Kuiper
module F = Kuiper.Float32
module S = Kuiper.Spec.Sigmoid

(* Use the existing float type and the same real-valued approximation
   relation as softmax. xr is a ghost input, erased from generated CUDA.
   This is an abstract approximation proof, not a quantitative error bound. *)
inline_for_extraction noextract
fn sigmoid_approx (#xr : erased real) (x : f32)
  requires pure (x %~ xr)
  returns r : f32
  ensures pure (r %~ S.sigmoid_real (reveal xr))
{
  let xr = reveal xr;
  let negative = F.neg x;
  F.neg_approx x xr;
  let exponential = fexp negative;
  exp_approx negative (0.0R -. xr);
  let denominator = F.add_rn one exponential;
  F.add_rn_approx one exponential 1.0R (exp (0.0R -. xr));
  exp_positive (0.0R -. xr);
  let result = F.div_rn one denominator;
  F.div_rn_approx one denominator 1.0R (1.0R +. exp (0.0R -. xr));
  result
}

[@@CPrologue "__device__"]
fn sigmoid (x : f32)
  returns r : f32
  ensures pure (r %~ S.sigmoid_real (to_real x))
{
  to_real_ok x;
  sigmoid_approx #(hide (to_real x)) x
}
