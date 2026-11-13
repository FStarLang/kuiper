module Luce.RoPE.Constants

open FStar.Real
module Pow = FStar.Math.Pow
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module A = Kuiper.Approximates.Base
module Data = Luce.RoPE.Constants.Data

inline_for_extraction noextract
let inverse_frequency (j:nat{j < 32}) : FP.t =
  F.of_literal (Data.data j).output

val inverse_frequency_approx (j:nat{j < 32})
  : Lemma (A.v_approximates (inverse_frequency j)
      (1.0R /. Pow.pow 10000000.0R (of_int j /. 32.0R)))
