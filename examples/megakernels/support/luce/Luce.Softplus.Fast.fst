module Luce.Softplus.Fast

#lang-pulse

open Kuiper
module F32 = Kuiper.Float32
module Fast = Kuiper.Float32.FastMath
module Softplus = Luce.Softplus

inline_for_extraction noextract
fn softplus20 (x : F32.t)
  preserves gpu
  returns result : F32.t
  ensures pure (forall (r : real). x %~ r ==> result %~ Softplus.model r)
{
  let result = if (lt (of_int #F32.t 20L) x) {
    x
  } else {
    let scaled = F32.mul_rn_ftz x (of_literal #F32.t "1.44269504088896340736");
    let exponential = F32.exp2_approx_ftz scaled;
    let argument = F32.add_rn_ftz (one #F32.t) exponential;
    Fast.log argument
  };
  // Authorized LUCE-local numerical assumption about this actual composite.
  // Not branch agreement, native equality, an error bound, or a derived theorem.
  assume pure (forall (r : real). x %~ r ==> result %~ Softplus.model r);
  result
}
