module Kuiper.Floating.Base

open Kuiper.Scalars.Base
open FStar.Tactics.Typeclasses { solve, tcinstance }

inline_for_extraction noextract
class floating (t : Type) = {
  [@@@tcinstance]
  is_scalar : scalar t;
  sub : t -> t -> t;
  div : t -> t -> t;
  exp : t -> t;
  log : t -> t;
  sqrt : t -> t;
  rsqrt : t -> t;
  sin : t -> t;
  cos : t -> t;
  tan : t -> t;
  asin : t -> t;
  acos : t -> t;
  atan : t -> t;
  sinh : t -> t;
  cosh : t -> t;
  tanh : t -> t;
  ceil : t -> t;
  floor : t -> t;
  round : t -> t;
  fabs : t -> t;
  erf : t -> t;
  log2 : t -> t;
  log10 : t -> t;
  exp2 : t -> t;
  pow : t -> t -> t;
  atan2 : t -> t -> t;
  fmin : t -> t -> t;
  fmax : t -> t -> t;
  fmod : t -> t -> t;
  copysign : t -> t -> t;
  fma : t -> t -> t -> t;
}

inline_for_extraction noextract
let abs (#t:Type) {| floating t |} (x : t) : t =
  if x `gte` zero then x else sub zero x
