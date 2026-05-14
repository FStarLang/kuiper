module Kuiper.Floating

open Kuiper.Scalars
open FStar.Tactics.Typeclasses { solve, tcinstance }

module F16 = Kuiper.Float16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64

inline_for_extraction
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

let abs (#t:Type) {| floating t |} (x : t) : t =
  if x `gte` zero then x else sub zero x

inline_for_extraction
instance _ : floating F16.t =
  let open F16 in
  {
    is_scalar = solve;
    sub; div; exp; log;
    sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
    sinh; cosh; tanh; ceil; floor; round; fabs; erf;
    log2; log10; exp2; pow; atan2; fmin; fmax; fmod;
    copysign; fma;
  }

inline_for_extraction
instance _ : floating F32.t =
  let open F32 in
  {
    is_scalar = solve;
    sub; div; exp; log;
    sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
    sinh; cosh; tanh; ceil; floor; round; fabs; erf;
    log2; log10; exp2; pow; atan2; fmin; fmax; fmod;
    copysign; fma;
  }

inline_for_extraction
instance _ : floating F64.t =
  let open F64 in
  {
    is_scalar = solve;
    sub; div; exp; log;
    sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
    sinh; cosh; tanh; ceil; floor; round; fabs; erf;
    log2; log10; exp2; pow; atan2; fmin; fmax; fmod;
    copysign; fma;
  }
