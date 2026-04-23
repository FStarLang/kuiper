module Kuiper.Scalars

open Kuiper.Sized
open FStar.Tactics.Typeclasses { solve, tcinstance }

module U8  = FStar.UInt8
module U16 = FStar.UInt16
module U32 = FStar.UInt32
module U64 = FStar.UInt64

module F16 = Kuiper.Float16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64

(* There are no scalar instances for signed ints, we do not have
total unconditional operations on them. *)

inline_for_extraction
class scalar (t : Type) = {
  [@@@tcinstance]
  is_sized : sized t;
  add : t -> t -> t;
  mul : t -> t -> t;
  zero : t;
  one : t;
  gt : t -> t -> bool;
  gte : t -> t -> bool;
}

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
instance _ : scalar U8.t =
  let open FStar.UInt8 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; gt; gte;
  }

inline_for_extraction
instance _ : scalar U16.t =
  let open FStar.UInt16 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; gt; gte;
  }

inline_for_extraction
instance _ : scalar U32.t =
  let open FStar.UInt32 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; gt; gte;
  }

inline_for_extraction
instance _ : scalar U64.t =
  let open FStar.UInt64 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; gt; gte;
  }

inline_for_extraction
instance _ : scalar F16.t =
  let open Kuiper.Float16 in
  {
    is_sized = solve;
    add; mul ; zero; one; gt; gte;
  }

inline_for_extraction
instance _ : floating F16.t =
  let open Kuiper.Float16 in
  {
    is_scalar = solve;
    sub; div; exp; log;
    sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
    sinh; cosh; tanh; ceil; floor; round; fabs; erf;
    log2; log10; exp2; pow; atan2; fmin; fmax; fmod;
    copysign; fma;
  }

inline_for_extraction
instance _ : scalar F32.t =
  let open Kuiper.Float32 in
  {
    is_sized = solve;
    add; mul ; zero; one; gt; gte;
  }

inline_for_extraction
instance _ : floating F32.t =
  let open Kuiper.Float32 in
  {
    is_scalar = solve;
    sub; div; exp; log;
    sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
    sinh; cosh; tanh; ceil; floor; round; fabs; erf;
    log2; log10; exp2; pow; atan2; fmin; fmax; fmod;
    copysign; fma;
  }

inline_for_extraction
instance _ : scalar F64.t =
  let open Kuiper.Float64 in
  {
    is_sized = solve;
    add; mul ; zero; one; gt; gte;
  }

inline_for_extraction
instance _ : floating F64.t =
  let open Kuiper.Float64 in
  {
    is_scalar = solve;
    sub; div; exp; log;
    sqrt; rsqrt; sin; cos; tan; asin; acos; atan;
    sinh; cosh; tanh; ceil; floor; round; fabs; erf;
    log2; log10; exp2; pow; atan2; fmin; fmax; fmod;
    copysign; fma;
  }

noextract
instance _ : scalar Real.real =
  let open FStar.Real in
  {
  is_sized = { size = 0sz; default = 0.0R };
  add = (+.);
  mul = ( *. );
  zero = 0.0R;
  one = 1.0R;
  // FIXME: reals cannot be compared in Tot.
  // We're overdue for restructuring the class hierarchy.
  gt = (fun _ _ -> false);
  gte = (fun _ _ -> false);
}
