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
  sub : t -> t -> t;
  mul : t -> t -> t;
  zero : t;
  one : t;
}

inline_for_extraction
class floating (t : Type) = {
  [@@@tcinstance]
  is_scalar : scalar t;
  div : t -> t -> t;
  exp : t -> t;
}

inline_for_extraction
instance _ : scalar U8.t = {
  is_sized = solve;
  add = U8.add_mod;
  sub = U8.sub_mod;
  mul = U8.mul_mod;
  zero = U8.zero;
  one = U8.one;
}

inline_for_extraction
instance _ : scalar U16.t = {
  is_sized = solve;
  add = U16.add_mod;
  sub = U16.sub_mod;
  mul = U16.mul_mod;
  zero = U16.zero;
  one = U16.one;
}

inline_for_extraction
instance _ : scalar U32.t = {
  is_sized = solve;
  add = U32.add_mod;
  sub = U32.sub_mod;
  mul = U32.mul_mod;
  zero = U32.zero;
  one = U32.one;
}

inline_for_extraction
instance _ : scalar U64.t = {
  is_sized = solve;
  add = U64.add_mod;
  sub = U64.sub_mod;
  mul = U64.mul_mod;
  zero = U64.zero;
  one = U64.one;
}

inline_for_extraction
instance _ : scalar F16.t = {
  is_sized = solve;
  add = F16.add;
  sub = F16.sub;
  mul = F16.mul;
  zero = F16.zero;
  one = F16.one;
}

inline_for_extraction
instance _ : floating F16.t = {
  is_scalar = solve;
  div = F16.div;
  exp = F16.exp;
}

inline_for_extraction
instance _ : scalar F32.t = {
  is_sized = solve;
  add = F32.add;
  sub = F32.sub;
  mul = F32.mul;
  zero = F32.zero;
  one = F32.one;
}

inline_for_extraction
instance _ : floating F32.t = {
  is_scalar = solve;
  div = F32.div;
  exp = F32.exp;
}

inline_for_extraction
instance _ : scalar F64.t = {
  is_sized = solve;
  add = F64.add;
  sub = F64.sub;
  mul = F64.mul;
  zero = F64.zero;
  one = F64.one;
}

inline_for_extraction
instance _ : floating F64.t = {
  is_scalar = solve;
  div = F64.div;
  exp = F64.exp;
}
