module GPU.Scalars

open GPU.Sized

module U8  = FStar.UInt8
module U16 = FStar.UInt16
module U32 = FStar.UInt32
module U64 = FStar.UInt64
module I8  = FStar.Int8
module I16 = FStar.Int16
module I32 = FStar.Int32
module I64 = FStar.Int64
module F32 = FStar.Float
module SZ  = FStar.SizeT

inline_for_extraction instance _ : sized U8.t  = { size = 1sz; }
inline_for_extraction instance _ : sized U16.t = { size = 2sz; }
inline_for_extraction instance _ : sized U32.t = { size = 4sz; }
inline_for_extraction instance _ : sized U64.t = { size = 8sz; }
inline_for_extraction instance _ : sized I8.t  = { size = 1sz; }
inline_for_extraction instance _ : sized I16.t = { size = 2sz; }
inline_for_extraction instance _ : sized I32.t = { size = 4sz; }
inline_for_extraction instance _ : sized I64.t = { size = 8sz; }
inline_for_extraction instance _ : sized F32.t = { size = 4sz; }
inline_for_extraction instance _ : sized SZ.t  = { size = 8sz; }

(* There are no simple_scalar instances for signed ints, we do not have
total unconditional operations on them. *)

inline_for_extraction
class simple_scalar (t : Type) = {
  [@@@FStar.Tactics.Typeclasses.tcinstance]
  is_sized : sized t;
  add : t -> t -> t;
  sub : t -> t -> t;
  mul : t -> t -> t;
  zero : t;
  one : t;
}

inline_for_extraction
instance _ : simple_scalar U8.t = {
  is_sized = FStar.Tactics.Typeclasses.solve;
  add = U8.add_underspec;
  sub = U8.sub_underspec;
  mul = U8.mul_underspec;
  zero = U8.zero;
  one = U8.one;
}

inline_for_extraction
instance _ : simple_scalar U16.t = {
  is_sized = FStar.Tactics.Typeclasses.solve;
  add = U16.add_underspec;
  sub = U16.sub_underspec;
  mul = U16.mul_underspec;
  zero = U16.zero;
  one = U16.one;
}

inline_for_extraction
instance _ : simple_scalar U32.t = {
  is_sized = FStar.Tactics.Typeclasses.solve;
  add = U32.add_underspec;
  sub = U32.sub_underspec;
  mul = U32.mul_underspec;
  zero = U32.zero;
  one = U32.one;
}

inline_for_extraction
instance _ : simple_scalar U64.t = {
  is_sized = FStar.Tactics.Typeclasses.solve;
  add = U64.add_underspec;
  sub = U64.sub_underspec;
  mul = U64.mul_underspec;
  zero = U64.zero;
  one = U64.one;
}

inline_for_extraction
instance _ : simple_scalar F32.t = {
  is_sized = FStar.Tactics.Typeclasses.solve;
  add = F32.add;
  sub = F32.sub;
  mul = F32.mul;
  zero = F32.zero;
  one = F32.one;
}
