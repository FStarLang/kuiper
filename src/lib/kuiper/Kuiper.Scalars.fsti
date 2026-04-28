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
  lt : t -> t -> bool;
  lte : t -> t -> bool;
  gt : t -> t -> bool;
  gte : t -> t -> bool;
}

inline_for_extraction
instance _ : scalar U8.t =
  let open FStar.UInt8 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; gt; gte;
  }

inline_for_extraction
instance _ : scalar U16.t =
  let open FStar.UInt16 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; gt; gte;
  }

inline_for_extraction
instance _ : scalar U32.t =
  let open FStar.UInt32 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; gt; gte;
  }

inline_for_extraction
instance _ : scalar U64.t =
  let open FStar.UInt64 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; gt; gte;
  }

inline_for_extraction
instance _ : scalar F16.t =
  let open Kuiper.Float16 in
  {
    is_sized = solve;
    add; mul ; zero; one; lt; lte; gt; gte;
  }

inline_for_extraction
instance _ : scalar F32.t =
  let open Kuiper.Float32 in
  {
    is_sized = solve;
    add; mul ; zero; one; lt; lte; gt; gte;
  }

inline_for_extraction
instance _ : scalar F64.t =
  let open Kuiper.Float64 in
  {
    is_sized = solve;
    add; mul ; zero; one; lt; lte; gt; gte;
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
  lt = (fun _ _ -> false);
  lte = (fun _ _ -> false);
  gt = (fun _ _ -> false);
  gte = (fun _ _ -> false);
}
