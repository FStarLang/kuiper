module Kuiper.Scalars

open Kuiper.Sized
open FStar.Tactics.Typeclasses { solve, tcinstance }
include Kuiper.Scalars.Base

(* There are no scalar instances for signed ints, we do not have
total unconditional operations on them. The instances for float types
are in their own modules. *)

inline_for_extraction noextract
instance _ : scalar UInt8.t =
  let open FStar.UInt8 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; eq;
    valid = (fun _ -> true);
  }

inline_for_extraction
instance _ : scalar UInt16.t =
  let open FStar.UInt16 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; eq;
    valid = (fun _ -> true);
  }

inline_for_extraction
instance _ : scalar UInt32.t =
  let open FStar.UInt32 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; eq;
    valid = (fun _ -> true);
  }

inline_for_extraction
instance _ : scalar UInt64.t =
  let open FStar.UInt64 in
  {
    is_sized = solve;
    add = add_mod;
    mul = mul_mod;
    zero; one; lt; lte; eq;
    valid = (fun _ -> true);
  }
