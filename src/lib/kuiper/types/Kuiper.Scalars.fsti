module Kuiper.Scalars

open Kuiper.Sized
open FStar.Tactics.Typeclasses { solve, tcinstance }
include Kuiper.Scalars.Base

(* There are no scalar instances for signed ints, we do not have
total unconditional operations on them. The instances for float types
are in their own modules. *)

inline_for_extraction noextract
instance val is_scalar_u8 : scalar UInt8.t

inline_for_extraction noextract
instance val is_scalar_u16 : scalar UInt16.t

inline_for_extraction noextract
instance val is_scalar_u32 : scalar UInt32.t

inline_for_extraction noextract
instance val is_scalar_u64 : scalar UInt64.t

(* Sizes reached through the scalar superclass, for vectorized copies. *)
val lem_sizeof_u8  () : Lemma (size #UInt8.t  == 1sz)
val lem_sizeof_u16 () : Lemma (size #UInt16.t == 2sz)
val lem_sizeof_u32 () : Lemma (size #UInt32.t == 4sz)
val lem_sizeof_u64 () : Lemma (size #UInt64.t == 8sz)
