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
