module Kuiper.Sized

module SZ = Kuiper.SizeT
open Kuiper.Canonical

(* Note: most times, the extraction plugin will generate sizeof expressions
for the corresponding type, and not really use the number inside here. However,
when reading shmem descriptions, these are inductive types that package the
type + the sized evidence together, and there, the type is erased into unit.
So in that case, we do take the size from here. So they MUST be correct. Ideally,
this typeclass would be empty, and we would always use sizeof.

Another sharp edge is that we sometimes need to reason about alignment and
vectorized copies, and for that we need the prover to know about these sizes. *)
inline_for_extraction noextract
class sized (t:Type) = {
  size : SZ.t;
  default: t;
}

inline_for_extraction noextract instance sizedu8  : sized UInt8.t  = { size = 1sz; default = 0uy }
assume Canonical_sizedu8 : canonical sizedu8

inline_for_extraction noextract instance sizedu16 : sized UInt16.t = { size = 2sz; default = 0us }
assume Canonical_sizedu16 : canonical sizedu16

inline_for_extraction noextract instance sizedu32 : sized UInt32.t = { size = 4sz; default = 0ul }
assume Canonical_sizedu32 : canonical sizedu32

inline_for_extraction noextract instance sizedu64 : sized UInt64.t = { size = 8sz; default = 0uL }
assume Canonical_sizedu64 : canonical sizedu64

(* Note, we extract F*'s SizeT into uint32_t. *)
inline_for_extraction noextract instance sizedsz : sized SizeT.t  = { size = 4sz; default = 0sz }
assume Canonical_sizedsz : canonical sizedsz
