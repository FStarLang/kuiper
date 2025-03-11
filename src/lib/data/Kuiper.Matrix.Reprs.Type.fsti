module Kuiper.Matrix.Reprs.Type
#lang-pulse

open Kuiper
open Kuiper.Bijection
module SZ = FStar.SizeT

[@@erasable]
noeq
type mlayout (rows cols : nat) = {
  bij : natlt rows & natlt cols =~ natlt (rows * cols);
}

inline_for_extraction
type mrepr = rows:nat -> cols:nat -> mlayout rows cols

(* Concrete layout accessors. The erased is important
to allow constructing the record (i.e. instances) without a
concrete nat in scope. *)
inline_for_extraction
class clayout (#rows #cols : erased nat) (l : mlayout rows cols) = {
  c_to    : (i:SZ.t{i < rows}) -> (j:SZ.t{j < cols}) -> r:SZ.t{SZ.v r == l.bij.ff (SZ.v i, SZ.v j)};
  c_from1 : (idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == fst (l.bij.gg (SZ.v idx))};
  c_from2 : (idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == snd (l.bij.gg (SZ.v idx))};
}

type crepr_t (r : mrepr) =
  rows:SZ.t -> cols:SZ.t{SZ.fits (rows * cols)} -> clayout (r rows cols)

inline_for_extraction
class crepr (r:mrepr) = {
  map : crepr_t r;
}

inline_for_extraction noextract
let clayout_from_crepr
  (rows : SZ.t) (cols : SZ.t{SZ.fits (rows * cols)})
  (m : mrepr) (d : crepr m)
  : clayout (m rows cols)
  = d.map rows cols
