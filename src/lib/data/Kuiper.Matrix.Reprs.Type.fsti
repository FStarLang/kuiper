module Kuiper.Matrix.Reprs.Type
#lang-pulse

open Kuiper
open Kuiper.Bijection
open FStar.Tactics.Typeclasses
module SZ = FStar.SizeT

[@@erasable]
noeq
type mlayout (rows cols : erased nat) = {
  bij : natlt rows & natlt cols =~ natlt (rows * cols);
}

(* erased helps *)
let mlayout_size (#rows #cols : erased nat) (_ : mlayout rows cols) : GTot nat =
  rows * cols

inline_for_extraction
type mrepr = rows:nat -> cols:nat -> mlayout rows cols

(* Concrete layout accessors. The erased is important
to allow constructing the record (i.e. instances) without a
concrete nat in scope. *)
inline_for_extraction
class clayout (#rows #cols : erased nat) (l : mlayout rows cols) = {
  [@@@no_method]  m_rows : (x:SZ.t {SZ.v x == reveal rows});
  [@@@no_method]  m_cols : (x:SZ.t {SZ.v x == reveal cols});
  [@@@no_method]  c_to    : (i:SZ.t{i < rows}) -> (j:SZ.t{j < cols}) -> r:SZ.t{SZ.v r == l.bij.ff (SZ.v i, SZ.v j)};
  [@@@no_method]  c_from1 : (idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == fst (l.bij.gg (SZ.v idx))};
  [@@@no_method]  c_from2 : (idx:SZ.t{idx < rows * cols}) -> r:SZ.t{SZ.v r == snd (l.bij.gg (SZ.v idx))};
}

#push-options "--warn_error -288"
val clayout_fits (#rows #cols : nat) (#l : mlayout rows cols)
  (c : clayout l)
  : Lemma (SZ.fits (mlayout_size l))
          [SMTPat (has_type c (clayout l))]
#pop-options

inline_for_extraction
type crepr_t (r : mrepr) =
  rows:SZ.t -> cols:SZ.t{SZ.fits (rows * cols)} -> clayout (r rows cols)

inline_for_extraction
class crepr (r:mrepr) = {
  map : crepr_t r;
}

inline_for_extraction noextract
instance clayout_from_crepr
  (rows : SZ.t) (cols : SZ.t{SZ.fits (rows * cols)})
  (m : mrepr) (d : crepr m)
  : clayout (m rows cols)
  = d.map rows cols
