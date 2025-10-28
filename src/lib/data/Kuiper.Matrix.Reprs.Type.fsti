module Kuiper.Matrix.Reprs.Type
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.Injection
open FStar.Tactics.Typeclasses
module SZ = Kuiper.SizeT

[@@erasable]
noeq
type mlayout (rows cols : nat) = {
  len : nat;
  map : natlt rows & natlt cols @~> natlt len;
}

let cell_of_pos #rows #cols
  (l : mlayout rows cols) (i : natlt rows) (j : natlt cols) : GTot nat =
  l.map.f (i, j)

let is_full_layout (l : mlayout 'rows 'cols) : prop =
  is_surj l.map.f

let full_mlayout (rows cols : nat) =
  l : mlayout rows cols { is_full_layout l }

let mlayout_size (#rows #cols : nat) (l : mlayout rows cols) : GTot nat =
  l.len

inline_for_extraction
type mrepr = rows:nat -> cols:nat -> full_mlayout rows cols

(* Concrete layout accessors. The erased is important
to allow constructing the record (i.e. instances) without a
concrete nat in scope. *)
inline_for_extraction
class clayout (#rows #cols : erased nat) (l : mlayout rows cols) = {
  [@@@no_method]  m_len  : (x:SZ.t {SZ.v x == reveal l.len});
  [@@@no_method]  m_rows : (x:SZ.t {SZ.v x == reveal rows});
  [@@@no_method]  m_cols : (x:SZ.t {SZ.v x == reveal cols});
  [@@@no_method]  c_to   : (i:SZ.t{i < rows}) -> (j:SZ.t{j < cols}) -> r:SZ.t{SZ.v r == l.map.f (SZ.v i, SZ.v j)};
}

val full_layout_size_lt #rows #cols (l : mlayout rows cols)
  : Lemma (ensures  l.len >= rows * cols)
          [SMTPat (has_type l (mlayout rows cols))]

val full_layout_size #rows #cols (l : mlayout rows cols)
  : Lemma (requires is_full_layout l)
          (ensures  l.len == rows * cols)
          [SMTPat (is_full_layout l)]

val clayout_fits (#rows #cols : nat) (#l : mlayout rows cols)
  (c : clayout l)
  : Lemma (SZ.fits (mlayout_size l))
          [SMTPat (has_type c (clayout l))]

inline_for_extraction
type crepr_t (r : mrepr) =
  rows:SZ.t -> cols:SZ.t{SZ.fits (rows * cols)} -> clayout (r rows cols)

inline_for_extraction
class crepr (r:mrepr) = {
  map : crepr_t r;
}

inline_for_extraction noextract
instance clayout_from_crepr
  (rows : nat) (cols : nat{SZ.fits (rows * cols)})
  {| concrete_sz rows, concrete_sz cols |}
  (m : mrepr) (d : crepr m)
  : clayout (m rows cols)
  = d.map (concr rows) (concr cols)

inline_for_extraction noextract
class strided_row_major (#rows #cols : erased nat) (l : mlayout rows cols) = {
  [@@@no_method]
  offset : sz;
  [@@@no_method]
  stride : sz;
  [@@@no_method]
  pf : i:natlt rows -> j:natlt cols ->
         squash (l.map.f (i,j) == offset + stride * i + j);
}

inline_for_extraction noextract
class strided_col_major (#rows #cols : erased nat) (l : mlayout rows cols) = {
  [@@@no_method]
  offset : sz;
  [@@@no_method]
  stride : sz;
  [@@@no_method]
  pf : i:natlt rows -> j:natlt cols ->
         squash (l.map.f (i,j) == offset + stride * j + i);
}
