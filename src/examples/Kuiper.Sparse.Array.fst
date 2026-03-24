module Kuiper.Sparse.Array

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
module SZ = FStar.SizeT

// This is here to force extraction.
let _ = 1ul

(* Sparse array *)
noextract
let valid_pos (#nnz l : nat) (s : lseq nat nnz) : prop
= in_bounds 0 l s /\ sorted s

noeq
inline_for_extraction
type sarray (et : Type0)
  (l : erased nat) =
  // ^ longitud "virtual" del array
{ nnz   : sz; // número de no-zeros len   : (len : sz {SZ.v len == reveal l}); // longitud "real" del array virtual
  elems : gpu_array et nnz; // elementos (no zero)
  pos   : gpu_array sz nnz; // posición de cada elemento
}

let unsparse
  (#et:Type0) {| scalar et |}
  (nnz l : nat)
  (elems : lseq et nnz)
  (pos   : lseq nat nnz)
  : GTot (lseq et l)
=
  let open FStar.Seq in
  init l fun i ->
    if mem i pos
      then elems @! index_mem i pos
      else zero

// MAYBE unfold
let sarray_pts_to'
  (#et:Type0) {| d : scalar et |} (#l : nat)
  (a : sarray et l)
  (#[Tactics.exact (`1.0R)] f : perm)
  (s : seq et)
  (v_elems : lseq et a.nnz)
  (v_pos   : lseq sz a.nnz)
  : slprop
=
    a.elems |-> Frac f v_elems **
    a.pos   |-> Frac f v_pos **
    pure (
      valid_pos l (cast_pos #a.nnz v_pos <: lseq nat a.nnz)
      /\ s == unsparse a.nnz l v_elems (cast_pos v_pos)
    )

let sarray_pts_to
  (#et:Type0) {| d : scalar et |} #l
  (a : sarray et l)
  (#[Tactics.exact (`1.0R)] f : perm)
  (s : seq et)
  : slprop
=
  exists* (v_elems : lseq et a.nnz) (v_pos : lseq sz a.nnz).
    sarray_pts_to' a #f s v_elems v_pos

inline_for_extraction noextract
unfold
instance has_pts_to_sarray
  (#et: Type0) (#l : nat) {| scalar et |}
  : has_pts_to (sarray et l) (seq et) =
{
  pts_to = sarray_pts_to;
}
