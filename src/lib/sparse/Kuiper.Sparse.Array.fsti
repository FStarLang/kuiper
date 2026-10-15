module Kuiper.Sparse.Array

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
module SZ = FStar.SizeT

noeq
inline_for_extraction
type sarray (et : Type0)
  (l : erased nat) =
  // ^ longitud "virtual" del array
{ nnz   : sz; // número de no-zeros len   : (len : sz {SZ.v len == reveal l}); // longitud "real" del array virtual
  elems : larray et nnz; // elementos (no zero)
  pos   : larray sz nnz; // posición de cada elemento
}

unfold
let sarray_pts_to'
  (#et:Type0) {| d : scalar et |} (#l : nat)
  (a : sarray et l)
  (#[full_default ()] f : perm)
  (s : seq et)
  (v_elems : lseq et a.nnz)
  (v_pos   : lseq sz a.nnz)
  : slprop
=
    a.elems |-> Frac f v_elems **
    a.pos   |-> Frac f v_pos   **
    pure (
      valid_pos l (cast_pos #a.nnz v_pos)
      /\ s == seq_unsparse a.nnz l v_elems (cast_pos v_pos)
    )

let sarray_pts_to
  (#et:Type0) {| d : scalar et |} #l
  (a : sarray et l)
  (#[full_default ()] f : perm)
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

ghost
fn sarray_pts_to_eq
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (#f1 f2 : perm)
  (#v1 #v2 : seq et)
  requires
    sarray_pts_to a #f1 v1 **
    sarray_pts_to a #f2 v2
  ensures
    sarray_pts_to a #f1 v2 **
    sarray_pts_to a #f2 v2

ghost
fn sarray_share_n
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (n : pos)
  (#f : perm)
  (#s : seq et)
  requires
    a |-> Frac f s
  ensures
    forall+ (_ : natlt n). a |-> Frac (f /. n) s

ghost
fn sarray_share
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (#f : perm)
  (#s : seq et)
  requires
    sarray_pts_to a #f s
  ensures
    sarray_pts_to a #(f /. 2) s **
    sarray_pts_to a #(f /. 2) s

ghost
fn sarray_gather_n
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (n : pos)
  (#f : perm)
  (#s : seq et)
  requires
    forall+ (_ : natlt n). a |-> Frac (f /. n) s
  ensures
    a |-> Frac f s

ghost
fn sarray_gather
  (#et:Type0) {| scalar et |}
  (#l : nat)
  (a : sarray et l)
  (#f : perm)
  (#s : seq et)
  requires
    sarray_pts_to a #(f /. 2) s **
    sarray_pts_to a #(f /. 2) s
  ensures
    sarray_pts_to a #f s