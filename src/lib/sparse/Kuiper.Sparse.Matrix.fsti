module Kuiper.Sparse.Matrix

#lang-pulse
open Kuiper
open Kuiper.Sparse.Common
open Kuiper.Chest
open Kuiper.EMatrix
module SZ = Kuiper.SizeT

// CSR
inline_for_extraction
noeq
type smatrix (et : Type0)
  (rows cols : erased nat) =
{
  nnz       : sz; // número de no-zeros
  elems     : larray et nnz; // elementos (no zero)
  col_ind   : larray sz nnz; // columna de cada elemento
  row_off   : larray sz (rows + 1); // posición de cada comienzo de fila
}

let is_global_smatrix
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  : prop
  = is_global_array m.elems
    /\ is_global_array m.col_ind
    /\ is_global_array m.row_off

let valid_smatrix
  (#nnz rows cols : nat)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  : prop
=
  // los offsets de fila están ordenados y dentro de rango
  (row_off @! 0 == 0) /\
  (row_off @! rows == nnz) /\
  // maybe separar esta propiedad en otra definicion
  (forall i j. {:pattern (row_off @! i); (row_off @! j)} i < j ==> row_off @! i <= row_off @! j) /\
  // indices de columna son posiciones validas por cada fila
  (in_bounds 0 cols col_ind) /\
  (forall (i : natlt rows).
    sorted_slice col_ind (row_off @! i) (row_off @! (i + 1))
  )

let matrix_unsparse
  (#et:Type0) {| scalar et |}
  (#nnz rows cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  : Ghost (chest2 et rows cols)
    (requires valid_smatrix rows cols col_ind row_off)
    (ensures fun _ -> true)
=
  mk2 fun i j ->
    let ri = row_off @! i in
    let re = row_off @! (i + 1) in
    if mem_slice j col_ind ri re
      then elems @! index_mem_slice j col_ind ri re
      else zero

unfold
let smatrix_pts_to'
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (e : chest2 et rows cols)
  : slprop
=
  m.elems   |-> Frac f v_elems **
  m.col_ind |-> Frac f v_col_ind **
  m.row_off |-> Frac f v_row_off **
  pure (
    valid_smatrix rows cols (cast_pos v_col_ind) (cast_pos v_row_off) /\
    e == matrix_unsparse rows cols v_elems (cast_pos v_col_ind) (cast_pos v_row_off)
  )

let smatrix_pts_to
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (e : chest2 et rows cols)
  : slprop
=
  exists* (v_elems    : lseq et m.nnz).
  exists* (v_col_ind  : lseq sz m.nnz).
  exists* (v_row_off  : lseq sz (rows + 1)).
    m.elems   |-> Frac f v_elems **
    m.col_ind |-> Frac f v_col_ind **
    m.row_off |-> Frac f v_row_off **
    pure (
      valid_smatrix rows cols (cast_pos v_col_ind) (cast_pos v_row_off) /\
      e == matrix_unsparse rows cols v_elems (cast_pos v_col_ind) (cast_pos v_row_off)
    )

inline_for_extraction noextract
unfold
instance has_pts_to_smatrix
  (#et: Type0) (#rows #cols : nat) {| scalar et |}
  : has_pts_to (smatrix et rows cols) (chest2 et rows cols) =
{
  pts_to = smatrix_pts_to;
}

instance val is_send_across_smatrix
  #et {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (vis : visibility)
  (#_ : squash (visibility_of m.elems == vis))
  (#_ : squash (visibility_of m.col_ind == vis))
  (#_ : squash (visibility_of m.row_off == vis))
  (#f : perm)
  (v : chest2 et rows cols)
  : is_send_across vis (smatrix_pts_to m #f v)

ghost
fn smatrix_share_n'
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (em : chest2 et rows cols)
  (k : pos)
  requires smatrix_pts_to' m #f v_elems v_col_ind v_row_off em
  ensures forall+ (_ : natlt k).
    smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off  em

ghost
fn smatrix_share_n
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : chest2 et rows cols)
  requires
    smatrix_pts_to m #f em
  ensures
    forall+ (_ : natlt k).
      smatrix_pts_to m #(f /. k) em

ghost
fn smatrix_gather_n'
  (#et:Type0) {| d : scalar et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (em : chest2 et rows cols)
  (k : pos)
  requires forall+ (_ : natlt k).
    smatrix_pts_to' m #(f /. k) v_elems v_col_ind v_row_off  em
  ensures smatrix_pts_to' m #f v_elems v_col_ind v_row_off em

ghost
fn smatrix_gather_n
  (#et:Type0) {| scalar et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (k : pos)
  (#f : perm)
  (#em : chest2 et rows cols)
  requires
    forall+ (_ : natlt k). smatrix_pts_to m #(f /. k) em
  ensures
    smatrix_pts_to m #f em

val unsparse_row_lemma
  (#et:Type0) {| scalar et |}
  (#nnz rows cols : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  (i : natlt rows)
  : Lemma
    (requires valid_smatrix rows cols col_ind row_off)
    (ensures
      ematrix_row (matrix_unsparse rows cols elems col_ind row_off) i ==
      seq_unsparse
        ((row_off @! i + 1) - (row_off @! i)) cols
        (Seq.slice elems (row_off @! i) (row_off @! i + 1))
        (Seq.slice col_ind (row_off @! i) (row_off @! i + 1))
    )