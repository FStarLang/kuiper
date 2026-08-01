module Kuiper.Array2.Vectorized
#lang-pulse

(* Vectorized read and write for Array2 (Tensor-backed) layouts. *)

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Chest

open Kuiper.Tensor { array2, layout2, idx2 }
open Kuiper.Array2.Strided
module T = Kuiper.Tensor

(* Ownership of a run of [w] consecutive cells of row [i], starting at
   column [j], holding [v]. This is the cell-level (as opposed to
   whole-matrix) precondition of a vectorized access. *)
let row_cells
  (#et : Type0) {| sized et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] gm : array2 et l)
  (f : perm)
  (i : natlt rows)
  (j : nat)
  (w : nat { j + w <= cols })
  (v : seq et { Seq.length v == w })
  : slprop
= forall+ (x : natlt w).
    T.tensor_pts_to_cell gm #f (idx2 i (j + x)) (Seq.index v x)

ghost
fn row_cells_to_slice
  (#et : Type0) {| sized et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (#f : perm)
  (v : seq et { Seq.length v == w })
  requires row_cells gm f i j w v
  ensures
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j) (cell_of_pos l i j + w) v

ghost
fn row_slice_to_cells
  (#et : Type0) {| sized et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat)
  (w : pos { j + w <= cols })
  (#f : perm)
  (v : seq et { Seq.length v == w })
  requires
    pts_to_slice (T.core gm) #f
      (cell_of_pos l i j) (cell_of_pos l i j + w) v
  ensures row_cells gm f i j w v

inline_for_extraction noextract
fn array2_vec_read
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : chest2 et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  pure (aligned' 16 (T.core gm) (cell_of_pos l i j))
  requires  pure (aligned 16 arr)
  requires  arr |-> s
  requires  pure (Pulse.Lib.Array.length arr == chunk et)
  ensures   arr |-> Seq.init_ghost (chunk et) (fun x -> acc2 em i (j + x))

(* [em] with row [i], columns [j .. j+w), overwritten by [v]. *)
let chest2_row_blit
  (#et : Type) (#rows #cols : nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  (j : nat)
  (w : nat { j + w <= cols })
  (v : seq et { Seq.length v == w })
  : chest2 et rows cols
= mk2 (fun r c ->
    if r = i && j <= c && c < j + w
    then Seq.index v (c - j)
    else acc2 em r c)

(* Vectorized write of a [chunk et]-wide run of cells, dual to
   [array2_vec_read]. Only the run itself needs to be owned. *)
inline_for_extraction noextract
fn array2_vec_write_cells
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (arr : array et)
  (#f : perm)
  (old nv : erased (seq et))
  (#_ : squash (Seq.length old == chunk et /\ Seq.length nv == chunk et))
  ()
  preserves gpu
  preserves arr |-> Frac f nv
  requires  row_cells gm 1.0R i j (chunk et) (reveal old)
  requires  pure (aligned' 16 (T.core gm) (cell_of_pos l i j))
  requires  pure (aligned 16 arr)
  ensures   row_cells gm 1.0R i j (chunk et) (reveal nv)

(* Vectorized write into a fully-owned matrix. *)
inline_for_extraction noextract
fn array2_vec_write
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#em : chest2 et rows cols)
  (arr : array et)
  (#f : perm)
  (nv : erased (seq et))
  (#_ : squash (Seq.length nv == chunk et))
  ()
  preserves gpu
  preserves arr |-> Frac f nv
  requires  gm |-> em
  requires  pure (aligned' 16 (T.core gm) (cell_of_pos l i j))
  requires  pure (aligned 16 arr)
  ensures   gm |-> chest2_row_blit em i j (chunk et) (reveal nv)
