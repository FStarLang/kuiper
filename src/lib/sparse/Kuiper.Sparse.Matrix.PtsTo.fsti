module Kuiper.Sparse.Matrix.PtsTo

#lang-pulse

open Kuiper
open FStar.Tactics.V2 { exact }
open Kuiper.Sparse.Common
open Kuiper.Tensor
open Kuiper.Array.Vectorized
open Kuiper.EMatrix
open Kuiper.Array2.Strided { strided_row_major }

module Math = Kuiper.Sparse.Math

(* permits over cells *)

let matrix_live_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : layout2 rows cols)
  (gm : array2 et lm)
  (i : natlt rows)
  (j : natlt cols)
: slprop
= exists* (v : et). Cell gm (idx2 i j) |-> v

let matrix_pts_to_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
  (v : lseq et (chunk et))
: GTot slprop
=
  forall+ (k : natlt (chunk et)).
    Cell gm (idx2 i (j + k <: natlt cols)) |-> Seq.index v k

let matrix_live_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
: slprop
= exists* v. matrix_pts_to_vec gm i j v

unfold
let matrix_pts_to_vec_slice
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { j + chunk et <= cols })
  (#n : nat)
  (v : lseq et n)
  (k : nat { k + chunk et <= n })
: slprop
= matrix_pts_to_vec gm i j (seq_chunk v k)

let matrix_pts_to_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
: slprop
= when__ (j < cols) (fun _ -> matrix_pts_to_vec gm i j v)

let matrix_live_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
: slprop
= exists* v. matrix_pts_to_vec_in_bounds gm i j v

unfold
let matrix_pts_to_vec_slice_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#n : nat)
  (v : lseq et n)
  (k : nat { k + chunk et <= n })
: slprop
= matrix_pts_to_vec_in_bounds gm i j (seq_chunk v k)

unfold
let matrix_pts_to_cell_in_matrix_in_bounds
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat)
  (em : chest2 et rows cols)
: slprop
=
  when__ (j < cols) (fun _ ->
    Cell gm (idx2 i (j <: natlt cols)) |-> (acc2 em i j)
  )

unfold
let matrix_pts_to_vec_in_matrix_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : chest2 et rows cols)
: slprop
=
  when__ (j < cols) (fun _ ->
    matrix_pts_to_vec gm i j (ematrix_row_chunk em i j)
  )

let pts_to_tile_in_matrix
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat)
  (em : chest2 et rows cols)
  (row_tile : nat)
: slprop
=
  forall+ (k : natlt row_tile).
    matrix_pts_to_cell_in_matrix_in_bounds gm i (j + k) em

(* thread owns in matrix *)

unfold
let thread_live_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (row_tile nthr : nat)
  (k : natlt (row_tile / chunk et))
: slprop
= matrix_live_vec_in_bounds gm i (offset_chunk et j k nthr)

let thread_live_tile_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (row_tile nthr : nat)
: slprop
=
  forall+ (k : natlt (row_tile / chunk et)).
    thread_live_vec gm i j row_tile nthr k

unfold
let thread_pts_to_vec_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
  (k : natlt (row_tile / chunk et))
: slprop
=
  matrix_pts_to_vec_slice_in_bounds gm
    i (offset_chunk et j k nthr)
    s (k * chunk et)

let thread_pts_to_tile_vec_underspec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (nthr : nat)
: slprop
=
  forall+ (k : natlt (row_tile / chunk et)).
    thread_pts_to_vec_underspec gm i j s nthr k

unfold
let thread_pts_to_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : chest2 et rows cols)
  (nthr k : nat)
: slprop
=
  matrix_pts_to_vec_in_matrix_in_bounds gm
    i (offset_chunk et j k nthr) em

let thread_pts_to_tile_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (em : chest2 et rows cols)
  (row_tile nthr : nat)
: slprop
=
  forall+ (k : natlt (row_tile / chunk et)).
    thread_pts_to_vec gm i j em nthr k

let thread_offset
  (et : Type0) {| sized et, has_vec_cpy et |}
  (j tid : nat)
: Pure nat (requires chunk et /? j) (ensures fun off -> chunk et /? off)
=
  Math.lineal_divides (chunk et) j (chunk et) tid;
  j + tid * v (chunk et)

(* helpers *)

ghost
fn fold_matrix_pts_to_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
  (#_ : squash (j < cols))
  requires matrix_pts_to_vec gm i j v
  ensures matrix_pts_to_vec_in_bounds gm i j v

ghost
fn fold_matrix_pts_to_vec_slice_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#n : nat)
  (v : lseq et n)
  (k : nat { k + chunk et <= n })
  (#_ : squash (j < cols))
  requires matrix_pts_to_vec_slice gm i j v k
  ensures matrix_pts_to_vec_slice_in_bounds gm i j v k

ghost
fn fold_matrix_pts_to_vec_not_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
  requires pure (j >= cols)
  ensures matrix_pts_to_vec_in_bounds gm i j v

ghost
fn fold_matrix_pts_to_vec_slice_not_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#n : nat)
  (v : lseq et n)
  (k : nat { k + chunk et <= n })
  requires pure (j >= cols)
  ensures matrix_pts_to_vec_slice_in_bounds gm i j v k

ghost
fn unfold_matrix_pts_to_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
  (#_ : squash (j < cols))
  requires matrix_pts_to_vec_in_bounds gm i j v
  ensures matrix_pts_to_vec gm i j v

ghost
fn unfold_matrix_live_vec_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#_ : squash (j < cols))
  requires matrix_live_vec_in_bounds gm i j
  ensures matrix_live_vec gm i j

ghost
fn unfold_matrix_pts_to_vec_not_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (v : lseq et (chunk et))
  requires pure (j >= cols)
  requires matrix_pts_to_vec_in_bounds gm i j v

ghost
fn unfold_matrix_live_vec_not_in_bounds
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  requires pure (j >= cols)
  requires matrix_live_vec_in_bounds gm i j

ghost
fn fold_thread_pts_to_tile_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : nat { chunk et /? j })
  (#row_tile : nat { chunk et /? row_tile })
  (s : lseq et row_tile)
  (em : chest2 et rows cols)
  (nthr : nat)
  requires thread_pts_to_tile_vec_underspec gm i j s nthr
  requires pure (is_ematrix_tile em i j s nthr)
  ensures thread_pts_to_tile_vec gm i j em row_tile nthr

ghost
fn thread_pts_to_tile_vec_gather
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
  (em : chest2 et rows cols)
  (nthr : nat)
  (trow_tile : nat { chunk et /? trow_tile })
  requires forall+ (tid : natlt nthr).
    thread_pts_to_tile_vec gm i (thread_offset et j tid) em trow_tile nthr
  ensures pts_to_tile_in_matrix gm i j em (trow_tile * nthr)

ghost
fn gather_pts_to_tile_in_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (em : chest2 et rows cols)
  (row_tile : pos)
  requires forall+ (b : natlt (cols `divup` row_tile)).
    pts_to_tile_in_matrix gm i (b * row_tile) em row_tile
  ensures forall+ (j : natlt cols).
    Cell gm (idx2 i j) |-> acc2 em i j

ghost
fn gahter_pts_to_tile_in_matrix
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (em : chest2 et rows cols)
  (row_tile : pos)
  requires array_exists (core gm)
  requires pure (fits (tlayout_ulen l))
  requires forall+ (r : natlt rows) (b : natlt (cols `divup` row_tile)).
    pts_to_tile_in_matrix gm r (b * row_tile) em row_tile
  ensures gm |-> em