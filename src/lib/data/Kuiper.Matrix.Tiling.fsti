module Kuiper.Matrix.Tiling
#lang-pulse

(* An API for tiling matrices, implemented with array views. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Injection
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
open Pulse.Lib.Trade
module SZ = Kuiper.SizeT

include Kuiper.EMatrix.Tiling

let tile_inj_f
  (#rows #cols : nat)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : (natlt trows & natlt tcols
    -> natlt rows & natlt cols)
=
   (fun (i, j) -> (tr * trows + i, tc * tcols + j))

let tile_inj
  (#rows #cols : nat)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : (natlt trows & natlt tcols
    @~> natlt rows & natlt cols)
= {
   f = tile_inj_f trows tcols tr tc;
   is_inj = ez;
}

let subtile_layout
  (#rows #cols : nat)
  (l : mlayout rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : mlayout trows tcols =
  {
    len = l.len;
    map = inj_comp (tile_inj trows tcols tr tc) l.map;
  }

inline_for_extraction noextract
instance val strided_row_major_subtile (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (#_ : squash (SZ.fits (mlayout_size l)))
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_row_major (subtile_layout l trows tcols tr tc)

val lemma_subtile_strided_row_major_offset
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (mlayout_size l))
          (ensures
            SZ.v (strided_row_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tr * trows) + tc * tcols)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).offset]

val lemma_subtile_strided_row_major_stride
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_row_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (mlayout_size l))
          (ensures
            (strided_row_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).stride]

inline_for_extraction noextract
instance val strided_col_major_subtile (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (#_ : squash (SZ.fits (mlayout_size l)))
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_col_major (subtile_layout l trows tcols tr tc)

val lemma_subtile_strided_col_major_offset
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (mlayout_size l))
          (ensures
            SZ.v (strided_col_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tc * tcols) + tr * trows)
          [SMTPat (strided_col_major_subtile l trows tcols tr tc).offset]

val lemma_subtile_strided_col_major_stride
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| sub : strided_col_major l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (mlayout_size l))
          (ensures
            (strided_col_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_col_major_subtile l trows tcols tr tc).stride]

inline_for_extraction noextract
instance val c_subtile_layout
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  {| clayout l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : clayout (subtile_layout l trows tcols tr tc)

inline_for_extraction noextract
val gpu_matrix_subtile
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {0 < trows /\ trows /? rows})
  (tcols : erased nat {0 < tcols /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  : Tot (gpu_matrix et (subtile_layout l trows tcols tr tc))

val gpu_matrix_subtile_base
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {0 < trows /\ trows /? rows})
  (tcols : erased nat {0 < tcols /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  : Lemma (
      core (gpu_matrix_subtile gm trows tcols tr tc)
      ==
      core gm
    )
    [SMTPat (core (gpu_matrix_subtile gm trows tcols tr tc))]

let is_gpu_matrix_subtile_global
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (#gm : gpu_matrix et l)
  (#trows : erased nat {0 < trows /\ trows /? rows})
  (#tcols : erased nat {0 < tcols /\ tcols /? cols})
  (#tr : enatlt (rows / trows))
  (#tc : enatlt (cols / tcols))
: Lemma
  (ensures
    is_global_matrix (gpu_matrix_subtile gm trows tcols tr tc) <==>
    is_global_matrix gm)
  [SMTPat (is_global_matrix (gpu_matrix_subtile gm trows tcols tr tc))]
= admit() // gpu_matrix_subtile_base gm trows tcols tr tc

val cell_convert_eq
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {0 < trows /\ trows /? rows})
  (tcols : erased nat {0 < tcols /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  (f : perm)
  (v : et)
: Lemma (
  gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) #f i j v
  ==
  gpu_matrix_pts_to_cell gm #f (tr * trows + i) (tc * tcols + j) v
)

ghost
fn subcell_to_cell
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {0 < trows /\ trows /? rows})
  (tcols : erased nat {0 < tcols /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  (#f : perm)
  (#v : et)
  requires
    gpu_matrix_pts_to_cell gm #f (tr * trows + i) (tc * tcols + j) v
  ensures
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) #f i j v

ghost
fn cell_to_subcell
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {0 < trows /\ trows /? rows})
  (tcols : erased nat {0 < tcols /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  (#f : perm)
  (#v : et)
  requires
    gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) #f i j v
  ensures
    gpu_matrix_pts_to_cell gm #f (tr * trows + i) (tc * tcols + j) v

ghost
fn gpu_matrix_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)

ghost
fn gpu_matrix_untile'
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tf : natlt (rows / trows) -> natlt (cols / tcols) -> ematrix et trows tcols)
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
      (gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (tf tr tc))
  ensures
    gm |-> Frac f (ematrix_from_tiles trows tcols tf)

ghost
fn gpu_matrix_untile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
  ensures
    gm |-> Frac f em

ghost
fn gpu_matrix_untile_underspec
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        (exists* (em : ematrix et trows tcols).
          gpu_matrix_subtile gm trows tcols tr tc |-> Frac f em)
  ensures
    (exists* (em : ematrix et rows cols). gm |-> Frac f em)

ghost
fn gpu_matrix_extract_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc) **
    (forall* (tm' : ematrix et trows tcols).
      gpu_matrix_subtile gm trows tcols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm'))

inline_for_extraction noextract
fn gpu_matrix_extract_tile_st
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat { 0 < trows /\ trows /? rows })
  (tcols : erased nat { 0 < tcols /\ tcols /? cols })
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns tc_tile : gpu_matrix et (subtile_layout l trows tcols tr tc)
  // use rewrites_to?
  ensures pure (tc_tile == gpu_matrix_subtile gm trows tcols tr tc)
  ensures
    tc_tile |-> Frac f (ematrix_subtile em trows tcols tr tc) **
    (forall* (tm' : ematrix et trows tcols).
      tc_tile |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm'))

ghost
fn gpu_matrix_extract_tile_ro
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : nat { 0 < trows /\ trows /? rows })
  (tcols : nat { 0 < tcols /\ tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    factored
      (gpu_matrix_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)

inline_for_extraction noextract
fn gpu_matrix_extract_tile_ro'
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {0 < trows /\ trows /? rows })
  (tcols : erased nat {0 < tcols /\ tcols /? cols })
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns gm' : gpu_matrix et (subtile_layout l trows tcols tr tc)
  ensures
    rewrites_to gm' (gpu_matrix_subtile gm trows tcols tr tc) **
    factored
      (gm' |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)

(* Explode a matrix into tiled per-cell ownership.
   Combines explode + factor + subcell_to_cell in one step.

   Input: gm |-> em (full matrix ownership)
   Output: forall+ tr tc i j. subtile_cell(tr, tc, i, j) with value macc em (tr*trows+i) (tc*tcols+j)
*)
ghost
fn gpu_matrix_explode_tiled
  (#et : Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  requires
    gm |-> em
  ensures
    forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
            (i : natlt trows) (j : natlt tcols).
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
        (macc em (tr * trows + i) (tc * tcols + j))

(* Implode a tiled per-cell ownership back to full matrix.
   Reverse of gpu_matrix_explode_tiled.

   Input: forall+ tr tc i j. subtile_cell(tr, tc, i, j) with value val_fn(tr, tc, i, j)
   Output: gm |-> em' where macc em' (tr*trows+i) (tc*tcols+j) == val_fn(tr, tc, i, j)
*)
ghost
fn gpu_matrix_implode_tiled
  (#et : Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (val_fn : natlt (rows / trows) -> natlt (cols / tcols) -> natlt trows -> natlt tcols -> GTot et)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
            (i : natlt trows) (j : natlt tcols).
      gpu_matrix_pts_to_cell (gpu_matrix_subtile gm trows tcols tr tc) i j
        (val_fn tr tc i j)
  ensures
    gm |-> mkM (fun (row : natlt rows) (col : natlt cols) ->
      val_fn (row / trows) (col / tcols) (row % trows) (col % tcols))
