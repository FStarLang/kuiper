module Kuiper.Tensor.Tiling
#lang-pulse

(* An API for tiling matrices, implemented with array views. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Injection
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout
open Kuiper.Tensor
open Kuiper.Shape
open Pulse.Lib.Trade { (@==>) }
module SZ = Kuiper.SizeT

include Kuiper.EMatrix.Tiling

(* Linear bound on tiled indices, so SMT can show tr * trows + i < rows without nonlinear reasoning.
   No SMTPat here (it would leak into every client); Tiling.fst adds a local pattern wrapper. *)
#push-options "--fuel 0 --ifuel 0"
let tile_row_bound (rows : nat) (trows : pos) (tr : nat)
  : Lemma (requires trows /? rows /\ tr < rows / trows)
          (ensures 0 <= tr * trows /\ tr * trows + trows <= rows)
  = Kuiper.Divides.lemma_divides_mod trows rows;
    FStar.Math.Lemmas.lemma_div_exact rows trows;
    FStar.Math.Lemmas.lemma_mult_le_right trows (tr + 1) (rows / trows);
    FStar.Math.Lemmas.distributivity_add_left tr 1 trows;
    FStar.Math.Lemmas.nat_times_nat_is_nat tr trows;
    FStar.Math.Lemmas.swap_mul trows (rows / trows)

(* Linear bound on tile coordinates, so SMT can show row / trows < rows / trows. *)
let tile_coord_bound (rows : nat) (trows : pos) (row : nat)
  : Lemma (requires trows /? rows /\ row < rows)
          (ensures 0 <= row / trows /\ row / trows < rows / trows)
  = Kuiper.Divides.lemma_divides_mod trows rows;
    FStar.Math.Lemmas.lemma_div_exact rows trows;
    FStar.Math.Lemmas.lemma_div_le row rows trows;
    FStar.Math.Lemmas.nat_over_pos_is_nat row trows;
    FStar.Math.Lemmas.euclidean_division_definition row trows;
    FStar.Math.Lemmas.swap_mul trows (rows / trows);
    if row / trows = rows / trows then
      FStar.Math.Lemmas.lemma_mod_lt row trows
#pop-options

let tile_inj_f
  (#rows #cols : nat)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : abs ((trows @| tcols @| INil)) -> abs ((rows @| cols @| INil))
=
   tile_row_bound rows trows tr;
   tile_row_bound cols tcols tc;
   (fun (i, (j, ())) -> (tr * trows + i, (tc * tcols + j, ())))

(* Injectivity of tile_inj_f, proved by explicit destructuring. *)
#push-options "--using_facts_from '* -Kuiper.Divides.lemma_divides_product'"
let tile_inj_f_inj
  (#rows #cols : nat)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (x y : abs ((trows @| tcols @| INil)))
  : Lemma (requires tile_inj_f trows tcols tr tc x == tile_inj_f trows tcols tr tc y)
          (ensures x == y)
  = let (i1, (j1, ())) = x <: natlt trows & (natlt tcols & unit) in
    let (i2, (j2, ())) = y <: natlt trows & (natlt tcols & unit) in
    assert (tile_inj_f trows tcols tr tc x == (tr * trows + i1, (tc * tcols + j1, ())));
    assert (tile_inj_f trows tcols tr tc y == (tr * trows + i2, (tc * tcols + j2, ())));
    assert (i1 == i2 /\ j1 == j2)
#pop-options

let tile_inj
  (#rows #cols : nat)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : (abs ((trows @| tcols @| INil)) @~> abs ((rows @| cols @| INil)))
= {
   f      = tile_inj_f trows tcols tr tc;
   is_inj = (fun x y -> tile_inj_f_inj trows tcols tr tc x y);
}

let subtile_layout
  (#rows #cols : nat)
  (l : layout2 rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : layout2 trows tcols =
  {
    ulen = l.ulen;
    imap = inj_comp (tile_inj trows tcols tr tc) l.imap;
  }

inline_for_extraction noextract
instance val c_subtile_layout
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| cc : ctlayout l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {tr < rows / trows})
  (tc    : erased int {tc < cols / tcols})
  {| concrete_sz trows, concrete_sz tcols, concrete_sz tr, concrete_sz tc |}
  : ctlayout (subtile_layout l trows tcols tr tc)

inline_for_extraction noextract
val array2_subtile
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  : Tot (array2 et (subtile_layout l trows tcols tr tc))

val array2_subtile_base
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  : Lemma (
      core (array2_subtile gm trows tcols tr tc)
      ==
      core gm
    )
    [SMTPat (core (array2_subtile gm trows tcols tr tc))]

let is_array2_subtile_global
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (#gm : array2 et l)
  (#trows : erased nat {trows > 0 /\ trows /? rows})
  (#tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (#tr : enatlt (rows / trows))
  (#tc : enatlt (cols / tcols))
: Lemma
  (ensures
    is_global (array2_subtile gm trows tcols tr tc) <==>
    is_global gm)
  [SMTPat (is_global (array2_subtile gm trows tcols tr tc))]
= array2_subtile_base gm trows tcols tr tc

(* Kuiper.Divides.lemma_divides_product causes a matching loop in this signature's VCs. *)
#push-options "--using_facts_from '* -Kuiper.Divides.lemma_divides_product'"
val cell_convert_eq
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  (f : perm)
  (v : et)
  : Lemma (
    tensor_pts_to_cell (array2_subtile gm trows tcols tr tc) #f (idx2 i j) v
    ==
    tensor_pts_to_cell gm #f (idx2 (tr * trows + i) (tc * tcols + j)) v
  )
#pop-options

ghost
fn array2_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures array_exists (core gm)
  ensures
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)

ghost
fn array2_untile_with_exists'
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tf : natlt (rows / trows) -> natlt (cols / tcols) -> chest2 et trows tcols)
  (#f : perm)
  requires array_exists (core gm)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
      (array2_subtile gm trows tcols tr tc |-> Frac f (tf tr tc))
  ensures
    gm |-> Frac f (ematrix_from_tiles trows tcols tf)

ghost
fn array2_untile'
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tf : natlt (rows / trows) -> natlt (cols / tcols) -> chest2 et trows tcols)
  (#f : perm)
  requires pure (nonempty (abs ((rows / trows) @| (cols / tcols) @| INil)))
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
      (array2_subtile gm trows tcols tr tc |-> Frac f (tf tr tc))
  ensures
    gm |-> Frac f (ematrix_from_tiles trows tcols tf)

ghost
fn array2_untile_with_exists
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : chest2 et rows cols)
  (#f : perm)
  requires array_exists (core gm)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
  ensures
    gm |-> Frac f em

ghost
fn array2_untile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : chest2 et rows cols)
  (#f : perm)
  requires pure (nonempty (abs ((rows / trows) @| (cols / tcols) @| INil)))
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
  ensures
    gm |-> Frac f em

ghost
fn array2_untile_underspec_with_exists
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#f : perm)
  requires array_exists (core gm)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        (exists* (em : chest2 et trows tcols).
          array2_subtile gm trows tcols tr tc |-> Frac f em)
  ensures
    (exists* (em : chest2 et rows cols). gm |-> Frac f em)

ghost
fn array2_untile_underspec
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#f : perm)
  requires pure (nonempty (abs ((rows / trows) @| (cols / tcols) @| INil)))
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        (exists* (em : chest2 et trows tcols).
          array2_subtile gm trows tcols tr tc |-> Frac f em)
  ensures
    (exists* (em : chest2 et rows cols). gm |-> Frac f em)

ghost
fn array2_extract_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc) **
    (forall* (tm' : chest2 et trows tcols).
      array2_subtile gm trows tcols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm'))

inline_for_extraction noextract
fn array2_extract_tile_st
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : erased nat { trows > 0 /\ trows /? rows })
  (tcols : erased nat { tcols > 0 /\ tcols /? cols })
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns tc_tile : array2 et (subtile_layout l trows tcols tr tc)
  // use rewrites_to?
  ensures pure (tc_tile == array2_subtile gm trows tcols tr tc)
  ensures
    tc_tile |-> Frac f (ematrix_subtile em trows tcols tr tc) **
    (forall* (tm' : chest2 et trows tcols).
      tc_tile |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm'))

ghost
fn array2_extract_tile_ro
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : nat { trows > 0 /\ trows /? rows })
  (tcols : nat { tcols > 0 /\ tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    factored
      (array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)

inline_for_extraction noextract
fn array2_extract_tile_ro'
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : erased nat {trows > 0 /\ trows /? rows })
  (tcols : erased nat {tcols > 0 /\ tcols /? cols })
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns gm' : array2 et (subtile_layout l trows tcols tr tc)
  ensures
    rewrites_to gm' (array2_subtile gm trows tcols tr tc) **
    factored
      (gm' |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)

(* Kuiper.Divides.lemma_divides_product causes a matching loop in these signatures' VCs. *)
#push-options "--using_facts_from '* -Kuiper.Divides.lemma_divides_product'"
(* Explode a matrix into tiled per-cell ownership.
   Combines explode + factor + subcell_to_cell in one step.

   Input: gm |-> em (full matrix ownership)
   Output: forall+ tr tc i j. subtile_cell(tr, tc, i, j) with value acc2 em (tr*trows+i) (tc*tcols+j)
*)
ghost
fn array2_explode_tiled
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : chest2 et rows cols)
  requires
    gm |-> em
  ensures array_exists (core gm)
  ensures
    forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
            (i : natlt trows) (j : natlt tcols).
      tensor_pts_to_cell (array2_subtile gm trows tcols tr tc) (idx2 i j)
        (acc2 em (tr * trows + i) (tc * tcols + j))

(* Implode a tiled per-cell ownership back to full matrix.
   Reverse of array2_explode_tiled.

   Input: forall+ tr tc i j. subtile_cell(tr, tc, i, j) with value val_fn(tr, tc, i, j)
   Output: gm |-> em' where acc2 em' (tr*trows+i) (tc*tcols+j) == val_fn(tr, tc, i, j)
*)
ghost
fn array2_implode_tiled
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (val_fn : natlt (rows / trows) -> natlt (cols / tcols) -> natlt trows -> natlt tcols -> GTot et)
  requires array_exists (core gm)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
            (i : natlt trows) (j : natlt tcols).
      tensor_pts_to_cell (array2_subtile gm trows tcols tr tc) (idx2 i j)
        (val_fn tr tc i j)
  ensures
    gm |-> mk2 (fun (row : natlt rows) (col : natlt cols) ->
      val_fn (row / trows) (col / tcols) (row % trows) (col % tcols))
#pop-options
