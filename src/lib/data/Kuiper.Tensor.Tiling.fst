module Kuiper.Tensor.Tiling
#lang-pulse

(* An API for tiling matrices, implemented with array views. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Injection
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout
open Kuiper.Tensor
open Kuiper.Array2
open Kuiper.Index
open Pulse.Lib.Trade { (@==>) }
module M = Kuiper.Array2
module SZ = Kuiper.SizeT

include Kuiper.EMatrix.Tiling

#restart-solver
#push-options "--split_queries always --z3rlimit 20"
inline_for_extraction noextract
let c_subtile_layout2
  (#rows #cols : erased nat)
  (l : M.layout rows cols)
  {| cc : ctlayout l |}
  (trows : erased int {0 < trows /\ trows /? rows})
  (tcols : erased int {0 < tcols /\ tcols /? cols})
  (tr    : erased int {tr < rows / trows})
  (tc    : erased int {tc < cols / tcols})
  {| concrete_sz trows, concrete_sz tcols, concrete_sz tr, concrete_sz tc |}
  : ctlayout (subtile_layout l trows tcols tr tc)
  = {
      ulen_fits = ();
      all_fit = ();
      cimap = (fun (x : conc (desc trows tcols)) ->
                match x with | (i, (j, ())) ->
                let x' =
                  (concr trows *^ concr tr +^ i,
                   (concr tcols *^ concr tc +^ j,
                    ())) in
                cc.cimap x');
  }
#pop-options

(* FIXME. The definition above works just fine. But, if we remove this
indirection (and presumably have the val on the fsti ascribe it), the
function fails to verify. *)
inline_for_extraction noextract
let c_subtile_layout = c_subtile_layout2

inline_for_extraction noextract
let array2_subtile
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  : Tot (array2 et (subtile_layout l trows tcols tr tc))
  = M.from_array (subtile_layout l trows tcols tr tc) (M.core gm)

let array2_subtile_base
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
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
  = ()

let cell_convert_eq
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
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
    M.pts_to_cell (array2_subtile gm trows tcols tr tc) #f (i, j) v
    ==
    M.pts_to_cell gm #f (tr * trows + i, tc * tcols + j) v
  )
  = M.pts_to_cell_eq gm (tr * trows + i, tc * tcols + j) f v;
    M.pts_to_cell_eq (array2_subtile gm trows tcols tr tc) (i, j) f v;
    ()

ghost
fn array2_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
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
        array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
{
  admit();
}

ghost
fn array2_untile'
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tf : natlt (rows / trows) -> natlt (cols / tcols) -> ematrix et trows tcols)
  (#f : perm)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
      (array2_subtile gm trows tcols tr tc |-> Frac f (tf tr tc))
  ensures
    gm |-> Frac f (ematrix_from_tiles trows tcols tf)
{
  admit();
}

ghost
fn array2_untile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
  ensures
    gm |-> Frac f em
{
  admit();
}

ghost
fn array2_extract_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc) **
    (forall* (tm' : ematrix et trows tcols).
      array2_subtile gm trows tcols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm'))
{
  admit();
}

inline_for_extraction noextract
fn array2_extract_tile_st
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : erased nat { trows > 0 /\ trows /? rows })
  (tcols : erased nat { tcols > 0 /\ tcols /? cols })
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns tc_tile : array2 et (subtile_layout l trows tcols tr tc)
  // use rewrites_to?
  ensures pure (tc_tile == array2_subtile gm trows tcols tr tc)
  ensures
    tc_tile |-> Frac f (ematrix_subtile em trows tcols tr tc) **
    (forall* (tm' : ematrix et trows tcols).
      tc_tile |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm'))
{
  array2_extract_tile gm trows tcols tr tc;
  array2_subtile gm trows tcols tr tc;
}

ghost
fn array2_extract_tile_ro
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : nat { trows > 0 /\ trows /? rows })
  (tcols : nat { tcols > 0 /\ tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    factored
      (array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)
{
  admit();
}

inline_for_extraction noextract
fn array2_extract_tile_ro'
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : erased nat {trows > 0 /\ trows /? rows })
  (tcols : erased nat {tcols > 0 /\ tcols /? cols })
  (tr : enatlt (rows / trows))
  (tc : enatlt (cols / tcols))
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns gm' : array2 et (subtile_layout l trows tcols tr tc)
  ensures
    rewrites_to gm' (array2_subtile gm trows tcols tr tc) **
    factored
      (gm' |-> Frac f (ematrix_subtile em trows tcols tr tc))
      (gm |-> Frac f em)
{
  array2_extract_tile_ro gm trows tcols tr tc;
  array2_subtile gm trows tcols tr tc;
}

(* Explode a matrix into tiled per-cell ownership.
   Combines explode + factor + subcell_to_cell in one step.

   Input: gm |-> em (full matrix ownership)
   Output: forall+ tr tc i j. subtile_cell(tr, tc, i, j) with value macc em (tr*trows+i) (tc*tcols+j)
*)
ghost
fn array2_explode_tiled
  (#et : Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  requires
    gm |-> em
  ensures
    forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
            (i : natlt trows) (j : natlt tcols).
      M.pts_to_cell (array2_subtile gm trows tcols tr tc) (i, j)
        (macc em (tr * trows + i) (tc * tcols + j))
{
  admit();
}

(* Implode a tiled per-cell ownership back to full matrix.
   Reverse of array2_explode_tiled.

   Input: forall+ tr tc i j. subtile_cell(tr, tc, i, j) with value val_fn(tr, tc, i, j)
   Output: gm |-> em' where macc em' (tr*trows+i) (tc*tcols+j) == val_fn(tr, tc, i, j)
*)
ghost
fn array2_implode_tiled
  (#et : Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (val_fn : natlt (rows / trows) -> natlt (cols / tcols) -> natlt trows -> natlt tcols -> GTot et)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
            (i : natlt trows) (j : natlt tcols).
      M.pts_to_cell (array2_subtile gm trows tcols tr tc) (i, j)
        (val_fn tr tc i j)
  ensures
    gm |-> mkM (fun (row : natlt rows) (col : natlt cols) ->
      val_fn (row / trows) (col / tcols) (row % trows) (col % tcols))
{
  admit();
}

#push-options "--z3rlimit 40"
(* Combinator for teardown of approximate kernels.
   Collects per-cell existentials into a matrix-level existential,
   and transforms per-cell approximation facts into matrix-level approximation.

   Usage: After separating gA/gB, call this on the remaining forall+ of cells
   to get a single existential matrix that approximates the spec.

   TODO: this needs a thourough review. *)
ghost
fn array2_collect_approx_tiled
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (ntr : nat { ntr == rows / trows })
  (ntc : nat { ntc == cols / tcols })
  (spec_fn : natlt rows -> natlt cols -> et -> prop)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+ (bid : natlt (ntr * ntc)) (tid : natlt (trows * tcols)).
      exists* (v : et).
        M.pts_to_cell
          (array2_subtile gm trows tcols (bid / ntc) (bid % ntc))
          ((tid / tcols <: natlt trows), (tid % tcols <: natlt tcols)) v **
        pure (spec_fn ((bid / ntc) * trows + (tid / tcols))
                      ((bid % ntc) * tcols + (tid % tcols)) v)
  returns vf : (natlt (ntr * ntc) -> natlt (trows * tcols) -> GTot et)
  ensures
    gm |-> mkM (fun (row : natlt rows) (col : natlt cols) ->
      vf ((row / trows) * ntc + (col / tcols)) ((row % trows) * tcols + (col % tcols))) **
    pure (forall (row : natlt rows) (col : natlt cols).
      spec_fn row col
        (vf ((row / trows) * ntc + (col / tcols)) ((row % trows) * tcols + (col % tcols))))
{
  admit();
}
#pop-options
