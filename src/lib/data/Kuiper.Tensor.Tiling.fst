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
open Kuiper.Shape
open Pulse.Lib.Trade { (@==>) }
module M = Kuiper.Array2
module SZ = Kuiper.SizeT

include Kuiper.EMatrix.Tiling

#push-options "--split_queries always --z3rlimit 20"
inline_for_extraction noextract
let c_subtile_layout
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

#push-options "--z3rlimit 80 --split_queries always"
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
  M.ilower gm;
  forevery_factor_2 rows (rows / trows) trows
    cols (cols / tcols) tcols
    _;
  forevery_mid_flip _;
  ghost
  fn aux (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
    requires
      forall+ (i : natlt trows) (j : natlt tcols).
        M.pts_to_cell gm #f ((tr * trows + i <: natlt rows), (tc * tcols + j <: natlt cols)) (macc em (tr * trows + i) (tc * tcols + j))
    ensures
      array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em trows tcols tr tc)
  {
    forevery_map_2
      (fun (i:natlt trows) (j:natlt tcols) ->
        M.pts_to_cell gm #f ((tr * trows + i <: natlt rows), (tc * tcols + j <: natlt cols))
          (macc em (tr * trows + i) (tc * tcols + j)))
      (fun (i:natlt trows) (j:natlt tcols) ->
        M.pts_to_cell (array2_subtile gm trows tcols tr tc) #f (i, j)
          (macc (ematrix_subtile em trows tcols tr tc) i j))
      fn i j {
        cell_convert_eq gm trows tcols tr tc i j f
          (macc em (tr * trows + i) (tc * tcols + j));
        rewrite
          M.pts_to_cell gm #f ((tr * trows + i <: natlt rows), (tc * tcols + j <: natlt cols))
            (macc em (tr * trows + i) (tc * tcols + j))
        as
          M.pts_to_cell (array2_subtile gm trows tcols tr tc) #f (i, j)
            (macc (ematrix_subtile em trows tcols tr tc) i j);
      };
    M.iraise (array2_subtile gm trows tcols tr tc);
  };
  forevery_map_2 _ _ aux;
}
#pop-options

#push-options "--z3rlimit 40 --split_queries always"
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
  let em = ematrix_from_tiles trows tcols tf;
  ghost
  fn aux (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
    requires
      array2_subtile gm trows tcols tr tc |-> Frac f (tf tr tc)
    ensures
      forall+ (i : natlt trows) (j : natlt tcols).
        M.pts_to_cell gm #f ((tr * trows + i <: natlt rows), (tc * tcols + j <: natlt cols))
          (macc em (tr * trows + i) (tc * tcols + j))
  {
    M.ilower (array2_subtile gm trows tcols tr tc);
    forevery_map_2
      (fun (i:natlt trows) (j:natlt tcols) ->
        M.pts_to_cell (array2_subtile gm trows tcols tr tc) #f (i, j)
          (macc (tf tr tc) i j))
      (fun (i:natlt trows) (j:natlt tcols) ->
        M.pts_to_cell gm #f ((tr * trows + i <: natlt rows), (tc * tcols + j <: natlt cols))
          (macc em (tr * trows + i) (tc * tcols + j)))
      fn i j {
        cell_convert_eq gm trows tcols tr tc i j f (macc (tf tr tc) i j);
        assert pure ((tr * trows + i) / trows == tr);
        assert pure ((tc * tcols + j) / tcols == tc);
        assert pure ((tr * trows + i) % trows == i);
        assert pure ((tc * tcols + j) % tcols == j);
        assert pure (macc em (tr * trows + i) (tc * tcols + j) == macc (tf tr tc) i j);
        rewrite
          M.pts_to_cell (array2_subtile gm trows tcols tr tc) #f (i, j)
            (macc (tf tr tc) i j)
        as
          M.pts_to_cell gm #f ((tr * trows + i <: natlt rows), (tc * tcols + j <: natlt cols))
            (macc em (tr * trows + i) (tc * tcols + j));
      };
  };
  forevery_map_2 _ _ aux;
  forevery_mid_flip _;
  forevery_unfactor_2 rows (rows / trows) trows
    cols (cols / tcols) tcols
    (fun i j -> M.pts_to_cell gm #f (i, j) (macc em i j));
  M.iraise gm;
}
#pop-options

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
  array2_untile' gm trows tcols _;
  from_subtiles_id em trows tcols;
  rewrite each ematrix_from_tiles trows tcols (ematrix_subtile em trows tcols)
            as em;
}

#push-options "--z3rlimit 40 --split_queries always"
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
  M.pts_to_ref gm;
  array2_tile gm trows tcols;
  forevery_flatten _;
  forevery_remove _ (tr, tc);
  ghost
  fn aux (tm' : ematrix et trows tcols)
    requires
      forall+
        (tr'tc' : natlt (rows / trows) & natlt (cols / tcols) { tr'tc' =!= (tr, tc) } ).
          array2_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em trows tcols (fst tr'tc') (snd tr'tc'))
    ensures
      array2_subtile gm trows tcols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_tile em trows tcols tr tc tm')
  {
    let em' = update_tile em trows tcols tr tc tm';
    assert pure (forall (tc' : natlt (cols / tcols)) (tr' : natlt (rows / trows)).
      tc =!= tc' \/ tr =!= tr' ==>
        (ematrix_subtile em trows tcols tr' tc'
         ==
         ematrix_subtile em' trows tcols tr' tc')
    );
    forevery_ext
      (fun (tr'tc' : natlt (rows / trows) & natlt (cols / tcols) { tr'tc' =!= (tr, tc) } ) ->
        array2_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em trows tcols (fst tr'tc') (snd tr'tc')))
      (fun (tr'tc' : natlt (rows / trows) & natlt (cols / tcols) { tr'tc' =!= (tr, tc) } ) ->
        array2_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em' trows tcols (fst tr'tc') (snd tr'tc')));
    ghost
    fn aux ()
      requires
        forall+
        (tr'tc' : natlt (rows / trows) & natlt (cols / tcols) { tr'tc' =!= (tr, tc) } ).
          array2_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em' trows tcols (fst tr'tc') (snd tr'tc'))
      requires
        array2_subtile gm trows tcols tr tc |-> Frac f tm'
      ensures
        gm |-> Frac f (update_tile em trows tcols tr tc tm')
    {
      assert pure (ematrix_subtile em' trows tcols tr tc == tm');
      rewrite
        array2_subtile gm trows tcols tr tc |-> Frac f tm'
      as
        array2_subtile gm trows tcols tr tc |-> Frac f (ematrix_subtile em' trows tcols tr tc);
      forevery_insert
        #(natlt (rows / trows) & natlt (cols / tcols))
        #(fun tr'tc' -> tr'tc' =!= (tr, tc))
        (fun (tr'tc' : natlt (rows / trows) & natlt (cols / tcols)) ->
          array2_subtile gm trows tcols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_subtile em' trows tcols (fst tr'tc') (snd tr'tc')))
        (tr, tc);
      forevery_unrefine _;
      forevery_unflatten' _;
      array2_untile gm trows tcols #em';
      ()
    };
    Pulse.Lib.Trade.intro_trade _ _ _ aux;
  };
  Pulse.Lib.Forall.intro_forall _ aux;
}
#pop-options

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
  array2_extract_tile gm trows tcols tr tc;
  Pulse.Lib.Forall.elim_forall (ematrix_subtile em trows tcols tr tc);
  rewrite each (update_tile em trows tcols tr tc (ematrix_subtile em trows tcols tr tc))
    as em;
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
  array2_tile gm trows tcols;
  ghost
  fn aux (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
    requires
      array2_subtile gm trows tcols tr tc |-> ematrix_subtile em trows tcols tr tc
    ensures
      forall+ (i : natlt trows) (j : natlt tcols).
        M.pts_to_cell (array2_subtile gm trows tcols tr tc) (i, j)
          (macc em (tr * trows + i) (tc * tcols + j))
  {
    M.ilower (array2_subtile gm trows tcols tr tc);
    forevery_ext_2
      (fun (i:natlt trows) (j:natlt tcols) ->
        M.pts_to_cell (array2_subtile gm trows tcols tr tc) (i, j)
          (macc (ematrix_subtile em trows tcols tr tc) i j))
      (fun (i:natlt trows) (j:natlt tcols) ->
        M.pts_to_cell (array2_subtile gm trows tcols tr tc) (i, j)
          (macc em (tr * trows + i) (tc * tcols + j)));
  };
  forevery_map_2 _ _ aux;
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
  let em' = mkM (fun (row : natlt rows) (col : natlt cols) ->
    val_fn (row / trows) (col / tcols) (row % trows) (col % tcols));

  ghost
  fn aux (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
    requires
      forall+ (i : natlt trows) (j : natlt tcols).
        M.pts_to_cell (array2_subtile gm trows tcols tr tc) (i, j)
          (val_fn tr tc i j)
    ensures
      array2_subtile gm trows tcols tr tc |-> ematrix_subtile em' trows tcols tr tc
  {
    assert pure (forall (i:natlt trows) (j:natlt tcols). (tr * trows + i) / trows == tr);
    assert pure (forall (i:natlt trows) (j:natlt tcols). (tc * tcols + j) / tcols == tc);
    assert pure (forall (i:natlt trows) (j:natlt tcols). (tr * trows + i) % trows == i);
    assert pure (forall (i:natlt trows) (j:natlt tcols). (tc * tcols + j) % tcols == j);
    forevery_ext_2
      (fun (i:natlt trows) (j:natlt tcols) ->
        M.pts_to_cell (array2_subtile gm trows tcols tr tc) (i, j)
          (val_fn tr tc i j))
      (fun (i:natlt trows) (j:natlt tcols) ->
        M.pts_to_cell (array2_subtile gm trows tcols tr tc) (i, j)
          (macc (ematrix_subtile em' trows tcols tr tc) i j));
    M.iraise (array2_subtile gm trows tcols tr tc);
  };
  forevery_map_2 _ _ aux;
  array2_untile gm trows tcols;
}
