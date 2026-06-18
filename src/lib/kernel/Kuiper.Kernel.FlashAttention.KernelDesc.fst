module Kuiper.Kernel.FlashAttention.KernelDesc

(* ─────────────────────────────────────────────────────────────────────────
   kernel_desc and host launch.

   Configuration: 1 block, [br] threads.  We require [bc == br] so that
   every thread id [tid : szlt br] is also valid as [szlt bc] (the
   [flashattention_kf_no_smem] function refines [tid] by both bounds).

   Per-thread resources (in [kpre tid]):
     - [gK |-> Frac (fK /. br) eK]                         (sharded)
     - [gV |-> Frac (fV /. br) eV]                         (sharded)
     - [gQ |-> Frac (fQ /. br) eQ]                         (sharded)
     - per-thread strided sub-views [gOt, glt, gmt]
       (rows {i*br + tid for i = 0..n/br}).
     - row [tid] of the shmem S matrix, viewed as the [array1] [gSt].

   The strided extraction of (n/br) rows starting at offset tid with
   stride br is non-trivial; the ghost proofs are stubbed with [admit()]
   for now. The host wrapper, [flashattention_launch], wires everything
   into [launch_sync].
   ───────────────────────────────────────────────────────────────────── *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Array
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Tiling
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, c_l2_row_major }

module M = Kuiper.Array2 
module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.SHMem
open Kuiper.Index
open Pulse.Lib.Trade { (@==>) }

inline_for_extraction noextract
instance c_stride_subtile_layout
  (#rows #cols : erased nat)
  (l : M.layout rows cols)
  {| cc : ctlayout l |}
  (srows : erased int {0 < srows /\ srows /? rows})
  (scols : erased int {0 < scols /\ scols /? cols})
  (tr    : erased int {tr < srows})
  (tc    : erased int {tc < scols})
  {| concrete_sz srows, concrete_sz scols, concrete_sz tr, concrete_sz tc |}
  : ctlayout (stride_subtile_layout l srows scols tr tc)
  = {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (x : conc (M.desc (rows/srows) (cols/scols))) ->
                match x with | (i, (j, ())) ->
                let x' =
                  (concr tr +^ i *^ concr srows,
                   (concr tc +^ j *^ concr scols,
                    ())) in
                cc.cimap x');
  }

(* ─────────────────────────────────────────────────────────────────────────
   STRIDE TILE API.  Mirrors [Kuiper.Tensor.Tiling], but each tile is a
   _strided_ sub-view: tile (tr, tc) gathers the elements at original
   indices (i * srows + tr, j * scols + tc).  The proofs are identical to
   the contiguous case except for the index arithmetic and the quantifier
   reordering needed to bring (tr, tc) to the front of the factored
   ownership.
   ───────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
let array2_stride_subtile
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  : Tot (M.array2 et (stride_subtile_layout l srows scols tr tc))
  = M.from_array (stride_subtile_layout l srows scols tr tc) (M.core gm)

let array2_stride_subtile_base
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  : Lemma (
      M.core (array2_stride_subtile gm srows scols tr tc)
      ==
      M.core gm
    )
    [SMTPat (M.core (array2_stride_subtile gm srows scols tr tc))]
  = ()

let stride_cell_convert_eq
  (#et : _)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  (i : natlt (rows / srows))
  (j : natlt (cols / scols))
  (f : perm)
  (v : et)
  : Lemma (
    M.pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (i, j) v
    ==
    M.pts_to_cell gm #f (i * srows + tr, j * scols + tc) v
  )
  = M.pts_to_cell_eq gm (i * srows + tr, j * scols + tc) f v;
    M.pts_to_cell_eq (array2_stride_subtile gm srows scols tr tc) (i, j) f v;
    ()

(* ── EMatrix-level lemmas for the stride decomposition ─────────────────── *)

let lem_stride_eucl (s:pos) (q:nat) (r:nat{r < s})
  : Lemma ((q * s + r) % s == r /\ (q * s + r) / s == q)
          [SMTPatOr [[SMTPat ((q * s + r) % s)]; [SMTPat ((q * s + r) / s)]]]
  = FStar.Math.Lemmas.lemma_mod_plus r q s;
    FStar.Math.Lemmas.lemma_div_plus r q s

#push-options "--z3rlimit 20"
let from_stride_subtiles_id
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  : Lemma (ematrix_stride_from_tiles srows scols (ematrix_stride_subtile em srows scols)
           ==
           em)
          [SMTPat (ematrix_stride_from_tiles srows scols (ematrix_stride_subtile em srows scols))]
= assert (equal (ematrix_stride_from_tiles srows scols (ematrix_stride_subtile em srows scols)) em);
  ()
#pop-options

let update_stride_tile_self
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  (tr : natlt srows)
  (tc : natlt scols)
  : Lemma (update_stride_tile em srows scols tr tc (ematrix_stride_subtile em srows scols tr tc)
           ==
           em)
          [SMTPat (update_stride_tile em srows scols tr tc (ematrix_stride_subtile em srows scols tr tc))]
= assert (equal (update_stride_tile em srows scols tr tc (ematrix_stride_subtile em srows scols tr tc)) em)

#push-options "--split_queries always --z3rlimit 20"
let subtile_of_update_stride_tile
  (#et : _)
  (#rows #cols : _)
  (em : ematrix et rows cols)
  (srows : pos {srows /? rows})
  (scols : pos {scols /? cols})
  (tr : natlt srows)
  (tc : natlt scols)
  (etile : ematrix et (rows/srows) (cols/scols))
  (tr' : natlt srows)
  (tc' : natlt scols)
  : Lemma (ematrix_stride_subtile (update_stride_tile em srows scols tr tc etile) srows scols tr' tc'
           ==
           (if tr = tr' && tc = tc' then etile else ematrix_stride_subtile em srows scols tr' tc'))
          [SMTPat (ematrix_stride_subtile (update_stride_tile em srows scols tr tc etile) srows scols tr' tc')]
  = if tr' = tr && tc' = tc then
      assert (equal (ematrix_stride_subtile (update_stride_tile em srows scols tr tc etile) srows scols tr' tc') etile)
    else
      assert (equal (ematrix_stride_subtile (update_stride_tile em srows scols tr tc etile) srows scols tr' tc') (ematrix_stride_subtile em srows scols tr' tc'))
#pop-options

(* ── array2-level ghost reshuffles ─────────────────────────────────────── *)

#push-options "--z3rlimit 80 --split_queries always"
ghost
fn array2_stride_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    forall+
      (tr : natlt srows)
      (tc : natlt scols).
        array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc)
{
  M.ilower gm;
  forevery_factor_2 rows (rows / srows) srows
    cols (cols / scols) scols
    _;
  // order: (i, tr, j, tc); bring inner (j, tc) -> (tc, j)
  forevery_map_2
    (fun (i:natlt (rows / srows)) (tr:natlt srows) ->
      forall+ (j:natlt (cols / scols)) (tc:natlt scols).
        M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
    (fun (i:natlt (rows / srows)) (tr:natlt srows) ->
     forall+ (tc:natlt scols) (j:natlt (cols / scols)).
       M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
         (macc em (i * srows + tr) (j * scols + tc)))
    fn i tr {
      forevery_commute _;
    };
  // order: (i, tr, tc, j)
  forevery_commute _;
  // order: (tr, i, tc, j)
  forevery_mid_flip _;
  // order: (tr, tc, i, j)
  ghost
  fn aux (tr : natlt srows) (tc : natlt scols)
    requires
      forall+ (i : natlt (rows / srows)) (j : natlt (cols / scols)).
        M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc))
    ensures
      array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc)
  {
    forevery_map_2
      (fun (i:natlt (rows / srows)) (j:natlt (cols / scols)) ->
        M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
      (fun (i:natlt (rows / srows)) (j:natlt (cols / scols)) ->
        M.pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (i, j)
          (macc (ematrix_stride_subtile em srows scols tr tc) i j))
      fn i j {
        stride_cell_convert_eq gm srows scols tr tc i j f
          (macc em (i * srows + tr) (j * scols + tc));
        rewrite
          M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
            (macc em (i * srows + tr) (j * scols + tc))
        as
          M.pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (i, j)
            (macc (ematrix_stride_subtile em srows scols tr tc) i j);
      };
    M.iraise (array2_stride_subtile gm srows scols tr tc);
  };
  forevery_map_2 _ _ aux;
}
#pop-options

#push-options "--z3rlimit 40 --split_queries always"
ghost
fn array2_stride_untile'
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (tf : natlt srows -> natlt scols -> ematrix et (rows/srows) (cols/scols))
  (#f : perm)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+
      (tr : natlt srows)
      (tc : natlt scols).
      (array2_stride_subtile gm srows scols tr tc |-> Frac f (tf tr tc))
  ensures
    gm |-> Frac f (ematrix_stride_from_tiles srows scols tf)
{
  let em = ematrix_stride_from_tiles srows scols tf;
  ghost
  fn aux (tr : natlt srows) (tc : natlt scols)
    requires
      array2_stride_subtile gm srows scols tr tc |-> Frac f (tf tr tc)
    ensures
      forall+ (i : natlt (rows / srows)) (j : natlt (cols / scols)).
        M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc))
  {
    M.ilower (array2_stride_subtile gm srows scols tr tc);
    forevery_map_2
      (fun (i:natlt (rows / srows)) (j:natlt (cols / scols)) ->
        M.pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (i, j)
          (macc (tf tr tc) i j))
      (fun (i:natlt (rows / srows)) (j:natlt (cols / scols)) ->
        M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
      fn i j {
        stride_cell_convert_eq gm srows scols tr tc i j f (macc (tf tr tc) i j);
        assert pure ((i * srows + tr) % srows == tr);
        assert pure ((j * scols + tc) % scols == tc);
        assert pure ((i * srows + tr) / srows == i);
        assert pure ((j * scols + tc) / scols == j);
        assert pure (macc em (i * srows + tr) (j * scols + tc) == macc (tf tr tc) i j);
        rewrite
          M.pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (i, j)
            (macc (tf tr tc) i j)
        as
          M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
            (macc em (i * srows + tr) (j * scols + tc));
      };
  };
  forevery_map_2 _ _ aux;
  // order: (tr, tc, i, j); rearrange to (i, tr, j, tc)
  forevery_mid_flip _;
  // order: (tr, i, tc, j)
  forevery_commute _;
  // order: (i, tr, tc, j); swap inner (tc, j) -> (j, tc)
  forevery_map_2
    (fun (i:natlt (rows / srows)) (tr:natlt srows) ->
      forall+ (tc:natlt scols) (j:natlt (cols / scols)).
        M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
    (fun (i:natlt (rows / srows)) (tr:natlt srows) ->
      forall+ (j:natlt (cols / scols)) (tc:natlt scols).
        M.pts_to_cell gm #f ((i * srows + tr <: natlt rows), (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
    fn i tr {
       forevery_commute _;
    };
  // order: (i, tr, j, tc)
  forevery_unfactor_2 rows (rows / srows) srows
    cols (cols / scols) scols
    (fun i j -> M.pts_to_cell gm #f (i, j) (macc em i j));
  M.iraise gm;
}
#pop-options

ghost
fn array2_stride_untile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    pure (SZ.fits (M.layout_size l))
  requires
    forall+
      (tr : natlt srows)
      (tc : natlt scols).
        array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc)
  ensures
    gm |-> Frac f em
{
  array2_stride_untile' gm srows scols _;
  from_stride_subtiles_id em srows scols;
  rewrite each ematrix_stride_from_tiles srows scols (ematrix_stride_subtile em srows scols)
            as em;
}

#push-options "--z3rlimit 40 --split_queries always"
ghost
fn array2_extract_stride_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (tr : natlt srows)
  (tc : natlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc) **
    (forall* (tm' : ematrix et (rows/srows) (cols/scols)).
      array2_stride_subtile gm srows scols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_stride_tile em srows scols tr tc tm'))
{
  M.pts_to_ref gm;
  array2_stride_tile gm srows scols;
  forevery_flatten _;
  forevery_remove _ (tr, tc);
  ghost
  fn aux (tm' : ematrix et (rows/srows) (cols/scols))
    requires
      forall+
        (tr'tc' : natlt srows & natlt scols { tr'tc' =!= (tr, tc) } ).
          array2_stride_subtile gm srows scols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_stride_subtile em srows scols (fst tr'tc') (snd tr'tc'))
    ensures
      array2_stride_subtile gm srows scols tr tc |-> Frac f tm' @==>
      gm |-> Frac f (update_stride_tile em srows scols tr tc tm')
  {
    let em' = update_stride_tile em srows scols tr tc tm';
    assert pure (forall (tc' : natlt scols) (tr' : natlt srows).
      tc =!= tc' \/ tr =!= tr' ==>
        (ematrix_stride_subtile em srows scols tr' tc'
         ==
         ematrix_stride_subtile em' srows scols tr' tc')
    );
    forevery_ext
      (fun (tr'tc' : natlt srows & natlt scols { tr'tc' =!= (tr, tc) } ) ->
        array2_stride_subtile gm srows scols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_stride_subtile em srows scols (fst tr'tc') (snd tr'tc')))
      (fun (tr'tc' : natlt srows & natlt scols { tr'tc' =!= (tr, tc) } ) ->
        array2_stride_subtile gm srows scols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_stride_subtile em' srows scols (fst tr'tc') (snd tr'tc')));
    ghost
    fn aux ()
      requires
        forall+
        (tr'tc' : natlt srows & natlt scols { tr'tc' =!= (tr, tc) } ).
          array2_stride_subtile gm srows scols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_stride_subtile em' srows scols (fst tr'tc') (snd tr'tc'))
      requires
        array2_stride_subtile gm srows scols tr tc |-> Frac f tm'
      ensures
        gm |-> Frac f (update_stride_tile em srows scols tr tc tm')
    {
      assert pure (ematrix_stride_subtile em' srows scols tr tc == tm');
      rewrite
        array2_stride_subtile gm srows scols tr tc |-> Frac f tm'
      as
        array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em' srows scols tr tc);
      forevery_insert
        #(natlt srows & natlt scols)
        #(fun tr'tc' -> tr'tc' =!= (tr, tc))
        (fun (tr'tc' : natlt srows & natlt scols) ->
          array2_stride_subtile gm srows scols (fst tr'tc') (snd tr'tc') |-> Frac f (ematrix_stride_subtile em' srows scols (fst tr'tc') (snd tr'tc')))
        (tr, tc);
      forevery_unrefine _;
      forevery_unflatten' _;
      array2_stride_untile gm srows scols #em';
      ()
    };
    Pulse.Lib.Trade.intro_trade _ _ _ aux;
  };
  Pulse.Lib.Forall.intro_forall _ aux;
}
#pop-options

inline_for_extraction noextract
fn array2_extract_stride_tile_st
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat { srows > 0 /\ srows /? rows })
  (scols : erased nat { scols > 0 /\ scols /? cols })
  (tr : enatlt srows)
  (tc : enatlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns tc_tile : M.array2 et (stride_subtile_layout l srows scols tr tc)
  ensures pure (tc_tile == array2_stride_subtile gm srows scols tr tc)
  ensures
    tc_tile |-> Frac f (ematrix_stride_subtile em srows scols tr tc) **
    (forall* (tm' : ematrix et (rows/srows) (cols/scols)).
      tc_tile |-> Frac f tm' @==>
      gm |-> Frac f (update_stride_tile em srows scols tr tc tm'))
{
  array2_extract_stride_tile gm srows scols tr tc;
  array2_stride_subtile gm srows scols tr tc;
}

ghost
fn array2_extract_stride_tile_ro
  (#et:Type0)
  (#rows #cols : nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : nat { srows > 0 /\ srows /? rows })
  (scols : nat { scols > 0 /\ scols /? cols })
  (tr : natlt srows)
  (tc : natlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  ensures
    factored
      (array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc))
      (gm |-> Frac f em)
{
  array2_extract_stride_tile gm srows scols tr tc;
  Pulse.Lib.Forall.elim_forall (ematrix_stride_subtile em srows scols tr tc);
  rewrite each (update_stride_tile em srows scols tr tc (ematrix_stride_subtile em srows scols tr tc))
    as em;
}

inline_for_extraction noextract
fn array2_extract_stride_tile_ro'
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : M.layout rows cols)
  (gm : M.array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows })
  (scols : erased nat {scols > 0 /\ scols /? cols })
  (tr : enatlt srows)
  (tc : enatlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns gm' : M.array2 et (stride_subtile_layout l srows scols tr tc)
  ensures
    rewrites_to gm' (array2_stride_subtile gm srows scols tr tc) **
    factored
      (gm' |-> Frac f (ematrix_stride_subtile em srows scols tr tc))
      (gm |-> Frac f em)
{
  array2_extract_stride_tile_ro gm srows scols tr tc;
  array2_stride_subtile gm srows scols tr tc;
}


(* The shmem layout we use for S: a single [SHArray et (bc * br)] viewed
   as a [br * bc] row-major matrix, so that row [tid] is the per-thread
   scratch slice expected by [flashattention_kf_no_smem]. *)

inline_for_extraction noextract
let fa_shmems (et : Type0) {| Kuiper.Sized.sized et |}
  (bc br : szp { SZ.fits (bc * br) })
  : list shmem_desc
  = [SHArray et (bc *^ br)]

(* The Array2 layout we use to view the S shmem array. *)
inline_for_extraction noextract
let fa_lS (bc br : szp) : M.layout br bc = l2_row_major br bc

inline_for_extraction noextract
instance fa_lS_ct (bc br : szp { SZ.fits (bc * br) })
  : ctlayout (fa_lS bc br) = c_l2_row_major _ _

(* Lift the raw shmem array to an Array2 view. *)
inline_for_extraction noextract
let fa_gS
  (#et : Type0) {| Kuiper.Sized.sized et |}
  (bc br : szp { SZ.fits (bc * br) })
  (sh : c_shmems (fa_shmems et bc br))
  : M.array2 et (fa_lS bc br)
  = M.from_array (fa_lS bc br) sh._1
(*


(* Outer setup/teardown for the full kernel_desc. nblk = 1, so the
   [forall+ bid : natlt 1. block_pre bid] is just [block_pre 0]. *)
// FOR NOW: assume 
// - 1 block
// - gS is in global instead of shared memory, and no barriers
// - br = bc = nthreads
ghost
fn setup_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (#lS: M.layout nthr nthr)
  (#lK #lV #lQ #lO: M.layout n d)
  (#ll #lm: layout n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS : M.array2 et lS { M.is_global gS })
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (gO : M.array2 et lO { M.is_global gO })
  (gl : array1 et ll { Array1.is_global gl })
  (gm : array1 et lm { Array1.is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  norewrite
  requires
    (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
    live gS ** live gO ** live gl ** live gm
  ensures
    (forall+ (tid : natlt nthr).

      (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
      (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm)) **
    emp
{
  forevery_singleton_intro #(natlt 1) (fun _bid ->
    (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
    (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm));
}

ghost
fn teardown_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (#lO : M.layout n d)
  (gO : M.array2 et lO { M.is_global gO })
  (#ll #lm : layout n)
  (gl : array1 et ll { Array1.is_global gl })
  (gm : array1 et lm { Array1.is_global gm })
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  ()
  norewrite
  requires
    (forall+ (_bid : natlt 1).
      (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
      (exists* (eO' : ematrix et n d). gO |-> eO') **
      (exists* (vl' : lseq et n). gl |-> vl') **
      (exists* (vm' : lseq et n). gm |-> vm')) **
    emp
  ensures
    (gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
    (exists* (eO' : ematrix et n d). gO |-> eO') **
    (exists* (vl' : lseq et n). gl |-> vl') **
    (exists* (vm' : lseq et n). gm |-> vm')
{
  forevery_singleton_elim #(natlt 1) _;
}

(* Block-level setup/teardown: split shmem S matrix into per-thread rows;
   shard read-only perms; explode write-side matrices into per-thread
   strided sub-tiles; bundle into [kpre_fa tid]. *)
ghost
fn block_setup_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (#lO : M.layout n d)
  (gO : M.array2 et lO { M.is_global gO })
  (#ll #lm : layout n)
  (gl : array1 et ll { Array1.is_global gl })
  (gm : array1 et lm { Array1.is_global gm })
  (eK eV eQ eO : ematrix et n d)
  (vl vm : erased (lseq et n))
  (fK fV fQ : perm)
  (sh : c_shmems (fa_shmems et bc br))
  (_bid : natlt 1)
  ()
  norewrite
  requires
    live_c_shmems sh **
    ((gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
     (gO |-> eO) ** (gl |-> vl) ** (gm |-> vm))
  ensures
    (forall+ (tid : natlt br).
       kpre_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh tid) **
    emp
{
  admit ()
}

ghost
fn block_teardown_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d bc br : szp { bc == br /\ bc /? n /\ br /? n /\ SZ.fits (bc * br) })
  (#lK #lV #lQ : M.layout n d)
  (gK : M.array2 et lK { M.is_global gK })
  (gV : M.array2 et lV { M.is_global gV })
  (gQ : M.array2 et lQ { M.is_global gQ })
  (#lO : M.layout n d)
  (gO : M.array2 et lO { M.is_global gO })
  (#ll #lm : layout n)
  (gl : array1 et ll { Array1.is_global gl })
  (gm : array1 et lm { Array1.is_global gm })
  (eK eV eQ : ematrix et n d)
  (fK fV fQ : perm)
  (sh : c_shmems (fa_shmems et bc br))
  (_bid : natlt 1)
  ()
  norewrite
  requires
    (forall+ (tid : natlt br).
       kpost_fa n d bc br gK gV gQ gO gl gm eK eV eQ fK fV fQ sh tid) **
    emp
  ensures
    live_c_shmems sh **
    ((gK |-> Frac fK eK) ** (gV |-> Frac fV eV) ** (gQ |-> Frac fQ eQ) **
     (exists* (eO' : ematrix et n d). gO |-> eO') **
     (exists* (vl' : lseq et n). gl |-> vl') **
     (exists* (vm' : lseq et n). gm |-> vm'))
{
  admit ()
}
*)