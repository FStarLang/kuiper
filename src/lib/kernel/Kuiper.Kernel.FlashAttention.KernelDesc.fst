module Kuiper.Kernel.FlashAttention.KernelDesc

(* ─────────────────────────────────────────────────────────────────────────
   kernel_desc setup / teardown for FlashAttention.

   Configuration: 1 block, [nthr] threads, [bc == br == nthr].  No shared
   memory, no barriers.

   Per-thread resources (in [kpre_post_outer_fa tid]):
     - [gK |-> Frac (fK /. nthr) eK]                      (sharded, read-only)
     - [gV |-> Frac (fV /. nthr) eV]                      (sharded, read-only)
     - [gQ |-> Frac (fQ /. nthr) eQ]                      (sharded, read-only)
     - row [tid] of gS (contiguous)                       (scratch)
     - strided rows {i*nthr + tid} of gO                  (output)
     - strided columns {j*nthr + tid} of gl, gm           (stats)

   [setup_fa] / [teardown_fa] split / reassemble the full global resources
   into these per-thread strided sub-views.
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

module SZ = Kuiper.SizeT
module Trade = Pulse.Lib.Trade
module B = Kuiper.Barrier
open Kuiper.Shape
open Pulse.Lib.Trade { (@==>) }
open Kuiper.Math { even, odd }

(* ── array2-over-tensor cell / row helpers (old Array2 shims) ──────────── *)

ghost
fn milower
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      tensor_pts_to_cell a #f (idx2 r c) (macc s r c))
{
  tensor_ilower2 a;
}

ghost
fn miraise
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (#f : perm) (#s : ematrix et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      tensor_pts_to_cell a #f (idx2 r c) (macc s r c))
  ensures
    a |-> Frac f s
{
  tensor_iraise2 a;
}

let mrow
  (#et : Type0) (#rows #cols : erased nat) (#l : layout2 rows cols)
  (a : array2 et l) (i : erased nat{i < rows})
  : array1 et (mrow_layout a i)
  = sliceof a 0 i

ghost
fn mextract_row
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (i : natlt rows)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    mrow a i |-> Frac f (tr_val (ematrix_row s i)) **
    (forall* (s' : lseq et cols).
      mrow a i |-> Frac f (tr_val s') @==>
      a |-> Frac f (ematrix_upd_row s i s'))
{
  tensor_extract_slice a 0 i #f #s;

  assert pure (Kuiper.Chest.equal
    (chest_slice 0 i s)
    (tr_val (ematrix_row s i)));
  rewrite sliceof a 0 i |-> Frac f (chest_slice 0 i s)
       as mrow a i |-> Frac f (tr_val (ematrix_row s i));

  Pulse.Lib.Forall.intro_forall
    #_
    #(fun (s' : lseq et cols) ->
      mrow a i |-> Frac f (tr_val s')
      @==> a |-> Frac f (ematrix_upd_row s i s'))
    (forall* (s' : chest (modulo_i 0 (mdesc rows cols)) et).
      sliceof a 0 i |-> Frac f s'
      @==> a |-> Frac f (chest_update_slice 0 i s s'))
    fn s' {
      Pulse.Lib.Trade.intro_trade
        (mrow a i |-> Frac f (tr_val s'))
        (a |-> Frac f (ematrix_upd_row s i s'))
        (forall* (s' : chest (modulo_i 0 (mdesc rows cols)) et).
              sliceof a 0 i |-> Frac f s'
              @==> a |-> Frac f (chest_update_slice 0 i s s'))
        fn _ {
          assert pure (modulo_i 0 (mdesc rows cols) == cols @| INil);
          let w : chest (modulo_i 0 (mdesc rows cols)) et = tr_val s';
          Pulse.Lib.Forall.elim_forall w;
          rewrite mrow a i |-> Frac f (tr_val s')
               as sliceof a 0 i |-> Frac f w;
          Pulse.Lib.Trade.elim_trade _ _;
          assert pure (Kuiper.Chest.equal
            (chest_update_slice 0 i s w)
            (ematrix_upd_row s i s'));
          rewrite each chest_update_slice 0 i s w
               as ematrix_upd_row s i s';
          ();
        };
    };
  ();
}

ghost
fn mextract_row_ro
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (i : natlt rows)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    a |-> Frac f s
  ensures
    factored
      (mrow a i |-> Frac f (tr_val (ematrix_row s i)))
      (a |-> Frac f s)
{
  mextract_row a i;
  Pulse.Lib.Forall.elim_forall (ematrix_row s i);
  assert pure (Kuiper.Chest.equal
    (ematrix_upd_row s i (ematrix_row s i))
    s);
}

ghost
fn mrestore_row
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (i : natlt rows)
  (#f : perm) (#s : ematrix et rows cols)
  requires
    factored
      (mrow a i |-> Frac f (tr_val (ematrix_row s i)))
      (a |-> Frac f s)
  ensures
    a |-> Frac f s
{
  Pulse.Lib.Trade.elim_trade _ _;
}

inline_for_extraction noextract
instance c_stride_subtile_layout
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
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
    cimap = (fun (x : conc ((rows/srows) @| (cols/scols) @| INil)) ->
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
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  : Tot (array2 et (stride_subtile_layout l srows scols tr tc))
  = from_array (stride_subtile_layout l srows scols tr tc) (core gm)

let array2_stride_subtile_base
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  : Lemma (
      core (array2_stride_subtile gm srows scols tr tc)
      ==
      core gm
    )
    [SMTPat (core (array2_stride_subtile gm srows scols tr tc))]
  = ()

let stride_cell_convert_eq
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows})
  (scols : erased nat {scols > 0 /\ scols /? cols})
  (tr : enatlt srows)
  (tc : enatlt scols)
  (i : natlt (rows / srows))
  (j : natlt (cols / scols))
  (f : perm)
  (v : et)
  : Lemma (
    tensor_pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (idx2 i j) v
    ==
    tensor_pts_to_cell gm #f (idx2 (i * srows + tr) (j * scols + tc)) v
  )
  = tensor_pts_to_cell_eq gm (idx2 ((i * srows + tr) <: natlt rows) ((j * scols + tc) <: natlt cols)) f v;
    tensor_pts_to_cell_eq (array2_stride_subtile gm srows scols tr tc) (idx2 i j) f v;
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
  (#l : layout2 rows cols)
  (gm : array2 et l)
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
  milower gm;
  forevery_factor_2 rows (rows / srows) srows
    cols (cols / scols) scols
    _;
  // order: (i, tr, j, tc); bring inner (j, tc) -> (tc, j)
  forevery_map_2
    (fun (i:natlt (rows / srows)) (tr:natlt srows) ->
      forall+ (j:natlt (cols / scols)) (tc:natlt scols).
        tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
    (fun (i:natlt (rows / srows)) (tr:natlt srows) ->
     forall+ (tc:natlt scols) (j:natlt (cols / scols)).
       tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
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
        tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc))
    ensures
      array2_stride_subtile gm srows scols tr tc |-> Frac f (ematrix_stride_subtile em srows scols tr tc)
  {
    forevery_map_2
      (fun (i:natlt (rows / srows)) (j:natlt (cols / scols)) ->
        tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
      (fun (i:natlt (rows / srows)) (j:natlt (cols / scols)) ->
        tensor_pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (idx2 i j)
          (macc (ematrix_stride_subtile em srows scols tr tc) i j))
      fn i j {
        stride_cell_convert_eq gm srows scols tr tc i j f
          (macc em (i * srows + tr) (j * scols + tc));
        rewrite
          tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
            (macc em (i * srows + tr) (j * scols + tc))
        as
          tensor_pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (idx2 i j)
            (macc (ematrix_stride_subtile em srows scols tr tc) i j);
      };
    miraise (array2_stride_subtile gm srows scols tr tc);
  };
  forevery_map_2 _ _ aux;
}
#pop-options

#push-options "--z3rlimit 40 --split_queries always"
ghost
fn array2_stride_untile'
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (tf : natlt srows -> natlt scols -> ematrix et (rows/srows) (cols/scols))
  (#f : perm)
  requires
    pure (SZ.fits (tlayout_ulen l))
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
        tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc))
  {
    milower (array2_stride_subtile gm srows scols tr tc);
    forevery_map_2
      (fun (i:natlt (rows / srows)) (j:natlt (cols / scols)) ->
        tensor_pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (idx2 i j)
          (macc (tf tr tc) i j))
      (fun (i:natlt (rows / srows)) (j:natlt (cols / scols)) ->
        tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
      fn i j {
        stride_cell_convert_eq gm srows scols tr tc i j f (macc (tf tr tc) i j);
        assert pure ((i * srows + tr) % srows == tr);
        assert pure ((j * scols + tc) % scols == tc);
        assert pure ((i * srows + tr) / srows == i);
        assert pure ((j * scols + tc) / scols == j);
        assert pure (macc em (i * srows + tr) (j * scols + tc) == macc (tf tr tc) i j);
        rewrite
          tensor_pts_to_cell (array2_stride_subtile gm srows scols tr tc) #f (idx2 i j)
            (macc (tf tr tc) i j)
        as
          tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
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
        tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
    (fun (i:natlt (rows / srows)) (tr:natlt srows) ->
      forall+ (j:natlt (cols / scols)) (tc:natlt scols).
        tensor_pts_to_cell gm #f (idx2 (i * srows + tr <: natlt rows) (j * scols + tc <: natlt cols))
          (macc em (i * srows + tr) (j * scols + tc)))
    fn i tr {
       forevery_commute _;
    };
  // order: (i, tr, j, tc)
  forevery_unfactor_2 rows (rows / srows) srows
    cols (cols / scols) scols
    (fun i j -> tensor_pts_to_cell gm #f (idx2 i j) (macc em i j));
  miraise gm;
}
#pop-options

ghost
fn array2_stride_untile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (srows : pos { srows /? rows })
  (scols : pos { scols /? cols })
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    pure (SZ.fits (tlayout_ulen l))
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
  (#l : layout2 rows cols)
  (gm : array2 et l)
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
  tensor_pts_to_ref gm;
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
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (srows : erased nat { srows > 0 /\ srows /? rows })
  (scols : erased nat { scols > 0 /\ scols /? cols })
  (tr : enatlt srows)
  (tc : enatlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns tc_tile : array2 et (stride_subtile_layout l srows scols tr tc)
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
  (#l : layout2 rows cols)
  (gm : array2 et l)
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
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (srows : erased nat {srows > 0 /\ srows /? rows })
  (scols : erased nat {scols > 0 /\ scols /? cols })
  (tr : enatlt srows)
  (tc : enatlt scols)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em
  returns gm' : array2 et (stride_subtile_layout l srows scols tr tc)
  ensures
    rewrites_to gm' (array2_stride_subtile gm srows scols tr tc) **
    factored
      (gm' |-> Frac f (ematrix_stride_subtile em srows scols tr tc))
      (gm |-> Frac f em)
{
  array2_extract_stride_tile_ro gm srows scols tr tc;
  array2_stride_subtile gm srows scols tr tc;
}



(* ─────────────────────────────────────────────────────────────────────────
   kernel_desc setup / teardown.

   1 block, [nthr] threads, [bc == br == nthr].  We split the full global
   resources into the per-thread strided sub-views of [kpre_post_outer_fa]:
     - gK/gV/gQ : fractionally shared (read only) among the [nthr] threads;
     - gS       : per-thread (contiguous) row [tid]              (scratch);
     - gO       : per-thread strided rows {i*nthr + tid}         (output);
     - gl/gm    : per-thread strided columns {j*nthr + tid}      (stats).
   ───────────────────────────────────────────────────────────────────── *)

(* Collapse a trailing singleton ([natlt 1]) forevery dimension. *)
ghost
fn collapse_inner1 (#nn:nat) (q : natlt nn -> natlt 1 -> slprop)
  requires forall+ (tr:natlt nn) (tc:natlt 1). q tr tc
  ensures  forall+ (tr:natlt nn). q tr 0
{
  forevery_map
    (fun (tr:natlt nn) -> forall+ (tc:natlt 1). q tr tc)
    (fun (tr:natlt nn) -> q tr 0)
    fn tr { forevery_singleton_elim #(natlt 1) (fun (tc:natlt 1) -> q tr tc); };
}

(* Introduce a trailing singleton ([natlt 1]) forevery dimension. *)
ghost
fn expand_inner1 (#nn:nat) (q : natlt nn -> natlt 1 -> slprop)
  requires forall+ (tr:natlt nn). q tr 0
  ensures  forall+ (tr:natlt nn) (tc:natlt 1). q tr tc
{
  forevery_map
    (fun (tr:natlt nn) -> q tr 0)
    (fun (tr:natlt nn) -> forall+ (tc:natlt 1). q tr tc)
    fn tr { forevery_singleton_intro #(natlt 1) (fun (tc:natlt 1) -> q tr tc); };
}

ghost
fn setup_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (#lS: layout2 nthr nthr)
  (#lK #lV #lQ #lO: layout2 n d)
  (#ll #lm: layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS : array2 et lS)
  (gK : array2 et lK { is_global gK })
  (gV : array2 et lV { is_global gV })
  (gQ : array2 et lQ { is_global gQ })
  (gO : array2 et lO { is_global gO })
  (gl : array2 et ll { is_global gl })
  (gm : array2 et lm { is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  ()
  norewrite
  requires
    full_io_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ
  ensures
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa n d nthr lS lO ll lm
{
  // K, V, Q : fractional sharing.
  tensor_share_n gK (SZ.v nthr);
  tensor_share_n gV (SZ.v nthr);
  tensor_share_n gQ (SZ.v nthr);

  // gS : contiguous row [tid].
  unfold (live gS); with eS. assert (gS |-> eS);
  array2_tile gS 1 (SZ.v nthr);
  forevery_rw_size2 (SZ.v nthr / 1) (SZ.v nthr) (SZ.v nthr / SZ.v nthr) 1
    #(fun (tr:natlt (SZ.v nthr / 1)) (tc:natlt (SZ.v nthr / SZ.v nthr)) ->
        array2_subtile gS 1 (SZ.v nthr) tr tc |-> Frac 1.0R (ematrix_subtile eS 1 (SZ.v nthr) tr tc));
  collapse_inner1 #(SZ.v nthr)
    (fun (tr:natlt (SZ.v nthr)) (tc:natlt 1) ->
        array2_subtile gS 1 (SZ.v nthr) tr tc |-> Frac 1.0R (ematrix_subtile eS 1 (SZ.v nthr) tr tc));

  // gO : strided rows {i*nthr + tid}.
  unfold (live gO); with eO. assert (gO |-> eO);
  array2_stride_tile gO (SZ.v nthr) 1;
  collapse_inner1 #(SZ.v nthr)
    (fun (tr:natlt (SZ.v nthr)) (tc:natlt 1) ->
        array2_stride_subtile gO (SZ.v nthr) 1 tr tc |-> Frac 1.0R (ematrix_stride_subtile eO (SZ.v nthr) 1 tr tc));

  // gl : strided columns {j*nthr + tid}.
  unfold (live gl); with el. assert (gl |-> el);
  array2_stride_tile gl 1 (SZ.v nthr);
  forevery_singleton_elim #(natlt 1)
    (fun (tr:natlt 1) -> forall+ (tc:natlt (SZ.v nthr)).
        array2_stride_subtile gl 1 (SZ.v nthr) tr tc |-> Frac 1.0R (ematrix_stride_subtile el 1 (SZ.v nthr) tr tc));

  // gm : strided columns {j*nthr + tid}.
  unfold (live gm); with em. assert (gm |-> em);
  array2_stride_tile gm 1 (SZ.v nthr);
  forevery_singleton_elim #(natlt 1)
    (fun (tr:natlt 1) -> forall+ (tc:natlt (SZ.v nthr)).
        array2_stride_subtile gm 1 (SZ.v nthr) tr tc |-> Frac 1.0R (ematrix_stride_subtile em 1 (SZ.v nthr) tr tc));

  // Bundle the 7 per-thread foreverys.
  forevery_zip3 #(natlt (SZ.v nthr))
    (fun (_:natlt (SZ.v nthr)) -> gK |-> Frac (fK /. (SZ.v nthr)) eK)
    (fun (_:natlt (SZ.v nthr)) -> gV |-> Frac (fV /. (SZ.v nthr)) eV)
    (fun (_:natlt (SZ.v nthr)) -> gQ |-> Frac (fQ /. (SZ.v nthr)) eQ);
  forevery_zip3 #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) -> array2_subtile gS 1 (SZ.v nthr) tid 0 |-> Frac 1.0R (ematrix_subtile eS 1 (SZ.v nthr) tid 0))
    (fun (tid:natlt (SZ.v nthr)) -> array2_stride_subtile gl 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile el 1 (SZ.v nthr) 0 tid))
    (fun (tid:natlt (SZ.v nthr)) -> array2_stride_subtile gm 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile em 1 (SZ.v nthr) 0 tid));
  forevery_zip3 #(natlt (SZ.v nthr))
    (fun (_:natlt (SZ.v nthr)) ->
       (gK |-> Frac (fK /. (SZ.v nthr)) eK) ** (gV |-> Frac (fV /. (SZ.v nthr)) eV) ** (gQ |-> Frac (fQ /. (SZ.v nthr)) eQ))
    (fun (tid:natlt (SZ.v nthr)) ->
       (array2_subtile gS 1 (SZ.v nthr) tid 0 |-> Frac 1.0R (ematrix_subtile eS 1 (SZ.v nthr) tid 0)) **
       (array2_stride_subtile gl 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile el 1 (SZ.v nthr) 0 tid)) **
       (array2_stride_subtile gm 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile em 1 (SZ.v nthr) 0 tid)))
    (fun (tid:natlt (SZ.v nthr)) -> array2_stride_subtile gO (SZ.v nthr) 1 tid 0 |-> Frac 1.0R (ematrix_stride_subtile eO (SZ.v nthr) 1 tid 0));

  forevery_map #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) ->
       ((gK |-> Frac (fK /. (SZ.v nthr)) eK) ** (gV |-> Frac (fV /. (SZ.v nthr)) eV) ** (gQ |-> Frac (fQ /. (SZ.v nthr)) eQ)) **
       ((array2_subtile gS 1 (SZ.v nthr) tid 0 |-> Frac 1.0R (ematrix_subtile eS 1 (SZ.v nthr) tid 0)) **
        (array2_stride_subtile gl 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile el 1 (SZ.v nthr) 0 tid)) **
        (array2_stride_subtile gm 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile em 1 (SZ.v nthr) 0 tid))) **
       (array2_stride_subtile gO (SZ.v nthr) 1 tid 0 |-> Frac 1.0R (ematrix_stride_subtile eO (SZ.v nthr) 1 tid 0)))
    (fun (tid:natlt (SZ.v nthr)) ->
       kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid)
    fn tid { () };
}

ghost
fn teardown_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (#lS: layout2 nthr nthr)
  (#lK #lV #lQ #lO: layout2 n d)
  (#ll #lm: layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gS : array2 et lS)
  (gK : array2 et lK { is_global gK })
  (gV : array2 et lV { is_global gV })
  (gQ : array2 et lQ { is_global gQ })
  (gO : array2 et lO { is_global gO })
  (gl : array2 et ll { is_global gl })
  (gm : array2 et lm { is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa n d nthr lS lO ll lm
  ensures
    full_io_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ
{
  // Split each tid back into the 7 separate pieces (4 single existentials).
  forevery_map #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) ->
       kpre_post_outer_fa n d nthr gS gK gV gQ gO gl gm eK eV eQ fK fV fQ tid)
    (fun (tid:natlt (SZ.v nthr)) ->
       ((gK |-> Frac (fK /. (SZ.v nthr)) eK) ** (gV |-> Frac (fV /. (SZ.v nthr)) eV) ** (gQ |-> Frac (fQ /. (SZ.v nthr)) eQ)) **
       (((exists* (eS:ematrix et (SZ.v nthr) (SZ.v nthr)). array2_subtile gS 1 (SZ.v nthr) tid 0 |-> Frac 1.0R (ematrix_subtile eS 1 (SZ.v nthr) tid 0)) **
         (exists* (el:ematrix et 1 n). array2_stride_subtile gl 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile el 1 (SZ.v nthr) 0 tid)) **
         (exists* (em:ematrix et 1 n). array2_stride_subtile gm 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile em 1 (SZ.v nthr) 0 tid))) **
        (exists* (eO:ematrix et n d). array2_stride_subtile gO (SZ.v nthr) 1 tid 0 |-> Frac 1.0R (ematrix_stride_subtile eO (SZ.v nthr) 1 tid 0))))
    fn tid { () };

  // Peel into K**V**Q, the S/l/m trio, and the O existential.
  forevery_unzip3 #(natlt (SZ.v nthr))
    (fun (_:natlt (SZ.v nthr)) ->
       (gK |-> Frac (fK /. (SZ.v nthr)) eK) ** (gV |-> Frac (fV /. (SZ.v nthr)) eV) ** (gQ |-> Frac (fQ /. (SZ.v nthr)) eQ))
    (fun (tid:natlt (SZ.v nthr)) ->
       (exists* (eS:ematrix et (SZ.v nthr) (SZ.v nthr)). array2_subtile gS 1 (SZ.v nthr) tid 0 |-> Frac 1.0R (ematrix_subtile eS 1 (SZ.v nthr) tid 0)) **
       (exists* (el:ematrix et 1 n). array2_stride_subtile gl 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile el 1 (SZ.v nthr) 0 tid)) **
       (exists* (em:ematrix et 1 n). array2_stride_subtile gm 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile em 1 (SZ.v nthr) 0 tid)))
    (fun (tid:natlt (SZ.v nthr)) ->
       exists* (eO:ematrix et n d). array2_stride_subtile gO (SZ.v nthr) 1 tid 0 |-> Frac 1.0R (ematrix_stride_subtile eO (SZ.v nthr) 1 tid 0));

  // K, V, Q : gather the fractional shares.
  forevery_unzip3 #(natlt (SZ.v nthr))
    (fun (_:natlt (SZ.v nthr)) -> gK |-> Frac (fK /. (SZ.v nthr)) eK)
    (fun (_:natlt (SZ.v nthr)) -> gV |-> Frac (fV /. (SZ.v nthr)) eV)
    (fun (_:natlt (SZ.v nthr)) -> gQ |-> Frac (fQ /. (SZ.v nthr)) eQ);
  tensor_gather_n gK (SZ.v nthr);
  tensor_gather_n gV (SZ.v nthr);
  tensor_gather_n gQ (SZ.v nthr);

  // Split the trio into the three separate existential foreverys.
  forevery_unzip3 #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) -> exists* (eS:ematrix et (SZ.v nthr) (SZ.v nthr)). array2_subtile gS 1 (SZ.v nthr) tid 0 |-> Frac 1.0R (ematrix_subtile eS 1 (SZ.v nthr) tid 0))
    (fun (tid:natlt (SZ.v nthr)) -> exists* (el:ematrix et 1 n). array2_stride_subtile gl 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile el 1 (SZ.v nthr) 0 tid))
    (fun (tid:natlt (SZ.v nthr)) -> exists* (em:ematrix et 1 n). array2_stride_subtile gm 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile em 1 (SZ.v nthr) 0 tid));

  // gS : contiguous row [tid] -> reassemble.
  let gsfun = forevery_exists
    (fun (tid:natlt (SZ.v nthr)) (e:ematrix et (SZ.v nthr) (SZ.v nthr)) ->
        array2_subtile gS 1 (SZ.v nthr) tid 0 |-> Frac 1.0R (ematrix_subtile e 1 (SZ.v nthr) tid 0));
  expand_inner1 #(SZ.v nthr)
    (fun (tr:natlt (SZ.v nthr)) (tc:natlt 1) ->
        array2_subtile gS 1 (SZ.v nthr) tr tc |-> Frac 1.0R (ematrix_subtile (gsfun tr) 1 (SZ.v nthr) tr tc));
  forevery_rw_size2 (SZ.v nthr) (SZ.v nthr / 1) 1 (SZ.v nthr / SZ.v nthr)
    #(fun (tr:natlt (SZ.v nthr)) (tc:natlt 1) ->
        array2_subtile gS 1 (SZ.v nthr) tr tc |-> Frac 1.0R (ematrix_subtile (gsfun tr) 1 (SZ.v nthr) tr tc));
  array2_untile' gS 1 (SZ.v nthr)
    (fun (tr:natlt (SZ.v nthr / 1)) (tc:natlt (SZ.v nthr / SZ.v nthr)) ->
        ematrix_subtile (gsfun tr) 1 (SZ.v nthr) tr tc) #1.0R;

  // gl : strided columns -> reassemble.
  let glfun = forevery_exists
    (fun (tid:natlt (SZ.v nthr)) (e:ematrix et 1 n) ->
        array2_stride_subtile gl 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile e 1 (SZ.v nthr) 0 tid));
  forevery_singleton_intro #(natlt 1)
    (fun (tr:natlt 1) -> forall+ (tc:natlt (SZ.v nthr)).
        array2_stride_subtile gl 1 (SZ.v nthr) tr tc |-> Frac 1.0R (ematrix_stride_subtile (glfun tc) 1 (SZ.v nthr) tr tc));
  array2_stride_untile' gl 1 (SZ.v nthr)
    (fun (tr:natlt 1) (tc:natlt (SZ.v nthr)) ->
        ematrix_stride_subtile (glfun tc) 1 (SZ.v nthr) tr tc) #1.0R;

  // gm : strided columns -> reassemble.
  let gmfun = forevery_exists
    (fun (tid:natlt (SZ.v nthr)) (e:ematrix et 1 n) ->
        array2_stride_subtile gm 1 (SZ.v nthr) 0 tid |-> Frac 1.0R (ematrix_stride_subtile e 1 (SZ.v nthr) 0 tid));
  forevery_singleton_intro #(natlt 1)
    (fun (tr:natlt 1) -> forall+ (tc:natlt (SZ.v nthr)).
        array2_stride_subtile gm 1 (SZ.v nthr) tr tc |-> Frac 1.0R (ematrix_stride_subtile (gmfun tc) 1 (SZ.v nthr) tr tc));
  array2_stride_untile' gm 1 (SZ.v nthr)
    (fun (tr:natlt 1) (tc:natlt (SZ.v nthr)) ->
        ematrix_stride_subtile (gmfun tc) 1 (SZ.v nthr) tr tc) #1.0R;

  // gO : strided rows -> reassemble.
  let gOfun = forevery_exists
    (fun (tid:natlt (SZ.v nthr)) (e:ematrix et n d) ->
        array2_stride_subtile gO (SZ.v nthr) 1 tid 0 |-> Frac 1.0R (ematrix_stride_subtile e (SZ.v nthr) 1 tid 0));
  expand_inner1 #(SZ.v nthr)
    (fun (tr:natlt (SZ.v nthr)) (tc:natlt 1) ->
        array2_stride_subtile gO (SZ.v nthr) 1 tr tc |-> Frac 1.0R (ematrix_stride_subtile (gOfun tr) (SZ.v nthr) 1 tr tc));
  array2_stride_untile' gO (SZ.v nthr) 1
    (fun (tr:natlt (SZ.v nthr)) (tc:natlt 1) ->
        ematrix_stride_subtile (gOfun tr) (SZ.v nthr) 1 tr tc) #1.0R;
}

(* ─────────────────────────────────────────────────────────────────────────
   Shared-memory block setup / teardown.

   The block owns a single flat shared array [fst sh : larray et (nthr*nthr)].
   We view it as the [nthr x nthr] gS scratch matrix ([gS_of_sh]) and then
   reuse the existing per-thread split/reassembly ([setup_fa]/[teardown_fa]).
   No threads communicate through gS (each owns a disjoint row), so no
   barriers are needed.
   ───────────────────────────────────────────────────────────────────── *)

ghost
fn block_setup_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (lS : full_layout2 nthr nthr)
  (#lK #lV #lQ #lO: layout2 n d)
  (#ll #lm: layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : array2 et lK { is_global gK })
  (gV : array2 et lV { is_global gV })
  (gQ : array2 et lQ { is_global gQ })
  (gO : array2 et lO { is_global gO })
  (gl : array2 et ll { is_global gl })
  (gm : array2 et lm { is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (sh : c_shmems (shmems_desc_fa et nthr))
  (bid : natlt 1sz)
  ()
  norewrite
  requires
    live_c_shmems sh **
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
  ensures
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa n d nthr lS lO ll lm
{
  // Expose the raw shared array.
  unfold_live_c_shmems_cons sh #1.0R;
  unfold_live_c_shmems_nil (snd sh) #1.0R;
  unfold_live_c_shmem (fst sh) #1.0R;
  gpu_pts_to_ref (fst sh);

  // View it as the gS matrix.
  tensor_abs' lS (fst sh);
  rewrite each (from_array lS (fst sh)) as (gS_of_sh n d nthr lS sh);

  // Reuse the existing per-thread split; [live (gS_of_sh ...)] together with
  // [full_io_fa_nos] matches [setup_fa]'s [full_io_fa] precondition.
  setup_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ ();
}

ghost
fn block_teardown_fa
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) })
  (lS : full_layout2 nthr nthr)
  (#lK #lV #lQ #lO: layout2 n d)
  (#ll #lm: layout2 1 n)
  {| ctlayout lS, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : array2 et lK { is_global gK })
  (gV : array2 et lV { is_global gV })
  (gQ : array2 et lQ { is_global gQ })
  (gO : array2 et lO { is_global gO })
  (gl : array2 et ll { is_global gl })
  (gm : array2 et lm { is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (sh : c_shmems (shmems_desc_fa et nthr))
  (bid : natlt 1sz)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa n d nthr lS lO ll lm
  ensures
    live_c_shmems sh **
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
{
  // Reuse the existing per-thread reassembly; this yields [full_io_fa] which
  // unfolds to [full_io_fa_nos ** live (gS_of_sh ...)].
  teardown_fa n d nthr (gS_of_sh n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ ();

  // Fold the gS matrix view back into the raw shared array.
  rewrite each (gS_of_sh n d nthr lS sh) as (from_array lS (fst sh));
  tensor_concr (from_array lS (fst sh));
  rewrite each (core (from_array lS (fst sh))) as (fst sh);
  fold_live_c_shmem (fst sh) #1.0R;
  fold_live_c_shmems_nil (snd sh) #1.0R;
  fold_live_c_shmems_cons sh #1.0R;
}

(* ═════════════════════════════════════════════════════════════════════════
   SHARED-MEMORY VARIANT (K, V, Q caching with a real barrier).
   ───────────────────────────────────────────────────────────────────────── *)

(* Content-free split of an [rows x cols] array into per-thread write-rows. *)
ghost
fn rows_split
  (#et:Type0)
  (#rows : nat)
  (#cols : nat { cols > 0 })
  (#l : layout2 rows cols)
  (a : array2 et l)
  requires
    exists* (e:ematrix et rows cols). a |-> e
  ensures
    forall+ (tid:natlt rows).
      exists* (r:ematrix et 1 cols). array2_subtile a 1 cols tid 0 |-> Frac 1.0R r
{
  with e. assert (a |-> e);
  array2_tile a 1 cols;
  forevery_rw_size2 (rows / 1) rows (cols / cols) 1
    #(fun (tr:natlt (rows / 1)) (tc:natlt (cols / cols)) ->
        array2_subtile a 1 cols tr tc |-> Frac 1.0R (ematrix_subtile e 1 cols tr tc));
  collapse_inner1 #rows
    (fun (tr:natlt rows) (tc:natlt 1) ->
        array2_subtile a 1 cols tr tc |-> Frac 1.0R (ematrix_subtile e 1 cols tr tc));
  forevery_map #(natlt rows)
    (fun (tid:natlt rows) -> array2_subtile a 1 cols tid 0 |-> Frac 1.0R (ematrix_subtile e 1 cols tid 0))
    (fun (tid:natlt rows) -> exists* (r:ematrix et 1 cols). array2_subtile a 1 cols tid 0 |-> Frac 1.0R r)
    fn tid { () };
}

(* Content-free reassembly of per-thread write-rows into the whole array. *)
ghost
fn rows_gather
  (#et:Type0)
  (#rows : nat)
  (#cols : nat { cols > 0 })
  (#l : layout2 rows cols)
  (a : array2 et l)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid:natlt rows).
      exists* (r:ematrix et 1 cols). array2_subtile a 1 cols tid 0 |-> Frac 1.0R r)
  ensures
    exists* (e:ematrix et rows cols). a |-> e
{
  let rf = forevery_exists
    (fun (tid:natlt rows) (r:ematrix et 1 cols) ->
        array2_subtile a 1 cols tid 0 |-> Frac 1.0R r);
  expand_inner1 #rows
    (fun (tr:natlt rows) (tc:natlt 1) ->
        array2_subtile a 1 cols tr tc |-> Frac 1.0R (rf tr));
  forevery_rw_size2 rows (rows / 1) 1 (cols / cols)
    #(fun (tr:natlt rows) (tc:natlt 1) ->
        array2_subtile a 1 cols tr tc |-> Frac 1.0R (rf tr));
  array2_untile' a 1 cols
    (fun (tr:natlt (rows / 1)) (tc:natlt (cols / cols)) -> rf tr) #1.0R;
}

#push-options "--z3rlimit 60 --fuel 0 --ifuel 0"
ghost
fn fa_barrier_ok
  (#et:Type0) {| scalar et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * d) })
  (#lKV : full_layout2 nthr d)
  {| ctlayout lKV |}
  (sK sV : array2 et lKV)
  (it : nat)
  requires
    forall+ (i:natlt (SZ.v nthr)). fa_barrier_side_rin n d nthr sK sV it i
  ensures
    forall+ (i:natlt (SZ.v nthr)). fa_barrier_side_rout n d nthr sK sV it i
{
  if (it >= 2 * SZ.v (n /^ nthr)) {
    forevery_map #(natlt (SZ.v nthr))
      (fun (i:natlt (SZ.v nthr)) -> fa_barrier_side_rin n d nthr sK sV it i)
      (fun (i:natlt (SZ.v nthr)) -> fa_barrier_side_rout n d nthr sK sV it i)
      fn i {
        rewrite (fa_barrier_side_rin n d nthr sK sV it i) as emp;
        rewrite emp as (fa_barrier_side_rout n d nthr sK sV it i);
      };
  } else {
    let ev = even it;
    if ev {
      assert pure (it < 2 * SZ.v (n /^ nthr));
      assert pure (even it);
      (* even: each thread gives back its row, receives a fractional read. *)
      forevery_map #(natlt (SZ.v nthr))
        (fun (i:natlt (SZ.v nthr)) -> fa_barrier_side_rin n d nthr sK sV it i)
        (fun (i:natlt (SZ.v nthr)) ->
          (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) i 0 |-> Frac 1.0R r) **
          (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) i 0 |-> Frac 1.0R r))
        fn i {
          rewrite (fa_barrier_side_rin n d nthr sK sV it i)
               as ((exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) i 0 |-> Frac 1.0R r) **
                   (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) i 0 |-> Frac 1.0R r));
        };
      forevery_unzip #(natlt (SZ.v nthr))
        (fun (i:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) i 0 |-> Frac 1.0R r)
        (fun (i:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) i 0 |-> Frac 1.0R r);
      rows_gather sK;
      rows_gather sV;
      with x. assert (sK |-> x);
      tensor_share_n sK (SZ.v nthr);
      with y. assert (sV |-> y);
      tensor_share_n sV (SZ.v nthr);
      forevery_map #(natlt (SZ.v nthr))
        (fun (_:natlt (SZ.v nthr)) -> sK |-> Frac (1.0R /. (SZ.v nthr)) x)
        (fun (_:natlt (SZ.v nthr)) -> exists* (x':ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x')
        fn i { () };
      forevery_map #(natlt (SZ.v nthr))
        (fun (_:natlt (SZ.v nthr)) -> sV |-> Frac (1.0R /. (SZ.v nthr)) y)
        (fun (_:natlt (SZ.v nthr)) -> exists* (y':ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y')
        fn i { () };
      forevery_zip #(natlt (SZ.v nthr))
        (fun (_:natlt (SZ.v nthr)) -> exists* (x':ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x')
        (fun (_:natlt (SZ.v nthr)) -> exists* (y':ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y');
      forevery_map #(natlt (SZ.v nthr))
        (fun (i:natlt (SZ.v nthr)) ->
          (exists* (x':ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x') **
          (exists* (y':ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y'))
        (fun (i:natlt (SZ.v nthr)) -> fa_barrier_side_rout n d nthr sK sV it i)
        fn i {
          rewrite ((exists* (x':ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x') **
                   (exists* (y':ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y'))
               as (fa_barrier_side_rout n d nthr sK sV it i);
        };
    } else {
      assert pure (it < 2 * SZ.v (n /^ nthr));
      assert pure (odd it);
      (* odd: each thread gives back its fractional read, receives its row. *)
      forevery_map #(natlt (SZ.v nthr))
        (fun (i:natlt (SZ.v nthr)) -> fa_barrier_side_rin n d nthr sK sV it i)
        (fun (i:natlt (SZ.v nthr)) ->
          (exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x) **
          (exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y))
        fn i {
          rewrite (fa_barrier_side_rin n d nthr sK sV it i)
               as ((exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x) **
                   (exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y));
        };
      forevery_unzip #(natlt (SZ.v nthr))
        (fun (i:natlt (SZ.v nthr)) -> exists* (x:ematrix et (SZ.v nthr) (SZ.v d)). sK |-> Frac (1.0R /. (SZ.v nthr)) x)
        (fun (i:natlt (SZ.v nthr)) -> exists* (y:ematrix et (SZ.v nthr) (SZ.v d)). sV |-> Frac (1.0R /. (SZ.v nthr)) y);
      tensor_gather_n_underspec sK (SZ.v nthr);
      tensor_gather_n_underspec sV (SZ.v nthr);
      rows_split sK;
      rows_split sV;
      forevery_zip #(natlt (SZ.v nthr))
        (fun (i:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) i 0 |-> Frac 1.0R r)
        (fun (i:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) i 0 |-> Frac 1.0R r);
      forevery_map #(natlt (SZ.v nthr))
        (fun (i:natlt (SZ.v nthr)) ->
          (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) i 0 |-> Frac 1.0R r) **
          (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) i 0 |-> Frac 1.0R r))
        (fun (i:natlt (SZ.v nthr)) -> fa_barrier_side_rout n d nthr sK sV it i)
        fn i {
          rewrite ((exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sK 1 (SZ.v d) i 0 |-> Frac 1.0R r) **
                   (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile sV 1 (SZ.v d) i 0 |-> Frac 1.0R r))
               as (fa_barrier_side_rout n d nthr sK sV it i);
        };
    }
  }
}
#pop-options

(* ─────────────────────────────────────────────────────────────────────────
   Shared-memory block setup / teardown (four shared arrays sK, sV, sQ, gS).
   We view the four flat shared arrays as the matrices, reuse [setup_fa] for
   the non-shared part (gS scratch + global strided sub-views) and split the
   K/V/Q caches into per-thread write-rows via [rows_split].
   ───────────────────────────────────────────────────────────────────────── *)

ghost
fn block_setup_fa_smem
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) /\ SZ.fits (nthr * d) })
  (lS : full_layout2 nthr nthr)
  (lKV : full_layout2 nthr d)
  (#lK #lV #lQ #lO: layout2 n d)
  (#ll #lm: layout2 1 n)
  {| ctlayout lS, ctlayout lKV, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : array2 et lK { is_global gK })
  (gV : array2 et lV { is_global gV })
  (gQ : array2 et lQ { is_global gQ })
  (gO : array2 et lO { is_global gO })
  (gl : array2 et ll { is_global gl })
  (gm : array2 et lm { is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  (bid : natlt 1sz)
  ()
  norewrite
  requires
    live_c_shmems sh **
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
  ensures
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa_smem n d nthr
         (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh)
         gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa_smem n d nthr lS lKV lO ll lm
{
  // Expose the four raw shared arrays.
  unfold_live_c_shmems_cons sh #1.0R;
  unfold_live_c_shmems_cons (snd sh) #1.0R;
  unfold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  unfold_live_c_shmems_cons (snd (snd (snd sh))) #1.0R;
  unfold_live_c_shmems_nil (snd (snd (snd (snd sh)))) #1.0R;
  unfold_live_c_shmem (fst sh) #1.0R;
  unfold_live_c_shmem (fst (snd sh)) #1.0R;
  unfold_live_c_shmem (fst (snd (snd sh))) #1.0R;
  unfold_live_c_shmem (fst (snd (snd (snd sh)))) #1.0R;
  gpu_pts_to_ref (fst sh);
  gpu_pts_to_ref (fst (snd sh));
  gpu_pts_to_ref (fst (snd (snd sh)));
  gpu_pts_to_ref (fst (snd (snd (snd sh))));

  // View them as the sK/sV/sQ caches and the gS scratch matrix.
  tensor_abs' lKV (fst sh);
  tensor_abs' lKV (fst (snd sh));
  tensor_abs' lKV (fst (snd (snd sh)));
  tensor_abs' lS (fst (snd (snd (snd sh))));
  rewrite each (from_array lKV (fst sh)) as (sK_of_sh n d nthr lKV sh);
  rewrite each (from_array lKV (fst (snd sh))) as (sV_of_sh n d nthr lKV sh);
  rewrite each (from_array lKV (fst (snd (snd sh)))) as (sQ_of_sh n d nthr lKV sh);
  rewrite each (from_array lS (fst (snd (snd (snd sh))))) as (gS_of_sh' n d nthr lS sh);

  // Non-shared part: reuse the existing split (live gS_of_sh' + full_io_fa_nos
  // matches setup_fa's full_io_fa precondition).
  setup_fa n d nthr (gS_of_sh' n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ ();

  // Shared caches: split into per-thread write-rows.
  rows_split (sK_of_sh n d nthr lKV sh);
  rows_split (sV_of_sh n d nthr lKV sh);
  rows_split (sQ_of_sh n d nthr lKV sh);

  // Bundle gS-pre + the three write-rows into kpre_post_outer_fa_smem.
  forevery_zip3 #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sK_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r)
    (fun (tid:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sV_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r)
    (fun (tid:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sQ_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r);
  forevery_zip #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) ->
       kpre_post_outer_fa n d nthr (gS_of_sh' n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid)
    (fun (tid:natlt (SZ.v nthr)) ->
       (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sK_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
       (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sV_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
       (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sQ_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r));
  forevery_map #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) ->
       (kpre_post_outer_fa n d nthr (gS_of_sh' n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
       ((exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sK_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
        (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sV_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
        (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sQ_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r)))
    (fun (tid:natlt (SZ.v nthr)) ->
       kpre_post_outer_fa_smem n d nthr
         (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh)
         gK gV gQ gO gl gm eK eV eQ fK fV fQ tid)
    fn tid { () };
}

ghost
fn block_teardown_fa_smem
  (#et : Type0) {| scalar et, floating et |}
  (n d nthr : szp { nthr /? n /\ SZ.fits (nthr * nthr) /\ SZ.fits (nthr * d) })
  (lS : full_layout2 nthr nthr)
  (lKV : full_layout2 nthr d)
  (#lK #lV #lQ #lO: layout2 n d)
  (#ll #lm: layout2 1 n)
  {| ctlayout lS, ctlayout lKV, ctlayout lK, ctlayout lV, ctlayout lQ, ctlayout lO, ctlayout ll, ctlayout lm |}
  (gK : array2 et lK { is_global gK })
  (gV : array2 et lV { is_global gV })
  (gQ : array2 et lQ { is_global gQ })
  (gO : array2 et lO { is_global gO })
  (gl : array2 et ll { is_global gl })
  (gm : array2 et lm { is_global gm })
  (eK eV eQ : ematrix et n d)
  (#fK #fV #fQ : perm)
  (sh : c_shmems (shmems_desc_fa_smem et n d nthr))
  (bid : natlt 1sz)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpre_post_outer_fa_smem n d nthr
         (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh)
         gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
    frame_fa_smem n d nthr lS lKV lO ll lm
  ensures
    live_c_shmems sh **
    full_io_fa_nos n d nthr gK gV gQ gO gl gm eK eV eQ fK fV fQ
{
  // Peel off the three write-rows from the gS-pre.
  forevery_map #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) ->
       kpre_post_outer_fa_smem n d nthr
         (gS_of_sh' n d nthr lS sh) (sK_of_sh n d nthr lKV sh) (sV_of_sh n d nthr lKV sh) (sQ_of_sh n d nthr lKV sh)
         gK gV gQ gO gl gm eK eV eQ fK fV fQ tid)
    (fun (tid:natlt (SZ.v nthr)) ->
       (kpre_post_outer_fa n d nthr (gS_of_sh' n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid) **
       ((exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sK_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
        (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sV_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
        (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sQ_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r)))
    fn tid { () };
  forevery_unzip #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) ->
       kpre_post_outer_fa n d nthr (gS_of_sh' n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ fK fV fQ tid)
    (fun (tid:natlt (SZ.v nthr)) ->
       (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sK_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
       (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sV_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r) **
       (exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sQ_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r));
  forevery_unzip3 #(natlt (SZ.v nthr))
    (fun (tid:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sK_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r)
    (fun (tid:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sV_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r)
    (fun (tid:natlt (SZ.v nthr)) -> exists* (r:ematrix et 1 (SZ.v d)). array2_subtile (sQ_of_sh n d nthr lKV sh) 1 (SZ.v d) tid 0 |-> Frac 1.0R r);

  // Reassemble the three caches and the gS scratch.
  rows_gather (sK_of_sh n d nthr lKV sh);
  rows_gather (sV_of_sh n d nthr lKV sh);
  rows_gather (sQ_of_sh n d nthr lKV sh);
  teardown_fa n d nthr (gS_of_sh' n d nthr lS sh) gK gV gQ gO gl gm eK eV eQ ();

  // Fold the four matrix views back into the raw shared arrays.
  rewrite each (sK_of_sh n d nthr lKV sh) as (from_array lKV (fst sh));
  rewrite each (sV_of_sh n d nthr lKV sh) as (from_array lKV (fst (snd sh)));
  rewrite each (sQ_of_sh n d nthr lKV sh) as (from_array lKV (fst (snd (snd sh))));
  rewrite each (gS_of_sh' n d nthr lS sh) as (from_array lS (fst (snd (snd (snd sh)))));
  tensor_concr (from_array lKV (fst sh));
  tensor_concr (from_array lKV (fst (snd sh)));
  tensor_concr (from_array lKV (fst (snd (snd sh))));
  tensor_concr (from_array lS (fst (snd (snd (snd sh)))));
  rewrite each (core (from_array lKV (fst sh))) as (fst sh);
  rewrite each (core (from_array lKV (fst (snd sh)))) as (fst (snd sh));
  rewrite each (core (from_array lKV (fst (snd (snd sh))))) as (fst (snd (snd sh)));
  rewrite each (core (from_array lS (fst (snd (snd (snd sh)))))) as (fst (snd (snd (snd sh))));
  fold_live_c_shmem (fst sh) #1.0R;
  fold_live_c_shmem (fst (snd sh)) #1.0R;
  fold_live_c_shmem (fst (snd (snd sh))) #1.0R;
  fold_live_c_shmem (fst (snd (snd (snd sh)))) #1.0R;
  fold_live_c_shmems_nil (snd (snd (snd (snd sh)))) #1.0R;
  fold_live_c_shmems_cons (snd (snd (snd sh))) #1.0R;
  fold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  fold_live_c_shmems_cons (snd sh) #1.0R;
  fold_live_c_shmems_cons sh #1.0R;
}
