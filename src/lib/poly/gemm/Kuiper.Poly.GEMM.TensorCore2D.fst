module Kuiper.Poly.GEMM.TensorCore2D

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 60"

open Pulse.Lib.Array
open Pulse.Lib.Trade
open Kuiper.Matrix.Reprs.Type
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Matrix
open Kuiper.EMatrix { ematrix }
open Kuiper.VArray {
  varray,
  varray_pts_to,
  varray_pts_to_cell
}
open Kuiper.TensorCore
open Kuiper.Float16
open Kuiper.Matrix.Tiling
open Kuiper.Poly.GEMM.Copy.Vec
open Kuiper.Poly.GEMM.Tiled.Common.Vec

open Kuiper.Approximates
open Kuiper.Spec.GEMM

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module R = Kuiper.Matrix.Reprs
module FB = Kuiper.Poly.GEMM.FlipFlopBarrier

open Kuiper.Poly.GEMM.TensorCore2D.KernelDesc

let fragarrayAcc_approximates (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wm wn : nat)
  ([@@@mkey] arr : array (fragment et FragAcc tm tn tk FragLAcc) { Pulse.Lib.Array.length arr == wm*wn})
  (rm : ematrix real (wm*tm) (wn*tn))
  : slprop
  =
    exists* (em : seq (ematrix et tm tn)).
      arr |-> em **
      pure (
        (Seq.length em == wm*wn) /\
        forall (i : natlt wm) (j : natlt wn). (em @! (i * wn + j)) %~ (ematrix_subtile rm tm tn i j))

let fragarrayA_approximates (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wm : nat)
  (arr : array (fragment et FragA tm tn tk FragLRM) { Pulse.Lib.Array.length arr == wm})
  (rm : ematrix real (wm*tm) tk)
  : slprop
  =
    exists* (eAs : seq (ematrix et tm tk)).
      arr |-> eAs **
      pure (
        (Seq.length eAs == wm) /\
        forall (i : natlt wm).
          (eAs @! i) %~ (ematrix_subtile rm tm tk i 0))

let fragarrayB_approximates (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wn : nat)
  (arr : array (fragment et FragB tm tn tk FragLRM) { Pulse.Lib.Array.length arr == wn})
  (rm : ematrix real tk (wn*tn))
  : slprop
  =
    exists* (eBs : seq (ematrix et tk tn)).
      arr |-> eBs **
      pure (
        (Seq.length eBs == wn) /\
        forall (i : natlt wn).
          (eBs @! i) %~ (ematrix_subtile rm tk tn 0 i))

inline_for_extraction noextract
fn populate_fragments_a
  (#et : Type0)
  {| scalar et, real_like et |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (frags : array (fragment et FragA tm tn tk FragLRM))
  (gm : gpu_matrix et (R.row_major bm bk))
  (#em : ematrix et bm bk)
  (rm : ematrix real bm bk {em %~ rm})
  (#f : perm)
  (arow : szlt (bm/(wm*tm)))
  (dotIdx : szlt (bk/tk))
  (#_ : squash (Pulse.Lib.Array.length frags == wm))
preserves
  gpu **
  gm |-> Frac f em
requires
  live frags
ensures
  fragarrayA_approximates wm frags (ematrix_subtile rm (wm*tm) tk arow dotIdx)
{
    gpu_matrix_pts_to_ref gm;
    array_fragment_pts_to_ref frags;

    let tile_for_tc_a_tiles =
      gpu_matrix_extract_tile_ro' gm (wm*tm) (SZ.v tk) (SZ.v arow) (SZ.v dotIdx);
    let mut i0 = 0sz;
    while ((!i0 <^ wm))
      invariant live i0
      invariant
        (exists* ems.
          frags |-> ems **
          pure (Seq.length ems == wm /\ !i0 <= wm /\
            forall (i : natlt !i0).
              (ems @! i) %~ (ematrix_subtile rm tm tk (arow*wm+i) dotIdx)))
    {
      // Guido: why is there a zero here? Can this really be right?
      // Guido: I see. tile_for_tc_a_tiles is a very rectangular tile with height equal to one tile.
      //        Zero is the only possible value here. Still seems a bit odd.
      let a_tile =
        gpu_matrix_extract_tile_ro' tile_for_tc_a_tiles (SZ.v tm) (SZ.v tk) (SZ.v !i0) 0;
      array_fragment_extract frags !i0;

      mma_loadA frags.(!i0) a_tile;
      Pulse.Lib.Forall.elim_forall
        (ematrix_subtile (ematrix_subtile em (wm*tm) tk arow dotIdx) tm tk !i0 0);

      ambig_trade_elim ();
      ambig_trade_elim ();

      i0 := !i0 +^ 1sz;
    };
    ambig_trade_elim ();
    fold fragarrayA_approximates wm frags (ematrix_subtile rm (wm*tm) tk arow dotIdx);
    ()
}

inline_for_extraction noextract
fn populate_fragments_b
  (#et : Type0)
  {| scalar et, real_like et |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (frags : array (fragment et FragB tm tn tk FragLRM))
  (gm : gpu_matrix et (R.row_major bk bn))
  (#em : ematrix et bk bn)
  (rm : ematrix real bk bn {em %~ rm})
  (#f : perm)
  (bcol : szlt (bn/(wn*tn)))
  (dotIdx : szlt (bk/tk))
  (#_ : squash (Pulse.Lib.Array.length frags == wn))
preserves
  gpu **
  gm |-> Frac f em
requires
  live frags
ensures
  fragarrayB_approximates wn frags (ematrix_subtile rm tk (wn*tn) dotIdx bcol)
{
    gpu_matrix_pts_to_ref gm;
    array_fragment_pts_to_ref frags;

    let tile_for_tc_b_tiles = gpu_matrix_extract_tile_ro' gm (SZ.v tk) (wn*tn) (SZ.v dotIdx) (SZ.v bcol);
    let mut i1 = 0sz;
    while ((!i1 <^ wn))
      invariant live i1
      invariant
        (exists* ems.
          frags |-> ems **
          pure (Seq.length ems == wn /\ !i1 <= wn /\
            forall (i : natlt !i1).
              (ems @! i) %~ (ematrix_subtile rm tk tn dotIdx (bcol*wn+i))))
    {
      let b_tile = gpu_matrix_extract_tile_ro' tile_for_tc_b_tiles (SZ.v tk) (SZ.v tn) 0 (SZ.v !i1);

      array_fragment_pts_to_ref frags;
      array_fragment_extract frags !i1;

      mma_loadB frags.(!i1) b_tile;
      Pulse.Lib.Forall.elim_forall
        (ematrix_subtile (ematrix_subtile em tk (wn*tn) dotIdx bcol) tk tn 0 !i1);

      ambig_trade_elim ();
      ambig_trade_elim ();

      i1 := !i1 +^ 1sz;
    };
    ambig_trade_elim ();
    fold fragarrayB_approximates wn frags (ematrix_subtile rm tk (wn*tn) dotIdx bcol);
    ()
}

let arrayfragments_fade
  (tm tn tk wm wn : szp)
  (i : natlt wm)
  (j : natlt wn)
  (resIdxM : natle wm)
  (resIdxN : natle wn)
  (rA : ematrix real (wm*tm) tk)
  (rB : ematrix real tk (wn*tn))
  (rAcc : ematrix real (wm*tm) (wn*tn))
: ematrix real tm tn
=
  let flat_idx = i * wn + j in
  let num_res_computed = resIdxM * wn + resIdxN in
  if flat_idx < num_res_computed
  then ematrix_subtile rAcc tm tn i j `matplus`
    (matmul (ematrix_subtile rA tm tk i 0) (ematrix_subtile rB tk tn 0 j))
  else ematrix_subtile rAcc tm tn i j

#push-options "--z3rlimit 80"
#restart-solver // Trying to avoid infinite loop in batch mode!?
inline_for_extraction noextract
fn fragarray_mma
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc, real_like et_ab, real_like et_acc |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (aFrags     : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags     : array (fragment et_ab FragB tm tn tk FragLRM))
  (accumFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (rA : ematrix real (wm*tm) tk)
  (rB : ematrix real tk (wn*tn))
  (rAcc : ematrix real (wm*tm) (wn*tn))
  (dotIdx : szlt (bk/tk))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
  preserves
    fragarrayA_approximates wm aFrags rA **
    fragarrayB_approximates wn bFrags rB
  requires
    pure (valid_frag_et_comb et_ab et_acc)
  requires
    fragarrayAcc_approximates wm wn accumFrags rAcc
  ensures
    fragarrayAcc_approximates wm wn accumFrags (rAcc `matplus` (matmul rA rB))
{
  unfold fragarrayA_approximates wm aFrags;
  unfold fragarrayB_approximates wn bFrags;
  unfold fragarrayAcc_approximates wm wn accumFrags;

  with eAs. assert aFrags |-> eAs;
  with eBs. assert bFrags |-> eBs;

  let mut resIdxM = 0sz;
  while ((!resIdxM <^ wm))
    invariant live resIdxM
    invariant
      exists* (eAcc : seq (ematrix et_acc tm tn)).
        accumFrags |-> eAcc **
        pure (
          !resIdxM <= wm /\
          (Seq.length eAcc == wm*wn) /\
          forall (i : natlt wm) (j : natlt wn).
            (eAcc @! (i * wn + j)) %~
              (arrayfragments_fade tm tn tk wm wn i j !resIdxM 0 rA rB rAcc))
  {
    let mut resIdxN = 0sz;
    while ((!resIdxN <^ wn))
      invariant live resIdxN
      invariant
        exists* (eAcc : seq (ematrix et_acc tm tn)).
          accumFrags |-> eAcc **
          pure (
            !resIdxN <= wn /\
            (Seq.length eAcc == wm*wn) /\
            forall (i : natlt wm) (j : natlt wn).
              (eAcc @! (i * wn + j)) %~
                (arrayfragments_fade tm tn tk wm wn i j !resIdxM !resIdxN rA rB rAcc))
    {
      with eAccs. assert accumFrags |-> eAccs;

      array_fragment_pts_to_ref aFrags;
      array_fragment_pts_to_ref bFrags;
      array_fragment_pts_to_ref accumFrags;

      array_fragment_extract_ro aFrags !resIdxM;
      array_fragment_extract_ro bFrags !resIdxN;
      array_fragment_extract accumFrags (!resIdxM * wn + !resIdxN);

      let a_frag = aFrags.(!resIdxM);
      let b_frag = bFrags.(!resIdxN);
      let acc_frag = accumFrags.(!resIdxM *^ wn +^ !resIdxN);

      with eAt. assert a_frag |-> eAt;
      with eBt. assert b_frag |-> eBt;
      with eAcct. assert acc_frag |-> eAcct;
      assert pure (eAt %~ (ematrix_subtile rA tm tk !resIdxM 0));
      assert pure (eBt %~ (ematrix_subtile rB tk tn 0 !resIdxN));
      assert pure (eAcct %~ (ematrix_subtile rAcc tm tn !resIdxM !resIdxN));

      mma_sync' a_frag b_frag acc_frag;

      Kuiper.TensorCore.Base.emma_approx_lemma eAcct eAt eBt
        (ematrix_subtile rAcc tm tn !resIdxM !resIdxN)
        (ematrix_subtile rA tm tk !resIdxM 0)
        (ematrix_subtile rB tk tn 0 !resIdxN);

      ambig_trade_elim ();
      ambig_trade_elim ();

      with v. assert acc_frag `fragment_pts_to` v;
      Pulse.Lib.Forall.elim_forall v;

      ambig_trade_elim ();

      assert array_fragment_pts_to accumFrags (Seq.Base.upd eAccs
            (!resIdxM * wn + !resIdxN)
            (emma (eAccs @! (!resIdxM * wn + !resIdxN))
                (eAs @! !resIdxM)
                (eBs @! !resIdxN)));

      resIdxN := !resIdxN +^ 1sz;
    };

    resIdxM := !resIdxM +^ 1sz;
  };

  with eAcc. assert accumFrags |-> eAcc;
  assert pure (
    forall (i : natlt wm) (j : natlt wn).
              (eAcc @! (i * wn + j)) %~ (arrayfragments_fade tm tn tk wm wn i j wm 0 rA rB rAcc));
  assert pure (
    forall (i : natlt wm) (j : natlt wn).
      (eAcc @! (i * wn + j)) %~
                ematrix_subtile rAcc tm tn i j `matplus`
                  (matmul (ematrix_subtile rA tm tk i 0) (ematrix_subtile rB tk tn 0 j)));

  assert pure (
    forall (i : natlt wm) (j : natlt wn).
              (eAcc @! (i * wn + j)) %~ (ematrix_subtile (rAcc `matplus` (matmul rA rB)) tm tn i j));
  assert pure (Seq.length eAcc == wm*wn);

  fold fragarrayA_approximates wm aFrags rA;
  fold fragarrayB_approximates wn bFrags rB;
  fold fragarrayAcc_approximates wm wn accumFrags (rAcc `matplus` (matmul rA rB));
}
#pop-options

// Working around apparent bug below
inline_for_extraction noextract
let sz_succ (x:SZ.t{SZ.fits (x+1)}) : SZ.t = x +^ 1sz

inline_for_extraction noextract
fn subproducts_tc_2d
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc, real_like et_ab, real_like et_acc |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (aFrags     : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags     : array (fragment et_ab FragB tm tn tk FragLRM))
  (accumFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (gA : gpu_matrix et_ab (R.row_major bm bk))
  (gB : gpu_matrix et_ab (R.row_major bk bn))
  (#eA : ematrix et_ab bm bk)
  (#eB : ematrix et_ab bk bn)
  (rA : ematrix real bm bk {eA %~ rA})
  (rB : ematrix real bk bn {eB %~ rB})
  (rAcc : ematrix real (wm*tm) (wn*tn))
  (#fA #fB : perm)
  (arow : szlt (bm/(wm*tm)))
  (bcol : szlt (bn/(wn*tn)))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (valid_frag_et_comb et_ab et_acc)
  preserves
    // aFrags and bFrags are swap space, we don't specify much about them
    live aFrags ** live bFrags
  requires
    fragarrayAcc_approximates wm wn accumFrags rAcc
  ensures
    fragarrayAcc_approximates wm wn accumFrags
      (rAcc `matplus` matmul (ematrix_subtile rA (wm*tm) bk arow 0)
                             (ematrix_subtile rB bk (wn*tn) 0 bcol))
{
  rewrite each rAcc
  as __gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn)) arow bcol 0;

  let mut dotIdx : sz = 0sz;
  while ((!dotIdx <^ (bk/^tk)))
    invariant live aFrags ** live bFrags
    invariant
      exists* (vdotIdx : sz { vdotIdx <= (bk/tk) }).
        dotIdx |-> vdotIdx **
        fragarrayAcc_approximates wm wn accumFrags
          (__gmatmul_single rAcc matmul matplus
            (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn)) arow bcol !dotIdx)
  {
    populate_fragments_a bm bn bk tm tn tk wm wn aFrags gA rA arow !dotIdx;
    populate_fragments_b bm bn bk tm tn tk wm wn bFrags gB rB bcol !dotIdx;

    fragarray_mma bm bn bk tm tn tk wm wn aFrags bFrags accumFrags
      (ematrix_subtile rA (wm*tm) tk arow !dotIdx)
      (ematrix_subtile rB tk (wn*tn) !dotIdx bcol)
      (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn)) arow bcol !dotIdx)
      !dotIdx;

    unfold fragarrayA_approximates wm aFrags;
    unfold fragarrayB_approximates wn bFrags;

    __gmatmul_single_lemma rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
      arow bcol (!dotIdx +^ 1sz);
    rewrite each
        (matplus (__gmatmul_single rAcc matmul matplus
                (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
                arow bcol !dotIdx)
            (matmul (ematrix_subtile rA (wm*tm) tk arow !dotIdx)
                    (ematrix_subtile rB tk (wn*tn) !dotIdx bcol)))
    as
        (__gmatmul_single rAcc matmul matplus
            (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
            arow bcol (!dotIdx +^ 1sz));

    // Weird issue here. We'd like to just add one and be done but
    // that doesn't seem to work due to some SZ.v (x+^1sz) vs SZ.v x + 1
    // issues
    with vdotIdx. assert dotIdx |-> vdotIdx;
    dotIdx := sz_succ !dotIdx;// +^ 1sz;S
    rewrite each
      (SZ.v vdotIdx + 1)
    as
      (SZ.v (sz_succ vdotIdx));

    ()
  };

  assert pure (!dotIdx == bk/^tk);
  assert pure (SZ.v (bk/^tk) == bk/tk);
  with vdotIdx. assert (dotIdx |-> vdotIdx ** pure (vdotIdx == bk/^tk));

  rewrite each vdotIdx as (bk/^tk);
  assert (fragarrayAcc_approximates wm wn accumFrags
    (__gmatmul_single rAcc matmul matplus
          (ematrix_tiled rA (wm*tm) tk)
          (ematrix_tiled rB tk (wn*tn))
          arow
          bcol
          (bk/^tk)));

  matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ()) (wm*tm) (wn*tn) tk rAcc rA rB arow bcol;
  rewrite each (
    __gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn)) arow bcol (bk/^tk))
  as (rAcc `matplus` matmul (ematrix_subtile rA (wm*tm) bk arow 0)
                            (ematrix_subtile rB bk (wn*tn) 0 bcol));
  ()
}

let em_fade_tiles
  (tm tn wm wn : pos)
  (idxI : natle wm)
  (idxJ : natle wn)
  (rm1 rm2 : ematrix real (wm*tm) (wn*tn))
: ematrix real (wm*tm) (wn*tn)
=
  ematrix_from_tiles tm tn (fun i j ->
    let flat_idx = i * wn + j in
    let num_copied = idxI * wn + idxJ in
    if flat_idx < num_copied
    then ematrix_subtile rm2 tm tn i j
    else ematrix_subtile rm1 tm tn i j)

#push-options "--z3rlimit 80 --split_queries always"
let lemma_update_tile_fade_approximates
  (#et : Type0) {| scalar et, real_like et|}
  (tm tn wm wn : pos)
  (idxI : natlt wm)
  (idxJ : natlt wn)
  (em : ematrix et (wm*tm) (wn*tn))
  (etile : ematrix et tm tn)
  (rm1 rm2 : ematrix real (wm*tm) (wn*tn))
: Lemma
  (requires
    (em %~ (em_fade_tiles tm tn wm wn idxI idxJ rm1 rm2)) /\
    (etile %~ (ematrix_subtile rm2 tm tn idxI idxJ)))
  (ensures (update_tile em tm tn idxI idxJ etile) %~ (em_fade_tiles tm tn wm wn idxI (idxJ + 1) rm1 rm2))
=
  () // would be nice to spell it out probably
#pop-options

inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et, real_like et |}
  (#rows : erased nat)
  // cols is concretized so using size is more succinct
  (#cols : sz)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc))
  (rAcc : ematrix real (wm*tm) (wn*tn))
  (gC : gpu_matrix et (R.row_major rows cols))
  // (#eC : ematrix et rows cols)
  (#_ : squash (SZ.fits (rows * cols)))
  (bid : szlt (rows/bm * (cols/bn)))
  (wid : szlt (bm/(wm*tm) * (bn/(wn*tn))))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
  preserves
    gpu **
    fragarrayAcc_approximates wm wn accumFrags rAcc
  requires
    pure (SZ.fits (wm * wn)) **
    (exists* eWarpTile. warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile)
  ensures
    warp_tile_approximates gC bm bn tm tn wm wn bid wid rAcc
{
  with (eWarpTile : ematrix _ _ _). assert warp_tile_pts_to gC (v bm) (v bn) (v tm) (v tn) (v wm) (v wn) (v bid) (v wid) eWarpTile;
  let rWarpTile = to_real_matrix eWarpTile;

  lemma_to_real_matrix_approximates eWarpTile;
  assert pure (eWarpTile %~ rWarpTile);
  assert pure (eWarpTile %~ ematrix_from_tiles tm tn (ematrix_subtile rWarpTile tm tn));
  assert pure (eWarpTile %~ em_fade_tiles tm tn wm wn 0 0 rWarpTile rAcc);

  let mut i = 0sz;
  while ((!i <^ wm))
    invariant
      live i
    invariant
      exists* (eWarpTile: ematrix et (wm*tm) (wn*tn)).
        warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile **
          pure (!i <= wm /\
            eWarpTile %~ (em_fade_tiles tm tn wm wn !i 0 rWarpTile rAcc))
  {
    let mut j = 0sz;
    while ((!j <^ wn))
      invariant live j
      invariant
        exists* (eWarpTile: ematrix et (wm*tm) (wn*tn)).
          warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile **
            pure (!i <= wm /\ !j <= wn /\
              eWarpTile %~ (em_fade_tiles tm tn wm wn !i !j rWarpTile rAcc))
    {
      with eWarpTile. assert warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile;
      unfold warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile;

      let tile_for_tc_tiles = warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (wm*tm) (wn*tn) (SZ.v wid);
      rewrite each _ as tile_for_tc_tiles;

      let tc_tile = gpu_matrix_extract_tile_st tile_for_tc_tiles (SZ.v tm) (SZ.v tn) (SZ.v !i) (SZ.v !j);

      let vi = !i;
      let vj = !j;
      let eidx : erased nat = vi * wn + vj;

      assert pure (vi < wm);
      assert pure (vj < wn);
      assert pure (eidx < wm * wn);
      assert pure (SZ.fits eidx);
      let idx = !i *^ wn +^ !j;

      unfold fragarrayAcc_approximates wm wn accumFrags rAcc;
      with eAccumFrags. assert accumFrags `array_fragment_pts_to` eAccumFrags;

      array_fragment_pts_to_ref accumFrags;
      array_fragment_extract_ro accumFrags idx;
      mma_store accumFrags.(idx) tc_tile;

      Pulse.Lib.Forall.elim_forall (Seq.Base.index eAccumFrags idx);
      ambig_trade_elim ();
      ambig_trade_elim ();
      fold fragarrayAcc_approximates wm wn accumFrags rAcc;

      rewrite each tile_for_tc_tiles as _;
      with eWarpTile'. fold warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile';

      lemma_update_tile_fade_approximates tm tn wm wn !i !j eWarpTile (eAccumFrags @! idx) rWarpTile rAcc;

      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  with eWarpTile'.
    assert (warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile');
  assert pure (eWarpTile' %~ (em_fade_tiles tm tn wm wn wm wn rWarpTile rAcc));
  assert pure (eWarpTile' %~ (ematrix_from_tiles tm tn (ematrix_subtile rAcc tm tn)));
  assert pure (eWarpTile' %~ rAcc);

  fold warp_tile_approximates gC bm bn tm tn wm wn bid wid rAcc;
  ()
}

inline_for_extraction noextract
fn populate_acc_with_zero
  (#et : Type0) {| sc : scalar et, real_like et |}
  (tm tn tk wm wn : szp)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
requires
  live accumFrags
ensures
  fragarrayAcc_approximates wm wn accumFrags (const_matrix 0.0R)
{
  array_fragment_pts_to_ref accumFrags;

  let mut fi : sz = 0sz;
  while ((!fi <^ wm*^wn))
    invariant
      live fi **
      (exists* (eAcc : seq (ematrix et tm tn)).
        accumFrags |-> eAcc **
        pure (
          Seq.length eAcc == wm*wn /\ !fi <= wm*wn  /\
          forall (i : natlt !fi).
            (eAcc @! i) %~ (ematrix_subtile (const_matrix #_ #(wm*tm) #(wn*tn) 0.0R) tm tn (i/wn) (i%wn))))
  {
    array_fragment_pts_to_ref accumFrags;
    array_fragment_extract accumFrags !fi;
    mma_fill accumFrags.(!fi) sc.zero;

    Pulse.Lib.Forall.elim_forall (fill_value sc.zero);
    ambig_trade_elim();

    fi := !fi +^ 1sz;
  };
  fold fragarrayAcc_approximates wm wn accumFrags (const_matrix 0.0R);
  ()
}

let loop_invariant_lemma
  (rows shared cols : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (bk /?+ shared))
  (mrow : natlt (rows / bm))
  (mcol : natlt (cols / bn))
  (warpRow : natlt (bm/(wm*tm)))
  (warpCol : natlt (bn/(wn*tn)))
  (gwRow : natlt (rows/(wm*tm)) { gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : natlt (cols/(wn*tn)) { gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (vk : natlt (shared / bk))
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rAcc0 : ematrix real (wm*tm) (wn*tn) { rAcc0 == const_matrix 0.0R })
  (rAcc  : ematrix real (wm*tm) (wn*tn))
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  (#_ : squash (rAcc  ==
          (__gmatmul_single rAcc0 matmul matplus
            (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn))
              gwRow
              gwCol
              vk)))
  (rA_sub : ematrix real bm bk { rA_sub == ematrix_subtile rA bm bk mrow vk })
  (rB_sub : ematrix real bk bn { rB_sub == ematrix_subtile rB bk bn vk mcol })
: Lemma (
        rAcc `matplus` matmul (ematrix_subtile rA_sub (wm*tm) bk warpRow 0)
                               (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol)
        ==
        __gmatmul_single rAcc0 matmul matplus
          (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) gwRow gwCol (vk + 1)
    )
= let lhs : ematrix real (wm*tm) (wn*tn) = rAcc `matplus` matmul (ematrix_subtile rA_sub (wm*tm) bk warpRow 0)
                                   (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol) in
  let rhs : ematrix real (wm*tm) (wn*tn) =
        __gmatmul_single rAcc0 matmul matplus
          (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) gwRow gwCol (vk + 1)
  in
  let aux3 () : Lemma ((wm * tm) * gwRow == bm * mrow + warpRow * (wm*tm)) =
    calc (==) {
      (wm*tm) * gwRow;
      == {}
      (wm * tm) * (mrow * (bm/(wm*tm)) + warpRow);
      == { Math.Lemmas.distributivity_add_right (wm*tm) (mrow * (bm/(wm*tm))) warpRow }
      (wm * tm) * (mrow * (bm/(wm*tm))) + (wm*tm)*warpRow;
      == {}
      mrow * ((wm * tm) * (bm/(wm*tm))) + (wm*tm)*warpRow;
      == { Math.Lemmas.lemma_div_exact bm (wm*tm) }
      mrow * bm + (wm*tm)*warpRow;
      == {}
      bm * mrow + warpRow * (wm*tm);
    }
  in
  let aux4 () : Lemma ((wn * tn) * gwCol == bn * mcol + warpCol * (wn*tn)) =
    calc (==) {
      (wn*tn) * gwCol;
      == {}
      (wn * tn) * (mcol * (bn/(wn*tn)) + warpCol);
      == { Math.Lemmas.distributivity_add_right (wn*tn) (mcol * (bn/(wn*tn))) warpCol }
      (wn * tn) * (mcol * (bn/(wn*tn))) + (wn*tn)*warpCol;
      == {}
      mcol * ((wn * tn) * (bn/(wn*tn))) + (wn*tn)*warpCol;
      == { Math.Lemmas.lemma_div_exact bn (wn*tn) }
      mcol * bn + (wn*tn)*warpCol;
      == {}
      bn * mcol + warpCol * (wn*tn);
    }
  in
  aux3();
  aux4();
  let aux1 () : Lemma (
                  ematrix_subtile rA_sub (wm*tm) bk warpRow 0
                  ==
                  macc (ematrix_tiled rA (wm*tm) bk) gwRow vk
                )
  = assert (ematrix_subtile rA_sub (wm*tm) bk warpRow 0
            `equal` macc (ematrix_tiled rA (wm*tm) bk) gwRow vk)
  in
  let aux2 () : Lemma (
                  ematrix_subtile rB_sub bk (wn*tn) 0 warpCol
                  ==
                  macc (ematrix_tiled rB bk (wn*tn)) vk gwCol
                )
  = assert (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol
            `equal` macc (ematrix_tiled rB bk (wn*tn)) vk gwCol)
  in
  aux1 ();
  aux2 ();

  let aux (i : natlt (wm*tm)) (j : natlt (wn*tn))
    : Lemma (macc lhs i j == macc rhs i j)
    = calc (==) {
        macc lhs i j;
        == {}
        macc (__gmatmul_single rAcc0 matmul matplus
               (ematrix_tiled rA (wm*tm) bk)
               (ematrix_tiled rB bk (wn*tn)) gwRow gwCol vk
              `matplus`
                 matmul (ematrix_subtile rA_sub (wm*tm) bk warpRow 0)
                        (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol)) i j;
        == {}
        macc (__gmatmul_single rAcc0 matmul matplus
               (ematrix_tiled rA (wm*tm) bk)
               (ematrix_tiled rB bk (wn*tn)) gwRow gwCol vk
              `matplus`
                 matmul (macc (ematrix_tiled rA (wm*tm) bk) gwRow vk)
                        (macc (ematrix_tiled rB bk (wn*tn)) vk gwCol)) i j;
        == { __gmatmul_single_lemma rAcc0 matmul matplus
               (ematrix_tiled rA (wm*tm) bk)
               (ematrix_tiled rB bk (wn*tn)) gwRow gwCol (vk + 1) }
        macc (__gmatmul_single rAcc0 matmul matplus
               (ematrix_tiled rA (wm*tm) bk)
               (ematrix_tiled rB bk (wn*tn)) gwRow gwCol (vk+1)) i j;
        == {}
        macc rhs i j;
      }
  in
  Classical.forall_intro_2 aux;
  assert (Kuiper.EMatrix.equal lhs rhs);
  ()

inline_for_extraction noextract
fn kf
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, sc : scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA |}
  (gA : gpu_matrix et_ab lA)
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : gpu_matrix et_ab lB)
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols))
  (#eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (rows * shared)))
  (#_ : squash (SZ.fits (rows * cols)))
  (#_ : squash (SZ.fits (shared * cols)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#fA #fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (nthr : erased nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : szlt (rows/bm * (cols/bn)))
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid **
    thread_id nthr tid **
    block_id (rows/bm * (cols/bn)) bid
  ensures
    gpu **
    kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid **
    thread_id nthr tid **
    block_id (rows/bm * (cols/bn)) bid
{
  unfold_c_shmems sh (`%shmems_desc);
  let (sarA, (sarB, _)) = sh;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  unfold FB.barrier_tok eA eB (R.row_major bm bk) (R.row_major bk bn) sarA sarB 0 nthr bid tid;

  gpu_matrix_abs' (R.row_major bm bk) sarA;
  let sA = from_array (R.row_major bm bk) sarA;
  rewrite each _ as sA; //from_array (R.row_major bm bk) sarA as sA;

  gpu_matrix_abs' (R.row_major bk bn) sarB;
  let sB = from_array (R.row_major bk bn) sarB;
  rewrite each _ as sB; //from_array (R.row_major bk bn) sarB as sB;

  let num_k_tiles = shared /^ bk;
  let num_n_tiles = cols /^ bn;
  let mrow = bid /^ num_n_tiles;
  assert pure (mrow < rows / bm);
  let mcol = bid %^ num_n_tiles;
  assert pure (mcol < cols / bn);

  let wid = tid /^ warp_sz;
  let warpRow : szlt (bm / (wm*tm)) = wid /^ (bn/^(wn*^tn));
  let warpCol : szlt (bn / (wn*tn)) = wid %^ (bn/^(wn*^tn));

  (* Tensor core fragments *)
  let aFrags = __alloc_array_fragment et_ab FragA tm tn tk FragLRM wm;
  let bFrags = __alloc_array_fragment et_ab FragB tm tn tk FragLRM wn;
  let accFrags = __alloc_array_fragment et_c FragAcc tm tn tk FragLAcc (wm *^ wn);

  // Fill accumulators with 0 
  populate_acc_with_zero tm tn tk wm wn accFrags;
  let rAcc0 : ematrix real (wm*tm) (wn*tn) = const_matrix 0.0R;
  assert (rewrites_to rAcc0 (const_matrix 0.0R));

  rewrite fragarrayAcc_approximates wm wn accFrags rAcc0
       as fragarrayAcc_approximates wm wn accFrags
            (__gmatmul_single rAcc0 matmul matplus
              (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) mrow mcol 0);

  rewrite
    (exists* (x : ematrix _ _ _). sA |-> Frac (1.0R /. nthr) x) **
    (exists* (x : ematrix _ _ _). sB |-> Frac (1.0R /. nthr) x)
  as
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr);

  let gwRow : enatlt (rows/(wm*tm)) = mrow * (bm/(wm*tm)) + warpRow;
  let gwCol : enatlt (cols/(wn*tn)) = mcol * (bn/(wn*tn)) + warpCol;

  let mut bkIdx : sz = 0sz;
  while ((!bkIdx <^ num_k_tiles))
    invariant
      exists* (vbkIdx : sz { vbkIdx <= num_k_tiles }).
        bkIdx |-> vbkIdx **
        fragarrayAcc_approximates wm wn accFrags
          (__gmatmul_single rAcc0 matmul matplus
            (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn))
              gwRow // (mrow * bm/(wm*tm) + warpRow)
              gwCol // (mcol * bn/(wn*tn) + warpCol)
              !bkIdx)
    invariant
      live aFrags **
      live bFrags
    invariant
      (exists* em1. FB.bp_sharing sA em1 nthr) **
      (exists* em2. FB.bp_sharing sB em2 nthr) **
      B.barrier_tok (FB.barrier_p eA eB sA sB nthr bid) (FB.barrier_q eA eB sA sB nthr bid) (2 * !bkIdx) tid //**
  {
    even_2x !bkIdx;
    assert pure((2 * !bkIdx % 2 = 0) == true);
    assert pure (even (2 * !bkIdx));

    #set-options "--z3rlimit 100 --retry 3" {
    rewrite
        (exists* em1. FB.bp_sharing sA em1 nthr) **
        (exists* em2. FB.bp_sharing sB em2 nthr)
      as FB.barrier_p eA eB sA sB nthr bid (2 * !bkIdx) tid;
    };

    B.barrier_wait ();
    rewrite FB.barrier_q eA eB sA sB nthr bid (2 * !bkIdx) tid
        as live_strided_chunks sA nthr tid **
           live_strided_chunks sB nthr tid;

    copy_tiles_out_of_matrices_vec bm bn bk sA sB gA gB mrow !bkIdx mcol (bm/^(wm*^tm)*^(bn/^(wn*^tn))*^warp_sz) tid;

    assert B.barrier_tok (FB.barrier_p eA eB sA sB nthr bid) (FB.barrier_q eA eB sA sB nthr bid) (2 * !bkIdx + 1) tid;
    odd_2x1 !bkIdx;
    assert (pure (odd (2 * !bkIdx + 1)));
    (* sigh.. *)
    #set-options "--z3rlimit 100 --retry 3" {
    rewrite own_strided_chunks sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid **
            own_strided_chunks sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid
         as FB.barrier_p eA eB sA sB nthr bid (2 * !bkIdx + 1) tid;
    };

    B.barrier_wait ();

    even_2x (!bkIdx + 1);
    #set-options "--z3refresh --z3rlimit_factor 2 --fuel 0 --ifuel 0" {
      assert pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 2);
      assert pure (odd (2 * !bkIdx + 1));
      assert pure ((2 * !bkIdx + 1) < 2 * shared / bk);
      assert pure (even (2 * !bkIdx + 2))
    };
    rewrite
      FB.barrier_q eA eB sA sB nthr bid (2 * !bkIdx + 1) tid
    as
      FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr **
      FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    unfold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    unfold FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    let rA_sub = ematrix_subtile rA bm bk mrow !bkIdx;
    let rB_sub = ematrix_subtile rB bk bn !bkIdx mcol;
    with rAcc. assert fragarrayAcc_approximates wm wn accFrags rAcc;
    subproducts_tc_2d bm bn bk tm tn tk wm wn aFrags bFrags accFrags
      sA sB
      rA_sub rB_sub
      rAcc
      warpRow warpCol;
    assert
      fragarrayAcc_approximates wm wn accFrags
        (rAcc `matplus` matmul (ematrix_subtile rA_sub (wm*tm) bk warpRow 0)
                               (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol));

    loop_invariant_lemma
      rows shared cols
      bm bn bk tm tn tk wm wn
      mrow mcol
      warpRow warpCol
      gwRow gwCol
      !bkIdx
      rA rB
      rAcc0 rAcc
      rA_sub rB_sub;

    fold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    fold FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    bkIdx := !bkIdx +^ 1sz;
  };

  assert
        fragarrayAcc_approximates wm wn accFrags
          (__gmatmul_single rAcc0 matmul matplus
            (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn))
              gwRow // (mrow * bm/(wm*tm) + warpRow)
              gwCol // (mcol * bn/(wn*tn) + warpCol)
              (shared / bk));

  assert pure (gwRow == warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size));
  assert pure (gwCol == warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size));

  matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ())
    (wm*tm) (wn*tn) bk
    rAcc0 rA rB
    gwRow gwCol;

  let rAcc' : ematrix real (wm*tm) (wn*tn) =
    gmatmul_single rAcc0 matmul matplus
     (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn))
       gwRow gwCol;

  assert pure (
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) gwRow gwCol !bkIdx)
      == rAcc');

  let rAcc'' : ematrix real (wm*tm) (wn*tn) =
    MS.matmul (ematrix_subtile rA (wm*tm) shared (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
              (ematrix_subtile rB shared  (wn*tn) 0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)));

  assert pure (matplus (const_matrix 0.0R) rAcc'' `equal` rAcc'');
  // ^ This is needed so we can use the result of the matmul_tiles_lemma
  // above...  very boring.

  assert pure (rAcc' == rAcc'');
  rewrite
    fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) gwRow gwCol !bkIdx)
  as
    fragarrayAcc_approximates wm wn accFrags rAcc'';

  with em1. unfold FB.bp_sharing sA em1 nthr;
  with em2. unfold FB.bp_sharing sB em2 nthr;

  rewrite each (tid / 32) as wid;
  epilogue bm bn bk tm tn tk wm wn accFrags rAcc'' gC bid wid;
  rewrite each v wid as (tid / 32);

  with vaFrags. assert aFrags |-> vaFrags; drop_ (aFrags |-> vaFrags);
  with vbFrags. assert bFrags |-> vbFrags; drop_ (bFrags |-> vbFrags);
  unfold fragarrayAcc_approximates wm wn accFrags rAcc'';
  with vaccumFrags. assert accFrags |-> vaccumFrags; drop_ (accFrags |-> vaccumFrags);

  gpu_matrix_concr sA; rewrite each core sA as sarA;
  gpu_matrix_concr sB; rewrite each core sB as sarB;

  #set-options "--z3rlimit 100 --retry 3" {
    rewrite
      B.barrier_tok (FB.barrier_p eA eB sA sB nthr bid)
        (FB.barrier_q eA eB sA sB nthr bid)
        (2 * v !bkIdx)
        (v tid)
    as
      B.barrier_tok (FB.barrier_p eA eB (from_array (R.row_major (v bm) (v bk)) sarA)
            (from_array (R.row_major (v bk) (v bn)) sarB)
            nthr bid)
        (FB.barrier_q eA eB (from_array (R.row_major (v bm) (v bk)) sarA)
            (from_array (R.row_major (v bk) (v bn)) sarB)
            nthr bid)
        (2 * (shared / bk))
        (v tid);
    fold FB.barrier_tok eA eB (R.row_major bm bk) (R.row_major bk bn) sarA sarB (2 * (shared / bk)) nthr bid tid;
  };

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  // Silly rewrite.
  rewrite each rAcc''
    as MS.matmul (ematrix_subtile rA (wm*tm) shared
            (warp_tile_i #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
          (ematrix_subtile rB shared (wn*tn)
            0 (warp_tile_j #rows #cols bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)));

  fold_c_shmems sh (`%shmems_desc);
  ()
}

#push-options "--fuel 1 --ifuel 1 --split_queries no --z3rlimit_factor 10"
#restart-solver
inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared) {| clayout lA |}
  (gA : gpu_matrix et_ab lA  { is_global_matrix gA })
  (#eA : ematrix et_ab rows shared)
  (#lB : mlayout shared cols) {| clayout lB |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : gpu_matrix et_ab lB { is_global_matrix gB })
  (#eB : ematrix et_ab shared cols)
  (gC : gpu_matrix et_c (R.row_major rows cols) { is_global_matrix gC })
  // ^ Why does this have a fixed layout?
  (#_ : squash (SZ.fits (rows * cols)))
  (#eC : ematrix et_c rows cols)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ rows))
  (#_ : squash (bn /?+ cols))
  (#_ : squash (bk /?+ shared))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == rows/bm * (cols/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (nblk <= max_blocks))
  (#_ : squash (nthr <= max_threads))
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  (#_ : squash (wm * tm /?+ rows)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ cols)) // idem
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** pure (eA %~ rA) **
       gB |-> Frac fB eB ** pure (eB %~ rB) **
       gC |-> eC ** pure (eC %~ rC))
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       (exists* (eC' : ematrix et_c rows cols).
         gC |-> eC' ** pure (eC' %~ MS.matmul rA rB)))
= {
  nblk;
  nthr;

  shmems_desc = shmems_desc et_ab bm bn bk;

  frame = pure (SZ.fits (mlayout_size (R.row_major rows cols)));
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid);

  setup      = setup    gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB rA rB rC;
  teardown   = teardown gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB rA rB rC;
  block_teardown = block_teardown gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB rA rB rC;

  kpre      = kpre  gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr;
  kpost     = kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr;

  f = kf gA #eA gB #eB gC #eC bm bn bk tm tn tk wm wn rA rB rC (SZ.v nthr);

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=solve;
  kpost_sendable=solve;
}
#pop-options
