module Kuiper.Kernel.GEMM.TensorCore2D.Epilogue

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 20"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Float16
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.Spec.GEMM
open Kuiper.TensorCore
open Pulse.Lib.Array
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module Chest = Kuiper.Chest

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
open Kuiper.Kernel.GEMM.TensorCore2D.Fade
open Kuiper.Kernel.GEMM.TensorCore2D.FadeUpdate
#restart-solver

let flat_fragment_index_bound (wm wn : pos) (i : natlt wm) (j : natlt wn)
  : Lemma (i * wn + j < wm * wn)
= FStar.Math.Lemmas.lemma_eucl_div_bound j i wn;
  FStar.Math.Lemmas.lemma_mult_le_right wn (i + 1) wm

#push-options "--fuel 1 --ifuel 1"
inline_for_extraction noextract
fn epilogue
  (#et : Type0) {| scalar et, real_like et |}
  (#m : erased nat)
  // n is concretized so using size is more succinct
  (#n : sz)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc))
  (rAcc : chest2 real (wm*tm) (wn*tn))
  (gC : array2 et (rm m n))
  // Fused output combine: the real-domain [comb] and its approximation-compatible
  // device element combine [ecomb].  The stored result approximates
  // [chest_comb comb rCtile rAcc] (old C combined with the accumulator).
  (comb : real -> real -> real)
  (ecomb : et -> et -> et)
  (rCtile : chest2 real (wm*tm) (wn*tn))
  (#_ : squash (Kuiper.Approximates.approx2 ecomb comb))
  // (#eC : chest2 et m n)
  (#_ : squash (SZ.fits (m * n)))
  (bid : szlt (m/bm * (n/bn)))
  (wid : szlt (bm/(wm*tm) * (bn/(wn*tn))))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
  preserves
    gpu **
    fragarrayAcc_approximates wm wn accumFrags rAcc
  requires
    pure (SZ.fits (wm * wn)) **
    warp_tile_approximates gC bm bn tm tn wm wn bid wid rCtile
  ensures
    warp_tile_approximates gC bm bn tm tn wm wn bid wid (Chest.chest_comb comb rCtile rAcc)
{
  unfold warp_tile_approximates gC bm bn tm tn wm wn bid wid rCtile;
  with (eWarpTile : chest2 _ _ _). assert warp_tile_pts_to gC (v bm) (v bn) (v tm) (v tn) (v wm) (v wn) (v bid) (v wid) eWarpTile;
  assert pure (eWarpTile %~ rCtile);
  assert pure (eWarpTile %~ ematrix_from_tiles tm tn (ematrix_subtile rCtile tm tn));
  assert pure (eWarpTile %~ em_fade_comb_tiles comb tm tn wm wn 0 0 rCtile rAcc);

  let mut i = 0sz;
  while (!i <^ wm)
    invariant
      live i
    invariant
      exists* (eWarpTile: chest2 et (wm*tm) (wn*tn)).
        warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile **
          pure (!i <= wm /\
            eWarpTile %~ (em_fade_comb_tiles comb tm tn wm wn !i 0 rCtile rAcc))
    decreases (wm - !i)
  {
    let mut j = 0sz;
    while (!j <^ wn)
      invariant live j
      invariant
        exists* (eWarpTile: chest2 et (wm*tm) (wn*tn)).
          warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile **
            pure (!i <= wm /\ !j <= wn /\
              eWarpTile %~ (em_fade_comb_tiles comb tm tn wm wn !i !j rCtile rAcc))
      decreases (wn - !j)
    {
      with eWarpTile. assert warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile;
      unfold warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile;

      let tile_for_tc_tiles = warp_tile (block_tile gC (SZ.v bm) (SZ.v bn) (SZ.v bid)) (wm*tm) (wn*tn) (SZ.v wid);
      rewrite each _ as tile_for_tc_tiles;

      FStar.Math.Lemmas.cancel_mul_div wm tm;
      FStar.Math.Lemmas.cancel_mul_div wn tn;
      let tc_tile = array2_extract_tile_st tile_for_tc_tiles (SZ.v tm) (SZ.v tn) (SZ.v !i) (SZ.v !j);

      let vi = !i;
      let vj = !j;
      let eidx : erased nat = vi * wn + vj;

      assert pure (vi < wm);
      assert pure (vj < wn);
      flat_fragment_index_bound wm wn vi vj;
      assert pure (eidx < wm * wn);
      assert pure (SZ.fits eidx);
      let idx = !i *^ wn +^ !j;

      unfold fragarrayAcc_approximates wm wn accumFrags rAcc;
      with eAccumFrags. assert accumFrags `array_fragment_pts_to` eAccumFrags;

      array_fragment_pts_to_ref accumFrags;
      array_fragment_extract_ro accumFrags idx;
      // Read-modify-write: combine the resident C tile with the accumulator.
      mma_store_comb (fun acc old -> ecomb old acc) accumFrags.(idx) tc_tile;

      // The tile now holds the fused combine of the resident C tile [m0] with
      // the accumulator [f0]; instantiate the [array2_extract_tile_st] trade at
      // that combined value so the warp tile folds back correctly.
      Pulse.Lib.Forall.elim_forall
        (Chest.chest_comb ecomb
          (ematrix_subtile eWarpTile tm tn !i !j)
          (Seq.Base.index eAccumFrags idx));
      ambig_trade_elim ();
      ambig_trade_elim ();
      fold fragarrayAcc_approximates wm wn accumFrags rAcc;

      rewrite each tile_for_tc_tiles as _;
      with eWarpTile'. fold warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile';

      lemma_update_tile_fade_comb_approximates ecomb comb tm tn wm wn !i !j eWarpTile (Seq.index eAccumFrags idx) rCtile rAcc;

      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  with eWarpTile'.
    assert (warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile');
  // After the outer loop !i == wm, so the invariant gives
  //   eWarpTile' %~ em_fade_comb_tiles comb tm tn wm wn wm 0 rCtile rAcc.
  // With idxI = wm every tile is combined, so the fade collapses to
  //   chest_comb comb rCtile rAcc.
  em_fade_comb_tiles_full comb tm tn wm wn 0 rCtile rAcc;
  assert pure (eWarpTile' %~ Chest.chest_comb comb rCtile rAcc);

  fold warp_tile_approximates gC bm bn tm tn wm wn bid wid (Chest.chest_comb comb rCtile rAcc);
  ()
}
#pop-options

inline_for_extraction noextract
fn populate_acc_with_zero
  (#et : Type0) {| sc : scalar et, real_like et |}
  (tm tn tk wm wn : szp)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
requires
  live accumFrags
ensures
  fragarrayAcc_approximates wm wn accumFrags (const _ 0.0R)
{
  array_fragment_pts_to_ref accumFrags;

  let mut fi : sz = 0sz;
  while (!fi <^ wm*^wn)
    invariant
      live fi **
      (exists* (eAcc : seq (chest2 et tm tn)).
        accumFrags |-> eAcc **
        pure (
          Seq.length eAcc == wm*wn /\ !fi <= wm*wn  /\
          forall (i : natlt !fi).
            Seq.index eAcc i %~ const _ 0.0R))
    decreases (wm*^wn - !fi)
  {
    array_fragment_pts_to_ref accumFrags;
    array_fragment_extract accumFrags !fi;
    mma_fill accumFrags.(!fi) sc.zero;

    Pulse.Lib.Forall.elim_forall (fill_value sc.zero);
    ambig_trade_elim();

    fi := !fi +^ 1sz;
  };
  fold fragarrayAcc_approximates wm wn accumFrags (const _ 0.0R);
  ()
}
