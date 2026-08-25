module Kuiper.Kernel.GEMM.TensorCore2D.Populate

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 120"

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
module MU = Kuiper.Kernel.GEMM.Util

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.ProofSupport
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
inline_for_extraction noextract
fn populate_fragments_a
  (#et : Type0)
  {| scalar et, real_like et |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (frags : array (fragment et FragA tm tn tk FragLRM))
  (gm : array2 et (rm bm bk))
  (#em : chest2 et bm bk)
  (rm : chest2 real bm bk {em %~ rm})
  (emA : et -> et)
  (mapA : real -> real)
  (#_ : squash (MU.approx1 emA mapA))
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
  fragarrayA_approximates wm frags
    (ematrix_subtile (Chest.chest_map mapA rm) (wm*tm) tk arow dotIdx)
{
    tensor_pts_to_ref gm;
    array_fragment_pts_to_ref frags;

    let tile_for_tc_a_tiles =
      array2_extract_tile_ro' gm (wm*tm) (SZ.v tk) (SZ.v arow) (SZ.v dotIdx);
    FStar.Math.Lemmas.cancel_mul_div (SZ.v wm) (SZ.v tm);
    let mut i0 = 0sz;
    while (!i0 <^ wm)
      invariant live i0
      invariant
        (exists* ems.
          frags |-> ems **
          pure (Seq.length ems == wm /\ !i0 <= wm /\
            forall (i : natlt wm). i < !i0 ==>
              (Seq.index ems i) %~
                (ematrix_subtile
                  (ematrix_subtile (Chest.chest_map mapA rm) (wm*tm) tk arow dotIdx)
                  tm tk i 0)))
      decreases (wm - !i0)
    {
      let a_tile =
        array2_extract_tile_ro' tile_for_tc_a_tiles (SZ.v tm) (SZ.v tk) (SZ.v !i0) 0;
      array_fragment_extract frags !i0;

      mma_loadA_map emA frags.(!i0) a_tile;
      Pulse.Lib.Forall.elim_forall
        (Chest.chest_map emA
          (ematrix_subtile (ematrix_subtile em (wm*tm) tk arow dotIdx) tm tk !i0 0));

      // Preserve the nested tile structure used by the postcondition. This
      // avoids flattening [arow*wm+i] and its nonlinear index arithmetic.
      subtile_approx em rm (wm*tm) tk arow dotIdx;
      subtile_approx
        (ematrix_subtile em (wm*tm) tk arow dotIdx)
        (ematrix_subtile rm (wm*tm) tk arow dotIdx)
        tm tk !i0 0;
      MU.chest_map_approx emA mapA
        (ematrix_subtile (ematrix_subtile em (wm*tm) tk arow dotIdx) tm tk !i0 0)
        (ematrix_subtile (ematrix_subtile rm (wm*tm) tk arow dotIdx) tm tk !i0 0);
      chest_map_subtile_comm mapA rm (wm*tm) tk arow dotIdx;
      chest_map_subtile_comm mapA
        (ematrix_subtile rm (wm*tm) tk arow dotIdx) tm tk !i0 0;

      ambig_trade_elim ();
      ambig_trade_elim ();

      i0 := !i0 +^ 1sz;
    };
    ambig_trade_elim ();
    fold fragarrayA_approximates wm frags
      (ematrix_subtile (Chest.chest_map mapA rm) (wm*tm) tk arow dotIdx);
    ()
}

#set-options "--z3rlimit 20"
inline_for_extraction noextract
fn populate_fragments_b
  (#et : Type0)
  {| scalar et, real_like et |}
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (frags : array (fragment et FragB tm tn tk FragLRM))
  (gm : array2 et (rm bk bn))
  (#em : chest2 et bk bn)
  (rm : chest2 real bk bn {em %~ rm})
  (emB : et -> et)
  (mapB : real -> real)
  (#_ : squash (MU.approx1 emB mapB))
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
  fragarrayB_approximates wn frags
    (ematrix_subtile (Chest.chest_map mapB rm) tk (wn*tn) dotIdx bcol)
{
    tensor_pts_to_ref gm;
    array_fragment_pts_to_ref frags;

    let tile_for_tc_b_tiles = array2_extract_tile_ro' gm (SZ.v tk) (wn*tn) (SZ.v dotIdx) (SZ.v bcol);
    FStar.Math.Lemmas.cancel_mul_div (SZ.v wn) (SZ.v tn);
    let mut i1 = 0sz;
    while (!i1 <^ wn)
      invariant live i1
      invariant
        (exists* ems.
          frags |-> ems **
          pure (Seq.length ems == wn /\ !i1 <= wn /\
            forall (i : natlt wn). i < !i1 ==>
              (Seq.index ems i) %~
                (ematrix_subtile
                  (ematrix_subtile (Chest.chest_map mapB rm) tk (wn*tn) dotIdx bcol)
                  tk tn 0 i)))
      decreases (wn - !i1)
    {
      let b_tile = array2_extract_tile_ro' tile_for_tc_b_tiles (SZ.v tk) (SZ.v tn) 0 (SZ.v !i1);

      array_fragment_pts_to_ref frags;
      array_fragment_extract frags !i1;

      mma_loadB_map emB frags.(!i1) b_tile;
      Pulse.Lib.Forall.elim_forall
        (Chest.chest_map emB
          (ematrix_subtile (ematrix_subtile em tk (wn*tn) dotIdx bcol) tk tn 0 !i1));

      subtile_approx em rm tk (wn*tn) dotIdx bcol;
      subtile_approx
        (ematrix_subtile em tk (wn*tn) dotIdx bcol)
        (ematrix_subtile rm tk (wn*tn) dotIdx bcol)
        tk tn 0 !i1;
      MU.chest_map_approx emB mapB
        (ematrix_subtile (ematrix_subtile em tk (wn*tn) dotIdx bcol) tk tn 0 !i1)
        (ematrix_subtile (ematrix_subtile rm tk (wn*tn) dotIdx bcol) tk tn 0 !i1);
      chest_map_subtile_comm mapB rm tk (wn*tn) dotIdx bcol;
      chest_map_subtile_comm mapB
        (ematrix_subtile rm tk (wn*tn) dotIdx bcol) tk tn 0 !i1;

      ambig_trade_elim ();
      ambig_trade_elim ();

      i1 := !i1 +^ 1sz;
    };
    ambig_trade_elim ();
    fold fragarrayB_approximates wn frags
      (ematrix_subtile (Chest.chest_map mapB rm) tk (wn*tn) dotIdx bcol);
    ()
}
