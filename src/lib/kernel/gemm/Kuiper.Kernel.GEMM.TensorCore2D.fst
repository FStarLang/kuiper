module Kuiper.Kernel.GEMM.TensorCore2D

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
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module CV2 = Kuiper.Kernel.GEMM.Copy.Vec2
module Chest = Kuiper.Chest
module MU = Kuiper.Kernel.GEMM.Util

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc

let fragarrayAcc_approximates (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wm wn : nat)
  ([@@@mkey] arr : array (fragment et FragAcc tm tn tk FragLAcc) { Pulse.Lib.Array.length arr == wm*wn})
  (rm : chest2 real (wm*tm) (wn*tn))
  : slprop
  =
    exists* (em : seq (chest2 et tm tn)).
      arr |-> em **
      pure (
        (Seq.length em == wm*wn) /\
        forall (i : natlt wm) (j : natlt wn). (Seq.index em (i * wn + j)) %~ (ematrix_subtile rm tm tn i j))

let fragarrayA_approximates (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wm : nat)
  (arr : array (fragment et FragA tm tn tk FragLRM) { Pulse.Lib.Array.length arr == wm})
  (rm : chest2 real (wm*tm) tk)
  : slprop
  =
    exists* (eAs : seq (chest2 et tm tk)).
      arr |-> eAs **
      pure (
        (Seq.length eAs == wm) /\
        forall (i : natlt wm).
          (Seq.index eAs i) %~ (ematrix_subtile rm tm tk i 0))

let fragarrayB_approximates (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wn : nat)
  (arr : array (fragment et FragB tm tn tk FragLRM) { Pulse.Lib.Array.length arr == wn})
  (rm : chest2 real tk (wn*tn))
  : slprop
  =
    exists* (eBs : seq (chest2 et tk tn)).
      arr |-> eBs **
      pure (
        (Seq.length eBs == wn) /\
        forall (i : natlt wn).
          (Seq.index eBs i) %~ (ematrix_subtile rm tk tn 0 i))

(* [chest_map] commutes with [ematrix_subtile]: mapping every element of a
   matrix and then extracting a subtile equals extracting the subtile of the
   mapped matrix.  Both sides are cellwise [f (acc2 em (..))].  Used as the
   bridge between the device-side element map [emA]/[emB] (applied to a loaded
   sub-tile) and the [chest_map mapA]/[chest_map mapB] form that the per-warp
   accumulator target uses.  Registered as an SMTPat that normalizes the
   [ematrix_subtile (chest_map ..)] ("map-outside") form to the
   [chest_map (ematrix_subtile ..)] ("map-inside") form. *)
let chest_map_subtile_comm
  (#et1 #et2 : Type0)
  (#rows #cols : nat)
  (f : et1 -> et2)
  (em : chest2 et1 rows cols)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (ematrix_subtile (Chest.chest_map f em) trows tcols tr tc
           == Chest.chest_map f (ematrix_subtile em trows tcols tr tc))
= Chest.ext (ematrix_subtile (Chest.chest_map f em) trows tcols tr tc)
            (Chest.chest_map f (ematrix_subtile em trows tcols tr tc))

(* Elementwise combine preserves approximation: if [approx2 ecomb comb] and the
   two operand chests approximate their real references, then their combined
   chest approximates the real combine.  Cellwise consequence of [approx2].
   Used at the epilogue to fuse the output combine. *)
let chest_comb_approx
  (#et : Type0) {| scalar et, real_like et |}
  (ecomb : et -> et -> et)
  (comb : real -> real -> real)
  (#rows #cols : nat)
  (e1 e2 : chest2 et rows cols)
  (r1 r2 : chest2 real rows cols)
  : Lemma
    (requires Kuiper.Approximates.approx2 ecomb comb /\ e1 %~ r1 /\ e2 %~ r2)
    (ensures Chest.chest_comb ecomb e1 e2 %~ Chest.chest_comb comb r1 r2)
= ()

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
    let mut i0 = 0sz;
    while (!i0 <^ wm)
      invariant live i0
      invariant
        (exists* ems.
          frags |-> ems **
          pure (Seq.length ems == wm /\ !i0 <= wm /\
            forall (i : natlt !i0).
              (Seq.index ems i) %~
                (ematrix_subtile (Chest.chest_map mapA rm) tm tk (arow*wm+i) dotIdx)))
      decreases (wm - !i0)
    {
      let a_tile =
        array2_extract_tile_ro' tile_for_tc_a_tiles (SZ.v tm) (SZ.v tk) (SZ.v !i0) 0;
      array_fragment_extract frags !i0;

      mma_loadA_map emA frags.(!i0) a_tile;
      Pulse.Lib.Forall.elim_forall
        (Chest.chest_map emA
          (ematrix_subtile (ematrix_subtile em (wm*tm) tk arow dotIdx) tm tk !i0 0));

      // The device map [emA] approximates the real map [mapA]; combined with the
      // fact that the loaded (unmapped) sub-tile approximates the real sub-tile
      // of [rm], the mapped fragment approximates [chest_map mapA] of the real
      // sub-tile, which equals the sub-tile of [chest_map mapA rm] (commutation).
      MU.chest_map_approx emA mapA
        (ematrix_subtile (ematrix_subtile em (wm*tm) tk arow dotIdx) tm tk !i0 0)
        (ematrix_subtile rm tm tk (arow*wm+ !i0) dotIdx);
      // Bridge chest_map (ematrix_subtile ..) to ematrix_subtile (chest_map ..)
      // (formerly an SMTPat; now applied explicitly to avoid E-matching blowup).
      chest_map_subtile_comm mapA rm tm tk (arow*wm+ !i0) dotIdx;

      ambig_trade_elim ();
      ambig_trade_elim ();

      i0 := !i0 +^ 1sz;
    };
    ambig_trade_elim ();
    fold fragarrayA_approximates wm frags
      (ematrix_subtile (Chest.chest_map mapA rm) (wm*tm) tk arow dotIdx);
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
    let mut i1 = 0sz;
    while (!i1 <^ wn)
      invariant live i1
      invariant
        (exists* ems.
          frags |-> ems **
          pure (Seq.length ems == wn /\ !i1 <= wn /\
            forall (i : natlt !i1).
              (Seq.index ems i) %~
                (ematrix_subtile (Chest.chest_map mapB rm) tk tn dotIdx (bcol*wn+i))))
      decreases (wn - !i1)
    {
      let b_tile = array2_extract_tile_ro' tile_for_tc_b_tiles (SZ.v tk) (SZ.v tn) 0 (SZ.v !i1);

      array_fragment_pts_to_ref frags;
      array_fragment_extract frags !i1;

      mma_loadB_map emB frags.(!i1) b_tile;
      Pulse.Lib.Forall.elim_forall
        (Chest.chest_map emB
          (ematrix_subtile (ematrix_subtile em tk (wn*tn) dotIdx bcol) tk tn 0 !i1));

      FStar.Math.Lemmas.paren_mul_right (SZ.v bcol) (SZ.v wn) (SZ.v tn);
      FStar.Math.Lemmas.distributivity_add_left (SZ.v bcol * SZ.v wn) (SZ.v !i1) (SZ.v tn);

      // Same reasoning as [populate_fragments_a]: mapped fragment approximates
      // [chest_map mapB] of the real sub-tile = sub-tile of [chest_map mapB rm].
      MU.chest_map_approx emB mapB
        (ematrix_subtile (ematrix_subtile em tk (wn*tn) dotIdx bcol) tk tn 0 !i1)
        (ematrix_subtile rm tk tn dotIdx (bcol*wn+ !i1));
      // Bridge chest_map (ematrix_subtile ..) to ematrix_subtile (chest_map ..).
      chest_map_subtile_comm mapB rm tk tn dotIdx (bcol*wn+ !i1);

      ambig_trade_elim ();
      ambig_trade_elim ();

      i1 := !i1 +^ 1sz;
    };
    ambig_trade_elim ();
    fold fragarrayB_approximates wn frags
      (ematrix_subtile (Chest.chest_map mapB rm) tk (wn*tn) dotIdx bcol);
    ()
}

let arrayfragments_fade
  (tm tn tk wm wn : szp)
  (i : natlt wm)
  (j : natlt wn)
  (resIdxM : natle wm)
  (resIdxN : natle wn)
  (rA : chest2 real (wm*tm) tk)
  (rB : chest2 real tk (wn*tn))
  (rAcc : chest2 real (wm*tm) (wn*tn))
: chest2 real tm tn
=
  if i < resIdxM || (i = resIdxM && j < resIdxN)
  then ematrix_subtile rAcc tm tn i j `matplus`
    (matmul (ematrix_subtile rA tm tk i 0) (ematrix_subtile rB tk tn 0 j))
  else ematrix_subtile rAcc tm tn i j

#push-options "--z3rlimit 80"
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
  (rA : chest2 real (wm*tm) tk)
  (rB : chest2 real tk (wn*tn))
  (rAcc : chest2 real (wm*tm) (wn*tn))
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
  while (!resIdxM <^ wm)
    invariant live resIdxM
    invariant
      exists* (eAcc : seq (chest2 et_acc tm tn)).
        accumFrags |-> eAcc **
        pure (
          !resIdxM <= wm /\
          (Seq.length eAcc == wm*wn) /\
          forall (i : natlt wm) (j : natlt wn).
            (Seq.index eAcc (i * wn + j)) %~
              (arrayfragments_fade tm tn tk wm wn i j !resIdxM 0 rA rB rAcc))
    decreases (wm - !resIdxM)
  {
    let mut resIdxN = 0sz;
    while (!resIdxN <^ wn)
      invariant live resIdxN
      invariant
        exists* (eAcc : seq (chest2 et_acc tm tn)).
          accumFrags |-> eAcc **
          pure (
            !resIdxN <= wn /\
            (Seq.length eAcc == wm*wn) /\
            forall (i : natlt wm) (j : natlt wn).
              (Seq.index eAcc (i * wn + j)) %~
                (arrayfragments_fade tm tn tk wm wn i j !resIdxM !resIdxN rA rB rAcc))
      decreases (wn - !resIdxN)
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
            (emma (Seq.index eAccs (!resIdxM * wn + !resIdxN))
                (Seq.index eAs !resIdxM)
                (Seq.index eBs !resIdxN)));

      resIdxN := !resIdxN +^ 1sz;
    };

    resIdxM := !resIdxM +^ 1sz;
  };

  with eAcc. assert accumFrags |-> eAcc;
  assert pure (
    forall (i : natlt wm) (j : natlt wn).
              (Seq.index eAcc (i * wn + j)) %~ (arrayfragments_fade tm tn tk wm wn i j wm 0 rA rB rAcc));
  assert pure (
    forall (i : natlt wm) (j : natlt wn).
      (Seq.index eAcc (i * wn + j)) %~
                ematrix_subtile rAcc tm tn i j `matplus`
                  (matmul (ematrix_subtile rA tm tk i 0) (ematrix_subtile rB tk tn 0 j)));

  assert pure (
    forall (i : natlt wm) (j : natlt wn).
              (Seq.index eAcc (i * wn + j)) %~ (ematrix_subtile (rAcc `matplus` (matmul rA rB)) tm tn i j));
  assert pure (Seq.length eAcc == wm*wn);

  fold fragarrayA_approximates wm aFrags rA;
  fold fragarrayB_approximates wn bFrags rB;
  fold fragarrayAcc_approximates wm wn accumFrags (rAcc `matplus` (matmul rA rB));
}
#pop-options

// Working around apparent bug below
inline_for_extraction noextract
let sz_succ (x:SZ.t{SZ.fits (x+1)}) : SZ.t = x +^ 1sz

// Helper lemma: proves the matmul accumulation step via extensional equality,
// working around the matplus normalization gap introduced by the chest2→chest
// refactor (matplus normalizes to Chest.M/on_domain_g in the slprop but the
// __gmatmul_single_lemma hypothesis keeps it opaque). By having ematrix_subtile
// directly in the ensures clause (not acc2), the conclusion normalizes
// consistently with the slprop when F* sends it to SMT.
#push-options "--z3rlimit 60"
let subproducts_step_eq
  (#bm #bk #bn : nat)
  (wm_tm tk_dim wn_tn : pos)
  (_ : squash (wm_tm /? bm /\ tk_dim /? bk /\ wn_tn /? bn))
  (rAcc : chest2 real wm_tm wn_tn)
  (rA : chest2 real bm bk) (rB : chest2 real bk bn)
  (row : natlt (bm/wm_tm)) (col : natlt (bn/wn_tn))
  (k : natlt (bk/tk_dim))
  : Lemma (
      matplus (__gmatmul_single rAcc matmul matplus
              (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
              row col k)
          (matmul (ematrix_subtile rA wm_tm tk_dim row k)
                  (ematrix_subtile rB tk_dim wn_tn k col))
      ==
      __gmatmul_single rAcc matmul matplus
          (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
          row col (k + 1))
  = __gmatmul_single_lemma rAcc matmul matplus
      (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
      row col (k + 1);
    macc_ematrix_tiled rA wm_tm tk_dim row k;
    macc_ematrix_tiled rB tk_dim wn_tn k col;
    assert (equal
      (matplus (__gmatmul_single rAcc matmul matplus
               (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
               row col k)
          (matmul (ematrix_subtile rA wm_tm tk_dim row k)
                  (ematrix_subtile rB tk_dim wn_tn k col)))
      (__gmatmul_single rAcc matmul matplus
          (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
          row col (k + 1)))
#pop-options

// Stateful function to advance fragarrayAcc_approximates by one matmul
// General ghost function to rewrite fragarrayAcc_approximates from mold
// to mnew, given that mold == mnew.
noextract
ghost fn rewrite_fragarrayAcc
  (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (wm wn : pos)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc)
                { Pulse.Lib.Array.length accumFrags == wm*wn })
  (mold mnew : chest2 real (wm*tm) (wn*tn))
  (#_ : squash (mold == mnew))
  requires fragarrayAcc_approximates wm wn accumFrags mold
  ensures fragarrayAcc_approximates wm wn accumFrags mnew
{
  ()
}

// Specialized ghost function for the inner-loop accumulation step.
// Calls the subproducts_step_eq lemma INSIDE the ghost fn body,
// then unfolds/folds with opaque params
// so the VC uses trivial congruence from the lemma's propositional equality.
#push-options "--z3rlimit 60"
noextract
ghost fn rewrite_fragarrayAcc_step
  (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (#bm #bk #bn : pos)
  (wm wn : pos)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc)
                { Pulse.Lib.Array.length accumFrags == wm*wn })
  (rAcc : chest2 real (wm*tm) (wn*tn))
  (rA : chest2 real bm bk) (rB : chest2 real bk bn)
  (arow : natlt (bm/(wm*tm))) (bcol : natlt (bn/(wn*tn)))
  (k : natlt (bk/tk))
  (#_ : squash ((wm*tm) /? bm /\ tk /? bk /\ (wn*tn) /? bn))
  requires fragarrayAcc_approximates wm wn accumFrags
    (matplus (__gmatmul_single rAcc matmul matplus
              (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
              arow bcol k)
            (matmul (ematrix_subtile rA (wm*tm) tk arow k)
                    (ematrix_subtile rB tk (wn*tn) k bcol)))
  ensures fragarrayAcc_approximates wm wn accumFrags
    (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
      arow bcol (k + 1))
{
  subproducts_step_eq (wm*tm) tk (wn*tn) () rAcc rA rB arow bcol k;
  unfold fragarrayAcc_approximates wm wn accumFrags;
  fold fragarrayAcc_approximates wm wn accumFrags
    (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
      arow bcol (k + 1));
}
#pop-options

// Ghost function to rewrite fragarrayAcc_approximates from the final
// tiled gmatmul_single form to the matplus/matmul form (post-loop).
noextract
ghost fn rewrite_fragarrayAcc_tiles
  (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (#bm #bk #bn : pos)
  (wm wn : pos)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc)
                { Pulse.Lib.Array.length accumFrags == wm*wn })
  (rAcc : chest2 real (wm*tm) (wn*tn))
  (rA : chest2 real bm bk) (rB : chest2 real bk bn)
  (arow : natlt (bm/(wm*tm))) (bcol : natlt (bn/(wn*tn)))
  (#_ : squash ((wm*tm) /? bm /\ tk /? bk /\ (wn*tn) /? bn))
  requires fragarrayAcc_approximates wm wn accumFrags
    (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
      arow bcol (bk/tk))
  ensures fragarrayAcc_approximates wm wn accumFrags
    (rAcc `matplus` matmul (ematrix_subtile rA (wm*tm) bk arow 0)
                           (ematrix_subtile rB bk (wn*tn) 0 bcol))
{
  matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ())
    (wm*tm) (wn*tn) tk rAcc rA rB arow bcol;
  unfold fragarrayAcc_approximates wm wn accumFrags;
  fold fragarrayAcc_approximates wm wn accumFrags
    (rAcc `matplus` matmul (ematrix_subtile rA (wm*tm) bk arow 0)
                           (ematrix_subtile rB bk (wn*tn) 0 bcol));
}

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
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
  (gA : array2 et_ab (rm bm bk))
  (gB : array2 et_ab (rm bk bn))
  (#eA : chest2 et_ab bm bk)
  (#eB : chest2 et_ab bk bn)
  (rA : chest2 real bm bk {eA %~ rA})
  (rB : chest2 real bk bn {eB %~ rB})
  (emA emB : et_ab -> et_ab)
  (mapA mapB : real -> real)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (rAcc : chest2 real (wm*tm) (wn*tn))
  (#fA #fB : perm)
  (arow : szlt (bm/(wm*tm)))
  (bcol : szlt (bn/(wn*tn)))
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
      (rAcc `matplus` matmul (ematrix_subtile (Chest.chest_map mapA rA) (wm*tm) bk arow 0)
                             (ematrix_subtile (Chest.chest_map mapB rB) bk (wn*tn) 0 bcol))
{
  rewrite each rAcc
  as __gmatmul_single rAcc matmul matplus
      (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) tk)
      (ematrix_tiled (Chest.chest_map mapB rB) tk (wn*tn)) arow bcol 0;

  let mut dotIdx : sz = 0sz;
  while (!dotIdx <^ (bk/^tk))
    invariant live aFrags ** live bFrags
    invariant
      exists* (vdotIdx : sz { vdotIdx <= (bk/tk) }).
        dotIdx |-> vdotIdx **
        fragarrayAcc_approximates wm wn accumFrags
          (__gmatmul_single rAcc matmul matplus
            (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) tk)
            (ematrix_tiled (Chest.chest_map mapB rB) tk (wn*tn)) arow bcol !dotIdx)
    decreases (bk/^tk - !dotIdx)
  {
    populate_fragments_a bm bn bk tm tn tk wm wn aFrags gA rA emA mapA arow !dotIdx;
    populate_fragments_b bm bn bk tm tn tk wm wn bFrags gB rB emB mapB bcol !dotIdx;

    fragarray_mma bm bn bk tm tn tk wm wn aFrags bFrags accumFrags
      (ematrix_subtile (Chest.chest_map mapA rA) (wm*tm) tk arow !dotIdx)
      (ematrix_subtile (Chest.chest_map mapB rB) tk (wn*tn) !dotIdx bcol)
      (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) tk)
      (ematrix_tiled (Chest.chest_map mapB rB) tk (wn*tn)) arow bcol !dotIdx)
      !dotIdx;

    unfold fragarrayA_approximates wm aFrags;
    unfold fragarrayB_approximates wn bFrags;

    // Ghost fn: advance the accumulator by one step — the lemma call and
    // unfold/fold happen inside the ghost fn with opaque params.
    // Pass !dotIdx directly (stt read), NOT a with-bound variable (ghost).
    rewrite_fragarrayAcc_step wm wn accumFrags rAcc
      (Chest.chest_map mapA rA) (Chest.chest_map mapB rB) arow bcol !dotIdx;

    with vdi. assert dotIdx |-> vdi;
    dotIdx := sz_succ !dotIdx;
    rewrite each
      (SZ.v vdi + 1)
    as
      (SZ.v (sz_succ vdi));

    ()
  };

  assert pure (!dotIdx == bk/^tk);
  assert pure (SZ.v (bk/^tk) == bk/tk);
  with vdotIdx. assert (dotIdx |-> vdotIdx ** pure (vdotIdx == bk/^tk));

  rewrite each vdotIdx as (bk/^tk);
  assert (fragarrayAcc_approximates wm wn accumFrags
    (__gmatmul_single rAcc matmul matplus
          (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) tk)
          (ematrix_tiled (Chest.chest_map mapB rB) tk (wn*tn))
          arow
          bcol
          (bk/^tk)));

  rewrite_fragarrayAcc_tiles wm wn accumFrags rAcc
    (Chest.chest_map mapA rA) (Chest.chest_map mapB rB) arow bcol;
  ()
}

let em_fade_tiles
  (tm tn wm wn : pos)
  (idxI : natle wm)
  (idxJ : natle wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: chest2 real (wm*tm) (wn*tn)
=
  ematrix_from_tiles tm tn (fun i j ->
    let flat_idx = i * wn + j in
    let num_copied = idxI * wn + idxJ in
    if flat_idx < num_copied
    then ematrix_subtile rm2 tm tn i j
    else ematrix_subtile rm1 tm tn i j)

// Once idxI reaches wm, every output tile index (i,j) satisfies
// i*wn+j < wm*wn <= wm*wn+idxJ, so the fade selects rm2 everywhere and the
// whole matrix collapses to rm2 (regardless of idxJ). Proven extensionally so
// that the chest2 stays opaque: from_subtiles_id no longer fires syntactically
// after the ematrix->chest refactor, so we bridge via `equal`.
#push-options "--z3rlimit 40"
let em_fade_tiles_full
  (tm tn wm wn : pos)
  (idxJ : natle wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: Lemma (em_fade_tiles tm tn wm wn wm idxJ rm1 rm2 == rm2)
= assert (equal (em_fade_tiles tm tn wm wn wm idxJ rm1 rm2) rm2)
#pop-options

#push-options "--z3rlimit 80 --split_queries always"
let lemma_update_tile_fade_approximates
  (#et : Type0) {| scalar et, real_like et|}
  (tm tn wm wn : pos)
  (idxI : natlt wm)
  (idxJ : natlt wn)
  (em : chest2 et (wm*tm) (wn*tn))
  (etile : chest2 et tm tn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: Lemma
  (requires
    (em %~ (em_fade_tiles tm tn wm wn idxI idxJ rm1 rm2)) /\
    (etile %~ (ematrix_subtile rm2 tm tn idxI idxJ)))
  (ensures (update_tile em tm tn idxI idxJ etile) %~ (em_fade_tiles tm tn wm wn idxI (idxJ + 1) rm1 rm2))
=
  () // would be nice to spell it out probably
#pop-options

(* Combine-fade variant of [em_fade_tiles]: after storing tiles up to
   (idxI, idxJ) with the fused output combine, each already-written tile holds
   [chest_comb comb] of the old C tile ([rm1]) and the accumulator tile ([rm2]);
   the remaining tiles still hold the original C ([rm1]). *)
let em_fade_comb_tiles
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxI : natle wm)
  (idxJ : natle wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: chest2 real (wm*tm) (wn*tn)
=
  ematrix_from_tiles tm tn (fun i j ->
    let flat_idx = i * wn + j in
    let num_copied = idxI * wn + idxJ in
    if flat_idx < num_copied
    then Chest.chest_comb comb (ematrix_subtile rm1 tm tn i j) (ematrix_subtile rm2 tm tn i j)
    else ematrix_subtile rm1 tm tn i j)

// Once idxI reaches wm every tile is combined, so the whole matrix collapses to
// [chest_comb comb rm1 rm2].  Proven extensionally, as [em_fade_tiles_full].
#push-options "--z3rlimit 40"
let em_fade_comb_tiles_full
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxJ : natle wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: Lemma (em_fade_comb_tiles comb tm tn wm wn wm idxJ rm1 rm2 == Chest.chest_comb comb rm1 rm2)
= assert (equal (em_fade_comb_tiles comb tm tn wm wn wm idxJ rm1 rm2) (Chest.chest_comb comb rm1 rm2))
#pop-options

#push-options "--z3rlimit 80 --split_queries always"
let lemma_update_tile_fade_comb_approximates
  (#et : Type0) {| scalar et, real_like et|}
  (ecomb : et -> et -> et)
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxI : natlt wm)
  (idxJ : natlt wn)
  (em : chest2 et (wm*tm) (wn*tn))
  (etile : chest2 et tm tn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: Lemma
  (requires
    Kuiper.Approximates.approx2 ecomb comb /\
    (em %~ (em_fade_comb_tiles comb tm tn wm wn idxI idxJ rm1 rm2)) /\
    (etile %~ (ematrix_subtile rm2 tm tn idxI idxJ)))
  (ensures
    (update_tile em tm tn idxI idxJ
      (Chest.chest_comb ecomb (ematrix_subtile em tm tn idxI idxJ) etile))
    %~ (em_fade_comb_tiles comb tm tn wm wn idxI (idxJ + 1) rm1 rm2))
=
  // The (idxI, idxJ) tile is still uncombined in the pre-state fade, so its
  // subtile of [em] approximates [ematrix_subtile rm1 tm tn idxI idxJ].
  chest_comb_approx ecomb comb
    (ematrix_subtile em tm tn idxI idxJ) etile
    (ematrix_subtile rm1 tm tn idxI idxJ) (ematrix_subtile rm2 tm tn idxI idxJ);
  ()
#pop-options

#restart-solver
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

      let tc_tile = array2_extract_tile_st tile_for_tc_tiles (SZ.v tm) (SZ.v tn) (SZ.v !i) (SZ.v !j);

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
      // Read-modify-write: combine the resident C tile with the accumulator.
      mma_store_comb ecomb accumFrags.(idx) tc_tile;

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

#push-options "--z3rlimit 100"
let loop_invariant_lemma
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm/(wm*tm)))
  (warpCol : natlt (bn/(wn*tn)))
  (gwRow : natlt (m/(wm*tm)) { gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : natlt (n/(wn*tn)) { gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (vk : natlt (k / bk))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rAcc0 : chest2 real (wm*tm) (wn*tn) { rAcc0 == const _ 0.0R })
  (rAcc  : chest2 real (wm*tm) (wn*tn))
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (#_ : squash (rAcc  ==
          (__gmatmul_single rAcc0 matmul matplus
            (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn))
              gwRow
              gwCol
              vk)))
  (rA_sub : chest2 real bm bk { rA_sub == ematrix_subtile rA bm bk mrow vk })
  (rB_sub : chest2 real bk bn { rB_sub == ematrix_subtile rB bk bn vk mcol })
: Lemma (
        rAcc `matplus` matmul (ematrix_subtile rA_sub (wm*tm) bk warpRow 0)
                               (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol)
        ==
        __gmatmul_single rAcc0 matmul matplus
          (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) gwRow gwCol (vk + 1)
    )
= let lhs : chest2 real (wm*tm) (wn*tn) = rAcc `matplus` matmul (ematrix_subtile rA_sub (wm*tm) bk warpRow 0)
                                   (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol) in
  let rhs : chest2 real (wm*tm) (wn*tn) =
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
                  acc2 (ematrix_tiled rA (wm*tm) bk) gwRow vk
                )
  = macc_ematrix_tiled rA (wm*tm) bk gwRow vk;
    assert (ematrix_subtile rA_sub (wm*tm) bk warpRow 0
            `equal` acc2 (ematrix_tiled rA (wm*tm) bk) gwRow vk)
  in
  let aux2 () : Lemma (
                  ematrix_subtile rB_sub bk (wn*tn) 0 warpCol
                  ==
                  acc2 (ematrix_tiled rB bk (wn*tn)) vk gwCol
                )
  = macc_ematrix_tiled rB bk (wn*tn) vk gwCol;
    assert (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol
            `equal` acc2 (ematrix_tiled rB bk (wn*tn)) vk gwCol)
  in
  aux1 ();
  aux2 ();

  let aux (i : natlt (wm*tm)) (j : natlt (wn*tn))
    : Lemma (acc2 lhs i j == acc2 rhs i j)
    = calc (==) {
        acc2 lhs i j;
        == {}
        acc2 (__gmatmul_single rAcc0 matmul matplus
               (ematrix_tiled rA (wm*tm) bk)
               (ematrix_tiled rB bk (wn*tn)) gwRow gwCol vk
              `matplus`
                 matmul (ematrix_subtile rA_sub (wm*tm) bk warpRow 0)
                        (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol)) i j;
        == {}
        acc2 (__gmatmul_single rAcc0 matmul matplus
               (ematrix_tiled rA (wm*tm) bk)
               (ematrix_tiled rB bk (wn*tn)) gwRow gwCol vk
              `matplus`
                 matmul (acc2 (ematrix_tiled rA (wm*tm) bk) gwRow vk)
                        (acc2 (ematrix_tiled rB bk (wn*tn)) vk gwCol)) i j;
        == { __gmatmul_single_lemma rAcc0 matmul matplus
               (ematrix_tiled rA (wm*tm) bk)
               (ematrix_tiled rB bk (wn*tn)) gwRow gwCol (vk + 1) }
        acc2 (__gmatmul_single rAcc0 matmul matplus
               (ematrix_tiled rA (wm*tm) bk)
               (ematrix_tiled rB bk (wn*tn)) gwRow gwCol (vk+1)) i j;
        == {}
        acc2 rhs i j;
      }
  in
  Classical.forall_intro_2 aux;
  assert (Kuiper.EMatrix.equal lhs rhs);
  ()
#pop-options

// Plain-`let` wrapper around the per-warp k-loop accumulator invariant.
// Its internal well-typedness obligations (ematrix_tiled divisibility,
// __gmatmul_single's row/col/to refinements) are discharged HERE, at the
// definition, exactly like [loop_invariant_lemma] discharges the same terms.
// Everywhere else this appears as a simple function application whose result
// type is read off the signature -- crucially, this lets the Pulse ghost fn
// [advance_kloop_invariant] mention the invariant in its requires/ensures
// binders without Pulse re-deriving __gmatmul_single's refinements (which it
// could not complete: "incomplete quantifiers" on the raw term in a binder).
// [kacc_inv] is [@@ "opaque_to_smt"]: it appears in kf's loop invariant and in
// [ktile_advance]'s pre/post, both checked in kf's LARGE ambient context.  Were
// it a transparent [let], Z3 would eagerly unfold it to the [__gmatmul_single]
// body, whose [acc2 (ematrix_tiled (chest_map ..))] subterms fire the
// [macc_ematrix_tiled] SMTPat and drive a non-terminating (multi-GB) E-matching
// blowup.  Keeping it opaque makes the invariant/framing checks purely
// syntactic on the folded symbol; the definition is exposed ONLY at the few
// ground points that need it, per index, via [kacc_inv_eq] / [reveal_opaque].
#push-options "--z3rlimit 100"
[@@ "opaque_to_smt"]
inline_for_extraction noextract
let kacc_inv
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (mrA : chest2 real m k)
  (mrB : chest2 real k n)
  (rAcc0 : chest2 real (wm*tm) (wn*tn))
  (gwRow : natlt (m/(wm*tm)))
  (gwCol : natlt (n/(wn*tn)))
  (vk : nat { vk <= k / bk })
: chest2 real (wm*tm) (wn*tn)
= __gmatmul_single rAcc0 matmul matplus
    (ematrix_tiled mrA (wm*tm) bk) (ematrix_tiled mrB bk (wn*tn)) gwRow gwCol vk
#pop-options

// Ground fact exposing [kacc_inv]'s (opaque) definition for one concrete index.
// Invoked at exactly the points where the invariant must connect to the raw
// [__gmatmul_single] machinery (advance_kloop_invariant's proof, kf's initial
// rewrite, kf's post-loop rewrite) -- and NOWHERE inside kf's loop body, so the
// [macc_ematrix_tiled] blowup stays contained.
let kacc_inv_eq
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (mrA : chest2 real m k)
  (mrB : chest2 real k n)
  (rAcc0 : chest2 real (wm*tm) (wn*tn))
  (gwRow : natlt (m/(wm*tm)))
  (gwCol : natlt (n/(wn*tn)))
  (vk : nat { vk <= k / bk })
: Lemma (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk
           == __gmatmul_single rAcc0 matmul matplus
                (ematrix_tiled mrA (wm*tm) bk) (ematrix_tiled mrB bk (wn*tn))
                gwRow gwCol vk)
= reveal_opaque (`%kacc_inv)
    (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk)

// Encapsulates the per-k-tile accumulator advance (subproducts' postcondition ->
// next loop invariant) in a SMALL context.  The heavy [loop_invariant_lemma]
// refinement discharge and the [chest_map_subtile_comm] SMTPat only see this
// fn's few parameters here, so E-matching stays bounded -- inlining this glue
// into [kf]'s large context caused a multi-GB Z3 blowup.
#push-options "--z3rlimit 100"
noextract
ghost fn advance_kloop_invariant
  (#et:Type0) {| scalar et, real_like et |}
  (m n k : nat)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm/(wm*tm)))
  (warpCol : natlt (bn/(wn*tn)))
  (gwRow : natlt (m/(wm*tm)) { gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : natlt (n/(wn*tn)) { gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (vk : natlt (k / bk))
  // The matrices here are already map-applied (e.g. [mrA = chest_map mapA rA]).
  // Taking them as plain [chest2 real] params (mirroring loop_invariant_lemma)
  // keeps the [ematrix_tiled] dimensions trivially well-typed; the [chest_map]/
  // subtile commutation is handled once, cheaply, at the (single) call site.
  (mrA : chest2 real m k)
  (mrB : chest2 real k n)
  (rAcc0 : chest2 real (wm*tm) (wn*tn) { rAcc0 == const _ 0.0R })
  (rAcc  : chest2 real (wm*tm) (wn*tn))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  // The invariant is stated via the [kacc_inv] wrapper, NOT the raw
  // [__gmatmul_single] application: Pulse cannot complete the well-typedness of
  // the raw term inside a binder ("incomplete quantifiers"), whereas [kacc_inv]
  // is a plain function whose result type comes straight from its signature.
  (#_ : squash (rAcc ==
          kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk))
  (mrA_sub : chest2 real bm bk { mrA_sub == ematrix_subtile mrA bm bk mrow vk })
  (mrB_sub : chest2 real bk bn { mrB_sub == ematrix_subtile mrB bk bn vk mcol })
  (accFrags : array (fragment et FragAcc tm tn tk FragLAcc)
              { Pulse.Lib.Array.length accFrags == wm*wn })
  requires
    fragarrayAcc_approximates wm wn accFrags
      (rAcc `matplus`
        matmul (ematrix_subtile mrA_sub (wm*tm) bk warpRow 0)
               (ematrix_subtile mrB_sub bk (wn*tn) 0 warpCol))
  ensures
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol (vk + 1))
{
  // [kacc_inv] is opaque_to_smt; expose it at [vk] so the incoming
  // [rAcc == kacc_inv .. vk] hypothesis connects to [loop_invariant_lemma]'s raw
  // [__gmatmul_single .. vk] form, and at [vk+1] for the closing rewrite.
  kacc_inv_eq m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol vk;
  kacc_inv_eq m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol (vk + 1);

  loop_invariant_lemma
    m n k
    bm bn bk tm tn tk wm wn
    mrow mcol
    warpRow warpCol
    gwRow gwCol
    vk
    mrA mrB
    rAcc0 rAcc
    mrA_sub mrB_sub;

  // [loop_invariant_lemma] proves the target as the raw __gmatmul_single form;
  // [kacc_inv ... (vk+1)] equals exactly that term (via [kacc_inv_eq] above).
  rewrite_fragarrayAcc wm wn accFrags
    (rAcc `matplus`
      matmul (ematrix_subtile mrA_sub (wm*tm) bk warpRow 0)
             (ematrix_subtile mrB_sub bk (wn*tn) 0 warpCol))
    (kacc_inv m n k bm bn bk tm tn tk wm wn mrA mrB rAcc0 gwRow gwCol (vk + 1));
}
#pop-options

// Ground fact: a subtile of an approximating matrix approximates the
// corresponding subtile of the reference.  Proving [eA_sub %~ rA_sub] (between
// two [ematrix_subtile] terms) DIRECTLY at the [ktile_advance] call site fires
// [lemma_approximates_intro]'s [SMTPat (m1 %~ m2)], which expands to
// [acc2 (ematrix_subtile ..)] terms and cascades into a non-terminating Z3
// E-matching blowup in kf's large context.  Instead, kf calls this lemma (proved
// in a CLEAN context) to obtain [eA_sub %~ rA_sub] as a ground hypothesis, so the
// call-site refinement discharges by assumption without triggering the cascade.
let subtile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (rm : chest2 real rows cols)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (requires em %~ rm)
          (ensures ematrix_subtile em trows tcols tr tc
                   %~ ematrix_subtile rm trows tcols tr tc)
= lemma_approximates_intro
    (ematrix_subtile em trows tcols tr tc)
    (ematrix_subtile rm trows tcols tr tc)

// Wraps the per-k-tile compute step (subproducts_tc_2d) TOGETHER with the
// ghost accumulator advance, exposing a pre/post stated ONLY in the clean
// [kacc_inv] (ematrix_tiled) form.  The map-aware generalization made
// subproducts' postcondition mention [ematrix_subtile (chest_map mapX ..)],
// which drives the [chest_map_subtile_comm] SMTPat.  Framing that call in kf's
// large ambient context (barrier / shared-memory / bp_sharing predicates, plus
// many ematrix_subtile terms) caused a multi-GB Z3 E-matching blowup.  By
// isolating subproducts' framing to this helper's SMALL context -- and keeping
// [chest_map]-subtile terms entirely OUT of kf (the helper's pre/post use
// [kacc_inv], which unfolds to [ematrix_tiled]-based __gmatmul_single, never
// [ematrix_subtile (chest_map ..)]) -- the SMTPat can no longer explode.
#push-options "--z3rlimit 100"
#restart-solver
inline_for_extraction noextract
fn ktile_advance
  (#et_ab #et_c : Type0)
  {| scalar et_ab, scalar et_c, real_like et_ab, real_like et_c |}
  // NOTE: m n k are [szp] (sizet), NOT [nat].  kf's dims are sizet; the
  // sz->nat coercion [sizet_to_nat] is GTot, so taking [nat] params here would
  // make kf's stateful call to this fn GHOST (Pulse "Application of a stateful
  // computation cannot have a ghost effect").  subproducts_tc_2d uses szp/szlt
  // for the same reason.  Coercions to nat happen inside slprops / ghost-fn
  // arguments below, where GTot is fine.
  (m n k : szp)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (aFrags     : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags     : array (fragment et_ab FragB tm tn tk FragLRM))
  (accFrags   : array (fragment et_c FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm*wn))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#eA_sub : chest2 et_ab bm bk)
  (#eB_sub : chest2 et_ab bk bn)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (vk : szlt (k / bk))
  (rA_sub : chest2 real bm bk
            { eA_sub %~ rA_sub /\ rA_sub == ematrix_subtile rA bm bk mrow vk })
  (rB_sub : chest2 real bk bn
            { eB_sub %~ rB_sub /\ rB_sub == ematrix_subtile rB bk bn vk mcol })
  (emA emB : et_ab -> et_ab)
  (mapA mapB : real -> real)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (rAcc0 : chest2 real (wm*tm) (wn*tn) { rAcc0 == const _ 0.0R })
  (warpRow : szlt (bm/(wm*tm)))
  (warpCol : szlt (bn/(wn*tn)))
  // NOTE: gwRow/gwCol are [enatlt] (ERASED) to match kf's erased locals.  If
  // they were informative [natlt], passing kf's erased gwRow/gwCol here would
  // require a [reveal], which gives this STATEFUL application a ghost effect
  // (Pulse "Application of a stateful computation cannot have a ghost effect").
  (gwRow : enatlt (m/(wm*tm)) { gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : enatlt (n/(wn*tn)) { gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (#fA #fB : perm)
  ()
  preserves gpu
  preserves sA |-> Frac fA eA_sub
  preserves sB |-> Frac fB eB_sub
  preserves live aFrags ** live bFrags
  requires
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol vk)
  ensures
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol (vk + 1))
{
  with rAcc. assert fragarrayAcc_approximates wm wn accFrags rAcc;

  subproducts_tc_2d bm bn bk tm tn tk wm wn aFrags bFrags accFrags
    sA sB
    rA_sub rB_sub
    emA emB mapA mapB
    rAcc
    warpRow warpCol;

  // advance_kloop_invariant needs its subtile args in [ematrix_subtile (chest_map
  // ..) ..] form, but kf supplies them as [chest_map .. (ematrix_subtile ..)].
  // Bridge with the commutation lemma explicitly (its SMTPat was removed to avoid
  // an E-matching blowup, so it must be invoked by hand here).
  chest_map_subtile_comm mapA rA bm bk mrow vk;
  chest_map_subtile_comm mapB rB bk bn vk mcol;

  advance_kloop_invariant
    m n k
    bm bn bk tm tn tk wm wn
    mrow mcol
    warpRow warpCol
    gwRow gwCol
    vk
    (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
    rAcc0 rAcc
    (Chest.chest_map mapA rA_sub) (Chest.chest_map mapB rB_sub)
    accFrags;
}
#pop-options

#restart-solver
inline_for_extraction noextract
fn kf
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, sc : scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab k n)
  (gC : array2 et_c (rm m n))
  (#eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#fA #fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  // Fused pre-maps / combine (real domain) and their device realizations.
  (mapA mapB : real -> real)
  (comb : real -> real -> real)
  (emA emB : et_ab -> et_ab)
  (ecomb : et_c -> et_c -> et_c)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (#_ : squash (Kuiper.Approximates.approx2 ecomb comb))
  (nthr : erased nat {nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (sh : c_shmems (shmems_desc et_ab bm bn bk))
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr sh bid tid **
    thread_id nthr tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok (FB.contract eA eB (rm bm bk) (rm bk bn) (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB mapA mapB comb rA rB rC nthr sh bid tid **
    thread_id nthr tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok (FB.contract eA eB (rm bm bk) (rm bk bn) (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state (2 * (k / bk))
{
  unfold_c_shmems sh (`%shmems_desc);
  let (sarA, (sarB, _)) = sh;

  gpu_pts_to_ref sarA;
  gpu_pts_to_ref sarB;

  tensor_abs' (rm bm bk) sarA;
  let sA = from_array (rm bm bk) sarA;
  rewrite each _ as sA; //from_array (rm bm bk) sarA as sA;

  tensor_abs' (rm bk bn) sarB;
  let sB = from_array (rm bk bn) sarB;
  rewrite each _ as sB; //from_array (rm bk bn) sarB as sB;

  let num_k_tiles = k /^ bk;
  let num_n_tiles = n /^ bn;
  let mrow = bid /^ num_n_tiles;
  assert pure (mrow < m / bm);
  let mcol = bid %^ num_n_tiles;
  assert pure (mcol < n / bn);

  let wid = tid /^ warp_size;
  let warpRow : szlt (bm / (wm*tm)) = wid /^ (bn/^(wn*^tn));
  let warpCol : szlt (bn / (wn*tn)) = wid %^ (bn/^(wn*^tn));

  (* Tensor core fragments *)
  let aFrags = __alloc_array_fragment et_ab FragA tm tn tk FragLRM wm;
  let bFrags = __alloc_array_fragment et_ab FragB tm tn tk FragLRM wn;
  let accFrags = __alloc_array_fragment et_c FragAcc tm tn tk FragLAcc (wm *^ wn);

  // Fill accumulators with 0
  populate_acc_with_zero tm tn tk wm wn accFrags;
  let rAcc0 : chest2 real (wm*tm) (wn*tn) = const _ 0.0R;
  assert (rewrites_to rAcc0 (const _ 0.0R));

  let gwRow : enatlt (m/(wm*tm)) = mrow * (bm/(wm*tm)) + warpRow;
  let gwCol : enatlt (n/(wn*tn)) = mcol * (bn/(wn*tn)) + warpCol;

  // Establish the accumulator invariant in the FOLDED, opaque [kacc_inv] form.
  // [kacc_inv] is [@@ "opaque_to_smt"], so it stays syntactically inert in kf's
  // large context: the [macc_ematrix_tiled] SMTPat (acc2 (ematrix_tiled ..))
  // never sees the [ematrix_tiled (chest_map ..)] terms buried inside
  // [__gmatmul_single], which previously drove a non-terminating (multi-GB) Z3
  // blowup at the loop invariant check and the [ktile_advance] call framing.
  // [kacc_inv_eq] exposes the definition only here (vk = 0) and once after the
  // loop -- never inside the loop body.
  kacc_inv_eq m n k bm bn bk tm tn tk wm wn
    (Chest.chest_map mapA rA) (Chest.chest_map mapB rB) rAcc0 gwRow gwCol 0;
  rewrite fragarrayAcc_approximates wm wn accFrags rAcc0
       as fragarrayAcc_approximates wm wn accFrags
            (kacc_inv m n k bm bn bk tm tn tk wm wn
              (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
              rAcc0 gwRow gwCol 0);

  rewrite
    (exists* (x : chest2 _ _ _). sA |-> Frac (1.0R /. nthr) x) **
    (exists* (x : chest2 _ _ _). sB |-> Frac (1.0R /. nthr) x)
  as
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr);

  let mut bkIdx : sz = 0sz;
  while (!bkIdx <^ num_k_tiles)
    invariant
      exists* (vbkIdx : sz { vbkIdx <= num_k_tiles }).
        bkIdx |-> vbkIdx **
        fragarrayAcc_approximates wm wn accFrags
          (kacc_inv m n k bm bn bk tm tn tk wm wn
            (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
            rAcc0 gwRow gwCol !bkIdx)
    invariant
      live aFrags **
      live bFrags
    invariant
      (exists* em1. FB.bp_sharing sA em1 nthr) **
      (exists* em2. FB.bp_sharing sB em2 nthr) **
      B.barrier_state (2 * !bkIdx)
    decreases (num_k_tiles - !bkIdx)
  {
    even_2x !bkIdx;
    assert pure((2 * !bkIdx % 2 = 0) == true);
    assert pure (even (2 * !bkIdx));

    FB.fold_barrier_p_even eA eB sA sB nthr bid !bkIdx tid;
    rewrite (FB.barrier_p eA eB sA sB nthr bid) (2 * !bkIdx) tid
         as (FB.contract eA eB (rm bm bk) (rm bk bn) sarA sarB nthr bid).rin (2 * !bkIdx) tid;

    B.barrier_wait ();

    rewrite (FB.contract eA eB (rm bm bk) (rm bk bn) sarA sarB nthr bid).rout (2 * !bkIdx) tid
         as (FB.barrier_q eA eB sA sB nthr bid) (2 * !bkIdx) tid;
    FB.unfold_barrier_q_even eA eB sA sB nthr bid !bkIdx tid;

    // FlipFlopBarrier2 returns FB.live_strided_chunks; the copy helper consumes
    // Copy.Vec2.live_strided_chunks. They share the same in_chunk predicate but
    // are distinct symbols, so bridge across them (as BlockTiling2D does).
    unfold FB.live_strided_chunks sA nthr tid;
    with eA0. assert (FB.own_strided_chunks sA eA0 nthr tid);
    rewrite FB.own_strided_chunks sA eA0 nthr tid as CV2.own_strided_chunks sA eA0 nthr tid;
    fold CV2.live_strided_chunks sA nthr tid;
    unfold FB.live_strided_chunks sB nthr tid;
    with eB0. assert (FB.own_strided_chunks sB eB0 nthr tid);
    rewrite FB.own_strided_chunks sB eB0 nthr tid as CV2.own_strided_chunks sB eB0 nthr tid;
    fold CV2.live_strided_chunks sB nthr tid;

    copy_tiles_out_of_matrices_vec bm bn bk sA sB gA gB mrow !bkIdx mcol (bm/^(wm*^tm)*^(bn/^(wn*^tn))*^warp_size) tid;

    // The copy helper yields Copy.Vec2.own_strided_chunks; convert back to FB's
    // for the barrier's odd phase.
    rewrite CV2.own_strided_chunks sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid
         as FB.own_strided_chunks sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid;
    rewrite CV2.own_strided_chunks sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid
         as FB.own_strided_chunks sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid;

    odd_2x1 !bkIdx;
    assert (pure (odd (2 * !bkIdx + 1)));
    FB.fold_barrier_p_odd eA eB sA sB nthr bid mrow mcol !bkIdx tid;
    rewrite (FB.barrier_p eA eB sA sB nthr bid) (2 * !bkIdx + 1) tid
         as (FB.contract eA eB (rm bm bk) (rm bk bn) sarA sarB nthr bid).rin (2 * !bkIdx + 1) tid;

    B.barrier_wait ();

    even_2x (!bkIdx + 1);
    assert pure (2 * (!bkIdx + 1) == 2 * !bkIdx + 2);
    assert pure (odd (2 * !bkIdx + 1));
    assert pure ((2 * !bkIdx + 1) < 2 * k / bk);
    assert pure (even (2 * !bkIdx + 2));
    rewrite (FB.contract eA eB (rm bm bk) (rm bk bn) sarA sarB nthr bid).rout (2 * !bkIdx + 1) tid
         as (FB.barrier_q eA eB sA sB nthr bid) (2 * !bkIdx + 1) tid;
    FB.unfold_barrier_q_odd eA eB sA sB nthr bid mrow mcol !bkIdx tid;

    unfold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    unfold FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    let rA_sub = ematrix_subtile rA bm bk mrow !bkIdx;
    let rB_sub = ematrix_subtile rB bk bn !bkIdx mcol;
    let vbk = !bkIdx;

    // Establish the input-subtile approximations [eA_sub %~ rA_sub] /
    // [eB_sub %~ rB_sub] as GROUND hypotheses (proved in [subtile_approximates]'s
    // clean context from the ground [eA %~ rA] / [eB %~ rB]).  This lets the
    // [ktile_advance] refinement discharge by assumption, instead of firing
    // [lemma_approximates_intro]'s [SMTPat (m1 %~ m2)] on the [ematrix_subtile]
    // terms and cascading into a non-terminating Z3 blowup in kf's huge context.
    subtile_approximates eA rA bm bk mrow vbk;
    subtile_approximates eB rB bk bn vbk mcol;

    // The loop invariant is now stated in the FOLDED [kacc_inv] form, which is
    // exactly [ktile_advance]'s pre/post form -- so NO per-iteration bridge
    // rewrites are needed here.  The whole per-k-tile compute step (subproducts +
    // accumulator advance) runs inside [ktile_advance]'s SMALL context, keeping
    // the [chest_map]/[ematrix_subtile] terms and the [macc_ematrix_tiled] SMTPat
    // out of kf's large ambient context entirely (which previously exploded Z3).
    ktile_advance
      m n k
      bm bn bk tm tn tk wm wn
      aFrags bFrags accFrags
      sA sB
      rA rB
      mrow mcol vbk
      rA_sub rB_sub
      emA emB mapA mapB
      rAcc0
      warpRow warpCol
      gwRow gwCol
      ();

    fold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    fold FB.bp_sharing sB (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    bkIdx := !bkIdx +^ 1sz;
  };

  // After the loop the accumulator invariant holds at [!bkIdx == num_k_tiles],
  // whose value is [k/bk].  Expose the opaque [kacc_inv] back to the raw
  // [__gmatmul_single] form -- ONCE, here in the post-loop context -- so the
  // [matmul_tiles_lemma]/[rAcc'] reasoning below (stated in raw form) applies.
  assert pure (SZ.v num_k_tiles == k / bk);
  kacc_inv_eq m n k bm bn bk tm tn tk wm wn
    (Chest.chest_map mapA rA) (Chest.chest_map mapB rB) rAcc0 gwRow gwCol (k / bk);
  rewrite fragarrayAcc_approximates wm wn accFrags
            (kacc_inv m n k bm bn bk tm tn tk wm wn
              (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
              rAcc0 gwRow gwCol (k / bk))
       as fragarrayAcc_approximates wm wn accFrags
            (__gmatmul_single rAcc0 matmul matplus
              (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
              (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn))
                gwRow gwCol (k / bk));

  assert
        fragarrayAcc_approximates wm wn accFrags
          (__gmatmul_single rAcc0 matmul matplus
            (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
            (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn))
              gwRow // (mrow * bm/(wm*tm) + warpRow)
              gwCol // (mcol * bn/(wn*tn) + warpCol)
              (k / bk));

  assert pure (gwRow == warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size));
  assert pure (gwCol == warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size));

  matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ())
    (wm*tm) (wn*tn) bk
    rAcc0 (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
    gwRow gwCol;

  let rAcc' : chest2 real (wm*tm) (wn*tn) =
    gmatmul_single rAcc0 matmul matplus
     (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
     (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn))
       gwRow gwCol;

  assert pure (
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
        (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn)) gwRow gwCol !bkIdx)
      == rAcc');

  // The per-warp matmul over the PRE-MAPPED inputs (mapA/mapB applied
  // elementwise via chest_map).  This is the matmul component of [wt_target].
  let rAcc'' : chest2 real (wm*tm) (wn*tn) =
    MS.matmul (ematrix_subtile (Chest.chest_map mapA rA) (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
              (ematrix_subtile (Chest.chest_map mapB rB) k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)));

  assert pure (matplus (const _ 0.0R) rAcc'' `equal` rAcc'');
  // ^ This is needed so we can use the result of the matmul_tiles_lemma
  // above...  very boring.

  assert pure (rAcc' == rAcc'');
  rewrite
    fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) bk)
        (ematrix_tiled (Chest.chest_map mapB rB) bk (wn*tn)) gwRow gwCol !bkIdx)
  as
    fragarrayAcc_approximates wm wn accFrags rAcc'';

  with em1. unfold FB.bp_sharing sA em1 nthr;
  with em2. unfold FB.bp_sharing sB em2 nthr;

  rewrite each (tid / 32) as wid;
  // The C-input warp tile (from [kpre1]) approximates this subtile of [rC];
  // [epilogue] reads it back to fuse the output combine.  Bind it so the fold
  // back into [wt_target] below has a stable syntactic form.
  let rCtile : chest2 real (wm*tm) (wn*tn) =
    ematrix_subtile rC (wm*tm) (wn*tn)
      (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (SZ.v wid))
      (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (SZ.v wid));
  assert (rewrites_to rCtile
    (ematrix_subtile rC (wm*tm) (wn*tn)
      (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (SZ.v wid))
      (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (SZ.v wid))));
  epilogue bm bn bk tm tn tk wm wn accFrags rAcc'' gC comb ecomb rCtile bid wid;
  rewrite each v wid as (tid / 32);

  with vaFrags. assert aFrags |-> vaFrags; drop_ (aFrags |-> vaFrags);
  with vbFrags. assert bFrags |-> vbFrags; drop_ (bFrags |-> vbFrags);
  unfold fragarrayAcc_approximates wm wn accFrags rAcc'';
  with vaccumFrags. assert accFrags |-> vaccumFrags; drop_ (accFrags |-> vaccumFrags);

  tensor_concr sA; rewrite each core sA as sarA;
  tensor_concr sB; rewrite each core sB as sarB;

  rewrite each sarA as fst sh;
  rewrite each sarB as fst (snd sh);

  // Fold the combined output tile into the opaque [wt_target] that [kpost1]
  // expects.  The [rewrite each v wid as (tid / 32)] above already inlined
  // [rCtile], so match on the explicit combined form here.
  rewrite each
    (Chest.chest_comb comb
      (ematrix_subtile rC (wm*tm) (wn*tn)
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size))
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)))
      rAcc'')
    as (wt_target mapA mapB comb bm bn bk tm tn tk wm wn rA rB rC nthr bid (tid / warp_size));

  fold_c_shmems sh (`%shmems_desc);

  ()
}

#restart-solver
#push-options "--fuel 1 --ifuel 1 --split_queries no --z3rlimit_factor 10"
inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
  {| real_like et_ab, real_like et_c |}
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA  { is_global gA })
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB { is_global gB })
  (#eB : chest2 et_ab k n)
  (gC : array2 et_c (rm m n) { is_global gC })
  // ^ Why does this have a fixed layout?
  (#_ : squash (SZ.fits (m * n)))
  (#eC : chest2 et_c m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#fA #fB : perm)
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
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
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  // Fused elementwise pre-maps on the inputs and combine on the output, threaded
  // in the REAL domain, plus their approximation-compatible DEVICE realizations.
  (mapA mapB : real -> real)
  (comb : real -> real -> real)
  (emA emB : et_ab -> et_ab)
  (ecomb : et_c -> et_c -> et_c)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (#_ : squash (Kuiper.Approximates.approx2 ecomb comb))
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** pure (eA %~ rA) **
       gB |-> Frac fB eB ** pure (eB %~ rB) **
       gC |-> eC ** pure (eC %~ rC))
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       (exists* (eC' : chest2 et_c m n).
         gC |-> eC' ** pure (eC' %~ MS.gmmcomb mapA mapB comb rC rA rB)))
= {
  nblk;
  nthr;

  shmems_desc = shmems_desc et_ab bm bn bk;

  barrier_contract = (fun bid ptrs -> FB.contract eA eB (rm bm bk) (rm bk bn) (fst ptrs) (fst (snd ptrs)) nthr bid);
  barrier_count    = (fun _bid -> 2 * (SZ.v k / SZ.v bk));
  barrier_ok = (fun bid ptrs -> FB.barrier_p_to_q_transform eA eB (rm bm bk) (rm bk bn) (fst ptrs) (fst (snd ptrs)) nthr bid);

  frame = pure (SZ.fits ((rm m n).ulen));
  block_pre  = (fun bid -> forall+ (tid : natlt nthr). kpre1  gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr). kpost1 gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB mapA mapB comb rA rB rC nthr bid tid);

  setup      = setup    gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB rA rB rC;
  teardown   = teardown gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB mapA mapB comb rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB rA rB rC;
  block_teardown = block_teardown gA eA gB eB gC eC bm bn bk tm tn tk wm wn nblk nthr fA fB mapA mapB comb rA rB rC;

  kpre      = kpre  gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB rA rB rC nthr;
  kpost     = kpost gA eA gB eB gC eC bm bn bk tm tn tk wm wn fA fB mapA mapB comb rA rB rC nthr;

  f = kf gA #eA gB #eB gC #eC bm bn bk tm tn tk wm wn rA rB rC mapA mapB comb emA emB ecomb (SZ.v nthr);

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=solve;
  kpost_sendable=solve;
}
#pop-options
