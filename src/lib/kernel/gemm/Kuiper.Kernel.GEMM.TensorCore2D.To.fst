module Kuiper.Kernel.GEMM.TensorCore2D.To

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

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc


open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.Epilogue
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
              (Seq.index ems i) %~ (ematrix_subtile rm tm tk (arow*wm+i) dotIdx)))
      decreases (wm - !i0)
    {
      let a_tile =
        array2_extract_tile_ro' tile_for_tc_a_tiles (SZ.v tm) (SZ.v tk) (SZ.v !i0) 0;
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
  (gm : array2 et (rm bk bn))
  (#em : chest2 et bk bn)
  (rm : chest2 real bk bn {em %~ rm})
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
              (Seq.index ems i) %~ (ematrix_subtile rm tk tn dotIdx (bcol*wn+i))))
      decreases (wn - !i1)
    {
      let b_tile = array2_extract_tile_ro' tile_for_tc_b_tiles (SZ.v tk) (SZ.v tn) 0 (SZ.v !i1);

      array_fragment_pts_to_ref frags;
      array_fragment_extract frags !i1;

      mma_loadB frags.(!i1) b_tile;
      Pulse.Lib.Forall.elim_forall
        (ematrix_subtile (ematrix_subtile em tk (wn*tn) dotIdx bcol) tk tn 0 !i1);

      FStar.Math.Lemmas.paren_mul_right (SZ.v bcol) (SZ.v wn) (SZ.v tn);
      FStar.Math.Lemmas.distributivity_add_left (SZ.v bcol * SZ.v wn) (SZ.v !i1) (SZ.v tn);
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
  (gA : array2 et_ab (rm bm bk))
  (gB : array2 et_ab (rm bk bn))
  (#eA : chest2 et_ab bm bk)
  (#eB : chest2 et_ab bk bn)
  (rA : chest2 real bm bk {eA %~ rA})
  (rB : chest2 real bk bn {eB %~ rB})
  (rAcc : chest2 real (wm*tm) (wn*tn))
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
  while (!dotIdx <^ (bk/^tk))
    invariant live aFrags ** live bFrags
    invariant
      exists* (vdotIdx : sz { vdotIdx <= (bk/tk) }).
        dotIdx |-> vdotIdx **
        fragarrayAcc_approximates wm wn accumFrags
          (__gmatmul_single rAcc matmul matplus
            (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn)) arow bcol !dotIdx)
    decreases (bk/^tk - !dotIdx)
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

    // Ghost fn: advance the accumulator by one step — the lemma call and
    // unfold/fold happen inside the ghost fn with opaque params.
    // Pass !dotIdx directly (stt read), NOT a with-bound variable (ghost).
    rewrite_fragarrayAcc_step wm wn accumFrags rAcc rA rB arow bcol !dotIdx;

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
          (ematrix_tiled rA (wm*tm) tk)
          (ematrix_tiled rB tk (wn*tn))
          arow
          bcol
          (bk/^tk)));

  rewrite_fragarrayAcc_tiles wm wn accumFrags rAcc rA rB arow bcol;
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
    (exists* eWarpTile. warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile)
  ensures
    warp_tile_approximates gC bm bn tm tn wm wn bid wid rAcc
{
  with (eWarpTile : chest2 _ _ _). assert warp_tile_pts_to gC (v bm) (v bn) (v tm) (v tn) (v wm) (v wn) (v bid) (v wid) eWarpTile;
  let rWarpTile = to_real_matrix eWarpTile;

  Kuiper.Chest.lemma_to_real_chest_approximates eWarpTile;
  assert pure (eWarpTile %~ rWarpTile);
  assert pure (eWarpTile %~ ematrix_from_tiles tm tn (ematrix_subtile rWarpTile tm tn));
  assert pure (eWarpTile %~ em_fade_tiles tm tn wm wn 0 0 rWarpTile rAcc);

  let mut i = 0sz;
  while (!i <^ wm)
    invariant
      live i
    invariant
      exists* (eWarpTile: chest2 et (wm*tm) (wn*tn)).
        warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile **
          pure (!i <= wm /\
            eWarpTile %~ (em_fade_tiles tm tn wm wn !i 0 rWarpTile rAcc))
    decreases (wm - !i)
  {
    let mut j = 0sz;
    while (!j <^ wn)
      invariant live j
      invariant
        exists* (eWarpTile: chest2 et (wm*tm) (wn*tn)).
          warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile **
            pure (!i <= wm /\ !j <= wn /\
              eWarpTile %~ (em_fade_tiles tm tn wm wn !i !j rWarpTile rAcc))
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
      mma_store accumFrags.(idx) tc_tile;

      Pulse.Lib.Forall.elim_forall (Seq.Base.index eAccumFrags idx);
      ambig_trade_elim ();
      ambig_trade_elim ();
      fold fragarrayAcc_approximates wm wn accumFrags rAcc;

      rewrite each tile_for_tc_tiles as _;
      with eWarpTile'. fold warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile';

      lemma_update_tile_fade_approximates tm tn wm wn !i !j eWarpTile (Seq.index eAccumFrags idx) rWarpTile rAcc;

      j := !j +^ 1sz;
    };
    i := !i +^ 1sz;
  };

  with eWarpTile'.
    assert (warp_tile_pts_to gC bm bn tm tn wm wn bid wid eWarpTile');
  // After the outer loop !i == wm, so the invariant gives
  //   eWarpTile' %~ em_fade_tiles tm tn wm wn wm 0 rWarpTile rAcc.
  // With idxI = wm every tile is an rAcc subtile, so the fade collapses to rAcc.
  em_fade_tiles_full tm tn wm wn 0 rWarpTile rAcc;
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

#set-options "--split_queries always"
inline_for_extraction noextract
fn kf
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab,
     scalar et_cd, scalar et_acc |}
  {| real_like et_ab, real_like et_cd, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
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
  (gC : array2 et_cd (rm m n))
  (#eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
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
  (#_ : squash (SZ.fits (tm * tn + warp_size)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#fA #fB #fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nthr : szp {SZ.v nthr == bm/(wm*tm)*(bn/(wn*tn))*warp_size})
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre_to
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn
      fA fB fC rA rB rC
      ((m /^ bm) *^ (n /^ bn)) nthr sh bid tid **
    thread_id nthr tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok (FB.contract eA eB (rm bm bk) (rm bk bn) (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost_to comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn
      fA fB fC rA rB rC
      ((m /^ bm) *^ (n /^ bn)) nthr sh bid tid **
    thread_id nthr tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok (FB.contract eA eB (rm bm bk) (rm bk bn) (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state (2 * (k / bk))
{
  unfold kpre_to
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    fA fB fC rA rB rC
    ((m /^ bm) *^ (n /^ bn)) nthr sh bid tid;
  unfold kpre1_to
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    fA fB fC rA rB rC
    ((m /^ bm) *^ (n /^ bn)) nthr bid tid;
  unfold shared_thread_live bm bn bk tm tn nthr sh tid;

  let (sarA, (sarB, (sarAcc, sarTail))) = sh;
  unfold live_c_shmem sarA #(1.0R /. nthr);
  unfold live_c_shmem sarB #(1.0R /. nthr);

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
  let accFrags = __alloc_array_fragment et_acc FragAcc tm tn tk FragLAcc (wm *^ wn);

  // Fill accumulators with 0
  populate_acc_with_zero tm tn tk wm wn accFrags;
  let rAcc0 : chest2 real (wm*tm) (wn*tn) = const _ 0.0R;
  assert (rewrites_to rAcc0 (const _ 0.0R));

  rewrite fragarrayAcc_approximates wm wn accFrags rAcc0
       as fragarrayAcc_approximates wm wn accFrags
            (__gmatmul_single rAcc0 matmul matplus
              (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) mrow mcol 0);

  rewrite
    (exists* (x : chest2 _ _ _). sA |-> Frac (1.0R /. nthr) x) **
    (exists* (x : chest2 _ _ _). sB |-> Frac (1.0R /. nthr) x)
  as
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr);

  let gwRow : enatlt (m/(wm*tm)) = mrow * (bm/(wm*tm)) + warpRow;
  let gwCol : enatlt (n/(wn*tn)) = mcol * (bn/(wn*tn)) + warpCol;

  let mut bkIdx : sz = 0sz;
  while (!bkIdx <^ num_k_tiles)
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
      m n k
      bm bn bk tm tn tk wm wn
      mrow mcol
      warpRow warpCol
      gwRow gwCol
      !bkIdx
      rA rB
      rAcc0 rAcc
      rA_sub rB_sub;

    // Rewrite the accumulator slprop from the matplus/matmul form
    // to the __gmatmul_single form using the equality from loop_invariant_lemma.
    rewrite_fragarrayAcc wm wn accFrags
      (rAcc `matplus` matmul (ematrix_subtile rA_sub (wm*tm) bk warpRow 0)
                              (ematrix_subtile rB_sub bk (wn*tn) 0 warpCol))
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) gwRow gwCol (!bkIdx + 1));

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
              (k / bk));

  assert pure (gwRow == warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size));
  assert pure (gwCol == warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size));

  matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ())
    (wm*tm) (wn*tn) bk
    rAcc0 rA rB
    gwRow gwCol;

  let rAcc' : chest2 real (wm*tm) (wn*tn) =
    gmatmul_single rAcc0 matmul matplus
     (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn))
       gwRow gwCol;

  assert pure (
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm*tm) bk) (ematrix_tiled rB bk (wn*tn)) gwRow gwCol !bkIdx)
      == rAcc');

  let rAcc'' : chest2 real (wm*tm) (wn*tn) =
    MS.matmul (ematrix_subtile rA (wm*tm) k (warp_tile_i #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)) 0)
              (ematrix_subtile rB k  (wn*tn) 0 (warp_tile_j #m #n bm bn bk tm tn tk wm wn nthr bid (tid / warp_size)));

  assert pure (matplus (const _ 0.0R) rAcc'' `equal` rAcc'');
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

  epilogue_to #et_ab #et_cd #et_acc comb comb_r #m #n
    gC #(fC /. (m / bm * (n / bn) * nthr)) #eC #rC #_ gD
    bm bn bk tm tn tk wm wn nthr
    #_ #_ #_ #_ #_
    (sarA, (sarB, (sarAcc, sarTail)))
    accFrags rAcc'' bid tid #_;

  MS.matmul_decompose_lemma rA rB
    (wm * tm) (wn * tn) gwRow gwCol;
  let rOutLocal =
    chest_comb comb_r
      (ematrix_subtile
        (ematrix_subtile rC bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)
      rAcc'';
  let rOutTarget =
    ematrix_subtile
      (ematrix_subtile
        (MS.mmcomb comb_r rC rA rB)
        bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;
  rewrite each
    (SZ.v bid / (SZ.v n / SZ.v bn))
  as SZ.v mrow;
  rewrite each
    (SZ.v bid % (SZ.v n / SZ.v bn))
  as SZ.v mcol;
  rewrite each
    (SZ.v tid / warp_size)
  as SZ.v wid;
  rewrite each
    (SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v tn)))
  as SZ.v warpRow;
  rewrite each
    (SZ.v wid % (SZ.v bn / (SZ.v wn * SZ.v tn)))
  as SZ.v warpCol;
  rewrite each
    (chest_comb comb_r
      (ematrix_subtile
        (ematrix_subtile rC bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)
      rAcc'')
  as rOutLocal;
  assert pure (
    (SZ.v bm / (SZ.v wm * SZ.v tm)) *
      (SZ.v wm * SZ.v tm) == SZ.v bm);
  assert pure (
    (SZ.v bn / (SZ.v wn * SZ.v tn)) *
      (SZ.v wn * SZ.v tn) == SZ.v bn);
  assert pure (
    gwRow * (SZ.v wm * SZ.v tm) ==
    SZ.v mrow * SZ.v bm +
      SZ.v warpRow * (SZ.v wm * SZ.v tm));
  assert pure (
    gwCol * (SZ.v wn * SZ.v tn) ==
    SZ.v mcol * SZ.v bn +
      SZ.v warpCol * (SZ.v wn * SZ.v tn));
  Kuiper.EMatrix.lemma_equal_intro rOutLocal rOutTarget;
  Kuiper.Chest.ext rOutLocal rOutTarget;
  rewrite
    output_lane_approximates
      gD bm bn tm tn wm wn bid tid rOutLocal
  as
    output_lane_approximates
      gD bm bn tm tn wm wn bid tid rOutTarget;

  with vaFrags. assert aFrags |-> vaFrags; drop_ (aFrags |-> vaFrags);
  with vbFrags. assert bFrags |-> vbFrags; drop_ (bFrags |-> vbFrags);
  unfold fragarrayAcc_approximates wm wn accFrags rAcc'';
  with vaccumFrags. assert accFrags |-> vaccumFrags; drop_ (accFrags |-> vaccumFrags);

  tensor_concr sA; rewrite each core sA as sarA;
  tensor_concr sB; rewrite each core sB as sarB;

  fold live_c_shmem sarA #(1.0R /. nthr);
  fold live_c_shmem sarB #(1.0R /. nthr);
  fold shared_thread_live bm bn bk tm tn nthr
    (sarA, (sarB, (sarAcc, sarTail))) tid;
  rewrite each rOutTarget as
    ematrix_subtile
      (ematrix_subtile
        (MS.mmcomb comb_r rC rA rB)
        bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;
  rewrite each (SZ.v mrow) as (SZ.v bid / (SZ.v n / SZ.v bn));
  rewrite each (SZ.v mcol) as (SZ.v bid % (SZ.v n / SZ.v bn));
  rewrite each (SZ.v warpRow) as
    ((SZ.v tid / warp_size) / (SZ.v bn / (SZ.v wn * SZ.v tn)));
  rewrite each (SZ.v warpCol) as
    ((SZ.v tid / warp_size) % (SZ.v bn / (SZ.v wn * SZ.v tn)));
  fold kpost1_to comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    fA fB fC rA rB rC
    ((m /^ bm) *^ (n /^ bn)) nthr bid tid;
  rewrite
    shared_thread_live bm bn bk tm tn nthr
      (sarA, (sarB, (sarAcc, sarTail))) tid
  as
    shared_thread_live bm bn bk tm tn nthr sh tid;
  fold kpost_to comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    fA fB fC rA rB rC
    ((m /^ bm) *^ (n /^ bn)) nthr sh bid tid;

  ()
}

let scratch_fits_of_kernel_constraints
  (m n : szp)
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  : squash (SZ.fits ((nthr / warp_size) * tm * tn))
=
  FStar.Math.Lemmas.cancel_mul_div
    (SZ.v bm / (SZ.v wm * SZ.v tm) *
      (SZ.v bn / (SZ.v wn * SZ.v tn)))
    (SZ.v warp_size);
  assert (SZ.v nthr / SZ.v warp_size ==
    SZ.v bm / (SZ.v wm * SZ.v tm) *
    (SZ.v bn / (SZ.v wn * SZ.v tn)));
  FStar.Math.Lemmas.div_exact_r
    (SZ.v bm) (SZ.v wm * SZ.v tm);
  assert (
    SZ.v wm * SZ.v tm *
    (SZ.v bm / (SZ.v wm * SZ.v tm)) == SZ.v bm);
  FStar.Math.Lemmas.div_exact_r
    (SZ.v bn) (SZ.v wn * SZ.v tn);
  assert (
    SZ.v wn * SZ.v tn *
    (SZ.v bn / (SZ.v wn * SZ.v tn)) == SZ.v bn);
  assert (
    (SZ.v nthr / SZ.v warp_size) * SZ.v tm * SZ.v tn <=
    SZ.v bm * SZ.v bn);
  Kuiper.Divides.lemma_nat_divides_pos_divides (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_divides_le (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_nat_divides_pos_divides (SZ.v bn) (SZ.v n);
  Kuiper.Divides.lemma_divides_le (SZ.v bn) (SZ.v n);
  assert (SZ.v bm * SZ.v bn <= SZ.v m * SZ.v n);
  ()

let squash_and
  (#p #q : prop)
  (#_ : squash p)
  (#_ : squash q)
  : squash (p /\ q)
= ()

let warp_divides_thread_count
  (bm bn tm tn wm wn : szp)
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  : squash (warp_size /?+ nthr)
=
  FStar.Math.Lemmas.cancel_mul_mod
    (SZ.v bm / (SZ.v wm * SZ.v tm) *
      (SZ.v bn / (SZ.v wn * SZ.v tn)))
    (SZ.v warp_size);
  ()

let shmem_inv_components_to
  (#et_ab #et_acc : Type0)
  {| sized et_ab, sized et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmems_inv sh))
  : squash (
      c_shmem_inv (fst sh) /\
      c_shmem_inv (fst (snd sh)) /\
      c_shmem_inv (fst (snd (snd sh))))
=
  assert (c_shmem_inv (fst sh));
  assert (c_shmems_inv (snd sh));
  assert (c_shmem_inv (fst (snd sh)));
  assert (c_shmems_inv (snd (snd sh)));
  assert (c_shmem_inv (fst (snd (snd sh))));
  ()

let scratch_tile_sendable_to
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmem_inv (fst (snd (snd sh)))))
  (tid : natlt nthr)
  (eAcc : chest2 et_acc tm tn)
  : is_send_across block_of (
      scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
        |-> Frac (1.0R /. warp_size) eAcc)
= is_send_across_tensor
    (scratch_tile bm bn bk tm tn nthr sh (tid / warp_size))
    block_of #_ #(1.0R /. warp_size) eAcc

let scratch_tile_live_sendable_to
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmem_inv (fst (snd (snd sh)))))
  (tid : natlt nthr)
  : is_send_across block_of (
      scratch_tile_live bm bn bk tm tn nthr sh tid)
=
  let ff (eAcc : chest2 et_acc tm tn) :
    is_send_across block_of (
      scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
        |-> Frac (1.0R /. warp_size) eAcc) =
    scratch_tile_sendable_to bm bn bk tm tn nthr sh #_ tid eAcc in
  is_send_across_exists
    (fun eAcc ->
      scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
        |-> Frac (1.0R /. warp_size) eAcc)
    #ff

let shared_thread_live_sendable_to
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (#_ : squash (c_shmems_inv sh))
  (tid : natlt nthr)
  : is_send_across block_of (
      shared_thread_live bm bn bk tm tn nthr sh tid)
=
  let sh_invs =
    shmem_inv_components_to bm bn bk tm tn nthr sh #_ in
  let scratch_send =
    scratch_tile_live_sendable_to bm bn bk tm tn nthr sh #_ tid in
  solve

#push-options "--fuel 1 --ifuel 1 --split_queries no --z3rlimit_factor 10"
inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
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
  (gC : array2 et_cd (rm m n) { is_global gC })
  (#mn_fits : squash (SZ.fits (m * n)))
  (#eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n) { is_global gD })
  (#eD : chest2 et_cd m n)
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#bm_div_m : squash (bm /?+ m))
  (#bn_div_n : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#shared_fits : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#fA #fB #fC : perm)
  (nblk : szp{SZ.v nblk == m/bm * (n/bn)})
  (nthr : szp{SZ.v nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + nthr-1)))
  (#_ : squash (SZ.fits (bk*bn + nthr-1)))
  (#_ : squash (nblk <= max_blocks))
  (#threads_limit : squash (nthr <= max_threads))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#_ : squash (wm * tm /?+ m)) // obvious, but SMT is flaky
  (#_ : squash (wn * tn /?+ n)) // idem
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** pure (eA %~ rA) **
       gB |-> Frac fB eB ** pure (eB %~ rB) **
       gC |-> Frac fC eC ** pure (eC %~ rC) **
       live gD)
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       gC |-> Frac fC eC **
       (exists* (eD' : chest2 et_cd m n).
         gD |-> eD' ** pure (eD' %~ MS.mmcomb comb_r rC rA rB)))
=
 FStar.Math.Lemmas.cancel_mul_div (SZ.v wm) (SZ.v tm);
 FStar.Math.Lemmas.cancel_mul_div (SZ.v wn) (SZ.v tn);
 let block_divides =
   squash_and
     #(bm /?+ m) #(bn /?+ n)
     #bm_div_m #bn_div_n in
 let scratch_fits =
   scratch_fits_of_kernel_constraints
     m n bm bn bk tm tn tk wm wn
     #mn_fits #bm_div_m #bn_div_n nthr in
 let warp_divides =
   warp_divides_thread_count bm bn tm tn wm wn nthr in
{
  nblk;
  nthr;

  shmems_desc = shmems_desc_to et_ab et_acc bm bn bk tm tn nthr
    #shared_fits #scratch_fits #warp_divides;

  barrier_contract = (fun bid ptrs -> FB.contract eA eB (rm bm bk) (rm bk bn) (fst ptrs) (fst (snd ptrs)) nthr bid);
  barrier_count    = (fun _bid -> 2 * (SZ.v k / SZ.v bk));
  barrier_ok = (fun bid ptrs -> FB.barrier_p_to_q_transform eA eB (rm bm bk) (rm bk bn) (fst ptrs) (fst (snd ptrs)) nthr bid);

  frame = pure (SZ.fits ((rm m n).ulen));
  block_pre  = (fun bid -> forall+ (tid : natlt nthr).
    kpre1_to gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn #block_divides fA fB fC rA rB rC
      nblk nthr bid tid);
  block_post = (fun bid -> forall+ (tid : natlt nthr).
    kpost1_to comb_r gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn #block_divides fA fB fC rA rB rC
      nblk nthr bid tid);

  setup = setup_to
    gA eA gB eB #_ #_ gC eC gD #mn_fits
    bm bn bk tm tn tk wm wn #block_divides nblk nthr
    fA fB fC rA rB rC;
  teardown = teardown_to comb_r
    gA eA gB eB gC eC gD #mn_fits
    bm bn bk tm tn tk wm wn #block_divides nblk nthr
    fA fB fC rA rB rC;

  block_frame    = (fun _ar _bid -> emp);
  block_setup = block_setup_to
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    #block_divides #shared_fits #scratch_fits #warp_divides
    fA fB fC rA rB rC nblk nthr;
  block_teardown = block_teardown_to comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    #block_divides #shared_fits #scratch_fits #warp_divides
    fA fB fC rA rB rC nblk nthr;

  kpre = kpre_to
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    #block_divides #shared_fits #scratch_fits #warp_divides
    fA fB fC rA rB rC nblk nthr;
  kpost = kpost_to comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn
    #block_divides #shared_fits #scratch_fits #warp_divides
    fA fB fC rA rB rC nblk nthr;

  f = kf comb comb_r
    gA #eA gB #eB gC #eC gD
    bm bn bk tm tn tk wm wn
    rA rB rC nthr;

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable = (fun sh sh_inv _bid _tid ->
    let shared_send = shared_thread_live_sendable_to
      bm bn bk tm tn nthr
      #shared_fits #scratch_fits #warp_divides sh #sh_inv _tid in
    solve);
  kpost_sendable = (fun sh sh_inv _bid _tid ->
    let shared_send = shared_thread_live_sendable_to
      bm bn bk tm tn nthr
      #shared_fits #scratch_fits #warp_divides sh #sh_inv _tid in
    solve);
}
#pop-options
