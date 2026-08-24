module Kuiper.Kernel.GEMM.TensorCore2D.FragmentMMA

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

open Kuiper.Kernel.GEMM.TensorCore2D.ProofSupport
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
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
  (tm tn tk wm wn : szp)
  (aFrags     : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags     : array (fragment et_ab FragB tm tn tk FragLRM))
  (accumFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (rA : chest2 real (wm*tm) tk)
  (rB : chest2 real tk (wn*tn))
  (rAcc : chest2 real (wm*tm) (wn*tn))
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
