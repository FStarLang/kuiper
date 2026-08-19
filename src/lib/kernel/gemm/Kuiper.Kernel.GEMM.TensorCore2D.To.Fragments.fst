module Kuiper.Kernel.GEMM.TensorCore2D.To.Fragments

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

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

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc


open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState
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

#push-options "--z3rlimit 120"
(* Flattening a nested sub-tile: the [i]-th [tm]-row sub-tile of the
   [(w*tm)]-row tile [tr] is the [(tr*w+i)]-th [tm]-row sub-tile.  Used to
   reconcile the loop invariant (which indexes [rm] directly) with the folded
   [fragarray*_approximates] (which indexes the nested sub-tile).  Same helpers
   as in [Kuiper.Kernel.GEMM.TensorCore2D]; with one SMT query per proof
   obligation these equalities are no longer found on their own. *)
let subtile_of_subtile_eq
  (#et : Type0)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (w : pos)
  (trows : pos { w * trows /? rows /\ trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / (w * trows)))
  (i : natlt w)
  (tc : natlt (cols / tcols))
  (pf : squash (tr * w + i < rows / trows))
  : Lemma (ematrix_subtile (ematrix_subtile em (w * trows) tcols tr tc) trows tcols i 0
           == ematrix_subtile em trows tcols (tr * w + i) tc)
= FStar.Math.Lemmas.paren_mul_right tr w trows;
  FStar.Math.Lemmas.distributivity_add_left (tr * w) i trows;
  Kuiper.Chest.ext (ematrix_subtile (ematrix_subtile em (w * trows) tcols tr tc) trows tcols i 0)
            (ematrix_subtile em trows tcols (tr * w + i) tc)

(* Column-dimension analogue. *)
let subtile_of_subtile_eq_cols
  (#et : Type0)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (w : pos)
  (trows : pos { trows /? rows })
  (tcols : pos { w * tcols /? cols /\ tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / (w * tcols)))
  (j : natlt w)
  (pf : squash (tc * w + j < cols / tcols))
  : Lemma (ematrix_subtile (ematrix_subtile em trows (w * tcols) tr tc) trows tcols 0 j
           == ematrix_subtile em trows tcols tr (tc * w + j))
= FStar.Math.Lemmas.paren_mul_right tc w tcols;
  FStar.Math.Lemmas.distributivity_add_left (tc * w) j tcols;
  Kuiper.Chest.ext (ematrix_subtile (ematrix_subtile em trows (w * tcols) tr tc) trows tcols 0 j)
            (ematrix_subtile em trows tcols tr (tc * w + j))

(* [tr < rows/(w*trows)] and [i < w] imply [tr*w + i < rows/trows], because
   [rows/trows == w * (rows/(w*trows))].  All the multiplication/division
   rearrangements are spelled out: query splitting no longer finds them. *)
let subtile_index_bound (rows : nat) (w trows : pos) (tr : nat)
  : Lemma (requires w * trows /? rows /\ trows /? rows /\ tr < rows / (w * trows))
          (ensures forall (i : natlt w). tr * w + i < rows / trows)
= let q = rows / (w * trows) in
  FStar.Math.Lemmas.paren_mul_right w trows q;
  FStar.Math.Lemmas.swap_mul trows q;
  FStar.Math.Lemmas.paren_mul_right w q trows;
  FStar.Math.Lemmas.cancel_mul_div (w * q) trows;
  FStar.Math.Lemmas.lemma_mult_le_right w (tr + 1) q;
  FStar.Math.Lemmas.distributivity_add_left tr 1 w;
  FStar.Math.Lemmas.swap_mul q w

(* [subtile_of_subtile_eq] for every sub-tile index at once. *)
let subtile_of_subtile_eq_all
  (#et : Type0)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (w : pos)
  (trows : pos { w * trows /? rows /\ trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / (w * trows)))
  (tc : natlt (cols / tcols))
  (pf : squash (forall (i : natlt w). tr * w + i < rows / trows))
  : Lemma (forall (i : natlt w).
             ematrix_subtile (ematrix_subtile em (w * trows) tcols tr tc) trows tcols i 0
             == ematrix_subtile em trows tcols (tr * w + i) tc)
= introduce forall (i : natlt w).
      ematrix_subtile (ematrix_subtile em (w * trows) tcols tr tc) trows tcols i 0
      == ematrix_subtile em trows tcols (tr * w + i) tc
  with subtile_of_subtile_eq em w trows tcols tr i tc ()

(* Column-dimension analogue of [subtile_of_subtile_eq_all]. *)
let subtile_of_subtile_eq_all_cols
  (#et : Type0)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (w : pos)
  (trows : pos { trows /? rows })
  (tcols : pos { w * tcols /? cols /\ tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / (w * tcols)))
  (pf : squash (forall (j : natlt w). tc * w + j < cols / tcols))
  : Lemma (forall (j : natlt w).
             ematrix_subtile (ematrix_subtile em trows (w * tcols) tr tc) trows tcols 0 j
             == ematrix_subtile em trows tcols tr (tc * w + j))
= introduce forall (j : natlt w).
      ematrix_subtile (ematrix_subtile em trows (w * tcols) tr tc) trows tcols 0 j
      == ematrix_subtile em trows tcols tr (tc * w + j)
  with subtile_of_subtile_eq_cols em w trows tcols tr tc j ()

#pop-options

#push-options "--z3rlimit 30"
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
    // The loop invariant tracks each fragment against
    // [ematrix_subtile rm tm tk (arow*wm+i) dotIdx]; the fold wants the
    // nested-sub-tile form.  Supply the flattening equality for all i.
    subtile_index_bound bm wm tm arow;
    subtile_of_subtile_eq_all rm wm tm tk arow dotIdx ();
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
    // The loop invariant's [ematrix_subtile rm tk tn dotIdx (bcol*wn+i)] needs
    // [bcol*wn+i < bn/tn] just to be well-typed; establish it up front.
    subtile_index_bound bn wn tn bcol;
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
      // [elim_forall] yields the nested-sub-tile form on the [em] side; the
      // invariant indexes [rm] directly.  Supply the flattening equality.
      subtile_index_bound bn wn tn bcol;
      subtile_of_subtile_eq_cols rm wn tk tn dotIdx bcol !i1 ();
      ambig_trade_elim ();
      ambig_trade_elim ();

      i1 := !i1 +^ 1sz;
    };
    ambig_trade_elim ();
    subtile_index_bound bn wn tn bcol;
    subtile_of_subtile_eq_all_cols rm wn tk tn dotIdx bcol ();
    fold fragarrayB_approximates wn frags (ematrix_subtile rm tk (wn*tn) dotIdx bcol);
    ()
}
#pop-options

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
