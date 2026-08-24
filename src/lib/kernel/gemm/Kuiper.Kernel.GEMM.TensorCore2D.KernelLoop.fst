module Kuiper.Kernel.GEMM.TensorCore2D.KernelLoop

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 10"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
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
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
open Kuiper.Kernel.GEMM.TensorCore2D.Subproducts
open Kuiper.Kernel.GEMM.TensorCore2D.KLoop

let unfold_fb_contract () : FStar.Tactics.V2.Tac unit =
  FStar.Tactics.V2.norm [delta_only [`%FB.contract]; iota; primops];
  Pulse.Lib.Core.slprop_equiv_norm ()

ghost
fn bp_to_rin
  (#etA #etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows}) (#bk : pos{bk /?+ shared}) (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk) (l2 : full_layout2 bk bn)
  (sar1 : larray etA (bm * bk)) (sar2 : larray etB (bk * bn))
  (sa1 : array2 etA l1) (sa2 : array2 etB l2)
  (nthr : pos) (bid : natlt (rows/bm * (cols/bn)))
  (it : nat) (tid : natlt nthr)
  requires
    FB.barrier_p eA eB sa1 sa2 nthr bid it tid **
    pure (sa1 == from_array l1 sar1 /\ sa2 == from_array l2 sar2)
  ensures
    (FB.contract eA eB l1 l2 sar1 sar2 nthr bid).rin it tid
{
  rewrite each sa1 as (from_array l1 sar1);
  rewrite each sa2 as (from_array l2 sar2);
  rewrite FB.barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
       as (FB.contract eA eB l1 l2 sar1 sar2 nthr bid).rin it tid
       by unfold_fb_contract ();
}

ghost
fn rout_to_bq
  (#etA #etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows}) (#bk : pos{bk /?+ shared}) (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk) (l2 : full_layout2 bk bn)
  (sar1 : larray etA (bm * bk)) (sar2 : larray etB (bk * bn))
  (sa1 : array2 etA l1) (sa2 : array2 etB l2)
  (nthr : pos) (bid : natlt (rows/bm * (cols/bn)))
  (it : nat) (tid : natlt nthr)
  requires
    (FB.contract eA eB l1 l2 sar1 sar2 nthr bid).rout it tid **
    pure (sa1 == from_array l1 sar1 /\ sa2 == from_array l2 sar2)
  ensures
    FB.barrier_q eA eB sa1 sa2 nthr bid it tid
{
  rewrite (FB.contract eA eB l1 l2 sar1 sar2 nthr bid).rout it tid
       as FB.barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
       by unfold_fb_contract ();
  rewrite each (from_array l1 sar1) as sa1;
  rewrite each (from_array l2 sar2) as sa2;
}

#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let lemma_double_div (k bk : pos)
  : Lemma (requires bk /?+ k)
          (ensures 2 * k / bk == 2 * (k / bk))
  = Kuiper.Divides.lemma_nat_divides_pos_divides bk k;
    assert (bk * (k / bk) == k);
    FStar.Math.Lemmas.cancel_mul_div (2 * (k / bk)) bk

let double_succ (x : nat) : Lemma (2 * (x + 1) == 2 * x + 2) = ()

let le_not_lt_eq (x y : nat)
  : Lemma (requires x <= y /\ ~(x < y)) (ensures x == y)
  = ()
#pop-options

ghost
fn advance_barrier_state
  (x : nat)
  (next : nat { next == x + 1 })
  requires B.barrier_state (2 * x + 1 + 1)
  ensures B.barrier_state (2 * next)
{
  rewrite B.barrier_state (2 * x + 1 + 1)
       as B.barrier_state (2 * next);
}

#push-options "--fuel 1 --ifuel 1 --z3rlimit 10"
inline_for_extraction noextract
fn run_loop
  (#et_ab #et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_c |}
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
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (#fA #fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (mapA mapB : real -> real)
  (emA emB : et_ab -> et_ab)
  (#_ : squash (MU.approx1 emA mapA))
  (#_ : squash (MU.approx1 emB mapB))
  (nthr : erased nat {
    nthr == bm/(wm*tm) * (bn/(wn*tn)) * warp_size })
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (sarA : larray et_ab (bm * bk))
  (sarB : larray et_ab (bk * bn))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#_ : squash (
    sA == from_array (rm bm bk) sarA /\
    sB == from_array (rm bk bn) sarB))
  (bid : szlt (m/bm * (n/bn)))
  (tid : szlt nthr)
  (mrow : szlt (m/bm) {
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) })
  (mcol : szlt (n/bn) {
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) })
  (warpRow : szlt (bm/(wm*tm)))
  (warpCol : szlt (bn/(wn*tn)))
  (gwRow : enatlt (m/(wm*tm)) {
    gwRow == mrow * (bm/(wm*tm)) + warpRow })
  (gwCol : enatlt (n/(wn*tn)) {
    gwCol == mcol * (bn/(wn*tn)) + warpCol })
  (rAcc0 : chest2 real (wm*tm) (wn*tn) {
    rAcc0 == const _ 0.0R })
  (aFrags : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags : array (fragment et_ab FragB tm tn tk FragLRM))
  (accFrags : array (fragment et_c FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  ()
  preserves
    gpu **
    gA |-> Frac (fA /. (m/bm * (n/bn) * nthr)) eA **
    gB |-> Frac (fB /. (m/bm * (n/bn) * nthr)) eB **
    thread_id nthr tid **
    block_id (m/bm * (n/bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (rm bk bn) sarA sarB nthr bid) **
    live aFrags **
    live bFrags
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (eA %~ rA /\ eB %~ rB) **
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol 0) **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state 0
  ensures
    fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol (k / bk)) **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (2 * (k / bk))
{
  let num_k_tiles = k /^ bk;
  let mut bkIdx : sz = 0sz;
  while (!bkIdx <^ num_k_tiles)
    invariant
      exists* (vbkIdx : sz { vbkIdx <= num_k_tiles }).
        bkIdx |-> vbkIdx **
        fragarrayAcc_approximates wm wn accFrags
          (kacc_inv m n k bm bn bk tm tn tk wm wn
            (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
            rAcc0 gwRow gwCol !bkIdx)
    invariant live aFrags ** live bFrags
    invariant
      (exists* em1. FB.bp_sharing sA em1 nthr) **
      (exists* em2. FB.bp_sharing sB em2 nthr) **
      B.barrier_state (2 * !bkIdx)
    decreases (num_k_tiles - !bkIdx)
  {
    even_2x !bkIdx;
    assert pure ((2 * !bkIdx % 2 = 0) == true);
    assert pure (even (2 * !bkIdx));

    FB.fold_barrier_p_even eA eB sA sB nthr bid !bkIdx tid;
    bp_to_rin eA eB (rm bm bk) (rm bk bn)
      sarA sarB sA sB nthr bid (2 * !bkIdx) tid;
    B.barrier_wait ();
    rout_to_bq eA eB (rm bm bk) (rm bk bn)
      sarA sarB sA sB nthr bid (2 * !bkIdx) tid;
    FB.unfold_barrier_q_even eA eB sA sB nthr bid !bkIdx tid;

    unfold FB.live_strided_chunks sA nthr tid;
    with eA0. assert (FB.own_strided_chunks sA eA0 nthr tid);
    rewrite FB.own_strided_chunks sA eA0 nthr tid
      as CV2.own_strided_chunks sA eA0 nthr tid;
    fold CV2.live_strided_chunks sA nthr tid;
    unfold FB.live_strided_chunks sB nthr tid;
    with eB0. assert (FB.own_strided_chunks sB eB0 nthr tid);
    rewrite FB.own_strided_chunks sB eB0 nthr tid
      as CV2.own_strided_chunks sB eB0 nthr tid;
    fold CV2.live_strided_chunks sB nthr tid;

    copy_tiles_out_of_matrices_vec bm bn bk sA sB gA gB
      mrow !bkIdx mcol
      (bm/^(wm*^tm)*^(bn/^(wn*^tn))*^warp_size) tid;

    rewrite CV2.own_strided_chunks sA
      (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid
      as FB.own_strided_chunks sA
        (ematrix_subtile eA bm bk mrow !bkIdx) nthr tid;
    rewrite CV2.own_strided_chunks sB
      (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid
      as FB.own_strided_chunks sB
        (ematrix_subtile eB bk bn !bkIdx mcol) nthr tid;

    odd_2x1 !bkIdx;
    assert pure (odd (2 * !bkIdx + 1));
    FB.fold_barrier_p_odd eA eB sA sB nthr bid mrow mcol !bkIdx tid;
    bp_to_rin eA eB (rm bm bk) (rm bk bn)
      sarA sarB sA sB nthr bid (2 * !bkIdx + 1) tid;
    B.barrier_wait ();

    even_2x (!bkIdx + 1);
    lemma_double_div (SZ.v k) (SZ.v bk);
    double_succ !bkIdx;
    assert pure (odd (2 * !bkIdx + 1));
    assert pure ((2 * !bkIdx + 1) < 2 * k / bk);
    assert pure (even (2 * !bkIdx + 2));
    rout_to_bq eA eB (rm bm bk) (rm bk bn)
      sarA sarB sA sB nthr bid (2 * !bkIdx + 1) tid;
    FB.unfold_barrier_q_odd eA eB sA sB nthr bid mrow mcol !bkIdx tid;

    unfold FB.bp_sharing sA
      (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    unfold FB.bp_sharing sB
      (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    let rA_sub = ematrix_subtile rA bm bk mrow !bkIdx;
    let rB_sub = ematrix_subtile rB bk bn !bkIdx mcol;
    let vbk = !bkIdx;
    subtile_approx eA rA bm bk mrow vbk;
    subtile_approx eB rB bk bn vbk mcol;

    ktile_advance
      m n k bm bn bk tm tn tk wm wn
      aFrags bFrags accFrags sA sB rA rB
      mrow mcol vbk rA_sub rB_sub
      emA emB mapA mapB rAcc0
      warpRow warpCol gwRow gwCol ();

    fold FB.bp_sharing sA
      (ematrix_subtile eA bm bk mrow !bkIdx) nthr;
    fold FB.bp_sharing sB
      (ematrix_subtile eB bk bn !bkIdx mcol) nthr;

    let next_bkIdx = !bkIdx +^ 1sz;
    advance_barrier_state (SZ.v !bkIdx) (SZ.v next_bkIdx);
    bkIdx := next_bkIdx;
  };

  assert pure (SZ.v num_k_tiles == k / bk);
  le_not_lt_eq (SZ.v !bkIdx) (SZ.v num_k_tiles);
  rewrite fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol !bkIdx)
    as fragarrayAcc_approximates wm wn accFrags
      (kacc_inv m n k bm bn bk tm tn tk wm wn
        (Chest.chest_map mapA rA) (Chest.chest_map mapB rB)
        rAcc0 gwRow gwCol (k / bk));
  rewrite B.barrier_state (2 * !bkIdx)
       as B.barrier_state (2 * (k / bk));
}
#pop-options
