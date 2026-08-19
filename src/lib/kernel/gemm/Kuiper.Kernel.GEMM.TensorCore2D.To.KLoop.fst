module Kuiper.Kernel.GEMM.TensorCore2D.To.KLoop

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
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module CV2 = Kuiper.Kernel.GEMM.Copy.Vec2
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc


open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState
open Kuiper.Kernel.GEMM.TensorCore2D.To.Fragments
open Kuiper.Kernel.GEMM.TensorCore2D.To.KLoopInvariant

(* [matmul_tiles_lemma]'s monoid-law arguments.  Inline [fun _ -> ()] lambdas
   have their obligation checked in the caller's enormous context, where the
   query now times out; discharging them once here, in a clean context, is
   cheap.  (Same fix as in [Kuiper.Kernel.GEMM.TensorCore2D].) *)
let real_add_zero (x : real) : squash (add x zero == x /\ add zero x == x) = ()
let real_add_assoc (x y z : real) : squash (add x (add y z) == add (add x y) z) = ()

(* Ditto for the "adding the zero matrix is the identity" fact needed to line
   up [matmul_tiles_lemma]'s conclusion: in [k_loop]'s context the query times
   out, in a clean context it is immediate. *)
let matplus_zero_left (#r #c : nat) (x : chest2 real r c)
  : Lemma (matplus (const _ 0.0R) x `equal` x)
  = ()

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

(* [FB.fold_barrier_p_*] produces [FB.barrier_p ...], but the barrier token is
   stated in terms of the contract's [.rin]/[.rout], which are built over the
   *raw* arrays.  Neither the projection out of the record literal built by
   [FB.contract] nor the [from_array l (core sa)] substitution reduces for the
   SMT solver anymore, so do the projection by normalization and the
   substitution with [rewrite each] (same bridge as in
   [Kuiper.Kernel.GEMM.TensorCore2D] / [BlockTiling2D]). *)
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

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn k_loop_step
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc |}
  {| real_like et_ab, real_like et_acc |}
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
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m /\ wn * tn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#_ : squash (eA %~ rA /\ eB %~ rB))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#fA #fB : perm)
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (mrow : szlt (m / bm) {
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) })
  (mcol : szlt (n / bn) {
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) })
  (warpRow : szlt (bm / (wm * tm)) {
    SZ.v warpRow == (SZ.v tid / warp_size) /
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (warpCol : szlt (bn / (wn * tn)) {
    SZ.v warpCol == (SZ.v tid / warp_size) %
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (gwRow : enatlt (m / (wm * tm)) {
    gwRow == mrow * (bm / (wm * tm)) + warpRow})
  (gwCol : enatlt (n / (wn * tn)) {
    gwCol == mcol * (bn / (wn * tn)) + warpCol})
  (rAcc0 : chest2 real (wm * tm) (wn * tn) {
    rAcc0 == const _ 0.0R})
  (aFrags : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags : array (fragment et_ab FragB tm tn tk FragLRM))
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  (v : szle (k / bk))
  (#_ : squash (SZ.v v < k / bk))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid **
    block_id (m / bm * (n / bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (rm bk bn)
        (core sA) (core sB) nthr bid)
  requires
    fragarrayAcc_approximates wm wn accFrags
      (tiled_partial_matmul rAcc0
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol v) **
    live aFrags **
    live bFrags **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (barrier_iteration v)
  returns vnext : szle (k / bk)
  ensures
    pure (SZ.v vnext == SZ.v v + 1) **
    fragarrayAcc_approximates wm wn accFrags
      (tiled_partial_matmul rAcc0
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol (v + 1)) **
    live aFrags **
    live bFrags **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (barrier_iteration vnext)
{
  even_2x v;
  assert pure ((2 * v % 2 = 0) == true);
  assert pure (even (2 * v));

  FB.fold_barrier_p_even eA eB sA sB nthr bid v tid;
  bp_to_rin eA eB (rm bm bk) (rm bk bn) (core sA) (core sB) sA sB nthr bid (2 * v) tid;

  B.barrier_wait ();

  rout_to_bq eA eB (rm bm bk) (rm bk bn) (core sA) (core sB) sA sB nthr bid (2 * v) tid;
  FB.unfold_barrier_q_even eA eB sA sB nthr bid v tid;

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
    mrow v mcol
    (bm /^ (wm *^ tm) *^ (bn /^ (wn *^ tn)) *^ warp_size)
    tid;

  rewrite
    CV2.own_strided_chunks sA (ematrix_subtile eA bm bk mrow v) nthr tid
    as FB.own_strided_chunks sA (ematrix_subtile eA bm bk mrow v) nthr tid;
  rewrite
    CV2.own_strided_chunks sB (ematrix_subtile eB bk bn v mcol) nthr tid
    as FB.own_strided_chunks sB (ematrix_subtile eB bk bn v mcol) nthr tid;

  odd_2x1 v;
  assert pure (odd (2 * v + 1));
  FB.fold_barrier_p_odd eA eB sA sB nthr bid mrow mcol v tid;
  bp_to_rin eA eB (rm bm bk) (rm bk bn) (core sA) (core sB) sA sB nthr bid (2 * v + 1) tid;

  B.barrier_wait ();

  even_2x (v + 1);
  assert pure (2 * (v + 1) == 2 * v + 2);
  assert pure (odd (2 * v + 1));
  assert pure ((2 * v + 1) < 2 * k / bk);
  assert pure (even (2 * v + 2));
  rout_to_bq eA eB (rm bm bk) (rm bk bn) (core sA) (core sB) sA sB nthr bid (2 * v + 1) tid;
  FB.unfold_barrier_q_odd eA eB sA sB nthr bid mrow mcol v tid;

  unfold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow v) nthr;
  unfold FB.bp_sharing sB (ematrix_subtile eB bk bn v mcol) nthr;

  let rA_sub = ematrix_subtile rA bm bk mrow v;
  let rB_sub = ematrix_subtile rB bk bn v mcol;
  with rAcc. assert fragarrayAcc_approximates wm wn accFrags rAcc;
  subproducts_tc_2d bm bn bk tm tn tk wm wn
    aFrags bFrags accFrags sA sB
    rA_sub rB_sub rAcc warpRow warpCol;

  loop_invariant_lemma
    m n k bm bn bk tm tn tk wm wn
    mrow mcol warpRow warpCol gwRow gwCol v
    rA rB rAcc0 rAcc rA_sub rB_sub;
  assert pure (
    rAcc `matplus`
      matmul (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
             (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol)
    ==
    tiled_partial_matmul rAcc0
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol (v + 1));
  rewrite_fragarrayAcc wm wn accFrags
    (rAcc `matplus`
      matmul (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
             (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol))
    (tiled_partial_matmul rAcc0
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol (v + 1));

  fold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow v) nthr;
  fold FB.bp_sharing sB (ematrix_subtile eB bk bn v mcol) nthr;
  v +^ 1sz
}
#pop-options

inline_for_extraction noextract
fn compute_acc
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc |}
  {| real_like et_ab, real_like et_acc |}
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
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
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
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#_ : squash (eA %~ rA /\ eB %~ rB))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (rm bk bn))
  (#fA #fB : perm)
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (mrow : szlt (m / bm) {
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) })
  (mcol : szlt (n / bn) {
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) })
  (warpRow : szlt (bm / (wm * tm)) {
    SZ.v warpRow == (SZ.v tid / warp_size) /
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (warpCol : szlt (bn / (wn * tn)) {
    SZ.v warpCol == (SZ.v tid / warp_size) %
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid **
    block_id (m / bm * (n / bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (rm bk bn)
        (core sA) (core sB) nthr bid)
  requires
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state 0
  returns accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc)
  ensures
    pure (Pulse.Lib.Array.length accFrags == wm * wn) **
    fragarrayAcc_approximates wm wn accFrags
      (kernel_warp_matmul rA rB
        bm bn bk tm tn tk wm wn nthr bid tid) **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (tile_barrier_iteration k bk)
{
  let num_k_tiles = k /^ bk;
  (* [mrow * (bm/(wm*tm)) + warpRow < m/(wm*tm)] (and its column analogue): the
     nested block/warp tile index bound.  With one SMT query per proof
     obligation the nonlinear steps are no longer found on their own, so spell
     them out (same sequence as in [...To.KernelBody]). *)
  Kuiper.Divides.lemma_div_product (wm * tm) bm m;
  FStar.Math.Lemmas.lemma_eucl_div_bound
    warpRow mrow (bm / (wm * tm));
  FStar.Math.Lemmas.lemma_mult_le_left
    (bm / (wm * tm)) (mrow + 1) (m / bm);
  let gwRow : enatlt (m / (wm * tm)) =
    mrow * (bm / (wm * tm)) + warpRow;
  Kuiper.Divides.lemma_div_product (wn * tn) bn n;
  FStar.Math.Lemmas.lemma_eucl_div_bound
    warpCol mcol (bn / (wn * tn));
  FStar.Math.Lemmas.lemma_mult_le_left
    (bn / (wn * tn)) (mcol + 1) (n / bn);
  let gwCol : enatlt (n / (wn * tn)) =
    mcol * (bn / (wn * tn)) + warpCol;

  let aFrags =
    __alloc_array_fragment et_ab FragA tm tn tk FragLRM wm;
  let bFrags =
    __alloc_array_fragment et_ab FragB tm tn tk FragLRM wn;
  let accFrags =
    __alloc_array_fragment et_acc FragAcc tm tn tk FragLAcc (wm *^ wn);

  populate_acc_with_zero tm tn tk wm wn accFrags;
  let rAcc0 : chest2 real (wm * tm) (wn * tn) = const _ 0.0R;
  assert (rewrites_to rAcc0 (const _ 0.0R));
  rewrite fragarrayAcc_approximates wm wn accFrags rAcc0
    as fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol 0);

  let mut bkIdx : szle num_k_tiles = 0sz;
  while (!bkIdx <^ num_k_tiles)
    invariant
      exists* (vbkIdx : szle num_k_tiles).
        bkIdx |-> vbkIdx **
        fragarrayAcc_approximates wm wn accFrags
          (__gmatmul_single rAcc0 matmul matplus
            (ematrix_tiled rA (wm * tm) bk)
            (ematrix_tiled rB bk (wn * tn))
            gwRow gwCol !bkIdx)
    invariant live aFrags ** live bFrags
    invariant
      (exists* em1. FB.bp_sharing sA em1 nthr) **
      (exists* em2. FB.bp_sharing sB em2 nthr) **
      B.barrier_state (2 * !bkIdx)
    decreases (num_k_tiles - !bkIdx)
  {
    let vnext = k_loop_step gA gB
      bm bn bk tm tn tk wm wn
      rA rB nthr sA sB
      bid tid mrow mcol warpRow warpCol gwRow gwCol rAcc0
      aFrags bFrags accFrags !bkIdx;
    rewrite each (SZ.v !bkIdx + 1) as SZ.v vnext;
    bkIdx := !bkIdx +^ 1sz;
  };

  assert
    fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol (k / bk));
  assert pure (
    gwRow == warp_tile_i #m #n bm bn bk tm tn tk wm wn
      nthr bid (tid / warp_size));
  assert pure (
    gwCol == warp_tile_j #m #n bm bn bk tm tn tk wm wn
      nthr bid (tid / warp_size));

  matmul_tiles_lemma real_add_zero real_add_assoc
    (wm * tm) (wn * tn) bk rAcc0 rA rB gwRow gwCol;
  let rAcc' : chest2 real (wm * tm) (wn * tn) =
    gmatmul_single rAcc0 matmul matplus
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol;
  assert pure (
    (__gmatmul_single rAcc0 matmul matplus
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol !bkIdx) == rAcc');

  let rAcc : chest2 real (wm * tm) (wn * tn) =
    MS.matmul
      (ematrix_subtile rA (wm * tm) k
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)) 0)
      (ematrix_subtile rB k (wn * tn) 0
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)));
  matplus_zero_left rAcc;
  assert pure (rAcc' == rAcc);
  rewrite
    fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol !bkIdx)
    as fragarrayAcc_approximates wm wn accFrags rAcc;

  with vaFrags. assert aFrags |-> vaFrags;
  drop_ (aFrags |-> vaFrags);
  with vbFrags. assert bFrags |-> vbFrags;
  drop_ (bFrags |-> vbFrags);
  accFrags
}
