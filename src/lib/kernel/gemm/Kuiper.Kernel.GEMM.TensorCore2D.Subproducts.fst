module Kuiper.Kernel.GEMM.TensorCore2D.Subproducts

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
open Kuiper.Kernel.GEMM.TensorCore2D.Populate
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentMMA
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

    fragarray_mma tm tn tk wm wn aFrags bFrags accumFrags
      (ematrix_subtile (Chest.chest_map mapA rA) (wm*tm) tk arow !dotIdx)
      (ematrix_subtile (Chest.chest_map mapB rB) tk (wn*tn) !dotIdx bcol)
      (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled (Chest.chest_map mapA rA) (wm*tm) tk)
      (ematrix_tiled (Chest.chest_map mapB rB) tk (wn*tn)) arow bcol !dotIdx);

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
