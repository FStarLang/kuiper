module Kuiper.Kernel.GEMM.TensorCore2D.To.Fragments

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Spec.GEMM
open Kuiper.TensorCore

module Chest = Kuiper.Chest
module Acc = Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
module Generic = Kuiper.Kernel.GEMM.TensorCore2D.Subproducts
module State = Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState

// Transport an accumulator predicate across an extensional matrix equality.
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

// The To kernel is the identity-map specialization of the mapped pipeline.
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
    live aFrags ** live bFrags
  requires
    fragarrayAcc_approximates wm wn accumFrags rAcc
  ensures
    fragarrayAcc_approximates wm wn accumFrags
      (rAcc `matplus` matmul (ematrix_subtile rA (wm*tm) bk arow 0)
                             (ematrix_subtile rB bk (wn*tn) 0 bcol))
{
  unfold State.fragarrayAcc_approximates wm wn accumFrags rAcc;
  fold Acc.fragarrayAcc_approximates wm wn accumFrags rAcc;

  Generic.subproducts_tc_2d bm bn bk tm tn tk wm wn
    aFrags bFrags accumFrags gA gB rA rB
    (fun (x:et_ab) -> x) (fun (x:et_ab) -> x)
    (fun (x:real) -> x) (fun (x:real) -> x)
    rAcc arow bcol;

  assert pure (equal (Chest.chest_map (fun x -> x) rA) rA);
  assert pure (equal (Chest.chest_map (fun x -> x) rB) rB);

  unfold Acc.fragarrayAcc_approximates wm wn accumFrags;
  fold State.fragarrayAcc_approximates wm wn accumFrags
    (rAcc `matplus` matmul (ematrix_subtile rA (wm*tm) bk arow 0)
                           (ematrix_subtile rB bk (wn*tn) 0 bcol));
}
