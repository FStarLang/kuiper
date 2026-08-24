module Kuiper.Kernel.GEMM.TensorCore2D.ProofSupport

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 120"

open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.TensorCore
open Pulse.Lib.Array

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
