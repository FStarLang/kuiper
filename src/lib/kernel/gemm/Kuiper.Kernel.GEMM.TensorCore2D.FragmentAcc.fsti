module Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc

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

module Chest = Kuiper.Chest

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

val subtile_approx
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (rr : chest2 real rows cols)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (requires em %~ rr)
          (ensures ematrix_subtile em trows tcols tr tc
                   %~ ematrix_subtile rr trows tcols tr tc)

val chest_map_subtile_comm
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

val chest_comb_approx
  (#et : Type0) {| scalar et, real_like et |}
  (ecomb : et -> et -> et)
  (comb : real -> real -> real)
  (#rows #cols : nat)
  (e1 e2 : chest2 et rows cols)
  (r1 r2 : chest2 real rows cols)
  : Lemma
    (requires Kuiper.Approximates.approx2 ecomb comb /\ e1 %~ r1 /\ e2 %~ r2)
    (ensures Chest.chest_comb ecomb e1 e2 %~ Chest.chest_comb comb r1 r2)
