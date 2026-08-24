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

(* Taking a sub-tile preserves approximation (cellwise consequence of [%~]). *)
let subtile_approx
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
= ()

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
