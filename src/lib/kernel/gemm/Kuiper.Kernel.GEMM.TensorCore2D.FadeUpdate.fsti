module Kuiper.Kernel.GEMM.TensorCore2D.FadeUpdate

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
open Kuiper.Kernel.GEMM.TensorCore2D.FragmentAcc
open Kuiper.Kernel.GEMM.TensorCore2D.Fade

val lemma_update_tile_fade_comb_approximates
  (#et : Type0) {| scalar et, real_like et|}
  (ecomb : et -> et -> et)
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxI : natlt wm)
  (idxJ : natlt wn)
  (em : chest2 et (wm*tm) (wn*tn))
  (etile : chest2 et tm tn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: Lemma
  (requires
    Kuiper.Approximates.approx2 ecomb comb /\
    (em %~ (em_fade_comb_tiles comb tm tn wm wn idxI idxJ rm1 rm2)) /\
    (etile %~ (ematrix_subtile rm2 tm tn idxI idxJ)))
  (ensures
    (update_tile em tm tn idxI idxJ
      (Chest.chest_comb ecomb (ematrix_subtile em tm tn idxI idxJ) etile))
    %~ (em_fade_comb_tiles comb tm tn wm wn idxI (idxJ + 1) rm1 rm2))
