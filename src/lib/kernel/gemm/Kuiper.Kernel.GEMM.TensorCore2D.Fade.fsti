module Kuiper.Kernel.GEMM.TensorCore2D.Fade

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

(* After storing tiles up to
   (idxI, idxJ) with the fused output combine, each already-written tile holds
   [chest_comb comb] of the old C tile ([rm1]) and the accumulator tile ([rm2]);
   the remaining tiles still hold the original C ([rm1]). *)
let em_fade_comb_tiles
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxI : natle wm)
  (idxJ : natle wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: chest2 real (wm*tm) (wn*tn)
=
  ematrix_from_tiles tm tn (fun i j ->
    let flat_idx = i * wn + j in
    let num_copied = idxI * wn + idxJ in
    if flat_idx < num_copied
    then Chest.chest_comb comb (ematrix_subtile rm1 tm tn i j) (ematrix_subtile rm2 tm tn i j)
    else ematrix_subtile rm1 tm tn i j)

val em_fade_comb_tiles_full
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxJ : natle wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: Lemma (em_fade_comb_tiles comb tm tn wm wn wm idxJ rm1 rm2 == Chest.chest_comb comb rm1 rm2)

val em_fade_comb_current_subtile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxI : natlt wm)
  (idxJ : natlt wn)
  (em : chest2 et (wm*tm) (wn*tn))
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
  : Lemma
    (requires em %~ em_fade_comb_tiles comb tm tn wm wn idxI idxJ rm1 rm2)
    (ensures ematrix_subtile em tm tn idxI idxJ
             %~ ematrix_subtile rm1 tm tn idxI idxJ)

val ematrix_subtile_approximates_forall
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m1 : chest2 et rows cols)
  (m2 : chest2 real rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  : Lemma
    (requires forall (tr:natlt (rows/trows)) (tc:natlt (cols/tcols)).
                ematrix_subtile m1 trows tcols tr tc %~ ematrix_subtile m2 trows tcols tr tc)
    (ensures m1 %~ m2)

val flat_idx_neq
  (wn : pos) (idxI tr : nat) (idxJ tc : natlt wn)
  : Lemma
    (requires tr <> idxI \/ tc <> idxJ)
    (ensures tr * wn + tc <> idxI * wn + idxJ)
