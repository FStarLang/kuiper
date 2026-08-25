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

(* Keep threshold-bump stability standalone so the nonlinear [flat_idx_neq]
   step is discharged in a small context. *)
#push-options "--z3rlimit 40"
let fade_comb_subtile_stable
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxI : natlt wm)
  (idxJ : natlt wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
  (tr : natlt wm)
  (tc : natlt wn)
  : Lemma
    (requires tr <> idxI \/ tc <> idxJ)
    (ensures ematrix_subtile (em_fade_comb_tiles comb tm tn wm wn idxI (idxJ+1) rm1 rm2) tm tn tr tc
          == ematrix_subtile (em_fade_comb_tiles comb tm tn wm wn idxI idxJ rm1 rm2) tm tn tr tc)
=
  FStar.Math.Lemmas.cancel_mul_div wm tm;
  FStar.Math.Lemmas.cancel_mul_div wn tn;
  flat_idx_neq wn idxI tr idxJ tc
#pop-options

#push-options "--z3rlimit 80"
let lemma_update_tile_fade_comb_approximates
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
=
  // [ematrix_subtile]'s index refinements need [(wm*tm)/tm == wm] and
  // [(wn*tn)/tn == wn]; supply them up front.
  FStar.Math.Lemmas.cancel_mul_div wm tm;
  FStar.Math.Lemmas.cancel_mul_div wn tn;
  // The (idxI, idxJ) tile is still uncombined in the pre-state fade, so its
  // subtile of [em] approximates [ematrix_subtile rm1 tm tn idxI idxJ].
  em_fade_comb_current_subtile_approximates
    comb tm tn wm wn idxI idxJ em rm1 rm2;
  chest_comb_approx ecomb comb
    (ematrix_subtile em tm tn idxI idxJ) etile
    (ematrix_subtile rm1 tm tn idxI idxJ) (ematrix_subtile rm2 tm tn idxI idxJ);
  // Reduce the whole-matrix goal to a per-tile one: for the just-written
  // tile the two facts above already give what is needed (via
  // [subtile_of_update_tile] and [tiles_from_subtiles_id]); for every other
  // tile, its flat index differs from [idxI*wn+idxJ] (since (tr,tc) is not
  // (idxI, idxJ) and columns are bounded by [wn]), so the fade's boundary
  // test is unaffected by advancing [idxJ] to [idxJ + 1], and the tile is
  // untouched by [update_tile].
  introduce forall (tr:natlt wm) (tc:natlt wn).
    ematrix_subtile
      (update_tile em tm tn idxI idxJ
        (Chest.chest_comb ecomb (ematrix_subtile em tm tn idxI idxJ) etile))
      tm tn tr tc
    %~ ematrix_subtile (em_fade_comb_tiles comb tm tn wm wn idxI (idxJ + 1) rm1 rm2) tm tn tr tc
  with begin
    if tr = idxI && tc = idxJ
    then ()
    else begin
      fade_comb_subtile_stable comb tm tn wm wn idxI idxJ rm1 rm2 tr tc;
      subtile_approx
        em (em_fade_comb_tiles comb tm tn wm wn idxI idxJ rm1 rm2)
        tm tn tr tc
    end
  end;
  ematrix_subtile_approximates_forall
    (update_tile em tm tn idxI idxJ
      (Chest.chest_comb ecomb (ematrix_subtile em tm tn idxI idxJ) etile))
    (em_fade_comb_tiles comb tm tn wm wn idxI (idxJ + 1) rm1 rm2)
    tm tn
#pop-options
