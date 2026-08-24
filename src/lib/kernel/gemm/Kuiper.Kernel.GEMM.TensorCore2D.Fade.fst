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

(* The tile exactly at the fade boundary has not been combined yet.  Keeping
   this definitional fact separate avoids unfolding the whole fade beneath a
   quantified approximation relation. *)
let em_fade_comb_current_subtile_eq
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxI : natlt wm)
  (idxJ : natlt wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
  : Lemma
      (ematrix_subtile
          (em_fade_comb_tiles comb tm tn wm wn idxI idxJ rm1 rm2)
          tm tn idxI idxJ
       == ematrix_subtile rm1 tm tn idxI idxJ)
= FStar.Math.Lemmas.cancel_mul_div wm tm;
  FStar.Math.Lemmas.cancel_mul_div wn tn;
  assert (equal
    (ematrix_subtile
      (em_fade_comb_tiles comb tm tn wm wn idxI idxJ rm1 rm2)
      tm tn idxI idxJ)
    (ematrix_subtile rm1 tm tn idxI idxJ))

// Once idxI reaches wm every tile is combined, so the whole matrix collapses to
// [chest_comb comb rm1 rm2].  Proven extensionally, as [em_fade_tiles_full].
(* Every tile index (i/tm, j/tn) of a (wm*tm) x (wn*tn) matrix has flat index
   below wm*wn, so once idxI = wm the fade's boundary test holds everywhere.
   Stated explicitly: query splitting no longer lets Z3 find this nonlinear
   bound on its own. *)
let div_lt_pos (i : nat) (q w : pos) : Lemma (requires i < w * q) (ensures i / q < w)
= if i / q >= w then begin
    FStar.Math.Lemmas.lemma_mult_le_right q w (i/q);
    FStar.Math.Lemmas.euclidean_division_definition i q
  end

let tile_flat_index_bound (tm tn wm wn : pos)
  : Lemma (forall (i:natlt (wm*tm)) (j:natlt (wn*tn)). (i/tm)*wn + (j/tn) < wm*wn)
= introduce forall (i:natlt (wm*tm)) (j:natlt (wn*tn)). (i/tm)*wn + (j/tn) < wm*wn
  with begin
    div_lt_pos i tm wm;
    div_lt_pos j tn wn;
    FStar.Math.Lemmas.lemma_mult_le_right wn (i/tm + 1) wm
  end

#push-options "--z3rlimit 20"
let em_fade_comb_tiles_full
  (comb : real -> real -> real)
  (tm tn wm wn : pos)
  (idxJ : natle wn)
  (rm1 rm2 : chest2 real (wm*tm) (wn*tn))
: Lemma (em_fade_comb_tiles comb tm tn wm wn wm idxJ rm1 rm2 == Chest.chest_comb comb rm1 rm2)
= tile_flat_index_bound tm tn wm wn;
  introduce forall (i : natlt (wm*tm)) (j : natlt (wn*tn)).
    Chest.acc2 (em_fade_comb_tiles comb tm tn wm wn wm idxJ rm1 rm2) i j
    == Chest.acc2 (Chest.chest_comb comb rm1 rm2) i j
  with begin
    assert (i/tm*wn + j/tn < wm*wn)
  end;
  Kuiper.EMatrix.lemma_equal_intro
    (em_fade_comb_tiles comb tm tn wm wn wm idxJ rm1 rm2)
    (Chest.chest_comb comb rm1 rm2);
  Chest.ext
    (em_fade_comb_tiles comb tm tn wm wn wm idxJ rm1 rm2)
    (Chest.chest_comb comb rm1 rm2)
#pop-options

#push-options "--z3rlimit 20"
let em_fade_comb_current_subtile_approximates
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
= FStar.Math.Lemmas.cancel_mul_div wm tm;
  FStar.Math.Lemmas.cancel_mul_div wn tn;
  subtile_approx em
    (em_fade_comb_tiles comb tm tn wm wn idxI idxJ rm1 rm2)
    tm tn idxI idxJ;
  em_fade_comb_current_subtile_eq comb tm tn wm wn idxI idxJ rm1 rm2

(* [%~] on a [chest2] quantifies over the *flat* index, so bridging a
   per-tile approximation up to a whole-matrix one (or vice versa) needs an
   explicit hop through that flat quantifier. *)
let approximates_elim
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m1 : chest2 et rows cols)
  (m2 : chest2 real rows cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (requires m1 %~ m2) (ensures Chest.acc2 m1 i j %~ Chest.acc2 m2 i j)
= ()

let ematrix_subtile_approximates_forall
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
= introduce forall (i:natlt rows) (j:natlt cols).
    Chest.acc2 m1 i j %~ Chest.acc2 m2 i j
  with begin
    approximates_elim
      (ematrix_subtile m1 trows tcols (i/trows) (j/tcols))
      (ematrix_subtile m2 trows tcols (i/trows) (j/tcols))
      (i%trows) (j%tcols)
  end;
  lemma_approximates_intro m1 m2
#pop-options

(* Injectivity of the row-major flattening used by the fade boundary test. *)
let flat_idx_neq
  (wn : pos) (idxI tr : nat) (idxJ tc : natlt wn)
  : Lemma
    (requires tr <> idxI \/ tc <> idxJ)
    (ensures tr * wn + tc <> idxI * wn + idxJ)
= if tr = idxI then ()
  else if tr < idxI
  then begin
    FStar.Math.Lemmas.lemma_eucl_div_bound tc tr wn;
    FStar.Math.Lemmas.lemma_mult_le_left wn (tr + 1) idxI
  end
  else begin
    FStar.Math.Lemmas.lemma_eucl_div_bound idxJ idxI wn;
    FStar.Math.Lemmas.lemma_mult_le_left wn (idxI + 1) tr
  end
