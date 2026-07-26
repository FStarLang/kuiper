module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Epilogue

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
module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc



open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCellUpdate

let epilogue_chest_approx
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, real_like et_cd, scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #cols : nat)
  (eC : chest2 et_cd rows cols)
  (eAcc : chest2 et_acc rows cols)
  (rC rAcc : chest2 real rows cols)
  : Lemma
      (requires eC %~ rC /\ eAcc %~ rAcc)
      (ensures epilogue_chest comb eC eAcc %~ chest_comb comb_r rC rAcc)
= ()

let ematrix_subtile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (e : chest2 et rows cols)
  (r : chest2 real rows cols)
  (trows : pos{trows /?+ rows})
  (tcols : pos{tcols /?+ cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma
      (requires e %~ r)
      (ensures
        ematrix_subtile e trows tcols tr tc
          %~ ematrix_subtile r trows tcols tr tc)
= ()


#push-options "--split_queries no"
inline_for_extraction noextract
fn epilogue_fragment_from_warp
  (#et_cd #et_acc : Type0)
  {| scd : scalar et_cd, real_like et_cd,
     sacc : scalar et_acc, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n : szp)
  (c : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn rows cols wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (warpRow : szlt (bm / (wm * rows)))
  (warpCol : szlt (bn / (wn * cols)))
  (bid : szlt (m / bm * (n / bn)))
  (wid : szlt (bm / (wm * rows) * (bn / (wn * cols))))
  (#_ : squash (
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) /\
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) /\
    SZ.v warpRow == SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v cols)) /\
    SZ.v warpCol == SZ.v wid % (SZ.v bn / (SZ.v wn * SZ.v cols))))
  (#fC : perm)
  (#eC : chest2 et_cd m n)
  (#rC : chest2 real m n)
  (#_ : squash (eC %~ rC))
  (#lAcc : layout2 rows cols) {| T.ctlayout lAcc |}
  (acc : array2 et_acc lAcc)
  (#eAcc : chest2 et_acc rows cols)
  (#rAcc : chest2 real rows cols)
  (#_ : squash (eAcc %~ rAcc))
  (d : array2 et_cd (rm m n))
  (idx : szlt (wm * wn))
  (lane : szlt warp_size)
  (#_ : squash (SZ.fits (rows * cols + warp_size)))
  preserves
    gpu **
    c |-> Frac fC eC **
    acc |-> Frac (1.0R /. warp_size) eAcc
  requires
    live_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      lane
  ensures
    own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (epilogue_chest comb
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile eC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        eAcc)
      lane **
    pure (
      epilogue_chest comb
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile eC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        eAcc
      %~
      chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile rC bm bn (SZ.v mrow) (SZ.v mcol))
            (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
          rows cols (SZ.v idx / wn) (SZ.v idx % wn))
        rAcc)
{
  let eTarget : chest2 et_cd (SZ.v rows) (SZ.v cols) =
    epilogue_fragment_target comb eC
      bm bn rows cols wm wn
      (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
      (SZ.v idx) eAcc;
  epilogue_fragment_target_eq comb eC
    bm bn rows cols wm wn
    (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
    (SZ.v idx) eAcc;
  unfold live_lane_cells
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    (SZ.v lane);
  with (eD0 : chest2 _ _ _).
    assert own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      eD0 (SZ.v lane);
  lane_fade_start eD0 eTarget (SZ.v lane);
  own_lane_cells_rw
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    lane eD0 (lane_fade eD0 eTarget lane lane);

  let area = rows *^ cols;
  let mut flat : sz = lane;
  while (!flat <^ area)
    invariant live flat
    invariant pure (!flat % warp_size == lane)
    invariant pure (!flat <= rows * cols + warp_size)
    invariant
      own_lane_cells
        (output_fragment d bm bn rows cols wm wn
          (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
        (lane_fade eD0 eTarget lane !flat)
        lane
    decreases (area + warp_size - !flat)
  {
    epilogue_fragment_step comb c
      bm bn rows cols wm wn
      mrow mcol warpRow warpCol bid wid
      acc d idx lane eD0 eTarget !flat;
    let vflat = !flat;
    assert pure (SZ.v vflat < SZ.v vflat + warp_size);
    assert pure (
      SZ.v (vflat +^ warp_size) == SZ.v vflat + warp_size);
    assert pure (
      SZ.v area + warp_size - SZ.v (vflat +^ warp_size)
        < SZ.v area + warp_size - SZ.v vflat);
    flat := vflat +^ warp_size;
  };

  lane_fade_done eD0 eTarget lane !flat;
  own_lane_cells_rw
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    lane
    (lane_fade eD0 eTarget lane !flat)
    eTarget;
  let eCBlock = ematrix_subtile eC bm bn (SZ.v mrow) (SZ.v mcol);
  let rCBlock = ematrix_subtile rC bm bn (SZ.v mrow) (SZ.v mcol);
  ematrix_subtile_approximates eC rC
    (SZ.v bm) (SZ.v bn) (SZ.v mrow) (SZ.v mcol);
  let eCWarp = ematrix_subtile eCBlock
    (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol);
  let rCWarp = ematrix_subtile rCBlock
    (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol);
  ematrix_subtile_approximates eCBlock rCBlock
    (SZ.v wm * SZ.v rows) (SZ.v wn * SZ.v cols)
    (SZ.v warpRow) (SZ.v warpCol);
  let eCFrag = ematrix_subtile eCWarp rows cols
    (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn);
  let rCFrag = ematrix_subtile rCWarp rows cols
    (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn);
  ematrix_subtile_approximates eCWarp rCWarp
    (SZ.v rows) (SZ.v cols)
    (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn);
  epilogue_chest_approx comb comb_r eCFrag eAcc rCFrag rAcc;
}
#pop-options
