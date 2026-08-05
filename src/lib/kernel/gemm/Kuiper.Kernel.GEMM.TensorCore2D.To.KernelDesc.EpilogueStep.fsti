module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15 --split_queries always"

open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCellUpdate

let lane_coincide
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (lane : natlt warp_size)
  (em1 em2 : chest2 et rows cols)
  : prop
= forall (i : natlt rows) (j : natlt cols).
    in_lane rows cols lane (i, j) ==> acc2 em1 i j == acc2 em2 i j

ghost
fn own_lane_cells_rw
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (lane : natlt warp_size)
  (em1 em2 : chest2 et rows cols)
  (#_ : squash (lane_coincide lane em1 em2))
  requires own_lane_cells m em1 lane
  ensures own_lane_cells m em2 lane
let lane_fade
  (#et : Type0) {| scalar et |}
  (#rows #cols : pos)
  (em0 em1 : chest2 et rows cols)
  (lane : natlt warp_size)
  (upto : nat)
  : chest2 et rows cols
= mk2 (fun i j ->
    let flat = i * cols + j in
    if in_lane rows cols lane (i, j) && flat < upto
    then acc2 em1 i j
    else acc2 em0 i j)

let lane_fade_start
  (#et : Type0) {| scalar et |}
  (#rows #cols : pos)
  (em0 em1 : chest2 et rows cols)
  (lane : natlt warp_size)
  : Lemma (lane_coincide lane em0 (lane_fade em0 em1 lane lane))
= ()

let lane_fade_step
  (#et : Type0) {| scalar et |}
  (#rows #cols : pos)
  (em0 em1 : chest2 et rows cols)
  (lane : natlt warp_size)
  (flat : nat{flat < rows * cols /\ flat % warp_size == lane})
  : Lemma (
      let row = flat / cols in
      let col = flat % cols in
      lane_coincide lane
        (upd2 (lane_fade em0 em1 lane flat) row col (acc2 em1 row col))
        (lane_fade em0 em1 lane (flat + warp_size)))
=
  let row = flat / cols in
  let col = flat % cols in
  let a = upd2 (lane_fade em0 em1 lane flat) row col (acc2 em1 row col) in
  let b = lane_fade em0 em1 lane (flat + warp_size) in
  FStar.Math.Lemmas.lemma_div_mod flat cols;
  introduce forall (i : natlt rows) (j : natlt cols).
    in_lane rows cols lane (i, j) ==> acc2 a i j == acc2 b i j
  with introduce _ ==> _
  with (
    if i = row && j = col then ()
    else begin
      let flat' = i * cols + j in
      if flat' >= flat then begin
        (if flat' = flat then lemma_eucl_unique cols row col i j);
        (if flat' < flat + warp_size
         then FStar.Math.Lemmas.lemma_mod_plus_injective warp_size flat (flat' - flat) 0)
      end
    end
  )

let lane_fade_done
  (#et : Type0) {| scalar et |}
  (#rows #cols : pos)
  (em0 em1 : chest2 et rows cols)
  (lane : natlt warp_size)
  (upto : nat{rows * cols <= upto})
  : Lemma (lane_coincide lane (lane_fade em0 em1 lane upto) em1)
= ()

inline_for_extraction noextract
fn epilogue_fragment_step
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
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
  (#lAcc : layout2 rows cols) {| T.ctlayout lAcc |}
  (acc : array2 et_acc lAcc)
  (#eAcc : chest2 et_acc rows cols)
  (d : array2 et_cd (rm m n))
  (idx : szlt (wm * wn))
  (lane : szlt warp_size)
  (#_ : squash (SZ.fits (rows * cols + warp_size)))
  (eD0 eTarget : chest2 et_cd rows cols)
  (#_ : squash (
    eTarget ==
    epilogue_fragment_target comb eC
      bm bn rows cols wm wn
      (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
      (SZ.v idx) eAcc))
  (vf : sz{
    SZ.v vf < rows * cols /\
    SZ.v vf % warp_size == lane /\
    SZ.v vf <= rows * cols + warp_size})
  preserves
    gpu **
    c |-> Frac fC eC **
    acc |-> Frac (1.0R /. warp_size) eAcc
  requires
    own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (lane_fade eD0 eTarget lane vf)
      lane
  ensures
    own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (lane_fade eD0 eTarget lane (SZ.v vf + warp_size))
      lane
