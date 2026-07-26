module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15 --split_queries always"

open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

let output_fragment_cell_convert_eq
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (mi : natlt wm)
  (nj : natlt wn)
  (i : natlt tm)
  (j : natlt tn)
  (f : perm)
  (v : et)
  : Lemma (
      let blockRow = bid / (n / bn) in
      let blockCol = bid % (n / bn) in
      let warpRow = wid / (bn / (wn * tn)) in
      let warpCol = wid % (bn / (wn * tn)) in
      let fragRow = tiled_cell (wm * tm) tm mi i in
      let fragCol = tiled_cell (wn * tn) tn nj j in
      let blockCellRow = tiled_cell bm (wm * tm) warpRow fragRow in
      let blockCellCol = tiled_cell bn (wn * tn) warpCol fragCol in
      tensor_pts_to_cell
        (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
        #f (idx2 i j) v
      ==
      tensor_pts_to_cell gD #f
        (idx2
          (tiled_cell m bm blockRow blockCellRow)
          (tiled_cell n bn blockCol blockCellCol))
        v)
=
  Math.Lemmas.lemma_div_exact (wm * tm) tm;
  Math.Lemmas.lemma_div_exact (wn * tn) tn;
  assert ((wm * tm) / tm == wm);
  assert ((wn * tn) / tn == wn);
  let blockRow = bid / (n / bn) in
  let blockCol = bid % (n / bn) in
  let warpRow = wid / (bn / (wn * tn)) in
  let warpCol = wid % (bn / (wn * tn)) in
  let fragRow = tiled_cell (wm * tm) tm mi i in
  let fragCol = tiled_cell (wn * tn) tn nj j in
  let blockCellRow = tiled_cell bm (wm * tm) warpRow fragRow in
  let blockCellCol = tiled_cell bn (wn * tn) warpCol fragCol in
  let dBlock = block_tile gD bm bn bid in
  let dWarp = warp_tile dBlock (wm * tm) (wn * tn) wid in
  cell_convert_eq dWarp tm tn mi nj i j f v;
  cell_convert_eq dBlock (wm * tm) (wn * tn)
    warpRow warpCol fragRow fragCol f v;
  cell_convert_eq gD bm bn
    blockRow blockCol blockCellRow blockCellCol f v;
  ()
