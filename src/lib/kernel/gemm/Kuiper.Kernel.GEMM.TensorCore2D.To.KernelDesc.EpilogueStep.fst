module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCellUpdate


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
{
  unfold own_lane_cells m em1 lane;
  forevery_map
    #(ij : (natlt rows & natlt cols){in_lane rows cols lane ij})
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em1 ij._1 ij._2))
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em2 ij._1 ij._2))
    fn ij {
      rewrite
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em1 ij._1 ij._2)
      as
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em2 ij._1 ij._2);
    };
  fold own_lane_cells m em2 lane;
}


#push-options "--fuel 2 --ifuel 1 --z3rlimit 60"
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
{
  let row : szlt rows = vf /^ cols;
  let col : szlt cols = vf %^ cols;
  assert pure (row < rows);
  assert pure (col < cols);
  let eij : erased (natlt rows & natlt cols) =
    Mktuple2 #(natlt rows) #(natlt cols) (SZ.v row) (SZ.v col);
  assert pure (in_lane rows cols lane eij);
  let warpFraction = precip warp_size;
  assert pure (warpFraction == 1.0R /. warp_size);
  assert pure (1.0R /. warp_size >. 0.0R);

  let em = lane_fade eD0 eTarget lane vf;
  unfold own_lane_cells
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    em lane;
  forevery_remove'
    #(natlt rows & natlt cols)
    (fun ij -> in_lane rows cols lane ij)
    (fun ij -> tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 ij._1 ij._2)
      (acc2 em ij._1 ij._2))
    eij;

  assert tensor_pts_to_cell
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    (idx2 (reveal eij)._1 (reveal eij)._2)
    (acc2 em (reveal eij)._1 (reveal eij)._2);
  rewrite
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (reveal eij)._1 (reveal eij)._2)
      (acc2 em (reveal eij)._1 (reveal eij)._2)
  as
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (SZ.v row) (SZ.v col))
      (acc2 em (SZ.v row) (SZ.v col));
  epilogue_cell_update comb c
    bm bn rows cols wm wn
    mrow mcol warpRow warpCol bid wid
    acc d idx row col;
  rewrite
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (SZ.v row) (SZ.v col))
      (acc2
        (epilogue_fragment_target comb eC
          bm bn rows cols wm wn
          (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
          (SZ.v idx) eAcc)
        (SZ.v row) (SZ.v col))
  as
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (SZ.v row) (SZ.v col))
      (acc2 eTarget (SZ.v row) (SZ.v col));

  let em' =
    upd2 em (SZ.v row) (SZ.v col)
      (acc2 eTarget (SZ.v row) (SZ.v col));
  rewrite each (acc2 eTarget (SZ.v row) (SZ.v col)) as
    acc2 em' (SZ.v row) (SZ.v col);
  rewrite
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (SZ.v row) (SZ.v col))
      (acc2 em' (SZ.v row) (SZ.v col))
  as
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (reveal eij)._1 (reveal eij)._2)
      (acc2 em' (reveal eij)._1 (reveal eij)._2);
  forevery_ext
    #(ij : (natlt rows & natlt cols){
      in_lane rows cols lane ij /\ ij =!= eij})
    (fun ij -> tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 ij._1 ij._2)
      (acc2 em ij._1 ij._2))
    (fun ij -> tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 ij._1 ij._2)
      (acc2 em' ij._1 ij._2));
  forevery_insert
    #(natlt rows & natlt cols)
    #(fun ij -> in_lane rows cols lane ij /\ ij =!= eij)
    (fun ij -> tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 ij._1 ij._2)
      (acc2 em' ij._1 ij._2))
    eij;
  forevery_refine_ext
    #(natlt rows & natlt cols)
    #(fun ij ->
      (in_lane rows cols lane ij /\ ij =!= eij) \/ reveal eij == ij)
    (fun ij -> in_lane rows cols lane ij)
    (fun ij -> tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 ij._1 ij._2)
      (acc2 em' ij._1 ij._2));
  fold own_lane_cells
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    em' lane;

  lane_fade_step eD0 eTarget lane vf;
  assert pure (lane_coincide lane em'
    (lane_fade eD0 eTarget lane (SZ.v vf + warp_size)));
  own_lane_cells_rw
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    lane em'
    (lane_fade eD0 eTarget lane (SZ.v vf + warp_size));
}
#pop-options
