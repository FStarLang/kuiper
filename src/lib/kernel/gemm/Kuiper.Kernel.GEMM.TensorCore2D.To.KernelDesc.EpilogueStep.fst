module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Array2.Strided
open Kuiper.Array2.Vectorized
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Pulse.Lib.Array

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module VG = Kuiper.Array2.Vectorized.Group

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCellUpdate
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueChunkUpdate


(* Two group indices congruent mod [warp_size] are at least a full warp
   apart. *)
let group_gap (a b : nat) (lane : natlt warp_size)
  : Lemma (requires a % warp_size == lane /\ b % warp_size == lane /\ a < b)
          (ensures a + warp_size <= b)
= FStar.Math.Lemmas.euclidean_division_definition a warp_size;
  FStar.Math.Lemmas.euclidean_division_definition b warp_size;
  if b / warp_size <= a / warp_size then
    FStar.Math.Lemmas.lemma_mult_le_right warp_size
      (b / warp_size) (a / warp_size)

(* Fading group [vg] is the only difference between [lane_fade] at [vg]
   and at [vg + warp_size]. *)
let lane_fade_others
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
  (#rows #cols : pos)
  (em0 em1 : chest2 et rows cols)
  (lane : natlt warp_size)
  (vg : nat { vg % warp_size == lane })
  : Lemma (forall (i : natlt rows) (j : natlt cols).
      in_lane (chunk et #_ #hvc) rows cols lane (i, j) /\
      VG.group_of (chunk et #_ #hvc) cols i j <> vg ==>
        acc2 (lane_fade em0 em1 lane vg) i j
        == acc2 (lane_fade em0 em1 lane (vg + warp_size)) i j)
= introduce forall (i : natlt rows) (j : natlt cols).
      in_lane (chunk et #_ #hvc) rows cols lane (i, j) /\
      VG.group_of (chunk et #_ #hvc) cols i j <> vg ==>
        acc2 (lane_fade em0 em1 lane vg) i j
        == acc2 (lane_fade em0 em1 lane (vg + warp_size)) i j
  with introduce _ ==> _
  with _. (
    let g = VG.group_of (chunk et #_ #hvc) cols i j in
    if vg < g then group_gap vg g lane)


ghost
fn own_lane_cells_rw
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
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
    #(ij : (natlt rows & natlt cols){
      in_lane (chunk et #_ #hvc) rows cols lane ij})
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


#push-options "--fuel 2 --ifuel 1 --z3rlimit 60 --split_queries no"
inline_for_extraction noextract
fn epilogue_fragment_step
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, hvc : has_vec_cpy et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#m #n : szp)
  {| str : strided_row_major (rm m n) |}
  (c : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn rows cols wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (#_ : squash (chunk et_cd /?+ cols))
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
  (vg : sz{
    SZ.v vg < VG.ngroups (chunk et_cd) rows cols /\
    SZ.v vg % warp_size == lane})
  preserves
    gpu **
    c |-> Frac fC eC **
    acc |-> Frac (1.0R /. warp_size) eAcc
  requires
    pure (aligned 16 (T.core c) /\ aligned 16 (T.core d) /\
          aligned_strided_row_major (chunk et_cd) str)
  requires
    own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (lane_fade eD0 eTarget lane vg)
      lane
  ensures
    own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (lane_fade eD0 eTarget lane (SZ.v vg + warp_size))
      lane
{
  VG.group_bounds (chunk et_cd) rows cols (SZ.v vg);
  assert pure (SZ.fits (SZ.v vg * chunk et_cd));
  let flat = vg *^ chunk et_cd;
  let row : szlt rows = flat /^ cols;
  let col : szlt cols = flat %^ cols;
  assert pure (SZ.v row == VG.grow (chunk et_cd) rows cols (SZ.v vg));
  assert pure (SZ.v col == VG.gcol (chunk et_cd) rows cols (SZ.v vg));

  let em = lane_fade eD0 eTarget lane vg;
  let em' = lane_fade eD0 eTarget lane (SZ.v vg + warp_size);

  unfold own_lane_cells
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    em lane;
  VG.cells_extract_group
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    (chunk et_cd) em
    (fun ij -> in_lane (chunk et_cd) rows cols lane ij)
    (SZ.v vg) ();

  // The residual cells are untouched by the update.
  lane_fade_others eD0 eTarget (SZ.v lane) (SZ.v vg);
  forevery_ext
    #(ij : (natlt rows & natlt cols){
      in_lane (chunk et_cd) rows cols lane ij /\
      VG.group_of (chunk et_cd) cols ij._1 ij._2 =!= SZ.v vg})
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

  rewrite
    row_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      1.0R
      (VG.grow (chunk et_cd) rows cols (SZ.v vg))
      (VG.gcol (chunk et_cd) rows cols (SZ.v vg))
      (chunk et_cd)
      (VG.group_seq (chunk et_cd) em (SZ.v vg))
  as
    row_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      1.0R (SZ.v row) (SZ.v col) (chunk et_cd)
      (VG.group_seq (chunk et_cd) em (SZ.v vg));

  // The updated run is exactly group [vg] of [em'].
  assert pure (forall (x : natlt (chunk et_cd)).
    acc2 em' (VG.grow (chunk et_cd) rows cols (SZ.v vg))
             (VG.gcol (chunk et_cd) rows cols (SZ.v vg) + x)
    == acc2
        (epilogue_fragment_target comb eC
          bm bn rows cols wm wn
          (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
          (SZ.v idx) eAcc)
        (SZ.v row) (SZ.v col + x));

  epilogue_chunk_update comb c
    bm bn rows cols wm wn
    mrow mcol warpRow warpCol bid wid
    acc d idx row col
    (VG.group_seq (chunk et_cd) em (SZ.v vg))
    (VG.group_seq (chunk et_cd) em' (SZ.v vg)) ();

  rewrite
    row_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      1.0R (SZ.v row) (SZ.v col) (chunk et_cd)
      (VG.group_seq (chunk et_cd) em' (SZ.v vg))
  as
    row_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      1.0R
      (VG.grow (chunk et_cd) rows cols (SZ.v vg))
      (VG.gcol (chunk et_cd) rows cols (SZ.v vg))
      (chunk et_cd)
      (VG.group_seq (chunk et_cd) em' (SZ.v vg));

  VG.cells_restore_group
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    (chunk et_cd) em'
    (fun ij -> in_lane (chunk et_cd) rows cols lane ij)
    (SZ.v vg) ();
  fold own_lane_cells
    (output_fragment d bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
    em' lane;
}
#pop-options
