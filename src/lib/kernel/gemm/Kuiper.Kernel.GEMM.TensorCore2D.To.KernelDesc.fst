module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

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

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module CV2 = Kuiper.Kernel.GEMM.Copy.Vec2
module BW = Kuiper.Barrier.Warp

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc


module MS = Kuiper.Spec.GEMM

let in_lane_covers_all
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  : Lemma (exists lane. in_lane rows cols lane ij)
= let lane : natlt warp_size =
    (ij._1 * cols + ij._2) % warp_size in
  assert (in_lane rows cols lane ij);
  ()

let in_lane_no_overlap
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  (lane1 lane2 : natlt warp_size)
  : Lemma
      (requires in_lane rows cols lane1 ij /\ in_lane rows cols lane2 ij)
      (ensures lane1 == lane2)
= ()

ghost
fn split_array2_into_lane_cells
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#em : chest2 et rows cols)
  requires m |-> em
  ensures forall+ (lane : natlt warp_size). own_lane_cells m em lane
{
  tensor_ilower2 m;
  forevery_flatten _;
  Classical.forall_intro (in_lane_covers_all rows cols);
  forevery_refine_ext #_ #(fun _ -> True)
    (fun (ij : natlt rows & natlt cols) -> exists lane. in_lane rows cols lane ij) _;
  Classical.forall_intro_3
    (fun ij lane1 -> Classical.move_requires
      (in_lane_no_overlap rows cols ij lane1));
  forevery_split_or_n _ _;
  forevery_map
    (fun lane ->
      forall+ (ij : (natlt rows & natlt cols){
        in_lane rows cols lane ij}).
        tensor_pts_to_cell m (idx2 ij._1 ij._2)
          (acc2 em ij._1 ij._2))
    (fun lane -> own_lane_cells m em lane)
    fn lane { fold own_lane_cells m em lane };
}

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

ghost
fn join_array2_from_lane_cells
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  (#em : chest2 et rows cols)
  requires forall+ (lane : natlt warp_size). own_lane_cells m em lane
  ensures m |-> em
{
  forevery_map
    (fun lane -> own_lane_cells m em lane)
    (fun lane ->
      forall+ (ij : (natlt rows & natlt cols){in_lane rows cols lane ij}).
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    fn lane { unfold own_lane_cells m em lane };
  forevery_join_or_n
    (fun (lane : natlt warp_size) ij -> in_lane rows cols lane ij)
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  Classical.forall_intro (in_lane_covers_all rows cols);
  Classical.forall_intro_3
    (fun ij lane1 -> Classical.move_requires
      (in_lane_no_overlap rows cols ij lane1));
  forevery_refine_ext #_
    #(fun (ij : natlt rows & natlt cols) ->
      exists lane. in_lane rows cols lane ij)
    (fun _ -> True) _;
  forevery_unflatten' _;
  tensor_iraise2 m;
}

ghost
fn join_lane_cells_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  (r : chest2 real rows cols)
  requires
    forall+ (lane : natlt warp_size).
      exists* (em : chest2 et rows cols).
        own_lane_cells m em lane ** pure (em %~ r)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)
{
  let ff = forevery_exists #(natlt warp_size)
    (fun lane em -> own_lane_cells m em lane ** pure (em %~ r));
  let em' : chest2 et rows cols =
    mk2 (fun i j ->
      let lane : natlt warp_size = (i * cols + j) % warp_size in
      acc2 (ff lane) i j);

  forevery_unzip
    (fun lane -> own_lane_cells m (ff lane) lane)
    (fun lane -> pure (ff lane %~ r));
  forevery_elim_pure (fun lane -> ff lane %~ r);
  assert pure (em' %~ r);
  forevery_map
    (fun lane -> own_lane_cells m (ff lane) lane)
    (fun lane -> own_lane_cells m em' lane)
    fn lane {
      assert pure (lane_coincide lane (ff lane) em');
      own_lane_cells_rw m lane (ff lane) em';
    };
  join_array2_from_lane_cells m;
}

ghost
fn array2_untile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? cols})
  {| enumerable (natlt (rows / trows)),
     enumerable (natlt (cols / tcols)) |}
  (r : chest2 real rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (#_ : squash (SZ.fits (rows / trows)))
  (#_ : squash (SZ.fits (cols / tcols)))
  requires
    forall+ (tr : natlt (rows / trows))
             (tc : natlt (cols / tcols)).
      exists* (em : chest2 et trows tcols).
        array2_subtile m trows tcols tr tc |-> em **
        pure (em %~ ematrix_subtile r trows tcols tr tc)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)
{
  let ff = forevery_exists_2
    #(natlt (rows / trows)) #_ #(natlt (cols / tcols)) #_
    (fun tr tc (em : chest2 et trows tcols) ->
      array2_subtile m trows tcols tr tc |-> em **
      pure (em %~ ematrix_subtile r trows tcols tr tc));
  forevery_extract_pure_2
    #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc **
      pure (ff tr tc %~ ematrix_subtile r trows tcols tr tc))
    (fun tr tc ->
      ff tr tc %~ ematrix_subtile r trows tcols tr tc)
    fn tr tc { () };
  assert pure (forall (tr : natlt (rows / trows))
                      (tc : natlt (cols / tcols)).
    ff tr tc %~ ematrix_subtile r trows tcols tr tc);
  forevery_map_2
    #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc **
      pure (ff tr tc %~ ematrix_subtile r trows tcols tr tc))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc)
    fn tr tc { () };
  array2_untile' m trows tcols ff;
  assert pure (ematrix_from_tiles trows tcols ff %~ r);
}

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
= ()

let lane_fade_done
  (#et : Type0) {| scalar et |}
  (#rows #cols : pos)
  (em0 em1 : chest2 et rows cols)
  (lane : natlt warp_size)
  (upto : nat{rows * cols <= upto})
  : Lemma (lane_coincide lane (lane_fade em0 em1 lane upto) em1)
= ()


let tiled_cell
  (extent : pos)
  (tile : pos{tile /?+ extent})
  (ti : natlt (extent / tile))
  (i : natlt tile)
  : natlt extent
= ti * tile + i

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
    epilogue_chest comb
      (ematrix_subtile
        (ematrix_subtile
          (ematrix_subtile eC bm bn (SZ.v mrow) (SZ.v mcol))
          (wm * rows) (wn * cols) (SZ.v warpRow) (SZ.v warpCol))
        rows cols (SZ.v idx / wn) (SZ.v idx % wn))
      eAcc;
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
    let vf = !flat;
    let row : szlt rows = vf /^ cols;
    let col : szlt cols = vf %^ cols;
    assert pure (row < rows);
    assert pure (col < cols);
    let eij : erased (natlt rows & natlt cols) =
      Mktuple2 #(natlt rows) #(natlt cols) (SZ.v row) (SZ.v col);
    assert pure (in_lane rows cols lane eij);
    assert pure (SZ.v idx / SZ.v wn < SZ.v wm);
    assert pure (SZ.v idx % SZ.v wn < SZ.v wn);
    assert pure (
      SZ.v warpRow * (SZ.v wm * SZ.v rows)
        + (SZ.v idx / SZ.v wn) * SZ.v rows + SZ.v row
      < SZ.v bm);
    assert pure (
      SZ.v warpCol * (SZ.v wn * SZ.v cols)
        + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col
      < SZ.v bn);
    assert pure (
      SZ.v mrow * SZ.v bm
        + SZ.v warpRow * (SZ.v wm * SZ.v rows)
        + (SZ.v idx / SZ.v wn) * SZ.v rows + SZ.v row
      < SZ.v m);
    assert pure (
      SZ.v mcol * SZ.v bn
        + SZ.v warpCol * (SZ.v wn * SZ.v cols)
        + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col
      < SZ.v n);
    let globalRow : szlt m =
      mrow *^ bm +^ warpRow *^ (wm *^ rows)
        +^ (idx /^ wn) *^ rows +^ row;
    let globalCol : szlt n =
      mcol *^ bn +^ warpCol *^ (wn *^ cols)
        +^ (idx %^ wn) *^ cols +^ col;
    let warpFraction = precip warp_size;
    assert pure (warpFraction == 1.0R /. warp_size);
    assert pure (1.0R /. warp_size >. 0.0R);

    with em. unfold own_lane_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      em lane;
    forevery_remove'
      #(natlt rows & natlt cols)
      (fun ij -> in_lane rows cols lane ij)
      _ eij;

    let cv = tensor_read c (globalRow, (globalCol, ()));
    let av = tensor_read acc (row, (col, ()));
    let dv = comb cv av;

    assert tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (reveal eij)._1 (reveal eij)._2)
      (acc2 em (reveal eij)._1 (reveal eij)._2);
    output_fragment_cell_convert_eq d
      bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid)
      (SZ.v idx / wn) (SZ.v idx % wn)
      (reveal eij)._1 (reveal eij)._2
      1.0R (acc2 em (reveal eij)._1 (reveal eij)._2);
    rewrite
      tensor_pts_to_cell
        (output_fragment d bm bn rows cols wm wn
          (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
        (idx2 (reveal eij)._1 (reveal eij)._2)
        (acc2 em (reveal eij)._1 (reveal eij)._2)
    as
      tensor_pts_to_cell d
        (idx2 (SZ.v globalRow) (SZ.v globalCol))
        (acc2 em (reveal eij)._1 (reveal eij)._2);
    tensor_write_cell d
      (globalRow, (globalCol, ()))
      dv;

    let em' = upd2 em (reveal eij)._1 (reveal eij)._2 dv;
    output_fragment_cell_convert_eq d
      bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid)
      (SZ.v idx / wn) (SZ.v idx % wn)
      (reveal eij)._1 (reveal eij)._2
      1.0R dv;
    rewrite
      tensor_pts_to_cell d
        (idx2 (SZ.v globalRow) (SZ.v globalCol))
        dv
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
      _ eij;
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

    lane_fade_step eD0 eTarget lane !flat;
    own_lane_cells_rw
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      lane em'
      (lane_fade eD0 eTarget lane (!flat + warp_size));
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

#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
ghost
fn split_output_to_lanes
  (#et : Type0) {| scalar et |}
  (#m #n : szp)
  (gD : array2 et (rm m n))
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  requires live gD
  ensures
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_live gD bm bn tm tn wm wn bid tid
{
  with (eD : chest2 _ _ _). assert gD |-> eD;
  array2_tile gD (SZ.v bm) (SZ.v bn) #eD #1.0R;
  forevery_unfactor' nblk (m / bm) (n / bn) _;

  forevery_map
    (fun (bid : natlt nblk) ->
      array2_subtile gD (SZ.v bm) (SZ.v bn)
        (bid / (n / bn)) (bid % (n / bn))
        |-> ematrix_subtile eD bm bn
              (bid / (n / bn)) (bid % (n / bn)))
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        output_lane_live gD bm bn tm tn wm wn bid tid)
    fn bid {
      rewrite each
        array2_subtile gD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn))
      as block_tile gD (SZ.v bm) (SZ.v bn) bid;
      let dBlock = block_tile gD (SZ.v bm) (SZ.v bn) bid;
      let eBlock =
        ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn));
      array2_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
        (wm * tm) (wn * tn)
        #(ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn)))
        #1.0R;
      rewrite each
        block_tile gD (SZ.v bm) (SZ.v bn) bid
        as dBlock;
      rewrite each
        ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn))
        as eBlock;
      forevery_unfactor'
        (nthr / warp_size)
        (bm / (wm * tm))
        (bn / (wn * tn))
        _;

      forevery_map
        (fun (wid : natlt (nthr / warp_size)) ->
          array2_subtile dBlock
            (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn)))
          |-> ematrix_subtile eBlock (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn))))
        (fun (wid : natlt (nthr / warp_size)) ->
          forall+ (lane : natlt warp_size)
                   (mi : natlt wm)
                   (nj : natlt wn).
            live_lane_cells
              (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
              lane)
        fn wid {
          rewrite each
            array2_subtile dBlock
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))
          as warp_tile dBlock
            (wm * tm) (wn * tn) wid;
          let dWarp = warp_tile dBlock
            (wm * tm) (wn * tn) wid;
          let eWarp =
            ematrix_subtile eBlock
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)));
          array2_tile
            (warp_tile dBlock (wm * tm) (wn * tn) wid)
            (SZ.v tm) (SZ.v tn)
            #(ematrix_subtile eBlock (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn))))
            #1.0R;
          rewrite each
            warp_tile dBlock (wm * tm) (wn * tn) wid
            as dWarp;
          rewrite each
            ematrix_subtile eBlock (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))
            as eWarp;
          forevery_rw_size2
            ((wm * tm) / tm) wm
            ((wn * tn) / tn) wn
            #(fun (mi : natlt ((wm * tm) / tm))
                  (nj : natlt ((wn * tn) / tn)) ->
              array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                |-> ematrix_subtile eWarp tm tn mi nj);

          forevery_map_2
            (fun (mi : natlt wm) (nj : natlt wn) ->
              array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                |-> ematrix_subtile eWarp tm tn mi nj)
            (fun (mi : natlt wm) (nj : natlt wn) ->
              forall+ (lane : natlt warp_size).
                live_lane_cells
                  (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            fn mi nj {
              rewrite each
                array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                as output_fragment gD bm bn tm tn wm wn bid wid mi nj;
              split_array2_into_lane_cells
                (output_fragment gD bm bn tm tn wm wn bid wid mi nj);
              forevery_map
                (fun lane ->
                  own_lane_cells
                    (output_fragment gD bm bn tm tn wm wn
                      bid wid mi nj)
                    (ematrix_subtile eWarp tm tn mi nj)
                    lane)
                (fun lane ->
                  live_lane_cells
                    (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                    lane)
                fn lane {
                  fold live_lane_cells
                    (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                    lane;
                };
            };

          forevery_map
            (fun (mi : natlt wm) ->
              forall+ (nj : natlt wn) (lane : natlt warp_size).
                live_lane_cells
                  (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            (fun (mi : natlt wm) ->
              forall+ (lane : natlt warp_size) (nj : natlt wn).
                live_lane_cells
                  (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            fn mi { forevery_commute _ };
          forevery_commute
            (fun (mi : natlt wm) (lane : natlt warp_size) ->
              forall+ (nj : natlt wn).
                live_lane_cells
                  (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                  lane);
        };

      forevery_unfactor' nthr (nthr / warp_size) warp_size _;
      forevery_map
        (fun (tid : natlt nthr) ->
          forall+ (mi : natlt wm) (nj : natlt wn).
            live_lane_cells
              (output_fragment gD bm bn tm tn wm wn
                bid (tid / warp_size) mi nj)
              (tid % warp_size))
        (fun tid ->
          output_lane_live gD bm bn tm tn wm wn bid tid)
        fn tid {
          fold output_lane_live gD bm bn tm tn wm wn bid tid;
        };
    };
}
#pop-options

ghost
fn setup_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, real_like et_cd |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    gA |-> Frac fA eA ** pure (eA %~ rA) **
    gB |-> Frac fB eB ** pure (eB %~ rB) **
    gC |-> Frac fC eC ** pure (eC %~ rC) **
    live gD
  ensures
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid) **
    pure (SZ.fits ((rm m n).ulen))
{
  tensor_share_n gA (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_share_n gB (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_share_n gC (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  split_output_to_lanes gD bm bn tm tn wm wn nblk nthr;

  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC)
    (fun bid tid ->
      output_lane_live gD bm bn tm tn wm wn bid tid);
  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_live gD bm bn tm tn wm wn bid tid);
  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    _;
  forevery_rw_size2
    (SZ.v nblk) (m / bm * (n / bn))
    (SZ.v nthr)
      (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
    #(fun (bid : natlt nblk) (tid : natlt nthr) ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_live gD bm bn tm tn wm wn bid tid);
  forevery_map_2
    #(natlt (m / bm * (n / bn)))
    #(natlt (
      bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_live gD bm bn tm tn wm wn bid tid)
    (fun bid tid ->
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    fn bid tid {
      fold kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid;
    };
  forevery_rw_size2
    (m / bm * (n / bn)) (SZ.v nblk)
    (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
      (SZ.v nthr)
    #(fun bid tid ->
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid);
}

ghost
fn split_scratch_to_threads
  (#et_ab #et_acc : Type0)
  {| sized et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  requires live_c_shmem (fst (snd (snd sh)))
  ensures forall+ (tid : natlt nthr).
    scratch_tile_live bm bn bk tm tn nthr sh tid
{
  let (_, (_, (sarAcc, _))) = sh;
  unfold_live_c_shmem sarAcc;
  with vAcc. assert sarAcc |-> vAcc;
  gpu_pts_to_ref sarAcc;
  tensor_abs' (scratch_layout tm tn nthr) sarAcc;
  let sAcc = scratch_matrix bm bn bk tm tn nthr sh;
  rewrite each from_array (scratch_layout tm tn nthr) sarAcc as sAcc;
  array2_tile sAcc (SZ.v tm) (SZ.v tn);
  forevery_rw_size2
    (((nthr / warp_size) * tm) / tm) (nthr / warp_size)
    (tn / tn) 1
    #(fun
        (wid : natlt (((nthr / warp_size) * tm) / tm))
        (tc : natlt (tn / tn)) ->
      array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc
        |-> ematrix_subtile
              (from_seq (scratch_layout tm tn nthr) vAcc)
              tm tn wid tc);
  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      forall+ (tc : natlt 1).
        array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc
          |-> ematrix_subtile
                (from_seq (scratch_layout tm tn nthr) vAcc)
                tm tn wid tc)
    (fun (wid : natlt (nthr / warp_size)) ->
      array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0
        |-> ematrix_subtile
              (from_seq (scratch_layout tm tn nthr) vAcc)
              tm tn wid 0)
    fn wid {
      forevery_singleton_elim #(natlt 1)
        (fun (tc : natlt 1) ->
          array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc
            |-> ematrix_subtile
                  (from_seq (scratch_layout tm tn nthr) vAcc)
                  tm tn wid tc);
    };

  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0
        |-> ematrix_subtile
              (from_seq (scratch_layout tm tn nthr) vAcc)
              tm tn wid 0)
    (fun (wid : natlt (nthr / warp_size)) ->
      forall+ (lane : natlt warp_size).
        scratch_tile_live bm bn bk tm tn nthr sh
          (wid * warp_size + lane))
    fn wid {
      rewrite each array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0
        as scratch_tile bm bn bk tm tn nthr sh wid;
      tensor_share_n
        (scratch_tile bm bn bk tm tn nthr sh wid)
        warp_size;
      forevery_map
        #(natlt warp_size)
        (fun lane ->
          scratch_tile bm bn bk tm tn nthr sh wid
            |-> Frac (1.0R /. warp_size)
              (ematrix_subtile
                (from_seq (scratch_layout tm tn nthr) vAcc)
                tm tn wid 0))
        (fun lane ->
          scratch_tile_live bm bn bk tm tn nthr sh
            (wid * warp_size + lane))
        fn lane {
          assert pure (
            (wid * warp_size + lane) / warp_size == wid);
          rewrite each scratch_tile bm bn bk tm tn nthr sh wid
            as scratch_tile bm bn bk tm tn nthr sh
              ((wid * warp_size + lane) / warp_size);
          fold scratch_tile_live bm bn bk tm tn nthr sh
            (wid * warp_size + lane);
        };
    };
  forevery_unfactor' nthr (nthr / warp_size) warp_size _;
  forevery_map
    #(natlt nthr)
    (fun tid ->
      scratch_tile_live bm bn bk tm tn nthr sh
        ((tid / warp_size) * warp_size + tid % warp_size))
    (fun tid -> scratch_tile_live bm bn bk tm tn nthr sh tid)
    fn tid {
      unfold scratch_tile_live bm bn bk tm tn nthr sh
        ((tid / warp_size) * warp_size + tid % warp_size);
      assert pure (
        (tid / warp_size) * warp_size + tid % warp_size == tid);
      rewrite each scratch_tile bm bn bk tm tn nthr sh
        (((tid / warp_size) * warp_size + tid % warp_size) / warp_size)
        as scratch_tile bm bn bk tm tn nthr sh (tid / warp_size);
      fold scratch_tile_live bm bn bk tm tn nthr sh tid;
    };
}

ghost
fn gather_scratch_from_threads
  (#et_ab #et_acc : Type0)
  {| sized et_ab, scalar et_acc |}
  (bm bn bk tm tn nthr : szp)
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (warp_size /?+ nthr))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  requires forall+ (tid : natlt nthr).
    scratch_tile_live bm bn bk tm tn nthr sh tid
  ensures live_c_shmem (fst (snd (snd sh)))
{
  forevery_map
    (fun tid -> scratch_tile_live bm bn bk tm tn nthr sh tid)
    (fun tid ->
      exists* (eAcc : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh (tid / warp_size)
          |-> Frac (1.0R /. warp_size) eAcc)
    fn tid {
      unfold scratch_tile_live bm bn bk tm tn nthr sh tid;
    };
  forevery_factor' nthr (nthr / warp_size) warp_size
    (fun wid _lane ->
      exists* (eAcc : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh wid
          |-> Frac (1.0R /. warp_size) eAcc);
  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      forall+ (_lane : natlt warp_size).
        exists* (eAcc : chest2 et_acc tm tn).
          scratch_tile bm bn bk tm tn nthr sh wid
            |-> Frac (1.0R /. warp_size) eAcc)
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eAcc : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh wid |-> eAcc)
    fn wid {
      tensor_gather_n_underspec
        (scratch_tile bm bn bk tm tn nthr sh wid)
        warp_size;
    };

  let sarAcc = fst (snd (snd sh));
  let sAcc = scratch_matrix bm bn bk tm tn nthr sh;
  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eAcc : chest2 et_acc tm tn).
        scratch_tile bm bn bk tm tn nthr sh wid |-> eAcc)
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eAcc : chest2 et_acc tm tn).
        array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0 |-> eAcc)
    fn wid {
      rewrite each scratch_tile bm bn bk tm tn nthr sh wid
        as array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0;
    };
  forevery_map
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eAcc : chest2 et_acc tm tn).
        array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid 0 |-> eAcc)
    (fun (wid : natlt (nthr / warp_size)) ->
      forall+ (tc : natlt 1).
        exists* (eAcc : chest2 et_acc tm tn).
          array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc |-> eAcc)
    fn wid {
      forevery_singleton_intro #(natlt 1)
        (fun tc ->
          exists* (eAcc : chest2 et_acc tm tn).
            array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc |-> eAcc);
    };
  forevery_rw_size2
    (nthr / warp_size) (((nthr / warp_size) * tm) / tm)
    1 (tn / tn)
    #(fun (wid : natlt (nthr / warp_size)) (tc : natlt 1) ->
      exists* (eAcc : chest2 et_acc tm tn).
        array2_subtile sAcc (SZ.v tm) (SZ.v tn) wid tc |-> eAcc);
  array2_untile_underspec sAcc (SZ.v tm) (SZ.v tn);
  with eAcc. assert sAcc |-> eAcc;
  tensor_concr sAcc;
  rewrite each core sAcc as sarAcc;
  with vAcc. assert sarAcc |-> vAcc;
  fold_live_c_shmem sarAcc;
  rewrite each sarAcc as fst (snd (snd sh));
}

ghost
fn block_setup_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, real_like et_ab,
     scalar et_cd, real_like et_cd, scalar et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (
    ((bm / (wm * tm) * (bn / (wn * tn)) * warp_size) / warp_size)
      * tm * tn)))
  (#_ : squash (
    warp_size /?+ (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
  ensures
    (forall+ (tid : natlt nthr).
      kpre_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid) **
    emp
{
  unfold_live_c_shmems_cons sh #1.0R;
  unfold_live_c_shmems_cons (snd sh) #1.0R;
  unfold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  unfold_live_c_shmems_nil (snd (snd (snd sh))) #1.0R;

  gpu_live_c_shmem_share_underspec (fst sh) #1.0R #nthr;
  gpu_live_c_shmem_share_underspec (fst (snd sh)) #1.0R #nthr;
  split_scratch_to_threads bm bn bk tm tn nthr sh;

  forevery_zip #(natlt nthr)
    (fun tid ->
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr))
    (fun tid -> scratch_tile_live bm bn bk tm tn nthr sh tid);
  forevery_zip #(natlt nthr)
    (fun tid -> live_c_shmem (fst sh) #(1.0R /. nthr))
    (fun tid ->
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
      scratch_tile_live bm bn bk tm tn nthr sh tid);
  forevery_map
    (fun tid ->
      live_c_shmem (fst sh) #(1.0R /. nthr) **
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
      scratch_tile_live bm bn bk tm tn nthr sh tid)
    (fun tid -> shared_thread_live bm bn bk tm tn nthr sh tid)
    fn tid {
      fold shared_thread_live bm bn bk tm tn nthr sh tid;
    };
  forevery_zip #(natlt nthr)
    (fun tid ->
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    (fun tid -> shared_thread_live bm bn bk tm tn nthr sh tid);
  forevery_map
    (fun tid ->
      kpre1_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid **
      shared_thread_live bm bn bk tm tn nthr sh tid)
    (fun tid ->
      kpre_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid)
    fn tid {
      fold kpre_to gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid;
    };
}

ghost
fn block_teardown_to
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd, scalar et_acc |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (SZ.fits (
    ((bm / (wm * tm) * (bn / (wn * tn)) * warp_size) / warp_size)
      * tm * tn)))
  (#_ : squash (
    warp_size /?+ (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)))
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : natlt nblk)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid) **
    emp
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt nthr).
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
{
  forevery_map
    (fun tid ->
      kpost_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid)
    (fun tid ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid **
      shared_thread_live bm bn bk tm tn nthr sh tid)
    fn tid {
      unfold kpost_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr sh bid tid;
    };
  forevery_unzip #(natlt nthr)
    (fun tid ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    (fun tid -> shared_thread_live bm bn bk tm tn nthr sh tid);

  forevery_map
    (fun tid -> shared_thread_live bm bn bk tm tn nthr sh tid)
    (fun tid ->
      live_c_shmem (fst sh) #(1.0R /. nthr) **
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
      scratch_tile_live bm bn bk tm tn nthr sh tid)
    fn tid {
      unfold shared_thread_live bm bn bk tm tn nthr sh tid;
    };
  forevery_unzip #(natlt nthr)
    (fun _ -> live_c_shmem (fst sh) #(1.0R /. nthr))
    (fun tid ->
      live_c_shmem (fst (snd sh)) #(1.0R /. nthr) **
      scratch_tile_live bm bn bk tm tn nthr sh tid);
  forevery_unzip #(natlt nthr)
    (fun _ -> live_c_shmem (fst (snd sh)) #(1.0R /. nthr))
    (fun tid -> scratch_tile_live bm bn bk tm tn nthr sh tid);

  gpu_live_c_shmem_gather_underspec (fst sh) #1.0R #nthr;
  gpu_live_c_shmem_gather_underspec (fst (snd sh)) #1.0R #nthr;
  gather_scratch_from_threads bm bn bk tm tn nthr sh;

  fold_live_c_shmems_nil (snd (snd (snd sh))) #1.0R;
  fold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  fold_live_c_shmems_cons (snd sh) #1.0R;
  fold_live_c_shmems_cons sh #1.0R;
}

#push-options "--z3rlimit 120 --fuel 1 --ifuel 1"
ghost
fn teardown_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid) **
    pure (SZ.fits ((rm m n).ulen))
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> Frac fC eC **
    (exists* (eD : chest2 et_cd m n).
      gD |-> eD ** pure (eD %~ MS.mmcomb comb_r rC rA rB))
{
  forevery_rw_size2
    (SZ.v nblk) (m / bm * (n / bn))
    (SZ.v nthr)
      (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
    #(fun (bid : natlt nblk) (tid : natlt nthr) ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid);
  forevery_map_2
    #(natlt (m / bm * (n / bn)))
    #(natlt (
      bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
    (fun bid tid ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))))
    fn bid tid {
      unfold kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid;
    };
  forevery_unzip_2
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));
  forevery_unzip_2
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));
  forevery_unzip_2
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC)
    (fun bid tid ->
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));

  forevery_rw_size2
    (m / bm * (n / bn)) (SZ.v nblk)
    (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
      (SZ.v nthr)
    #(fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  forevery_rw_size2
    (m / bm * (n / bn)) (SZ.v nblk)
    (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
      (SZ.v nthr)
    #(fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  forevery_rw_size2
    (m / bm * (n / bn)) (SZ.v nblk)
    (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
      (SZ.v nthr)
    #(fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  forevery_rw_size2
    (m / bm * (n / bn)) (SZ.v nblk)
    (bm / (wm * tm) * (bn / (wn * tn)) * warp_size)
      (SZ.v nthr)
    #(fun bid tid ->
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn)))));

  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_gather_n gA (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_gather_n gB (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  tensor_gather_n gC (nblk * nthr);

  forevery_map
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        output_lane_approximates
          gD bm bn tm tn wm wn bid tid
          (ematrix_subtile
            (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
              bm bn (bid / (n / bn)) (bid % (n / bn)))
            (wm * tm) (wn * tn)
            ((tid / warp_size) / (bn / (wn * tn)))
            ((tid / warp_size) % (bn / (wn * tn)))))
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    fn bid {
      forevery_ext
        (fun (tid : natlt nthr) ->
          output_lane_approximates
            gD bm bn tm tn wm wn bid tid
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              ((tid / warp_size) / (bn / (wn * tn)))
              ((tid / warp_size) % (bn / (wn * tn)))))
        (fun (tid : natlt nthr) ->
          output_lane_approximates
            gD bm bn tm tn wm wn bid
            ((tid / warp_size) * warp_size + tid % warp_size)
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              ((tid / warp_size) / (bn / (wn * tn)))
              ((tid / warp_size) % (bn / (wn * tn)))));
      forevery_factor' nthr (nthr / warp_size) warp_size
        (fun wid lane ->
          output_lane_approximates
            gD bm bn tm tn wm wn bid
            (wid * warp_size + lane)
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))));
      forevery_map
        (fun (wid : natlt (nthr / warp_size)) ->
          forall+ (lane : natlt warp_size).
            output_lane_approximates
              gD bm bn tm tn wm wn bid (wid * warp_size + lane)
              (ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn)))))
        (fun (wid : natlt (nthr / warp_size)) ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn) wid |-> eWarp **
            pure (eWarp %~
              ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn)))))
        fn wid {
          forevery_map
            (fun (lane : natlt warp_size) ->
              output_lane_approximates
                gD bm bn tm tn wm wn bid
                (wid * warp_size + lane)
                (ematrix_subtile
                  (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                    bm bn (bid / (n / bn)) (bid % (n / bn)))
                  (wm * tm) (wn * tn)
                  (wid / (bn / (wn * tn)))
                  (wid % (bn / (wn * tn)))))
            (fun (lane : natlt warp_size) ->
              forall+ (mi : natlt wm) (nj : natlt wn).
                exists* (eFrag : chest2 et_cd tm tn).
                  own_lane_cells
                    (output_fragment gD bm bn tm tn wm wn
                      bid wid mi nj)
                    eFrag lane **
                  pure (eFrag %~
                    ematrix_subtile
                      (ematrix_subtile
                        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                          bm bn (bid / (n / bn)) (bid % (n / bn)))
                        (wm * tm) (wn * tn)
                        (wid / (bn / (wn * tn)))
                        (wid % (bn / (wn * tn))))
                      tm tn mi nj))
            fn lane {
              assert pure (
                (wid * warp_size + lane) / warp_size == wid);
              assert pure (
                (wid * warp_size + lane) % warp_size == lane);
              unfold output_lane_approximates
                gD bm bn tm tn wm wn bid
                (wid * warp_size + lane)
                (ematrix_subtile
                  (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                    bm bn (bid / (n / bn)) (bid % (n / bn)))
                  (wm * tm) (wn * tn)
                  (wid / (bn / (wn * tn)))
                  (wid % (bn / (wn * tn))));
              rewrite each
                ((wid * warp_size + lane) / warp_size)
              as wid;
              rewrite each
                ((wid * warp_size + lane) % warp_size)
              as lane;
              forevery_ext_2
                (fun (mi : natlt wm) (nj : natlt wn) ->
                  exists* (eFrag : chest2 et_cd tm tn).
                    own_lane_cells
                      (output_fragment gD bm bn tm tn wm wn
                        bid wid mi nj)
                      eFrag
                      lane **
                    pure (eFrag %~
                      ematrix_subtile
                        (ematrix_subtile
                          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                            bm bn (bid / (n / bn)) (bid % (n / bn)))
                          (wm * tm) (wn * tn)
                          (wid / (bn / (wn * tn)))
                          (wid % (bn / (wn * tn))))
                        tm tn mi nj))
                (fun (mi : natlt wm) (nj : natlt wn) ->
                  exists* (eFrag : chest2 et_cd tm tn).
                    own_lane_cells
                      (output_fragment gD bm bn tm tn wm wn
                        bid wid mi nj)
                      eFrag lane **
                    pure (eFrag %~
                      ematrix_subtile
                        (ematrix_subtile
                          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                            bm bn (bid / (n / bn)) (bid % (n / bn)))
                          (wm * tm) (wn * tn)
                          (wid / (bn / (wn * tn)))
                          (wid % (bn / (wn * tn))))
                        tm tn mi nj));
            };

          forevery_commute
            (fun (lane : natlt warp_size) (mi : natlt wm) ->
              forall+ (nj : natlt wn).
                exists* (eFrag : chest2 et_cd tm tn).
                  own_lane_cells
                    (output_fragment gD bm bn tm tn wm wn
                      bid wid mi nj)
                    eFrag lane **
                  pure (eFrag %~
                    ematrix_subtile
                      (ematrix_subtile
                        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                          bm bn (bid / (n / bn)) (bid % (n / bn)))
                        (wm * tm) (wn * tn)
                        (wid / (bn / (wn * tn)))
                        (wid % (bn / (wn * tn))))
                      tm tn mi nj));
          forevery_map
            (fun (mi : natlt wm) ->
              forall+ (lane : natlt warp_size) (nj : natlt wn).
                exists* (eFrag : chest2 et_cd tm tn).
                  own_lane_cells
                    (output_fragment gD bm bn tm tn wm wn
                      bid wid mi nj)
                    eFrag lane **
                  pure (eFrag %~
                    ematrix_subtile
                      (ematrix_subtile
                        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                          bm bn (bid / (n / bn)) (bid % (n / bn)))
                        (wm * tm) (wn * tn)
                        (wid / (bn / (wn * tn)))
                        (wid % (bn / (wn * tn))))
                      tm tn mi nj))
            (fun (mi : natlt wm) ->
              forall+ (nj : natlt wn) (lane : natlt warp_size).
                exists* (eFrag : chest2 et_cd tm tn).
                  own_lane_cells
                    (output_fragment gD bm bn tm tn wm wn
                      bid wid mi nj)
                    eFrag lane **
                  pure (eFrag %~
                    ematrix_subtile
                      (ematrix_subtile
                        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                          bm bn (bid / (n / bn)) (bid % (n / bn)))
                        (wm * tm) (wn * tn)
                        (wid / (bn / (wn * tn)))
                        (wid % (bn / (wn * tn))))
                      tm tn mi nj))
            fn mi { forevery_commute _ };

          forevery_map_2
            (fun mi nj ->
              forall+ (lane : natlt warp_size).
                exists* (eFrag : chest2 et_cd tm tn).
                  own_lane_cells
                    (output_fragment gD bm bn tm tn wm wn
                      bid wid mi nj)
                    eFrag lane **
                  pure (eFrag %~
                    ematrix_subtile
                      (ematrix_subtile
                        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                          bm bn (bid / (n / bn)) (bid % (n / bn)))
                        (wm * tm) (wn * tn)
                        (wid / (bn / (wn * tn)))
                        (wid % (bn / (wn * tn))))
                      tm tn mi nj))
            (fun mi nj ->
              exists* (eFrag : chest2 et_cd tm tn).
                output_fragment gD bm bn tm tn wm wn bid wid mi nj
                  |-> eFrag **
                pure (eFrag %~
                  ematrix_subtile
                    (ematrix_subtile
                      (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                        bm bn (bid / (n / bn)) (bid % (n / bn)))
                      (wm * tm) (wn * tn)
                      (wid / (bn / (wn * tn)))
                      (wid % (bn / (wn * tn))))
                    tm tn mi nj))
            fn mi nj {
              join_lane_cells_approximates
                (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
                (ematrix_subtile
                  (ematrix_subtile
                    (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                      bm bn (bid / (n / bn)) (bid % (n / bn)))
                    (wm * tm) (wn * tn)
                    (wid / (bn / (wn * tn)))
                    (wid % (bn / (wn * tn))))
                  tm tn mi nj);
            };

          let dWarp =
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn) wid;
          forevery_map_2
            (fun mi nj ->
              exists* (eFrag : chest2 et_cd tm tn).
                output_fragment gD bm bn tm tn wm wn bid wid mi nj
                  |-> eFrag **
                pure (eFrag %~
                  ematrix_subtile
                    (ematrix_subtile
                      (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                        bm bn (bid / (n / bn)) (bid % (n / bn)))
                      (wm * tm) (wn * tn)
                      (wid / (bn / (wn * tn)))
                      (wid % (bn / (wn * tn))))
                    tm tn mi nj))
            (fun mi nj ->
              exists* (eFrag : chest2 et_cd tm tn).
                array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag **
                pure (eFrag %~
                  ematrix_subtile
                    (ematrix_subtile
                      (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                        bm bn (bid / (n / bn)) (bid % (n / bn)))
                      (wm * tm) (wn * tn)
                      (wid / (bn / (wn * tn)))
                      (wid % (bn / (wn * tn))))
                    tm tn mi nj))
            fn mi nj {
              rewrite each
                output_fragment gD bm bn tm tn wm wn bid wid mi nj
              as array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj;
            };
          forevery_rw_size2
            wm ((wm * tm) / tm)
            wn ((wn * tn) / tn)
            #(fun (mi : natlt wm) (nj : natlt wn) ->
              exists* (eFrag : chest2 et_cd tm tn).
                array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                  |-> eFrag **
                pure (eFrag %~
                  ematrix_subtile
                    (ematrix_subtile
                      (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                        bm bn (bid / (n / bn)) (bid % (n / bn)))
                      (wm * tm) (wn * tn)
                      (wid / (bn / (wn * tn)))
                      (wid % (bn / (wn * tn))))
                    tm tn mi nj));
          array2_untile_approximates dWarp (SZ.v tm) (SZ.v tn)
            (ematrix_subtile
              (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                bm bn (bid / (n / bn)) (bid % (n / bn)))
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn))));
          rewrite each dWarp as
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn) wid;
        };

      forevery_map
        #(natlt (nthr / warp_size))
        (fun (wid : natlt (nthr / warp_size)) ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn) wid
              |-> eWarp **
            pure (eWarp %~
              ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn)))))
        (fun (wid : natlt (nthr / warp_size)) ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn)
              ((wid / (bn / (wn * tn))) * (bn / (wn * tn)) +
                wid % (bn / (wn * tn)))
              |-> eWarp **
            pure (eWarp %~
              ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn)))))
        fn wid {
          FStar.Math.Lemmas.euclidean_division_definition
            wid (bn / (wn * tn));
          rewrite each wid as
            ((wid / (bn / (wn * tn))) * (bn / (wn * tn)) +
              wid % (bn / (wn * tn)));
        };
      forevery_factor'
        (nthr / warp_size)
        (bm / (wm * tm))
        (bn / (wn * tn))
        (fun wr wc ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn)
              (wr * (bn / (wn * tn)) + wc)
              |-> eWarp **
            pure (eWarp %~
              ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn) wr wc));
      let dBlock = block_tile gD (SZ.v bm) (SZ.v bn) bid;
      rewrite each block_tile gD (SZ.v bm) (SZ.v bn) bid
        as dBlock;
      forevery_map_2
        #(natlt (bm / (wm * tm)))
        #(natlt (bn / (wn * tn)))
        (fun wr wc ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            warp_tile dBlock (wm * tm) (wn * tn)
              (wr * (bn / (wn * tn)) + wc)
              |-> eWarp **
            pure (eWarp %~
              ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn) wr wc))
        (fun wr wc ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            array2_subtile dBlock (wm * tm) (wn * tn) wr wc
              |-> eWarp **
            pure (eWarp %~
              ematrix_subtile
                (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
                  bm bn (bid / (n / bn)) (bid % (n / bn)))
                (wm * tm) (wn * tn) wr wc))
        fn wr wc {
          assert pure (
            (wr * (bn / (wn * tn)) + wc) / (bn / (wn * tn)) == wr);
          assert pure (
            (wr * (bn / (wn * tn)) + wc) % (bn / (wn * tn)) == wc);
          rewrite each
            warp_tile dBlock (wm * tm) (wn * tn)
              (wr * (bn / (wn * tn)) + wc)
          as
            array2_subtile dBlock (wm * tm) (wn * tn) wr wc;
        };
      array2_untile_approximates dBlock
        (wm * tm) (wn * tn)
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn (bid / (n / bn)) (bid % (n / bn)));
      rewrite each dBlock
        as block_tile gD (SZ.v bm) (SZ.v bn) bid;
    };

  forevery_map
    #(natlt nblk)
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          ((bid / (n / bn)) * (n / bn) + bid % (n / bn))
          |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn))))
    fn bid {
      FStar.Math.Lemmas.euclidean_division_definition bid (n / bn);
      rewrite each bid as
        ((bid / (n / bn)) * (n / bn) + bid % (n / bn));
    };
  forevery_factor' nblk (m / bm) (n / bn)
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc));
  forevery_map_2
    #(natlt (m / bm)) #(natlt (n / bn))
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc))
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        array2_subtile gD (SZ.v bm) (SZ.v bn) br bc |-> eBlock **
        pure (eBlock %~
          ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc))
    fn br bc {
      rewrite each block_tile gD (SZ.v bm) (SZ.v bn)
        (br * (n / bn) + bc)
        as array2_subtile gD (SZ.v bm) (SZ.v bn) br bc;
    };
  array2_untile_approximates gD (SZ.v bm) (SZ.v bn)
    (MS.mmcomb comb_r rC rA rB);
}
#pop-options
