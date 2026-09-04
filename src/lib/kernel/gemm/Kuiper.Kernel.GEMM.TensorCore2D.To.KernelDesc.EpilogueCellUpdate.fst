module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCellUpdate

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 60"

open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell

let epilogue_fragment_target
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#m #n : nat)
  (eC : chest2 et_cd m n)
  (bm bn rows cols wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm / (wm * rows)))
  (warpCol : natlt (bn / (wn * cols)))
  (idx : natlt (wm * wn))
  (eAcc : chest2 et_acc rows cols)
  : chest2 et_cd rows cols
= epilogue_chest comb
    (ematrix_subtile
      (ematrix_subtile
        (ematrix_subtile eC bm bn mrow mcol)
        (wm * rows) (wn * cols) warpRow warpCol)
      rows cols (idx / wn) (idx % wn))
    eAcc

let epilogue_fragment_target_eq
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#m #n : nat)
  (eC : chest2 et_cd m n)
  (bm bn rows cols wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * rows /?+ bm /\ wn * cols /?+ bn))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm / (wm * rows)))
  (warpCol : natlt (bn / (wn * cols)))
  (idx : natlt (wm * wn))
  (eAcc : chest2 et_acc rows cols)
  : Lemma (
      epilogue_fragment_target comb eC
        bm bn rows cols wm wn
        mrow mcol warpRow warpCol idx eAcc
      ==
      epilogue_chest comb
        (ematrix_subtile
          (ematrix_subtile
            (ematrix_subtile eC bm bn mrow mcol)
            (wm * rows) (wn * cols) warpRow warpCol)
          rows cols (idx / wn) (idx % wn))
        eAcc)
= ()

let epilogue_chest_acc2
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#rows #cols : nat)
  (eC : chest2 et_cd rows cols)
  (eAcc : chest2 et_acc rows cols)
  (row : natlt rows)
  (col : natlt cols)
  : Lemma (
      acc2 (epilogue_chest comb eC eAcc) row col
      == comb (acc2 eC row col) (acc2 eAcc row col))
= Kuiper.EMatrix.macc_mkM
    (fun i j -> comb (acc2 eC i j) (acc2 eAcc i j)) row col

(* [acc2] of the epilogue target tile at a warp-local cell [(row, col)] is the
   combine of the corresponding *global* C cell with the accumulator cell.
   Peeling [epilogue_chest] plus the three nested [ematrix_subtile]s is a long
   delta chain that, with one SMT query per proof obligation, no longer goes
   through inside [epilogue_cell_update]'s Pulse proof state, so discharge it
   here in a minimal context. *)
#push-options "--z3rlimit 20"
let epilogue_fragment_target_cell
  (#et_cd #et_acc : Type0)
  {| scalar et_cd, scalar et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (#m #n : nat)
  (eC : chest2 et_cd m n)
  (bm bn rows cols wm wn : pos)
  (#pf : squash (bm /?+ m /\ bn /?+ n /\
                 wm * rows /?+ bm /\ wn * cols /?+ bn))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm / (wm * rows)))
  (warpCol : natlt (bn / (wn * cols)))
  (idx : natlt (wm * wn))
  (eAcc : chest2 et_acc rows cols)
  (row : natlt rows)
  (col : natlt cols)
  (globalRow : natlt m)
  (globalCol : natlt n)
  (_ : squash (
     globalRow == mrow * bm + warpRow * (wm * rows) + (idx / wn) * rows + row /\
     globalCol == mcol * bn + warpCol * (wn * cols) + (idx % wn) * cols + col))
  : Lemma (
      acc2
        (epilogue_fragment_target comb eC bm bn rows cols wm wn
           mrow mcol warpRow warpCol idx eAcc)
        row col
      == comb (acc2 eC globalRow globalCol) (acc2 eAcc row col))
= epilogue_fragment_target_eq comb eC bm bn rows cols wm wn
    mrow mcol warpRow warpCol idx eAcc;
  let fragRow = tiled_cell (wm * rows) rows (idx / wn) row in
  let fragCol = tiled_cell (wn * cols) cols (idx % wn) col in
  let warpCellRow = tiled_cell bm (wm * rows) warpRow fragRow in
  let warpCellCol = tiled_cell bn (wn * cols) warpCol fragCol in
  let globalRow' = tiled_cell m bm mrow warpCellRow in
  let globalCol' = tiled_cell n bn mcol warpCellCol in
  assert (globalRow == globalRow');
  assert (globalCol == globalCol');
  epilogue_chest_acc2 comb
    (ematrix_subtile
      (ematrix_subtile
        (ematrix_subtile eC bm bn mrow mcol)
        (wm * rows) (wn * cols) warpRow warpCol)
      rows cols (idx / wn) (idx % wn))
    eAcc row col;
  FStar.Math.Lemmas.nat_over_pos_is_nat idx wn;
  subtile_acc2
    (ematrix_subtile
      (ematrix_subtile eC bm bn mrow mcol)
      (wm * rows) (wn * cols) warpRow warpCol)
    rows cols (idx / wn) (idx % wn) row col;
  subtile_acc2
    (ematrix_subtile eC bm bn mrow mcol)
    (wm * rows) (wn * cols) warpRow warpCol
    fragRow fragCol;
  subtile_acc2 eC bm bn mrow mcol
    warpCellRow warpCellCol;
  ()
#pop-options

inline_for_extraction noextract
fn epilogue_cell_update
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
  (row : szlt rows)
  (col : szlt cols)
  (#old : erased et_cd)
  preserves
    gpu **
    c |-> Frac fC eC **
    acc |-> Frac (1.0R /. warp_size) eAcc
  requires
    tensor_pts_to_cell
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      (idx2 (SZ.v row) (SZ.v col))
      (reveal old)
  ensures
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
{
  assert pure (SZ.v idx / SZ.v wn < SZ.v wm);
  assert pure (SZ.v idx % SZ.v wn < SZ.v wn);
  assert pure (
    (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col
    < SZ.v wn * SZ.v cols);
  assert pure (
    SZ.v warpCol + 1
    <= SZ.v bn / (SZ.v wn * SZ.v cols));
  FStar.Math.Lemmas.div_exact_r
    (SZ.v bn) (SZ.v wn * SZ.v cols);
  assert pure (
    (SZ.v bn / (SZ.v wn * SZ.v cols))
      * (SZ.v wn * SZ.v cols)
    == SZ.v bn);
  assert pure (
    SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col
    < (SZ.v warpCol + 1) * (SZ.v wn * SZ.v cols));
  assert pure (
    (SZ.v warpCol + 1) * (SZ.v wn * SZ.v cols)
    <= SZ.v bn);
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

  let cv = tensor_read c (globalRow, (globalCol, ()));
  let av = tensor_read acc (row, (col, ()));
  let dv = comb cv av;

  output_cell_to_global d bm bn rows cols wm wn
    (SZ.v bid) (SZ.v wid)
    (SZ.v idx / wn) (SZ.v idx % wn)
    (SZ.v row) (SZ.v col)
    (SZ.v globalRow) (SZ.v globalCol)
    1.0R (reveal old);
  tensor_write_cell d
    (globalRow, (globalCol, ()))
    dv;
  global_cell_to_output d bm bn rows cols wm wn
    (SZ.v bid) (SZ.v wid)
    (SZ.v idx / wn) (SZ.v idx % wn)
    (SZ.v row) (SZ.v col)
    (SZ.v globalRow) (SZ.v globalCol)
    1.0R dv;
  epilogue_fragment_target_cell comb eC bm bn rows cols wm wn
    (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol) (SZ.v idx) eAcc
    (SZ.v row) (SZ.v col) (SZ.v globalRow) (SZ.v globalCol) ();
  rewrite each dv as
    acc2
      (epilogue_fragment_target comb eC
        bm bn rows cols wm wn
        (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
        (SZ.v idx) eAcc)
      (SZ.v row) (SZ.v col);
}
