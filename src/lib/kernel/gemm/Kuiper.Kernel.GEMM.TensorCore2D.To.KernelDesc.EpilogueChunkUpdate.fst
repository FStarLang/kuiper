module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueChunkUpdate

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 40 --split_queries no"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Array2.Vectorized
open Kuiper.Array2.Strided
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module ML = FStar.Math.Lemmas

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCellUpdate

(* Pointwise: cell [(i, j+x)] of the fragment is cell
   [(globalRow, globalCol+x)] of the global matrix. *)
let frag_global_cell_eq
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
  (j : nat)
  (w : nat { j + w <= tn })
  (globalRow : natlt m {
    globalRow == tiled_cell m bm (bid / (n / bn))
      (tiled_cell bm (wm * tm) (wid / (bn / (wn * tn)))
        (tiled_cell (wm * tm) tm mi i)) })
  (globalCol : nat { globalCol + w <= n /\
    globalCol == bid % (n / bn) * bn
      + wid % (bn / (wn * tn)) * (wn * tn) + nj * tn + j })
  (f : perm)
  (v : seq et { Seq.length v == w })
  (x : natlt w)
  : Lemma (
      T.tensor_pts_to_cell
        (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
        #f (idx2 i (j + x)) (Seq.index v x)
      == T.tensor_pts_to_cell gD #f
           (idx2 globalRow (globalCol + x)) (Seq.index v x))
= output_fragment_cell_convert_eq gD bm bn tm tn wm wn
    bid wid mi nj i (j + x) f (Seq.index v x)

(* A run of [w] consecutive cells of an output fragment is the same run of
   cells of the underlying global matrix. *)
ghost
fn row_cells_frag_to_global
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
  (j : nat)
  (w : nat { j + w <= tn })
  (globalRow : natlt m {
    globalRow == tiled_cell m bm (bid / (n / bn))
      (tiled_cell bm (wm * tm) (wid / (bn / (wn * tn)))
        (tiled_cell (wm * tm) tm mi i)) })
  (globalCol : nat { globalCol + w <= n /\
    globalCol == bid % (n / bn) * bn
      + wid % (bn / (wn * tn)) * (wn * tn) + nj * tn + j })
  (f : perm)
  (v : seq et { Seq.length v == w })
  requires
    row_cells (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
      f i j w v
  ensures
    row_cells gD f globalRow globalCol w v
{
  FStar.Classical.forall_intro
    (frag_global_cell_eq gD bm bn tm tn wm wn bid wid mi nj
      i j w globalRow globalCol f v);
  unfold row_cells (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
    f i j w v;
  forevery_ext #(natlt w)
    (fun x -> T.tensor_pts_to_cell
      (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
      #f (idx2 i (j + x)) (Seq.index v x))
    (fun x -> T.tensor_pts_to_cell gD #f
      (idx2 globalRow (globalCol + x)) (Seq.index v x));
  fold row_cells gD f globalRow globalCol w v;
}

ghost
fn row_cells_global_to_frag
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
  (j : nat)
  (w : nat { j + w <= tn })
  (globalRow : natlt m {
    globalRow == tiled_cell m bm (bid / (n / bn))
      (tiled_cell bm (wm * tm) (wid / (bn / (wn * tn)))
        (tiled_cell (wm * tm) tm mi i)) })
  (globalCol : nat { globalCol + w <= n /\
    globalCol == bid % (n / bn) * bn
      + wid % (bn / (wn * tn)) * (wn * tn) + nj * tn + j })
  (f : perm)
  (v : seq et { Seq.length v == w })
  requires
    row_cells gD f globalRow globalCol w v
  ensures
    row_cells (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
      f i j w v
{
  FStar.Classical.forall_intro
    (frag_global_cell_eq gD bm bn tm tn wm wn bid wid mi nj
      i j w globalRow globalCol f v);
  unfold row_cells gD f globalRow globalCol w v;
  forevery_ext #(natlt w)
    (fun x -> T.tensor_pts_to_cell gD #f
      (idx2 globalRow (globalCol + x)) (Seq.index v x))
    (fun x -> T.tensor_pts_to_cell
      (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
      #f (idx2 i (j + x)) (Seq.index v x));
  fold row_cells (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
    f i j w v;
}

let divides_helper
  (d : pos)
  (a b r c : int)
  : Lemma (requires d /? a /\ d /? b /\ d /? c)
          (ensures d /? (a + b * r + c))
= Kuiper.Divides.lemma_divides_product_l d b r;
  Kuiper.Divides.lemma_divides_sum d a (b * r);
  Kuiper.Divides.lemma_divides_sum d (a + b * r) c

(* [chunk et_cd] divides the global column of the start of a chunk. *)
let global_col_divides
  (w : pos)
  (bn rows cols wm wn : pos)
  (#_ : squash (wn * cols /?+ bn))
  (mcol warpCol mi : nat)
  (col : nat)
  : Lemma (requires w /? cols /\ w /? col)
          (ensures w /? (mcol * bn + warpCol * (wn * cols) + mi * cols + col))
= Kuiper.Divides.lemma_divides_product_r w wn cols;
  ML.div_exact_r bn (wn * cols);
  Kuiper.Divides.lemma_divides_product_r w (bn / (wn * cols)) (wn * cols);
  assert (bn == (bn / (wn * cols)) * (wn * cols));
  Kuiper.Divides.lemma_divides_product_r w mcol bn;
  Kuiper.Divides.lemma_divides_product_r w warpCol (wn * cols);
  Kuiper.Divides.lemma_divides_product_r w mi cols;
  Kuiper.Divides.lemma_divides_sum w (mcol * bn) (warpCol * (wn * cols));
  Kuiper.Divides.lemma_divides_sum w
    (mcol * bn + warpCol * (wn * cols)) (mi * cols);
  Kuiper.Divides.lemma_divides_sum w
    (mcol * bn + warpCol * (wn * cols) + mi * cols) col

(* The target values of a run of fragment cells, in terms of the
   corresponding run of global [C] cells. *)
let target_run_eq
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
  (row : natlt rows)
  (col : nat)
  (w : nat { col + w <= cols })
  (globalRow : natlt m)
  (globalCol : nat)
  (pf : squash (
    globalCol + w <= n /\
    globalRow == mrow * bm + warpRow * (wm * rows)
                 + (idx / wn) * rows + row /\
    globalCol == mcol * bn + warpCol * (wn * cols)
                 + (idx % wn) * cols + col))
  (x : natlt w)
  : Lemma (
      acc2 (epilogue_fragment_target comb eC bm bn rows cols wm wn
              mrow mcol warpRow warpCol idx eAcc) row (col + x)
      == comb (acc2 eC globalRow (globalCol + x))
              (acc2 eAcc row (col + x)))
= epilogue_fragment_target_eq comb eC bm bn rows cols wm wn
    mrow mcol warpRow warpCol idx eAcc

#push-options "--z3rlimit 60"
(* Vectorized epilogue update of a [chunk et_cd]-wide run of cells of an
   output fragment: one vector load of [C], [chunk] scalar accumulator
   reads, and one vector store to [D]. *)
inline_for_extraction noextract
fn epilogue_chunk_update
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
  (row : szlt rows)
  (col : szlt cols { chunk et_cd /? SZ.v col /\
                     SZ.v col + chunk et_cd <= SZ.v cols })
  (old nv : erased (seq et_cd))
  (#_ : squash (
    Seq.length old == chunk et_cd /\
    Seq.length nv == chunk et_cd /\
    (forall (x : natlt (chunk et_cd)).
      Seq.index nv x ==
      acc2
        (epilogue_fragment_target comb eC
          bm bn rows cols wm wn
          (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
          (SZ.v idx) eAcc)
        (SZ.v row) (SZ.v col + x))))
  ()
  preserves
    gpu **
    c |-> Frac fC eC **
    acc |-> Frac (1.0R /. warp_size) eAcc
  requires
    pure (aligned 16 (T.core c) /\ aligned 16 (T.core d) /\
          aligned_strided_row_major (chunk et_cd) str)
  requires
    row_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      1.0R (SZ.v row) (SZ.v col) (chunk et_cd) (reveal old)
  ensures
    row_cells
      (output_fragment d bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / wn) (SZ.v idx % wn))
      1.0R (SZ.v row) (SZ.v col) (chunk et_cd) (reveal nv)
{
  assert pure (SZ.v idx / SZ.v wn < SZ.v wm);
  assert pure (SZ.v idx % SZ.v wn < SZ.v wn);
  assert pure (
    (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col + (chunk et_cd)
    <= SZ.v wn * SZ.v cols);
  assert pure (SZ.v warpCol + 1 <= SZ.v bn / (SZ.v wn * SZ.v cols));
  ML.div_exact_r (SZ.v bn) (SZ.v wn * SZ.v cols);
  assert pure (
    (SZ.v bn / (SZ.v wn * SZ.v cols)) * (SZ.v wn * SZ.v cols) == SZ.v bn);
  assert pure (
    SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col + (chunk et_cd)
    <= (SZ.v warpCol + 1) * (SZ.v wn * SZ.v cols));
  assert pure ((SZ.v warpCol + 1) * (SZ.v wn * SZ.v cols) <= SZ.v bn);
  assert pure (
    SZ.v warpRow * (SZ.v wm * SZ.v rows)
      + (SZ.v idx / SZ.v wn) * SZ.v rows + SZ.v row
    < SZ.v bm);
  assert pure (
    SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col + (chunk et_cd)
    <= SZ.v bn);
  assert pure (
    SZ.v mrow * SZ.v bm + SZ.v warpRow * (SZ.v wm * SZ.v rows)
      + (SZ.v idx / SZ.v wn) * SZ.v rows + SZ.v row
    < SZ.v m);
  assert pure (
    SZ.v mcol * SZ.v bn + SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col + (chunk et_cd)
    <= SZ.v n);
  let globalRow : szlt m =
    mrow *^ bm +^ warpRow *^ (wm *^ rows) +^ (idx /^ wn) *^ rows +^ row;
  let globalCol : szlt (n -^ (chunk et_cd) +^ 1sz) =
    mcol *^ bn +^ warpCol *^ (wn *^ cols) +^ (idx %^ wn) *^ cols +^ col;

  // The start of the run is [chunk]-aligned within both [c] and [d].
  global_col_divides (chunk et_cd) (SZ.v bn) (SZ.v rows) (SZ.v cols) (SZ.v wm) (SZ.v wn)
    (SZ.v mcol) (SZ.v warpCol) (SZ.v idx % SZ.v wn) (SZ.v col);
  str.pf globalRow globalCol;
  divides_helper
    (chunk et_cd) str.offset str.stride (SZ.v globalRow) (SZ.v globalCol);
  assert pure ((chunk et_cd) /? cell_of_pos (rm m n) (SZ.v globalRow) (SZ.v globalCol));
  assert pure ((chunk et_cd) * size #et_cd == 16);
  assert pure (
    16 /?+ (cell_of_pos (rm m n) (SZ.v globalRow) (SZ.v globalCol)
            * size #et_cd));

  let mut cbuf = [| zero #et_cd #_; (chunk et_cd) |];
  assume pure (aligned 16 cbuf); // FIXME local arrays do not need alignment
  array2_vec_read c globalRow globalCol cbuf;

  let mut obuf = [| zero #et_cd #_; (chunk et_cd) |];
  assume pure (aligned 16 obuf); // FIXME local arrays do not need alignment

  let mut k = 0sz;
  while (!k <^ (chunk et_cd))
    invariant live k
    invariant pure (SZ.v !k <= SZ.v (chunk et_cd))
    invariant exists* (sc so : seq et_cd).
      cbuf |-> sc ** obuf |-> so **
      pure (
        sc == Seq.init_ghost (SZ.v (chunk et_cd)) (fun x ->
          acc2 eC (SZ.v globalRow) (SZ.v globalCol + x)) /\
        Seq.length so == SZ.v (chunk et_cd) /\
        Seq.equal so (Seq.init_ghost (SZ.v (chunk et_cd)) (fun x ->
          if x < SZ.v !k
          then comb (acc2 eC (SZ.v globalRow) (SZ.v globalCol + x))
                    (acc2 eAcc (SZ.v row) (SZ.v col + x))
          else zero #et_cd)))
    decreases (SZ.v (chunk et_cd) - SZ.v !k)
  {
    let vk = !k;
    assert pure (SZ.v vk < SZ.v (chunk et_cd));
    with sc. assert cbuf |-> sc;
    let cv = Pulse.Lib.Array.op_Array_Access cbuf vk #_ #sc;
    assert pure (SZ.v col + SZ.v (chunk et_cd) <= SZ.v cols);
    let ccol : szlt cols = col +^ vk;
    let av = tensor_read acc (row, (ccol, ()));
    with so. assert obuf |-> so;
    Pulse.Lib.Array.op_Array_Assignment obuf vk (comb cv av) #so;
    with so'. assert obuf |-> so';
    assert pure (Seq.equal so' (Seq.init_ghost (SZ.v (chunk et_cd)) (fun x ->
      if x < SZ.v vk + 1
      then comb (acc2 eC (SZ.v globalRow) (SZ.v globalCol + x))
                (acc2 eAcc (SZ.v row) (SZ.v col + x))
      else zero #et_cd)));
    k := vk +^ 1sz;
  };

  with so. assert obuf |-> so;
  FStar.Classical.forall_intro
    (target_run_eq comb eC
      (SZ.v bm) (SZ.v bn) (SZ.v rows) (SZ.v cols) (SZ.v wm) (SZ.v wn)
      (SZ.v mrow) (SZ.v mcol) (SZ.v warpRow) (SZ.v warpCol)
      (SZ.v idx) eAcc (SZ.v row) (SZ.v col) (chunk et_cd)
      (SZ.v globalRow) (SZ.v globalCol) ());
  assert pure (Seq.equal so (reveal nv));

  row_cells_frag_to_global d
    (SZ.v bm) (SZ.v bn) (SZ.v rows) (SZ.v cols) (SZ.v wm) (SZ.v wn)
    (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn)
    (SZ.v row) (SZ.v col) (chunk et_cd)
    (SZ.v globalRow) (SZ.v globalCol) 1.0R (reveal old);
  array2_vec_write_cells d globalRow globalCol obuf old nv ();
  row_cells_global_to_frag d
    (SZ.v bm) (SZ.v bn) (SZ.v rows) (SZ.v cols) (SZ.v wm) (SZ.v wn)
    (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn)
    (SZ.v row) (SZ.v col) (chunk et_cd)
    (SZ.v globalRow) (SZ.v globalCol) 1.0R (reveal nv);
}
#pop-options
