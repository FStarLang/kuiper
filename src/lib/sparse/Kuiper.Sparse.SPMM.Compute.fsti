module Kuiper.Sparse.SPMM.Compute

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Spec.GEMM
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.Common
module Array2 = Kuiper.Array2
open Kuiper.Tensor { ctlayout }

(* Definiciones auxiliares *)

let submatrix
  (#a : Type0)
  (#rows #cols : nat)
  (em : ematrix a rows cols)
  (m : natle rows)
  (srows : nat{m + srows <= rows})
  (n : natle cols)
  (scols : nat{n + scols <= cols})
: ematrix a srows scols
= mkM (fun i j -> macc em (m + i) (n + j))

let col_strided_matrix
  (#a : Type0)
  (#rows #cols : nat)
  (em : ematrix a rows cols)
  (step : pos)
: ematrix a rows (cols `divup` step)
=
  mkM (fun i j -> macc em i (j * step))

let step_sparse_cols
  (cols n scols : nat)
  (step : pos)
  : GTot nat
= let n' = min n cols in min scols (cols - n') `divup` step

let step_submatrix
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  (scols : nat)
  (n : nat)
  (step : pos)
: Pure (ematrix et rows (step_sparse_cols cols n scols step))
  (requires true)
  (ensures fun _ -> true)
=
  let n' = min n cols in
  col_strided_matrix (submatrix em 0 rows n' (min scols (cols - n'))) step

(* Definiciones sparse *)

let _sparse_row_x_mat_acc
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
  (to : natle nnz)
: Ghost (lseq et block)
  (requires valid_pos shared pos)
  (ensures fun _ -> true)
=
  Seq.init_ghost block (fun i ->
    if i < cols
      then _sparse_dprod_acc (acc @! i) elems pos (ematrix_col em i) to
      else acc @! i
  )

let sparse_row_x_mat_acc
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
: Ghost (lseq et block)
  (requires valid_pos shared pos)
  (ensures fun _ -> true)
=
  _sparse_row_x_mat_acc acc elems pos em nnz

(* Lemas para combinar resultados *)

(* Sparse compute *)

let compute_result
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (bw bx : pos{bw /? bx})
  (#nnz : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (eB : ematrix et shared cols)
  (out : lseq et (bx / bw))
  (off : natlt bw)
  (n : natlt cols)
: Ghost (lseq et (bx / bw))
  (requires valid_pos shared col_ind)
  (ensures fun _ -> true)
=
  sparse_row_x_mat_acc
    out elems col_ind
    (step_submatrix eB (bx - off) (n + off) bw)

val compute_step
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (bw bx : pos{bw /? bx})
  (#nnz : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (eB : ematrix et shared cols)
  (out : lseq et (bx / bw))
  (off : natlt bw)
  (n : natlt cols)
  (from to : natle nnz{from <= to})
: Lemma
  (requires valid_pos shared col_ind)
  (ensures
    compute_result
      bw bx #(to - from)
      (Seq.slice elems from to) (Seq.slice col_ind from to) eB
      (compute_result
        bw bx #from
        (Seq.slice elems 0 from) (Seq.slice col_ind 0 from)
        eB out off n
      )
      off n ==
    compute_result
      bw bx #to
      (Seq.slice elems 0 to) (Seq.slice col_ind 0 to)
      eB out off n
  )

val compute_lemma
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (bw bx : pos{bw /? bx})
  (#nnz : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (out : lseq et (bx / bw))
  (off : natlt bw)
  (n : natlt cols)
  (i : natlt rows)
  (x : natlt (bx / bw))
: Lemma
  (requires
    valid_pos shared col_ind /\
    unsparse _ _ elems col_ind == ematrix_row eA i /\
    n + off + x * bw < cols /\
    forall i. out @! i == zero
  )
  (ensures
    compute_result bw bx elems col_ind eB out off n @! x ==
    matmul_single eA eB i (n + off + x * bw)
  )

// TODO ver si se pueden simplificar más los argumentos
inline_for_extraction noextract
fn compute
  (#et : Type0) {| scalar et |}
  (#shared #cols : sz)
  // creo que este refinamiento no hace falta
  (blockWidth blockItemsK blockItemsX : szp{blockWidth /? blockItemsX})
  // fragmentos sparse
  (elems_tile : gpu_array et blockItemsK)
  (col_ind_tile : gpu_array sz blockItemsK)
  (#fA : perm)
  (nnz : sz)
  (#v_elems : erased (lseq et nnz))
  (#v_col_ind : erased (lseq sz nnz))
  (#_ : squash(valid_pos shared (cast_pos v_col_ind)))
  // matriz densa B
  (#lB : Array2.layout shared cols)
  {| ctlayout lB |}
  (gB : Array2.t et lB)
  (#fB : perm)
  (#eB : erased (ematrix et shared cols))
  // resultado parcial
  (out : larray et (blockItemsX /^ blockWidth))
  (#v_out : erased (seq et))
  (#_ : squash (len v_out == blockItemsX /^ blockWidth))
  (tid : szlt blockWidth)
  (n_idx : szlt cols)
  norewrite
  preserves
    gpu **
    gpu_pts_to_slice elems_tile #fA 0 nnz v_elems **
    gpu_pts_to_slice col_ind_tile #fA 0 nnz v_col_ind **
    gB |-> Frac fB eB
  requires
    pure (fits (cols + blockItemsX)) **
    out |-> v_out
  ensures
    out |->
      compute_result
        blockWidth blockItemsX
        v_elems (cast_pos v_col_ind) eB v_out
        tid n_idx
