module Kuiper.Sparse.SPMM

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.Sparse
open Kuiper.EMatrix

let lseq (a:Type) (n:nat) = erased (Seq.lseq a n)

inline_for_extraction noextract
fn spmm
  (#et : Type0) {| scalar et |}
  (rows shared cols : szp)
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {k /? blockItemsK /\ k /? blockItemsX}))
  (blockChunks : sz{SZ.v blockChunks == blockItemsX / blockWidth}) // Ver nota abajo
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et lB {is_global gB})
  (#fB : perm)
  (gC : array2 et lC {is_global gC})
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (#eA : chest2 et rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#eC : chest2 et rows cols)
  //(#_ : size_req rows shared cols)
  norewrite
  preserves
    cpu **
    //on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (smatrix_pts_to' gA #fA elems col_ind row_off eA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (blockItemsX /? cols) **
    on gpu_loc (live gC) **
    pure (rows * (cols `divup` blockItemsX) <= max_blocks) **
    pure (blockWidth <= max_threads)
  ensures on gpu_loc (gC |-> MS.matmul eA eB)
