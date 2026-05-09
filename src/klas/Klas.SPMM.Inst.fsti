module Klas.SPMM.Inst

open Kuiper
open Kuiper.Sparse
module SZ = Kuiper.SizeT
module Array2 = Kuiper.Array2
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM

#lang-pulse

let lseq (a:Type) (n:nat) = erased (Seq.lseq a n)

inline_for_extraction noextract
fn inst
  (et : Type0) {| scalar et |}
  // (rB : mrepr) {| crepr rB |}
  // (rC : mrepr) {| crepr rC |}
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {k /? blockItemsK /\ k /? blockItemsX}))
  (rows shared cols : szp)
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#fA : perm)
  (row_indices : gpu_array sz rows)
  (fri : perm)
  (gB : Array2.t et (rm shared cols) { Array2.is_global gB})
  (#fB : perm)
  (gC : Array2.t et (rm rows cols) { Array2.is_global gC})
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (#eA : ematrix et rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
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
