module Klas.SPMM.Inst

open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.SPMM
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.EMatrix
open Kuiper.Array.Vectorized
open Kuiper.Array2.Strided
module MS = Kuiper.Spec.GEMM

#lang-pulse

inline_for_extraction noextract
fn inst
  (et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (rows shared cols : szp)
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et (rm shared cols) { is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#fB : perm)
  (gC : array2 et (rm rows cols) { is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  // matriz sparse gA
  (elems : erased (lseq et gA.nnz))
  (col_ind : erased (lseq sz gA.nnz))
  (row_off : erased (lseq sz (rows + 1)))
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
{
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;
  prod_divides blockWidth (chunk et) blockItemsX;
  lemma_divides_trans (chunk et) blockItemsX cols;
  assert pure (chunk et /? cols);
  assert pure (chunk et /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v shared) #(SZ.v cols) (chunk et);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v rows) #(SZ.v cols) (chunk et);
  spmm #et
    rows shared cols
    blockItemsK blockItemsX blockWidth (blockItemsX /^ blockWidth)
    gA row_indices fri gB gC
    elems col_ind row_off
    #eA
    row_perm
    #eB #eC
}
