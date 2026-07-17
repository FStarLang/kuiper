module Klas.SPMM.Inst

open Kuiper
open Kuiper.Sparse
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.Tensor.Layout { trepr2, ctrepr2 }
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.EMatrix
open Kuiper.Array.Vectorized
module MS = Kuiper.Spec.GEMM

#lang-pulse

inline_for_extraction noextract
fn inst
  (et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (repB repC : trepr2)
  {| crepB : ctrepr2 repB, crepC : ctrepr2 repC |}
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    k /? blockItemsX
  }))
  (rows shared cols : szp)
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et (repB shared cols) { is_global gB})
  (#fB : perm)
  (gC : array2 et (repC rows cols) { is_global gC})
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
