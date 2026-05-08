module Klas.SPMM

open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.SPMM
module SZ = Kuiper.SizeT
open Kuiper.Matrix.Reprs
module M = Kuiper.Matrix
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM

#lang-pulse

let lseq (a:Type) (n:nat) = erased (Seq.lseq a n)

inline_for_extraction noextract
fn inst
  (et : Type0) {| scalar et |}
  (rB : mrepr) {| crepr rB |}
  (rC : mrepr) {| crepr rC |}
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {k /? blockItemsK /\ k /? blockItemsX}))
  (blockChunks : sz{SZ.v blockChunks == blockItemsX / blockWidth}) // Ver nota abajo
  (rows shared cols : szp)
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#fA : perm)
  (row_indices : gpu_array sz rows)
  (fri : perm)
  (gB : M.gpu_matrix et (rB shared cols) {M.is_global_matrix gB})
  (#fB : perm)
  (gC : M.gpu_matrix et (rC rows cols) {M.is_global_matrix gC})
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
    pure (rows * cols / blockItemsX <= max_blocks) **
    pure (blockWidth <= max_threads)
  ensures on gpu_loc (gC |-> MS.matmul eA eB)
{
  M.gpu_matrix_pts_to_ref_located gB;
  M.gpu_matrix_pts_to_ref_located gC;
  spmm #et
    rows shared cols
    blockItemsK blockItemsX blockWidth blockChunks
    #(rB shared cols) #(rC rows cols)
    gA row_indices fri gB gC
    elems col_ind row_off
    #eA
    row_perm
    #eB #eC
}

let spmm_u32 = inst u32 row_major row_major 128sz 128sz 32sz 4sz
let spmm_f32 = inst f32 row_major row_major 128sz 128sz 32sz 4sz
