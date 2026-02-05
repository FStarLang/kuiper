module Kuiper.Poly.GEMM.Util

#lang-pulse

open Kuiper
open Kuiper.Approximates
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)

inline_for_extraction noextract
fn matmul_tiled_dotprod
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols #tile : szp)
  (#lA : mlayout (mrows * tile)   (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA #eB : ematrix et _ _)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  // ensures
  //   pure (res == MS.matmul_single #et #_ #(rows * tile) #(shared * tile) #(cols * tile) eA eB i j shared)

(* Real-valued matrix for specification purposes *)
let ematrix_to_real (#et:Type) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  : GTot (ematrix real rows cols)
  = mkM (fun i j -> to_real (macc em i j))

(* Real-valued matmul_single using real arithmetic (which IS associative) *)
let real_matmul_single
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  (row : natlt rows)
  (col : natlt cols)
  : GTot real
  = MS.__gmatmul_single 0.0R ( *. ) ( +. )
      (ematrix_to_real m1) (ematrix_to_real m2)
      row col shared

(* Real-valued gemm_single: combines initial value with real matmul using comb_r *)
let real_gemm_single
  (comb_r : binop real)
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  (m0 : ematrix et rows cols)
  (row : natlt rows)
  (col : natlt cols)
  : GTot real
  = comb_r (to_real (macc m0 row col)) (real_matmul_single m1 m2 row col)

(* Real-valued GEMM matrix: each cell is real_gemm_single *)
let real_mmcomb
  (comb_r : binop real)
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m0 : ematrix et rows cols)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  : GTot (ematrix real rows cols)
  = mkM (fun i j -> real_gemm_single comb_r m1 m2 m0 i j)

(* Version of matmul_tiled_dotprod with approximate postcondition *)
inline_for_extraction noextract
fn matmul_tiled_dotprod'
  (#et : Type0) {| scalar et, real_like et |}
  (#mrows #mshared #mcols #tile : szp)
  (#lA : mlayout (mrows * tile)   (mshared * tile))
  (#lB : mlayout (mshared * tile) (mcols   * tile))
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA #eB : ematrix et _ _)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res %~ real_matmul_single eA eB (bi * tile + i) (bj * tile + j))

(* Used by SHMEM, Blocktiling1D *)
inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 : mlayout tile tile) {| clayout l1 |}
  (#l2 : mlayout tile tile) {| clayout l2 |}
  (m1 : M.gpu_matrix et l1)
  (m2 : M.gpu_matrix et l2)
  (j : szlt tile)
  (#acc0 : erased (seq et))
  (#v1 #v2 : ematrix et tile tile)
  (#f : perm)
  preserves
    gpu **
    m1 |-> Frac f v1 **
    m2 |-> Frac f v2
  requires
    pure (Seq.length acc0 == tile) **
    acc |-> acc0
  ensures
    exists* acc'.
      pure (Seq.length acc' == tile) **
      (acc |-> acc')
